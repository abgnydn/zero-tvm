#!/usr/bin/env node
// MOE-GROUP-PROBE — is MoE chunking Phase B worth building?
//
//   node --experimental-strip-types scripts/moe-group-probe.mjs
//
// Phase A (2026-08-13) gave the MoE block a token dimension, so a whole chunk
// runs in seven dispatches instead of seven per token. It did NOT change what
// the expert matmuls READ: every (token, slot) pair still pulls its expert's
// full weight rows, so a 256-token chunk at top-8 reads 2048 expert-sized
// slabs where only ~128 distinct experts exist.
//
// Phase B sorts the pairs by expert so each expert's weights are read once for
// all the rows that chose it. That is a permutation kernel, per-expert offsets
// and a grouped GEMM — a real build. This probe measures the ceiling FIRST, on
// synthetic buffers at the real qwen30b shapes, so the build is justified by a
// number rather than by the arithmetic looking convincing.
//
// Arm A: the SHIPPED moe kernel, grid (N/4, tokens, slots) — what runs today.
// Arm B: the SHIPPED E5 chunk GEMM, once per expert, M = that expert's rows.
//
// Arm B leaves out the gather/scatter Phase B would need. That is deliberate
// and it is why this is a CEILING: the permutation moves tokens*K activations
// (8 MB here) against the hundreds of MB of weight traffic the grouping saves,
// so it cannot change the verdict, only shave the win.

import { installShims } from './native/shims.mjs'
import { toF16, f16Array } from '../tests/kernels/half.mjs'
import { int4MatmulWGSL, int4MatmulEntry, int4MatmulSgE5WGSL } from '../src/compiler/shaders/int4_matmul.gen.ts'

await installShims({ unsafe: true })

// qwen30b's expert matmul: K = d, N = moe_intermediate, 128 experts top-8.
const K = Number(process.env.K) || 2048
const N = Number(process.env.N) || 768
const E = Number(process.env.E) || 128
const TOK = Number(process.env.TOK) || 256
const SLOTS = Number(process.env.SLOTS) || 8
const PAIRS = TOK * SLOTS

const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix']
    .filter((f) => adapter.features.has(f)),
})
device.addEventListener?.('uncapturederror', (e) => console.error('[gpu]', e.error?.message))

const rnd = (() => { let st = 91; return () => (st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff })()
const ST = GPUBufferUsage.STORAGE
const buf = (data, usage) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(b, 0, data)
  return b
}
const pipeOf = (code, entry) => device.createComputePipeline({
  layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: entry },
})

const KP = K / 8, SPR = K / 64
// One expert stack: [E, N, K]. 4-bit + f16 scale/bias per group of 64.
const expertWords = N * KP, expertScales = N * SPR
const w = new Uint32Array(E * expertWords); for (let i = 0; i < w.length; i++) w[i] = (rnd() * 0xffffffff) >>> 0
const wB = buf(w, ST)
const scB = buf(f16Array(Array.from({ length: E * expertScales }, () => toF16(rnd() * 0.05 + 0.01))), ST)
const biB = buf(f16Array(Array.from({ length: E * expertScales }, () => toF16(rnd() * 0.1 - 0.05))), ST)
const actB = buf(f16Array(Array.from({ length: TOK * K }, () => toF16(rnd() * 2 - 1))), ST)

// Expert choice per (token, slot). Uniform-random over E is the FAVOURABLE case
// for Phase B: every expert gets ~PAIRS/E rows. Real routing is skewed, which
// makes some groups larger (better) and some size-1 (no better than today), so
// treat this as the ceiling it is.
const ids = new Uint32Array(PAIRS); for (let i = 0; i < PAIRS; i++) ids[i] = Math.floor(rnd() * E) % E
const idsB = buf(ids, ST)
const perExpert = new Array(E).fill(0); for (const e of ids) perExpert[e]++
const used = perExpert.filter((c) => c > 0).length

const time = async (encodeOnce) => {
  const once = (iters) => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    for (let i = 0; i < iters; i++) encodeOnce(pass)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  once(2); await device.queue.onSubmittedWorkDone()
  const t0 = performance.now()
  once(10); await device.queue.onSubmittedWorkDone()
  return (performance.now() - t0) / 10
}

// ---- Arm A: the shipped moe kernel, one dispatch over (rows, token, slot) ----
const MOE_OPTS = { affine: true, moe: true, subgroups: true, rowsPerWG: 4 }
const moePipe = pipeOf(int4MatmulWGSL(MOE_OPTS), int4MatmulEntry(MOE_OPTS))
const moeOut = device.createBuffer({ size: PAIRS * N * 2, usage: ST })
const moePod = buf(new Uint32Array([KP, SPR, N, 0, N, K, SLOTS * N, SLOTS]), GPUBufferUsage.UNIFORM)
const moeBg = device.createBindGroup({
  layout: moePipe.getBindGroupLayout(0),
  entries: [moeOut, actB, scB, wB, moePod, biB, idsB].map((b, i) => ({ binding: i, resource: { buffer: b } })),
})
// An invalid bind group makes WebGPU SKIP the dispatch, and a skipped dispatch
// times as ~0 ms rather than as an error — the first run of this probe read
// 0.013 ms for arm A that way. Fail loudly instead.
{
  device.pushErrorScope('validation')
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(moePipe); pass.setBindGroup(0, moeBg)
  pass.dispatchWorkgroups(N / 4, TOK, SLOTS)
  pass.end()
  device.queue.submit([enc.finish()])
  const err = await device.popErrorScope()
  if (err) throw new Error(`arm A did not run: ${err.message.split('\n')[0]}`)
}
const msA = await time((pass) => {
  pass.setPipeline(moePipe); pass.setBindGroup(0, moeBg)
  pass.dispatchWorkgroups(N / 4, TOK, SLOTS)
})

// ---- Arm B: the shipped E5 chunk GEMM, once per expert ----
// Weights/scales are bound at that expert's offset, so E5 sees an ordinary
// single-matrix GEMM — which is exactly what Phase B would hand it.
const e5Pipe = pipeOf(int4MatmulSgE5WGSL(true), 'int4_matmul_sg_e5_affine')
const CAP = 64   // E5 stages a 64-row tile whatever M is
const e5Out = device.createBuffer({ size: Math.max(CAP, 64) * N * 2 * E, usage: ST })
const groups = []
for (let e = 0; e < E; e++) {
  const m = perExpert[e]
  if (!m) continue
  groups.push({
    m,
    bg: device.createBindGroup({
      layout: e5Pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: e5Out, offset: e * CAP * N * 2, size: CAP * N * 2 } },
        { binding: 1, resource: { buffer: actB } },
        { binding: 2, resource: { buffer: scB, offset: e * expertScales * 2, size: expertScales * 2 } },
        { binding: 3, resource: { buffer: wB, offset: e * expertWords * 4, size: expertWords * 4 } },
        { binding: 4, resource: { buffer: buf(new Uint32Array([KP, SPR, N, m]), GPUBufferUsage.UNIFORM) } },
        { binding: 5, resource: { buffer: biB, offset: e * expertScales * 2, size: expertScales * 2 } },
      ],
    }),
  })
}
const msB = await time((pass) => {
  pass.setPipeline(e5Pipe)
  for (const g of groups) {
    pass.setBindGroup(0, g.bg)
    pass.dispatchWorkgroups(Math.ceil(N / 32), Math.ceil(g.m / 64), 1)
  }
})

const bytesA = PAIRS * (expertWords * 4 + expertScales * 4)
const bytesB = used * (expertWords * 4 + expertScales * 4)
console.log(`\nMoE expert matmul, qwen30b shapes: K=${K} N=${N} experts=${E}, chunk ${TOK} tokens x ${SLOTS} slots = ${PAIRS} pairs`)
console.log(`  ${used}/${E} experts used, ${(PAIRS / used).toFixed(1)} rows per expert on average\n`)
console.log(`  A  shipped moe kernel, 1 dispatch      ${msA.toFixed(3)} ms   weights read ${(bytesA / 2 ** 20).toFixed(0)} MiB`)
console.log(`  B  E5 per expert, ${String(groups.length).padStart(3)} dispatches     ${msB.toFixed(3)} ms   weights read ${(bytesB / 2 ** 20).toFixed(0)} MiB`)
console.log(`\n  grouping is ${(msA / msB).toFixed(2)}x faster here (weight traffic ${(bytesA / bytesB).toFixed(1)}x lower)`)
console.log(`  ceiling, not a forecast: no gather/scatter, and uniform routing is the kindest case.`)

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
