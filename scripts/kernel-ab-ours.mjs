#!/usr/bin/env node
// KERNEL-AB-OURS — time OUR int4 matmul alone, at shapes an MLX run can match.
//
//   node --experimental-strip-types scripts/kernel-ab-ours.mjs --json
//
// The LM Studio comparison in BENCH.md is a RUNTIME result: both arms are whole
// pipelines, and nothing in it isolates a kernel. This is the other
// measurement — one kernel, one shape, nothing else in the timing loop — so it
// can be put beside MLX's own `mx.quantized_matmul` at the same shape.
//
// Same quantization on both sides or the comparison is meaningless: 4-bit,
// group 64, AFFINE (w = s*q + b). That is what MLX writes and what our affine
// family reads.
//
// M=1 picks the decode kernel (int4_matmul_tiled_vec4h_affine, the wide-load
// matvec) and M>1 picks E5, the subgroup-matrix chunk GEMM — the same ladder
// variants.ts resolves at runtime, not a special benchmark path.
//
// CROSS-PROCESS, and that is a real weakness. Every other A/B in this repo
// alternates arms inside ONE process; MLX is Python and we are Node, so the
// two cannot share one. scripts/kernel-ab.mjs is the driver that alternates
// the two processes round by round, which is the same compromise
// lmstudio-ab.mjs already makes. Do not compare a run of this against a run of
// the MLX side taken at another time.
//
// VOID on battery: `pmset -g batt` must say charged/charging first.

import { installShims } from './native/shims.mjs'
import { toF16, f16Array } from '../tests/kernels/half.mjs'
import { int4MatmulWGSL, int4MatmulEntry, int4MatmulSgE5WGSL } from '../src/compiler/shaders/int4_matmul.gen.ts'

const JSON_OUT = process.argv.includes('--json')
const ITERS = Number(process.env.ITERS) || 200

await installShims({ unsafe: true })
const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice({
  requiredFeatures: ['shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix']
    .filter((f) => adapter.features.has(f)),
  requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                    maxBufferSize: adapter.limits.maxBufferSize },
})

// Qwen3.5-9B-MLX-4bit, the model the LM Studio A/B runs. These four are where
// a decode token's time goes (BENCH.md): the FFN pair dominates, o_proj and
// the attention projection follow.
const SHAPES = [
  { name: 'ffn_gate_up', K: 4096, N: 24576 },
  { name: 'ffn_down', K: 12288, N: 4096 },
  { name: 'o_proj', K: 4096, N: 4096 },
  { name: 'c_attn', K: 4096, N: 10240 },
]
const MS = (process.env.MS || '1,256').split(',').map(Number)

const rnd = (() => { let st = 7; return () => (st = (st * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff })()
const ST = GPUBufferUsage.STORAGE
const buf = (data, usage) => {
  const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(b, 0, data)
  return b
}
const pipeOf = (code, entry) => device.createComputePipeline({
  layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: entry },
})

// The two shipped affine kernels, exactly as variants.ts resolves them.
const MATVEC = { affine: true, subgroups: true, rowsPerWG: 4, vec4Half: true }
const matvecPipe = pipeOf(int4MatmulWGSL(MATVEC), int4MatmulEntry(MATVEC))
const e5Pipe = pipeOf(int4MatmulSgE5WGSL(true), 'int4_matmul_sg_e5_affine')

const out = []
for (const { name, K, N } of SHAPES) {
  const KP = K / 8, SPR = K / 64
  const wBytes1 = N * KP * 4 + N * SPR * 4
  // ROTATE OVER DISTINCT WEIGHT SETS. Looping one matrix measures the cache:
  // o_proj's 8 MiB sits in the M2 Max's SLC after iteration 1, and the run
  // reports a bandwidth decode never sees. Real decode walks 32 layers of
  // DIFFERENT weights per token, so every read is cold. Enough copies to blow
  // past ~256 MiB of cache, capped so a big shape does not exhaust memory.
  const COPIES = Math.max(2, Math.min(24, Math.ceil((256 * 2 ** 20) / wBytes1)))
  const wSets = Array.from({ length: COPIES }, () => {
    const w = new Uint32Array(N * KP); for (let i = 0; i < w.length; i++) w[i] = (rnd() * 0xffffffff) >>> 0
    return {
      wB: buf(w, ST),
      scB: buf(f16Array(Array.from({ length: N * SPR }, () => toF16(rnd() * 0.05 + 0.01))), ST),
      biB: buf(f16Array(Array.from({ length: N * SPR }, () => toF16(rnd() * 0.1 - 0.05))), ST),
    }
  })

  for (const M of MS) {
    const CAP = Math.max(M, 64)   // E5 stages a 64-row tile whatever M is
    const aB = buf(f16Array(Array.from({ length: CAP * K }, () => toF16(rnd() * 2 - 1))), ST)
    // ONE OUTPUT PER ROTATION, not one shared. Dispatches that all write the
    // same buffer are a write-after-write hazard and the driver serializes
    // them — so a shared output measures single-dispatch LATENCY, while the
    // engine (and MLX's arm, whose every call returns a fresh array) overlaps
    // independent work. Sharing one made this read ~39 ms of kernels for a
    // token the engine decodes in 24.8 ms, which is what exposed it.
    const oBs = Array.from({ length: COPIES }, () => device.createBuffer({ size: CAP * N * 2, usage: ST }))
    const pod = buf(new Uint32Array([KP, SPR, N, M]), GPUBufferUsage.UNIFORM)
    const wide = M > 1
    const pipe = wide ? e5Pipe : matvecPipe
    const bgs = wSets.map((ws, bgi) => device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [oBs[bgi], aB, ws.scB, ws.wB, pod, ws.biB].map((b, j) => ({ binding: j, resource: { buffer: b } })),
    }))
    const bg = bgs[0]
    const grid = wide
      ? [Math.ceil(N / 32), Math.ceil(M / 64), 1]     // E5 tile: 64(M) x 32(N)
      : [Math.ceil(N / 4), 1, 1]                      // matvec: 4 rows per workgroup

    // A bind group WebGPU rejects makes it skip the dispatch, and a skipped
    // dispatch times as ~0 ms rather than as an error. Validate once first.
    device.pushErrorScope('validation')
    {
      const enc = device.createCommandEncoder(); const p = enc.beginComputePass()
      p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(...grid); p.end()
      device.queue.submit([enc.finish()])
    }
    const err = await device.popErrorScope()
    if (err) throw new Error(`${name} M=${M} did not run: ${err.message.split('\n')[0]}`)

    const once = (n) => {
      const enc = device.createCommandEncoder(); const p = enc.beginComputePass()
      p.setPipeline(pipe)
      for (let i = 0; i < n; i++) { p.setBindGroup(0, bgs[i % bgs.length]); p.dispatchWorkgroups(...grid) }
      p.end(); device.queue.submit([enc.finish()])
    }
    once(3); await device.queue.onSubmittedWorkDone()
    const t0 = performance.now()
    once(ITERS); await device.queue.onSubmittedWorkDone()
    const ms = (performance.now() - t0) / ITERS

    // Weight bytes are what a quantized matmul actually moves: 4 bits per value
    // plus an f16 scale and bias per group of 64.
    const wBytes = wBytes1
    out.push({
      name, M, K, N, ms, copies: COPIES,
      gbPerS: wBytes / (ms / 1000) / 1e9,
      gflops: (2 * M * K * N) / (ms / 1000) / 1e9,
      kernel: wide ? 'int4_matmul_sg_e5_affine' : int4MatmulEntry(MATVEC),
    })
    aB.destroy(); for (const o of oBs) o.destroy(); pod.destroy()
  }
  for (const ws of wSets) { ws.wB.destroy(); ws.scB.destroy(); ws.biB.destroy() }
}

if (JSON_OUT) {
  console.log(JSON.stringify({ engine: 'zero-tvm', iters: ITERS, results: out }))
} else {
  console.log(`\nzero-tvm int4 affine matmul (4-bit, group 64), ${ITERS} dispatches over rotating weight sets\n`)
  console.log(`  ${'shape'.padEnd(13)} ${'M'.padStart(4)} ${'K'.padStart(6)} ${'N'.padStart(6)} `
    + `${'ms'.padStart(8)} ${'GB/s'.padStart(7)} ${'GFLOP/s'.padStart(9)}  kernel`)
  for (const r of out) {
    console.log(`  ${r.name.padEnd(13)} ${String(r.M).padStart(4)} ${String(r.K).padStart(6)} ${String(r.N).padStart(6)} `
      + `${r.ms.toFixed(3).padStart(8)} ${r.gbPerS.toFixed(0).padStart(7)} ${r.gflops.toFixed(0).padStart(9)}  ${r.kernel}`)
  }
}

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
