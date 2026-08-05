// REAL-WEIGHT KERNEL VALIDATION — the piece the synthetic suite cannot cover.
//
// tests/kernels/run.mjs proves a kernel matches OUR OWN reference on OUR OWN
// random data. That catches arithmetic bugs; it cannot catch a misread of a
// vendor's on-disk quantization (nibble order, group size, whether the bias is
// affine or symmetric). Every new model format raises exactly those questions,
// so this file validates a kernel against tensors pulled out of a real
// published checkpoint, with the reference produced by the VENDOR's own library.
//
// Convention (matches the e2e suites): skip LOUDLY when the weights are absent
// rather than failing, so CI stays green on a machine without the checkpoint.
//
//   node scripts/make-kernel-ref.mjs --model qwen36moe    # writes the bundle
//   node tests/kernels/real-weights.mjs
//
// Adding a format: write a producer in scripts/make-kernel-ref.mjs that emits
// the same bundle shape (meta.json + raw .bin files); nothing here changes.
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { resolve, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { getDevice, pipelineFor, buffer, runCompute, BU, MM } from './gpu.mjs'
import { f16Array, f16BitsToF32 } from './half.mjs'
import { int4MatmulWGSL, int4MatmulEntry } from '../../src/compiler/shaders/int4_matmul.gen.ts'
import { withPrelude } from '../../src/compiler/shader-prelude.ts'
import { QWEN35_4B } from '../../src/compiler/model-spec.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const REFS = join(ROOT, '.weights-local/kernel-refs')
const SHADERS = join(ROOT, 'src/compiler/shaders')

const read = (dir, name, T) => {
  const b = readFileSync(join(dir, `${name}.bin`))
  return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT)
}

/** Copy already-computed storage buffers back to the host. */
const readBack = async (device, pairs) => {
  const enc = device.createCommandEncoder()
  const outs = pairs.map(([, bytes]) => device.createBuffer({ size: bytes, usage: BU.COPY_DST | BU.MAP_READ }))
  pairs.forEach(([buf, bytes], i) => enc.copyBufferToBuffer(buf, 0, outs[i], 0, bytes))
  device.queue.submit([enc.finish()])
  await Promise.all(outs.map((b) => b.mapAsync(MM.READ)))
  return outs.map((b) => b.getMappedRange().slice(0))
}

/** int4 affine matmul (MLX layout) against a bundle produced from real weights. */
async function affineMatmul(device, dir, meta) {
  const { N, K, K_PACKED: KP, GROUPS_PER_ROW: SPR } = meta
  const weights = read(dir, 'weights_u32', Uint32Array)
  const scales = read(dir, 'scales_f16', Uint16Array)
  const bias2 = read(dir, 'bias2_f16', Uint16Array)
  const x = read(dir, 'x_f16', Uint16Array)
  const ref = read(dir, 'y_ref_f32', Float32Array)

  // withPrelude keeps `enable` directives first — WGSL rejects them after globals.
  const src = withPrelude(int4MatmulWGSL({ affine: true }))
  const entry = int4MatmulEntry({ affine: true })
  const outBytes = N * 2
  const out = device.createBuffer({ size: outBytes, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, x, BU.STORAGE | BU.COPY_DST),
    buffer(device, scales, BU.STORAGE | BU.COPY_DST),
    buffer(device, weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([KP, SPR, N, 0]), BU.UNIFORM | BU.COPY_DST),
    buffer(device, bias2, BU.STORAGE | BU.COPY_DST), // @binding(5)
  ]
  const bytes = await runCompute(device, pipelineFor(device, src, entry), buffers, [N], 0, outBytes)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const scale = Math.max(...Array.from(ref, Math.abs))
  const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / scale))
  return { pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)} (vs ${meta.reference})` }
}


// ── MoE block: router + K+1 expert slots, 6 dispatches ───────────────────────
//
// The whole block, in the layout the engine's loader is meant to build:
// the shared expert is index E of every stacked expert tensor and its gate is
// row E of the router, so nothing about it is a special case. Slot selection is
// grid `z` plus an ids[] lookup, so the expert never touches a bind offset.
//
//   router_logits (E+1 wg) -> router_topk (1 wg)
//     -> gate (F/4,1,SLOTS) -> up (F/4,1,SLOTS) -> silu_mul -> down (D/4,1,SLOTS)
//     -> combine
//
async function moeBlock(device, dir, meta) {
  const D = meta.hidden, F = meta.moe_intermediate, K = meta.top_k
  const SLOTS = K + 1                       // top-k routed + 1 shared
  const rd = (n, T) => read(dir, n, T)
  const x = rd('x_f16', Uint16Array)
  const yRef = rd('y_ref_f32', Float32Array)
  const idx = rd('topk_idx_u32', Uint32Array)
  const score = rd('topk_score_f32', Float32Array)
  const st = (n, T) => buffer(device, rd(n, T), BU.STORAGE | BU.COPY_DST)

  const W = {}
  for (const proj of ['gate_proj', 'up_proj', 'down_proj']) {
    W[`${proj}_w`] = st(`exp_${proj}_w_u32`, Uint32Array)
    W[`${proj}_s`] = st(`exp_${proj}_s_f16`, Uint16Array)
    W[`${proj}_b`] = st(`exp_${proj}_b2_f16`, Uint16Array)
  }
  const xBuf = buffer(device, x, BU.STORAGE | BU.COPY_DST)

  // ---- router: 8-bit affine matvec (one workgroup per expert) then top-k ----
  const E = meta.num_experts
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8')), e)
  const rtL = shader('moe_router_logits.wgsl', 'moe_router_logits')
  const rtK = shader('moe_router_topk.wgsl', 'moe_router_topk')
  const rIdx = device.createBuffer({ size: SLOTS * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const rScore = device.createBuffer({ size: SLOTS * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const rLogits = device.createBuffer({ size: (E + 1) * 4, usage: BU.STORAGE })
  const bg = (p, bufs) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(rtL)
    pass.setBindGroup(0, bg(rtL, [
      rLogits, xBuf,
      st('router_w_u32', Uint32Array), st('router_s_f16', Uint16Array), st('router_b_f16', Uint16Array),
      buffer(device, new Uint32Array([D, E + 1]), BU.UNIFORM | BU.COPY_DST),
    ]))
    pass.dispatchWorkgroups(E + 1)   // row E is the shared-expert gate
    pass.setPipeline(rtK)
    pass.setBindGroup(0, bg(rtK, [rIdx, rScore, rLogits,
      buffer(device, new Uint32Array([E, K, meta.norm_topk_prob ? 1 : 0]), BU.UNIFORM | BU.COPY_DST)]))
    pass.dispatchWorkgroups(1)
    pass.end()
    device.queue.submit([enc.finish()])
  }
  const [idxBytes, scBytes] = await readBack(device, [[rIdx, SLOTS * 4], [rScore, SLOTS * 4]])
  // The last slot is the shared expert, which the reference reports separately.
  const gpuIdx = new Uint32Array(idxBytes).subarray(0, K)
  const gpuScore = new Float32Array(scBytes).subarray(0, K)

  // Order-invariant: the kernel emits descending, mlx ascending; the block sum
  // does not depend on either.
  const byIdx = (a, b) => a[0] - b[0]
  const refPairs = Array.from(idx, (v, i2) => [v, score[i2]]).sort(byIdx)
  const gotPairs = Array.from(gpuIdx, (v, i2) => [v, gpuScore[i2]]).sort(byIdx)
  if (!refPairs.every((r, i2) => r[0] === gotPairs[i2][0])) {
    return { pass: false, detail: `router picked ${[...gpuIdx].sort((a, b) => a - b)}, reference ${[...idx].sort((a, b) => a - b)}` }
  }
  const scoreErr = Math.max(...refPairs.map((r, i2) => Math.abs(r[1] - gotPairs[i2][1]) / r[1]))

  // ---- buffers laid out by slot so one dispatch covers every expert ----
  const gu = device.createBuffer({ size: SLOTS * 2 * F * 2, usage: BU.STORAGE })     // [slot][gate|up]
  const h = device.createBuffer({ size: SLOTS * F * 2, usage: BU.STORAGE })
  const dOut = device.createBuffer({ size: SLOTS * D * 2, usage: BU.STORAGE })
  const acc = device.createBuffer({ size: D * 4, usage: BU.STORAGE | BU.COPY_SRC })

  // Tiled + subgroups: 4 output rows per workgroup, subgroupAdd instead of a
  // tree reduction, so the 8 activations a thread loads are reused 4 times.
  const MOE_OPTS = { affine: true, moe: true, subgroups: true, rowsPerWG: 4 }
  const mmMoe = pipelineFor(device, withPrelude(int4MatmulWGSL(MOE_OPTS)), int4MatmulEntry(MOE_OPTS))
  const RPW = 4
  const silu = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'silu_mul.wgsl'), 'utf8'),
                                               { ...QWEN35_4B, ffn: F }), 'silu_mul')
  const comb = shader('moe_combine.wgsl', 'moe_combine')

  const u32 = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  // IN_SLOT_STRIDE 0 = every slot reads the same activation (gate/up); D/F for down.
  const podGateMoe = u32([D / 8, D / 64, F, 0, 2 * F])
  const podDownMoe = u32([F / 8, F / 64, D, F, D])

  const bgMoe = (out, outOff, outSize, inp, proj, pod) => device.createBindGroup({
    layout: mmMoe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: out, offset: outOff, size: outSize } },
      { binding: 1, resource: { buffer: inp } },
      { binding: 2, resource: { buffer: W[`${proj}_s`] } },
      { binding: 3, resource: { buffer: W[`${proj}_w`] } },
      { binding: 4, resource: { buffer: pod } },
      { binding: 5, resource: { buffer: W[`${proj}_b`] } },
      { binding: 6, resource: { buffer: rIdx } },
    ],
  })
  const GU_SLOT = 2 * F * 2                 // bytes per slot in `gu`
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()

  // Every slot — the K routed experts AND the shared one — in the same dispatch.
  pass.setPipeline(mmMoe)
  pass.setBindGroup(0, bgMoe(gu, 0, SLOTS * GU_SLOT, xBuf, 'gate_proj', podGateMoe))
  pass.dispatchWorkgroups(F / RPW, 1, SLOTS)
  // up writes the second half of each slot: same kernel, output bound 1024 B in.
  pass.setBindGroup(0, bgMoe(gu, F * 2, SLOTS * GU_SLOT - F * 2, xBuf, 'up_proj', podGateMoe))
  pass.dispatchWorkgroups(F / RPW, 1, SLOTS)

  pass.setPipeline(silu)
  pass.setBindGroup(0, device.createBindGroup({
    layout: silu.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: h } }, { binding: 1, resource: { buffer: gu } },
              { binding: 2, resource: { buffer: buffer(device, new Int32Array([SLOTS, Math.ceil(SLOTS * F / 256)]), BU.UNIFORM | BU.COPY_DST) } }],
  }))
  pass.dispatchWorkgroups(Math.ceil(SLOTS * F / 256))

  pass.setPipeline(mmMoe)
  pass.setBindGroup(0, bgMoe(dOut, 0, SLOTS * D * 2, h, 'down_proj', podDownMoe))
  pass.dispatchWorkgroups(D / RPW, 1, SLOTS)

  pass.setPipeline(comb)
  pass.setBindGroup(0, bg(comb, [acc, dOut, rScore, u32([D, SLOTS])]))
  pass.dispatchWorkgroups(Math.ceil(D / 256))
  pass.end()

  const outBuf = device.createBuffer({ size: D * 4, usage: BU.COPY_DST | BU.MAP_READ })
  enc.copyBufferToBuffer(acc, 0, outBuf, 0, D * 4)
  device.queue.submit([enc.finish()])
  await outBuf.mapAsync(MM.READ)
  const got = new Float32Array(outBuf.getMappedRange().slice(0))

  // Tolerance is set by the REFERENCE's precision, not ours: mlx runs the block
  // in bfloat16 (~4e-3) and gate→silu→down→sum compounds it. Evidence this is
  // not covering a bug — the same algorithm in f32 on CPU gives the same figure,
  // and the error is spread across all 2048 outputs rather than concentrated in
  // a few, which is what an indexing or routing mistake looks like.
  const scale = Math.max(...Array.from(yRef, Math.abs))
  let maxRel = 0
  for (let i2 = 0; i2 < D; i2++) maxRel = Math.max(maxRel, Math.abs(got[i2] - yRef[i2]) / scale)
  return {
    pass: maxRel < 0.012 && scoreErr < 0.01,
    detail: `router top-${K} exact, scores ${scoreErr.toExponential(2)} | full block ${maxRel.toExponential(2)} | ${SLOTS} slots in 6 dispatches`,
  }
}

const KERNELS = { affine_matmul: affineMatmul, moe_block: moeBlock }

async function main() {
  if (!existsSync(REFS) || readdirSync(REFS).length === 0) {
    console.log('SKIP  real-weight kernel checks — no bundles in .weights-local/kernel-refs')
    console.log('      produce one with:  node scripts/make-kernel-ref.mjs --model qwen36moe')
    process.exit(0)
  }
  const { device, info } = await getDevice({ f16: true })
  console.log(`adapter: ${info.description || info.vendor || 'unknown'}`)
  let failed = 0
  for (const name of readdirSync(REFS)) {
    const dir = join(REFS, name)
    const metaPath = join(dir, 'meta.json')
    if (!existsSync(metaPath)) continue
    const meta = JSON.parse(readFileSync(metaPath, 'utf8'))
    const fn = KERNELS[meta.kernel]
    if (!fn) {
      console.log(`SKIP  ${name.padEnd(22)} unknown kernel '${meta.kernel}'`)
      continue
    }
    const res = await fn(device, dir, meta)
    if (!res.pass) failed++
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${name.padEnd(22)} ${res.detail}`)
    console.log(`      source: ${meta.model} :: ${meta.tensor}`)
  }
  console.log(failed ? `\n${failed} real-weight check(s) FAILED` : '\nreal-weight checks correct')
  process.exit(failed ? 1 : 0)
}
main().catch((e) => { console.error(e); process.exit(1) })
