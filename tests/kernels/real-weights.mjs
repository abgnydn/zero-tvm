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
import { DEEPSEEK_V2_LITE, QWEN35_4B, QWEN36_35B_A3B, ropeInvFreqTable } from '../../src/compiler/model-spec.ts'

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

/** int4/int3 affine matmul (MLX layout) against a bundle from real weights.
 *  The bundle's meta.kernel picks the variant, so one function serves both. */
async function affineMatmul(device, dir, meta) {
  const { N, K, K_PACKED: KP, GROUPS_PER_ROW: SPR } = meta
  const q3 = meta.kernel === 'affine_matmul_q3'
  const weights = read(dir, 'weights_u32', Uint32Array)
  const scales = read(dir, 'scales_f16', Uint16Array)
  const bias = read(dir, 'bias_f16', Uint16Array)
  const x = read(dir, 'x_f16', Uint16Array)
  const ref = read(dir, 'y_ref_f32', Float32Array)

  // withPrelude keeps `enable` directives first — WGSL rejects them after globals.
  const src = withPrelude(int4MatmulWGSL({ affine: true, q3 }))
  const entry = int4MatmulEntry({ affine: true, q3 })
  const outBytes = N * 2
  const out = device.createBuffer({ size: outBytes, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, x, BU.STORAGE | BU.COPY_DST),
    buffer(device, scales, BU.STORAGE | BU.COPY_DST),
    buffer(device, weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([KP, SPR, N, 0]), BU.UNIFORM | BU.COPY_DST),
    buffer(device, bias, BU.STORAGE | BU.COPY_DST), // @binding(5)
  ]
  const bytes = await runCompute(device, pipelineFor(device, src, entry), buffers, [N], 0, outBytes)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const scale = Math.max(...Array.from(ref, Math.abs))
  const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / scale))
  return { pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)} (vs ${meta.reference})` }
}


// ── MoE block: router + K+1 expert slots, 7 dispatches ───────────────────────
//
// The whole block, in the layout the engine's loader is meant to build:
// the shared expert is index E of every stacked expert tensor and its gate is
// row E of the router, so nothing about it is a special case. Slot selection is
// grid `z` plus an ids[] lookup, so the expert never touches a bind offset.
//
//   router_logits (E+1 wg) -> router_topk (1 wg)
//     -> gate (F/4,1,SLOTS) -> up (F/4,1,SLOTS) -> silu_mul -> down (D/4,1,SLOTS)
//     -> combine                                              [7 dispatches]
//
async function moeBlock(device, dir, meta) {
  const D = meta.hidden, F = meta.moe_intermediate, K = meta.top_k
  const bits = meta.bits ?? 4               // expert precision; router is always 8
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
    W[`${proj}_b`] = st(`exp_${proj}_b_f16`, Uint16Array)
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
      // hasShared = 1: Qwen3.6 has a shared expert. This field is not optional —
      // moe_router_topk's PODArgs grew from three u32 to four when MoE-without-a-
      // shared-expert landed, and a 12-byte uniform against a 16-byte struct is a
      // bind-group validation failure. WebGPU answers that by skipping the
      // dispatch, so the output keeps its zeros and reads as "the router picked
      // expert 0 eight times" rather than as an error.
      buffer(device, new Uint32Array([E, K, meta.norm_topk_prob ? 1 : 0, 1]), BU.UNIFORM | BU.COPY_DST)]))
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
  const acc = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })

  // Tiled + subgroups: 4 output rows per workgroup, subgroupAdd instead of a
  // tree reduction, so the 8 activations a thread loads are reused 4 times.
  const MOE_OPTS = { affine: true, moe: true, subgroups: true, rowsPerWG: 4, q3: bits === 3 }
  const mmMoe = pipelineFor(device, withPrelude(int4MatmulWGSL(MOE_OPTS)), int4MatmulEntry(MOE_OPTS))
  // The wide-load sibling, graded against the SAME mlx reference. 4-bit only —
  // q3's bitstream has no wide path. Both K's in this block (D=2048 for
  // gate/up, F=512 for down) satisfy the K % 512 gate, so this grades every
  // dispatch the engine would route wide.
  const WIDE_OPTS = { ...MOE_OPTS, vec4Half: true }
  const mmWide = bits === 4
    ? pipelineFor(device, withPrelude(int4MatmulWGSL(WIDE_OPTS)), int4MatmulEntry(WIDE_OPTS))
    : null
  const RPW = 4
  const silu = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'silu_mul.wgsl'), 'utf8'),
                                               { ...QWEN35_4B, ffn: F }), 'silu_mul')
  const comb = shader('moe_combine.wgsl', 'moe_combine')

  const u32 = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  // IN_SLOT_STRIDE 0 = every slot reads the same activation (gate/up); D/F for down.
  // K_PACKED is words per row: K*bits/32 (4-bit: K/8, 3-bit: 3K/32).
  // EIGHT u32: the five slot fields plus the three TOKEN strides chunked
  // prefill dispatches on. This test is one token, so grid y is 1 and the token
  // strides never multiply by anything — but they are part of the struct, and a
  // 20-byte uniform against a 32-byte one is a bind failure, not a wrong number.
  const podGateMoe = u32([(D * bits) / 32, D / 64, F, 0, 2 * F, D, SLOTS * 2 * F, SLOTS])
  const podDownMoe = u32([(F * bits) / 32, F / 64, D, F, D, SLOTS * F, SLOTS * D, SLOTS])

  const bgMoe = (pipe, out, outOff, outSize, inp, proj, pod) => device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
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
  pass.setBindGroup(0, bgMoe(mmMoe, gu, 0, SLOTS * GU_SLOT, xBuf, 'gate_proj', podGateMoe))
  pass.dispatchWorkgroups(F / RPW, 1, SLOTS)
  // up writes the second half of each slot: same kernel, output bound 1024 B in.
  pass.setBindGroup(0, bgMoe(mmMoe, gu, F * 2, SLOTS * GU_SLOT - F * 2, xBuf, 'up_proj', podGateMoe))
  pass.dispatchWorkgroups(F / RPW, 1, SLOTS)

  pass.setPipeline(silu)
  pass.setBindGroup(0, device.createBindGroup({
    layout: silu.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: h } }, { binding: 1, resource: { buffer: gu } },
              { binding: 2, resource: { buffer: buffer(device, new Int32Array([SLOTS, Math.ceil(SLOTS * F / 256)]), BU.UNIFORM | BU.COPY_DST) } }],
  }))
  pass.dispatchWorkgroups(Math.ceil(SLOTS * F / 256))

  pass.setPipeline(mmMoe)
  pass.setBindGroup(0, bgMoe(mmMoe, dOut, 0, SLOTS * D * 2, h, 'down_proj', podDownMoe))
  pass.dispatchWorkgroups(D / RPW, 1, SLOTS)

  pass.setPipeline(comb)
  pass.setBindGroup(0, bg(comb, [acc, dOut, rScore, u32([D, SLOTS])]))
  pass.dispatchWorkgroups(Math.ceil(D / 256))
  pass.end()

  const outBuf = device.createBuffer({ size: D * 2, usage: BU.COPY_DST | BU.MAP_READ })
  enc.copyBufferToBuffer(acc, 0, outBuf, 0, D * 2)
  device.queue.submit([enc.finish()])
  await outBuf.mapAsync(MM.READ)
  const got = Array.from(new Uint16Array(outBuf.getMappedRange().slice(0)), f16BitsToF32)

  // The reference runs in f32, so this tolerance is set by OUR f16 activations
  // and int4 weights rather than by the reference's own rounding. It was 0.012
  // when the reference was bfloat16 — that is not a tolerance, it is the
  // reference's noise floor, and it is wide enough to hide a real bug.
  const scale = Math.max(...Array.from(yRef, Math.abs))
  let maxRel = 0
  for (let i2 = 0; i2 < D; i2++) maxRel = Math.max(maxRel, Math.abs(got[i2] - yRef[i2]) / scale)

  // Wide arm: the SAME chain re-recorded with the vec4h pipeline into the same
  // buffers (every write covers its whole region, and the narrow readback has
  // completed). Graded against the same reference at the same bar — a wide
  // kernel that read slot 0's activations for every slot (the bug the old ban
  // guarded against) fails this by orders of magnitude, not by epsilon.
  let maxRelW = -1
  if (mmWide) {
    const encW = device.createCommandEncoder()
    const passW = encW.beginComputePass()
    passW.setPipeline(mmWide)
    passW.setBindGroup(0, bgMoe(mmWide, gu, 0, SLOTS * GU_SLOT, xBuf, 'gate_proj', podGateMoe))
    passW.dispatchWorkgroups(F / RPW, 1, SLOTS)
    passW.setBindGroup(0, bgMoe(mmWide, gu, F * 2, SLOTS * GU_SLOT - F * 2, xBuf, 'up_proj', podGateMoe))
    passW.dispatchWorkgroups(F / RPW, 1, SLOTS)
    passW.setPipeline(silu)
    passW.setBindGroup(0, device.createBindGroup({
      layout: silu.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: h } }, { binding: 1, resource: { buffer: gu } },
                { binding: 2, resource: { buffer: buffer(device, new Int32Array([SLOTS, Math.ceil(SLOTS * F / 256)]), BU.UNIFORM | BU.COPY_DST) } }],
    }))
    passW.dispatchWorkgroups(Math.ceil(SLOTS * F / 256))
    passW.setPipeline(mmWide)
    passW.setBindGroup(0, bgMoe(mmWide, dOut, 0, SLOTS * D * 2, h, 'down_proj', podDownMoe))
    passW.dispatchWorkgroups(D / RPW, 1, SLOTS)
    passW.setPipeline(comb)
    passW.setBindGroup(0, bg(comb, [acc, dOut, rScore, u32([D, SLOTS])]))
    passW.dispatchWorkgroups(Math.ceil(D / 256))
    passW.end()
    const outW = device.createBuffer({ size: D * 2, usage: BU.COPY_DST | BU.MAP_READ })
    encW.copyBufferToBuffer(acc, 0, outW, 0, D * 2)
    device.queue.submit([encW.finish()])
    await outW.mapAsync(MM.READ)
    const gotW = Array.from(new Uint16Array(outW.getMappedRange().slice(0)), f16BitsToF32)
    maxRelW = 0
    for (let i2 = 0; i2 < D; i2++) maxRelW = Math.max(maxRelW, Math.abs(gotW[i2] - yRef[i2]) / scale)
  }

  const widePass = mmWide ? maxRelW < 0.005 : true
  return {
    pass: maxRel < 0.005 && scoreErr < 0.005 && widePass,
    detail: `router top-${K} exact, scores ${scoreErr.toExponential(2)} | full block ${maxRel.toExponential(2)}`
      + (mmWide ? ` | wide ${maxRelW.toExponential(2)}` : '')
      + ` | ${SLOTS} slots in 7 dispatches`,
  }
}


// The shipped spec, so these tests exercise what the engine will actually use
// rather than a shape derived beside it.
const QWEN36 = QWEN36_35B_A3B

/** Chain kernels in one pass, then read the requested buffers back. */
async function runChain(device, steps, reads) {
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  for (const [pipeline, entries, wg] of steps) {
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((e, i) => ({ binding: i, resource: e.buffer ? e : { buffer: e } })),
    }))
    pass.dispatchWorkgroups(wg[0], wg[1] ?? 1, wg[2] ?? 1)
  }
  pass.end()
  device.queue.submit([enc.finish()])
  return readBack(device, reads)
}

/** max |got - ref| / max|ref| over the whole vector. Loops rather than
 *  spreading into Math.max — these run to 500k elements. */
function relErr(got, ref) {
  let scale = 0
  for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]))
  let m = 0
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]) / scale)
  return m
}

/**
 * Layer 0's gated-DeltaNet sub-block against mlx_lm's own GatedDeltaNet.
 *
 * The GDN kernels are already checked against a JS reference at Qwen3.5 dims
 * (compile-qwen35.mjs). This checks the two things that reference cannot: the
 * weights are MLX-affine rather than MLC-symmetric, and a reference we wrote
 * ourselves can share a misreading with the kernel we wrote ourselves.
 *
 * The four input projections are CONCATENATED into one [gdnProjRows, d] matmul
 * here, the way the engine's loader fuses them, so `z` and `[a|b]` are read out
 * of one buffer by 256-aligned bind offsets — which is also what makes
 * z_stride / ab_stride = gdnProjRows the natural setting.
 */
async function qwen36Gdn(device, dir, meta) {
  const S = QWEN36
  const rd = (n, T) => read(dir, n, T)
  const st = (n, T) => buffer(device, rd(n, T), BU.STORAGE | BU.COPY_DST)
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8'), S), e)
  const D = S.d, QKV = S.gdnQkvDim, VD = S.gdnVDim, ROWS = S.gdnProjRows, VH = S.gdnVHeads

  // One fused [ROWS, D] affine projection: qkv | z | a | b.
  const cat = (names, T) => {
    const parts = names.map((n) => rd(n, T))
    const out = new T(parts.reduce((a, p) => a + p.length, 0))
    let o = 0
    for (const p of parts) { out.set(p, o); o += p.length }
    return buffer(device, out, BU.STORAGE | BU.COPY_DST)
  }
  const P = ['qkv', 'z', 'a', 'b']
  const projW = cat(P.map((n) => `${n}_w_u32`), Uint32Array)
  const projS = cat(P.map((n) => `${n}_s_f16`), Uint16Array)
  const projB = cat(P.map((n) => `${n}_b_f16`), Uint16Array)

  const MM = { affine: true, subgroups: true, rowsPerWG: 4 }
  const F32 = { affine: true, subgroups: true, rowsPerWG: 4, outF32: true }
  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL(MM), S), int4MatmulEntry(MM))
  const mmF32 = pipelineFor(device, withPrelude(int4MatmulWGSL(F32), S), int4MatmulEntry(F32))
  const RPW = 4
  const norm = shader('rms_norm.wgsl', 'rms_norm')
  const conv = shader('gdn_conv.wgsl', 'gdn_conv')
  const gates = shader('gdn_gates.wgsl', 'gdn_gates')
  const recur = shader('gdn_recur.wgsl', 'gdn_recur')
  const gnorm = shader('gdn_norm_out.wgsl', 'gdn_norm_out')

  const u32 = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  const i32 = (a) => buffer(device, new Int32Array(a), BU.UNIFORM | BU.COPY_DST)
  const sb = (n) => device.createBuffer({ size: n, usage: BU.STORAGE | BU.COPY_SRC })

  const xBuf = st('x_f16', Uint16Array)
  const xn = sb(D * 2)
  const proj = sb(ROWS * 2)                 // [qkv | z | a | b], f16
  const convOut = sb(QKV * 2)
  const gateBuf = sb(2 * VH * 4)
  const recurOut = sb(VD * 4)
  const gnormOut = sb(VD * 2)
  const y = sb(D * 4)
  const convState = st('conv_state_f16', Uint16Array)
  const state = buffer(device, rd('recur_state_f32', Float32Array), BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)

  // CONTROL: the same projection driven by mlx's OWN normalised input instead of
  // ours. Separates "our matmul is wrong" from "our f16 input differs slightly
  // and the dot products cancel" — without it the projection's error is the
  // largest in the chain with no way to say why.
  const xnRef = buffer(device, f16Array(rd('xnorm_ref_f32', Float32Array)), BU.STORAGE | BU.COPY_DST)
  const projCtl = sb(ROWS * 2)

  const zOff = QKV * 2, abOff = (QKV + VD) * 2   // 16384 and 24576 bytes — both 256-aligned
  const steps = [
    [norm, [xn, xBuf, st('norm1_gamma_f16', Uint16Array), u32([D])], [1]],
    [mm, [proj, xn, projS, projW, u32([D / 8, D / 64, ROWS]), projB], [ROWS / RPW]],
    [conv, [convOut, { buffer: proj, offset: 0, size: QKV * 2 }, convState,
            st('conv1d_f16', Uint16Array), i32([meta.prefill, Math.ceil(QKV / 256)])],
     [Math.ceil(QKV / 256)]],
    [gates, [gateBuf, { buffer: proj, offset: abOff, size: ROWS * 2 - abOff },
             st('A_log_f32', Float32Array), st('dt_bias_f32', Float32Array), i32([1, ROWS, 1])], [1]],
    [recur, [recurOut, convOut, gateBuf, state, i32([1, VH])], [VH]],
    [gnorm, [gnormOut, recurOut, st('gnorm_gamma_f16', Uint16Array),
             { buffer: proj, offset: zOff, size: ROWS * 2 - zOff }, i32([1, ROWS, VH])], [VH]],
    // out_proj writes f32 so the comparison is not limited by an f16 round-trip.
    [mmF32, [y, gnormOut, st('out_s_f16', Uint16Array), st('out_w_u32', Uint32Array),
             u32([VD / 8, VD / 64, D]), st('out_b_f16', Uint16Array)], [D / RPW]],
    [mm, [projCtl, xnRef, projS, projW, u32([D / 8, D / 64, ROWS]), projB], [ROWS / RPW]],
  ]

  // Every stage is compared, not just the ends: an error that grows smoothly is
  // f16/int4 compounding through a recurrence, one that jumps is a bug, and the
  // two are indistinguishable from the block output alone.
  const bytes = await runChain(device, steps, [
    [xn, D * 2], [proj, ROWS * 2], [convOut, QKV * 2], [recurOut, VD * 4],
    [gnormOut, VD * 2], [y, D * 4], [state, VH * S.gdnStatePerHead * 4],
    [projCtl, ROWS * 2],
  ])
  const h = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
  const f = (b) => new Float32Array(b)
  const stages = [
    ['input_layernorm', h(bytes[0]), 'xnorm_ref_f32'],
    ['fused proj', h(bytes[1]), 'proj_ref_f32'],
    ['conv+silu', h(bytes[2]), 'conv_ref_f32'],
    ['recurrence', f(bytes[3]), 'recur_ref_f32'],
    ['gated norm', h(bytes[4]), 'gnorm_ref_f32'],
    ['out_proj', f(bytes[5]), 'y_ref_f32'],
    ['state out', f(bytes[6]), 'recur_state_out_f32'],
    ['proj(ref x)', h(bytes[7]), 'proj_ref_f32'],
  ].map(([name, got, ref]) => [name, relErr(got, rd(ref, Float32Array))])

  // The reference is f32, so these numbers are ours: f16 activations through
  // int4 weights. They should rise SMOOTHLY along the chain (~5e-4 at the first
  // projection to ~2e-3 after out_proj) — a jump between neighbours is a bug,
  // a slow climb is compounding. Reading the block output alone cannot tell
  // those apart, which is why every stage is compared.
  const worst = Math.max(...stages.map(([, e]) => e))
  return {
    pass: worst < 0.005,
    detail: stages.map(([n, e]) => `${n} ${e.toExponential(2)}`).join(' | ')
          + ` (${ROWS}-row fused proj, 7 dispatches)`,
  }
}


/**
 * Layer 3's gated-attention sub-block against mlx_lm's own Qwen3NextAttention.
 * (is_linear = (i+1) % full_attention_interval != 0, so 3, 7, ... 39 are the
 * ten full-attention layers.)
 *
 * q_proj emits 8192 rows for 16 heads x 256: per head the first 256 are the
 * query and the next 256 are the SIGMOID GATE — already the interleaving
 * gated_qkv_split expects, so c_attn is just q_proj ++ k_proj ++ v_proj.
 *
 * The KV pages are SEEDED FROM MLX'S OWN CACHE rather than built by prefilling
 * through our chain. That is the stronger test: our RoPE has to agree with
 * mlx's for the current token AND our page layout has to match what mlx stored,
 * whereas a self-prefilled cache would be consistent with our own conventions
 * even if both were wrong.
 */
async function qwen36Attn(device, dir, meta) {
  const S = QWEN36
  const rd = (n, T) => read(dir, n, T)
  const st = (n, T) => buffer(device, rd(n, T), BU.STORAGE | BU.COPY_DST)
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8'), S), e)
  const D = S.d, HD = S.headDim, POS = meta.pos, PAGES = 1   // 4 tokens fit one 16-slot page

  const cat = (names, T) => {
    const parts = names.map((n) => rd(n, T))
    const out = new T(parts.reduce((a, p) => a + p.length, 0))
    let o = 0
    for (const p of parts) { out.set(p, o); o += p.length }
    return buffer(device, out, BU.STORAGE | BU.COPY_DST)
  }
  const P = ['q', 'k', 'v']
  const cW = cat(P.map((n) => `${n}_w_u32`), Uint32Array)
  const cS = cat(P.map((n) => `${n}_s_f16`), Uint16Array)
  const cB = cat(P.map((n) => `${n}_b_f16`), Uint16Array)

  const MM = { affine: true, subgroups: true, rowsPerWG: 4 }
  const F32 = { ...MM, outF32: true }
  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL(MM), S), int4MatmulEntry(MM))
  const mmF32 = pipelineFor(device, withPrelude(int4MatmulWGSL(F32), S), int4MatmulEntry(F32))
  const RPW = 4
  const norm = shader('rms_norm.wgsl', 'rms_norm')
  const split = shader('gated_qkv_split.wgsl', 'gated_qkv_split')
  const qkNorm = shader('qk_norm.wgsl', 'qk_norm')
  const rope = shader('rope.wgsl', 'rope_kernel')
  const kvAppend = shader('kv_append.wgsl', 'kv_append')
  const attention = shader('attention.wgsl', 'attention')
  const gateK = shader('attn_gate.wgsl', 'attn_gate')

  const u32 = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  const i32 = (a) => buffer(device, new Int32Array(a), BU.UNIFORM | BU.COPY_DST)
  const sb = (n) => device.createBuffer({ size: n, usage: BU.STORAGE | BU.COPY_SRC })

  // mlx cache is [1, kvHeads, seq, headDim]; the page is
  // [head][slot][dim] for K, the same shifted by vPageOffset for V.
  const seedPages = () => {
    const pg = new Uint16Array(PAGES * S.kvPageStride)
    const k = rd('k_cache_f16', Uint16Array), v = rd('v_cache_f16', Uint16Array)
    for (let h = 0; h < S.kvHeads; h++) {
      for (let p = 0; p < POS; p++) {
        for (let d = 0; d < HD; d++) {
          const src = (h * POS + p) * HD + d
          const dst = h * S.headPageStride + p * HD + d
          pg[dst] = k[src]
          pg[dst + S.vPageOffset] = v[src]
        }
      }
    }
    return buffer(device, pg, BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
  }

  const xn = sb(D * 2)
  const cAttn = sb(S.cAttnDim * 2)
  const qkv = sb(S.qkvDim * 2)
  const gate = sb(S.qDim * 2)
  const qB = sb(S.qDim * 2), kB = sb(S.kvDim * 2), vB = sb(S.kvDim * 2)
  const attnOut = sb(S.qDim * 2)
  const y = sb(D * 4)
  const pages = seedPages()
  // position_map / length_info are STORAGE, not uniform — rope, kv_append and
  // attention all read them as arrays.
  const pos = buffer(device, new Int32Array([POS]), BU.STORAGE | BU.COPY_DST)
  const len = buffer(device, new Int32Array([POS + 1]), BU.STORAGE | BU.COPY_DST)
  const WGS_NORM = S.heads + S.kvHeads

  const steps = [
    [norm, [xn, st('x_f16', Uint16Array), st('norm1_gamma_f16', Uint16Array), u32([D])], [1]],
    [mm, [cAttn, xn, cS, cW, u32([D / 8, D / 64, S.cAttnDim]), cB], [S.cAttnDim / RPW]],
    [split, [qkv, gate, cAttn, i32([1, S.cAttnDim / 256])], [S.cAttnDim / 256]],
    [qkNorm, [qkv, st('q_norm_f16', Uint16Array), st('k_norm_f16', Uint16Array),
              i32([1, WGS_NORM])], [WGS_NORM]],
    [rope, [qB, kB, vB, qkv, pos, u32([1, 0, 1, S.qkvDim / 256]),
            buffer(device, ropeInvFreqTable(S), BU.STORAGE | BU.COPY_DST)], [S.qkvDim / 256]],
    [kvAppend, [kB, vB, pages, pos, u32([1, PAGES, 0, 0, S.kvDim / 256])], [S.kvDim / 256]],
    [attention, [qB, buffer(device, new Int32Array([0, PAGES]), BU.STORAGE | BU.COPY_DST),
                 buffer(device, new Int32Array([0]), BU.STORAGE | BU.COPY_DST), pages, len, attnOut,
                 buffer(device, (() => { const b = new ArrayBuffer(64); const iv = new Int32Array(b)
                   iv[0] = 1; iv[1] = PAGES; iv[2] = PAGES; new Float32Array(b)[7] = 1 / Math.sqrt(HD)
                   iv[8] = 1; return new Uint8Array(b) })(), BU.UNIFORM | BU.COPY_DST)], [1, S.heads]],
    [gateK, [attnOut, gate, u32([S.qDim / 256])], [S.qDim / 256]],
    [mmF32, [y, attnOut, st('o_s_f16', Uint16Array), st('o_w_u32', Uint32Array),
             u32([S.qDim / 8, S.qDim / 64, D]), st('o_b_f16', Uint16Array)], [D / RPW]],
  ]

  const bytes = await runChain(device, steps, [
    [xn, D * 2], [cAttn, S.cAttnDim * 2], [qB, S.qDim * 2], [kB, S.kvDim * 2],
    [attnOut, S.qDim * 2], [y, D * 4],
  ])
  const h = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
  const stages = [
    ['input_layernorm', h(bytes[0]), 'xnorm_ref_f32'],
    ['c_attn', h(bytes[1]), 'cattn_ref_f32'],
    ['q norm+rope', h(bytes[2]), 'q_rope_ref_f32'],
    ['k norm+rope', h(bytes[3]), 'k_rope_ref_f32'],
    ['attn*sigmoid(gate)', h(bytes[4]), 'gated_ref_f32'],
    ['o_proj', new Float32Array(bytes[5]), 'y_ref_f32'],
  ].map(([name, got, ref]) => [name, relErr(got, rd(ref, Float32Array))])

  const worst = Math.max(...stages.map(([, e]) => e))
  return {
    pass: worst < 0.005,
    detail: stages.map(([n, e]) => `${n} ${e.toExponential(2)}`).join(' | ')
          + ` (pos ${POS}, KV pages seeded from mlx, 9 dispatches)`,
  }
}


/**
 * A WHOLE decoder layer against mlx_lm's DecoderLayer:
 *
 *   r = linear_attn(input_layernorm(x));  h = x + r
 *   y = h + mlp(post_attention_layernorm(h))
 *
 * The other three bundles each prove one sub-block. This proves they COMPOSE,
 * which is what engine-core will do — in particular that the GDN block's f16
 * output is the dtype and layout the MoE block reads.
 *
 * Weights come from the qwen36gdn and qwen36moe_block bundles: all three are
 * layer 0, so duplicating 450 MB into a fourth bundle would say nothing new.
 * The qwen36layer bundle owns the x they are all run on, the seeded state, and
 * post_attention_layernorm, which neither of the others carries.
 *
 * The closing `y = h + m` runs on the GPU through add_norm, whose `residual`
 * output is exactly that sum. Its normalised output needs the NEXT layer's
 * gamma, which this bundle has no reason to carry, so that output is ignored
 * and only the residual is compared — which is the point, since it is the one
 * seam nothing else covers: moe_combine writes f16 and add_norm reads f16, and
 * WebGPU would silently accept an f32 buffer there.
 */
async function qwen36Layer(device, dir, meta) {
  const gdnDir = join(REFS, 'qwen36gdn'), moeDir = join(REFS, 'qwen36moe_block')
  if (!existsSync(gdnDir) || !existsSync(moeDir)) {
    return { pass: null, detail: 'needs the qwen36gdn and qwen36moe_block bundles (same layer 0 weights)' }
  }
  const S = QWEN36
  const rd = (n, T) => read(dir, n, T)
  const stFrom = (d, n, T) => buffer(device, read(d, n, T), BU.STORAGE | BU.COPY_DST)
  const st = (n, T) => stFrom(dir, n, T)
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8'), S), e)
  const D = S.d, F = meta.moe_intermediate, K = meta.top_k, SLOTS = K + 1, E = meta.num_experts
  const QKV = S.gdnQkvDim, VD = S.gdnVDim, ROWS = S.gdnProjRows, VH = S.gdnVHeads

  const cat = (d, names, T) => {
    const parts = names.map((n) => read(d, n, T))
    const out = new T(parts.reduce((a, p) => a + p.length, 0))
    let o = 0
    for (const p of parts) { out.set(p, o); o += p.length }
    return buffer(device, out, BU.STORAGE | BU.COPY_DST)
  }
  const P = ['qkv', 'z', 'a', 'b']
  const projW = cat(gdnDir, P.map((n) => `${n}_w_u32`), Uint32Array)
  const projS = cat(gdnDir, P.map((n) => `${n}_s_f16`), Uint16Array)
  const projB = cat(gdnDir, P.map((n) => `${n}_b_f16`), Uint16Array)
  const W = {}
  for (const proj of ['gate_proj', 'up_proj', 'down_proj']) {
    W[`${proj}_w`] = stFrom(moeDir, `exp_${proj}_w_u32`, Uint32Array)
    W[`${proj}_s`] = stFrom(moeDir, `exp_${proj}_s_f16`, Uint16Array)
    W[`${proj}_b`] = stFrom(moeDir, `exp_${proj}_b_f16`, Uint16Array)
  }

  const MM = { affine: true, subgroups: true, rowsPerWG: 4 }
  const MOE = { ...MM, moe: true }
  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL(MM), S), int4MatmulEntry(MM))
  const mmMoe = pipelineFor(device, withPrelude(int4MatmulWGSL(MOE), S), int4MatmulEntry(MOE))
  const RPW = 4
  const norm = shader('rms_norm.wgsl', 'rms_norm')
  const conv = shader('gdn_conv.wgsl', 'gdn_conv')
  const gates = shader('gdn_gates.wgsl', 'gdn_gates')
  const recur = shader('gdn_recur.wgsl', 'gdn_recur')
  const gnorm = shader('gdn_norm_out.wgsl', 'gdn_norm_out')
  const addNorm = shader('add_norm.wgsl', 'add_norm')
  const rtL = shader('moe_router_logits.wgsl', 'moe_router_logits')
  const rtK = shader('moe_router_topk.wgsl', 'moe_router_topk')
  const silu = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'silu_mul.wgsl'), 'utf8'),
                                               { ...S, ffn: F }), 'silu_mul')
  const comb = shader('moe_combine.wgsl', 'moe_combine')

  const u32 = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  const i32 = (a) => buffer(device, new Int32Array(a), BU.UNIFORM | BU.COPY_DST)
  const sb = (n) => device.createBuffer({ size: n, usage: BU.STORAGE | BU.COPY_SRC })

  const xBuf = st('x_f16', Uint16Array)
  const xn = sb(D * 2), proj = sb(ROWS * 2), convOut = sb(QKV * 2)
  const gateBuf = sb(2 * VH * 4), recurOut = sb(VD * 4), gnormOut = sb(VD * 2)
  const r = sb(D * 2), h = sb(D * 2), n2 = sb(D * 2)
  const rLog = sb((E + 1) * 4), rIdx = sb(SLOTS * 4), rScore = sb(SLOTS * 4)
  const GU = 2 * F * 2
  const gu = sb(SLOTS * GU), hh = sb(SLOTS * F * 2), dOut = sb(SLOTS * D * 2), moeOut = sb(D * 2)
  const y = sb(D * 2), yNorm = sb(D * 2)
  const convState = st('conv_state_f16', Uint16Array)
  const state = buffer(device, rd('recur_state_f32', Float32Array), BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)

  const zOff = QKV * 2, abOff = (QKV + VD) * 2
  const steps = [
    // ── GDN sub-block ──
    [norm, [xn, xBuf, stFrom(gdnDir, 'norm1_gamma_f16', Uint16Array), u32([D])], [1]],
    [mm, [proj, xn, projS, projW, u32([D / 8, D / 64, ROWS]), projB], [ROWS / RPW]],
    [conv, [convOut, { buffer: proj, offset: 0, size: QKV * 2 }, convState,
            stFrom(gdnDir, 'conv1d_f16', Uint16Array), i32([meta.prefill, Math.ceil(QKV / 256)])],
     [Math.ceil(QKV / 256)]],
    [gates, [gateBuf, { buffer: proj, offset: abOff, size: ROWS * 2 - abOff },
             stFrom(gdnDir, 'A_log_f32', Float32Array), stFrom(gdnDir, 'dt_bias_f32', Float32Array),
             i32([1, ROWS, 1])], [1]],
    [recur, [recurOut, convOut, gateBuf, state, i32([1, VH])], [VH]],
    [gnorm, [gnormOut, recurOut, stFrom(gdnDir, 'gnorm_gamma_f16', Uint16Array),
             { buffer: proj, offset: zOff, size: ROWS * 2 - zOff }, i32([1, ROWS, VH])], [VH]],
    [mm, [r, gnormOut, stFrom(gdnDir, 'out_s_f16', Uint16Array), stFrom(gdnDir, 'out_w_u32', Uint32Array),
          u32([VD / 8, VD / 64, D]), stFrom(gdnDir, 'out_b_f16', Uint16Array)], [D / RPW]],
    // ── residual + post-attention norm, one kernel ──
    // add_norm's grid is the TOKEN count (base = batch * D), not ceil(D/256):
    // one workgroup of 256 threads walks a whole row. Dispatching D/256 of them
    // runs seven extra workgroups whose reads fall off the end of the buffer.
    [addNorm, [xBuf, r, st('norm2_gamma_f16', Uint16Array), n2, h, u32([1])], [1]],
    // ── MoE block, 7 dispatches ──
    [rtL, [rLog, n2, stFrom(moeDir, 'router_w_u32', Uint32Array),
           stFrom(moeDir, 'router_s_f16', Uint16Array), stFrom(moeDir, 'router_b_f16', Uint16Array),
           u32([D, E + 1])], [E + 1]],
    [rtK, [rIdx, rScore, rLog, u32([E, K, meta.norm_topk_prob ? 1 : 0, 1])], [1]],   // 1 = has a shared expert
  ]
  const bgMoe = (out, outOff, outSize, inp, proj2, pod) => [
    { buffer: out, offset: outOff, size: outSize }, inp, W[`${proj2}_s`], W[`${proj2}_w`], pod,
    W[`${proj2}_b`], rIdx,
  ]
  const podGate = u32([D / 8, D / 64, F, 0, 2 * F, D, SLOTS * 2 * F, SLOTS])
  const podDown = u32([F / 8, F / 64, D, F, D, SLOTS * F, SLOTS * D, SLOTS])
  steps.push(
    [mmMoe, bgMoe(gu, 0, SLOTS * GU, n2, 'gate_proj', podGate), [F / RPW, 1, SLOTS]],
    [mmMoe, bgMoe(gu, F * 2, SLOTS * GU - F * 2, n2, 'up_proj', podGate), [F / RPW, 1, SLOTS]],
    [silu, [hh, gu, buffer(device, new Int32Array([SLOTS, Math.ceil(SLOTS * F / 256)]),
                           BU.UNIFORM | BU.COPY_DST)], [Math.ceil(SLOTS * F / 256)]],
    [mmMoe, bgMoe(dOut, 0, SLOTS * D * 2, hh, 'down_proj', podDown), [D / RPW, 1, SLOTS]],
    [comb, [moeOut, dOut, rScore, u32([D, SLOTS])], [Math.ceil(D / 256)]],
    // The layer's own closing residual, on the GPU. add_norm's `residual`
    // output IS h + m; its normalised output needs a gamma that belongs to the
    // NEXT layer, so any gamma will do here and only the residual is compared.
    // Running it proves the seam the MoE block feeds: moe_combine writes f16
    // and add_norm reads f16.
    [addNorm, [h, moeOut, st('norm2_gamma_f16', Uint16Array), yNorm, y, u32([1])], [1]],
  )

  const bytes = await runChain(device, steps,
    [[r, D * 2], [h, D * 2], [n2, D * 2], [moeOut, D * 2], [y, D * 2]])
  const hf = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
  const stages = [
    ['linear_attn', hf(bytes[0]), 'r_ref_f32'],
    ['x + r', hf(bytes[1]), 'h_ref_f32'],
    ['post_attn norm', hf(bytes[2]), 'norm2_ref_f32'],
    ['MoE', hf(bytes[3]), 'moe_ref_f32'],
    ['h + MoE', hf(bytes[4]), 'y_ref_f32'],
  ].map(([name, got, ref]) => [name, relErr(got, rd(ref, Float32Array))])

  const worst = Math.max(...stages.map(([, e]) => e))
  return {
    pass: worst < 0.005,
    detail: stages.map(([n, e]) => `${n} ${e.toExponential(2)}`).join(' | ') + ' (16 dispatches, all GPU)',
  }
}


/**
 * The MLX embedding table through embedding_affine.wgsl.
 *
 * embedding.wgsl is hard-symmetric with no bias binding; the MLX table is
 * affine over groups of 64. Feeding one to the other is in bounds and
 * error-free, so this check exists to make that difference loud.
 */
async function affineEmbedding(device, dir, meta) {
  const rd = (n, T) => read(dir, n, T)
  const st = (n, T) => buffer(device, rd(n, T), BU.STORAGE | BU.COPY_DST)
  const D = meta.hidden, N = meta.n_ids
  const S = { ...QWEN36, d: D }
  const pipe = pipelineFor(device,
    withPrelude(readFileSync(join(SHADERS, 'embedding_affine.wgsl'), 'utf8'), S), 'embedding_affine')
  const out = device.createBuffer({ size: N * D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const wgs = Math.ceil(N * D / 256)
  const bufs = [
    out, st('ids_u32', Uint32Array), st('scales_f16', Uint16Array), st('weights_u32', Uint32Array),
    buffer(device, new Int32Array([N, wgs]), BU.UNIFORM | BU.COPY_DST),
    st('bias_f16', Uint16Array),   // binding index == array index -> @binding(5)
  ]
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipe)
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  }))
  pass.dispatchWorkgroups(wgs)
  pass.end()
  device.queue.submit([enc.finish()])
  const [bytes] = await readBack(device, [[out, N * D * 2]])
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const err = relErr(got, rd('y_ref_f32', Float32Array))
  return { pass: err < 0.005, detail: `max rel err ${err.toExponential(2)} over ${N} rows (vs ${meta.reference})` }
}

/** MLA attention on a real DeepSeek layer: score against the latent cache,
 *  softmax, weighted-sum it back. The reference (scripts/make-mla-ref.py) got
 *  these numbers by computing MLA the OTHER way — up-projecting the latent into
 *  per-head K and V and running ordinary attention — so agreement here means
 *  the shader reproduces the vendor's attention, not just our rearrangement of
 *  it. Inputs go to the GPU as f16, which is the floor on the error below. */
async function mlaAttention(device, dir, meta) {
  const rd = (n, T) => read(dir, n, T)
  const { heads, kv_lora: L, rope: R, tokens: T, softmax_scale: scale } = meta
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8')), e)

  const scores = device.createBuffer({ size: heads * T * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const outLat = device.createBuffer({ size: heads * L * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const cacheC = buffer(device, f16Array(rd('ref_c', Float32Array)), BU.STORAGE | BU.COPY_DST)
  const scoreBufs = [
    scores,
    buffer(device, f16Array(rd('ref_qlat', Float32Array)), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(rd('ref_qpe', Float32Array)), BU.STORAGE | BU.COPY_DST),
    cacheC,
    buffer(device, f16Array(rd('ref_kpe', Float32Array)), BU.STORAGE | BU.COPY_DST),
    buffer(device, (() => {
      // {L, R, T} then a f32 scale — mixed types, so it is written by hand.
      const u = new ArrayBuffer(16), dv = new DataView(u)
      dv.setUint32(0, L, true); dv.setUint32(4, R, true)
      dv.setUint32(8, T, true); dv.setFloat32(12, scale, true)
      return new Uint8Array(u)
    })(), BU.UNIFORM | BU.COPY_DST),
  ]
  const combineBufs = [
    outLat, scores, cacheC,
    buffer(device, new Uint32Array([L, T, 0, 0]), BU.UNIFORM | BU.COPY_DST),
  ]

  // The whole chain, not just the attention: q_nope enters latent space through
  // kv_b_proj's K half, attention happens entirely against the latent cache,
  // and the result comes back out through the V half. Both hops are mla_proj —
  // the same kernel, because Wk was transposed at prepare time.
  const N_NOPE = meta.nope
  const qNope = buffer(device, f16Array(rd('ref_qnope', Float32Array)), BU.STORAGE | BU.COPY_DST)
  const qLat = device.createBuffer({ size: heads * L * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const oLatF16 = device.createBuffer({ size: heads * L * 2, usage: BU.STORAGE | BU.COPY_DST })
  const oHeads = device.createBuffer({ size: heads * N_NOPE * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const wkT = buffer(device, f16Array(rd('wk_t', Float32Array)), BU.STORAGE | BU.COPY_DST)
  const wv = buffer(device, f16Array(rd('wv', Float32Array)), BU.STORAGE | BU.COPY_DST)
  const dims = (N, K) => buffer(device, new Uint32Array([N, K, 0, 0]), BU.UNIFORM | BU.COPY_DST)

  // q_lat replaces the reference's, so the attention below is fed by the GPU's
  // own projection — otherwise a wrong Wk orientation would never show up here.
  scoreBufs[1] = qLat

  const proj = shader('mla_proj.wgsl', 'mla_proj')
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  const run = (pipe, bufs, x, y) => {
    pass.setPipeline(pipe)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    }))
    pass.dispatchWorkgroups(x, y, 1)
  }
  const wgs = (n) => Math.ceil(n / 64)
  run(proj, [qLat, qNope, wkT, dims(L, N_NOPE)], wgs(L), heads)
  run(shader('mla_scores.wgsl', 'mla_scores'), scoreBufs, T, heads)
  // scores is read_write in both: stage 2 turns the logits into probabilities
  // in place, so it must not be read back before the combine has run.
  run(shader('mla_combine.wgsl', 'mla_combine'), combineBufs, heads, 1)
  pass.end()
  // out_lat is f32 (it is a reduction target); mla_proj reads f16, so the hop
  // back out is a second pass after a narrowing copy.
  device.queue.submit([enc.finish()])
  const [latBytes] = await readBack(device, [[outLat, heads * L * 4]])
  device.queue.writeBuffer(oLatF16, 0, f16Array(new Float32Array(latBytes)))
  const enc2 = device.createCommandEncoder()
  const p2 = enc2.beginComputePass()
  p2.setPipeline(proj)
  p2.setBindGroup(0, device.createBindGroup({
    layout: proj.getBindGroupLayout(0),
    entries: [oLatF16, wv, dims(N_NOPE, L)].map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))
      .concat([{ binding: 0, resource: { buffer: oHeads } }]),
  }))
  p2.dispatchWorkgroups(wgs(N_NOPE), heads, 1)
  p2.end()
  device.queue.submit([enc2.finish()])
  const [headBytes] = await readBack(device, [[oHeads, heads * N_NOPE * 2]])

  const errLat = relErr(Array.from(new Float32Array(latBytes)), rd('ref_olat', Float32Array))
  const errOut = relErr(Array.from(new Uint16Array(headBytes), f16BitsToF32), rd('ref_oheads', Float32Array))
  // 1e-3, not 1e-2. Observed is 3-4e-4 (the f16 input floor), so a 2e-2 bound
  // left ~50x of headroom — enough for either shader to drop its f32
  // accumulator and still pass, which is the one thing these numbers exist to
  // catch. Inputs are seeded and the kernels deterministic, so a tight bound
  // is not a flaky one.
  return {
    pass: errLat < 1e-3 && errOut < 1e-3,
    detail: `latent context ${errLat.toExponential(2)} | back through kv_b's V half ${errOut.toExponential(2)}`
      + ` (${meta.cache_values_per_token} cached values/token vs ${meta.mha_equivalent_per_token} for MHA,`
      + ` q projected on GPU through Wk^T)`,
  }
}

/** DeepSeek-V2 layer 0's ATTENTION path, end to end on real weights. Two things
 *  are covered nowhere else — the pe-row permutation applied to QUANTIZED rows
 *  (what makes DeepSeek's interleaved RoPE free) and yarn's mscale^2 in the
 *  softmax scale.
 *
 *  NOT the dense FFN, though the bundle carries its weights and reference. That
 *  half is an ordinary SwiGLU already covered by other bundles; what makes
 *  layer 0 interesting for the mixed dense/MoE stack is engine WIRING (two FFN
 *  widths chosen per layer), which is not a kernel question and is still open.
 *  Said here because the earlier version of this comment claimed the FFN was
 *  covered and it never was. */
async function dsv2Layer(device, dir, meta) {
  const rd = (n, T) => read(dir, n, T)
  const { heads, nope: NOPE, rope: R, v: V, kv_lora: L, tokens: T,
          softmax_scale: scale, d: D } = meta
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8')), e)
  const affine = pipelineFor(device, withPrelude(int4MatmulWGSL({ affine: true })),
                             int4MatmulEntry({ affine: true }))
  const raw = (n, T2) => {
    const rec = meta.tensors[`${meta.layer}.${n}`]
    const b = readFileSync(join(dir, rec.file))
    return new T2(b.buffer, b.byteOffset, b.byteLength / T2.BYTES_PER_ELEMENT)
  }
  const f16buf = (n) => buffer(device, f16Array(rd(n, Float32Array)), BU.STORAGE | BU.COPY_DST)
  const u32u = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)

  // ── the de-interleave, as the loader will do it ──
  // Rows of a quantized matrix are independent — the affine scale groups run
  // along the INPUT axis — so weight, scales and biases shuffle together and
  // nothing is dequantized to do it. That independence is the whole reason
  // DeepSeek's interleaved RoPE can be free.
  // src[j] = which ORIGINAL row output row j takes. The de-interleave gathers
  // the even indices then the odd ones, so row j < R/2 comes from 2j and the
  // rest from 2(j-R/2)+1. Writing the inverse of this permutation still yields
  // a plausible matrix — it just computes a different model.
  const src = Array.from({ length: R }, (_, j) => (j < R / 2 ? 2 * j : 2 * (j - R / 2) + 1))
  const permute = (arr, rowLen, moves) => {
    const out = arr.slice()
    for (const [dst, s] of moves) out.set(arr.subarray(s * rowLen, (s + 1) * rowLen), dst * rowLen)
    return out
  }
  const qMoves = []
  for (let h = 0; h < heads; h++) {
    for (let j = 0; j < R; j++) {
      qMoves.push([h * (NOPE + R) + NOPE + j, h * (NOPE + R) + NOPE + src[j]])
    }
  }
  const kvMoves = Array.from({ length: R }, (_, j) => [L + j, L + src[j]])

  /** One affine matvec over rows that may have been shuffled. */
  const matvec = async (name, moves, N, K, xBuf) => {
    const KP = K / 8, SPR = K / 64
    const w = moves ? permute(raw(`${name}.weight`, Uint32Array), KP, moves) : raw(`${name}.weight`, Uint32Array)
    const sc = moves ? permute(raw(`${name}.scales`, Uint16Array), SPR, moves) : raw(`${name}.scales`, Uint16Array)
    const bi = moves ? permute(raw(`${name}.biases`, Uint16Array), SPR, moves) : raw(`${name}.biases`, Uint16Array)
    const out = device.createBuffer({ size: N * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const bytes = await runCompute(device, affine, [
      out, xBuf,
      buffer(device, sc, BU.STORAGE | BU.COPY_DST),
      buffer(device, w, BU.STORAGE | BU.COPY_DST),
      u32u([KP, SPR, N, 0]),
      buffer(device, bi, BU.STORAGE | BU.COPY_DST),
    ], [N], 0, N * 2)
    return Array.from(new Uint16Array(bytes), f16BitsToF32)
  }

  // ref_h1 is [T, D] — every position's normed hidden state. The projections
  // take ONE row, the query's. Handing over the whole matrix silently projects
  // token 0 instead, which produces a fully-formed wrong answer.
  const qi = meta.query_at
  const h1 = buffer(device, f16Array(rd('ref_h1', Float32Array).slice(qi * D, (qi + 1) * D)),
                    BU.STORAGE | BU.COPY_DST)
  const errQ = relErr(await matvec('self_attn.q_proj', qMoves, heads * (NOPE + R), D, h1),
                      rd('ref_qperm', Float32Array))
  const errKv = relErr(await matvec('self_attn.kv_a_proj_with_mqa', kvMoves, L + R, D, h1),
                       rd('ref_kvaperm', Float32Array))

  // ── MLA, from the reference's post-RoPE query and cache ──
  const wgs = (n) => Math.ceil(n / 64)
  const dims = (N, K) => u32u([N, K, 0, 0])
  const cacheC = f16buf('ref_c')
  const qLat = device.createBuffer({ size: heads * L * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const scores = device.createBuffer({ size: heads * T * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const outLat = device.createBuffer({ size: heads * L * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const oLat16 = device.createBuffer({ size: heads * L * 2, usage: BU.STORAGE | BU.COPY_DST })
  const oHeads = device.createBuffer({ size: heads * V * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const mlaProj = shader('mla_proj.wgsl', 'mla_proj')

  const pass1 = device.createCommandEncoder()
  const p1 = pass1.beginComputePass()
  const run = (pass, pipe, bufs, x, y) => {
    pass.setPipeline(pipe)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    }))
    pass.dispatchWorkgroups(x, y, 1)
  }
  run(p1, mlaProj, [qLat, f16buf('ref_qnope'), f16buf('wk_t'), dims(L, NOPE)], wgs(L), heads)
  // The scale carries yarn's mscale^2 (1.59 here) on top of 1/sqrt(nope+rope).
  // Leaving it out is a 1.59x error on every logit and looks like nothing.
  const skU = (() => {
    const u = new ArrayBuffer(16), dv = new DataView(u)
    dv.setUint32(0, L, true); dv.setUint32(4, R, true)
    dv.setUint32(8, T, true); dv.setFloat32(12, scale, true)
    return buffer(device, new Uint8Array(u), BU.UNIFORM | BU.COPY_DST)
  })()
  run(p1, shader('mla_scores.wgsl', 'mla_scores'),
      [scores, qLat, f16buf('ref_qpe'), cacheC, f16buf('ref_kpe'), skU], T, heads)
  run(p1, shader('mla_combine.wgsl', 'mla_combine'), [outLat, scores, cacheC, u32u([L, T, 0, 0])], heads, 1)
  p1.end()
  device.queue.submit([pass1.finish()])

  const [qlatB, scoreB, latB] = await readBack(device,
    [[qLat, heads * L * 2], [scores, heads * T * 4], [outLat, heads * L * 4]])
  const errQlat = relErr(Array.from(new Uint16Array(qlatB), f16BitsToF32), rd('ref_qlat', Float32Array))
  const errOlat = relErr(Array.from(new Float32Array(latB), (x) => x), rd('ref_olat', Float32Array))
  // scores comes back holding UNNORMALISED exp(s - max): mla_combine
  // exponentiates in place and applies 1/sum only to its output, so the buffer
  // is not a probability distribution. Normalise both sides before comparing —
  // the earlier version of this check compared against a softmax and failed at
  // 8.2e-1 while the kernel was correct, which the passing latent context
  // (o_lat = p @ c) had already proved.
  const refS = rd('ref_scores', Float32Array)
  const gotE = Array.from(new Float32Array(scoreB), (x) => x)
  const norm = (row) => {
    const m = Math.max(...row)
    const e = row.map((s) => Math.exp(s - m))
    const z = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / z)
  }
  const refP = [], gotP = []
  for (let h = 0; h < heads; h++) {
    refP.push(...norm(Array.from({ length: T }, (_, t) => refS[h * T + t])))
    const e = gotE.slice(h * T, (h + 1) * T)
    const z = e.reduce((a, b) => a + b, 0)
    gotP.push(...e.map((v) => v / z))
  }
  const errProb = relErr(gotP, refP)

  // ── back out through kv_b's V half, then o_proj ──
  device.queue.writeBuffer(oLat16, 0, f16Array(new Float32Array(latB)))
  const pass2 = device.createCommandEncoder()
  const p2 = pass2.beginComputePass()
  run(p2, mlaProj, [oHeads, oLat16, f16buf('wv'), dims(V, L)], wgs(V), heads)
  p2.end()
  device.queue.submit([pass2.finish()])
  const [headB] = await readBack(device, [[oHeads, heads * V * 2]])
  const errOheads = relErr(Array.from(new Uint16Array(headB), f16BitsToF32), rd('ref_oheads', Float32Array))

  const attn = await matvec('self_attn.o_proj', null, D, heads * V, oHeads)
  const errAttn = relErr(attn, rd('ref_attn_out', Float32Array))

  const prep = await dsv2Prep(device, dir, meta)
  const parts = [['q_proj', errQ], ['kv_a_proj', errKv], ['q->latent', errQlat],
                 ['softmax', errProb], ['latent ctx', errOlat], ['->head', errOheads],
                 ['o_proj', errAttn]]
  // See the note in mlaAttention: 2e-2 was ~50x the observed error and would
  // not have noticed an f32 accumulator becoming f16.
  return {
    pass: parts.every(([, e]) => e < 1e-3) && prep.pass,
    detail: parts.map(([n, e]) => `${n} ${e.toExponential(2)}`).join(' | ')
      + ` (pe rows permuted at load, yarn scale ${scale.toFixed(4)})\n      prep: ${prep.detail}`,
  }
}

/** The three PREP kernels, on the GPU, one dispatch at a time.
 *
 *  dsv2Layer proves nothing about these: it feeds ref_qnope, ref_qpe, ref_c and
 *  ref_kpe straight from numpy, so the RoPE convention, the kv_a RMSNorm and
 *  every cache address are simply assumed there. kv_a_layernorm.weight is in
 *  the bundle and, until this ran, was read by no GPU test in this repo.
 *
 *  Position 19 is not arbitrary: position 0 is rotation-identity (cos 1,
 *  sin 0), so a stale or wrong RoPE table passes there for free. */
async function dsv2Prep(device, dir, meta) {
  const rd = (n, T) => read(dir, n, T)
  const { heads, nope: N, rope: R, kv_lora: L, tokens: T, query_at: qi } = meta
  const shader = (f, e) => pipelineFor(device, withPrelude(readFileSync(join(SHADERS, f), 'utf8'),
    { ...DEEPSEEK_V2_LITE }), e)
  const stg = (n) => device.createBuffer({ size: n, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const f16in = (n) => buffer(device, f16Array(rd(n, Float32Array)), BU.STORAGE | BU.COPY_DST)
  const u32u = (a) => buffer(device, new Uint32Array(a), BU.UNIFORM | BU.COPY_DST)
  // The engine hands position through a STORAGE i32 array, exactly as
  // kv_append does, so the bind groups stay hoisted across tokens.
  const posMap = buffer(device, new Int32Array([qi]), BU.STORAGE | BU.COPY_DST)
  // kv_a_layernorm.weight ships as F16 in the checkpoint, so it is bound
  // verbatim rather than via the f32 ref dumps. Until this test it was read by
  // no GPU test in the repo.
  const rawF16 = (name) => {
    const rec = meta.tensors[`${meta.layer}.${name}`]
    const b = readFileSync(join(dir, rec.file))
    return buffer(device, new Uint8Array(b.buffer, b.byteOffset, b.byteLength), BU.STORAGE | BU.COPY_DST)
  }
  // The table the ENGINE would build — not one recomputed by the test — so a
  // wrong yarn table is a failure here rather than a shared assumption.
  const invFreq = buffer(device, ropeInvFreqTable(DEEPSEEK_V2_LITE), BU.STORAGE | BU.COPY_DST)

  const run = (pipe, bufs, x, y = 1) => {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipe)
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    }))
    pass.dispatchWorkgroups(x, y, 1)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // ── mla_q_split: q_proj's [heads, N+R] output into q_nope and rotated q_pe
  const qNope = stg(heads * N * 2)
  const qPe = stg(heads * R * 2)
  run(shader('mla_q_split.wgsl', 'mla_q_split'),
    [f16in('ref_qperm'), invFreq, posMap, qNope, qPe, u32u([N, R, 0, 0])], heads)

  // ── mla_kv_write: kv_a's [L+R] output into the cache at position qi
  const cacheC = stg(T * L * 2)
  const cacheKpe = stg(T * R * 2)
  run(shader('mla_kv_write.wgsl', 'mla_kv_write'),
    [f16in('ref_kvaperm'), rawF16('self_attn.kv_a_layernorm.weight'), invFreq, posMap, cacheC, cacheKpe,
     u32u([L, R, 0, 0])], 1)

  // ── mla_narrow: the f32 latent context down to f16 for mla_proj
  const narrowed = stg(heads * L * 2)
  run(shader('mla_narrow.wgsl', 'mla_narrow'),
    [narrowed, buffer(device, rd('ref_olat', Float32Array), BU.STORAGE | BU.COPY_DST),
     u32u([heads * L, 0, 0, 0])], Math.ceil((heads * L) / 256))

  const [a, b, c, d, e] = await readBack(device, [
    [qNope, heads * N * 2], [qPe, heads * R * 2],
    [cacheC, T * L * 2], [cacheKpe, T * R * 2], [narrowed, heads * L * 2],
  ])
  const f16 = (bytes) => Array.from(new Uint16Array(bytes), f16BitsToF32)
  // Only the query row of the cache was written; compare that row.
  const rowC = f16(c).slice(qi * L, (qi + 1) * L)
  const rowK = f16(d).slice(qi * R, (qi + 1) * R)
  const parts = [
    ['q_nope', relErr(f16(a), rd('ref_qnope', Float32Array))],
    ['q_pe(rope)', relErr(f16(b), rd('ref_qpe', Float32Array))],
    ['latent+rmsnorm', relErr(rowC, rd('ref_c', Float32Array).slice(qi * L, (qi + 1) * L))],
    ['k_pe(rope)', relErr(rowK, rd('ref_kpe', Float32Array).slice(qi * R, (qi + 1) * R))],
    ['narrow f32->f16', relErr(f16(e), rd('ref_olat', Float32Array))],
  ]
  return {
    pass: parts.every(([, v]) => v < 1e-3),
    detail: parts.map(([n, v]) => `${n} ${v.toExponential(2)}`).join(' | ')
      + ` (position ${qi}, rope table from the spec)`,
  }
}

const KERNELS = {
  affine_matmul: affineMatmul, affine_matmul_q3: affineMatmul,
  affine_embedding: affineEmbedding, moe_block: moeBlock,
  qwen36_gdn: qwen36Gdn, qwen36_attn: qwen36Attn, qwen36_layer: qwen36Layer,
  mla_attention: mlaAttention, dsv2_layer: dsv2Layer,
}

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
