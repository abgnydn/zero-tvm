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
import { QWEN35_4B, QWEN36_35B_A3B } from '../../src/compiler/model-spec.ts'

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
  const bias = read(dir, 'bias_f16', Uint16Array)
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
    buffer(device, bias, BU.STORAGE | BU.COPY_DST), // @binding(5)
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

  // The reference runs in f32, so this tolerance is set by OUR f16 activations
  // and int4 weights rather than by the reference's own rounding. It was 0.012
  // when the reference was bfloat16 — that is not a tolerance, it is the
  // reference's noise floor, and it is wide enough to hide a real bug.
  const scale = Math.max(...Array.from(yRef, Math.abs))
  let maxRel = 0
  for (let i2 = 0; i2 < D; i2++) maxRel = Math.max(maxRel, Math.abs(got[i2] - yRef[i2]) / scale)
  return {
    pass: maxRel < 0.005 && scoreErr < 0.005,
    detail: `router top-${K} exact, scores ${scoreErr.toExponential(2)} | full block ${maxRel.toExponential(2)} | ${SLOTS} slots in 6 dispatches`,
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
    [rope, [qB, kB, vB, qkv, pos, u32([1, 0, 1, S.qkvDim / 256])], [S.qkvDim / 256]],
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

const KERNELS = {
  affine_matmul: affineMatmul, moe_block: moeBlock,
  qwen36_gdn: qwen36Gdn, qwen36_attn: qwen36Attn,
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
