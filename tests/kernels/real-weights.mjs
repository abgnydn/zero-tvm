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
import { getDevice, pipelineFor, buffer, runCompute, runComputeReads, BU, MM } from './gpu.mjs'
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


// ── MoE block: routed experts ────────────────────────────────────────────────
//
// Expert e's slice of every stacked switch_mlp tensor is contiguous AND its
// byte stride (524288 for weights, 32768 for scales/bias) is a multiple of 256,
// so an expert is selected purely by the bind-group offset — the matmul kernel
// needs no notion of experts at all. (Per-expert dispatch costs ~7.9 µs each,
// which is fine for a correctness test; folding the expert into a grid
// dimension is the optimisation, not the semantics.)
const COMBINE_WGSL = `
enable f16;
@group(0) @binding(0) var<storage, read_write> acc : array<f32>;
@group(0) @binding(1) var<storage, read> src : array<f16>;
@group(0) @binding(2) var<uniform> args : Args;
@group(0) @binding(3) var<storage, read> gate : array<f32>;
struct Args { n : u32, score : f32, useGate : u32 }
@compute @workgroup_size(256)
fn moe_combine(@builtin(global_invocation_id) g : vec3<u32>) {
  if (g.x >= args.n) { return; }
  // Routed slots carry a router score; the shared expert instead carries
  // sigmoid(shared_expert_gate·x), so it is the same accumulate with a
  // different weight — not a separate code path.
  let wgt : f32 = select(args.score, 1.0 / (1.0 + exp(-gate[0])), args.useGate == 1u);
  acc[g.x] = acc[g.x] + wgt * f32(src[g.x]);
}`

async function moeBlock(device, dir, meta) {
  const D = meta.hidden, F = meta.moe_intermediate, K = meta.top_k
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

  // ---- router (8-bit affine, its own kernel) -> top-K on the GPU ----
  const rt = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'moe_router.wgsl'), 'utf8')), 'moe_router')
  const rIdx = device.createBuffer({ size: K * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const rScore = device.createBuffer({ size: K * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const [idxBytes, scBytes] = await runComputeReads(device, rt, [
    rIdx, rScore, xBuf,
    st('router_w_u32', Uint32Array), st('router_s_f16', Uint16Array), st('router_b_f16', Uint16Array),
    buffer(device, new Uint32Array([D, meta.num_experts, K, meta.norm_topk_prob ? 1 : 0]),
           BU.UNIFORM | BU.COPY_DST),
  ], [1], [{ index: 0, bytes: K * 4 }, { index: 1, bytes: K * 4 }])
  const gpuIdx = new Uint32Array(idxBytes)
  const gpuScore = new Float32Array(scBytes)

  // Order-invariant comparison: the kernel emits descending, mlx ascending, and
  // the block output does not depend on the order either way.
  const byIdx = (a, b) => a[0] - b[0]
  const refPairs = Array.from(idx, (v, i) => [v, score[i]]).sort(byIdx)
  const gotPairs = Array.from(gpuIdx, (v, i) => [v, gpuScore[i]]).sort(byIdx)
  const sameSet = refPairs.every((r, i) => r[0] === gotPairs[i][0])
  const scoreErr = sameSet
    ? Math.max(...refPairs.map((r, i) => Math.abs(r[1] - gotPairs[i][1]) / r[1]))
    : Infinity
  if (!sameSet) {
    return { pass: false, detail: `router picked ${[...gpuIdx].sort((a, b) => a - b)} but reference picked ${[...idx].sort((a, b) => a - b)}` }
  }
  const gu = device.createBuffer({ size: 2 * F * 2, usage: BU.STORAGE | BU.COPY_SRC })   // [gate|up] f16
  const h = device.createBuffer({ size: F * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const dOut = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const acc = device.createBuffer({ size: D * 4, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  device.queue.writeBuffer(acc, 0, new Float32Array(D))

  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL({ affine: true })), int4MatmulEntry({ affine: true }))
  const silu = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'silu_mul.wgsl'), 'utf8'),
                                               { ...QWEN35_4B, ffn: F }), 'silu_mul')
  const comb = pipelineFor(device, COMBINE_WGSL, 'moe_combine')

  const EXP_W = 524288, EXP_S = 32768                    // per-expert byte strides
  const pod = (kp, spr, n) => buffer(device, new Uint32Array([kp, spr, n, 0]), BU.UNIFORM | BU.COPY_DST)
  const podGU = pod(F * 2048 / 8 / F, 0, 0)              // placeholder, replaced below
  const podGate = buffer(device, new Uint32Array([D / 8, D / 64, F, 0]), BU.UNIFORM | BU.COPY_DST)
  const podDown = buffer(device, new Uint32Array([F / 8, F / 64, D, 0]), BU.UNIFORM | BU.COPY_DST)
  const podSilu = buffer(device, new Int32Array([1, (F + 255) / 256 | 0]), BU.UNIFORM | BU.COPY_DST)

  const bgMM = (out, outOff, outSize, inp, proj, e, podBuf) =>
    device.createBindGroup({
      layout: mm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: out, offset: outOff, size: outSize } },
        { binding: 1, resource: { buffer: inp } },
        { binding: 2, resource: { buffer: W[`${proj}_s`], offset: e * EXP_S, size: EXP_S } },
        { binding: 3, resource: { buffer: W[`${proj}_w`], offset: e * EXP_W, size: EXP_W } },
        { binding: 4, resource: { buffer: podBuf } },
        { binding: 5, resource: { buffer: W[`${proj}_b`], offset: e * EXP_S, size: EXP_S } },
      ],
    })

  const gateLogitPlaceholder = device.createBuffer({ size: 4, usage: BU.STORAGE })
  const enc = device.createCommandEncoder()
  for (let t = 0; t < K; t++) {
    const e = gpuIdx[t]
    const pass = enc.beginComputePass()
    pass.setPipeline(mm)
    pass.setBindGroup(0, bgMM(gu, 0, F * 2, xBuf, 'gate_proj', e, podGate))
    pass.dispatchWorkgroups(F)
    pass.setBindGroup(0, bgMM(gu, F * 2, F * 2, xBuf, 'up_proj', e, podGate))
    pass.dispatchWorkgroups(F)
    pass.setPipeline(silu)
    pass.setBindGroup(0, device.createBindGroup({
      layout: silu.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: h } }, { binding: 1, resource: { buffer: gu } },
                { binding: 2, resource: { buffer: podSilu } }],
    }))
    pass.dispatchWorkgroups(Math.ceil(F / 256))
    pass.setPipeline(mm)
    pass.setBindGroup(0, bgMM(dOut, 0, D * 2, h, 'down_proj', e, podDown))
    pass.dispatchWorkgroups(D)
    pass.setPipeline(comb)
    pass.setBindGroup(0, device.createBindGroup({
      layout: comb.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: acc } }, { binding: 1, resource: { buffer: dOut } },
                { binding: 2, resource: { buffer: buffer(device, new Uint32Array(
                    [D, new Uint32Array(new Float32Array([gpuScore[t]]).buffer)[0], 0]), BU.UNIFORM | BU.COPY_DST) } },
                { binding: 3, resource: { buffer: gateLogitPlaceholder } }],
    }))
    pass.dispatchWorkgroups(Math.ceil(D / 256))
    pass.end()
  }
  // ---- shared expert: same chain, no expert offset, gated by sigmoid ----
  const SW = {}
  for (const proj of ['gate_proj', 'up_proj', 'down_proj']) {
    SW[`${proj}_w`] = st(`shd_${proj}_w_u32`, Uint32Array)
    SW[`${proj}_s`] = st(`shd_${proj}_s_f16`, Uint16Array)
    SW[`${proj}_b`] = st(`shd_${proj}_b2_f16`, Uint16Array)
  }
  const mv = pipelineFor(device, withPrelude(readFileSync(join(SHADERS, 'int8_affine_matvec.wgsl'), 'utf8')),
                         'int8_affine_matvec')
  const gateLogit = device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC })
  const bgShared = (out, outOff, outSize, inp, proj, podBuf) =>
    device.createBindGroup({
      layout: mm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: out, offset: outOff, size: outSize } },
        { binding: 1, resource: { buffer: inp } },
        { binding: 2, resource: { buffer: SW[`${proj}_s`] } },
        { binding: 3, resource: { buffer: SW[`${proj}_w`] } },
        { binding: 4, resource: { buffer: podBuf } },
        { binding: 5, resource: { buffer: SW[`${proj}_b`] } },
      ],
    })
  {
    const pass = enc.beginComputePass()
    pass.setPipeline(mv)
    pass.setBindGroup(0, device.createBindGroup({
      layout: mv.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gateLogit } }, { binding: 1, resource: { buffer: xBuf } },
        { binding: 2, resource: { buffer: st('shdgate_w_u32', Uint32Array) } },
        { binding: 3, resource: { buffer: st('shdgate_s_f16', Uint16Array) } },
        { binding: 4, resource: { buffer: st('shdgate_b_f16', Uint16Array) } },
        { binding: 5, resource: { buffer: buffer(device, new Uint32Array([D, 1]), BU.UNIFORM | BU.COPY_DST) } },
      ],
    }))
    pass.dispatchWorkgroups(1)
    pass.setPipeline(mm)
    pass.setBindGroup(0, bgShared(gu, 0, F * 2, xBuf, 'gate_proj', podGate))
    pass.dispatchWorkgroups(F)
    pass.setBindGroup(0, bgShared(gu, F * 2, F * 2, xBuf, 'up_proj', podGate))
    pass.dispatchWorkgroups(F)
    pass.setPipeline(silu)
    pass.setBindGroup(0, device.createBindGroup({
      layout: silu.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: h } }, { binding: 1, resource: { buffer: gu } },
                { binding: 2, resource: { buffer: podSilu } }],
    }))
    pass.dispatchWorkgroups(Math.ceil(F / 256))
    pass.setPipeline(mm)
    pass.setBindGroup(0, bgShared(dOut, 0, D * 2, h, 'down_proj', podDown))
    pass.dispatchWorkgroups(D)
    pass.setPipeline(comb)
    pass.setBindGroup(0, device.createBindGroup({
      layout: comb.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: acc } }, { binding: 1, resource: { buffer: dOut } },
                { binding: 2, resource: { buffer: buffer(device, new Uint32Array([D, 0, 1]), BU.UNIFORM | BU.COPY_DST) } },
                { binding: 3, resource: { buffer: gateLogit } }],
    }))
    pass.dispatchWorkgroups(Math.ceil(D / 256))
    pass.end()
  }

  const outBuf = device.createBuffer({ size: D * 4, usage: BU.COPY_DST | BU.MAP_READ })
  enc.copyBufferToBuffer(acc, 0, outBuf, 0, D * 4)
  device.queue.submit([enc.finish()])
  await outBuf.mapAsync(MM.READ)
  const got = new Float32Array(outBuf.getMappedRange().slice(0))

  // Tolerance is set by the REFERENCE's precision, not ours. MLX runs the whole
  // block in bfloat16 (~4e-3 relative) and gate→silu→down→weighted-sum compounds
  // it. Evidence it is not covering for a bug: re-running this exact algorithm on
  // CPU in f32 — no GPU, no f16 intermediates — lands at max 1.24e-2 / median
  // 2.27e-3 on the routed part alone, i.e. the GPU path adds ~nothing (1.26e-2),
  // and the error is spread across all 2048 outputs rather than concentrated in
  // a few, which is what an indexing or routing mistake looks like. Measured
  // 4.32e-3 for the full block (the shared expert dominates the magnitude, so
  // routed noise shrinks relative to it); 1.2e-2 leaves ~3x headroom.
  const scale = Math.max(...Array.from(yRef, Math.abs))
  let maxRel = 0
  for (let i = 0; i < D; i++) maxRel = Math.max(maxRel, Math.abs(got[i] - yRef[i]) / scale)
  return {
    pass: maxRel < 0.012 && scoreErr < 0.01,
    detail: `router top-${K} exact, scores ${scoreErr.toExponential(2)} | FULL block ${maxRel.toExponential(2)}`,
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
