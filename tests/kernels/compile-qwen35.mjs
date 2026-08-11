// QWEN35_4B kernel-correctness suite — the GatedDeltaNet (GDN) kernel family
// plus the gated-attention pieces, all under shaderPrelude(QWEN35_4B):
// D=2560, HEADS=16, KV_HEADS=4, HEAD_DIM=256, ROTARY_DIM=64, ATTN_GATE=1,
// C_ATTN_DIM=10240, GDN 16 k-heads / 32 v-heads × 128, conv width 4,
// ROPE_THETA=1e7, RMS_EPS=1e-6 — against tests/kernels/ref-gdn.mjs (a
// formula-by-formula transcription of HF modeling_qwen3_5.py and mlc-llm
// qwen35_model.py). Deterministic (seeded RNG); exits non-zero on mismatch.
//
// Highlights:
//   - gdn_conv is exercised across 6 sequential decode steps and its ring
//     buffer must match the expected raw history BIT-EXACTLY.
//   - gdn_recur runs 8 steps as ONE seq_len=8 dispatch and as 8 seq_len=1
//     dispatches; both must match the CPU reference, and the two GPU paths
//     must agree BIT-EXACTLY (state round-trips through the f32 buffer).
//   - NEGATIVE CONTROL: a reference computed with the WRONG K→V head pairing
//     (h % GDN_K_HEADS, the GGUF tile order) must FAIL against the GPU
//     output — proving the tests pin the repeat-interleave pairing
//     (h / GDN_GVA_GROUP) that the MLC weight cache requires.
//   - full-chain tests: the complete GDN block (ONE fused int4 input
//     projection qkv‖z‖a‖b → conv → gates → recurrence → gated norm →
//     out_proj, engine-packed layout) over 4 tokens with state
//     carry, and the complete gated-attention block (c_attn → split →
//     qk_norm → partial rope → kv_append → attention → sigmoid gate →
//     o_proj) over 3 decode steps.
//   - compile_all: every .wgsl + int4 generator variant builds under
//     QWEN35_4B.
//
//   npm run test:kernels:qwen35

import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import {
  getDevice,
  probeSubgroupSize,
  buffer,
  runCompute,
  runComputeReads,
  pipelineFor,
  BU,
  MM,
} from './gpu.mjs'
import { toF16, f16Array, f16BitsToF32, f32ToF16Bits } from './half.mjs'
import { withPrelude, QWEN35_4B, ropeInvFreqTable } from '../../src/compiler/shader-prelude.ts'
import {
  int4MatmulWGSL,
  int4MatmulEntry,
  INT4_MATMUL_VARIANTS,
  int4MatmulTiledMWGSL,
} from '../../src/compiler/shaders/int4_matmul.gen.ts'
import {
  int4Matvec,
  refConvStep,
  expectedConvRing,
  refGates,
  refRecur,
  refGatedNorm,
  refGdnBlock,
  refGatedSplit,
  refPartialRope,
  refGatedAttnBlock,
  sigmoid,
} from './ref-gdn.mjs'

const Q = QWEN35_4B
const G = {
  kHeads: Q.gdnKHeads, vHeads: Q.gdnVHeads,
  headK: Q.gdnHeadK, headV: Q.gdnHeadV,
}
const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')
const wgsl = (name) => withPrelude(readFileSync(resolve(SHADERS, name), 'utf8'), Q)

// Deterministic PRNG (mulberry32) — same as run.mjs / compile-qwen.mjs.
function rng(seed) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const arr = (n, f) => Array.from({ length: n }, f)

function podBuffer(device, fields) {
  const size = Math.max(16, Math.ceil((fields.length * 4) / 16) * 16)
  const buf = new ArrayBuffer(size)
  const dv = new DataView(buf)
  fields.forEach((f, i) => {
    if (f.f32 !== undefined) dv.setFloat32(i * 4, f.f32, true)
    else if (f.u32 !== undefined) dv.setUint32(i * 4, f.u32, true)
    else dv.setInt32(i * 4, f.i32, true)
  })
  return buffer(device, new Uint8Array(buf), BU.UNIFORM | BU.COPY_DST)
}

// Random int4 weight records (q4f16_1) for an N×K projection.
function randInt4(r, N, K, scaleLo = 0.01, scaleHi = 0.06) {
  const KP = K / 8
  const SPR = K / 32
  const weights = new Uint32Array(N * KP)
  for (let i = 0; i < weights.length; i++) weights[i] = (r() * 0xffffffff) >>> 0
  const scales = new Float32Array(N * SPR)
  for (let i = 0; i < scales.length; i++) scales[i] = toF16(r() * (scaleHi - scaleLo) + scaleLo)
  return { weights, scales, KP, SPR }
}

const maxAbsDiff = (got, ref) => {
  let m = 0
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]))
  return m
}
const maxRelDiff = (got, ref, eps) => {
  let m = 0
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]) / (Math.abs(ref[i]) + eps))
  return m
}

// ── gdn_conv — depthwise causal conv + SiLU, ring buffer over 6 decode steps ─
async function testGdnConv(device) {
  const r = rng(21)
  const C = Q.gdnQkvDim, K = Q.gdnConvK, RING = K - 1, STEPS = 6
  const convW = arr(C * K, () => toF16(r() - 0.5))
  const xs = arr(STEPS, () => arr(C, () => toF16(r() * 2 - 1)))

  const pipe = pipelineFor(device, wgsl('gdn_conv.wgsl'), 'gdn_conv')
  const out = device.createBuffer({ size: C * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const xBuf = device.createBuffer({ size: C * 2, usage: BU.STORAGE | BU.COPY_DST })
  const state = device.createBuffer({ size: RING * C * 2, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC })
  device.queue.writeBuffer(state, 0, new Uint16Array(RING * C))   // zeroed before token 0
  const wBuf = buffer(device, f16Array(convW), BU.STORAGE | BU.COPY_DST)
  const GRID = C / 256

  let maxAbs = 0
  for (let t = 0; t < STEPS; t++) {
    device.queue.writeBuffer(xBuf, 0, f16Array(xs[t]))
    const pod = podBuffer(device, [{ i32: t }, { u32: GRID }])
    const bytes = await runCompute(device, pipe, [out, xBuf, state, wBuf, pod], [GRID], 0, C * 2)
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const ref = refConvStep(xs, t, convW, C, K)
    maxAbs = Math.max(maxAbs, maxAbsDiff(got, ref))
  }
  // Ring-buffer state after the last step must hold the raw history bit-exact.
  const [stateBytes] = await runComputeReads(
    device, pipe,
    // No extra dispatch — read the state via a harmless 0-workgroup-guard run:
    // easier: just copy it out with a runComputeReads on a no-op? Instead read
    // directly through a dedicated read of the buffer via a trivial encoder.
    [out, xBuf, state, wBuf, podBuffer(device, [{ i32: STEPS - 1 }, { u32: 0 }])],
    [1],
    [{ index: 2, bytes: RING * C * 2 }],
  )
  const gotRing = new Uint16Array(stateBytes)
  const wantRing = Uint16Array.from(expectedConvRing(xs, STEPS - 1, C, K), f32ToF16Bits)
  let ringBad = 0
  for (let i = 0; i < wantRing.length; i++) if (gotRing[i] !== wantRing[i]) ringBad++
  const pass = maxAbs < 5e-3 && ringBad === 0
  return {
    name: `gdn_conv k=${K} ${STEPS} steps`,
    pass,
    detail: `max abs err ${maxAbs.toExponential(2)}, ${ringBad} ring-state mismatches`,
  }
}

// ── gdn_gates — decay exp(g) + beta, incl. the softplus x>20 branch ──────────
async function testGdnGates(device) {
  const r = rng(22)
  const H = Q.gdnVHeads
  const aRaw = arr(H, () => toF16(r() * 2 - 1))
  aRaw[3] = toF16(25)   // softplus threshold branch: x = 25 + dt_bias > 20 → x
  const bRaw = arr(H, () => toF16(r() * 4 - 2))
  const aLog = arr(H, () => Math.fround(Math.log(r() * 3.5 + 0.5)))   // A ∈ (0.5, 4)
  const dtBias = arr(H, () => Math.fround(r() - 1))

  const pipe = pipelineFor(device, wgsl('gdn_gates.wgsl'), 'gdn_gates')
  const out = device.createBuffer({ size: 2 * H * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    // Packed [a | b] pair — the engine binds the fused-projection buffer at
    // the a-region offset, so the kernel sees a at [h], b at [H+h].
    buffer(device, f16Array([...aRaw, ...bRaw]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Float32Array(aLog), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Float32Array(dtBias), BU.STORAGE | BU.COPY_DST),
    // PODArgs {seq_len, ab_stride, packGridDimX} — single token, tight stride.
    podBuffer(device, [{ i32: 1 }, { i32: 2 * H }, { u32: 1 }]),
  ]
  const bytes = await runCompute(device, pipe, buffers, [1], 0, 2 * H * 4)
  const got = new Float32Array(bytes)
  const { gExp, beta } = refGates(aRaw, bRaw, aLog, dtBias)
  const maxAbs = Math.max(maxAbsDiff(got.slice(0, H), gExp), maxAbsDiff(got.slice(H), beta))
  return { name: 'gdn_gates softplus/sigmoid', pass: maxAbs < 5e-5, detail: `max abs err ${maxAbs.toExponential(2)}` }
}

// ── shared gdn_recur fixture: 8 tokens, random state, distinct per-head data ─
function buildRecurFixture() {
  const r = rng(23)
  const STEPS = 8
  const qs = arr(STEPS, () => arr(Q.gdnKDim, () => toF16(r() * 2 - 1)))
  const ks = arr(STEPS, () => arr(Q.gdnKDim, () => toF16(r() * 2 - 1)))
  const vs = arr(STEPS, () => arr(Q.gdnVDim, () => toF16(r() * 2 - 1)))
  const gExps = arr(STEPS, () => arr(Q.gdnVHeads, () => Math.fround(r() * 0.8 + 0.2)))
  const betas = arr(STEPS, () => arr(Q.gdnVHeads, () => Math.fround(r() * 0.9 + 0.05)))
  const state0 = new Float32Array(Q.gdnVHeads * Q.gdnStatePerHead)
  for (let i = 0; i < state0.length; i++) state0[i] = Math.fround(r() - 0.5)
  // Packed GPU-side layouts.
  const convOut = new Uint16Array(STEPS * Q.gdnQkvDim)
  const gates = new Float32Array(STEPS * 2 * Q.gdnVHeads)
  for (let t = 0; t < STEPS; t++) {
    for (let i = 0; i < Q.gdnKDim; i++) {
      convOut[t * Q.gdnQkvDim + i] = f32ToF16Bits(qs[t][i])
      convOut[t * Q.gdnQkvDim + Q.gdnKDim + i] = f32ToF16Bits(ks[t][i])
    }
    for (let i = 0; i < Q.gdnVDim; i++) convOut[t * Q.gdnQkvDim + 2 * Q.gdnKDim + i] = f32ToF16Bits(vs[t][i])
    for (let h = 0; h < Q.gdnVHeads; h++) {
      gates[t * 2 * Q.gdnVHeads + h] = gExps[t][h]
      gates[t * 2 * Q.gdnVHeads + Q.gdnVHeads + h] = betas[t][h]
    }
  }
  const refState = Float64Array.from(state0)
  const refOuts = refRecur(qs, ks, vs, gExps, betas, refState, G)
  return { STEPS, qs, ks, vs, gExps, betas, state0, convOut, gates, refOuts, refState }
}

// ── gdn_recur — 8 steps in one seq_len=8 dispatch vs CPU reference ───────────
async function testGdnRecurSeq(device, fx) {
  const { STEPS, convOut, gates, state0, refOuts, refState } = fx
  const pipe = pipelineFor(device, wgsl('gdn_recur.wgsl'), 'gdn_recur')
  const out = device.createBuffer({ size: STEPS * Q.gdnVDim * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const state = buffer(device, state0, BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
  const buffers = [
    out,
    buffer(device, convOut, BU.STORAGE | BU.COPY_DST),
    buffer(device, gates, BU.STORAGE | BU.COPY_DST),
    state,
    podBuffer(device, [{ i32: STEPS }, { u32: Q.gdnVHeads }]),
  ]
  const [outBytes, stateBytes] = await runComputeReads(device, pipe, buffers, [Q.gdnVHeads], [
    { index: 0, bytes: STEPS * Q.gdnVDim * 4 },
    { index: 3, bytes: state0.length * 4 },
  ])
  const got = new Float32Array(outBytes)
  const gotState = new Float32Array(stateBytes)
  fx.gpuSeqOut = got   // reused by the stepwise + negative-control tests
  const flatRef = refOuts.flat()
  const outErr = maxRelDiff(got, flatRef, 1e-2)
  const stateErr = maxRelDiff(gotState, refState, 1e-2)
  const pass = outErr < 1e-3 && stateErr < 1e-3
  return {
    name: 'gdn_recur seq=8 + state',
    pass,
    detail: `out rel err ${outErr.toExponential(2)}, state rel err ${stateErr.toExponential(2)}`,
  }
}

// ── gdn_recur — same 8 steps as 8 seq_len=1 dispatches: state carryover must
// be BIT-EXACT vs the single seq=8 dispatch ─────────────────────────────────
async function testGdnRecurStepwise(device, fx) {
  const { STEPS, convOut, gates, state0, gpuSeqOut } = fx
  const pipe = pipelineFor(device, wgsl('gdn_recur.wgsl'), 'gdn_recur')
  const out = device.createBuffer({ size: Q.gdnVDim * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const state = buffer(device, state0, BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
  const convBuf = device.createBuffer({ size: Q.gdnQkvDim * 2, usage: BU.STORAGE | BU.COPY_DST })
  const gatesBuf = device.createBuffer({ size: 2 * Q.gdnVHeads * 4, usage: BU.STORAGE | BU.COPY_DST })
  const pod = podBuffer(device, [{ i32: 1 }, { u32: Q.gdnVHeads }])
  const gotAll = new Float32Array(STEPS * Q.gdnVDim)
  for (let t = 0; t < STEPS; t++) {
    device.queue.writeBuffer(convBuf, 0, convOut.subarray(t * Q.gdnQkvDim, (t + 1) * Q.gdnQkvDim))
    device.queue.writeBuffer(gatesBuf, 0, gates.subarray(t * 2 * Q.gdnVHeads, (t + 1) * 2 * Q.gdnVHeads))
    const bytes = await runCompute(device, pipe, [out, convBuf, gatesBuf, state, pod], [Q.gdnVHeads], 0, Q.gdnVDim * 4)
    gotAll.set(new Float32Array(bytes), t * Q.gdnVDim)
  }
  let bits = 0
  for (let i = 0; i < gotAll.length; i++) if (gotAll[i] !== gpuSeqOut[i]) bits++
  return {
    name: 'gdn_recur 8×seq=1 carryover',
    pass: bits === 0,
    detail: `${bits} elements differ from the seq=8 dispatch (must be bit-exact)`,
  }
}

// ── NEGATIVE CONTROL — the wrong K→V pairing (h % GDN_K_HEADS) must FAIL ─────
function testGdnRecurPairingControl(fx) {
  const { qs, ks, vs, gExps, betas, state0, gpuSeqOut } = fx
  const wrongState = Float64Array.from(state0)
  const wrongOuts = refRecur(qs, ks, vs, gExps, betas, wrongState, G, (h) => h % Q.gdnKHeads)
  const err = maxRelDiff(gpuSeqOut, wrongOuts.flat(), 1e-2)
  return Promise.resolve({
    name: 'gdn_recur wrong-pairing control',
    pass: err > 1e-2,
    detail: `wrong-pairing ref diverges by ${err.toExponential(2)} (must be > 1e-2)`,
  })
}

// ── gdn_norm_out — per-head gated RMSNorm + silu(z) gate ─────────────────────
async function testGdnNormOut(device) {
  const r = rng(24)
  const SEQ = 2
  const x = new Float32Array(SEQ * Q.gdnVDim)
  for (let i = 0; i < x.length; i++) x[i] = Math.fround(r() * 6 - 3)
  const gamma = arr(Q.gdnHeadV, () => toF16(r() * 0.5 + 0.75))
  const z = arr(SEQ * Q.gdnVDim, () => toF16(r() * 4 - 2))

  const ref = []
  for (let t = 0; t < SEQ; t++) {
    ref.push(...refGatedNorm(
      Array.from(x.slice(t * Q.gdnVDim, (t + 1) * Q.gdnVDim)),
      gamma, z.slice(t * Q.gdnVDim, (t + 1) * Q.gdnVDim), Q.rmsEps, Q.gdnHeadV,
    ))
  }

  const pipe = pipelineFor(device, wgsl('gdn_norm_out.wgsl'), 'gdn_norm_out')
  const out = device.createBuffer({ size: SEQ * Q.gdnVDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const GRID = SEQ * Q.gdnVHeads
  const buffers = [
    out,
    buffer(device, x, BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(z), BU.STORAGE | BU.COPY_DST),
    // PODArgs {seq_len, z_stride, packGridDimX} — z tightly packed here.
    podBuffer(device, [{ i32: SEQ }, { i32: Q.gdnVDim }, { u32: GRID }]),
  ]
  const bytes = await runCompute(device, pipe, buffers, [GRID], 0, SEQ * Q.gdnVDim * 2)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const maxAbs = maxAbsDiff(got, ref)
  return { name: 'gdn_norm_out gated rmsnorm', pass: maxAbs < 2e-2, detail: `max abs err ${maxAbs.toExponential(2)}` }
}

// ── full GDN block — ONE fused int4 input projection (qkv‖z‖a‖b rows packed,
// mirroring the engine/loader layout) → conv → gates → recur → norm →
// out_proj, 4 decode steps with conv + recurrent state carry. The CPU
// reference (ref-gdn.mjs refGdnBlock) still computes the four projections
// separately — row concatenation is exactly equivalent for a row-major
// q4f16_1 matmul, so the reference stays authoritative for the packed GPU
// path. Downstream kernels read the z / [a|b] regions via 256-aligned
// bind-group offsets, exactly as the engine binds them. ─────────────────────
async function testGdnBlockChain(device) {
  const r = rng(25)
  const d = Q.d, STEPS = 4
  const w = {
    qkv: randInt4(r, Q.gdnQkvDim, d),
    z: randInt4(r, Q.gdnVDim, d),
    a: randInt4(r, Q.gdnVHeads, d, 0.001, 0.004),  // small: keeps softplus(a+dt_bias) in a lively decay range
    b: randInt4(r, Q.gdnVHeads, d),
    out: randInt4(r, d, Q.gdnVDim),
  }
  const convW = arr(Q.gdnQkvDim * Q.gdnConvK, () => toF16(r() - 0.5))
  const gamma = arr(Q.gdnHeadV, () => toF16(r() * 0.5 + 0.75))
  const aLog = arr(Q.gdnVHeads, () => Math.fround(Math.log(r() * 3.5 + 0.5)))
  const dtBias = arr(Q.gdnVHeads, () => Math.fround(r() - 1))
  const xs = arr(STEPS, () => arr(d, () => toF16(r() * 2 - 1)))

  const refState = { history: [], recur: new Float64Array(Q.gdnVHeads * Q.gdnStatePerHead) }
  const refOuts = refGdnBlock(xs, {
    qkvW: w.qkv.weights, qkvS: w.qkv.scales,
    zW: w.z.weights, zS: w.z.scales,
    aW: w.a.weights, aS: w.a.scales,
    bW: w.b.weights, bS: w.b.scales,
    outW: w.out.weights, outS: w.out.scales,
    convW, gamma, aLog, dtBias,
  }, Q, refState)

  // Pipelines: the scalar int4 matmul serves all five projections.
  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL(), Q), 'int4_matmul')
  const conv = pipelineFor(device, wgsl('gdn_conv.wgsl'), 'gdn_conv')
  const gatesP = pipelineFor(device, wgsl('gdn_gates.wgsl'), 'gdn_gates')
  const recur = pipelineFor(device, wgsl('gdn_recur.wgsl'), 'gdn_recur')
  const norm = pipelineFor(device, wgsl('gdn_norm_out.wgsl'), 'gdn_norm_out')

  const mk = (bytes, extra = 0) =>
    device.createBuffer({ size: bytes, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST | extra })
  const hidden = mk(d * 2)
  const projOut = mk(Q.gdnProjRows * 2)   // fused projection output [qkv | z | a | b]
  const convState = mk((Q.gdnConvK - 1) * Q.gdnQkvDim * 2)
  device.queue.writeBuffer(convState, 0, new Uint16Array((Q.gdnConvK - 1) * Q.gdnQkvDim))
  const convOut = mk(Q.gdnQkvDim * 2)
  const gatesBuf = mk(2 * Q.gdnVHeads * 4)
  const recurState = mk(Q.gdnVHeads * Q.gdnStatePerHead * 4)
  device.queue.writeBuffer(recurState, 0, new Float32Array(Q.gdnVHeads * Q.gdnStatePerHead))
  const recurOut = mk(Q.gdnVDim * 4)
  const normOut = mk(Q.gdnVDim * 2)
  const blockOut = mk(d * 2)

  // Pack qkv‖z‖a‖b rows into ONE weight/scale buffer pair — byte-for-byte
  // what the loader's packGdnProj produces (row-major q4f16_1: row concat ==
  // byte concat).
  const KP = d / 8, SPR = d / 32
  const projW = new Uint32Array(Q.gdnProjRows * KP)
  const projS = new Float32Array(Q.gdnProjRows * SPR)
  {
    let row = 0
    for (const rec of [w.qkv, w.z, w.a, w.b]) {
      projW.set(rec.weights, row * KP)
      projS.set(rec.scales, row * SPR)
      row += rec.weights.length / KP
    }
  }
  const projWBuf = buffer(device, projW, BU.STORAGE | BU.COPY_DST)
  const projSBuf = buffer(device, f16Array(Array.from(projS)), BU.STORAGE | BU.COPY_DST)
  const outWBuf = buffer(device, w.out.weights, BU.STORAGE | BU.COPY_DST)
  const outSBuf = buffer(device, f16Array(Array.from(w.out.scales)), BU.STORAGE | BU.COPY_DST)
  const convWBuf = buffer(device, f16Array(convW), BU.STORAGE | BU.COPY_DST)
  const gammaBuf = buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST)
  const aLogBuf = buffer(device, new Float32Array(aLog), BU.STORAGE | BU.COPY_DST)
  const dtBiasBuf = buffer(device, new Float32Array(dtBias), BU.STORAGE | BU.COPY_DST)

  const pods = {
    proj: buffer(device, new Uint32Array([KP, SPR, Q.gdnProjRows, 0]), BU.UNIFORM | BU.COPY_DST),
    out: buffer(device, new Uint32Array([w.out.KP, w.out.SPR, d, 0]), BU.UNIFORM | BU.COPY_DST),
    gates: podBuffer(device, [{ i32: 1 }, { i32: 2 * Q.gdnVHeads }, { u32: 1 }]),
    recur: podBuffer(device, [{ i32: 1 }, { u32: Q.gdnVHeads }]),
    norm: podBuffer(device, [{ i32: 1 }, { i32: Q.gdnVDim }, { u32: Q.gdnVHeads }]),
  }

  // Region views into projOut — same 256-aligned offsets the engine binds.
  const zRegion = { buffer: projOut, offset: Q.gdnQkvDim * 2, size: Q.gdnVDim * 2 }
  const abRegion = { buffer: projOut, offset: (Q.gdnQkvDim + Q.gdnVDim) * 2, size: 2 * Q.gdnVHeads * 2 }

  let maxRel = 0
  for (let t = 0; t < STEPS; t++) {
    device.queue.writeBuffer(hidden, 0, f16Array(xs[t]))
    const convPod = podBuffer(device, [{ i32: t }, { u32: Q.gdnQkvDim / 256 }])
    // One fused projection → conv → gates → recurrence → norm → out_proj
    // (same dispatch chain the engine records). Entries may be region views.
    const dispatch = async (pipe, bufs, grid) => {
      const bg = device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
      })
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(pipe)
      pass.setBindGroup(0, bg)
      pass.dispatchWorkgroups(grid)
      pass.end()
      device.queue.submit([enc.finish()])
    }
    await dispatch(mm, [projOut, hidden, projSBuf, projWBuf, pods.proj], Q.gdnProjRows)
    // gdn_conv binds the whole packed buffer; it reads only the qkv region.
    await dispatch(conv, [convOut, projOut, convState, convWBuf, convPod], Q.gdnQkvDim / 256)
    await dispatch(gatesP, [gatesBuf, abRegion, aLogBuf, dtBiasBuf, pods.gates], 1)
    await dispatch(recur, [recurOut, convOut, gatesBuf, recurState, pods.recur], Q.gdnVHeads)
    await dispatch(norm, [normOut, recurOut, gammaBuf, zRegion, pods.norm], Q.gdnVHeads)
    const bytes = await runCompute(device, mm, [blockOut, normOut, outSBuf, outWBuf, pods.out], [d], 0, d * 2)
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    maxRel = Math.max(maxRel, maxRelDiff(got, refOuts[t], 1e-2))
  }
  return {
    name: 'gdn_block chain 4 steps',
    pass: maxRel < 0.02,
    detail: `max rel err ${maxRel.toExponential(2)} across ${STEPS} decode steps`,
  }
}

// ── rope — PARTIAL RoPE: only ROTARY_DIM=64 of 256 dims rotate, theta 1e7 ────
async function testRopePartial(device) {
  const r = rng(26)
  const QKV = Q.qkvDim, HD = Q.headDim, pos = 37
  const qkv = arr(QKV, () => toF16(r() * 2 - 1))

  // Reference: split then rotate the rotary slice of each Q/K head in place.
  const q = qkv.slice(0, Q.qDim)
  const k = qkv.slice(Q.qDim, Q.qDim + Q.kvDim)
  const v = qkv.slice(Q.qDim + Q.kvDim)
  for (let h = 0; h < Q.heads; h++) refPartialRope(q, h * HD, pos, Q.rotaryDim, Q.ropeTheta)
  for (let h = 0; h < Q.kvHeads; h++) refPartialRope(k, h * HD, pos, Q.rotaryDim, Q.ropeTheta)

  const pipe = pipelineFor(device, wgsl('rope.wgsl'), 'rope_kernel')
  const mk = (n) => device.createBuffer({ size: n * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    mk(Q.qDim), mk(Q.kvDim), mk(Q.kvDim),
    buffer(device, f16Array(qkv), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array([pos]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1, 0, 1, QKV / 256]), BU.UNIFORM | BU.COPY_DST),
    buffer(device, ropeInvFreqTable(Q), BU.STORAGE | BU.COPY_DST), // inv_freq (binding 6)
  ]
  const [qb, kb, vb] = await runComputeReads(device, pipe, buffers, [QKV / 256], [
    { index: 0, bytes: Q.qDim * 2 },
    { index: 1, bytes: Q.kvDim * 2 },
    { index: 2, bytes: Q.kvDim * 2 },
  ])
  const dec = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
  const [gq, gk, gv] = [dec(qb), dec(kb), dec(vb)]
  let bad = Math.max(maxAbsDiff(gq, q), maxAbsDiff(gk, k))
  // The pass-through dims (>= ROTARY_DIM) and V must be bit-exact copies.
  let passBad = 0
  for (let h = 0; h < Q.heads; h++)
    for (let dd = Q.rotaryDim; dd < HD; dd++)
      if (gq[h * HD + dd] !== qkv[h * HD + dd]) passBad++
  for (let i = 0; i < Q.kvDim; i++) if (gv[i] !== v[i]) passBad++
  const pass = bad < 5e-3 && passBad === 0
  return {
    name: 'rope partial rotary=64 θ=1e7',
    pass,
    detail: `max abs err ${bad.toExponential(2)}, ${passBad} pass-through dims touched`,
  }
}

// ── qk_norm — per-head RMSNorm at head_dim 256 (EPT=8), eps 1e-6 ─────────────
async function testQkNorm256(device) {
  const r = rng(27)
  const QKV = Q.qkvDim, HD = Q.headDim
  const qkv = arr(QKV, () => toF16(r() * 2 - 1))
  const qGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))
  const kGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))

  const ref = qkv.slice()
  const normHead = (base, gamma) => {
    let ss = 0
    for (let dd = 0; dd < HD; dd++) ss += qkv[base + dd] * qkv[base + dd]
    const rinv = 1 / Math.sqrt(ss / HD + Q.rmsEps)
    for (let dd = 0; dd < HD; dd++) ref[base + dd] = toF16(qkv[base + dd] * rinv * gamma[dd])
  }
  for (let h = 0; h < Q.heads; h++) normHead(h * HD, qGamma)
  for (let h = 0; h < Q.kvHeads; h++) normHead(Q.qDim + h * HD, kGamma)

  const WGS = Q.heads + Q.kvHeads
  const pipe = pipelineFor(device, wgsl('qk_norm.wgsl'), 'qk_norm')
  const qkvBuf = buffer(device, f16Array(qkv), BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
  const buffers = [
    qkvBuf,
    buffer(device, f16Array(qGamma), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(kGamma), BU.STORAGE | BU.COPY_DST),
    podBuffer(device, [{ i32: 1 }, { u32: WGS }]),
  ]
  const bytes = await runCompute(device, pipe, buffers, [WGS], 0, QKV * 2)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const maxAbs = maxAbsDiff(got, ref)
  let vBad = 0
  for (let i = Q.qDim + Q.kvDim; i < QKV; i++) if (got[i] !== ref[i]) vBad++
  const pass = maxAbs < 5e-3 && vBad === 0
  return { name: 'qk_norm hd=256', pass, detail: `max abs err ${maxAbs.toExponential(2)}, ${vBad} V elems touched` }
}

// ── gated_qkv_split — c_attn interleaved [Q|gate] unpack, bit-exact ──────────
async function testGatedQkvSplit(device) {
  const r = rng(28)
  const cAttn = arr(Q.cAttnDim, () => toF16(r() * 2 - 1))
  const { qkv: refQkv, gate: refGate } = refGatedSplit(cAttn, Q)

  const pipe = pipelineFor(device, wgsl('gated_qkv_split.wgsl'), 'gated_qkv_split')
  const qkvOut = device.createBuffer({ size: Q.qkvDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const gateOut = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const GRID = Q.cAttnDim / 256
  const buffers = [
    qkvOut,
    gateOut,
    buffer(device, f16Array(cAttn), BU.STORAGE | BU.COPY_DST),
    podBuffer(device, [{ i32: 1 }, { u32: GRID }]),
  ]
  const [qkvBytes, gateBytes] = await runComputeReads(device, pipe, buffers, [GRID], [
    { index: 0, bytes: Q.qkvDim * 2 },
    { index: 1, bytes: Q.qDim * 2 },
  ])
  const gotQkv = new Uint16Array(qkvBytes)
  const gotGate = new Uint16Array(gateBytes)
  let bad = 0
  const wantQkv = f16Array(refQkv)
  const wantGate = f16Array(refGate)
  for (let i = 0; i < wantQkv.length; i++) if (gotQkv[i] !== wantQkv[i]) bad++
  for (let i = 0; i < wantGate.length; i++) if (gotGate[i] !== wantGate[i]) bad++
  return { name: 'gated_qkv_split 10240→6144+4096', pass: bad === 0, detail: `${bad} mismatched elements` }
}

// ── attn_gate — attn_out *= sigmoid(gate), in place ──────────────────────────
async function testAttnGate(device) {
  const r = rng(29)
  const N = Q.qDim
  const attnOut = arr(N, () => toF16(r() * 2 - 1))
  const gate = arr(N, () => toF16(r() * 8 - 4))
  const ref = attnOut.map((v, i) => toF16(v * sigmoid(gate[i])))

  const pipe = pipelineFor(device, wgsl('attn_gate.wgsl'), 'attn_gate')
  const io = buffer(device, f16Array(attnOut), BU.STORAGE | BU.COPY_DST | BU.COPY_SRC)
  const buffers = [
    io,
    buffer(device, f16Array(gate), BU.STORAGE | BU.COPY_DST),
    podBuffer(device, [{ u32: N / 256 }]),
  ]
  const bytes = await runCompute(device, pipe, buffers, [N / 256], 0, N * 2)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const maxAbs = maxAbsDiff(got, ref)
  return { name: 'attn_gate sigmoid', pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
}

// ── full gated-attention block — c_attn → split → qk_norm → partial rope →
// kv_append → attention → sigmoid gate → o_proj, 3 decode steps ─────────────
async function testGatedAttnBlockChain(device) {
  const r = rng(30)
  const d = Q.d, STEPS = 3, HD = Q.headDim
  const cAttnW = randInt4(r, Q.cAttnDim, d)
  const oW = randInt4(r, d, Q.qDim)
  const qGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))
  const kGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))
  const xs = arr(STEPS, () => arr(d, () => toF16(r() * 2 - 1)))

  const cache = { k: [], v: [] }
  const refOuts = xs.map((x, t) => refGatedAttnBlock(x, t, {
    cAttnW: cAttnW.weights, cAttnS: cAttnW.scales,
    oW: oW.weights, oS: oW.scales,
    qGamma, kGamma,
  }, Q, cache))

  const mm = pipelineFor(device, withPrelude(int4MatmulWGSL(), Q), 'int4_matmul')
  const split = pipelineFor(device, wgsl('gated_qkv_split.wgsl'), 'gated_qkv_split')
  const qkNorm = pipelineFor(device, wgsl('qk_norm.wgsl'), 'qk_norm')
  const rope = pipelineFor(device, wgsl('rope.wgsl'), 'rope_kernel')
  const kvAppend = pipelineFor(device, wgsl('kv_append.wgsl'), 'kv_append')
  const attn = pipelineFor(device, wgsl('attention.wgsl'), 'attention')
  const gateP = pipelineFor(device, wgsl('attn_gate.wgsl'), 'attn_gate')

  const NUM_PAGES = 1   // 3 tokens fit one 16-slot page
  const mk = (bytes) => device.createBuffer({ size: bytes, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  const hidden = mk(d * 2)
  const cAttnOut = mk(Q.cAttnDim * 2)
  const qkvBuf = mk(Q.qkvDim * 2)
  const gateBuf = mk(Q.qDim * 2)
  const qBuf = mk(Q.qDim * 2)
  const kBuf = mk(Q.kvDim * 2)
  const vBuf = mk(Q.kvDim * 2)
  const pages = mk(NUM_PAGES * Q.kvPageStride * 2)
  device.queue.writeBuffer(pages, 0, new Uint16Array(NUM_PAGES * Q.kvPageStride))
  const attnOut = mk(Q.qDim * 2)
  const outBuf = mk(d * 2)
  const posBuf = mk(4)
  const lenBuf = mk(4)
  const indptr = buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST)
  const pageVals = buffer(device, new Int32Array([0]), BU.STORAGE | BU.COPY_DST)
  const cAttnScales = buffer(device, f16Array(Array.from(cAttnW.scales)), BU.STORAGE | BU.COPY_DST)
  const cAttnWeights = buffer(device, cAttnW.weights, BU.STORAGE | BU.COPY_DST)
  const oScales = buffer(device, f16Array(Array.from(oW.scales)), BU.STORAGE | BU.COPY_DST)
  const oWeights = buffer(device, oW.weights, BU.STORAGE | BU.COPY_DST)
  const qGammaBuf = buffer(device, f16Array(qGamma), BU.STORAGE | BU.COPY_DST)
  const kGammaBuf = buffer(device, f16Array(kGamma), BU.STORAGE | BU.COPY_DST)

  const WGS_NORM = Q.heads + Q.kvHeads
  const ropeFreqsBuf = buffer(device, ropeInvFreqTable(Q), BU.STORAGE | BU.COPY_DST) // inv_freq (binding 6)
  const pods = {
    cAttn: buffer(device, new Uint32Array([cAttnW.KP, cAttnW.SPR, Q.cAttnDim, 0]), BU.UNIFORM | BU.COPY_DST),
    o: buffer(device, new Uint32Array([oW.KP, oW.SPR, d, 0]), BU.UNIFORM | BU.COPY_DST),
    split: podBuffer(device, [{ i32: 1 }, { u32: Q.cAttnDim / 256 }]),
    norm: podBuffer(device, [{ i32: 1 }, { u32: WGS_NORM }]),
    rope: buffer(device, new Uint32Array([1, 0, 1, Q.qkvDim / 256]), BU.UNIFORM | BU.COPY_DST),
    append: buffer(device, new Uint32Array([1, NUM_PAGES, 0, 0, Q.kvDim / 256]), BU.UNIFORM | BU.COPY_DST),
    attn: podBuffer(device, [
      { i32: 1 }, { i32: NUM_PAGES }, { i32: NUM_PAGES },
      { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 },
      { f32: 1 / Math.sqrt(HD) },
      { u32: 1 },
    ]),
    gate: podBuffer(device, [{ u32: Q.qDim / 256 }]),
  }

  const dispatch = async (pipe, bufs, grid) => {
    const bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    })
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipe)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, 1)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  let maxRel = 0
  for (let t = 0; t < STEPS; t++) {
    device.queue.writeBuffer(hidden, 0, f16Array(xs[t]))
    device.queue.writeBuffer(posBuf, 0, new Int32Array([t]))
    device.queue.writeBuffer(lenBuf, 0, new Int32Array([t + 1]))
    await dispatch(mm, [cAttnOut, hidden, cAttnScales, cAttnWeights, pods.cAttn], [Q.cAttnDim])
    await dispatch(split, [qkvBuf, gateBuf, cAttnOut, pods.split], [Q.cAttnDim / 256])
    await dispatch(qkNorm, [qkvBuf, qGammaBuf, kGammaBuf, pods.norm], [WGS_NORM])
    await dispatch(rope, [qBuf, kBuf, vBuf, qkvBuf, posBuf, pods.rope, ropeFreqsBuf], [Q.qkvDim / 256])
    await dispatch(kvAppend, [kBuf, vBuf, pages, posBuf, pods.append], [Q.kvDim / 256])
    await dispatch(attn, [qBuf, indptr, pageVals, pages, lenBuf, attnOut, pods.attn], [1, Q.heads])
    await dispatch(gateP, [attnOut, gateBuf, pods.gate], [Q.qDim / 256])
    const bytes = await runCompute(device, mm, [outBuf, attnOut, oScales, oWeights, pods.o], [d], 0, d * 2)
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    maxRel = Math.max(maxRel, maxRelDiff(got, refOuts[t], 1e-2))
  }
  // 3e-2 (vs the single-kernel 2e-2): this chain stacks EIGHT f16 buffer
  // round-trips (c_attn → split → qk_norm → rope → append → attention →
  // gate → o_proj) and the o_proj dot re-amplifies them; a wiring error
  // (e.g. swapped Q/gate interleave) diverges by O(1), not O(1e-2) — see the
  // gdn_recur pairing control for the failure magnitude of a real bug.
  return {
    name: 'gated_attn chain 3 steps',
    pass: maxRel < 0.03,
    detail: `max rel err ${maxRel.toExponential(2)} across ${STEPS} decode steps`,
  }
}

// Set in main() after the device probe — the chunked-prefill kernels
// (int4_matmul_batched_dyn) hard-assume 32-lane subgroups like every _sg
// variant; on adapters without that the chunked tests SKIP (the engine
// falls back to per-token prefill there too).
let HAS_SG32 = false
const skip = (name) => ({ name, pass: true, detail: 'SKIPPED — no 32-lane subgroup support (engine falls back to per-token prefill)' })

// Region-view-aware dispatch helper (bind-group entries may be
// { buffer, offset, size } views — same as the engine binds).
function dispatchViews(device, pipe, bufs, grid) {
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: b.buffer ? b : { buffer: b } })),
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipe)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(grid[0], grid[1] ?? 1, grid[2] ?? 1)
  pass.end()
  device.queue.submit([enc.finish()])
}

async function readBack(device, buf, bytes) {
  const out = device.createBuffer({ size: bytes, usage: BU.COPY_DST | BU.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(buf, 0, out, 0, bytes)
  device.queue.submit([enc.finish()])
  await out.mapAsync(MM.READ)
  return out.getMappedRange().slice(0)
}

// ── gdn_conv_seq + commit — chunked conv (2 chunks, engine stride) must be
// BIT-EXACT vs the per-token gdn_conv kernel, output AND ring state ──────────
async function testGdnConvSeq(device) {
  const r = rng(31)
  const C = Q.gdnQkvDim, K = Q.gdnConvK, RING = K - 1
  const STRIDE = Q.gdnProjRows      // batched fused-projection layout
  const STEPS = 7                   // chunks of 4 + 3 (exercises base_pos > 0)
  const xs = arr(STEPS, () => arr(C, () => toF16(r() * 2 - 1)))
  const convW = arr(C * K, () => toF16(r() - 0.5))
  const wBuf = buffer(device, f16Array(convW), BU.STORAGE | BU.COPY_DST)
  const GRID1 = C / 256

  // Per-token reference stream (the pinned gdn_conv kernel).
  const stepPipe = pipelineFor(device, wgsl('gdn_conv.wgsl'), 'gdn_conv')
  const refOut = new Uint16Array(STEPS * C)
  const stateA = device.createBuffer({ size: RING * C * 2, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC })
  device.queue.writeBuffer(stateA, 0, new Uint16Array(RING * C))
  {
    const out = device.createBuffer({ size: C * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const xBuf = device.createBuffer({ size: C * 2, usage: BU.STORAGE | BU.COPY_DST })
    for (let t = 0; t < STEPS; t++) {
      device.queue.writeBuffer(xBuf, 0, f16Array(xs[t]))
      const pod = podBuffer(device, [{ i32: t }, { u32: GRID1 }])
      const bytes = await runCompute(device, stepPipe, [out, xBuf, stateA, wBuf, pod], [GRID1], 0, C * 2)
      refOut.set(new Uint16Array(bytes), t * C)
    }
  }
  const ringRef = new Uint16Array(await readBack(device, stateA, RING * C * 2))

  // Chunked: same stream through gdn_conv_seq + gdn_conv_commit.
  const seqPipe = pipelineFor(device, wgsl('gdn_conv_seq.wgsl'), 'gdn_conv_seq')
  const commitPipe = pipelineFor(device, wgsl('gdn_conv_commit.wgsl'), 'gdn_conv_commit')
  const stateB = device.createBuffer({ size: RING * C * 2, usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC })
  device.queue.writeBuffer(stateB, 0, new Uint16Array(RING * C))
  const gotOut = new Uint16Array(STEPS * C)
  const CHUNKS = [[0, 4], [4, 3]]
  for (const [start, n] of CHUNKS) {
    // Batched raw buffer at the engine stride; non-qkv rows are random noise
    // to prove the kernel never reads past channel GDN_QKV_DIM.
    const raw = new Uint16Array(n * STRIDE)
    for (let i = 0; i < raw.length; i++) raw[i] = (r() * 0xffff) | 0
    for (let t = 0; t < n; t++)
      for (let c = 0; c < C; c++) raw[t * STRIDE + c] = f32ToF16Bits(xs[start + t][c])
    const rawBuf = buffer(device, raw, BU.STORAGE | BU.COPY_DST)
    const outBuf = device.createBuffer({ size: n * C * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const pod = podBuffer(device, [{ i32: start }, { i32: n }, { i32: STRIDE }, { u32: n * GRID1 }])
    const bytes = await runCompute(device, seqPipe, [outBuf, rawBuf, stateB, wBuf, pod], [n * GRID1], 0, n * C * 2)
    gotOut.set(new Uint16Array(bytes), start * C)
    const podCommit = podBuffer(device, [{ i32: start }, { i32: n }, { i32: STRIDE }, { u32: RING * GRID1 }])
    dispatchViews(device, commitPipe, [stateB, rawBuf, podCommit], [RING * GRID1])
  }
  const ringGot = new Uint16Array(await readBack(device, stateB, RING * C * 2))

  let outBad = 0
  for (let i = 0; i < refOut.length; i++) if (gotOut[i] !== refOut[i]) outBad++
  let ringBad = 0
  for (let i = 0; i < ringRef.length; i++) if (ringGot[i] !== ringRef[i]) ringBad++
  return {
    name: 'gdn_conv_seq 2 chunks',
    pass: outBad === 0 && ringBad === 0,
    detail: `${outBad} output / ${ringBad} ring mismatches vs per-token gdn_conv (must be bit-exact)`,
  }
}

// ── gdn_gates seq — strided batched gates BIT-EXACT vs per-token dispatches ──
async function testGdnGatesSeq(device) {
  const r = rng(34)
  const H = Q.gdnVHeads, SEQ = 3, STRIDE = Q.gdnProjRows
  const abs = arr(SEQ, () => arr(2 * H, () => toF16(r() * 4 - 2)))
  const aLog = arr(H, () => Math.fround(Math.log(r() * 3.5 + 0.5)))
  const dtBias = arr(H, () => Math.fround(r() - 1))
  const aLogBuf = buffer(device, new Float32Array(aLog), BU.STORAGE | BU.COPY_DST)
  const dtBuf = buffer(device, new Float32Array(dtBias), BU.STORAGE | BU.COPY_DST)
  const pipe = pipelineFor(device, wgsl('gdn_gates.wgsl'), 'gdn_gates')

  // Per-token reference.
  const ref = new Float32Array(SEQ * 2 * H)
  for (let t = 0; t < SEQ; t++) {
    const out = device.createBuffer({ size: 2 * H * 4, usage: BU.STORAGE | BU.COPY_SRC })
    const pod = podBuffer(device, [{ i32: 1 }, { i32: 2 * H }, { u32: 1 }])
    const bytes = await runCompute(
      device, pipe,
      [out, buffer(device, f16Array(abs[t]), BU.STORAGE | BU.COPY_DST), aLogBuf, dtBuf, pod],
      [1], 0, 2 * H * 4,
    )
    ref.set(new Float32Array(bytes), t * 2 * H)
  }

  // Batched strided (noise between tokens, as in the packed projection).
  const raw = new Uint16Array(SEQ * STRIDE)
  for (let i = 0; i < raw.length; i++) raw[i] = (r() * 0xffff) | 0
  for (let t = 0; t < SEQ; t++)
    for (let i = 0; i < 2 * H; i++) raw[t * STRIDE + i] = f32ToF16Bits(abs[t][i])
  const out = device.createBuffer({ size: SEQ * 2 * H * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const pod = podBuffer(device, [{ i32: SEQ }, { i32: STRIDE }, { u32: SEQ }])
  const bytes = await runCompute(
    device, pipe,
    [out, buffer(device, raw, BU.STORAGE | BU.COPY_DST), aLogBuf, dtBuf, pod],
    [SEQ], 0, SEQ * 2 * H * 4,
  )
  const got = new Float32Array(bytes)
  let bad = 0
  for (let i = 0; i < ref.length; i++) if (got[i] !== ref[i]) bad++
  return { name: 'gdn_gates seq=3 strided', pass: bad === 0, detail: `${bad} mismatches vs per-token (must be bit-exact)` }
}

// ── int4_matmul_batched_dyn — runtime-M GEMM vs CPU reference + M-invariance ─
async function testMatmulBatchedDyn(device) {
  if (!HAS_SG32) return skip('int4_matmul_batched_dyn')
  const r = rng(35)
  const K = Q.d, N = 64, M = 6, CAP = 8   // M=6 exercises the partial last block
  const w = randInt4(r, N, K)
  const input = arr(CAP * K, () => toF16(r() * 2 - 1))
  const pipe = pipelineFor(
    device,
    withPrelude(int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, mDyn: true }), Q),
    'int4_matmul_batched_dyn',
  )
  const inBuf = buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST)
  const wBuf = buffer(device, w.weights, BU.STORAGE | BU.COPY_DST)
  const sBuf = buffer(device, f16Array(Array.from(w.scales)), BU.STORAGE | BU.COPY_DST)
  const run = async (m) => {
    const out = device.createBuffer({ size: CAP * N * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
    device.queue.writeBuffer(out, 0, new Uint16Array(CAP * N))
    const pod = buffer(device, new Uint32Array([w.KP, w.SPR, N, m]), BU.UNIFORM | BU.COPY_DST)
    const bytes = await runCompute(device, pipe, [out, inBuf, sBuf, wBuf, pod], [N / 4], 0, CAP * N * 2)
    return new Uint16Array(bytes)
  }
  const got6 = await run(M)
  const got8 = await run(8)
  // CPU reference per batch row.
  let maxRel = 0
  for (let b = 0; b < M; b++) {
    const ref = int4Matvec(w.weights, w.scales, input.slice(b * K, (b + 1) * K), K, N)
    const row = Array.from(got6.subarray(b * N, (b + 1) * N), f16BitsToF32)
    maxRel = Math.max(maxRel, maxRelDiff(row, ref, 1e-2))
  }
  // Rows past M must stay untouched (zeroed).
  let tailBad = 0
  for (let i = M * N; i < CAP * N; i++) if (got6[i] !== 0) tailBad++
  // M-invariance: the same row is computed identically regardless of M.
  let mBad = 0
  for (let i = 0; i < M * N; i++) if (got6[i] !== got8[i]) mBad++
  const pass = maxRel < 1e-2 && tailBad === 0 && mBad === 0
  return {
    name: 'int4_matmul_batched_dyn',
    pass,
    detail: `max rel err ${maxRel.toExponential(2)} vs CPU, ${tailBad} rows-past-M writes, ${mBad} M-variance mismatches`,
  }
}

// ── int4_matmul_batched_dyn_affine — the same GEMM for MLX-affine weights ───
//
// WHY IT EXISTS. Chunked prefill dispatches int4_matmul_batched_dyn for every
// projection, and until now that kernel only understood MLC symmetric quant
// (w = s·(q−7), group 32). Every MLX checkpoint here is affine (w = s·q + b,
// group 64), so engine-core gated chunked prefill off for all of them and they
// prefilled ONE TOKEN AT A TIME. Prefill is the bottleneck for agentic use, so
// that gate was the expensive one.
//
// The reference below is deliberately written from the FORMULA (w = s·q + b),
// not from the kernel's factored form — the kernel folds the bias as b·Σx per
// group and this multiplies it out per element, so an error in that folding
// shows up as a mismatch rather than cancelling.
async function testMatmulBatchedDynAffine(device) {
  if (!HAS_SG32) return skip('int4_matmul_batched_dyn_affine')
  const r = rng(37)
  // Engine-scale: CHUNK_CAP is 64 and a chunk's M is whatever is left, so M=31
  // with CAP=64 is the shape a real prefill actually dispatches. The first
  // version of this test used M=6/CAP=8 and passed while the engine diverged.
  const K = Q.d, N = 256, M = 31, CAP = 64
  const KP = K / 8, SPR = K / 64             // group 64, not 32
  const weights = Uint32Array.from(arr(N * KP, () => (r() * 0xffffffff) >>> 0))
  const scales = arr(N * SPR, () => toF16(r() * 0.05 + 0.01))
  const biases = arr(N * SPR, () => toF16(r() * 0.1 - 0.05))
  const input = arr(CAP * K, () => toF16(r() * 2 - 1))

  const pipe = pipelineFor(
    device,
    withPrelude(int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, mDyn: true, affine: true }), Q),
    'int4_matmul_batched_dyn_affine',
  )
  const inBuf = buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST)
  const wBuf = buffer(device, weights, BU.STORAGE | BU.COPY_DST)
  const sBuf = buffer(device, f16Array(Array.from(scales)), BU.STORAGE | BU.COPY_DST)
  const bBuf = buffer(device, f16Array(Array.from(biases)), BU.STORAGE | BU.COPY_DST)
  const run = async (m) => {
    const out = device.createBuffer({ size: CAP * N * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
    device.queue.writeBuffer(out, 0, new Uint16Array(CAP * N))
    const pod = buffer(device, new Uint32Array([KP, SPR, N, m]), BU.UNIFORM | BU.COPY_DST)
    const bytes = await runCompute(device, pipe, [out, inBuf, sBuf, wBuf, pod, bBuf], [N / 4], 0, CAP * N * 2)
    return new Uint16Array(bytes)
  }
  const got6 = await run(M)
  // M-invariance must compare against a run that COVERS the same rows, so the
  // second M is CAP, not a smaller number. Comparing M=31 against M=8 counts
  // the 23 rows the second run never wrote as mismatches — (31-8)*256 = 5888 of
  // them, which looks exactly like a kernel bug.
  const got8 = await run(CAP)

  let maxRel = 0
  for (let b = 0; b < M; b++) {
    const x = input.slice(b * K, (b + 1) * K)
    const ref = arr(N, (_, row) => {
      let acc = 0
      for (let i = 0; i < K; i++) {
        const nib = (weights[row * KP + (i >> 3)] >>> ((i & 7) * 4)) & 15
        const g = i >> 6
        acc += (scales[row * SPR + g] * nib + biases[row * SPR + g]) * x[i]
      }
      return acc
    })
    const row = Array.from(got6.subarray(b * N, (b + 1) * N), f16BitsToF32)
    maxRel = Math.max(maxRel, maxRelDiff(row, ref, 1e-2))
  }
  let tailBad = 0
  for (let i = M * N; i < CAP * N; i++) if (got6[i] !== 0) tailBad++
  let mBad = 0
  for (let i = 0; i < M * N; i++) if (got6[i] !== got8[i]) mBad++
  const pass = maxRel < 1e-2 && tailBad === 0 && mBad === 0
  return {
    name: 'int4_matmul_batched_dyn_affine',
    pass,
    detail: `max rel err ${maxRel.toExponential(2)} vs CPU, ${tailBad} rows-past-M writes, ${mBad} M-variance mismatches`,
  }
}

// ── int4_matmul_tiled_m(+_affine) — the tiled prefill GEMM vs CPU reference ─
//
// The shapes are chosen to hit every edge the kernel guards:
//   M=47, CAP=64  -> TWO y-tiles (ceil(47/32)=2), second one ragged in M
//   N=200         -> SEVEN x-tiles, last one ragged in N (200 = 6*32 + 8)
// so a wrong guard shows up as a mismatch or a write past M, not silence.
// References computed from the FORMULA per element, never from the kernel's
// factored form — same discipline as the batched_dyn tests above.
function makeTiledTest(affine) {
  const name = affine ? 'int4_matmul_tiled_m_affine' : 'int4_matmul_tiled_m'
  return async function tiledTest(device) {
    const r = rng(affine ? 41 : 40)
    const K = Q.d, N = 200, M = 47, CAP = 64
    const GROUP = affine ? 64 : 32
    const KP = K / 8, SPR = K / GROUP
    const weights = Uint32Array.from(arr(N * KP, () => (r() * 0xffffffff) >>> 0))
    const scales = arr(N * SPR, () => toF16(r() * 0.05 + 0.01))
    const biases = affine ? arr(N * SPR, () => toF16(r() * 0.1 - 0.05)) : null
    const input = arr(CAP * K, () => toF16(r() * 2 - 1))

    const pipe = pipelineFor(device, withPrelude(int4MatmulTiledMWGSL(affine), Q), name)
    const inBuf = buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST)
    const wBuf = buffer(device, weights, BU.STORAGE | BU.COPY_DST)
    const sBuf = buffer(device, f16Array(Array.from(scales)), BU.STORAGE | BU.COPY_DST)
    const bBuf = affine ? buffer(device, f16Array(Array.from(biases)), BU.STORAGE | BU.COPY_DST) : null
    const run = async (m) => {
      const out = device.createBuffer({ size: CAP * N * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
      device.queue.writeBuffer(out, 0, new Uint16Array(CAP * N))
      const pod = buffer(device, new Uint32Array([KP, SPR, N, m]), BU.UNIFORM | BU.COPY_DST)
      const bufs = affine ? [out, inBuf, sBuf, wBuf, pod, bBuf] : [out, inBuf, sBuf, wBuf, pod]
      const bytes = await runCompute(device, pipe, bufs,
        [Math.ceil(N / 32), Math.ceil(m / 32)], 0, CAP * N * 2)
      return new Uint16Array(bytes)
    }
    const got = await run(M)
    const gotCap = await run(CAP)

    let maxRel = 0
    for (let b = 0; b < M; b++) {
      const x = input.slice(b * K, (b + 1) * K)
      const ref = arr(N, (_, row) => {
        let acc = 0
        for (let i = 0; i < K; i++) {
          const nib = (weights[row * KP + (i >> 3)] >>> ((i & 7) * 4)) & 15
          const g = Math.floor(i / GROUP)
          acc += affine
            ? (scales[row * SPR + g] * nib + biases[row * SPR + g]) * x[i]
            : scales[row * SPR + g] * (nib - 7) * x[i]
        }
        return acc
      })
      const gotRow = Array.from(got.subarray(b * N, (b + 1) * N), f16BitsToF32)
      maxRel = Math.max(maxRel, maxRelDiff(gotRow, ref, 1e-2))
    }
    let tailBad = 0
    for (let i = M * N; i < CAP * N; i++) if (got[i] !== 0) tailBad++
    let mBad = 0
    for (let i = 0; i < M * N; i++) if (got[i] !== gotCap[i]) mBad++
    const pass = maxRel < 1e-2 && tailBad === 0 && mBad === 0
    return {
      name, pass,
      detail: `max rel err ${maxRel.toExponential(2)} vs CPU, ${tailBad} rows-past-M writes, ${mBad} M-variance mismatches (M=47 CAP=64 N=200)`,
    }
  }
}
const testMatmulTiledM = makeTiledTest(false)
const testMatmulTiledMAffine = makeTiledTest(true)

// ── full chunked GDN chain — ONE chunk of 6 vs 6 single-token chunks (must be
// BIT-EXACT, f32 state included) and vs the sequential CPU reference ─────────
async function testGdnChunkChain(device) {
  if (!HAS_SG32) return skip('gdn_chunk_chain')
  const r = rng(36)
  const d = Q.d, S = 6, CAP = 8, STRIDE = Q.gdnProjRows
  const w = {
    qkv: randInt4(r, Q.gdnQkvDim, d),
    z: randInt4(r, Q.gdnVDim, d),
    a: randInt4(r, Q.gdnVHeads, d, 0.001, 0.004),
    b: randInt4(r, Q.gdnVHeads, d),
    out: randInt4(r, d, Q.gdnVDim),
  }
  const convW = arr(Q.gdnQkvDim * Q.gdnConvK, () => toF16(r() - 0.5))
  const gamma = arr(Q.gdnHeadV, () => toF16(r() * 0.5 + 0.75))
  const aLog = arr(Q.gdnVHeads, () => Math.fround(Math.log(r() * 3.5 + 0.5)))
  const dtBias = arr(Q.gdnVHeads, () => Math.fround(r() - 1))
  const xs = arr(S, () => arr(d, () => toF16(r() * 2 - 1)))

  const refState = { history: [], recur: new Float64Array(Q.gdnVHeads * Q.gdnStatePerHead) }
  const refOuts = refGdnBlock(xs, {
    qkvW: w.qkv.weights, qkvS: w.qkv.scales,
    zW: w.z.weights, zS: w.z.scales,
    aW: w.a.weights, aS: w.a.scales,
    bW: w.b.weights, bS: w.b.scales,
    outW: w.out.weights, outS: w.out.scales,
    convW, gamma, aLog, dtBias,
  }, Q, refState)

  // Packed projection (loader layout).
  const KP = d / 8, SPR = d / 32
  const projW = new Uint32Array(Q.gdnProjRows * KP)
  const projS = new Float32Array(Q.gdnProjRows * SPR)
  {
    let row = 0
    for (const rec of [w.qkv, w.z, w.a, w.b]) {
      projW.set(rec.weights, row * KP)
      projS.set(rec.scales, row * SPR)
      row += rec.weights.length / KP
    }
  }
  const dyn = pipelineFor(device, withPrelude(int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, mDyn: true }), Q), 'int4_matmul_batched_dyn')
  const conv = pipelineFor(device, wgsl('gdn_conv_seq.wgsl'), 'gdn_conv_seq')
  const commit = pipelineFor(device, wgsl('gdn_conv_commit.wgsl'), 'gdn_conv_commit')
  const gatesP = pipelineFor(device, wgsl('gdn_gates.wgsl'), 'gdn_gates')
  const recur = pipelineFor(device, wgsl('gdn_recur.wgsl'), 'gdn_recur')
  const norm = pipelineFor(device, wgsl('gdn_norm_out.wgsl'), 'gdn_norm_out')

  const projWBuf = buffer(device, projW, BU.STORAGE | BU.COPY_DST)
  const projSBuf = buffer(device, f16Array(Array.from(projS)), BU.STORAGE | BU.COPY_DST)
  const outWBuf = buffer(device, w.out.weights, BU.STORAGE | BU.COPY_DST)
  const outSBuf = buffer(device, f16Array(Array.from(w.out.scales)), BU.STORAGE | BU.COPY_DST)
  const convWBuf = buffer(device, f16Array(convW), BU.STORAGE | BU.COPY_DST)
  const gammaBuf = buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST)
  const aLogBuf = buffer(device, new Float32Array(aLog), BU.STORAGE | BU.COPY_DST)
  const dtBiasBuf = buffer(device, new Float32Array(dtBias), BU.STORAGE | BU.COPY_DST)

  const CONV_GRID1 = Q.gdnQkvDim / 256
  const RING = Q.gdnConvK - 1

  async function runChain(chunks) {
    const mk = (bytes) => device.createBuffer({ size: bytes, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
    const hidden = mk(CAP * d * 2)
    const projOut = mk(CAP * STRIDE * 2)
    const convState = mk(RING * Q.gdnQkvDim * 2)
    device.queue.writeBuffer(convState, 0, new Uint16Array(RING * Q.gdnQkvDim))
    const convOut = mk(CAP * Q.gdnQkvDim * 2)
    const gatesBuf = mk(CAP * 2 * Q.gdnVHeads * 4)
    const recurState = mk(Q.gdnVHeads * Q.gdnStatePerHead * 4)
    device.queue.writeBuffer(recurState, 0, new Float32Array(Q.gdnVHeads * Q.gdnStatePerHead))
    const recurOut = mk(CAP * Q.gdnVDim * 4)
    const normOut = mk(CAP * Q.gdnVDim * 2)
    const blockOut = mk(CAP * d * 2)
    const zRegion = { buffer: projOut, offset: Q.gdnQkvDim * 2, size: (CAP - 1) * STRIDE * 2 + Q.gdnVDim * 2 }
    const abRegion = { buffer: projOut, offset: (Q.gdnQkvDim + Q.gdnVDim) * 2, size: (CAP - 1) * STRIDE * 2 + 2 * Q.gdnVHeads * 2 }

    const outs = new Uint16Array(S * d)
    let start = 0
    for (const n of chunks) {
      const hid = new Uint16Array(n * d)
      for (let t = 0; t < n; t++)
        for (let i = 0; i < d; i++) hid[t * d + i] = f32ToF16Bits(xs[start + t][i])
      device.queue.writeBuffer(hidden, 0, hid)
      const pods = {
        proj: buffer(device, new Uint32Array([KP, SPR, Q.gdnProjRows, n]), BU.UNIFORM | BU.COPY_DST),
        conv: podBuffer(device, [{ i32: start }, { i32: n }, { i32: STRIDE }, { u32: n * CONV_GRID1 }]),
        commit: podBuffer(device, [{ i32: start }, { i32: n }, { i32: STRIDE }, { u32: RING * CONV_GRID1 }]),
        gates: podBuffer(device, [{ i32: n }, { i32: STRIDE }, { u32: n }]),
        recur: podBuffer(device, [{ i32: n }, { u32: Q.gdnVHeads }]),
        norm: podBuffer(device, [{ i32: n }, { i32: STRIDE }, { u32: n * Q.gdnVHeads }]),
        out: buffer(device, new Uint32Array([w.out.KP, w.out.SPR, d, n]), BU.UNIFORM | BU.COPY_DST),
      }
      dispatchViews(device, dyn, [projOut, hidden, projSBuf, projWBuf, pods.proj], [Q.gdnProjRows / 4])
      dispatchViews(device, conv, [convOut, projOut, convState, convWBuf, pods.conv], [n * CONV_GRID1])
      dispatchViews(device, commit, [convState, projOut, pods.commit], [RING * CONV_GRID1])
      dispatchViews(device, gatesP, [gatesBuf, abRegion, aLogBuf, dtBiasBuf, pods.gates], [n])
      dispatchViews(device, recur, [recurOut, convOut, gatesBuf, recurState, pods.recur], [Q.gdnVHeads])
      dispatchViews(device, norm, [normOut, recurOut, gammaBuf, zRegion, pods.norm], [n * Q.gdnVHeads])
      dispatchViews(device, dyn, [blockOut, normOut, outSBuf, outWBuf, pods.out], [d / 4])
      const bytes = await readBack(device, blockOut, n * d * 2)
      outs.set(new Uint16Array(bytes), start * d)
      start += n
    }
    const state = new Float32Array(await readBack(device, recurState, Q.gdnVHeads * Q.gdnStatePerHead * 4))
    return { outs, state }
  }

  const chunked = await runChain([S])
  const stepwise = await runChain([1, 1, 1, 1, 1, 1])
  let outBits = 0
  for (let i = 0; i < chunked.outs.length; i++) if (chunked.outs[i] !== stepwise.outs[i]) outBits++
  let stateBits = 0
  for (let i = 0; i < chunked.state.length; i++) if (chunked.state[i] !== stepwise.state[i]) stateBits++
  // vs CPU: mixed tolerance — |Δ| < max(2e-2, 2e-2·|ref|). Six recurrence
  // steps amplify f16 input rounding on near-zero outputs, so a pure
  // relative metric over-penalizes; the BIT-EXACT chunk-vs-stepwise check
  // above (same kernels, f32 state included) is the load-bearing equality,
  // and the per-token semantics vs CPU are pinned by gdn_block.
  let cpuWorst = 0
  for (let t = 0; t < S; t++) {
    for (let i = 0; i < d; i++) {
      const got = f16BitsToF32(chunked.outs[t * d + i])
      const ref = refOuts[t][i]
      const tol = Math.max(2e-2, 2e-2 * Math.abs(ref))
      cpuWorst = Math.max(cpuWorst, Math.abs(got - ref) / tol)
    }
  }
  const refStateF32 = Float32Array.from(refState.recur)
  const stateErr = maxRelDiff(chunked.state, refStateF32, 1e-2)
  // State tolerance 5e-3 (vs gdn_recur_seq's 1e-3): here the recurrence
  // inputs passed through the f16 projection + conv stages, adding their
  // rounding to the drift a direct-input recurrence doesn't see.
  const pass = outBits === 0 && stateBits === 0 && cpuWorst < 1 && stateErr < 5e-3
  return {
    name: 'gdn_chunk_chain 6 tok',
    pass,
    detail: `chunk-vs-stepwise: ${outBits} out / ${stateBits} f32-state bit diffs; vs CPU: out ${cpuWorst.toFixed(3)}×tol, state rel ${stateErr.toExponential(2)}`,
  }
}

// ── attention_prefill — chunk of 5 query tokens BIT-EXACT vs per-token decode ─
async function testAttentionPrefill(device) {
  const r = rng(37)
  const BASE = 3, SEQ = 5, T = BASE + SEQ   // 8 tokens fit one 16-slot page
  const NUM_PAGES = 1
  // Random K/V for positions 0..T-1, written straight into the page layout.
  const pageData = new Uint16Array(NUM_PAGES * Q.kvPageStride)
  for (let h = 0; h < Q.kvHeads; h++) {
    for (let t = 0; t < T; t++) {
      for (let dd = 0; dd < Q.headDim; dd++) {
        pageData[h * Q.headPageStride + t * Q.headDim + dd] = f32ToF16Bits(toF16(r() - 0.5))
        pageData[Q.vPageOffset + h * Q.headPageStride + t * Q.headDim + dd] = f32ToF16Bits(toF16(r() - 0.5))
      }
    }
  }
  const pages = buffer(device, pageData, BU.STORAGE | BU.COPY_DST)
  const qAll = arr(SEQ * Q.qDim, () => toF16(r() * 2 - 1))
  const pageVals = buffer(device, new Int32Array([0]), BU.STORAGE | BU.COPY_DST)
  const indptr = buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST)

  // Per-token reference via the decode attention kernel.
  const decodeAttn = pipelineFor(device, wgsl('attention.wgsl'), 'attention')
  const ref = new Uint16Array(SEQ * Q.qDim)
  {
    const qBuf = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_DST })
    const out = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
    for (let t = 0; t < SEQ; t++) {
      device.queue.writeBuffer(qBuf, 0, f16Array(qAll.slice(t * Q.qDim, (t + 1) * Q.qDim)))
      device.queue.writeBuffer(out, 0, new Uint16Array(Q.qDim))
      const lenBuf = buffer(device, new Int32Array([BASE + 1 + t]), BU.STORAGE | BU.COPY_DST)
      const pod = podBuffer(device, [
        { i32: 1 }, { i32: NUM_PAGES }, { i32: NUM_PAGES },
        { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 },
        { f32: 1 / Math.sqrt(Q.headDim) }, { u32: 1 },
      ])
      const bytes = await runCompute(
        device, decodeAttn,
        [qBuf, indptr, pageVals, pages, lenBuf, out, pod],
        [1, Q.heads], 5, Q.qDim * 2,
      )
      ref.set(new Uint16Array(bytes), t * Q.qDim)
    }
  }

  // Chunked: one dispatch, per-token causal kv_len.
  const prefillAttn = pipelineFor(device, wgsl('attention_prefill.wgsl'), 'attention_prefill')
  const qBatch = buffer(device, f16Array(qAll), BU.STORAGE | BU.COPY_DST)
  const outBatch = device.createBuffer({ size: SEQ * Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
  device.queue.writeBuffer(outBatch, 0, new Uint16Array(SEQ * Q.qDim))
  const pod = podBuffer(device, [{ i32: SEQ }, { i32: BASE + 1 }, { f32: 1 / Math.sqrt(Q.headDim) }])
  const bytes = await runCompute(device, prefillAttn, [qBatch, pageVals, pages, outBatch, pod], [SEQ, Q.heads], 3, SEQ * Q.qDim * 2)
  const got = new Uint16Array(bytes)
  let bad = 0
  for (let i = 0; i < ref.length; i++) if (got[i] !== ref[i]) bad++
  return { name: 'attention_prefill seq=5', pass: bad === 0, detail: `${bad} mismatches vs per-token decode attention (must be bit-exact)` }
}

// ── silu_mul — chunked FFN epilogue vs CPU ───────────────────────────────────
async function testSiluMul(device) {
  const r = rng(38)
  const SEQ = 3, F = Q.ffn
  const gu = arr(SEQ * 2 * F, () => toF16(r() * 6 - 3))
  const ref = new Array(SEQ * F)
  for (let t = 0; t < SEQ; t++)
    for (let i = 0; i < F; i++) {
      const g = gu[t * 2 * F + i]
      ref[t * F + i] = toF16(gu[t * 2 * F + F + i] * (g / (1 + Math.exp(-g))))
    }
  const pipe = pipelineFor(device, wgsl('silu_mul.wgsl'), 'silu_mul')
  const out = device.createBuffer({ size: SEQ * F * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const GRID = (SEQ * F) / 256
  const bytes = await runCompute(
    device, pipe,
    [out, buffer(device, f16Array(gu), BU.STORAGE | BU.COPY_DST), podBuffer(device, [{ i32: SEQ }, { u32: GRID }])],
    [GRID], 0, SEQ * F * 2,
  )
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  const maxAbs = maxAbsDiff(got, ref)
  return { name: 'silu_mul', pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
}

// ── test roster ──────────────────────────────────────────────────────────────
const recurFixture = buildRecurFixture()
const TESTS = [
  { label: 'gdn_conv', fn: testGdnConv },
  { label: 'gdn_gates', fn: testGdnGates },
  { label: 'gdn_recur_seq', fn: (d) => testGdnRecurSeq(d, recurFixture) },
  { label: 'gdn_recur_step', fn: (d) => testGdnRecurStepwise(d, recurFixture) },
  { label: 'gdn_recur_pairing', fn: () => testGdnRecurPairingControl(recurFixture) },
  { label: 'gdn_norm_out', fn: testGdnNormOut },
  { label: 'gdn_block', fn: testGdnBlockChain },
  { label: 'rope_partial', fn: testRopePartial },
  { label: 'qk_norm', fn: testQkNorm256 },
  { label: 'gated_qkv_split', fn: testGatedQkvSplit },
  { label: 'attn_gate', fn: testAttnGate },
  { label: 'gated_attn_block', fn: testGatedAttnBlockChain },
  // Chunked-prefill kernel family (perf round A):
  { label: 'gdn_conv_seq', fn: testGdnConvSeq },
  { label: 'gdn_gates_seq', fn: testGdnGatesSeq },
  { label: 'matmul_batched_dyn', fn: testMatmulBatchedDyn },
  { label: 'matmul_batched_dyn_affine', fn: testMatmulBatchedDynAffine },
  { label: 'matmul_tiled_m', fn: testMatmulTiledM },
  { label: 'matmul_tiled_m_affine', fn: testMatmulTiledMAffine },
  { label: 'gdn_chunk_chain', fn: testGdnChunkChain },
  { label: 'attention_prefill', fn: testAttentionPrefill },
  { label: 'silu_mul', fn: testSiluMul },
]

// ── compile-all gate — every shader must build under QWEN35_4B ───────────────
async function compileAll(device, subgroups) {
  const sources = readdirSync(SHADERS)
    .filter((f) => f.endsWith('.wgsl'))
    .sort()
    .map((f) => ({ name: f, code: wgsl(f) }))
  for (const v of INT4_MATMUL_VARIANTS) {
    sources.push({ name: `gen:${int4MatmulEntry(v)}`, code: withPrelude(int4MatmulWGSL(v), Q) })
  }

  const errors = []
  let compiled = 0
  let skippedSg = 0
  for (const s of sources) {
    if (!subgroups && /^\s*enable subgroups;/m.test(s.code)) {
      skippedSg++
      continue
    }
    const entryMatch = s.code.match(/@compute[\s\S]*?fn\s+([A-Za-z0-9_]+)/)
    if (!entryMatch) {
      errors.push(`${s.name}: no @compute entry point found`)
      continue
    }
    device.pushErrorScope('validation')
    const module = device.createShaderModule({ code: s.code })
    const info = module.getCompilationInfo ? await module.getCompilationInfo() : { messages: [] }
    const msgs = [...info.messages].filter((m) => m.type === 'error')
    if (msgs.length) {
      errors.push(`${s.name}: ${msgs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ')}`)
      await device.popErrorScope()
      continue
    }
    device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: entryMatch[1] },
    })
    const scopeErr = await device.popErrorScope()
    if (scopeErr) {
      errors.push(`${s.name}: ${String(scopeErr.message).split('\n')[0]}`)
      continue
    }
    compiled++
  }
  for (const e of errors) console.error(`      ${e}`)
  const skipNote = skippedSg ? `, ${skippedSg} sg-only skipped (no 'subgroups' feature)` : ''
  return {
    name: 'compile_all',
    pass: errors.length === 0,
    detail: `${compiled} shaders compile under QWEN35_4B (${sources.length - INT4_MATMUL_VARIANTS.length} .wgsl + ${INT4_MATMUL_VARIANTS.length} generated${skipNote})`,
  }
}

async function main() {
  let env
  try {
    env = await getDevice()
  } catch (e) {
    console.error(`ERROR: ${e.message}`)
    console.error(
      'No WebGPU adapter. On a Linux CI runner install lavapipe:\n' +
        '  apt-get install -y mesa-vulkan-drivers\n' +
        '  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json\n' +
        'or run on a machine with a real GPU.',
    )
    process.exit(1)
  }
  const { device, info, f16, subgroups } = env
  let sgSize = 0
  if (subgroups) {
    try {
      sgSize = await probeSubgroupSize(device)
    } catch (e) {
      console.error(`WARN: subgroup-size probe failed (${String(e).split('\n')[0]}); treating as no subgroups`)
    }
  }
  HAS_SG32 = subgroups && sgSize === 32
  const sgLabel = subgroups ? (sgSize ? `yes (size ${sgSize})` : 'yes (size unknown)') : 'no'
  console.log(
    `adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16} | subgroups: ${sgLabel} | spec: ${Q.id}`,
  )
  if (!f16) {
    console.error('FAIL: adapter lacks shader-f16; cannot exercise the f16 kernels')
    process.exit(1)
  }

  let failed = 0
  let ran = 0
  for (const t of TESTS) {
    let res
    try {
      res = await t.fn(device)
    } catch (e) {
      res = { pass: false, detail: String(e).split('\n')[0] }
    }
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${t.label.padEnd(18)} ${res.detail}`)
    ran++
    if (!res.pass) failed++
  }

  {
    let res
    try {
      res = await compileAll(device, subgroups)
    } catch (e) {
      res = { pass: false, detail: String(e).split('\n')[0] }
    }
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${'compile_all'.padEnd(18)} ${res.detail}`)
    ran++
    if (!res.pass) failed++
  }

  console.log(`\n${ran - failed}/${ran} Qwen3.5 kernels correct`)
  process.exit(failed ? 1 : 0)
}

main()
