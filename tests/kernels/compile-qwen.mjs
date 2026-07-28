// QWEN3_4B kernel-correctness suite (phase B of the Qwen3 port).
//
// Upgraded from phase A's compile-only gate: every GQA-sensitive kernel now
// runs under shaderPrelude(QWEN3_4B) — D=2560, HEADS=32, KV_HEADS=8,
// HEAD_DIM=128, GQA_GROUP=4, FFN=9728, VOCAB=151936, ROPE_THETA=1e6,
// RMS_EPS=1e-6 — against a plain-JS reference, on the local WebGPU adapter.
// Deterministic (seeded RNG). Exits non-zero on any mismatch.
//
// The attention tests seed every KV head with DISTINCT random content and a
// shuffled page table, so an implementation with the wrong query-head → KV-head
// mapping (anything other than h / GQA_GROUP) FAILS rather than silently
// passing on replicated data.
//
// The compile-all gate from phase A is kept at the end: every .wgsl file and
// every int4_matmul generator variant must still build under QWEN3_4B.
//
//   npm run test:kernels:qwen

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
import { withPrelude, QWEN3_4B } from '../../src/compiler/shader-prelude.ts'
import { resolveMatmul } from '../../src/zero-tvm/variants.ts'
import {
  int4MatmulWGSL,
  int4MatmulEntry,
  INT4_MATMUL_VARIANTS,
} from '../../src/compiler/shaders/int4_matmul.gen.ts'

const Q = QWEN3_4B
const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')
// Same render step as the app's createShaderModule sites, but with the
// QWEN3_4B prelude.
const wgsl = (name) => withPrelude(readFileSync(resolve(SHADERS, name), 'utf8'), Q)

// Deterministic PRNG (mulberry32) — same as run.mjs.
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

// int4 dot-product reference shared by the matmul-shaped tests.
function makeDot(weights, scales, input, KP, SPR) {
  return (row) => {
    let acc = 0
    for (let w = 0; w < KP; w++) {
      const packed = weights[row * KP + w]
      const scale = scales[row * SPR + (w >> 2)]
      const base = w * 8
      for (let n = 0; n < 8; n++) acc += input[base + n] * (((packed >>> (4 * n)) & 15) - 7) * scale
    }
    return acc
  }
}

// Qwen3 RoPE: full-head rotation, pairs at distance HALF_HEAD_DIM, theta 1e6.
const ropeFreq = (pos, dimLo) => pos / Math.pow(Q.ropeTheta, (dimLo * 2) / Q.headDim)

// ── vec4 K-divisibility gating (pure JS — resolveMatmul / variants.ts) ───────
// Sentinel "pipelines": resolveMatmul only reads/forwards these fields.
function testVec4Gating() {
  const mk = (label) => ({ __sentinel: label })
  const P = {
    int4Matmul: mk('scalar'), lmHead: mk('scalar_f32'),
    int4MatmulSg: mk('sg'), int4MatmulF32Sg: mk('sg_f32'),
    int4MatmulTiled: mk('tiled'), int4MatmulF32Tiled: mk('tiled_f32'),
    int4MatmulTiled8: mk('tiled8'), int4MatmulF32Tiled8: mk('tiled8_f32'),
    int4MatmulSgVec4: mk('sg_vec4'), int4MatmulF32SgVec4: mk('sg_vec4_f32'),
    int4MatmulTiledVec4: mk('tiled_vec4'), int4MatmulF32TiledVec4: mk('tiled_vec4_f32'),
  }
  const cases = [
    // [variant, k, expected label] — Qwen3 instances: d=2560, qDim=4096, ffn=9728
    ['tiled', Q.d, 'tiled'],          // K=2560 % 1024 ≠ 0 → vec4 falls back
    ['tiled', Q.qDim, 'tiled_vec4'],  // K=4096 → vec4 survives (o_proj)
    ['tiled', Q.ffn, 'tiled'],        // K=9728 → falls back (ffn_down)
    ['sg', Q.d, 'sg'],
    ['sg', Q.qDim, 'sg_vec4'],
    ['tiled', 3072, 'tiled_vec4'],    // Phi-3 d — unchanged resolution
    ['tiled', 8192, 'tiled_vec4'],    // Phi-3 ffn
    ['tiled', undefined, 'tiled_vec4'], // no K given — legacy behaviour
  ]
  const bad = []
  for (const [variant, k, want] of cases) {
    const got = resolveMatmul(variant, P, true, k)
    // rowsPerWG must not change when the gate strips vec4 (grids stay valid).
    const gotNo4 = resolveMatmul(variant, P, false, k)
    if (got.label !== want) bad.push(`${variant} K=${k}: got ${got.label}, want ${want}`)
    if (got.rowsPerWG !== gotNo4.rowsPerWG) bad.push(`${variant} K=${k}: rowsPerWG changed`)
  }
  return Promise.resolve({
    name: 'vec4_k_gating',
    pass: bad.length === 0,
    detail: bad.length ? bad.join('; ') : `${cases.length} resolveMatmul cases (d/qDim/ffn × tiled/sg)`,
  })
}

// ── int4_matmul — full gate_up shape: K=2560, N=9728 rows ────────────────────
function makeInt4MatmulTest(src, entry, { rowsPerWG = 1 } = {}) {
  return function int4MatmulTest(device) {
    const r = rng(1)
    const K = Q.d, M = Q.ffn, KP = K / 8, SPR = K / 32
    const input = arr(K, () => toF16(r() * 2 - 1))
    const scales = arr(M * SPR, () => toF16(r() * 0.05 + 0.01))
    const weights = Uint32Array.from(arr(M * KP, () => (r() * 0xffffffff) >>> 0))
    const dot = makeDot(weights, scales, input, KP, SPR)

    // Spot-check rows across the range (full-N ref would be ~25M MACs × N).
    const rows = [0, 1, 2, 3, 4863, 9724, 9725, 9726, 9727]

    const pipe = pipelineFor(device, withPrelude(src, Q), entry)
    const out = device.createBuffer({ size: M * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      out,
      buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([KP, SPR, M, 0]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runCompute(device, pipe, buffers, [M / rowsPerWG], 0, M * 2).then((bytes) => {
      const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxRel = Math.max(
        ...rows.map((row) => {
          const ref = toF16(dot(row))
          return Math.abs(got[row] - ref) / (Math.abs(ref) + 1e-3)
        }),
      )
      return { name: `${entry} K=2560 N=9728`, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── rms_norm — d=2560, eps 1e-6 from RMS_EPS ─────────────────────────────────
function testRmsNorm(device) {
  const r = rng(2)
  const D = Q.d
  const input = arr(D, () => toF16(r() * 2 - 1))
  const gamma = arr(D, () => toF16(r() * 0.5 + 0.75))
  let ss = 0
  for (const v of input) ss += v * v
  const rinv = 1 / Math.sqrt(ss / D + Q.rmsEps)
  const ref = input.map((v, i) => toF16(v * rinv * gamma[i]))

  const pipe = pipelineFor(device, wgsl('rms_norm.wgsl'), 'rms_norm')
  const out = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [1], 0, D * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
    return { name: 'rms_norm d=2560 eps=1e-6', pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
  })
}

// ── embedding — D=2560 dequant lookup (64-row vocab slice) ───────────────────
function testEmbedding(device) {
  const r = rng(4)
  const D = Q.d, VOCAB = 64, DP = D / 8, SPR = D / 32, token = 17
  const scales = arr(VOCAB * SPR, () => toF16(r() * 0.05 + 0.01))
  const weights = Uint32Array.from(arr(VOCAB * DP, () => (r() * 0xffffffff) >>> 0))
  const ref = arr(D, (_, i) => {
    const nib = (weights[token * DP + (i >> 3)] >>> ((i & 7) * 4)) & 15
    return toF16((nib - 7) * scales[token * SPR + (i >> 5)])
  })
  const pipe = pipelineFor(device, wgsl('embedding.wgsl'), 'embedding')
  const out = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, new Int32Array([token]), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
    buffer(device, weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1, D / 256]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [D / 256], 0, D * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const bad = got.reduce((m, g, i) => Math.max(m, Math.abs(g - ref[i])), 0)
    return { name: 'embedding d=2560', pass: bad === 0, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── rope — asymmetric [Q 4096 | K 1024 | V 1024] layout, theta 1e6 ───────────
function testRope(device) {
  const r = rng(5)
  const QKV = Q.qkvDim, HD = Q.headDim, HALF = Q.halfHeadDim, pos = 7
  const qkv = arr(QKV, () => toF16(r() * 2 - 1))
  const rope = (val, pair, dim) => {
    const freq = ropeFreq(pos, dim % HALF)
    return toF16(Math.cos(freq) * val + Math.sin(freq) * pair)
  }
  const q = new Array(Q.qDim), k = new Array(Q.kvDim), v = new Array(Q.kvDim)
  for (let within = 0; within < QKV; within++) {
    const dim = within % HD
    const val = qkv[within]
    if (within < Q.qDim + Q.kvDim) {
      // Q or K — rotate. Pair partner is ±HALF inside the same head.
      const pair = dim < HALF ? -qkv[within + HALF] : qkv[within - HALF]
      const out = rope(val, pair, dim)
      if (within < Q.qDim) q[within] = out
      else k[within - Q.qDim] = out
    } else {
      v[within - Q.qDim - Q.kvDim] = val
    }
  }
  const pipe = pipelineFor(device, wgsl('rope.wgsl'), 'rope_kernel')
  const mk = (n) => device.createBuffer({ size: n * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    mk(Q.qDim), mk(Q.kvDim), mk(Q.kvDim),
    buffer(device, f16Array(qkv), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array([pos]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1, 0, 1, QKV / 256]), BU.UNIFORM | BU.COPY_DST),
  ]
  const reads = [
    { index: 0, bytes: Q.qDim * 2 },
    { index: 1, bytes: Q.kvDim * 2 },
    { index: 2, bytes: Q.kvDim * 2 },
  ]
  return runComputeReads(device, pipe, buffers, [QKV / 256], reads).then(([qb, kb, vb]) => {
    const dec = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
    const [gq, gk, gv] = [dec(qb), dec(kb), dec(vb)]
    let bad = 0
    for (let i = 0; i < Q.qDim; i++) bad = Math.max(bad, Math.abs(gq[i] - q[i]))
    for (let i = 0; i < Q.kvDim; i++) bad = Math.max(bad, Math.abs(gk[i] - k[i]), Math.abs(gv[i] - v[i]))
    return { name: 'rope GQA theta=1e6', pass: bad < 5e-3, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── qk_norm — per-head RMSNorm of Q/K in place, V untouched ──────────────────
function testQkNorm(device) {
  const r = rng(11)
  const QKV = Q.qkvDim, HD = Q.headDim
  const qkv = arr(QKV, () => toF16(r() * 2 - 1))
  const qGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))
  const kGamma = arr(HD, () => toF16(r() * 0.5 + 0.75))

  // CPU reference: RMSNorm(headDim, eps) per Q head (q_gamma) and K head
  // (k_gamma); the V group must come back bit-identical.
  const ref = qkv.slice()
  const normHead = (base, gamma) => {
    let ss = 0
    for (let d = 0; d < HD; d++) ss += qkv[base + d] * qkv[base + d]
    const rinv = 1 / Math.sqrt(ss / HD + Q.rmsEps)
    for (let d = 0; d < HD; d++) ref[base + d] = toF16(qkv[base + d] * rinv * gamma[d])
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
  return runCompute(device, pipe, buffers, [WGS], 0, QKV * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    let maxAbs = 0
    for (let i = 0; i < QKV; i++) maxAbs = Math.max(maxAbs, Math.abs(got[i] - ref[i]))
    // V region must be untouched (bit-exact).
    let vBad = 0
    for (let i = Q.qDim + Q.kvDim; i < QKV; i++) if (got[i] !== ref[i]) vBad++
    const pass = maxAbs < 5e-3 && vBad === 0
    return { name: 'qk_norm per-head', pass, detail: `max abs err ${maxAbs.toExponential(2)}, ${vBad} V elems touched` }
  })
}

// ── qkv_fused — GQA decomposition + RoPE + paged KV append ───────────────────
// Full 6144-row projection; the reference spot-checks Q heads {0,15,31}
// (rotated, → q_out), K heads {0,7} (rotated, → per-KV-head page rows) and V
// heads {0,7} (raw, → V page region).
function makeQkvFusedTest(file, entry) {
  return function qkvFusedTest(device) {
    const r = rng(10)
    const D = Q.d, ROWS = Q.qkvDim, KP = Q.dPacked, SPR = Q.dScales
    const HD = Q.headDim, HALF = Q.halfHeadDim
    const POS = 20, NUM_PAGES = 2, PAGE = Q.kvPageStride
    const pageNo = (POS / Q.pageSize) | 0, slot = POS % Q.pageSize
    const PAIRS = Q.qkvPairs

    const hidden = new Float32Array(D)
    for (let i = 0; i < D; i++) hidden[i] = toF16(r() * 2 - 1)
    const scales = new Float32Array(ROWS * SPR)
    for (let i = 0; i < scales.length; i++) scales[i] = toF16(r() * 0.05 + 0.01)
    const weights = new Uint32Array(ROWS * KP)
    for (let i = 0; i < weights.length; i++) weights[i] = (r() * 0xffffffff) >>> 0
    const dot = makeDot(weights, scales, hidden, KP, SPR)

    const pipe = pipelineFor(device, wgsl(file), entry)
    const qOut = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const kvPages = device.createBuffer({
      size: NUM_PAGES * PAGE * 2,
      usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
    })
    device.queue.writeBuffer(kvPages, 0, new Uint16Array(NUM_PAGES * PAGE))
    const buffers = [
      qOut,
      kvPages,
      buffer(device, f16Array(Array.from(hidden)), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(Array.from(scales)), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([POS]), BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([0, 0, PAIRS, 0]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runComputeReads(device, pipe, buffers, [PAIRS], [
      { index: 0, bytes: Q.qDim * 2 },
      { index: 1, bytes: NUM_PAGES * PAGE * 2 },
    ]).then(([qBytes, pageBytes]) => {
      const gotQ = Array.from(new Uint16Array(qBytes), f16BitsToF32)
      const gotPages = Array.from(new Uint16Array(pageBytes), f16BitsToF32)
      let maxRel = 0
      const check = (got, want) => {
        maxRel = Math.max(maxRel, Math.abs(got - want) / (Math.abs(want) + 1e-3))
      }
      const rot = (dimLo, vLo, vHi) => {
        const freq = ropeFreq(POS, dimLo)
        const c = Math.cos(freq), s = Math.sin(freq)
        return [toF16(c * vLo - s * vHi), toF16(c * vHi + s * vLo)]
      }
      for (let dimLo = 0; dimLo < HALF; dimLo++) {
        // Q heads → q_out (rows [0, qDim)).
        for (const qh of [0, 15, 31]) {
          const rowLo = qh * HD + dimLo
          const [lo, hi] = rot(dimLo, dot(rowLo), dot(rowLo + HALF))
          check(gotQ[qh * HD + dimLo], lo)
          check(gotQ[qh * HD + dimLo + HALF], hi)
        }
        // K heads → pages at the KV-head stride (rows [qDim, qDim+kvDim)).
        for (const kh of [0, 7]) {
          const rowLo = Q.qDim + kh * HD + dimLo
          const [lo, hi] = rot(dimLo, dot(rowLo), dot(rowLo + HALF))
          const kBase = pageNo * PAGE + kh * Q.headPageStride + slot * HD
          check(gotPages[kBase + dimLo], lo)
          check(gotPages[kBase + dimLo + HALF], hi)
        }
        // V heads → V page region, no RoPE (rows [qDim+kvDim, qkvDim)).
        for (const vh of [0, 7]) {
          const rowLo = Q.qDim + Q.kvDim + vh * HD + dimLo
          const vBase = pageNo * PAGE + vh * Q.headPageStride + slot * HD + Q.vPageOffset
          check(gotPages[vBase + dimLo], toF16(dot(rowLo)))
          check(gotPages[vBase + dimLo + HALF], toF16(dot(rowLo + HALF)))
        }
      }
      return { name: `${entry} GQA`, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── shared GQA attention fixture ─────────────────────────────────────────────
// All 32 query heads over 8 KV heads with per-KV-head-DISTINCT random content
// and a shuffled page table. CPU reference maps head h → kv head
// floor(h / GQA_GROUP); a wrong mapping reads different (or unwritten) rows
// and fails.
function buildAttentionFixture(KV_LEN) {
  const r = rng(9)
  const HD = Q.headDim, PAGE = Q.kvPageStride
  const NUM_PAGES = Math.ceil(KV_LEN / Q.pageSize)
  const smScale = 1 / Math.sqrt(HD)
  // Deterministic shuffle: logical page i → physical page (i-1) mod P.
  const pageOrder = arr(NUM_PAGES, (_, i) => (i + NUM_PAGES - 1) % NUM_PAGES)

  const q = arr(Q.qDim, () => toF16(r() * 2 - 1))
  const pages = new Uint16Array(NUM_PAGES * PAGE)
  const kRef = arr(Q.kvHeads, () => arr(KV_LEN, () => new Array(HD)))
  const vRef = arr(Q.kvHeads, () => arr(KV_LEN, () => new Array(HD)))
  for (let kh = 0; kh < Q.kvHeads; kh++) {
    for (let t = 0; t < KV_LEN; t++) {
      const phys = pageOrder[(t / Q.pageSize) | 0], slot = t % Q.pageSize
      const base = phys * PAGE + kh * Q.headPageStride + slot * HD
      for (let d = 0; d < HD; d++) {
        const kv = toF16(r() * 2 - 1)
        const vv = toF16(r() * 2 - 1)
        kRef[kh][t][d] = kv
        vRef[kh][t][d] = vv
        pages[base + d] = f32ToF16Bits(kv)
        pages[base + Q.vPageOffset + d] = f32ToF16Bits(vv)
      }
    }
  }

  // CPU reference: standard softmax attention, query head h on KV head h/4.
  const ref = []
  for (let h = 0; h < Q.heads; h++) {
    const kh = Math.floor(h / Q.gqaGroup)
    const scores = arr(KV_LEN, (_, t) => {
      let dotv = 0
      for (let d = 0; d < HD; d++) dotv += q[h * HD + d] * kRef[kh][t][d]
      return dotv * smScale
    })
    const m = Math.max(...scores)
    const exps = scores.map((s) => Math.exp(s - m))
    const sum = exps.reduce((a, b) => a + b, 0)
    for (let d = 0; d < HD; d++) {
      let o = 0
      for (let t = 0; t < KV_LEN; t++) o += (exps[t] / sum) * vRef[kh][t][d]
      ref.push(toF16(o))
    }
  }
  return { q, pages, ref, pageOrder, NUM_PAGES, smScale, kRef, vRef }
}

// ── attention[_sg] — GQA head mapping + shuffled page table ──────────────────
function makeAttentionTest(file, entry) {
  return function attentionTest(device) {
    const KV_LEN = 40
    const { q, pages, ref, pageOrder, NUM_PAGES, smScale } = buildAttentionFixture(KV_LEN)

    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      buffer(device, f16Array(q), BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array(pageOrder), BU.STORAGE | BU.COPY_DST),
      buffer(device, pages, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([KV_LEN]), BU.STORAGE | BU.COPY_DST),
      out,
      podBuffer(device, [
        { i32: 1 }, { i32: NUM_PAGES }, { i32: NUM_PAGES },
        { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 },
        { f32: smScale },
        { u32: 1 },
      ]),
    ]
    return runComputeReads(device, pipe, buffers, [1, Q.heads], [
      { index: 5, bytes: Q.qDim * 2 },
    ]).then(([bytes]) => {
      const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
      return { name: `${entry} GQA 32q/8kv`, pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
    })
  }
}

// ── attention_splitk[_sg] + combine — GQA + partitioned KV range ─────────────
function makeAttentionSplitkTest(file, entry, N, KV_LEN) {
  return function attentionSplitkTest(device) {
    const HD = Q.headDim
    const { q, pages, ref, pageOrder, NUM_PAGES, smScale } = buildAttentionFixture(KV_LEN)

    const splitPipe = pipelineFor(device, wgsl(file), entry)
    const combinePipe = pipelineFor(device, wgsl('attention_combine.wgsl'), 'attention_combine')
    const partials = device.createBuffer({
      size: Q.heads * N * (HD + 2) * 4,
      usage: BU.STORAGE | BU.COPY_SRC,
    })
    const out = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const qBuf = buffer(device, f16Array(q), BU.STORAGE | BU.COPY_DST)
    const indptr = buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST)
    const values = buffer(device, new Int32Array(pageOrder), BU.STORAGE | BU.COPY_DST)
    const pagesBuf = buffer(device, pages, BU.STORAGE | BU.COPY_DST)
    const lengthInfo = buffer(device, new Int32Array([KV_LEN]), BU.STORAGE | BU.COPY_DST)
    const splitU = podBuffer(device, [
      { i32: 1 }, { i32: NUM_PAGES }, { i32: NUM_PAGES },
      { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 },
      { f32: smScale },
      { u32: N },
    ])
    const combineU = podBuffer(device, [{ u32: N }])

    const splitBG = device.createBindGroup({
      layout: splitPipe.getBindGroupLayout(0),
      entries: [qBuf, indptr, values, pagesBuf, lengthInfo, partials, splitU]
        .map((b, i) => ({ binding: i, resource: { buffer: b } })),
    })
    const combineBG = device.createBindGroup({
      layout: combinePipe.getBindGroupLayout(0),
      entries: [partials, out, combineU]
        .map((b, i) => ({ binding: i, resource: { buffer: b } })),
    })

    const enc = device.createCommandEncoder()
    const p1 = enc.beginComputePass()
    p1.setPipeline(splitPipe); p1.setBindGroup(0, splitBG)
    p1.dispatchWorkgroups(N, Q.heads, 1); p1.end()
    const p2 = enc.beginComputePass()
    p2.setPipeline(combinePipe); p2.setBindGroup(0, combineBG)
    p2.dispatchWorkgroups(1, Q.heads, 1); p2.end()
    const read = device.createBuffer({ size: Q.qDim * 2, usage: BU.COPY_DST | BU.MAP_READ })
    enc.copyBufferToBuffer(out, 0, read, 0, Q.qDim * 2)
    device.queue.submit([enc.finish()])

    return read.mapAsync(MM.READ).then(() => {
      const got = Array.from(new Uint16Array(read.getMappedRange().slice(0)), f16BitsToF32)
      read.unmap()
      const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
      const name = `${entry} GQA N=${N} kv=${KV_LEN}`
      return { name, pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
    })
  }
}

// ── kv_append — KV_DIM-wide rows into 8-head pages ───────────────────────────
function testKvAppend(device) {
  const r = rng(7)
  const ROW = Q.kvDim, PAGE = Q.kvPageStride, NUM_PAGES = 2, position = 20
  const HD = Q.headDim
  const kData = arr(ROW, () => toF16(r() * 2 - 1))
  const vData = arr(ROW, () => toF16(r() * 2 - 1))
  const pageNo = (position / Q.pageSize) | 0, slot = position % Q.pageSize
  const pipe = pipelineFor(device, wgsl('kv_append.wgsl'), 'kv_append')
  const pages = device.createBuffer({
    size: NUM_PAGES * PAGE * 2,
    usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
  })
  device.queue.writeBuffer(pages, 0, new Uint16Array(NUM_PAGES * PAGE))
  const buffers = [
    buffer(device, f16Array(kData), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(vData), BU.STORAGE | BU.COPY_DST),
    pages,
    buffer(device, new Int32Array([position]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1, NUM_PAGES, 0, 0, ROW / 256]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [ROW / 256], 2, NUM_PAGES * PAGE * 2).then((bytes) => {
    const got = new Uint16Array(bytes)
    const kBits = f16Array(kData), vBits = f16Array(vData)
    let bad = 0
    // The written slot must match…
    for (let head = 0; head < Q.kvHeads; head++) {
      for (let dim = 0; dim < HD; dim++) {
        const within = head * HD + dim
        const kOff = pageNo * PAGE + head * Q.headPageStride + slot * HD + dim
        if (got[kOff] !== kBits[within]) bad++
        if (got[kOff + Q.vPageOffset] !== vBits[within]) bad++
      }
    }
    // …and nothing else may have been written (2·KV_DIM non-zero f16s max).
    let nonZero = 0
    for (const w of got) if (w !== 0) nonZero++
    if (nonZero > 2 * ROW) bad += nonZero - 2 * ROW
    return { name: 'kv_append kvDim=1024', pass: bad === 0, detail: `${bad} mismatched cache slots` }
  })
}

// ── argmax[_sg] — vocab 151936 ───────────────────────────────────────────────
function makeArgmaxTest(file, entry) {
  return function argmaxTest(device) {
    const r = rng(3)
    const V = Q.vocab
    const logits = Float32Array.from(arr(V, () => r() * 20 - 10))
    let refIdx = 0
    for (let i = 1; i < V; i++) if (logits[i] > logits[refIdx]) refIdx = i

    const pipe = pipelineFor(device, wgsl(file), entry)
    const result = device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      buffer(device, logits, BU.STORAGE | BU.COPY_DST),
      result,
      buffer(device, new Uint32Array([V]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runCompute(device, pipe, buffers, [1], 1, 4).then((bytes) => {
      const got = new Int32Array(bytes)[0]
      return { name: `${entry} n=151936`, pass: got === refIdx, detail: `gpu=${got} ref=${refIdx}` }
    })
  }
}

// ── kv_quantize_int8 — headDim=128 (32 threads × 4 dims), 8 KV heads ─────────
function testKvQuantizeInt8(device) {
  const r = rng(12)
  const HD = Q.headDim, POS = 20, NUM_PAGES = 2
  const pageNo = (POS / Q.pageSize) | 0, slot = POS % Q.pageSize
  const kSlot = arr(Q.kvDim, () => toF16(r() * 2 - 1))
  const vSlot = arr(Q.kvDim, () => toF16(r() * 2 - 1))

  const pipe = pipelineFor(device, wgsl('kv_quantize_int8.wgsl'), 'kv_quantize_int8')
  const pagesI8 = device.createBuffer({
    size: NUM_PAGES * Q.kvI8PageWords * 4,
    usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
  })
  device.queue.writeBuffer(pagesI8, 0, new Uint32Array(NUM_PAGES * Q.kvI8PageWords))
  const scalesBuf = device.createBuffer({
    size: NUM_PAGES * Q.kvScalesPerPage * 2,
    usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
  })
  device.queue.writeBuffer(scalesBuf, 0, new Uint16Array(NUM_PAGES * Q.kvScalesPerPage))
  const WGS = Q.kvHeads * 2
  const buffers = [
    buffer(device, f16Array(kSlot), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(vSlot), BU.STORAGE | BU.COPY_DST),
    pagesI8,
    scalesBuf,
    buffer(device, new Int32Array([POS]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([0, 0, 0, WGS]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runComputeReads(device, pipe, buffers, [WGS], [
    { index: 2, bytes: NUM_PAGES * Q.kvI8PageWords * 4 },
    { index: 3, bytes: NUM_PAGES * Q.kvScalesPerPage * 2 },
  ]).then(([pageBytes, scaleBytes]) => {
    const gotWords = new Uint32Array(pageBytes)
    const gotScales = Array.from(new Uint16Array(scaleBytes), f16BitsToF32)
    let scaleBad = 0
    let maxIntDiff = 0
    let maxDequantErr = 0
    for (let head = 0; head < Q.kvHeads; head++) {
      for (let side = 0; side < 2; side++) {
        const src = side === 0 ? kSlot : vSlot
        let maxAbs = 0
        for (let d = 0; d < HD; d++) maxAbs = Math.max(maxAbs, Math.abs(src[head * HD + d]))
        let scale = maxAbs / 127
        if (scale < 1e-8) scale = 1e-8
        const scaleIdx = pageNo * Q.kvScalesPerPage + head * Q.kvScalesPerHead
                       + slot * Q.kvScalesPerSlot + side
        // Allow a couple f16 ulps: the GPU divides in f32, the ref in f64.
        if (Math.abs(gotScales[scaleIdx] - toF16(scale)) > Math.abs(scale) * 2e-3) scaleBad++
        const gotScale = gotScales[scaleIdx]
        for (let qw = 0; qw < Q.kvI8RowWords; qw++) {
          const word = gotWords[pageNo * Q.kvI8PageWords + head * Q.kvI8HeadWords
                              + slot * Q.kvI8SlotWords + side * Q.kvI8RowWords + qw]
          for (let b = 0; b < 4; b++) {
            const raw = (word >>> (b * 8)) & 0xff
            const gotInt = raw > 127 ? raw - 256 : raw
            const orig = src[head * HD + qw * 4 + b]
            const refInt = Math.max(-127, Math.min(127, Math.round(orig / scale)))
            maxIntDiff = Math.max(maxIntDiff, Math.abs(gotInt - refInt))
            maxDequantErr = Math.max(maxDequantErr, Math.abs(gotInt * gotScale - orig))
          }
        }
      }
    }
    // ±1 int tolerance absorbs round-half-to-even vs JS round-half-up; the
    // dequant bound is the real accuracy contract (≤ ~0.75·scale ≈ 6e-3).
    const pass = scaleBad === 0 && maxIntDiff <= 1 && maxDequantErr < 8e-3
    return {
      name: 'kv_quantize_int8 hd=128',
      pass,
      detail: `int diff ≤ ${maxIntDiff}, dequant err ${maxDequantErr.toExponential(2)}, ${scaleBad} bad scales`,
    }
  })
}

// ── attention_int8 — GQA + int8 pages/scales at headDim 128 ──────────────────
function testAttentionInt8(device) {
  const r = rng(13)
  const HD = Q.headDim, KV_LEN = 40
  const NUM_PAGES = Math.ceil(KV_LEN / Q.pageSize)
  const smScale = 1 / Math.sqrt(HD)
  const pageOrder = arr(NUM_PAGES, (_, i) => (i + NUM_PAGES - 1) % NUM_PAGES)

  const q = arr(Q.qDim, () => toF16(r() * 2 - 1))
  const words = new Uint32Array(NUM_PAGES * Q.kvI8PageWords)
  const scales = new Uint16Array(NUM_PAGES * Q.kvScalesPerPage)
  // kInt[kh][t][d], vInt[kh][t][d] + per-(kh,t,side) scales — DISTINCT per KV
  // head so a wrong h→kv_head mapping fails.
  const kInt = arr(Q.kvHeads, () => arr(KV_LEN, () => new Array(HD)))
  const vInt = arr(Q.kvHeads, () => arr(KV_LEN, () => new Array(HD)))
  const kScale = arr(Q.kvHeads, () => new Array(KV_LEN))
  const vScale = arr(Q.kvHeads, () => new Array(KV_LEN))
  for (let kh = 0; kh < Q.kvHeads; kh++) {
    for (let t = 0; t < KV_LEN; t++) {
      const phys = pageOrder[(t / Q.pageSize) | 0], slot = t % Q.pageSize
      kScale[kh][t] = toF16(r() * 0.01 + 0.005)
      vScale[kh][t] = toF16(r() * 0.01 + 0.005)
      const scaleBase = phys * Q.kvScalesPerPage + kh * Q.kvScalesPerHead + slot * Q.kvScalesPerSlot
      scales[scaleBase] = f32ToF16Bits(kScale[kh][t])
      scales[scaleBase + 1] = f32ToF16Bits(vScale[kh][t])
      const wordBase = phys * Q.kvI8PageWords + kh * Q.kvI8HeadWords + slot * Q.kvI8SlotWords
      for (let qw = 0; qw < Q.kvI8RowWords; qw++) {
        let kw = 0, vw = 0
        for (let b = 0; b < 4; b++) {
          const ki = ((r() * 255) | 0) - 127   // [-127, 127]
          const vi = ((r() * 255) | 0) - 127
          kInt[kh][t][qw * 4 + b] = ki
          vInt[kh][t][qw * 4 + b] = vi
          kw |= (ki & 0xff) << (b * 8)
          vw |= (vi & 0xff) << (b * 8)
        }
        words[wordBase + qw] = kw >>> 0
        words[wordBase + Q.kvI8RowWords + qw] = vw >>> 0
      }
    }
  }

  // CPU reference: dequant + standard softmax, query head h on KV head h/4.
  const ref = []
  for (let h = 0; h < Q.heads; h++) {
    const kh = Math.floor(h / Q.gqaGroup)
    const scores = arr(KV_LEN, (_, t) => {
      let dotv = 0
      for (let d = 0; d < HD; d++) dotv += q[h * HD + d] * kInt[kh][t][d]
      return dotv * kScale[kh][t] * smScale
    })
    const m = Math.max(...scores)
    const exps = scores.map((s) => Math.exp(s - m))
    const sum = exps.reduce((a, b) => a + b, 0)
    for (let d = 0; d < HD; d++) {
      let o = 0
      for (let t = 0; t < KV_LEN; t++) o += (exps[t] / sum) * vInt[kh][t][d] * vScale[kh][t]
      ref.push(toF16(o))
    }
  }

  const pipe = pipelineFor(device, wgsl('attention_int8.wgsl'), 'attention_int8')
  const out = device.createBuffer({ size: Q.qDim * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    buffer(device, f16Array(q), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array(pageOrder), BU.STORAGE | BU.COPY_DST),
    buffer(device, words, BU.STORAGE | BU.COPY_DST),
    buffer(device, scales, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array([KV_LEN]), BU.STORAGE | BU.COPY_DST),
    out,
    podBuffer(device, [
      { i32: 1 }, { i32: NUM_PAGES }, { i32: NUM_PAGES },
      { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 },  // elem offsets (incl. scales)
      { f32: smScale },
      { u32: 1 },
    ]),
  ]
  return runComputeReads(device, pipe, buffers, [1, Q.heads], [
    { index: 6, bytes: Q.qDim * 2 },
  ]).then(([bytes]) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
    return { name: 'attention_int8 GQA hd=128', pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
  })
}

// ── LM head — tied-embedding matmul: N=151936 rows, K=2560 ───────────────────
// The full quantized embed matrix (151936 × 320 u32 ≈ 195MB — matches the MLC
// ndarray-cache record [151936, 320]) is bound and every row computed on the
// GPU; the CPU reference spot-checks rows across the range. The >65535-row
// grid is folded into dispatch z exactly like engine-core's lmHead dispatch.
function testLmHead(device) {
  const V = Q.vocab, K = Q.d, KP = Q.dPacked, SPR = Q.dScales
  const weightsBytes = V * KP * 4
  if (device.limits.maxStorageBufferBindingSize < weightsBytes) {
    return Promise.resolve({
      name: 'lm_head N=151936',
      pass: true,
      skip: `adapter grants maxStorageBufferBindingSize ${device.limits.maxStorageBufferBindingSize} < ${weightsBytes}`,
    })
  }
  const r = rng(14)
  const input = arr(K, () => toF16(r() * 2 - 1))
  // Typed-array generation (145M elements total) — keep it flat and fast.
  const weights = new Uint32Array(V * KP)
  for (let i = 0; i < weights.length; i++) weights[i] = (r() * 0xffffffff) >>> 0
  const scales16 = new Uint16Array(V * SPR)
  const scales = new Float32Array(V * SPR)
  for (let i = 0; i < scales.length; i++) {
    const s = toF16(r() * 0.05 + 0.01)
    scales[i] = s
    scales16[i] = f32ToF16Bits(s)
  }
  const dot = makeDot(weights, scales, input, KP, SPR)
  const rows = [0, 1, 2, 75967, 151933, 151934, 151935]

  const pipe = pipelineFor(device, withPrelude(int4MatmulWGSL({ outF32: true }), Q), 'int4_matmul_f32')
  const out = device.createBuffer({ size: V * 4, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
    buffer(device, scales16, BU.STORAGE | BU.COPY_DST),
    buffer(device, weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([KP, SPR, V, 0]), BU.UNIFORM | BU.COPY_DST),
  ]
  // 151936 rows > 65535 per grid dimension → fold into z (kernels index
  // blockIdx.z * gridDim.x + blockIdx.x and guard on packGridDimX).
  const X = 16384
  return runCompute(device, pipe, buffers, [X, 1, Math.ceil(V / X)], 0, V * 4).then((bytes) => {
    const got = new Float32Array(bytes)
    const maxRel = Math.max(
      ...rows.map((row) => {
        const ref = dot(row)
        return Math.abs(got[row] - ref) / (Math.abs(ref) + 1e-3)
      }),
    )
    return { name: 'lm_head N=151936 K=2560', pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
  })
}

// ── test roster ──────────────────────────────────────────────────────────────
const TESTS = [
  { label: 'vec4_k_gating', fn: () => testVec4Gating() },
  { label: 'int4_matmul', fn: makeInt4MatmulTest(int4MatmulWGSL(), 'int4_matmul') },
  { label: 'rms_norm', fn: testRmsNorm },
  { label: 'embedding', fn: testEmbedding },
  { label: 'rope', fn: testRope },
  { label: 'qk_norm', fn: testQkNorm },
  { label: 'qkv_fused', fn: makeQkvFusedTest('qkv_fused.wgsl', 'qkv_fused') },
  { label: 'attention', fn: makeAttentionTest('attention.wgsl', 'attention') },
  { label: 'attn_splitk2', fn: makeAttentionSplitkTest('attention_splitk.wgsl', 'attention_splitk', 2, 40) },
  { label: 'attn_splitk4', fn: makeAttentionSplitkTest('attention_splitk.wgsl', 'attention_splitk', 4, 40) },
  { label: 'kv_append', fn: testKvAppend },
  { label: 'argmax', fn: makeArgmaxTest('argmax.wgsl', 'argmax_kernel') },
  { label: 'kv_quantize_int8', fn: testKvQuantizeInt8 },
  { label: 'attention_int8', fn: testAttentionInt8 },
  { label: 'lm_head', fn: testLmHead },
  // Subgroup variants (need the feature + 32-lane subgroups, like run.mjs).
  { label: 'int4_matmul_sg', fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true }), 'int4_matmul_sg'), minSg: 32 },
  { label: 'qkv_fused_sg', fn: makeQkvFusedTest('qkv_fused_sg.wgsl', 'qkv_fused_sg'), minSg: 32 },
  { label: 'attention_sg', fn: makeAttentionTest('attention_sg.wgsl', 'attention_sg'), minSg: 32 },
  { label: 'attn_splitk_sg', fn: makeAttentionSplitkTest('attention_splitk_sg.wgsl', 'attention_splitk_sg', 4, 40), minSg: 32 },
  { label: 'argmax_sg', fn: makeArgmaxTest('argmax_sg.wgsl', 'argmax_sg'), minSg: 16 },
]

// ── compile-all gate (phase A, kept verbatim in spirit) ──────────────────────
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
    detail: `${compiled} shaders compile under QWEN3_4B (${sources.length - INT4_MATMUL_VARIANTS.length} .wgsl + ${INT4_MATMUL_VARIANTS.length} generated${skipNote})`,
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
  let skipped = 0
  for (const t of TESTS) {
    const name = t.label
    if (t.minSg) {
      if (!subgroups) {
        console.log(`SKIP  ${name.padEnd(18)} adapter lacks the 'subgroups' WebGPU feature`)
        skipped++
        continue
      }
      if (sgSize < t.minSg) {
        console.log(
          `SKIP  ${name.padEnd(18)} needs subgroup size >= ${t.minSg}, adapter reports ${sgSize || 'unknown'}`,
        )
        skipped++
        continue
      }
    }
    let res
    try {
      res = await t.fn(device)
    } catch (e) {
      res = { pass: false, detail: String(e).split('\n')[0] }
    }
    if (res.skip) {
      console.log(`SKIP  ${name.padEnd(18)} ${res.skip}`)
      skipped++
      continue
    }
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${name.padEnd(18)} ${res.detail}`)
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

  const skipNote = skipped ? ` (${skipped} skipped — see SKIP lines above)` : ''
  console.log(`\n${ran - failed}/${ran} Qwen3 kernels correct${skipNote}`)
  process.exit(failed ? 1 : 0)
}

main()
