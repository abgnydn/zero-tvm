// Headless kernel-correctness suite.
//
// Loads the engine's real WGSL kernels from src/compiler/shaders/, runs each
// on the local WebGPU adapter (a real GPU on a dev machine, lavapipe in CI),
// and checks the GPU output against a plain-JS reference. Deterministic
// (seeded RNG) so CI is reproducible. Exits non-zero on any mismatch.
//
//   npm run test:kernels
//
// Covers the scalar kernels AND the shipped `_sg` subgroup variants. The
// subgroup tests need the WebGPU 'subgroups' feature and (for most kernels)
// a 32-lane subgroup; when the adapter can't provide that (lavapipe in CI
// often can't) they SKIP loudly with the reason instead of passing silently.

import { readFileSync } from 'node:fs'
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
} from './gpu.mjs'
import { toF16, f16Array, f16BitsToF32, f32ToF16Bits } from './half.mjs'

const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')
const wgsl = (name) => readFileSync(resolve(SHADERS, name), 'utf8')

// Small deterministic PRNG (mulberry32) so runs are reproducible.
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

// Pack a PODArgs-style uniform: ints then optional trailing f32/u32 fields.
// `fields` is [{ i32: n } | { f32: x } | { u32: n }, ...]; buffer is padded
// to 16 bytes.
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

// ── int4_matmul[_sg].wgsl : out[r] = dot(input, dequant(weights[r])) ─────────
function makeInt4MatmulTest(file, entry) {
  return function int4MatmulTest(device) {
    const r = rng(1)
    const K = 512, M = 8, KP = K / 8, SPR = K / 32 // K must be a multiple of 512
    const input = arr(K, () => toF16(r() * 2 - 1))
    const scales = arr(M * SPR, () => toF16(r() * 0.05 + 0.01))
    const weights = Uint32Array.from(arr(M * KP, () => (r() * 0xffffffff) >>> 0))

    const ref = arr(M, (_, row) => {
      let acc = 0
      for (let w = 0; w < KP; w++) {
        const packed = weights[row * KP + w]
        const scale = scales[row * SPR + (w >> 2)]
        for (let n = 0; n < 8; n++) {
          const nib = (packed >>> (4 * n)) & 15
          acc += input[w * 8 + n] * (nib - 7) * scale
        }
      }
      return toF16(acc)
    })

    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: M * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      out,
      buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([KP, SPR, M, 0]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runCompute(device, pipe, buffers, [M], 0, M * 2).then((bytes) => {
      const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / (Math.abs(ref[i]) + 1e-3)))
      return { name: entry, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── rms_norm.wgsl : out[i] = input[i] / sqrt(mean(input^2)+1e-5) * gamma[i] ──
function testRmsNorm(device) {
  const r = rng(2)
  const D = 3072 // hardcoded in the kernel
  const input = arr(D, () => toF16(r() * 2 - 1))
  const gamma = arr(D, () => toF16(r() * 0.5 + 0.75))

  let ss = 0
  for (const v of input) ss += v * v
  const rinv = 1 / Math.sqrt(ss / D + 1e-5)
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
    return { name: 'rms_norm', pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
  })
}

// ── argmax[_sg].wgsl : index of the (first) maximum logit ────────────────────
function makeArgmaxTest(file, entry) {
  return function argmaxTest(device) {
    const r = rng(3)
    const V = 32064 // Phi-3 vocab
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
      return { name: entry, pass: got === refIdx, detail: `gpu=${got} ref=${refIdx}` }
    })
  }
}

// ── embedding.wgsl : out[i] = dequant(embd_weight[token_id, i]) ──────────────
function testEmbedding(device) {
  const r = rng(4)
  const D = 3072, VOCAB = 64, DP = D / 8, SPR = D / 32, token = 17
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
    buffer(device, new Uint32Array([1, D / 256]), BU.UNIFORM | BU.COPY_DST), // seq_len, packGridDimX
  ]
  return runCompute(device, pipe, buffers, [D / 256], 0, D * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const bad = got.reduce((m, g, i) => Math.max(m, Math.abs(g - ref[i])), 0)
    return { name: 'embedding', pass: bad === 0, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── rope.wgsl : RoPE on Q/K heads, copy V ────────────────────────────────────
function testRope(device) {
  const r = rng(5)
  const QKV = 9216, HD = 96, pos = 7
  const qkv = arr(QKV, () => toF16(r() * 2 - 1))
  const rope = (val, pair, dim) => {
    const freq = pos / Math.pow(10000, ((dim % 48) * 2) / 96)
    return toF16(Math.cos(freq) * val + Math.sin(freq) * pair)
  }
  const q = new Array(3072), k = new Array(3072), v = new Array(3072)
  for (let head = 0; head < 96; head++) {
    for (let dim = 0; dim < HD; dim++) {
      const val = qkv[head * HD + dim]
      if (head < 64) {
        const pair = dim < 48 ? -qkv[head * HD + dim + 48] : qkv[head * HD + dim - 48]
        const out = rope(val, pair, dim)
        if (head < 32) q[head * HD + dim] = out
        else k[(head - 32) * HD + dim] = out
      } else {
        v[(head - 64) * HD + dim] = val
      }
    }
  }
  const pipe = pipelineFor(device, wgsl('rope.wgsl'), 'rope_kernel')
  const mk = () => device.createBuffer({ size: 3072 * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    mk(), mk(), mk(),
    buffer(device, f16Array(qkv), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Int32Array([pos]), BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([1, 0, 1, QKV / 256]), BU.UNIFORM | BU.COPY_DST), // apply_rope, offset, seq_len, packGridDimX
  ]
  const reads = [0, 1, 2].map((index) => ({ index, bytes: 3072 * 2 }))
  return runComputeReads(device, pipe, buffers, [QKV / 256], reads).then(([qb, kb, vb]) => {
    const dec = (b) => Array.from(new Uint16Array(b), f16BitsToF32)
    const [gq, gk, gv] = [dec(qb), dec(kb), dec(vb)]
    let bad = 0
    for (let i = 0; i < 3072; i++) {
      bad = Math.max(bad, Math.abs(gq[i] - q[i]), Math.abs(gk[i] - k[i]), Math.abs(gv[i] - v[i]))
    }
    return { name: 'rope', pass: bad < 5e-3, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── add_norm.wgsl : residual = A+B ; out = rmsnorm(residual) * gamma ─────────
function testAddNorm(device) {
  const r = rng(6)
  const D = 3072
  const A = arr(D, () => toF16(r() * 2 - 1))
  const B = arr(D, () => toF16(r() * 2 - 1))
  const gamma = arr(D, () => toF16(r() * 0.5 + 0.75))
  const resid = A.map((a, i) => toF16(a + B[i]))
  let ss = 0
  for (const x of resid) ss += x * x
  const rinv = 1 / Math.sqrt(ss / D + 1e-5)
  const ref = resid.map((x, i) => toF16(x * rinv * gamma[i]))
  const pipe = pipelineFor(device, wgsl('add_norm.wgsl'), 'add_norm')
  const out = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const residBuf = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    buffer(device, f16Array(A), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(B), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST),
    out, residBuf,
    buffer(device, new Uint32Array([1]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [1], 3, D * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const bad = got.reduce((m, g, i) => Math.max(m, Math.abs(g - ref[i])), 0)
    return { name: 'add_norm', pass: bad < 5e-3, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── kv_append.wgsl : write K,V into the paged cache at `position` ────────────
function testKvAppend(device) {
  const r = rng(7)
  const ROW = 3072, PAGE = 98304, NUM_PAGES = 2, position = 20
  const kData = arr(ROW, () => toF16(r() * 2 - 1))
  const vData = arr(ROW, () => toF16(r() * 2 - 1))
  const pageNo = (position / 16) | 0, slot = position % 16
  const pipe = pipelineFor(device, wgsl('kv_append.wgsl'), 'kv_append')
  const pages = device.createBuffer({
    size: NUM_PAGES * PAGE * 2,
    usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
  })
  device.queue.writeBuffer(pages, 0, new Uint16Array(NUM_PAGES * PAGE)) // zero-fill
  const buffers = [
    buffer(device, f16Array(kData), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(vData), BU.STORAGE | BU.COPY_DST),
    pages,
    buffer(device, new Int32Array([position]), BU.STORAGE | BU.COPY_DST),
    // ntoken, num_pages, pages_offset, position_map_offset, packGridDimX
    buffer(device, new Uint32Array([1, NUM_PAGES, 0, 0, ROW / 256]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [ROW / 256], 2, NUM_PAGES * PAGE * 2).then((bytes) => {
    const got = new Uint16Array(bytes)
    const kBits = f16Array(kData), vBits = f16Array(vData)
    let bad = 0
    for (let head = 0; head < 32; head++) {
      for (let dim = 0; dim < 96; dim++) {
        const within = head * 96 + dim
        const kOff = pageNo * PAGE + head * 1536 + slot * 96 + dim
        if (got[kOff] !== kBits[within]) bad++
        if (got[kOff + 49152] !== vBits[within]) bad++
      }
    }
    return { name: 'kv_append', pass: bad === 0, detail: `${bad} mismatched cache slots` }
  })
}

// ── fused_ffn[.wgsl|_tiled_sg.wgsl] : out[i] = SiLU(gate·x) * (up·x) ─────────
// Both variants accumulate the dot products in f32 (the scalar kernel was
// converted from f16 accumulation), so the reference is a plain f32 loop.
function makeFusedFfnTest(file, entry, wgPerDispatch) {
  return function fusedFfnTest(device) {
    const r = rng(8)
    const D = 3072, DP = D / 8, SPR = D / 32, ROWS = 8
    const maxRow = ROWS - 1 + 8192
    const input = arr(D, () => toF16(r() * 1.5 - 0.75))
    const scales = arr((maxRow + 1) * SPR, () => toF16(r() * 0.05 + 0.01))
    const weights = Uint32Array.from(arr((maxRow + 1) * DP, () => (r() * 0xffffffff) >>> 0))
    const dot = (row) => {
      let acc = 0
      for (let w = 0; w < DP; w++) {
        const packed = weights[row * DP + w]
        const scale = scales[row * SPR + (w >> 2)]
        for (let n = 0; n < 8; n++) {
          const nib = (packed >>> (4 * n)) & 15
          acc += input[w * 8 + n] * (nib - 7) * scale
        }
      }
      return acc
    }
    const ref = arr(ROWS, (_, i) => {
      const g = dot(i), u = dot(i + 8192)
      const silu = g * (1 / (1 + Math.exp(-g)))
      return toF16(u * silu)
    })
    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: ROWS * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      out,
      buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([ROWS]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runCompute(device, pipe, buffers, [wgPerDispatch(ROWS)], 0, ROWS * 2).then((bytes) => {
      const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / (Math.abs(ref[i]) + 1e-3)))
      return { name: entry, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── attention[_sg].wgsl : paged-KV decode attention (FlashDecoding) ──────────
// Seeds a 3-page KV cache with a deliberately shuffled page table, runs 2
// heads of single-token decode, and compares against a standard (non-online)
// CPU softmax-attention reference.
function makeAttentionTest(file, entry) {
  return function attentionTest(device) {
    const r = rng(9)
    const HEADS = 2, HD = 96, PAGE = 98304, NUM_PAGES = 3
    const KV_LEN = 40 // 3 pages: 16 + 16 + 8 slots
    const smScale = 1 / Math.sqrt(HD)
    const pageOrder = [2, 0, 1] // logical page → physical page (exercises the table)

    const q = arr(32 * HD, () => toF16(r() * 2 - 1)) // full Q buffer; heads 0..1 checked
    const pages = new Uint16Array(NUM_PAGES * PAGE)
    // kRef[h][t][d], vRef[h][t][d]
    const kRef = arr(HEADS, () => arr(KV_LEN, () => new Array(HD)))
    const vRef = arr(HEADS, () => arr(KV_LEN, () => new Array(HD)))
    for (let h = 0; h < HEADS; h++) {
      for (let t = 0; t < KV_LEN; t++) {
        const phys = pageOrder[t >> 4], slot = t & 15
        const base = phys * PAGE + h * 1536 + slot * 96
        for (let d = 0; d < HD; d++) {
          const kv = toF16(r() * 2 - 1)
          const vv = toF16(r() * 2 - 1)
          kRef[h][t][d] = kv
          vRef[h][t][d] = vv
          pages[base + d] = f32ToF16Bits(kv)
          pages[base + 49152 + d] = f32ToF16Bits(vv)
        }
      }
    }

    // CPU reference: standard softmax attention per head.
    const ref = []
    for (let h = 0; h < HEADS; h++) {
      const scores = arr(KV_LEN, (_, t) => {
        let dot = 0
        for (let d = 0; d < HD; d++) dot += q[h * HD + d] * kRef[h][t][d]
        return dot * smScale
      })
      const m = Math.max(...scores)
      const exps = scores.map((s) => Math.exp(s - m))
      const sum = exps.reduce((a, b) => a + b, 0)
      for (let d = 0; d < HD; d++) {
        let o = 0
        for (let t = 0; t < KV_LEN; t++) o += (exps[t] / sum) * vRef[h][t][d]
        ref.push(toF16(o))
      }
    }

    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: 3072 * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      buffer(device, f16Array(q), BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST), // indptr
      buffer(device, new Int32Array(pageOrder), BU.STORAGE | BU.COPY_DST),      // values
      buffer(device, pages, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([KV_LEN]), BU.STORAGE | BU.COPY_DST),       // length_info
      out,
      podBuffer(device, [
        { i32: 1 },            // B
        { i32: NUM_PAGES },    // max_num_pages
        { i32: NUM_PAGES },    // nnz_pages
        { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 }, // elem offsets
        { f32: smScale },
        { u32: 1 },            // packGridDimX
      ]),
    ]
    // One workgroup per (batch, head): dispatch (1, HEADS).
    return runComputeReads(device, pipe, buffers, [1, HEADS], [
      { index: 5, bytes: HEADS * HD * 2 },
    ]).then(([bytes]) => {
      const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
      return { name: entry, pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
    })
  }
}

// ── qkv_fused[_sg].wgsl : QKV int4 matmul + RoPE + paged KV append ──────────
// Full 9216-row projection on the GPU; the CPU reference spot-checks all
// three groups (Q / K / V) on a subset of heads, including the RoPE pair
// rotation and the paged-cache write position.
function makeQkvFusedTest(file, entry) {
  return function qkvFusedTest(device) {
    const r = rng(10)
    const D = 3072, ROWS = 9216, KP = 384, SPR = 96
    const POS = 20, NUM_PAGES = 2, PAGE = 98304
    const pageNo = (POS / 16) | 0, slot = POS % 16
    const PAIRS = 4608

    const hidden = new Float32Array(D)
    for (let i = 0; i < D; i++) hidden[i] = toF16(r() * 2 - 1)
    const scales = new Float32Array(ROWS * SPR)
    for (let i = 0; i < scales.length; i++) scales[i] = toF16(r() * 0.05 + 0.01)
    const weights = new Uint32Array(ROWS * KP)
    for (let i = 0; i < weights.length; i++) weights[i] = (r() * 0xffffffff) >>> 0

    const dot = (row) => {
      let acc = 0
      for (let w = 0; w < KP; w++) {
        const packed = weights[row * KP + w]
        const scale = scales[row * SPR + (w >> 2)]
        const base = w * 8
        for (let n = 0; n < 8; n++) acc += hidden[base + n] * (((packed >>> (4 * n)) & 15) - 7) * scale
      }
      return acc
    }

    const pipe = pipelineFor(device, wgsl(file), entry)
    const qOut = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const kvPages = device.createBuffer({
      size: NUM_PAGES * PAGE * 2,
      usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
    })
    device.queue.writeBuffer(kvPages, 0, new Uint16Array(NUM_PAGES * PAGE)) // zero-fill
    const buffers = [
      qOut,
      kvPages,
      buffer(device, f16Array(Array.from(hidden)), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(Array.from(scales)), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Int32Array([POS]), BU.STORAGE | BU.COPY_DST),
      // position_map_elem_offset, pages_elem_offset, packGridDimX
      buffer(device, new Uint32Array([0, 0, PAIRS, 0]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runComputeReads(device, pipe, buffers, [PAIRS], [
      { index: 0, bytes: D * 2 },
      { index: 1, bytes: NUM_PAGES * PAGE * 2 },
    ]).then(([qBytes, pageBytes]) => {
      const gotQ = Array.from(new Uint16Array(qBytes), f16BitsToF32)
      const gotPages = Array.from(new Uint16Array(pageBytes), f16BitsToF32)
      let maxRel = 0
      const check = (got, want) => {
        maxRel = Math.max(maxRel, Math.abs(got - want) / (Math.abs(want) + 1e-3))
      }
      for (const head of [0, 13, 31]) {
        for (let dimLo = 0; dimLo < 48; dimLo++) {
          const freq = POS / Math.pow(10000, (dimLo * 2) / 96)
          const c = Math.cos(freq), s = Math.sin(freq)
          for (let group = 0; group < 3; group++) {
            const rowLo = group * 3072 + head * 96 + dimLo
            const vLo = dot(rowLo), vHi = dot(rowLo + 48)
            if (group === 2) {
              const vBase = pageNo * PAGE + head * 1536 + slot * 96 + 49152 + dimLo
              check(gotPages[vBase], toF16(vLo))
              check(gotPages[vBase + 48], toF16(vHi))
            } else {
              const rotLo = toF16(c * vLo - s * vHi)
              const rotHi = toF16(c * vHi + s * vLo)
              if (group === 0) {
                check(gotQ[head * 96 + dimLo], rotLo)
                check(gotQ[head * 96 + dimLo + 48], rotHi)
              } else {
                const kBase = pageNo * PAGE + head * 1536 + slot * 96 + dimLo
                check(gotPages[kBase], rotLo)
                check(gotPages[kBase + 48], rotHi)
              }
            }
          }
        }
      }
      return { name: entry, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── test roster ──────────────────────────────────────────────────────────────
// Baseline tests run on any f16-capable adapter. Subgroup (`_sg`) variants
// additionally require the 'subgroups' feature and a minimum subgroup size
// (most kernels lay out their cross-lane reduction for 32-lane subgroups —
// same gate chat.ts applies before shipping them to a real GPU).
const TESTS = [
  { label: 'int4_matmul', fn: makeInt4MatmulTest('int4_matmul.wgsl', 'int4_matmul') },
  { label: 'rms_norm', fn: testRmsNorm },
  { label: 'argmax', fn: makeArgmaxTest('argmax.wgsl', 'argmax_kernel') },
  { label: 'embedding', fn: testEmbedding },
  { label: 'rope', fn: testRope },
  { label: 'add_norm', fn: testAddNorm },
  { label: 'kv_append', fn: testKvAppend },
  { label: 'fused_ffn', fn: makeFusedFfnTest('fused_ffn.wgsl', 'fused_ffn_kernel', (rows) => rows) },
  { label: 'attention', fn: makeAttentionTest('attention.wgsl', 'attention') },
  { label: 'qkv_fused', fn: makeQkvFusedTest('qkv_fused.wgsl', 'qkv_fused') },
  { label: 'int4_matmul_sg', fn: makeInt4MatmulTest('int4_matmul_sg.wgsl', 'int4_matmul_sg'), minSg: 32 },
  { label: 'argmax_sg', fn: makeArgmaxTest('argmax_sg.wgsl', 'argmax_sg'), minSg: 16 },
  { label: 'attention_sg', fn: makeAttentionTest('attention_sg.wgsl', 'attention_sg'), minSg: 32 },
  { label: 'qkv_fused_sg', fn: makeQkvFusedTest('qkv_fused_sg.wgsl', 'qkv_fused_sg'), minSg: 32 },
  {
    label: 'fused_ffn_tiled_sg',
    fn: makeFusedFfnTest('fused_ffn_tiled_sg.wgsl', 'fused_ffn_tiled_sg', (rows) => Math.ceil(rows / 4)),
    minSg: 32,
  },
]

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
    `adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16} | subgroups: ${sgLabel}`,
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
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${name.padEnd(18)} ${res.detail}`)
    ran++
    if (!res.pass) failed++
  }
  const skipNote = skipped ? ` (${skipped} skipped — see SKIP lines above)` : ''
  console.log(`\n${ran - failed}/${ran} kernels correct${skipNote}`)
  process.exit(failed ? 1 : 0)
}

main()
