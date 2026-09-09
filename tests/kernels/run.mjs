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
// Node ≥23 strips TS types on import, so the tests share the exact prelude
// and int4 generator the app compiles with — no duplicated logic.
import { withPrelude, PHI3, ropeInvFreqTable } from '../../src/compiler/shader-prelude.ts'
import {
  int4MatmulWGSL,
  int4MatmulEntry,
  INT4_MATMUL_VARIANTS,
} from '../../src/compiler/shaders/int4_matmul.gen.ts'

const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')
// Same render step as every app-side createShaderModule site: inject the
// model-shape prelude after any `enable` directives.
const wgsl = (name) => withPrelude(readFileSync(resolve(SHADERS, name), 'utf8'))

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

// ── int4_matmul[_sg|_tiled][_f32][_vec4] (generated) ─────────────────────────
// out[r] = dot(input, dequant(weights[r])). Options:
//   K         — reduction dim; scalar/sg variants need K % 512 == 0, the
//               _vec4 variants need K % 1024 == 0 (32 K-elements per thread
//               per iteration)
//   rowsPerWG — output rows per workgroup (dispatch M / rowsPerWG)
//   outF32    — read the output as f32 (LM-head variants)
function makeInt4MatmulTest(src, entry, { K = 512, rowsPerWG = 1, outF32 = false, affine = false, q3 = false } = {}) {
  return function int4MatmulTest(device) {
    const r = rng(1)
    // affine (MLX): one scale+bias per 64 values; symmetric (MLC): one scale per 32.
    // q3: 3-bit values as a continuous LSB-first bitstream, 3K/32 words per row.
    const M = 8, KP = q3 ? (K * 3) / 32 : K / 8, SPR = K / (affine ? 64 : 32)
    const input = arr(K, () => toF16(r() * 2 - 1))
    const scales = arr(M * SPR, () => toF16(r() * 0.05 + 0.01))
    const biases = affine ? arr(M * SPR, () => toF16(r() * 0.1 - 0.05)) : null
    let weights, q
    if (q3) {
      // Pack the row's values bit-serially so the STREAM layout itself is what
      // the kernel is tested against (values straddle word boundaries).
      q = arr(M * K, () => Math.floor(r() * 8))
      weights = new Uint32Array(M * KP)
      for (let row = 0; row < M; row++) {
        for (let i = 0; i < K; i++) {
          const v = q[row * K + i]
          const bit = 3 * i
          const w = row * KP + (bit >> 5)
          const sh = bit & 31
          weights[w] |= (v << sh) >>> 0
          if (sh > 29) weights[w + 1] |= v >>> (32 - sh)
        }
      }
    } else {
      weights = Uint32Array.from(arr(M * KP, () => (r() * 0xffffffff) >>> 0))
    }

    // Mirrors the kernel exactly: per weight word, one (scale, bias) pair; the
    // bias multiplies only THAT word's 8 activations, so summing the words lands
    // each group's bias term once (see the `affine` note in int4_matmul.gen.ts).
    const ref = arr(M, (_, row) => {
      let acc = 0
      if (q3) {
        // Per value: s_g·x·q + b_g·x, groups of 64 — same algebra as affine;
        // the packing difference lives entirely in the weights buffer above.
        for (let i = 0; i < K; i++) {
          const g = i >> 6
          acc += scales[row * SPR + g] * input[i] * q[row * K + i] + biases[row * SPR + g] * input[i]
        }
        return outF32 ? acc : toF16(acc)
      }
      for (let w = 0; w < K / 8; w++) {
        const packed = weights[row * KP + w]
        const g = affine ? w >> 3 : w >> 2
        const scale = scales[row * SPR + g]
        let dot = 0, xs = 0
        for (let n = 0; n < 8; n++) {
          const nib = (packed >>> (4 * n)) & 15
          // affine reads the nibble raw (w = s·q + b); symmetric offsets it.
          dot += input[w * 8 + n] * (affine ? nib : nib - 7)
          xs += input[w * 8 + n]
        }
        acc += scale * dot
        if (affine) acc += biases[row * SPR + g] * xs
      }
      return outF32 ? acc : toF16(acc)
    })

    const outBytes = M * (outF32 ? 4 : 2)
    const pipe = pipelineFor(device, withPrelude(src), entry)
    const out = device.createBuffer({ size: outBytes, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      out,
      buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
      buffer(device, weights, BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([KP, SPR, M, 0]), BU.UNIFORM | BU.COPY_DST),
      // binding index == array index, so the affine bias lands at @binding(5).
      ...(affine ? [buffer(device, f16Array(biases), BU.STORAGE | BU.COPY_DST)] : []),
    ]
    return runCompute(device, pipe, buffers, [M / rowsPerWG], 0, outBytes).then((bytes) => {
      const got = outF32
        ? Array.from(new Float32Array(bytes))
        : Array.from(new Uint16Array(bytes), f16BitsToF32)
      const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / (Math.abs(ref[i]) + 1e-3)))
      return { name: entry, pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
    })
  }
}

// ── rms_norm.wgsl : out[i] = input[i] / sqrt(mean(input^2)+1e-5) * gamma[i] ──
function testRmsNorm(device) {
  const r = rng(2)
  const D = PHI3.d // prelude const in the kernel
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

// ── moe_router_logits.wgsl : the [rows, D] affine matvec that picks experts.
//
// Two entry points, one per router width, because the width is a property of
// the CHECKPOINT and not of the family: Qwen3.6 ships the router as an 8-bit
// per-tensor override, Qwen3-30B-A3B leaves it at the model's base 4 bits.
// Running the wrong one is not an error — the 8-bit reader walks a 4-bit row at
// twice the stride and every expert's logit comes out of the next expert's
// bytes — so each width is checked here against the same reference formula:
//
//   logit[e] = Sum_g ( s[e,g] * Sum_i x[i]*q[e,i] + b[e,g] * Sum_i x[i] )
//
// with i running over group g's 64 values, q unsigned (MLX affine stores
// w = s*q + b directly, no -7 offset).
function makeRouterLogitsTest(bits) {
  // 16 = the checkpoint left the router in f16 (DeepSeek). Different file,
  // because its bind group has no scales or biases in it.
  const file = bits === 16 ? 'moe_router_logits_f16.wgsl' : 'moe_router_logits.wgsl'
  const entry = bits === 16 ? 'moe_router_logits_f16' : bits === 4 ? 'moe_router_logits_q4' : 'moe_router_logits'
  return function routerLogitsTest(device) {
    const r = rng(11 + bits)
    const D = 2048            // Qwen3-30B-A3B / Qwen3.6 hidden size
    const rows = 9            // 8 routed experts + a shared gate — shape is irrelevant here
    const GPR = D / 64        // affine groups per row
    const PER_WORD = 32 / bits
    const MAXQ = (1 << bits) - 1

    const x = arr(D, () => toF16(r() * 2 - 1))
    const q = arr(rows * D, () => Math.floor(r() * (MAXQ + 1)))
    const scales = arr(rows * GPR, () => toF16(r() * 0.02 + 0.005))
    const biases = arr(rows * GPR, () => toF16(r() * 0.2 - 0.1))

    // f16 path: plain [rows, D] weights, no groups at all.
    const w16 = bits === 16 ? arr(rows * D, () => toF16(r() * 0.1 - 0.05)) : null
    const ref = bits === 16
      ? arr(rows, (_, e) => {
        let acc = 0
        for (let i = 0; i < D; i++) acc += x[i] * w16[e * D + i]
        return acc
      })
      : arr(rows, (_, e) => {
      let acc = 0
      for (let g = 0; g < GPR; g++) {
        let dot = 0, xs = 0
        for (let j = 0; j < 64; j++) {
          const i = g * 64 + j
          dot += x[i] * q[e * D + i]
          xs += x[i]
        }
        acc += scales[e * GPR + g] * dot + biases[e * GPR + g] * xs
      }
      return acc
    })

    // Pack q into u32 words, little end first — the same order the kernel
    // unpacks with `word >> (bits * n)`.
    const words = new Uint32Array((rows * D) / PER_WORD)
    for (let w = 0; w < words.length; w++) {
      let v = 0
      for (let n = 0; n < PER_WORD; n++) v |= q[w * PER_WORD + n] << (bits * n)
      words[w] = v >>> 0
    }

    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: rows * 4, usage: BU.STORAGE | BU.COPY_SRC })
    // The f16 router carries its weights directly, so the reference below is
    // the same formula with scale 1 and bias 0 — see `w16`.
    const buffers = bits === 16
      ? [out,
         buffer(device, f16Array(x), BU.STORAGE | BU.COPY_DST),
         buffer(device, f16Array(w16), BU.STORAGE | BU.COPY_DST),
         buffer(device, new Uint32Array([D, rows]), BU.UNIFORM | BU.COPY_DST)]
      : [out,
         buffer(device, f16Array(x), BU.STORAGE | BU.COPY_DST),
         buffer(device, words, BU.STORAGE | BU.COPY_DST),
         buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
         buffer(device, f16Array(biases), BU.STORAGE | BU.COPY_DST),
         buffer(device, new Uint32Array([D, rows]), BU.UNIFORM | BU.COPY_DST)]
    // One workgroup per row, exactly as the engine dispatches it.
    return runCompute(device, pipe, buffers, [rows], 0, rows * 4).then((bytes) => {
      const got = Array.from(new Float32Array(bytes))
      // Inputs are f16-exact on both sides, so the only difference left is f32
      // summation order (per-word + tree reduction vs. per-group). A wrong
      // unpack width is off by O(1), nowhere near this.
      const maxRel = Math.max(...ref.map((v, i) => Math.abs(got[i] - v) / (Math.abs(v) + 1e-2)))
      return {
        name: entry,
        pass: maxRel < 1e-4,
        detail: `${rows} rows × ${D} · ${bits}-bit · max rel err ${maxRel.toExponential(2)}`,
      }
    })
  }
}

// ── moe_router_topk.wgsl : softmax over E, then the top-K, plus the OPTIONAL
// shared slot. The shared expert is what most large MoE checkpoints (Mixtral,
// Qwen3-MoE) do NOT have, so `hasShared = 0` is a first-class shape here, not
// a degenerate one: the router has E rows instead of E+1, the block runs K
// slots instead of K+1, and slot K must be left completely untouched.
function makeMoeTopkTest(E, K, normTopk, hasShared) {
  const label = `moe_router_topk_${hasShared ? 'shared' : 'noshared'}_e${E}k${K}`
  return function moeTopkTest(device) {
    const r = rng(7)
    const rows = E + (hasShared ? 1 : 0)
    const logits = arr(rows, () => r() * 8 - 4)

    // Reference: softmax over the ROUTED experts only (the shared gate, when
    // present, is row E and never enters the softmax — it takes a sigmoid).
    const peak = Math.max(...logits.slice(0, E))
    const ex = logits.slice(0, E).map((v) => Math.exp(v - peak))
    const denom = ex.reduce((a, b) => a + b, 0)
    // Descending score; ties break on the LOWER expert index (subgroupMin).
    const order = ex.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0] || a[1] - b[1])
    const sel = order.slice(0, K)
    const total = sel.reduce((a, [v]) => a + v, 0)
    const scale = normTopk ? 1 / total : 1 / denom
    const refIdx = sel.map(([, i]) => i)
    const refScore = sel.map(([v]) => v * scale)
    if (hasShared) {
      refIdx.push(E)
      refScore.push(1 / (1 + Math.exp(-logits[E])))
    }

    // Allocate K+1 either way and poison the last slot: on the no-shared path
    // the kernel must not write it, which a K-sized buffer could never detect.
    const POISON = 0xdeadbeef
    const idxBuf = buffer(device, new Uint32Array(K + 1).fill(POISON), BU.STORAGE | BU.COPY_SRC | BU.COPY_DST)
    const scoreBuf = buffer(device, new Float32Array(K + 1).fill(-999), BU.STORAGE | BU.COPY_SRC | BU.COPY_DST)

    const pipe = pipelineFor(device, wgsl('moe_router_topk.wgsl'), 'moe_router_topk')
    const buffers = [
      idxBuf,
      scoreBuf,
      buffer(device, new Float32Array(logits), BU.STORAGE | BU.COPY_DST),
      buffer(device, new Uint32Array([E, K, normTopk ? 1 : 0, hasShared ? 1 : 0]), BU.UNIFORM | BU.COPY_DST),
    ]
    return runComputeReads(device, pipe, buffers, [1], [
      { index: 0, bytes: (K + 1) * 4 },
      { index: 1, bytes: (K + 1) * 4 },
    ]).then(([idxBytes, scoreBytes]) => {
      const gotIdx = Array.from(new Uint32Array(idxBytes))
      const gotScore = Array.from(new Float32Array(scoreBytes))
      const n = refIdx.length
      const idxOk = refIdx.every((v, i) => v === gotIdx[i])
      const maxErr = Math.max(...refScore.map((v, i) => Math.abs(v - gotScore[i])))
      // The slot past the block must still hold the poison on the no-shared path.
      const tailClean = hasShared || (gotIdx[K] === POISON && gotScore[K] === -999)
      const sum = gotScore.slice(0, K).reduce((a, b) => a + b, 0)
      return {
        name: label,
        pass: idxOk && maxErr < 2e-6 && tailClean,
        detail: !idxOk
          ? `indices differ: ${gotIdx.slice(0, n)} vs ${refIdx}`
          : !tailClean
            ? `wrote past slot ${K - 1} on a no-shared block (idx=${gotIdx[K]}, score=${gotScore[K]})`
            : `${n} slots · max score err ${maxErr.toExponential(2)}`
              + ` · routed sum ${sum.toFixed(4)}${hasShared ? ` · shared ${gotScore[K].toFixed(4)}` : ' · slot K untouched'}`,
      }
    })
  }
}

// ── argmax[_sg].wgsl : index of the (first) maximum logit ────────────────────
function makeArgmaxTest(file, entry) {
  return function argmaxTest(device) {
    const r = rng(3)
    const V = PHI3.vocab
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

// ── sampler.wgsl : temperature / top-p / min-p over the logits ───────────────
// The kernel answers with a token INDEX, so this compares EXACTLY — no
// tolerance to hide behind. That only holds because the fixture is PEAKED:
// WGSL's exp() is allowed ~3 ULP and the kernel accumulates in f32, so the two
// arithmetics can disagree by ~1e-5 of the distribution's mass, and a flat
// fixture puts every draw within that of a CDF boundary. Uniform logits are
// exactly that trap — their top order statistics differ by ~6e-4, so the
// nucleus is thousands of near-identical tokens. iid Gumbel logits give the
// ~0.7 top probability a real LM head produces; every draw the fixtures below
// take sits more than 1e-3 of the mass from the nearest boundary.
const SAMPLE_PARTS = 64   // must match engine-core.ts's SAMPLE_PARTS

// lowbias32, the hash sampler.wgsl draws with, in u32 JS arithmetic.
function hash32(x) {
  let h = x >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  h = Math.imul(h, 0x7feb352d) >>> 0
  h = (h ^ (h >>> 15)) >>> 0
  h = Math.imul(h, 0x846ca68b) >>> 0
  h = (h ^ (h >>> 16)) >>> 0
  return h
}

// Greedy is NOT "the first maximiser": argmax.wgsl's tree reduction does not
// preserve index order (after the first round slot t may hold index t+128, and
// keeping the lower slot then keeps the later index), so on tied logits the
// answer is a property of the exact partitioning. sample_select runs that loop
// verbatim, and the reference has to say so — otherwise the tie fixture below
// would be asserting the wrong contract.
function argmaxRef(logits) {
  const V = logits.length
  const chunk = Math.ceil(V / 256)
  const val = new Float32Array(256), idx = new Int32Array(256)
  for (let t = 0; t < 256; t++) {
    let bv = Math.fround(-1e30), bi = 0
    for (let i = t * chunk, e = Math.min(t * chunk + chunk, V); i < e; i++) {
      if (logits[i] > bv) { bv = logits[i]; bi = i }
    }
    val[t] = bv; idx[t] = bi
  }
  for (let s = 128; s > 0; s >>= 1) {
    for (let t = 0; t < s; t++) if (val[t + s] > val[t]) { val[t] = val[t + s]; idx[t] = idx[t + s] }
  }
  return idx[0]
}

// The sampling math itself, written the obvious way: softmax the scaled
// logits, SORT for the nucleus cut (the kernel bisects for the same cutoff
// without sorting), intersect with min-p, walk the CDF. Filters are both
// computed against the full distribution and intersected — see the note in
// sampler.wgsl about why applying one to the other's output is a different
// sampler.
function samplerRef(logits, { temperature, topP = 1, minP = 0, seed = 0, counter = 0 }) {
  const V = logits.length
  const best = argmaxRef(logits)
  if (temperature <= 0) return best

  const invT = 1 / temperature
  const M = logits[best] * invT
  const w = new Float64Array(V)
  let D = 0
  for (let i = 0; i < V; i++) { w[i] = Math.exp(logits[i] * invT - M); D += w[i] }

  // min-p is a threshold on w directly: p_i >= min_p * p_max and p_max = 1/D.
  let tau = minP
  if (topP < 1) {
    const sorted = Array.from(w).sort((a, b) => b - a)
    const want = topP * D
    let cum = 0, k = 0
    while (k < V) { cum += sorted[k]; k++; if (cum >= want) break }
    tau = Math.max(tau, sorted[k - 1])
  }

  let kept = 0
  for (let i = 0; i < V; i++) if (w[i] >= tau) kept += w[i]
  const target = ((hash32(seed ^ hash32(counter)) >>> 8) / 16777216) * kept
  let run = 0, picked = best
  for (let i = 0; i < V; i++) {
    if (w[i] >= tau) { run += w[i]; picked = i; if (run > target) break }
  }
  return picked
}

// iid Gumbel, the default fixture (see the note above).
const gumbelLogits = (V) => {
  const r = rng(31)
  return Float32Array.from(arr(V, () => 2 * -Math.log(-Math.log(r()))))
}

function makeSamplerTest(V, cfg, draws, vsArgmax = false, makeLogits = gumbelLogits) {
  return async function samplerTest(device) {
    const logits = makeLogits(V)
    const ref = arr(draws, (_, c) => samplerRef(logits, { ...cfg, counter: c }))

    const reduce = pipelineFor(device, wgsl('sampler.wgsl'), 'sample_reduce')
    const select = pipelineFor(device, wgsl('sampler.wgsl'), 'sample_select')
    const logitBuf = buffer(device, logits, BU.STORAGE | BU.COPY_DST)
    const partials = device.createBuffer({ size: SAMPLE_PARTS * 2 * 4, usage: BU.STORAGE })
    // One output + one uniform per draw so every draw rides in ONE submit —
    // this harness pays ~100 ms per submit whatever is in it (see gpu.mjs).
    const outs = arr(draws, () => device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC }))
    const reads = arr(draws, () => device.createBuffer({ size: 4, usage: BU.COPY_DST | BU.MAP_READ }))

    const enc = device.createCommandEncoder()
    for (let c = 0; c < draws; c++) {
      const params = podBuffer(device, [
        { u32: V }, { u32: SAMPLE_PARTS }, { f32: cfg.temperature },
        { f32: cfg.topP ?? 1 }, { f32: cfg.minP ?? 0 }, { u32: cfg.seed }, { u32: c },
      ])
      for (const [pipe, wgs] of [[reduce, SAMPLE_PARTS], [select, 1]]) {
        const pass = enc.beginComputePass()
        pass.setPipeline(pipe)
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [logitBuf, partials, outs[c], params]
            .map((b, i) => ({ binding: i, resource: { buffer: b } })),
        }))
        pass.dispatchWorkgroups(wgs, 1, 1)
        pass.end()
      }
      enc.copyBufferToBuffer(outs[c], 0, reads[c], 0, 4)
    }
    device.queue.submit([enc.finish()])
    const got = []
    for (const b of reads) {
      await b.mapAsync(MM.READ)
      got.push(new Int32Array(b.getMappedRange())[0])
      b.unmap()
    }

    let pass = got.every((g, i) => g === ref[i])
    let detail = `gpu=[${got}] ref=[${ref}]`

    // Temperature 0 must be THE argmax path, not merely an argmax: every
    // reference comparison in this repo pins greedy decoding, so a tie broken
    // the other way would diverge silently from the harness it is checked with.
    if (vsArgmax) {
      const pipe = pipelineFor(device, wgsl('argmax.wgsl'), 'argmax_kernel')
      const result = device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC })
      const bytes = await runCompute(device, pipe, [
        logitBuf, result, buffer(device, new Uint32Array([V]), BU.UNIFORM | BU.COPY_DST),
      ], [1], 1, 4)
      const am = new Int32Array(bytes)[0]
      detail += ` argmax=${am}`
      pass = pass && got.every((g) => g === am)
    }
    return { name: 'sampler', pass, detail }
  }
}

// ── embedding.wgsl : out[i] = dequant(embd_weight[token_id, i]) ──────────────
function testEmbedding(device) {
  const r = rng(4)
  const D = PHI3.d, VOCAB = 64, DP = D / 8, SPR = D / 32, token = 17
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
  const QKV = PHI3.qkvDim, HD = PHI3.headDim, pos = 7
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
    buffer(device, ropeInvFreqTable(PHI3), BU.STORAGE | BU.COPY_DST), // inv_freq (binding 6)
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
  const D = PHI3.d
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
  const ROW = PHI3.kvDim, PAGE = PHI3.kvPageStride, NUM_PAGES = 2, position = 20
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
    const D = PHI3.d, DP = D / 8, SPR = D / 32, ROWS = 8
    const maxRow = ROWS - 1 + PHI3.ffn
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
      const g = dot(i), u = dot(i + PHI3.ffn)
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
    const HEADS = 2, HD = PHI3.headDim, PAGE = PHI3.kvPageStride, NUM_PAGES = 3
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
    const D = PHI3.d, ROWS = PHI3.qkvDim, KP = PHI3.dPacked, SPR = PHI3.dScales
    const POS = 20, NUM_PAGES = 2, PAGE = PHI3.kvPageStride
    const pageNo = (POS / 16) | 0, slot = POS % 16
    const PAIRS = PHI3.qkvPairs

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

// ── attention_splitk[_sg] + attention_combine : split-K flash-decode ─────────
// Same setup as the attention test (shuffled page table), but the KV range is
// partitioned across N workgroups per head writing (m, d, o[96]) partials,
// then a combine pass merges them. Parametrized over N and kv_len to cover:
// page count not divisible by N, empty partitions (fewer pages than N), and
// kv_len < N. Compared against the same standard-softmax CPU reference.
function makeAttentionSplitkTest(file, entry, N, KV_LEN) {
  return function attentionSplitkTest(device) {
    const r = rng(9)
    const HEADS = 2, HD = PHI3.headDim, PAGE = PHI3.kvPageStride
    const NUM_PAGES = Math.ceil(KV_LEN / 16)
    const smScale = 1 / Math.sqrt(HD)
    // Deterministic shuffle: logical page i → physical page (i-1) mod P
    // (same [2,0,1] order as the attention test when P=3).
    const pageOrder = arr(NUM_PAGES, (_, i) => (i + NUM_PAGES - 1) % NUM_PAGES)

    const q = arr(32 * HD, () => toF16(r() * 2 - 1))
    const pages = new Uint16Array(NUM_PAGES * PAGE)
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

    const splitPipe = pipelineFor(device, wgsl(file), entry)
    const combinePipe = pipelineFor(device, wgsl('attention_combine.wgsl'), 'attention_combine')
    // Partials sized for the full 32-head layout the engine uses (the kernel
    // indexes by absolute head id) — 98 f32 per (head, partition).
    const partials = device.createBuffer({
      size: 32 * N * 98 * 4,
      usage: BU.STORAGE | BU.COPY_SRC,
    })
    const out = device.createBuffer({ size: 3072 * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const qBuf = buffer(device, f16Array(q), BU.STORAGE | BU.COPY_DST)
    const indptr = buffer(device, new Int32Array([0, NUM_PAGES]), BU.STORAGE | BU.COPY_DST)
    const values = buffer(device, new Int32Array(pageOrder), BU.STORAGE | BU.COPY_DST)
    const pagesBuf = buffer(device, pages, BU.STORAGE | BU.COPY_DST)
    const lengthInfo = buffer(device, new Int32Array([KV_LEN]), BU.STORAGE | BU.COPY_DST)
    const splitU = podBuffer(device, [
      { i32: 1 },            // B
      { i32: NUM_PAGES },    // max_num_pages
      { i32: NUM_PAGES },    // nnz_pages
      { i32: 0 }, { i32: 0 }, { i32: 0 }, { i32: 0 }, // elem offsets
      { f32: smScale },
      { u32: N },            // num_splits
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

    // Two sequential passes: partial (N, HEADS) then combine (1, HEADS).
    const enc = device.createCommandEncoder()
    const p1 = enc.beginComputePass()
    p1.setPipeline(splitPipe); p1.setBindGroup(0, splitBG)
    p1.dispatchWorkgroups(N, HEADS, 1); p1.end()
    const p2 = enc.beginComputePass()
    p2.setPipeline(combinePipe); p2.setBindGroup(0, combineBG)
    p2.dispatchWorkgroups(1, HEADS, 1); p2.end()
    const read = device.createBuffer({ size: HEADS * HD * 2, usage: BU.COPY_DST | BU.MAP_READ })
    enc.copyBufferToBuffer(out, 0, read, 0, HEADS * HD * 2)
    device.queue.submit([enc.finish()])

    return read.mapAsync(MM.READ).then(() => {
      const got = Array.from(new Uint16Array(read.getMappedRange().slice(0)), f16BitsToF32)
      read.unmap()
      const maxAbs = Math.max(...got.map((g, i) => Math.abs(g - ref[i])))
      const name = `${entry} N=${N} kv=${KV_LEN}`
      return { name, pass: maxAbs < 5e-3, detail: `max abs err ${maxAbs.toExponential(2)}` }
    })
  }
}

// ── add3_norm.wgsl : residual = (A+B)+C ; out = rmsnorm(residual) * gamma ────
function testAdd3Norm(device) {
  const r = rng(6)
  const D = PHI3.d
  const A = arr(D, () => toF16(r() * 2 - 1))
  const B = arr(D, () => toF16(r() * 2 - 1))
  const C = arr(D, () => toF16(r() * 2 - 1))
  const gamma = arr(D, () => toF16(r() * 0.5 + 0.75))
  // Sequential f16 rounding, matching the two-step add_norm ping-pong.
  const resid = A.map((a, i) => toF16(toF16(a + B[i]) + C[i]))
  let ss = 0
  for (const x of resid) ss += x * x
  const rinv = 1 / Math.sqrt(ss / D + 1e-5)
  const refOut = resid.map((x, i) => toF16(x * rinv * gamma[i]))
  const pipe = pipelineFor(device, wgsl('add3_norm.wgsl'), 'add3_norm')
  const out = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const residBuf = device.createBuffer({ size: D * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    buffer(device, f16Array(A), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(B), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(C), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST),
    out, residBuf,
    buffer(device, new Uint32Array([1]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runComputeReads(device, pipe, buffers, [1], [
    { index: 4, bytes: D * 2 },
    { index: 5, bytes: D * 2 },
  ]).then(([outBytes, residBytes]) => {
    const gotOut = Array.from(new Uint16Array(outBytes), f16BitsToF32)
    const gotResid = Array.from(new Uint16Array(residBytes), f16BitsToF32)
    let bad = 0
    for (let i = 0; i < D; i++) {
      bad = Math.max(bad, Math.abs(gotOut[i] - refOut[i]), Math.abs(gotResid[i] - resid[i]))
    }
    return { name: 'add3_norm', pass: bad < 5e-3, detail: `max abs err ${bad.toExponential(2)}` }
  })
}

// ── fused_ffn[_tiled_sg]_prologue : add_norm folded into the FFN's phase 1 ───
// CPU reference is the SEQUENTIAL path: add_norm (f16 residual add, f16
// normed store) followed by the fused-FFN dots on the normed f16 vector —
// exactly what the addNorm1 → fused_ffn dispatch pair computes.
function makeFusedFfnPrologueTest(file, entry, wgPerDispatch) {
  return function fusedFfnPrologueTest(device) {
    const r = rng(8)
    const D = PHI3.d, DP = D / 8, SPR = D / 32, ROWS = 8
    const maxRow = ROWS - 1 + PHI3.ffn
    const residual = arr(D, () => toF16(r() * 1.5 - 0.75))
    const delta = arr(D, () => toF16(r() * 1.5 - 0.75))
    const gamma = arr(D, () => toF16(r() * 0.5 + 0.75))
    const scales = arr((maxRow + 1) * SPR, () => toF16(r() * 0.05 + 0.01))
    const weights = Uint32Array.from(arr((maxRow + 1) * DP, () => (r() * 0xffffffff) >>> 0))

    // Reference add_norm (matches testAddNorm's math).
    const summed = residual.map((a, i) => toF16(a + delta[i]))
    let ss = 0
    for (const x of summed) ss += x * x
    const rinv = 1 / Math.sqrt(ss / D + 1e-5)
    const normed = summed.map((x, i) => toF16(x * rinv * gamma[i]))

    const dot = (row) => {
      let acc = 0
      for (let w = 0; w < DP; w++) {
        const packed = weights[row * DP + w]
        const scale = scales[row * SPR + (w >> 2)]
        for (let n = 0; n < 8; n++) {
          const nib = (packed >>> (4 * n)) & 15
          acc += normed[w * 8 + n] * (nib - 7) * scale
        }
      }
      return acc
    }
    const ref = arr(ROWS, (_, i) => {
      const g = dot(i), u = dot(i + PHI3.ffn)
      const silu = g * (1 / (1 + Math.exp(-g)))
      return toF16(u * silu)
    })
    const pipe = pipelineFor(device, wgsl(file), entry)
    const out = device.createBuffer({ size: ROWS * 2, usage: BU.STORAGE | BU.COPY_SRC })
    const buffers = [
      out,
      buffer(device, f16Array(residual), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(delta), BU.STORAGE | BU.COPY_DST),
      buffer(device, f16Array(gamma), BU.STORAGE | BU.COPY_DST),
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

// ── test roster ──────────────────────────────────────────────────────────────
// Baseline tests run on any f16-capable adapter. Subgroup (`_sg`) variants
// additionally require the 'subgroups' feature and a minimum subgroup size
// (most kernels lay out their cross-lane reduction for 32-lane subgroups —
// same gate chat.ts applies before shipping them to a real GPU).
const TESTS = [
  { label: 'int4_matmul', fn: makeInt4MatmulTest(int4MatmulWGSL(), 'int4_matmul') },
  {
    // MLX-style affine quant (w = scale·q + bias, group 64) — the format the
    // Qwen3.6-35B-A3B MoE weights ship in. Caller passes MLX's bias verbatim.
    label: 'int4_matmul_affine',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ affine: true }), 'int4_matmul_affine', { affine: true }),
  },
  {
    label: 'int4_matmul_sg_affine',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true, affine: true }), 'int4_matmul_sg_affine', {
      affine: true,
    }),
    minSg: 32,
  },
  {
    // rowsPerWG=4: xsum is computed once per weight word and reused across the
    // 4 rows, so this also covers the shared-xsum path.
    label: 'int4_matmul_tiled_affine',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, affine: true }),
      'int4_matmul_tiled_affine',
      { affine: true, rowsPerWG: 4 },
    ),
    minSg: 32,
  },
  // AFFINE WIDE LOADS. The symmetric vec4 gets one scale per vec4 because its
  // groups are 32; affine groups are 64, so the scale index is v_off>>1 (vec4)
  // or v_off>>2 (vec2), and Σx must cover exactly the values the iteration
  // dequantizes. Both indices are off-by-a-shift bugs that produce plausible
  // numbers, so each width is graded at BOTH row counts.
  {
    label: 'int4_matmul_sg_vec4_affine',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, vec4: true, affine: true }),
      'int4_matmul_sg_vec4_affine', { affine: true, K: 1024 },
    ),
    minSg: 32,
  },
  {
    label: 'int4_matmul_tiled_vec4_affine',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, vec4: true, affine: true }),
      'int4_matmul_tiled_vec4_affine', { affine: true, rowsPerWG: 4, K: 1024 },
    ),
    minSg: 32,
  },
  {
    label: 'int4_matmul_sg_vec4h_affine',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, vec4Half: true, affine: true }),
      'int4_matmul_sg_vec4h_affine', { affine: true, K: 512 },
    ),
    minSg: 32,
  },
  {
    label: 'int4_matmul_tiled_vec4h_affine',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, vec4Half: true, affine: true }),
      'int4_matmul_tiled_vec4h_affine', { affine: true, rowsPerWG: 4, K: 512 },
    ),
    minSg: 32,
  },
  {
    // 3-bit experts (MLX bits=3): continuous-bitstream packing. The test's own
    // packer straddles word boundaries, so a kernel that mishandles the
    // straddle fails HERE, not inside a 15 GB model.
    label: 'int4_matmul_affine_q3',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ affine: true, q3: true }), 'int4_matmul_affine_q3', {
      affine: true,
      q3: true,
    }),
  },
  {
    label: 'int4_matmul_tiled_affine_q3',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, affine: true, q3: true }),
      'int4_matmul_tiled_affine_q3',
      { affine: true, q3: true, rowsPerWG: 4 },
    ),
    minSg: 32,
  },
  {
    label: 'int4_matmul_f32_affine',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ outF32: true, affine: true }), 'int4_matmul_f32_affine', {
      affine: true,
      outF32: true,
    }),
  },
  // ── RAGGED K: every case above uses K=512, so none of them would notice a
  // K-loop that truncates its tail. These use the two widths that actually
  // blocked real models — 768 (Qwen3-30B-A3B moe_intermediate) and 1408
  // (Qwen1.5-MoE ffn). Both are %64 (the scale-group width) but neither is
  // %512, so a trip-count division drops 256 / 384 elements per row and the
  // result is silently wrong rather than an error.
  {
    label: 'int4_matmul_affine_k768',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ affine: true }), 'int4_matmul_affine', {
      affine: true,
      K: 768,
    }),
  },
  {
    label: 'int4_matmul_tiled_affine_k1408',
    fn: makeInt4MatmulTest(
      int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, affine: true }),
      'int4_matmul_tiled_affine',
      { affine: true, rowsPerWG: 4, K: 1408 },
    ),
    minSg: 32,
  },
  {
    label: 'int4_matmul_affine_q3_k768',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ affine: true, q3: true }), 'int4_matmul_affine_q3', {
      affine: true,
      q3: true,
      K: 768,
    }),
  },
  // Both router widths against the same reference — the wrong one is silent.
  { label: 'moe_router_logits', fn: makeRouterLogitsTest(8) },
  { label: 'moe_router_logits_q4', fn: makeRouterLogitsTest(4) },
  { label: 'moe_router_logits_f16', fn: makeRouterLogitsTest(16) },
  // Qwen3.6 shape (256 experts, top-8, WITH a shared expert) and Qwen3-30B-A3B
  // shape (128 experts, top-8, WITHOUT one) — the two sides of the flag.
  { label: 'moe_topk_shared', fn: makeMoeTopkTest(256, 8, true, true), minSg: 32 },
  { label: 'moe_topk_noshared', fn: makeMoeTopkTest(128, 8, true, false), minSg: 32 },
  { label: 'moe_topk_noshared_nonorm', fn: makeMoeTopkTest(128, 8, false, false), minSg: 32 },
  {
    // K=1408, not 768: the _sg variant is a 32-thread workgroup, so 768/8 = 96
    // words divides evenly and would NOT be ragged here. 1408/8 = 176 leaves a
    // remainder of 16 against 32 lanes, which is the case worth guarding.
    label: 'int4_matmul_sg_k1408',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true }), 'int4_matmul_sg', { K: 1408 }),
    minSg: 32,
  },
  { label: 'rms_norm', fn: testRmsNorm },
  { label: 'argmax', fn: makeArgmaxTest('argmax.wgsl', 'argmax_kernel') },
  // Sampler fixtures. The two degenerate settings must land on the argmax
  // token exactly: temperature 0 short-circuits, and top_p 0 drives the
  // bisection to the top of its bracket so the nucleus is the peak alone.
  { label: 'sampler_greedy', fn: makeSamplerTest(PHI3.vocab, { temperature: 0, seed: 7 }, 1, true) },
  // Two logits tied at the maximum, placed so argmax.wgsl does NOT return the
  // first of them (chunk = ceil(32064/256) = 126; slot 0 carries 126*128 out
  // of the first reduction round and keeps it against slot 64's 126*64).
  // Anything that merely finds "an" argmax picks 8064 here.
  {
    label: 'sampler_tie',
    fn: makeSamplerTest(PHI3.vocab, { temperature: 0, seed: 7 }, 1, true, (V) => {
      const l = new Float32Array(V).fill(-1)
      l[126 * 64] = 9
      l[126 * 128] = 9
      return l
    }),
  },
  { label: 'sampler_topp0', fn: makeSamplerTest(PHI3.vocab, { temperature: 1.0, topP: 0, seed: 7 }, 2, true) },
  { label: 'sampler_temp', fn: makeSamplerTest(PHI3.vocab, { temperature: 1.5, seed: 7 }, 6) },
  { label: 'sampler_topp', fn: makeSamplerTest(PHI3.vocab, { temperature: 1.5, topP: 0.9, seed: 7 }, 6) },
  { label: 'sampler_minp', fn: makeSamplerTest(PHI3.vocab, { temperature: 1.5, minP: 0.02, seed: 7 }, 6) },
  { label: 'sampler_both', fn: makeSamplerTest(PHI3.vocab, { temperature: 1.5, topP: 0.95, minP: 0.001, seed: 7 }, 6) },
  // Qwen3.5's 248320 logits: 64 partitions of 3880, and a nucleus (1735
  // tokens) that no workgroup could have held.
  { label: 'sampler_wide', fn: makeSamplerTest(248320, { temperature: 1.5, topP: 0.99, seed: 7 }, 6) },
  { label: 'embedding', fn: testEmbedding },
  { label: 'rope', fn: testRope },
  { label: 'add_norm', fn: testAddNorm },
  { label: 'kv_append', fn: testKvAppend },
  { label: 'fused_ffn', fn: makeFusedFfnTest('fused_ffn.wgsl', 'fused_ffn_kernel', (rows) => rows) },
  { label: 'attention', fn: makeAttentionTest('attention.wgsl', 'attention') },
  { label: 'qkv_fused', fn: makeQkvFusedTest('qkv_fused.wgsl', 'qkv_fused') },
  { label: 'int4_matmul_sg', fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true }), 'int4_matmul_sg'), minSg: 32 },
  { label: 'argmax_sg', fn: makeArgmaxTest('argmax_sg.wgsl', 'argmax_sg'), minSg: 16 },
  { label: 'attention_sg', fn: makeAttentionTest('attention_sg.wgsl', 'attention_sg'), minSg: 32 },
  { label: 'qkv_fused_sg', fn: makeQkvFusedTest('qkv_fused_sg.wgsl', 'qkv_fused_sg'), minSg: 32 },
  {
    label: 'fused_ffn_tiled_sg',
    fn: makeFusedFfnTest('fused_ffn_tiled_sg.wgsl', 'fused_ffn_tiled_sg', (rows) => Math.ceil(rows / 4)),
    minSg: 32,
  },
  // ── opt-in experiment variants (?vec4 / ?vec4qkv / ?splitk / ?fuseprologue)
  // — built + correctness-tested here, throughput not yet measured ──────────
  // vec4 int4 matmuls: every knob combo variants.ts can resolve to
  // (tiled/sg × f16/f32 out). K=1024: the vec4 layout needs K % 1024 == 0.
  {
    label: 'int4_matmul_sg_vec4',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true, vec4: true }), 'int4_matmul_sg_vec4', { K: 1024 }),
    minSg: 32,
  },
  {
    label: 'int4_matmul_tiled_vec4',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, vec4: true }), 'int4_matmul_tiled_vec4', { K: 1024, rowsPerWG: 4 }),
    minSg: 32,
  },
  {
    label: 'int4_matmul_f32_sg_vec4',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ outF32: true, subgroups: true, vec4: true }), 'int4_matmul_f32_sg_vec4', { K: 1024, outF32: true }),
    minSg: 32,
  },
  {
    label: 'int4_matmul_f32_tiled_vec4',
    fn: makeInt4MatmulTest(int4MatmulWGSL({ outF32: true, subgroups: true, rowsPerWG: 4, vec4: true }), 'int4_matmul_f32_tiled_vec4', { K: 1024, rowsPerWG: 4, outF32: true }),
    minSg: 32,
  },
  { label: 'qkv_fused_sg_vec4', fn: makeQkvFusedTest('qkv_fused_sg_vec4.wgsl', 'qkv_fused_sg_vec4'), minSg: 32 },
  // split-K attention + combine. kv=40 is 3 pages: N=2 splits them 2+1
  // (pages not divisible by N); N=4 leaves one partition empty. kv=2 with
  // N=4 covers kv_len < N (three empty partitions).
  { label: 'attn_splitk2', fn: makeAttentionSplitkTest('attention_splitk.wgsl', 'attention_splitk', 2, 40) },
  { label: 'attn_splitk4', fn: makeAttentionSplitkTest('attention_splitk.wgsl', 'attention_splitk', 4, 40) },
  { label: 'attn_splitk4_short', fn: makeAttentionSplitkTest('attention_splitk.wgsl', 'attention_splitk', 4, 2) },
  {
    label: 'attn_splitk_sg',
    fn: makeAttentionSplitkTest('attention_splitk_sg.wgsl', 'attention_splitk_sg', 4, 40),
    minSg: 32,
  },
  // prologue fusion: add_norm folded into the FFN kernel + the 3-way
  // residual merge that replaces addNorm2.
  { label: 'add3_norm', fn: testAdd3Norm },
  {
    label: 'fused_ffn_prologue',
    fn: makeFusedFfnPrologueTest('fused_ffn_prologue.wgsl', 'fused_ffn_prologue', (rows) => rows),
  },
  {
    label: 'ffn_tiled_sg_prologue',
    fn: makeFusedFfnPrologueTest('fused_ffn_tiled_sg_prologue.wgsl', 'fused_ffn_tiled_sg_prologue', (rows) => Math.ceil(rows / 4)),
    minSg: 32,
  },
]

// ── compile-all gate ─────────────────────────────────────────────────────────
// Creates a shader module + compute pipeline for EVERY .wgsl file (through
// the same prelude path the app uses) and every shipped generator variant,
// and fails loudly on any compilation/validation error. This protects the
// shaders the correctness tests above don't execute.
async function compileAll(device, subgroups) {
  const sources = readdirSync(SHADERS)
    .filter((f) => f.endsWith('.wgsl'))
    .sort()
    .map((f) => ({ name: f, code: wgsl(f) }))
  for (const v of INT4_MATMUL_VARIANTS) {
    sources.push({ name: `gen:${int4MatmulEntry(v)}`, code: withPrelude(int4MatmulWGSL(v)) })
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
    detail: `${compiled} shaders compile clean (${sources.length - INT4_MATMUL_VARIANTS.length} .wgsl + ${INT4_MATMUL_VARIANTS.length} generated${skipNote})`,
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

  // Compile-all gate: every shader must at least build on this adapter.
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
  console.log(`\n${ran - failed}/${ran} kernels correct${skipNote}`)
  process.exit(failed ? 1 : 0)
}

main()
