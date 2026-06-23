// Headless kernel-correctness suite.
//
// Loads the engine's real WGSL kernels from src/compiler/shaders/, runs each
// on the local WebGPU adapter (a real GPU on a dev machine, lavapipe in CI),
// and checks the GPU output against a plain-JS reference. Deterministic
// (seeded RNG) so CI is reproducible. Exits non-zero on any mismatch.
//
//   npm run test:kernels
//
// This is the automated net the project previously lacked — kernel
// correctness was only checkable by hand via test-shaders.html.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { getDevice, buffer, runCompute, runComputeReads, pipelineFor, BU } from './gpu.mjs'
import { toF16, f16Array, f16BitsToF32, f16fma, f16add } from './half.mjs'

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

// ── int4_matmul.wgsl : out[r] = dot(input, dequant(weights[r])) ──────────────
function testInt4Matmul(device) {
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

  const pipe = pipelineFor(device, wgsl('int4_matmul.wgsl'), 'int4_matmul')
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
    return { name: 'int4_matmul', pass: maxRel < 0.02, detail: `max rel err ${maxRel.toExponential(2)}` }
  })
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

// ── argmax.wgsl : index of the (first) maximum logit ─────────────────────────
function testArgmax(device) {
  const r = rng(3)
  const V = 32064 // Phi-3 vocab
  const logits = Float32Array.from(arr(V, () => r() * 20 - 10))
  let refIdx = 0
  for (let i = 1; i < V; i++) if (logits[i] > logits[refIdx]) refIdx = i

  const pipe = pipelineFor(device, wgsl('argmax.wgsl'), 'argmax_kernel')
  const result = device.createBuffer({ size: 4, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    buffer(device, logits, BU.STORAGE | BU.COPY_DST),
    result,
    buffer(device, new Uint32Array([V]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [1], 1, 4).then((bytes) => {
    const got = new Int32Array(bytes)[0]
    return { name: 'argmax', pass: got === refIdx, detail: `gpu=${got} ref=${refIdx}` }
  })
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

// ── fused_ffn.wgsl : out[i] = SiLU(gate·x) * (up·x), f16 accumulation ────────
function testFusedFfn(device) {
  const r = rng(8)
  const D = 3072, DP = D / 8, SPR = D / 32, ROWS = 8
  const maxRow = ROWS - 1 + 8192
  const input = arr(D, () => toF16(r() * 1.5 - 0.75))
  const scales = arr((maxRow + 1) * SPR, () => toF16(r() * 0.05 + 0.01))
  const weights = Uint32Array.from(arr((maxRow + 1) * DP, () => (r() * 0xffffffff) >>> 0))
  // Reference replicates the kernel's f16 accumulation structure: 64 lanes,
  // each doing 6 chunks of fma, then a pairwise f16 tree reduction.
  const dot = (row) => {
    const lane = new Array(64)
    for (let l = 0; l < 64; l++) {
      let acc = 0
      for (let chunk = 0; chunk < 6; chunk++) {
        const w = l + chunk * 64, base = w * 8
        const packed = weights[row * DP + w], scale = scales[row * SPR + (w >> 2)]
        for (let n = 0; n < 8; n++) {
          const nib = (packed >>> (4 * n)) & 15
          acc = f16fma(input[base + n], toF16((nib - 7) * scale), acc)
        }
      }
      lane[l] = acc
    }
    for (const stride of [32, 16, 8, 4, 2, 1])
      for (let t = 0; t < stride; t++) lane[t] = f16add(lane[t], lane[t + stride])
    return lane[0]
  }
  const ref = arr(ROWS, (_, i) => {
    const g = dot(i), u = dot(i + 8192)
    const silu = toF16(g * (1 / (1 + Math.exp(-g))))
    return toF16(u * silu)
  })
  const pipe = pipelineFor(device, wgsl('fused_ffn.wgsl'), 'fused_ffn_kernel')
  const out = device.createBuffer({ size: ROWS * 2, usage: BU.STORAGE | BU.COPY_SRC })
  const buffers = [
    out,
    buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(scales), BU.STORAGE | BU.COPY_DST),
    buffer(device, weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([ROWS]), BU.UNIFORM | BU.COPY_DST),
  ]
  return runCompute(device, pipe, buffers, [ROWS], 0, ROWS * 2).then((bytes) => {
    const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
    const maxRel = Math.max(...got.map((g, i) => Math.abs(g - ref[i]) / (Math.abs(ref[i]) + 1e-3)))
    return { name: 'fused_ffn', pass: maxRel < 0.03, detail: `max rel err ${maxRel.toExponential(2)}` }
  })
}

const TESTS = [
  testInt4Matmul,
  testRmsNorm,
  testArgmax,
  testEmbedding,
  testRope,
  testAddNorm,
  testKvAppend,
  testFusedFfn,
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
  const { device, info, f16 } = env
  console.log(`adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16}`)
  if (!f16) {
    console.error('FAIL: adapter lacks shader-f16; cannot exercise the f16 kernels')
    process.exit(1)
  }

  let failed = 0
  for (const t of TESTS) {
    let res
    try {
      res = await t(device)
    } catch (e) {
      res = { name: t.name, pass: false, detail: String(e).split('\n')[0] }
    }
    console.log(`${res.pass ? 'PASS' : 'FAIL'}  ${res.name.padEnd(13)} ${res.detail}`)
    if (!res.pass) failed++
  }
  console.log(`\n${TESTS.length - failed}/${TESTS.length} kernels correct`)
  process.exit(failed ? 1 : 0)
}

main()
