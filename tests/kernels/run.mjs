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
import { getDevice, buffer, runCompute, pipelineFor, BU } from './gpu.mjs'
import { toF16, f16Array, f16BitsToF32 } from './half.mjs'

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

const TESTS = [testInt4Matmul, testRmsNorm, testArgmax]

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
