#!/usr/bin/env node
// INT8-AFFINE — the 8-bit dense kernels vs a CPU reference.
//
//   npm run test:kernels:int8
//
// kev-0.6b at 4-bit keeps 17/21 argmaxes against its own f32 forward; at
// 8-bit it keeps 21/21 (docs/kev-parity/sweep-kev.txt). These are the kernels
// that serve it: int8_affine_matvec (decode GEMV, revived from a dead file
// written for a MoE one-row use that never shipped) and int8_affine_batched
// (its chunked-prefill twin). Same MLX-affine math as the int4 family
// (w = s·q + b, group 64, unsigned byte per value, 4 per u32 word), f16
// write like every other projection kernel.
//
// The references below are written from the FORMULA (per-element w = s*q+b),
// not from the kernels' factored form (s·Σxq + b·Σx per group) — an error in
// that folding must show up as a mismatch rather than cancelling. Same rule
// the int4 affine tests hold.

import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import {
  getDevice,
  buffer,
  runCompute,
  pipelineFor,
  BU,
} from './gpu.mjs'
import { toF16, f16Array, f16BitsToF32 } from './half.mjs'
import { withPrelude } from '../../src/compiler/shader-prelude.ts'
import { PHI3 } from '../../src/compiler/model-spec.ts'

const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')
// The int8 shaders take every dimension from PODArgs — the prelude consts go
// unused. PHI3 is carried only so the file follows the suite convention.
const wgsl = (name) => withPrelude(readFileSync(resolve(SHADERS, name), 'utf8'), PHI3)

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
const maxRelDiff = (got, ref, eps) => {
  let m = 0
  for (let i = 0; i < ref.length; i++) m = Math.max(m, Math.abs(got[i] - ref[i]) / (Math.abs(ref[i]) + eps))
  return m
}

/** Random 8-bit affine weights: u8 quants packed 4/u32, f16 scales/biases. */
function randU8(r, N, K) {
  const KP = K / 4, SPR = K / 64
  const weights = new Uint32Array(N * KP)
  for (let i = 0; i < weights.length; i++) weights[i] = (r() * 0xffffffff) >>> 0
  const scales = new Float32Array(N * SPR)
  const biases = new Float32Array(N * SPR)
  for (let i = 0; i < scales.length; i++) {
    scales[i] = toF16(r() * 0.05 + 0.01)
    biases[i] = toF16(r() * 0.1 - 0.05)
  }
  return { weights, scales, biases, KP, SPR }
}

/** CPU reference from the formula: out[r] = Σᵢ x[i]·(s[g]·q[r,i] + b[g]). */
function refRow(weights, scales, biases, x, K, r) {
  const KP = K / 4
  let acc = 0
  for (let g = 0; g < K / 64; g++) {
    const s = scales[r * (K / 64) + g]
    const b = biases[r * (K / 64) + g]
    for (let i = 0; i < 64; i++) {
      const word = weights[r * KP + ((g * 64 + i) >> 2)]
      const q = (word >>> (8 * ((g * 64 + i) & 3))) & 255
      acc += x[g * 64 + i] * (s * q + b)
    }
  }
  return acc
}

async function testMatvec(device) {
  const r = rng(81)
  const K = 1024, N = 256   // kev-0.6b d × a slice of its ffn
  const w = randU8(r, N, K)
  const input = arr(K, () => toF16(r() * 2 - 1))
  const pipe = pipelineFor(device, wgsl('int8_affine_matvec.wgsl'), 'int8_affine_matvec')
  const bytes = await runCompute(device, pipe, [
    device.createBuffer({ size: N * 2, usage: BU.STORAGE | BU.COPY_SRC }),
    buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST),
    buffer(device, f16Array(Array.from(w.scales)), BU.STORAGE | BU.COPY_DST),
    buffer(device, w.weights, BU.STORAGE | BU.COPY_DST),
    buffer(device, new Uint32Array([K / 4, K / 64, N, 0]), BU.UNIFORM | BU.COPY_DST),
    buffer(device, f16Array(Array.from(w.biases)), BU.STORAGE | BU.COPY_DST),
  ], [N], 0, N * 2)
  const got = Array.from(new Uint16Array(bytes), f16BitsToF32)
  let maxRel = 0
  for (let rr = 0; rr < N; rr++) {
    const ref = refRow(w.weights, w.scales, w.biases, input, K, rr)
    maxRel = Math.max(maxRel, Math.abs(got[rr] - ref) / (Math.abs(ref) + 1e-2))
  }
  return { name: 'int8_affine_matvec', pass: maxRel < 1e-2, detail: `max rel err ${maxRel.toExponential(2)} vs CPU` }
}

async function testBatched(device) {
  const r = rng(82)
  const K = 1024, N = 256, M = 31, CAP = 64
  const w = randU8(r, N, K)
  const input = arr(CAP * K, () => toF16(r() * 2 - 1))
  const pipe = pipelineFor(device, wgsl('int8_affine_batched.wgsl'), 'int8_affine_batched')
  const inBuf = buffer(device, f16Array(input), BU.STORAGE | BU.COPY_DST)
  const run = async (m) => {
    const out = device.createBuffer({ size: CAP * N * 2, usage: BU.STORAGE | BU.COPY_SRC | BU.COPY_DST })
    device.queue.writeBuffer(out, 0, new Uint16Array(CAP * N))
    const pod = buffer(device, new Uint32Array([w.KP, w.SPR, N, m]), BU.UNIFORM | BU.COPY_DST)
    const bytes = await runCompute(device, pipe,
      [out, inBuf,
        buffer(device, f16Array(Array.from(w.scales)), BU.STORAGE | BU.COPY_DST),
        buffer(device, w.weights, BU.STORAGE | BU.COPY_DST),
        pod,
        buffer(device, f16Array(Array.from(w.biases)), BU.STORAGE | BU.COPY_DST)],
      [N, CAP], 0, CAP * N * 2)
    return new Uint16Array(bytes)
  }
  const gotM = await run(M)
  const gotCap = await run(CAP)
  let maxRel = 0
  for (let b = 0; b < M; b++) {
    const ref = []
    for (let rr = 0; rr < N; rr++) {
      ref.push(refRow(w.weights, w.scales, w.biases, input.slice(b * K, (b + 1) * K), K, rr))
    }
    const row = Array.from(gotM.subarray(b * N, (b + 1) * N), f16BitsToF32)
    maxRel = Math.max(maxRel, maxRelDiff(row, ref, 1e-2))
  }
  let tailBad = 0
  for (let i = M * N; i < CAP * N; i++) if (gotM[i] !== 0) tailBad++
  let mBad = 0
  for (let i = 0; i < M * N; i++) if (gotM[i] !== gotCap[i]) mBad++
  const pass = maxRel < 1e-2 && tailBad === 0 && mBad === 0
  return {
    name: 'int8_affine_batched',
    pass,
    detail: `max rel err ${maxRel.toExponential(2)} vs CPU, ${tailBad} rows-past-M writes, ${mBad} M-variance mismatches`,
  }
}

const TESTS = [
  { label: 'int8_affine_matvec', fn: testMatvec },
  { label: 'int8_affine_batched', fn: testBatched },
]

async function main() {
  let env
  try {
    env = await getDevice()
  } catch (e) {
    console.error(`ERROR: ${e.message}`)
    process.exit(1)
  }
  const { device, info, f16 } = env
  console.log(`adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16} | int8-affine`)
  if (!f16) {
    console.error('FAIL: adapter lacks shader-f16; cannot exercise the f16 kernels')
    process.exit(1)
  }
  let failed = 0
  for (const t of TESTS) {
    let res
    try {
      res = await t.fn(device)
    } catch (e) {
      res = { pass: false, detail: String(e).split('\n')[0] }
    }
    console.log(`${res.pass ? 'PASS ' : 'FAIL '} ${t.label.padEnd(18)} ${res.detail}`)
    if (!res.pass) failed++
  }
  console.log(`\nint8-affine: ${TESTS.length - failed} passed, ${failed} failed of ${TESTS.length}`)
  process.exit(failed ? 1 : 0)
}

main()
