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
import { getDevice, pipelineFor, buffer, runCompute, BU } from './gpu.mjs'
import { f16Array, f16BitsToF32 } from './half.mjs'
import { int4MatmulWGSL, int4MatmulEntry } from '../../src/compiler/shaders/int4_matmul.gen.ts'
import { withPrelude } from '../../src/compiler/shader-prelude.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const REFS = join(ROOT, '.weights-local/kernel-refs')

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

const KERNELS = { affine_matmul: affineMatmul }

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
