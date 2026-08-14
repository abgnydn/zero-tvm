#!/usr/bin/env node
// MOE-SLAB-BYTES — does the pooled loader put the SAME bytes in a slot that the
// unpooled loader puts at that expert's row base?
//
//   node --experimental-strip-types scripts/moe-slab-bytes.mjs [dir] [layer]
//
// Why this exists. moe-pool-test.mjs found the pooled engine producing
// different tokens from the unpooled one, and the follow-ups narrowed it hard:
//
//   - two unpooled runs are token-identical  (cross-process compare is sound)
//   - two POOLED runs are token-identical    (the pooled path is deterministic)
//   - a FULL pool, where nothing is ever evicted, still diverges
//   - moe_combine accumulates over the router's top-K request slots in the same
//     order either way, and the matmul only changes its row base
//
// Deterministic on both sides, same arithmetic, same order — so the remaining
// difference has to be the BYTES. The two paths build them differently:
//
//   unpooled  buildPlan -> buildBuffer  over the whole stacked tensor
//   pooled    slab-source.read -> convertSlab  over one expert's slice
//
// This compares them directly. No GPU, no engine, no weights uploaded: it reads
// ranges out of the shards on disk. Minutes, not a 16 GB load.

import { openSync, readSync, closeSync, statSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const DIR = process.argv[2] ?? '.weights-local/Qwen3-30B-A3B-4bit'
const LAYER = Number(process.argv[3] ?? 0)
const MODEL = process.env.MODEL ?? 'qwen30b'
/** Experts to check per tensor. All of them is 128 ranged reads x 9 tensors. */
const SAMPLE = Number(process.env.SAMPLE || 6)

if (!statSync(DIR, { throwIfNoEntry: false })) {
  console.error(`no such checkpoint dir: ${DIR}`)
  process.exit(1)
}

/** MlxSource over local files — the shards are already on this disk. */
const fds = new Map()
const fdFor = (file) => {
  if (!fds.has(file)) fds.set(file, openSync(join(DIR, file), 'r'))
  return fds.get(file)
}
const src = {
  async range(file, begin, end) {
    const n = end - begin
    const buf = Buffer.allocUnsafe(n)
    let got = 0
    while (got < n) {
      const r = readSync(fdFor(file), buf, got, n - got, begin + got)
      if (r <= 0) break
      got += r
    }
    if (got !== n) throw new Error(`${file}: read ${got} of ${n} at ${begin}`)
    return new Uint8Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + n))
  },
  async whole(file) {
    return new Uint8Array(readFileSync(join(DIR, file)))
  },
}

const { specForParam } = await import('../src/zero-tvm/model-registry.ts')
const spec = specForParam(MODEL)
if (!spec.moe) { console.error(`${MODEL} is not a MoE spec`); process.exit(1) }

const { openMlxCheckpoint, planModel, buildPlan } = await import('../src/zero-tvm/weight-loader-mlx.ts')
const { fetchSlabSource } = await import('../src/zero-tvm/slab-source.ts')

const { locate } = await openMlxCheckpoint(src)
const slabs = fetchSlabSource(spec, locate, src)

console.log(`\n${DIR}`)
console.log(`  ${spec.id} — layer ${LAYER}, ${spec.moe.experts} experts, sampling ${SAMPLE}\n`)

const PLANS = [
  ['gate', 'w'], ['gate', 's'], ['gate', 'b'],
  ['up', 'w'], ['up', 's'], ['up', 'b'],
  ['down', 'w'], ['down', 's'], ['down', 'b'],
]

// The stacked tensor each pooled slot is a slice of.
const all = planModel(spec, { start: LAYER, end: LAYER + 1 })
let failed = 0

for (const [proj, kind] of PLANS) {
  const name = `moe_${proj}_proj_${kind}`
  const entry = all.find((p) => p.plan.name === name && p.layer === LAYER)
  if (!entry) { console.log(`SKIP  ${name.padEnd(20)} no plan for this spec`); continue }

  const stride = slabs.slabBytes(LAYER, proj, kind)
  const full = await buildPlan(entry.plan, locate, src)
  const rows = Math.floor(full.byteLength / stride)

  // Spread the sample: first, last, and evenly between — a bug at the seam
  // between two shards or at the shared expert hides in the middle otherwise.
  const picks = SAMPLE >= rows
    ? Array.from({ length: rows }, (_, i) => i)
    : Array.from({ length: SAMPLE }, (_, i) => Math.round((i * (rows - 1)) / (SAMPLE - 1)))

  let bad = 0, firstBad = null
  for (const e of picks) {
    const slab = await slabs.read(LAYER, proj, kind, e)
    const want = full.subarray(e * stride, (e + 1) * stride)
    if (slab.byteLength !== want.byteLength) {
      bad++; firstBad ??= { e, why: `length ${slab.byteLength} vs ${want.byteLength}` }
      continue
    }
    let at = -1
    for (let i = 0; i < want.byteLength; i++) if (slab[i] !== want[i]) { at = i; break }
    if (at >= 0) {
      bad++
      firstBad ??= { e, why: `byte ${at}: 0x${slab[at].toString(16).padStart(2, '0')} vs `
        + `0x${want[at].toString(16).padStart(2, '0')}` }
    }
  }

  const ok = bad === 0
  if (!ok) failed++
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(20)} ${String(stride).padStart(9)} B/expert  `
    + `${rows} rows  ${picks.length - bad}/${picks.length} identical`
    + (firstBad ? `\n      expert ${firstBad.e}: ${firstBad.why}` : ''))
}

for (const fd of fds.values()) closeSync(fd)

console.log()
if (failed === 0) {
  console.log('The two load paths agree byte for byte. The pooled engine\'s divergence')
  console.log('is NOT the weights — look at the readback/submit split instead.')
  process.exit(0)
}
console.log(`${failed} tensor(s) differ. The pooled engine multiplies by different`)
console.log('numbers than the unpooled one, which is exactly enough to move a token.')
process.exit(1)
