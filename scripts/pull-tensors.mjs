#!/usr/bin/env node
// PULL-TENSORS — a handful of real tensors out of a checkpoint, by Range
// request, without downloading the checkpoint.
//
// A kernel-ref bundle needs ONE layer. DeepSeek-V2-Lite is 9 GB and its whole
// attention block is 7.7 MB of it, so fetching the repo to validate a kernel is
// three orders of magnitude of waste — and on a slow link it is the difference
// between an afternoon and a week.
//
//   node scripts/pull-tensors.mjs <hf-repo> --match '<regex>' --out <dir>
//   node scripts/pull-tensors.mjs mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx \
//        --match 'layers\.1\.(self_attn|input_layernorm)' --out .weights-local/kernel-refs/dsv2mla
//
// Writes <name>.bin verbatim (the shard's own bytes, no reinterpretation — a
// converter here would be one more place to misread a vendor's layout) plus
// meta.json carrying each tensor's shape and dtype from the safetensors header.
//
// --dry-run prints what it would fetch and the total, which is worth doing
// first on a metered connection.

import { mkdirSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { openMlxCheckpoint } from '../src/zero-tvm/weight-loader-mlx.ts'

const argv = process.argv.slice(2)
const flag = (name) => {
  const i = argv.indexOf(`--${name}`)
  return i < 0 ? null : argv[i + 1]
}
const repo = argv.find((a) => !a.startsWith('--') && !argv.includes(`--${a}`)) ?? argv[0]
const match = flag('match')
const outDir = flag('out')
const dryRun = argv.includes('--dry-run')
if (!repo || !match || (!outDir && !dryRun)) {
  console.error("usage: node scripts/pull-tensors.mjs <hf-repo> --match '<regex>' --out <dir> [--dry-run]")
  process.exit(2)
}

const base = `https://huggingface.co/${repo}/resolve/main`
const TIMEOUT = 120_000

async function whole(file) {
  const r = await fetch(`${base}/${file}`, { signal: AbortSignal.timeout(TIMEOUT) })
  if (!r.ok) throw new Error(`${file}: HTTP ${r.status}`)
  return new Uint8Array(await r.arrayBuffer())
}
async function range(file, begin, end) {
  const r = await fetch(`${base}/${file}`, {
    headers: { Range: `bytes=${begin}-${end - 1}` },
    signal: AbortSignal.timeout(TIMEOUT),
  })
  // A server that ignores Range answers 200 with the WHOLE file — on a 5 GB
  // shard that is exactly the download this script exists to avoid, so it is
  // an error rather than something to slice locally.
  if (r.status !== 206) throw new Error(`${file}: expected 206 for Range, got ${r.status}`)
  return new Uint8Array(await r.arrayBuffer())
}

const re = new RegExp(match)
console.log(`indexing ${repo} …`)
const ck = await openMlxCheckpoint({ whole, range })
const hits = ck.records.filter((r) => re.test(r)).sort()
if (!hits.length) {
  console.error(`no records match /${match}/ — the checkpoint has ${ck.records.length}`)
  process.exit(1)
}

const locs = hits.map((name) => ({ name, loc: ck.locate(name) }))
const total = locs.reduce((a, { loc }) => a + (loc.end - loc.begin), 0)
for (const { name, loc } of locs) {
  console.log(`  ${name.padEnd(52)} ${JSON.stringify(loc.info.shape)} ${loc.info.dtype} `
    + `${((loc.end - loc.begin) / 1e6).toFixed(2)} MB`)
}
console.log(`${locs.length} tensors, ${(total / 1e6).toFixed(1)} MB`)
if (dryRun) process.exit(0)

const out = resolve(outDir)
mkdirSync(out, { recursive: true })
const meta = { repo, match, tensors: {} }
let done = 0
for (const { name, loc } of locs) {
  const bytes = await range(loc.file, loc.begin, loc.end)
  // '/' is legal in a record name and not in a filename.
  const file = `${name.replace(/\//g, '__')}.bin`
  writeFileSync(join(out, file), bytes)
  meta.tensors[name] = { file, shape: loc.info.shape, dtype: loc.info.dtype, bytes: bytes.byteLength }
  done += bytes.byteLength
  process.stdout.write(`\r  fetched ${(done / 1e6).toFixed(1)}/${(total / 1e6).toFixed(1)} MB`)
}
writeFileSync(join(out, 'meta.json'), `${JSON.stringify(meta, null, 1)}\n`)
console.log(`\nwrote ${outDir}/ — ${locs.length} tensors + meta.json`)
