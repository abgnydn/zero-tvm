#!/usr/bin/env node
// MOE-SLAB-PROBE — what does one expert miss actually cost, end to end?
//
//   node --experimental-strip-types scripts/moe-slab-probe.mjs
//
// The residency projection prices a miss as a writeBuffer, which is the GPU
// half. The other half is reading the slab off disk, and that was assumed free
// on the strength of an OPFS figure in BENCH.md (7.6-8.7 GB/s) measured
// PAGE-CACHED and SEQUENTIAL. An expert miss is neither: it is a 1.3-2.5 MiB
// read at an effectively random offset in a 16 GB checkpoint. If that runs at
// disk speed rather than cache speed, the projection is wrong and the whole
// technique needs re-sizing.
//
// So this reads real slabs out of a real checkpoint and pushes them to the GPU,
// and reports the three costs separately: read, upload, and the two chained.
//
// CACHE HONESTY. A 16 GB working set on a 32 GB machine will be partly resident
// in the page cache no matter what, so the COLD arm below runs after
// `purge`-style pressure is not available and is therefore an UPPER bound on
// throughput, i.e. a LOWER bound on cost. It is labelled as such rather than
// quoted as the disk's speed.
//
// Node's fs stands in for OPFS. Both land on the same APFS volume; a browser
// adds its own layer, so treat these as the floor of what OPFS can do.

import { openSync, readSync, closeSync, statSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { installShims } from './native/shims.mjs'

const DIR = process.argv[2] ?? '.weights-local/Qwen3-30B-A3B-4bit'
const SLAB = Number(process.env.SLAB_MIB || 2.53) * 1024 * 1024
const N = Number(process.env.READS || 200)

const shards = readdirSync(DIR).filter((f) => f.endsWith('.safetensors')).map((f) => join(DIR, f))
if (!shards.length) { console.error(`no safetensors in ${DIR}`); process.exit(1) }
const sizes = shards.map((f) => statSync(f).size)
const totalBytes = sizes.reduce((a, b) => a + b, 0)
console.log(`\n${DIR}`)
console.log(`  ${shards.length} shards, ${(totalBytes / 2 ** 30).toFixed(1)} GiB`)
console.log(`  slab ${(SLAB / 2 ** 20).toFixed(2)} MiB x ${N} reads at random offsets\n`)

await installShims({ unsafe: true })
const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice()

const fds = shards.map((f) => openSync(f, 'r'))
const buf = Buffer.allocUnsafe(SLAB)
// Deterministic offsets, so a re-run measures the same thing.
let seed = 12345
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff
// DISJOINT offsets per arm. The first version reused one offset list, so the
// read-only pass warmed the page cache for the chained pass and "chained" came
// out FASTER than "read alone" — which is impossible, and is exactly how a
// benchmark reports a number the machine cannot deliver.
const makePicks = () => Array.from({ length: N }, () => {
  const i = Math.floor(rnd() * shards.length)
  return { i, off: Math.floor(rnd() * Math.max(1, sizes[i] - SLAB)) }
})
const picksRead = makePicks()
const picksChain = makePicks()

// ---- read only ----
let t0 = performance.now()
for (const p of picksRead) readSync(fds[p.i], buf, 0, SLAB, p.off)
const readMs = (performance.now() - t0) / N

// ---- upload only (same bytes, no disk) ----
const gpu = device.createBuffer({ size: SLAB, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
const view = new Uint8Array(buf.buffer, buf.byteOffset, SLAB & ~3)
for (let i = 0; i < 4; i++) device.queue.writeBuffer(gpu, 0, view)
await device.queue.onSubmittedWorkDone()
t0 = performance.now()
for (let i = 0; i < N; i++) device.queue.writeBuffer(gpu, 0, view)
await device.queue.onSubmittedWorkDone()
const upMs = (performance.now() - t0) / N

// ---- chained, which is what a miss actually is ----
t0 = performance.now()
for (const p of picksChain) {
  readSync(fds[p.i], buf, 0, SLAB, p.off)
  device.queue.writeBuffer(gpu, 0, view)
}
await device.queue.onSubmittedWorkDone()
const bothMs = (performance.now() - t0) / N

const gbps = (ms) => SLAB / (ms / 1000) / 1e9
console.log(`  ${'stage'.padEnd(22)} ${'ms'.padStart(8)} ${'GB/s'.padStart(8)}`)
console.log(`  ${'disk read'.padEnd(22)} ${readMs.toFixed(3).padStart(8)} ${gbps(readMs).toFixed(2).padStart(8)}`)
console.log(`  ${'writeBuffer upload'.padEnd(22)} ${upMs.toFixed(3).padStart(8)} ${gbps(upMs).toFixed(2).padStart(8)}`)
console.log(`  ${'chained (one miss)'.padEnd(22)} ${bothMs.toFixed(3).padStart(8)} ${gbps(bothMs).toFixed(2).padStart(8)}`)
console.log(`\n  A miss costs ${bothMs.toFixed(3)} ms. The projection in moe-replay.mjs`)
console.log(`  priced it at ${(SLAB / (9.1e9) * 1000).toFixed(3)} ms (upload only), so it is`)
console.log(`  ${(bothMs / (SLAB / 9.1e9 / 1000 * 1e6)).toFixed(1)}x optimistic if the read does not overlap.`)
console.log(`\n  Page cache: this machine has 32 GiB against a ${(totalBytes / 2 ** 30).toFixed(0)} GiB checkpoint,`)
console.log(`  so some of these reads were served from RAM. Treat the read row as an`)
console.log(`  UPPER bound on throughput.`)

import { writeFileSync } from 'node:fs'
writeFileSync('bench/quality/moe-slab-probe.json', JSON.stringify({
  slabBytes: SLAB, reads: N, readMs, uploadMs: upMs, missMs: bothMs,
  note: 'read arm ran at RAM speed (page cache); on a machine too small to hold '
      + 'the checkpoint — the case streaming exists for — it will be slower',
}, null, 2))
for (const fd of fds) closeSync(fd)
await new Promise((r) => process.stdout.write('', r))
process.exit(0)
