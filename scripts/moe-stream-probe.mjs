#!/usr/bin/env node
// MOE-STREAM-PROBE — the two costs the expert-streaming projection guessed at.
//
//   node --experimental-strip-types scripts/moe-stream-probe.mjs
//
// scripts/moe-replay.mjs converts cache hit rates into tokens per second. Its
// first version priced transfer at 2.4 GB/s — the low end of a writeBuffer
// range measured in BENCH.md for a DIFFERENT size — and ignored the router-id
// readback entirely. Both are measured here instead:
//
//   1. writeBuffer throughput AT THE EXPERT SLAB SIZE. Throughput is
//      size-dependent, and an expert slab is 1.3-2.5 MiB, not the multi-hundred
//      MiB a KV-pool save was measured at. A number carried over from another
//      size is not a measurement of this one.
//
//   2. mapAsync round-trip latency for a small buffer. WebGPU has no
//      GPU-waits-on-host primitive (verified: no fence/semaphore/event exists
//      in the API), so a streaming engine MUST return each layer's router ids
//      to JS before it knows which experts to fetch. That is 40-48 round trips
//      per token and it is the cost llama.cpp's Metal implementation avoids
//      with MTLSharedEvent. If it is large, streaming is latency-bound rather
//      than bandwidth-bound and no cache policy fixes it.
//
// Native (dawn.node), so the readback number is a floor: a browser adds its own
// scheduling on top.

import { writeFileSync } from 'node:fs'
import { installShims } from './native/shims.mjs'

await installShims({ unsafe: true })
const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice()

const MIB = 1024 * 1024
// The two shipped MoE shapes, plus neighbours to show the size dependence.
const SLABS = [0.5, 1.31, 2.0, 2.53, 4.0, 8.0]
const ITERS = 40

console.log(`\nwriteBuffer throughput by slab size (${ITERS} writes each)\n`)
console.log(`  ${'MiB'.padStart(6)} ${'GB/s'.padStart(8)} ${'ms each'.padStart(9)}`)
const write = {}
for (const mib of SLABS) {
  const bytes = Math.round(mib * MIB / 4) * 4      // writeBuffer requires a multiple of 4
  const src = new Uint8Array(bytes)
  const buf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  // Warm: the first write into a fresh buffer pays allocation, not bandwidth.
  for (let i = 0; i < 4; i++) device.queue.writeBuffer(buf, 0, src)
  await device.queue.onSubmittedWorkDone()
  const t0 = performance.now()
  for (let i = 0; i < ITERS; i++) device.queue.writeBuffer(buf, 0, src)
  await device.queue.onSubmittedWorkDone()
  const ms = (performance.now() - t0) / ITERS
  const gbps = bytes / (ms / 1000) / 1e9
  write[mib] = { gbps, ms }
  console.log(`  ${mib.toFixed(2).padStart(6)} ${gbps.toFixed(2).padStart(8)} ${ms.toFixed(3).padStart(9)}`)
  buf.destroy()
}

// ---- readback round trip ----
// A streaming engine needs the router ids on the CPU to issue a fetch. The
// realistic shape is: submit work, copy a tiny buffer out, await mapAsync.
const idsBytes = 64                       // top-8 u32 plus slack
const gpuBuf = device.createBuffer({ size: idsBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
const readBuf = device.createBuffer({ size: idsBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })

const roundTrip = async () => {
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, idsBytes)
  device.queue.submit([enc.finish()])
  await readBuf.mapAsync(GPUMapMode.READ)
  readBuf.unmap()
}
for (let i = 0; i < 5; i++) await roundTrip()
const N = 60
const t1 = performance.now()
for (let i = 0; i < N; i++) await roundTrip()
const rtMs = (performance.now() - t1) / N

console.log(`\nrouter-id readback (submit + copy + mapAsync + unmap)`)
console.log(`  ${rtMs.toFixed(3)} ms per round trip`)
console.log(`  x48 layers = ${(rtMs * 48).toFixed(1)} ms/token (qwen30b)`)
console.log(`  x40 layers = ${(rtMs * 40).toFixed(1)} ms/token (qwen36)`)
console.log(`\n  This is a FLOOR. dawn.node has no browser task scheduling in the way,`)
console.log(`  and a real engine also has compute queued behind these submits.`)

writeFileSync('bench/quality/moe-stream-probe.json', JSON.stringify({
  writeBuffer: write, readbackMs: rtMs, iters: ITERS,
}, null, 2))
console.log(`\nwrote bench/quality/moe-stream-probe.json`)

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
