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

// ---- does splitting the encoder per layer cost what it looks like? ----
// The projection's spread (36 vs 88 t/s) is entirely this question. A streaming
// engine cannot record a whole token into one command buffer: each layer's
// router ids must reach JS before its experts can be fetched. So the real cost
// is not the round trip in isolation, it is the round trip with a layer's
// compute queued behind it.
//
// Arm A is what the engine does today — one submit for the whole token.
// Arm B is the split: per layer, submit, copy ids, await, continue.
const LAYERS = 48
const busy = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read_write> o : array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  var a = f32(g.x) * 0.5;
  for (var i = 0; i < 900; i = i + 1) { a = a * 1.0000001 + 0.000001; }
  o[g.x % 64u] = a;
}` })
const busyPipe = device.createComputePipeline({ layout: 'auto', compute: { module: busy, entryPoint: 'main' } })
const outBuf = device.createBuffer({ size: 4096, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
const busyBg = device.createBindGroup({
  layout: busyPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outBuf } }] })
// Calibrated so one dispatch costs what one real layer costs (~0.236 ms for
// qwen30b: 11.3 ms of decode over 48 layers). Testing a 0.18 ms round trip
// against 0.07 ms of compute answers a question nobody asked.
let WG = 512

const oneLayer = (enc) => {
  const pass = enc.beginComputePass()
  pass.setPipeline(busyPipe); pass.setBindGroup(0, busyBg); pass.dispatchWorkgroups(WG); pass.end()
}

// Arm A — everything in one command buffer, one wait at the end.
const armA = async () => {
  const enc = device.createCommandEncoder()
  for (let i = 0; i < LAYERS; i++) oneLayer(enc)
  device.queue.submit([enc.finish()])
  await device.queue.onSubmittedWorkDone()
}
// Arm B — split, with a readback between layers.
const armB = async () => {
  for (let i = 0; i < LAYERS; i++) {
    const enc = device.createCommandEncoder()
    oneLayer(enc)
    enc.copyBufferToBuffer(gpuBuf, 0, readBuf, 0, idsBytes)
    device.queue.submit([enc.finish()])
    await readBuf.mapAsync(GPUMapMode.READ)
    readBuf.unmap()
  }
}
// Calibrate WG against the real per-layer compute time.
{
  const target = 0.236
  for (let attempt = 0; attempt < 6; attempt++) {
    const enc = device.createCommandEncoder()
    for (let i = 0; i < 20; i++) oneLayer(enc)
    device.queue.submit([enc.finish()])
    const t = performance.now()
    await device.queue.onSubmittedWorkDone()
    const per = (performance.now() - t) / 20
    if (per > target * 0.85 && per < target * 1.15) break
    WG = Math.max(64, Math.round(WG * Math.min(4, Math.max(0.25, target / Math.max(per, 1e-4)))))
  }
}

// Arm C — PIPELINED. Submit the next layer before awaiting the previous
// layer's readback, which is what next-layer speculation would let an engine
// do. Two staging buffers, because a mapped buffer cannot be re-copied into.
const readBuf2 = device.createBuffer({ size: idsBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
const ring = [readBuf, readBuf2]
const armC = async () => {
  let pending = null
  for (let i = 0; i < LAYERS; i++) {
    const rb = ring[i % 2]
    const enc = device.createCommandEncoder()
    oneLayer(enc)
    enc.copyBufferToBuffer(gpuBuf, 0, rb, 0, idsBytes)
    device.queue.submit([enc.finish()])
    const p = rb.mapAsync(GPUMapMode.READ)
    if (pending) { await pending.p; pending.rb.unmap() }
    pending = { p, rb }
  }
  if (pending) { await pending.p; pending.rb.unmap() }
}

for (let i = 0; i < 3; i++) { await armA(); await armB(); await armC() }
const time = async (fn, n) => {
  const t = performance.now()
  for (let i = 0; i < n; i++) await fn()
  return (performance.now() - t) / n
}
const msA = await time(armA, 12)
const msB = await time(armB, 12)
const msC = await time(armC, 12)
console.log(`\nencoder split, ${LAYERS} layers of synthetic compute`)
console.log(`  one submit per token   ${msA.toFixed(2)} ms   (${(msA / LAYERS).toFixed(3)} ms/layer)`)
console.log(`  split + readback       ${msB.toFixed(2)} ms   (${(msB / LAYERS).toFixed(3)} ms/layer)`)
console.log(`  the split costs        ${(msB - msA).toFixed(2)} ms/token  (${((msB / msA - 1) * 100).toFixed(0)}% slower)`)
console.log(`  pipelined (submit ahead) ${msC.toFixed(2)} ms   (${(msC / LAYERS).toFixed(3)} ms/layer)`)
console.log(`  per layer overhead     serial ${((msB - msA) / LAYERS).toFixed(3)} ms · pipelined ${((msC - msA) / LAYERS).toFixed(3)} ms`)
console.log(`  against a ${rtMs.toFixed(3)} ms isolated round trip and ${(msA / LAYERS).toFixed(3)} ms/layer of compute`)

writeFileSync('bench/quality/moe-stream-probe.json', JSON.stringify({
  splitOneSubmitMs: msA, splitPerLayerMs: msB, splitPipelinedMs: msC, layers: LAYERS,
  writeBuffer: write, readbackMs: rtMs, iters: ITERS,
}, null, 2))
console.log(`\nwrote bench/quality/moe-stream-probe.json`)

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
