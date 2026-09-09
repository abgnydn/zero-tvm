#!/usr/bin/env node
// MOE-OPTIMISTIC-PROBE — price the two mechanisms the optimistic pooled
// recorder would stand on, before building any of it.
//
//   node --experimental-strip-types scripts/moe-optimistic-probe.mjs
//
// The design: fire each MoE layer's id-readback WITHOUT awaiting it, keep
// recording on the GPU-resident slot map, replay a layer from checkpoint when
// a late-arriving readback reveals a miss. Its floor is set by:
//   A. the readback WAVE — 40 fire-and-forget copies awaited together. If
//      mapAsync overlaps, the wave costs ~one round trip; if Dawn serialises,
//      it costs 40 of them and the design is dead on arrival.
//   B. the CHECKPOINT — copying a GDN state (1 MiB f32) per layer per token.
// Both measured against the CURRENT pattern (submit + await per layer) on the
// same synthetic workload: tiny dispatch standing in for a layer, 36-byte copy
// standing in for the ids.
import { execFileSync } from 'node:child_process'
import { installShims } from './native/shims.mjs'

const LAYERS = 40
const ROUNDS = Number(process.env.ROUNDS) || 30
const batt = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8' })
if (!/AC Power/.test(batt) || /discharging/.test(batt)) {
  console.log('TIMINGS REFUSED — on battery.'); process.exit(1)
}
await installShims({ unsafe: true })
const adapter = await navigator.gpu.requestAdapter()
const device = await adapter.requestDevice()

// A trivial dispatch so every "layer" puts real work between copies.
const mod = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read_write> buf : array<u32>;
@compute @workgroup_size(64) fn tick(@builtin(global_invocation_id) id : vec3<u32>) {
  if (id.x < arrayLength(&buf)) { buf[id.x] = buf[id.x] + 1u; }
}` })
const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'tick' } })
const work = device.createBuffer({ size: 1 << 20, usage: GPUBufferUsage.STORAGE })
const bgWork = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: work } }] })
const ids = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]

// ── A1: the CURRENT pattern — submit, await map, next layer ──
const staging2 = [0, 1].map(() => device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }))
const serial = []
for (let r = 0; r < ROUNDS; r++) {
  const t0 = performance.now()
  for (let L = 0; L < LAYERS; L++) {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bgWork); pass.dispatchWorkgroups(64); pass.end()
    const st = staging2[L % 2]
    enc.copyBufferToBuffer(ids, 0, st, 0, 36)
    device.queue.submit([enc.finish()])
    await st.mapAsync(GPUMapMode.READ); st.unmap()
  }
  serial.push(performance.now() - t0)
}

// ── A2: the WAVE — fire every copy, await them all at the end ──
const stagingN = Array.from({ length: LAYERS }, () => device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }))
const wave = []
for (let r = 0; r < ROUNDS; r++) {
  const t0 = performance.now()
  const maps = []
  for (let L = 0; L < LAYERS; L++) {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bgWork); pass.dispatchWorkgroups(64); pass.end()
    enc.copyBufferToBuffer(ids, 0, stagingN[L], 0, 36)
    device.queue.submit([enc.finish()])
    maps.push(stagingN[L].mapAsync(GPUMapMode.READ))
  }
  await Promise.all(maps)
  for (const st of stagingN) st.unmap()
  wave.push(performance.now() - t0)
}

// ── A3: first-arrival latency inside a wave — how stale is a miss report? ──
const firstArrival = []
for (let r = 0; r < ROUNDS; r++) {
  const maps = []
  let tFire0 = 0
  for (let L = 0; L < LAYERS; L++) {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bgWork); pass.dispatchWorkgroups(64); pass.end()
    enc.copyBufferToBuffer(ids, 0, stagingN[L], 0, 36)
    if (L === 0) tFire0 = performance.now()
    device.queue.submit([enc.finish()])
    maps.push(stagingN[L].mapAsync(GPUMapMode.READ))
  }
  await maps[0]
  firstArrival.push(performance.now() - tFire0)
  await Promise.all(maps)
  for (const st of stagingN) st.unmap()
}

// ── B: checkpoint copies — 30 GDN states of 1 MiB, on-queue ──
const state = device.createBuffer({ size: 1 << 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
const snaps = Array.from({ length: 30 }, () => device.createBuffer({ size: 1 << 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }))
const ckpt = []
for (let r = 0; r < ROUNDS; r++) {
  const t0 = performance.now()
  const enc = device.createCommandEncoder()
  for (const s of snaps) enc.copyBufferToBuffer(state, 0, s, 0, 1 << 20)
  device.queue.submit([enc.finish()])
  await device.queue.onSubmittedWorkDone()
  ckpt.push(performance.now() - t0)
}

console.log(`${LAYERS} layers, medians of ${ROUNDS}:`)
console.log(`  A1 serial submit+await   ${med(serial).toFixed(2)} ms/token  (${(med(serial) / LAYERS * 1000).toFixed(0)} µs/layer) — the current pooled pattern`)
console.log(`  A2 fire-and-forget wave  ${med(wave).toFixed(2)} ms/token — the optimistic pattern`)
console.log(`  A3 first readback lands  ${med(firstArrival).toFixed(2)} ms after fire — the miss-detection staleness`)
console.log(`  B  30 x 1MiB checkpoints ${med(ckpt).toFixed(2)} ms/token`)
const saved = med(serial) - med(wave)
console.log(`\nwave saves ${saved.toFixed(2)} ms/token of pure readback stall.`)
console.log('Against the measured pooled decode (15.3 t/s = 65 ms/token vs 58.6 t/s = 17 ms unpooled,')
console.log('i.e. ~48 ms of pool overhead at the half pool), that is the fraction this design can recover')
console.log('before replay costs. If A2 is not several times cheaper than A1, stop here.')
process.exit(0)
