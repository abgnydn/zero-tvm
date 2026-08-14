#!/usr/bin/env node
// MOE-REPLAY — score cache policies against a real routing trace.
//
//   node scripts/moe-replay.mjs bench/quality/moe-trace-qwen30b.json
//
// Answers the one question that decides whether expert streaming is worth
// building: with only N of E experts resident per layer, how often does the
// router ask for one we do not have?
//
// A miss costs a disk read of one expert slab. Measured on this machine
// (BENCH.md): OPFS reads 7.6-8.7 GB/s page-cached, writeBuffer 2.4-6.7 GB/s.
// The slower of those bounds the answer, so the projections below use
// writeBuffer's low end and say so.
//
// Policies, because the literature disagrees about which wins:
//   LRU   evict least-recently-used. What llama.cpp's Metal PoC shipped.
//   LFU   evict least-frequently-used. One paper finds this beats LRU because
//         expert activation is SKEWED rather than time-local.
//   LFRU  frequency / age. Proposed in a vLLM RFC against LRU thrash.
//   OPT   Belady: evict whatever is needed furthest in the future. Not
//         implementable — it is the ceiling every real policy is measured
//         against, and the gap to it says how much prediction could buy.

import { readFileSync } from 'node:fs'
// The SHIPPED pool, not a second implementation of LRU. The number this script
// reports is the decision the engine will actually make; two LRUs would drift
// and the drift would not be visible until someone re-derived the projection.
import { ExpertPool } from '../src/zero-tvm/expert-pool.ts'

const file = process.argv[2] ?? 'bench/quality/moe-trace-qwen30b.json'
const t = JSON.parse(readFileSync(file, 'utf8'))
const { layers, experts, topK, slots, sharedExpertIndex, trace } = t

// Per-layer request streams. The shared expert is excluded: it runs for every
// token, so it is pinned resident by construction and counting it as a hit
// would flatter every policy equally.
const shared = sharedExpertIndex >= 0 ? sharedExpertIndex : -1
const streams = Array.from({ length: layers }, () => [])
for (const step of trace) {
  for (let L = 0; L < layers; L++) {
    const row = step.slice(L * slots, (L + 1) * slots)
    for (const e of row) if (e !== shared && e < experts) streams[L].push(e)
  }
}

const total = streams.reduce((a, s) => a + s.length, 0)
if (!total) { console.error('empty trace'); process.exit(1) }

// ---- skew, before any policy ----
const counts = new Map()
for (const s of streams) for (const e of s) counts.set(e, (counts.get(e) ?? 0) + 1)
const sorted = [...counts.values()].sort((a, b) => b - a)
const share = (n) => sorted.slice(0, n).reduce((a, b) => a + b, 0) / total
console.log(`\n${t.specId} — ${trace.length} decode steps, ${layers} layers, `
  + `${experts} experts, top-${topK}${shared >= 0 ? ' + shared (pinned, excluded)' : ''}`)
console.log(`  ${counts.size}/${experts} experts ever used`)
console.log(`  top 10%  of experts take ${(share(Math.ceil(experts * 0.1)) * 100).toFixed(1)}% of requests`)
console.log(`  top 25%  take ${(share(Math.ceil(experts * 0.25)) * 100).toFixed(1)}%`)
console.log(`  top 50%  take ${(share(Math.ceil(experts * 0.5)) * 100).toFixed(1)}%`)
console.log(`  (uniform routing would give 10% / 25% / 50%)`)

// ---- policies ----
// LRU runs through the real pool. Requests go one at a time here because the
// trace is flattened per layer; the engine resolves a whole top-K at once,
// which only ever helps (duplicates inside a request are charged once).
function runPool(stream, cap, pin) {
  const pool = new ExpertPool(cap, pin >= 0 ? { pin: [pin] } : {})
  for (const e of stream) pool.resolve([e])
  return pool.hitRate
}

function run(policy, stream, cap) {
  const resident = new Map()          // expert -> metadata
  let hits = 0, tick = 0
  for (let i = 0; i < stream.length; i++) {
    const e = stream[i]
    tick++
    if (resident.has(e)) {
      hits++
      const m = resident.get(e)
      m.last = tick; m.freq++
    } else {
      if (resident.size >= cap) {
        let victim = -1, worst = Infinity
        for (const [k, m] of resident) {
          let score
          if (policy === 'lru') score = m.last
          else if (policy === 'lfu') score = m.freq
          else if (policy === 'lfru') score = m.freq / Math.max(1, tick - m.last)
          else { // opt — Belady
            let next = Infinity
            for (let j = i + 1; j < stream.length; j++) if (stream[j] === k) { next = j; break }
            score = -next
          }
          if (score < worst) { worst = score; victim = k }
        }
        resident.delete(victim)
      }
      resident.set(e, { last: tick, freq: 1 })
    }
  }
  return hits / stream.length
}

const POLICIES = ['lru', 'lfu', 'lfru', 'opt']
const FRACTIONS = [0.10, 0.25, 0.50]

console.log(`\n  hit rate by resident pool size (per layer), higher is better\n`)
console.log(`  ${'pool'.padEnd(16)} ${POLICIES.map((p) => p.toUpperCase().padStart(7)).join('')}`)
const results = {}
for (const f of FRACTIONS) {
  const cap = Math.max(1, Math.round(experts * f))
  const row = POLICIES.map((p) => {
    // OPT is O(n^2) per layer; on a long trace that is minutes. Cap its work.
    const hs = streams.map((s) => (p === 'lru'
      ? runPool(s, cap, -1)
      : run(p, p === 'opt' ? s.slice(0, 4000) : s, cap)))
    return hs.reduce((a, b) => a + b, 0) / hs.length
  })
  results[f] = row
  console.log(`  ${`${cap}/${experts} (${(f * 100).toFixed(0)}%)`.padEnd(16) } `
    + row.map((h) => `${(h * 100).toFixed(1)}%`.padStart(7)).join(''))
}

// ---- can the PREVIOUS TOKEN stand in as the prefetch hint? ----
// This is the measurement that matters most, because the readback is the
// binding cost: WebGPU forces one GPU->CPU round trip per layer before that
// layer's experts can be fetched, and at 0.2 ms x 48 layers that is most of a
// token. Reading back ALL layers at once costs one round trip instead of
// forty-eight — but only works if we know what to fetch a whole token early.
//
// So: how much of token N+1's routing was already in token N's? If that is
// high, the design is one readback per token, prefetching the next token from
// the current one, and the per-layer stall disappears.
{
  let same = 0, totalReq = 0
  const perLayer = []
  for (let L = 0; L < layers; L++) {
    let s2 = 0, n2 = 0
    for (let i = 1; i < trace.length; i++) {
      const prev = new Set(trace[i - 1].slice(L * slots, (L + 1) * slots).filter((e) => e !== shared && e < experts))
      const cur = trace[i].slice(L * slots, (L + 1) * slots).filter((e) => e !== shared && e < experts)
      for (const e of cur) { n2++; if (prev.has(e)) s2++ }
    }
    perLayer.push(s2 / Math.max(n2, 1))
    same += s2; totalReq += n2
  }
  const overall = same / totalReq
  const lo = Math.min(...perLayer), hi = Math.max(...perLayer)
  console.log(`\n  previous-token overlap: ${(overall * 100).toFixed(1)}% of a token's experts`)
  console.log(`  were already chosen by the token before it (per-layer ${(lo * 100).toFixed(0)}%-${(hi * 100).toFixed(0)}%).`)
  console.log(`  A prefetch keyed on the previous token would cover that much with ONE`)
  console.log(`  readback per token instead of one per layer.`)
}

// ---- what that means in tokens per second ----
// Every constant below is MEASURED, not carried over. The first version of
// this projection priced transfer at 2.4 GB/s (the low end of a writeBuffer
// range measured at a different size) and ignored the router readback
// entirely. scripts/moe-stream-probe.mjs measures both at the sizes that
// actually occur; scripts/decode-bench-native.mjs and the rate-label protocol
// runs give compute.
const probe = JSON.parse(readFileSync('bench/quality/moe-stream-probe.json', 'utf8'))

// Bytes per expert: 3 matrices of ffn x d at `bits`, plus f16 scale+bias per
// group of 64. Derived, so a different model reprices itself.
const bits = t.bits ?? 4
const d = t.d ?? 2048, ffn = t.ffn ?? 768
const paramsPerExpert = 3 * ffn * d
const bytesPerExpert = paramsPerExpert * bits / 8 + (paramsPerExpert / 64) * 4
const slabMiB = bytesPerExpert / 2 ** 20

// writeBuffer throughput is size-dependent; take the probe's nearest slab.
const sizes = Object.keys(probe.writeBuffer).map(Number)
const near = sizes.reduce((a, b) => (Math.abs(b - slabMiB) < Math.abs(a - slabMiB) ? b : a))
const GBPS = probe.writeBuffer[near].gbps
const READBACK_MS = probe.readbackMs

// Decode compute with everything resident — what the engine does today.
const COMPUTE_MS = { 'qwen3-30b-a3b-4bit': 1000 / 88.4, 'qwen36-35b-a3b-q3': 1000 / 74.87 }[t.specId] ?? 12

console.log(`\n  projection — ${slabMiB.toFixed(2)} MiB per expert`)
console.log(`    transfer  ${GBPS.toFixed(2)} GB/s   (measured at ${near} MiB)`)
console.log(`    readback  ${READBACK_MS.toFixed(3)} ms x ${layers} layers = ${(READBACK_MS * layers).toFixed(1)} ms/token`)
console.log(`    compute   ${COMPUTE_MS.toFixed(1)} ms/token (all experts resident, measured)`)
// Per-layer compute against per-layer readback decides whether the round trip
// can hide at all. They are the same order of magnitude here, which is the
// whole reason the pipelined column below is worth computing.
const perLayerCompute = COMPUTE_MS / layers
console.log(`    split     ${((probe.splitPerLayerMs - probe.splitOneSubmitMs) / probe.layers).toFixed(3)} ms/layer serial, `
  + `${((probe.splitPipelinedMs - probe.splitOneSubmitMs) / probe.layers).toFixed(3)} ms/layer pipelined (measured)`)
console.log(`    per layer: ${perLayerCompute.toFixed(3)} ms compute vs ${READBACK_MS.toFixed(3)} ms isolated round trip`)
console.log(`\n  ${'pool'.padEnd(12)} ${'miss'.padStart(6)} ${'xfer ms'.padStart(8)} ${'serial'.padStart(9)} ${'overlapped'.padStart(11)} ${'pipelined'.padStart(10)}`)
for (const f of FRACTIONS) {
  const best = Math.max(...results[f].slice(0, 3))     // best IMPLEMENTABLE policy
  const misses = layers * topK * (1 - best)
  const xfer = misses * bytesPerExpert / (GBPS * 1e9) * 1000
  // SERIAL / PIPELINED overheads are MEASURED by moe-stream-probe.mjs, not
  // derived from the isolated round trip. Splitting the encoder per layer costs
  // 0.184 ms/layer when each readback is awaited before the next submit, and
  // 0.062 ms/layer when the next layer is submitted first — the round trip
  // partly hides, but only partly, and only if the engine pipelines.
  const splitSerial = (probe.splitPerLayerMs - probe.splitOneSubmitMs) / probe.layers
  const splitPipe = (probe.splitPipelinedMs - probe.splitOneSubmitMs) / probe.layers
  const serial = COMPUTE_MS + splitSerial * layers + xfer
  // OVERLAPPED: speculating the next layer's experts lets the fetch run under
  // compute, so transfer only costs what exceeds it. The readback does NOT
  // disappear — the prediction still has to come back to JS, and WebGPU has no
  // GPU-side wait for it.
  const overlapped = COMPUTE_MS + splitPipe * layers + Math.max(0, xfer - COMPUTE_MS)
  // PIPELINED, an upper bound. Running layer L+1's router on layer L's hidden
  // state produces the prediction one layer early, so its readback overlaps
  // layer L's compute instead of blocking layer L+1. Whether that works turns
  // on per-layer compute exceeding the round trip, which here it barely does.
  // Requires speculation to be correct; published accuracy is ~96% at L+1, and
  // a wrong guess costs a stall, so treat this as the ceiling and not a plan.
  const pipelined = COMPUTE_MS + splitPipe * layers * 0.5 + Math.max(0, xfer - COMPUTE_MS)
  console.log(`  ${`${Math.round(experts * f)}/${experts}`.padEnd(12)} `
    + `${`${(100 - best * 100).toFixed(0)}%`.padStart(6)} ${xfer.toFixed(0).padStart(8)} `
    + `${`${(1000 / serial).toFixed(1)} t/s`.padStart(9)} ${`${(1000 / overlapped).toFixed(1)} t/s`.padStart(11)} `
    + `${`${(1000 / pipelined).toFixed(1)} t/s`.padStart(9)}`)
}
console.log(`\n  Splitting the encoder per layer is unavoidable — WebGPU has no`)
console.log(`  GPU-waits-on-host primitive, so each layer's router ids must reach JS`)
console.log(`  before its experts can be fetched. Submitting the next layer BEFORE`)
console.log(`  awaiting the previous readback cuts that cost to a third, and is`)
console.log(`  therefore worth more than any cache-policy change: LRU already sits`)
console.log(`  within a point of Belady.`)
