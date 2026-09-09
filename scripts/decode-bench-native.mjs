#!/usr/bin/env node
// DECODE-BENCH-NATIVE — per-token decode rate on dawn.node, idle/hot A/B.
//
//   npm run dev                                        # weights mirror
//   node --experimental-strip-types scripts/decode-bench-native.mjs qwen35
//
// Alternates event-loop-idle and event-loop-hot phases IN ONE PROCESS so the
// dawn.node backoff effect (BENCH.md 2026-08-13) is measured against the same
// boot and the same thermal state — separate runs let machine drift
// masquerade as the effect (it did, twice, before this script existed).
// Rates are computed over tokens 8.. so prefill and warmup are excluded.
//
// Numbers are VOID unless the machine is on healthy power: check
// `pmset -g batt` says charged/charging on a real adapter FIRST — a 5%
// battery on a weak adapter hard-throttles the SoC and this bench cannot
// tell that from a slow kernel.

import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'qwen35'
await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const t0 = Date.now()
const { engine, tokenizer, spec } = await createEngineRaw({
  model,
  onProgress: (p) => { if (p.stage === 'ready') console.log(`[boot] ${p.message}`) },
})
console.log(`[boot] ${spec.id} in ${((Date.now() - t0) / 1000).toFixed(1)}s`)

let hot = 0
const spin = () => { if (hot > 0) setImmediate(spin) }

const prompts = [
  'Write a long essay about the history of navigation at sea.',
  'Explain how tides work in detail.',
  'Describe the water cycle thoroughly.',
  'Write a detailed history of mapmaking.',
  'Explain ocean currents at length.',
  'Describe how lighthouses were built.',
]
async function phase(label, useSpin, i) {
  if (useSpin) { hot = 1; setImmediate(spin) }
  const ids = tokenizer.encode(
    `<|im_start|>user\n${prompts[i]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`)
  const stamps = []
  await engine.generatePipelined(ids, 96, () => stamps.push(performance.now()))
  hot = 0
  const N = stamps.length
  if (N < 20) { console.log(`${label}: FAILED — only ${N} tokens`); return }
  const rate = ((N - 9) / ((stamps[N - 1] - stamps[8]) / 1000)).toFixed(1)
  console.log(`${label}: ${rate} tok/s`)
}

await phase('warmup', false, 0)
await phase('idle-1', false, 1)
await phase('hot-1 ', true, 2)
await phase('idle-2', false, 3)
await phase('hot-2 ', true, 4)
console.log('hot ≈ what agent-native serves (withHotLoop); idle = the backoff tax')
process.exit(0)
