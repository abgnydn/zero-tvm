#!/usr/bin/env node
// CHUNK-CAP-SWEEP — does a bigger prefill chunk pay on LONG prompts?
//
//   node --experimental-strip-types scripts/chunk-cap-sweep.mjs [llama32] [4096]
//
// The cap moved 64 -> 256 when E5 landed (BENCH.md "Sweep round 3": 64->256 was
// -14% prefill time at an 800-token prompt; 512 was +3.6% AT 800 TOKENS ONLYand
// left unshipped). Agent prompts run 4k-24k (docs/PAGING_PLAN.md §0.2), where a
// bigger tile has more rows to amortize against — this measures the caps that
// were never tried at the lengths that matter.
//
// One engine per (cap, round), fresh boot each time so no arm inherits a warm
// pipeline cache; caps interleaved round-robin so SoC heat decays across all
// arms equally rather than penalising whichever ran last. Prefill is timed as
// time-to-first-token of generatePipelined on a synthetic prompt (no cross-turn
// reuse — resetKVTracking per run).
import { execFileSync } from 'node:child_process'
import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'llama32'
const PROMPT = Number(process.argv[3]) || 4096
const CAPS = (process.env.CAPS ?? '64,256,512,1024').split(',').map(Number)
const ROUNDS = Number(process.env.ROUNDS) || 3

const batt = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8' })
if (!/AC Power/.test(batt) || /discharging/.test(batt)) {
  console.log('TIMINGS REFUSED — on battery. This sweep is timing-only; nothing to run.')
  process.exit(1)
}

await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const ids = Array.from({ length: PROMPT }, (_, i) => 1000 + ((i * 37) % 900))
const results = new Map(CAPS.map((c) => [c, []]))

for (let r = 0; r < ROUNDS; r++) {
  for (const cap of CAPS) {
    const built = await createEngineRaw({ model, chunkCap: cap })
    built.engine.resetKVTracking()
    const t0 = performance.now()
    let tFirst = 0
    await built.engine.generatePipelined(ids, 4, (t) => { if (!tFirst) tFirst = performance.now() }, () => false)
    const ms = tFirst - t0
    results.get(cap).push(PROMPT / (ms / 1000))
    console.log(`round ${r} cap ${cap}: ${(PROMPT / (ms / 1000)).toFixed(1)} tok/s prefill (${ms.toFixed(0)} ms)`)
    built.engine.destroy?.()
  }
}

const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
console.log(`\n${model}, ${PROMPT}-token prompt, medians of ${ROUNDS}:`)
const base = med(results.get(256) ?? results.get(CAPS[0]))
for (const cap of CAPS) {
  const m = med(results.get(cap))
  console.log(`  cap ${String(cap).padStart(4)}: ${m.toFixed(1)} tok/s  (${((m / base - 1) * 100).toFixed(1)}% vs 256)`)
}
process.exit(0)
