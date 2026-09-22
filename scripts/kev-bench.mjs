#!/usr/bin/env node
// KEV-BENCH — decision latency on the engine, measured INSIDE the page.
//
//   node --experimental-strip-types scripts/kev-bench.mjs [kev|kev4b]
//
// Three shapes, each timed with performance.now() around decide() in the tab
// (no puppeteer round trip in the number), median of 5 after one warm-up:
//   1. questions per call on ONE state: 1, 5, 10, 25 — every question is its
//      own branch, so this is the branch cost with the state hot
//   2. cold state: a fresh state text each call (prefix reuse finds nothing
//      past <state>), 4 questions — what a new ticket costs
//   3. hot state: the SAME state again with a NEW question — the agent-loop
//      case, where the ONNX path re-reads the state and this one does not

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const model = process.argv[2] ?? 'kev'
const STATE = 'From: ops@bigcorp.io\nSubject: Prod API returning 502s since 03:00 UTC\n\nAll our checkout requests fail with 502 from your gateway. We are losing roughly $4k an hour. Escalate immediately. Ticket refs: INC-2291. We are on the Enterprise plan.'
const Q = (i) => ({ type: 'noul', instructions: `Statement ${i}: the sender is on an enterprise plan.` })
const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)] }

await startHarness()
try {
  const page = await newPage(`/model-smoke.html?model=${model}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error', { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') throw new Error(await page.evaluate(() => window.__error))
  const time = (state, questions) => page.evaluate(async (s, qs) => {
    const t0 = performance.now(); await window.__kevDecide(s, qs); return performance.now() - t0
  }, state, questions)
  const stateTokens = await page.evaluate(async (s) => {
    const r = await window.__kevRecord({ state: s, questions: [] }); return r.state.length
  }, STATE)
  console.log(`model=${model} state=${stateTokens} tokens (state cut at 384 by kev's MAX_STATE)`)

  console.log('\n1. questions per call, state hot (median of 5):')
  for (const n of [1, 5, 10, 25]) {
    const qs = Array.from({ length: n }, (_, i) => Q(i))
    await time(STATE, qs)
    const ms = []
    for (let r = 0; r < 5; r++) ms.push(await time(STATE, qs))
    const m = median(ms)
    console.log(`   ${String(n).padStart(3)} questions -> ${m.toFixed(0).padStart(5)} ms  (${(m / n).toFixed(1)} ms/question)`)
  }

  console.log('\n2. cold state, 4 questions (each call a new state):')
  {
    const qs = [0, 1, 2, 3].map(Q)
    const ms = []
    for (let r = 0; r < 6; r++) ms.push(await time(`${STATE}\n\nRun ${r}: ${Math.random()}`, qs))
    console.log(`   median ${median(ms.slice(1)).toFixed(0)} ms  (first ${ms[0].toFixed(0)} ms)`)
  }

  console.log('\n3. hot state, one NEW question per call:')
  {
    await time(STATE, [Q(0)])
    const ms = []
    for (let r = 0; r < 6; r++) ms.push(await time(STATE, [Q(100 + r)]))
    console.log(`   median ${median(ms.slice(1)).toFixed(0)} ms`)
  }
} finally {
  await stopHarness()
}
