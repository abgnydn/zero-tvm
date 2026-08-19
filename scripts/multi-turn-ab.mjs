#!/usr/bin/env node
// MULTI-TURN A/B — is a wrong turn-2 answer the RENDERING or the REUSE?
//
// tests/e2e/multi-turn.test.ts asserts that turn 2 recalls a name given in turn
// 1. When it fails, two very different things look identical from the outside:
//
//   RENDERING  turn 2's prompt is correct and the model simply answers
//              differently than it did under the old prompt. Nothing to fix in
//              the engine; the test's expectation was calibrated on the old
//              rendering.
//   REUSE      turn 2's prompt is correct but the engine restored the wrong KV
//              or GDN state, so the model never saw turn 1. A real bug, and the
//              silent kind — the reply is fluent either way.
//
// `?reuse=0` separates them: it prefills every turn from zero, so the model gets
// the same prompt with no restored state. Same answer both ways -> rendering.
// Different -> reuse.
//
// This exists because the alternative is arguing about it. Needs a GPU and the
// qwen35 mirror primed, same as the e2e suite.
//
//   node scripts/multi-turn-ab.mjs                 # qwen35
//   MODEL=qwen36q3 node scripts/multi-turn-ab.mjs
import { bootAndWaitReady, newPage, startHarness, stopHarness } from '../tests/e2e/harness.js'

const MODEL = process.env.MODEL ?? 'qwen35'
const BOOT_TIMEOUT_MS = 10 * 60_000
const GEN_TIMEOUT_MS = 5 * 60_000

const TURNS = [
  'My name is Alice. Please remember it. Now: what is the capital of France? Answer in one short sentence.',
  'What is my name? Answer in one short sentence.',
]

async function sendTurn(page, text) {
  const before = await page.$$eval('.msg.ai', (els) => els.length)
  await page.type('#inp', text)
  await page.click('#btn')
  await page.waitForFunction(
    (n) => {
      const btn = document.getElementById('btn')
      return document.querySelectorAll('.msg.ai').length > n && !!btn && !btn.hidden
    },
    { timeout: GEN_TIMEOUT_MS, polling: 100 },
    before,
  )
  return page.$eval('.msg.ai:last-of-type .body', (el) => el.textContent || '')
}

async function run(query) {
  const page = await newPage(`/zero-tvm.html?model=${MODEL}${query}`)
  const prefill = []
  page.on('console', (m) => { if (m.text().includes('[engine] prefill:')) prefill.push(m.text()) })
  await bootAndWaitReady(page, BOOT_TIMEOUT_MS)
  const replies = []
  for (const t of TURNS) replies.push(await sendTurn(page, t))
  await page.close()
  return { replies, prefill }
}

await startHarness()
try {
  // Reuse OFF first: it is the control, and running it first means a failure in
  // the slow arm cannot be blamed on state left by the fast one.
  const off = await run('&reuse=0')
  const on = await run('')

  const show = (label, r) => {
    console.log(`\n=== ${label} ===`)
    r.replies.forEach((x, i) => console.log(`  turn ${i + 1}: ${JSON.stringify(x.slice(0, 140))}`))
    r.prefill.forEach((l) => console.log(`  ${l}`))
    console.log(`  recalls the name: ${r.replies[1].toLowerCase().includes('alice') ? 'YES' : 'no'}`)
  }
  show('reuse OFF (every turn prefills from zero)', off)
  show('reuse ON (the shipped path)', on)

  const a = off.replies[1].toLowerCase().includes('alice')
  const b = on.replies[1].toLowerCase().includes('alice')
  console.log(`
=== VERDICT ===`)
  if (a === b) {
    console.log(`  Both arms ${a ? 'recall' : 'fail to recall'} the name, so REUSE IS NOT THE CAUSE.`)
    if (!a) {
      console.log('  The prompt is the same in both arms, so this is what the model does')
      console.log('  with the prompt it is now given. Check the prompt against the vendor')
      console.log('  template (scripts/render-diff.py) before changing any engine code —')
      console.log("  and if it matches, the test's expectation was calibrated on the old one.")
    }
  } else {
    console.log('  THE ARMS DISAGREE — reuse changes the answer, which it must never do.')
    console.log(`  reuse off: ${a ? 'recalls' : 'does not recall'} · reuse on: ${b ? 'recalls' : 'does not recall'}`)
    console.log('  That is a restored-state bug. The reused prefix and the GDN rewind are')
    console.log('  the suspects; ?chunk=0 narrows it further.')
  }
} finally {
  await stopHarness()
}
