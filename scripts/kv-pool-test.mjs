#!/usr/bin/env node
// KV-POOL-TEST — does a cold restore compute what a full prefill computes?
//
//   node scripts/kv-pool-test.mjs [llama32|qwen35]
//
// The Phase-1 gate (docs/PAGING_PLAN.md). A session generates and saves; a
// FRESH engine (zeroed buffers — what a reload actually is) restores and
// continues; a second fresh engine prefills everything from zero as control.
// The continuations must be IDENTICAL. Per-token prefill in all arms so
// decode- and prefill-written KV share a kernel path and identity is the bar.
//
// Also: a fingerprint differing in ONE field (prefillPath) must refuse to
// restore — bytes under the wrong fingerprint are a different model's
// memories and nothing downstream can detect the swap.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'llama32'
const PROMPT = Array.from({ length: 120 }, (_, i) => 3000 + ((i * 41) % 700))
const EXTEND = Array.from({ length: 24 }, (_, i) => 5000 + ((i * 13) % 300))

let failed = false
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(30)} ${detail}`)
  if (!ok) failed = true
}

await startHarness()
try {
  const page = await newPage(`/model-smoke.html?model=${param}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  const r = await page.evaluate(
    (p, g, e, c) => window.__poolRun(p, g, e, c, true), PROMPT, 8, EXTEND, 16)

  // absorbed counts tokens whose FORWARD PASS ran: the last generated token
  // is an output that was never fed back, so prompt+gen-1 is the exact count
  // (±1 for the pipelined path's documented stop overrun).
  const expLo = PROMPT.length + 8 - 1, expHi = PROMPT.length + 8 + 1
  check('save captured the session', r.saved.tokens >= expLo && r.saved.tokens <= expHi,
    `${r.saved.tokens} tokens (prompt ${PROMPT.length}, ${8} generated)${r.saved.reason ? ' — ' + r.saved.reason : ''}`)
  check('restore attached', r.restoredCount === r.saved.tokens,
    `${r.restoredCount} of ${r.saved.tokens}`)
  check('engine reused the prefix', (r.reuseInfo?.reused ?? 0) >= r.saved.tokens,
    `prefill reused ${r.reuseInfo?.reused} of ${PROMPT.length + 8 + EXTEND.length}`)
  const same = r.control.length === r.restored.length && r.control.every((t, i) => t === r.restored[i])
  check('tokens IDENTICAL to control', same,
    same ? `${r.restored.length}/${r.restored.length}`
      : `diverge at ${r.control.findIndex((t, i) => t !== r.restored[i])}`)
  check('tampered fingerprint refused', r.tamperRestored === 0, r.tamperReason ?? '')
  check('restored continuation fast', r.restoredMs < 60_000, `${(r.restoredMs / 1000).toFixed(2)}s`)
  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
} finally {
  await stopHarness()
}
console.log(failed ? '\nKV pool BROKEN' : '\ncold restore computes what full prefill computes')
process.exit(failed ? 1 : 0)
