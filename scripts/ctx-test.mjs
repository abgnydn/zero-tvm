#!/usr/bin/env node
// CTX-TEST — does ?ctx=262144 actually run, or merely parse?
//
//   node scripts/ctx-test.mjs [qwen35]
//
// Three claims, three checks:
//   1. An 8 GiB KV allocation (1 GiB per attention layer) boots and generates.
//   2. Tokens are IDENTICAL to the default-ctx engine — a bigger page table
//      must be invisible to the numerics.
//   3. The LAST slot (position maxContext-1, page 16,383) is addressable:
//      one pipelineStep there, zero GPU errors. This is an ADDRESSING check,
//      not a retrieval-quality claim — the cache below it is uninitialised
//      and the token that comes back means nothing.
// What this does NOT prove: quality at long range (needle tests are a
// separate, hour-long affair), and survival under real memory pressure —
// Metal commits lazily, so a quiet 32 GB machine passing says little about a
// loaded 16 GB one.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'qwen35'
const CTX = Number(process.env.CTX) || 262144
const IDS = Array.from({ length: 96 }, (_, i) => 2000 + ((i * 53) % 800))

let failed = false
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(30)} ${detail}`)
  if (!ok) failed = true
}

await startHarness()
try {
  // Arm 1: the big engine.
  const big = await newPage(`/model-smoke.html?model=${param}&ctx=${CTX}`)
  await big.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await big.evaluate(() => window.__phase) === 'error') {
    throw new Error(await big.evaluate(() => window.__error))
  }
  const t0 = Date.now()
  const bigTokens = await big.evaluate((ids, n) => window.__generate(ids, n), IDS, 16)
  check(`boot + generate at ctx=${CTX}`, bigTokens.length === 16,
    `${((Date.now() - t0) / 1000).toFixed(1)}s, tokens [${bigTokens.slice(0, 5).join(',')}…]`)

  // The last slot. pipelineStep writes KV at position CTX-1 → the top page.
  // Wrong address math here is an uncaptured GPU error or a device loss, not
  // a wrong token.
  const probe = await big.evaluate(async (ctx) => {
    const t = performance.now()
    // engine is closed over by the page; reach it through __generate's engine
    // via a fresh single step. model-smoke exposes no raw engine handle, so
    // this rides window.__pipelineStep if present, else skips.
    if (!window.__pipelineStep) return { skipped: true }
    const r = await window.__pipelineStep(1000, ctx - 1)
    return { ms: performance.now() - t, token: r }
  }, CTX)
  const gpuErrsBig = await big.evaluate(() => window.__gpuErrs())
  if (probe.skipped) console.log('SKIP  last-slot probe                 (no __pipelineStep hook)')
  else check('last slot addressable', gpuErrsBig === 0, `position ${CTX - 1}, ${probe.ms.toFixed(0)}ms`)
  check('gpu errors (big engine)', gpuErrsBig === 0, String(gpuErrsBig))
  await big.close()

  // Arm 2: default ctx, same prompt — the numerics must not notice the table.
  const ref = await newPage(`/model-smoke.html?model=${param}`)
  await ref.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  const refTokens = await ref.evaluate((ids, n) => window.__generate(ids, n), IDS, 16)
  check('tokens identical to default ctx',
    bigTokens.length === refTokens.length && bigTokens.every((t, i) => t === refTokens[i]),
    bigTokens.every((t, i) => t === refTokens[i]) ? '16/16'
      : `diverge at ${bigTokens.findIndex((t, i) => t !== refTokens[i])}`)
} finally {
  await stopHarness()
}
console.log(failed ? '\nctx override BROKEN' : `\nctx=${CTX} runs — the budget was the only ceiling`)
process.exit(failed ? 1 : 0)
