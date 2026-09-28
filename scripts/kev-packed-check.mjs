#!/usr/bin/env node
// KEV-PACKED-CHECK — forwardHiddenPacked against plain `state + branch` rows on
// a fresh engine, for any spec: the packed chunk (attention specs) and the
// GDN-rewind path (hybrids) must both reproduce the branch-by-branch rows,
// and a second call on the same state (resident-state path) must too.
//
//   node --experimental-strip-types scripts/kev-packed-check.mjs qwen35
//   node --experimental-strip-types scripts/kev-packed-check.mjs kev

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const model = process.argv[2] ?? 'kev'
// Token ids only — no tokenizer needed; anything in-vocab works for a
// numerical identity check. 40-token state, three branches of 9 / 13 / 7.
const rng = (seed) => () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed }
const r = rng(7)
const ids = (n) => Array.from({ length: n }, () => 1000 + (r() % 30000))
const state = ids(40)
const branches = [9, 13, 7].map((n) => ({ ids: ids(n), positions: [2, n - 1] }))
const cos = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] } return d / Math.sqrt(na * nb) }
const maxAbs = (a, b) => a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0)

await startHarness()
let failed = 0
try {
  const page = await newPage(`/model-smoke.html?model=${model}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error', { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') throw new Error(await page.evaluate(() => window.__error))
  const t0 = performance.now()
  const { packed1, packed2, plain, reuse } = await page.evaluate((s, b) => window.__packedCheck(s, b), state, branches)
  console.log(`${model}: ${branches.length} branches, ${((performance.now() - t0) / 1000).toFixed(1)}s | last prefill ${JSON.stringify(reuse)}`)
  for (let k = 0; k < branches.length; k++) {
    for (let i = 0; i < branches[k].positions.length; i++) {
      const c1 = cos(packed1[k][i], plain[k][i]), c2 = cos(packed2[k][i], plain[k][i])
      const d1 = maxAbs(packed1[k][i], plain[k][i]), d2 = maxAbs(packed2[k][i], plain[k][i])
      const ok = c1 > 0.9999 && c2 > 0.9999
      if (!ok) failed++
      console.log(`${ok ? 'PASS' : 'FAIL'}  branch ${k} pos ${branches[k].positions[i]}: cos ${c1.toFixed(6)} / ${c2.toFixed(6)}  max|Δ| ${d1.toFixed(4)} / ${d2.toFixed(4)}  (call 1 / call 2 vs plain)`)
    }
  }
  if (await page.evaluate(() => window.__gpuErrs()) > 0) { failed++; console.log('FAIL  uncaptured GPU errors') }
} finally {
  await stopHarness()
}
console.log(failed ? `\n${failed} FAILED` : '\npacked/rewind rows match branch-by-branch rows')
process.exit(failed ? 1 : 0)
