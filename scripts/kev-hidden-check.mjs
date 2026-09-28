#!/usr/bin/env node
// KEV-HIDDEN-CHECK — bisect forwardHiddenAt on one fixture row: resident
// engine (reuse) vs fresh chunked vs fresh per-token, and each against the
// torch reference's probabilities through the same head. Prints cosine
// between the hidden rows of each arm and the pointer probabilities each
// arm yields.
//
//   node --experimental-strip-types scripts/kev-hidden-check.mjs --ref /tmp/ref-kev [--rec 2] [--row 0]

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const arg = (k, d) => { const i = process.argv.indexOf(k); return i < 0 ? d : process.argv[i + 1] }
const refDir = arg('--ref'); const recI = Number(arg('--rec', '2')) - 1; const rowI = Number(arg('--row', '0'))
const ref = JSON.parse(readFileSync(join(refDir, 'meta.json'), 'utf8'))
const row = ref.records[recI].rows[rowI]
const want = ref.records[recI].probs[rowI]
const positions = [...row.opts, row.decide]
console.log(`rec ${recI + 1} row ${rowI}: ${row.ids.length} ids, readout at ${positions.join(',')} | ref probs ${want.map((v) => v.toFixed(3)).join(' ')}`)

const cos = (a, b) => { let d = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] } return d / Math.sqrt(na * nb) }

await startHarness()
try {
  const page = await newPage('/model-smoke.html?model=kev')
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error', { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') throw new Error(await page.evaluate(() => window.__error))
  // Warm the resident engine's absorbed record with the PREVIOUS record's last
  // row, the way the parity run does, so `resident` reproduces the reuse case.
  if (recI > 0) {
    const prev = ref.records[recI - 1].rows.at(-1)
    await page.evaluate((ids, pos) => window.__hiddenAt(ids, pos), prev.ids, [prev.decide])
  }
  const r = await page.evaluate((ids, pos) => window.__hiddenAtCheck(ids, pos), row.ids, positions)
  const probsVia = await page.evaluate(async (arms) => {
    const { pointerLogits, softmax, loadKevHead } = await import('/src/zero-tvm/kev.ts')
    const { resolveModelBase } = await import('/src/zero-tvm/weight-loader.ts')
    const { specFromSearch } = await import('/src/zero-tvm/model-select.ts')
    const head = await loadKevHead(await resolveModelBase(specFromSearch(location.search)))
    const out = {}
    for (const [name, rows] of Object.entries(arms)) {
      const h = rows.map((v) => Float32Array.from(v))
      out[name] = softmax(pointerLogits(head, h[h.length - 1], h.slice(0, -1)))
    }
    return out
  }, r)
  for (const [name, rows] of Object.entries(r)) {
    const vsFresh = rows.map((v, i) => cos(v, r.freshPerToken[i]).toFixed(5)).join(' ')
    console.log(`${name.padEnd(14)} probs ${probsVia[name].map((v) => v.toFixed(3)).join(' ')} | cos vs freshPerToken per row: ${vsFresh}`)
  }
  const norms = r.freshPerToken.map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)).toFixed(2))
  console.log(`row L2 norms (freshPerToken): ${norms.join(' ')}`)
} finally {
  await stopHarness()
}
