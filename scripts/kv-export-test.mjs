#!/usr/bin/env node
// KV-EXPORT-TEST — does an exportKV/importKV round trip preserve the session?
//
//   node scripts/kv-export-test.mjs [llama32|qwen35] [--int8]
//
// kv-pool-test.mjs covers the integrated pool path (f16, through kv-pool.ts).
// This covers the RAW snapshot contract directly: the exported shape (layer
// bytes, per-row scales in int8 mode, GDN buffers on hybrids) and the
// continuation after import into a FRESH engine, against an uninterrupted
// control. A serialization bug (wrong scale stride, dropped GDN slice) shows
// up as fluent wrong text here instead of in a user's restored session.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'llama32'
const useInt8 = process.argv.includes('--int8')
const PROMPT = Array.from({ length: 60 }, (_, i) => 3000 + ((i * 41) % 700))

let failed = false
const check = (name, ok, detail = '') => {
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
    (p) => window.__kvRoundTrip(p, 8, 16, useInt8), PROMPT)
  check('export produced a snapshot', r.exported === true)
  if (!r.exported) throw new Error('exportKV returned null — engine refused a state it should attach')
  check('import accepted', r.imported === true)
  check('snapshot covers the prompt+generated prefix',
    r.shape.tokens >= PROMPT.length + 8 - 1 && r.shape.tokens <= PROMPT.length + 8 + 1,
    `${r.shape.tokens} tokens`)
  check('every layer present', r.shape.layers.length > 0 && r.shape.layers.every((b) => b > 0),
    `${r.shape.layers.length} layers`)
  if (useInt8) {
    check('int8 scales present for every layer',
      r.shape.scales.length === r.shape.layers.length && r.shape.scales.every((b) => b > 0),
      `${r.shape.scales.length} scale buffers`)
  } else {
    check('f16 snapshot carries no scales', r.shape.scales.length === 0)
  }
  const same = r.cont.length === r.direct.length && r.cont.every((t, i) => t === r.direct[i])
  check('restored continuation IDENTICAL to control', same,
    same ? `${r.direct.length}/${r.direct.length}`
      : `diverge at ${r.cont.findIndex((t, i) => t !== r.direct[i])}`)
  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
} finally {
  await stopHarness()
}
console.log(failed ? '\nKV export round trip BROKEN' : '\nexport/import preserves the session')
process.exit(failed ? 1 : 0)
