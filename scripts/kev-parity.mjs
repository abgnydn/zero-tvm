#!/usr/bin/env node
// KEV-PARITY — the engine's typed decisions against kev's own torch forward on
// the SAME merged weights (scripts/kev-ref.py), record by record.
//
//   1. cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/kev-ref.py \
//        --merged .weights-local/kev-0.6b-merged-bf16 --records tests/fixtures/kev-records.json \
//        --kev-src <kev checkout> --out /tmp/ref-kev
//   2. node scripts/kev-parity.mjs --ref /tmp/ref-kev [--model kev4b]
//
// Three things are compared, in the order they can fail:
//   - token ids of every row: MUST be identical (the encoder is a port, and a
//     tokenizer drift here would make every later number meaningless)
//   - readout positions (decide / </opt>): identical
//   - probabilities: argmax equal on every question; max |Δp| reported and
//     gated at 0.05 — the backbone here is the 4-bit affine quantization of
//     the f32 reference, so this is a fidelity bound, not an equality
//
// Exit 0 = the engine decides what kev decides.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const refDir = process.argv[process.argv.indexOf('--ref') + 1]
const model = process.argv.indexOf('--model') < 0 ? 'kev' : process.argv[process.argv.indexOf('--model') + 1]
if (process.argv.indexOf('--ref') < 0 || !refDir) {
  console.error('usage: node scripts/kev-parity.mjs --ref <kev-ref dir> [--model kev|kev4b]')
  process.exit(2)
}
const PROB_TOL = Number(process.env.KEV_PROB_TOL ?? '0.05')
const ref = JSON.parse(readFileSync(join(refDir, 'meta.json'), 'utf8'))
const records = JSON.parse(readFileSync(new URL('../tests/fixtures/kev-records.json', import.meta.url), 'utf8'))
if (ref.records.length !== records.length) throw new Error(`ref has ${ref.records.length} records, fixtures ${records.length}`)
console.log(`ref: ${ref.merged} | head T=${ref.temperature} | kev@${ref.kev_commit.slice(0, 8)} | ${records.length} records`)

await startHarness()
let failed = 0
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(22)} ${detail}`)
  if (!pass) failed++
}
try {
  const page = await newPage(`/model-smoke.html?model=${model}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(`page boot failed: ${await page.evaluate(() => window.__error)}`)
  }

  let maxDp = 0
  let argmaxMiss = 0
  let questions = 0
  const t0 = performance.now()
  for (let i = 0; i < records.length; i++) {
    const got = await page.evaluate((rec) => window.__kevRecord(rec), records[i])
    const want = ref.records[i]
    const idsOk = got.rows.every((r, k) => JSON.stringify(r.ids) === JSON.stringify(want.rows[k].ids))
    const posOk = got.rows.every((r, k) => r.decide === want.rows[k].decide && JSON.stringify(r.opts) === JSON.stringify(want.rows[k].opts))
    check(`rec ${i + 1} ids`, idsOk, idsOk ? `${got.rows.length} rows, state ${got.state.length} tok`
      : `MISMATCH row ${got.rows.findIndex((r, k) => JSON.stringify(r.ids) !== JSON.stringify(want.rows[k].ids))}`)
    check(`rec ${i + 1} readout`, posOk, posOk ? 'decide/opts equal' : 'decide or option positions differ')
    got.probs.forEach((p, k) => {
      questions++
      const q = want.probs[k]
      const am = p.indexOf(Math.max(...p))
      const ar = q.indexOf(Math.max(...q))
      const dp = Math.max(...p.map((v, j) => Math.abs(v - q[j])))
      maxDp = Math.max(maxDp, dp)
      if (am !== ar) argmaxMiss++
      console.log(`      q${k + 1} engine ${p.map((v) => v.toFixed(3)).join(' ')} | ref ${q.map((v) => v.toFixed(3)).join(' ')} | Δmax ${dp.toFixed(4)}${am !== ar ? '  ARGMAX DIFFERS' : ''}`)
    })
  }
  const ms = performance.now() - t0
  check('argmax', argmaxMiss === 0, `${questions - argmaxMiss}/${questions} questions agree`)
  check('max |Δp|', maxDp <= PROB_TOL, `${maxDp.toFixed(4)} (tol ${PROB_TOL})`)
  console.log(`${questions} questions in ${(ms / 1000).toFixed(1)}s incl. page evaluate overhead`)
  if (await page.evaluate(() => window.__gpuErrs()) > 0) check('gpu errors', false, 'uncaptured GPU errors during the run')
} finally {
  await stopHarness()
}
console.log(failed ? `\n${failed} check(s) FAILED` : '\nkev parity: engine decides what kev decides')
process.exit(failed ? 1 : 0)
