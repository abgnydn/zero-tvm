#!/usr/bin/env node
// VALIDATE-MODEL — the numerical half of the add-model pipeline: diff the
// browser engine against mlx_lm's own forward pass on the SAME checkpoint.
//
//   1. cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-ref.py \
//        --model <local checkpoint dir> --out /tmp/ref-<param>
//   2. node scripts/validate-model.mjs <param> --ref /tmp/ref-<param>
//
// Boots the e2e harness (vite + real Chrome — headless Chrome's shader-f16 is
// unreliable on macOS), opens model-smoke.html?model=<param>, and compares:
//   - final-position logits: argmax must match, cosine ≥ 0.999, top-5 overlap
//   - greedy continuation: token-exact for at least the first 8 tokens
//     (further drift is reported, not failed — one f16 rounding flip mid-way
//     legitimately forks the sequence)
//
// Exit 0 = the generated spec computes what mlx_lm computes.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv[2]
const refDir = process.argv[process.argv.indexOf('--ref') + 1]
if (!param || !refDir || process.argv.indexOf('--ref') < 0) {
  console.error('usage: node scripts/validate-model.mjs <model-param> --ref <mlx-ref dir>')
  process.exit(2)
}

const meta = JSON.parse(readFileSync(join(refDir, 'meta.json'), 'utf8'))
const refLogits = new Float32Array(readFileSync(join(refDir, 'logits.bin')).buffer.slice(0))
console.log(`ref: ${meta.model} | ${meta.prompt_ids.length} prompt ids | argmax ${meta.argmax} | ${refLogits.length} logits`)

await startHarness()
let failed = 0
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(16)} ${detail}`)
  if (!pass) failed++
}
try {
  const page = await newPage(`/model-smoke.html?model=${param}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(`page boot failed: ${await page.evaluate(() => window.__error)}`)
  }

  const got = new Float32Array(await page.evaluate((ids) => window.__forward(ids), meta.prompt_ids))
  if (got.length !== refLogits.length) throw new Error(`logit length ${got.length} != ref ${refLogits.length}`)

  let dot = 0, na = 0, nb = 0, argmax = 0
  for (let i = 0; i < got.length; i++) {
    dot += got[i] * refLogits[i]; na += got[i] ** 2; nb += refLogits[i] ** 2
    if (got[i] > got[argmax]) argmax = i
  }
  const cosine = dot / Math.sqrt(na * nb)
  const gotTop5 = [...got.keys()].sort((a, b) => got[b] - got[a]).slice(0, 5)
  const refTop5 = meta.top16.slice(0, 5).map(([id]) => id)
  const overlap = gotTop5.filter((id) => refTop5.includes(id)).length

  check('argmax', argmax === meta.argmax, `engine ${argmax} vs mlx ${meta.argmax}`)
  check('cosine', cosine >= 0.999, cosine.toFixed(6))
  check('top5 overlap', overlap >= 4, `${overlap}/5 (engine [${gotTop5}] vs mlx [${refTop5}])`)

  const gen = await page.evaluate((ids, n) => window.__generate(ids, n), meta.prompt_ids, meta.greedy.length)
  let diverge = gen.findIndex((t, i) => t !== meta.greedy[i])
  if (diverge < 0) diverge = gen.length
  check('greedy', diverge >= Math.min(8, meta.greedy.length),
    diverge >= gen.length ? `all ${gen.length} tokens match mlx_lm` : `diverges at token ${diverge}/${gen.length}`)
  console.log(`  engine: [${gen.slice(0, 12).join(', ')}…]`)
  console.log(`  mlx_lm: [${meta.greedy.slice(0, 12).join(', ')}…]  ${JSON.stringify(meta.greedy_text).slice(0, 90)}`)

  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed++
} finally {
  await stopHarness()
}
console.log(failed ? `\n${failed} check(s) FAILED` : '\nmodel validates against mlx_lm')
process.exit(failed ? 1 : 0)
