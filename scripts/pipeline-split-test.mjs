#!/usr/bin/env node
// PIPELINE-SPLIT-TEST — does the model still compute itself when its layers
// are cut in half?
//
// Builds THREE engines over one set of weights on one device: the whole model,
// and two pipeline stages [0, k) and [k, layers). The stages are driven by
// hand — token id into A, residual into B, token out of B — and the generated
// sequence must match the whole model's, token for token.
//
// One device on purpose. The hand-off here is a memcpy where a real split
// would be a DataChannel, so a mismatch can only come from the SPLIT itself:
// each stage keeping its own KV cache and GDN state, and every stage
// re-normalising from the bare residual. Get that wrong and the sequences
// diverge within a couple of tokens; get it right and the transport is
// plumbing.
//
//   node scripts/pipeline-split-test.mjs [--ref <mlx-ref dir>] [model-param]
//
// Give it a scripts/mlx-ref.py dump and it uses that prompt (and checks the
// answer against mlx_lm's). Without one it falls back to arbitrary token ids,
// which still proves agreement but on a degenerate sequence — a real prompt is
// far more sensitive, because a real answer walks through many distinct
// argmaxes instead of sitting in a repeating attractor.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const args = process.argv.slice(2)
const refIdx = args.indexOf('--ref')
const refDir = refIdx >= 0 ? args[refIdx + 1] : null
const param = args.filter((a, i) => !a.startsWith('--') && !(refIdx >= 0 && i === refIdx + 1))[0] ?? 'llama32'
const spec = specForParam(param)
const ref = refDir ? JSON.parse(readFileSync(join(refDir, 'meta.json'), 'utf8')) : null
const PROMPT = ref ? ref.prompt_ids : [1, 2, 3, 4]
const TOKENS = ref ? Math.min(8, ref.greedy.length) : 8

let failed = false
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(22)} ${detail}`)
  if (!pass) failed = true
}

await startHarness()
try {
  const page = await newPage(`/model-smoke.html?model=${param}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(`page boot failed: ${await page.evaluate(() => window.__error)}`)
  }
  console.log(`${spec.id}: ${spec.layers} layers, d=${spec.d} (hand-off is ${spec.d * 2} bytes/token)`)

  const whole = await page.evaluate((ids, n) => window.__generate(ids, n), PROMPT, TOKENS)
  console.log(`whole model: [${whole.join(', ')}]`)
  if (ref) {
    const want = ref.greedy.slice(0, whole.length)
    check('whole vs mlx_lm', whole.every((t, i) => t === want[i]),
      `${JSON.stringify(ref.greedy_text)} — the sequence the split must reproduce`)
  }

  for (const k of [1, Math.floor(spec.layers / 2), spec.layers - 1]) {
    const split = await page.evaluate((ids, kk, n) => window.__splitCheck(ids, kk, n), PROMPT, k, TOKENS)
    // generate() HALTS on a stop id without emitting it; the hand-driven split
    // loop runs a fixed count, so it emits that stop id as one extra token.
    // Reproducing every token the whole model emitted and then stopping where
    // it stopped IS agreement.
    const prefixSame = whole.every((t, i) => t === split[i])
    const extra = split.slice(whole.length)
    const same = prefixSame && (extra.length === 0 || (extra.length === 1 && spec.stops.includes(extra[0])))
    check(`split at layer ${k}`, same, same
      ? `${whole.length} tokens identical${extra.length ? ` (then the stop id ${extra[0]}, where the whole model halted)` : ''}`
      : `[${split.join(', ')}] vs [${whole.join(', ')}]`)
  }

  // The arrangement a real split uses: each stage loads ONLY its own layers,
  // which is the whole reason to split — the stage's VRAM is its share.
  if (spec.weightFormat === 'mlx-safetensors') {
    const k = Math.floor(spec.layers / 2)
    const r = await page.evaluate((ids, kk, n) => window.__splitLoadCheck(ids, kk, n), PROMPT, k, TOKENS)
    const prefixSame = whole.every((t, i) => t === r.tokens[i])
    const extra = r.tokens.slice(whole.length)
    const same = prefixSame && (extra.length === 0 || (extra.length === 1 && spec.stops.includes(extra[0])))
    const gb = (b) => (b / 1e9).toFixed(2)
    check('partial load matches', same, same ? `${whole.length} tokens identical` : `[${r.tokens}] vs [${whole}]`)
    const wholeBytes = await page.evaluate(() => window.__wholeBytes())
    // Neither stage may hold the whole model. The SUM can still exceed it: a
    // tied lm_head IS the embedding table, so a tied model carries that table
    // on both ends (llama32 does; Qwen3.6-35B is untied and does not).
    check('stages hold a share', r.bytesA < wholeBytes && r.bytesB < wholeBytes,
      `whole ${gb(wholeBytes)} GB · stage A ${gb(r.bytesA)} (layers 0-${k}) + stage B ${gb(r.bytesB)} `
      + `(layers ${k}-${spec.layers}) = ${gb(r.bytesA + r.bytesB)} GB`
      + (spec.tiedEmbeddings ? ' — tied embedding table sits on both ends' : ''))
  }

  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  await stopHarness()
}
console.log(failed ? '\npipeline split FAILED' : '\npipeline split correct — the halves compute what the whole does')
process.exit(failed ? 1 : 0)
