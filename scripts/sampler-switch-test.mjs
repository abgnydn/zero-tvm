#!/usr/bin/env node
// SAMPLER-SWITCH-TEST — does setSampling() actually switch the sampler?
//
//   node scripts/sampler-switch-test.mjs [llama32]
//
// The sampler KERNEL is pinned by tests/kernels/run.mjs, but the runtime
// uniform rewrite (engine.setSampling) was never called in any test — a
// wrong byte offset there produces fluent wrong text, never an error. Three
// fresh engines on one prompt: greedy twice (determinism control), sampled
// twice at one seed (reproducibility — draws are a pure function of
// seed+position), and greedy-after-sampling (null must restore greedy
// exactly). The sampled-vs-greedy difference is asserted at temperature 1.5
// over 8 tokens; a model that draws argmax 8 times running is broken in a
// way this would miss, but that failure is loud elsewhere (quality-eval).

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'llama32'
const PROMPT = Array.from({ length: 48 }, (_, i) => 3000 + ((i * 41) % 700))

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
  const r = await page.evaluate((p) => window.__samplerSwitch(p, 8), PROMPT)
  const eq = (a, b) => a.length === b.length && a.every((t, i) => t === b[i])
  check('greedy is deterministic', eq(r.g1, r.g2), `${r.g1.length} tokens`)
  check('same seed reproduces sampled draws', eq(r.s1, r.s2),
    eq(r.s1, r.s2) ? `${r.s1.length} tokens` : 'seed had no effect — uniform rewrite suspect')
  check('sampling engages (differs from greedy)', !eq(r.s1, r.g1),
    eq(r.s1, r.g1) ? 'temperature 1.5 drew argmax 8/8 — sampler may be stuck greedy' : 'differs')
  check('setSampling(null) restores greedy exactly', eq(r.g3, r.g1),
    eq(r.g3, r.g1) ? `${r.g3.length} tokens` : 'greedy-after-sample diverges — null did not restore the uniform')
  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
} finally {
  await stopHarness()
}
console.log(failed ? '\nsampler switching BROKEN' : '\nsetSampling switches both ways')
process.exit(failed ? 1 : 0)
