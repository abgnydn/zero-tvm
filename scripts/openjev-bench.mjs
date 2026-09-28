#!/usr/bin/env node
// OPENJEV-BENCH — the same decision shapes scripts/kev-bench.mjs times on the
// engine, through open-jev (Transformers.js → ONNX Runtime WebGPU) in the
// SAME Chrome with the SAME flags. The page is ~/dev/open-jev-test/bench.html,
// served by that project's vite; this only drives it and prints the result.
//
//   node --experimental-strip-types scripts/openjev-bench.mjs [kev-0.6b|kev-4b|open-jev] [q4f16|q4|fp16]

import { spawn } from 'node:child_process'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'
import { USER_DATA_DIR } from '../tests/e2e/harness.ts'

const model = process.argv[2] ?? 'kev-0.6b'
const dtype = process.argv[3] ?? 'auto'
const DIR = resolve(process.env.HOME, 'dev/open-jev-test')
const PORT = 5191

const vite = spawn(resolve(DIR, 'node_modules/.bin/vite'), ['--port', String(PORT), '--strictPort'], { cwd: DIR, stdio: ['ignore', 'pipe', 'pipe'] })
vite.stderr.on('data', (d) => process.stderr.write(d))
let up = false
vite.on('exit', (c) => { if (!up) { console.error(`vite exited ${c}`); process.exit(1) } })
for (let i = 0; i < 60 && !up; i++) {
  try { up = (await fetch(`http://localhost:${PORT}/bench.html`)).ok } catch { await new Promise((r) => setTimeout(r, 500)) }
}
if (!up) throw new Error('vite did not come up on ' + PORT)
const browser = await puppeteer.launch({
  headless: false,
  userDataDir: USER_DATA_DIR,
  // STOCK flags by default — what a visitor's Chrome runs. With the engine
  // harness's experimental flags (OJ_UNSAFE=1) Chrome exposes subgroup-matrix
  // and ORT's WebGPU provider picks its SubgroupMatrixMatMulNBits shader, which
  // fails validation on this Dawn ("Invalid ShaderModule") — so that
  // configuration measures nothing.
  args: process.env.OJ_UNSAFE === '1'
    ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--enable-dawn-features=allow_unsafe_apis']
    : [],
  defaultViewport: { width: 1100, height: 820 },
  protocolTimeout: 20 * 60 * 1000,
})
try {
  const page = await browser.newPage()
  page.on('console', (m) => { const t = m.text(); if (t.startsWith('[B]')) console.log(t) })
  await page.goto(`http://localhost:${PORT}/bench.html?model=${model}&dtype=${dtype}`)
  await page.waitForFunction(() => window.__phase === 'done' || window.__phase === 'error', { timeout: 20 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') throw new Error(await page.evaluate(() => window.__error))
  const r = await page.evaluate(() => window.__result)
  console.log(`\n${r.model} ${r.dtype} ${r.device}`)
  for (const [n, ms] of Object.entries(r.perCall)) console.log(`  ${String(n).padStart(3)} questions -> ${ms.toFixed(0).padStart(5)} ms  (${(ms / n).toFixed(1)} ms/question)`)
  console.log(`  cold state, 4 questions: ${r.cold.toFixed(0)} ms\n  hot state, one new question: ${r.hot.toFixed(0)} ms`)
} finally {
  await browser.close().catch(() => {})
  vite.kill('SIGTERM')
}
