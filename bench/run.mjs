// Auto-bench: measure Zero-TVM vs WebLLM decode tok/s on a real GPU, write
// bench/results.json, then sync the docs from it.
//
//   npm run bench                 # measure + write results.json + sync docs
//   BENCH_HW="M2 Pro, 19-core" npm run bench
//
// This drives the *real* browser engine, so it needs a machine with WebGPU (a
// dev Mac) and ~2 GB of Phi-3 weights (first run downloads them; cached after).
// It CANNOT run in a GPU-less CI sandbox — there it's a no-op-by-failure.
//
// Same boot flow as the e2e tests (tests/e2e/harness.ts): spawn a Vite dev
// server, launch Chrome with the WebGPU flags, click through the download
// gate, then call the engine's own window.bench() (chat.ts) and read
// window.webllmResult (webllm-bench/main.ts).

import { spawn, spawnSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import puppeteer from 'puppeteer'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = 5191
const BASE = `http://localhost:${PORT}`
const N_TOKENS = Number(process.env.BENCH_TOKENS ?? 128)
const N_RUNS = Number(process.env.BENCH_RUNS ?? 5)
const HARDWARE = process.env.BENCH_HW ?? 'unknown GPU'
const READY_MS = 12 * 60 * 1000 // first run downloads ~2 GB
// headless:false matches the e2e harness (most reliable on a desktop GPU under
// a display/Xvfb). In a container on a real GPU, BENCH_HEADLESS=new skips Xvfb.
const HEADLESS =
  process.env.BENCH_HEADLESS === 'new' ? 'new' : process.env.BENCH_HEADLESS === 'true'

async function waitForUrl(url, timeoutMs) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      if ((await fetch(url)).ok) return
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`timed out waiting for ${url}`)
}

async function bootReady(page, path) {
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  // Click the download gate if present; cached-skip path won't show it.
  try {
    await page.waitForSelector('#start-btn', { timeout: 8_000 })
    await page.click('#start-btn')
  } catch {
    /* auto-boot */
  }
  await page.waitForFunction(() => document.getElementById('badge')?.textContent === 'Ready', {
    timeout: READY_MS,
    polling: 500,
  })
}

const vite = spawn(
  resolve(ROOT, 'node_modules/.bin/vite'),
  ['--port', String(PORT), '--strictPort'],
  { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
)
vite.stderr.on('data', (b) => process.stderr.write(`[vite] ${b}`))

let browser
let ok = false
try {
  await waitForUrl(`${BASE}/zero-tvm.html`, 30_000)
  // Headless Linux + a real NVIDIA GPU (container / Colab) needs ANGLE-over-
  // Vulkan with no display surface — per Chrome's headless-GPU docs
  // (developer.chrome.com/docs/web-platform/webgpu/colab-headless). The desktop
  // path (headless:false, e.g. macOS/Metal) keeps the simpler flag set.
  const gpuArgs =
    HEADLESS === 'new'
      ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface']
      : ['--enable-features=Vulkan']
  browser = await puppeteer.launch({
    headless: HEADLESS,
    args: [
      '--no-sandbox', // required when running as root in a container
      '--enable-unsafe-webgpu',
      '--enable-dawn-features=allow_unsafe_apis',
      ...gpuArgs,
    ],
    protocolTimeout: READY_MS,
  })

  // 1) Zero-TVM decode median via the engine's built-in bench().
  const ztPage = await browser.newPage()
  ztPage.on('console', (m) => {
    if (/^\[bench\]/.test(m.text())) console.log(m.text())
  })
  await bootReady(ztPage, '/zero-tvm.html')
  const zt = await ztPage.evaluate((n, runs) => window.bench(n, runs), N_TOKENS, N_RUNS)
  console.log(`→ Zero-TVM median: ${zt.median.toFixed(2)} tok/s`)

  // 2) WebLLM decode median — webllm-bench.html runs on start and sets window.webllmResult.
  const wlPage = await browser.newPage()
  wlPage.on('console', (m) => {
    if (/median=|\[summary\]/.test(m.text())) console.log(m.text())
  })
  await bootReady(wlPage, '/webllm-bench.html')
  const wl = await wlPage
    .waitForFunction(() => window.webllmResult, { timeout: READY_MS, polling: 1000 })
    .then((h) => h.jsonValue())
  console.log(`→ WebLLM median: ${wl.median.toFixed(2)} tok/s`)

  const results = {
    ztDecode: +zt.median.toFixed(2),
    ztRuns: zt.runs?.map((x) => +x.tokPerS.toFixed(2)),
    webllmDecode: +wl.median.toFixed(2),
    hardware: HARDWARE,
    date: new Date().toISOString().slice(0, 10),
    nTokens: N_TOKENS,
    nRuns: N_RUNS,
  }
  writeFileSync(resolve(ROOT, 'bench/results.json'), JSON.stringify(results, null, 2) + '\n')
  console.log('\nwrote bench/results.json')
  ok = true
} finally {
  await browser?.close().catch(() => {})
  vite.kill('SIGTERM')
}

if (ok) spawnSync('node', [resolve(ROOT, 'bench/sync-docs.mjs'), '--write'], { cwd: ROOT, stdio: 'inherit' })
