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
// 127.0.0.1, not "localhost": on Linux (Colab/containers) localhost can resolve
// to IPv6 ::1 while Vite binds IPv4, so fetch/Chrome can't reach the dev server.
const HOST = '127.0.0.1'
const BASE = `http://${HOST}:${PORT}`
const N_TOKENS = Number(process.env.BENCH_TOKENS ?? 128)
const N_RUNS = Number(process.env.BENCH_RUNS ?? 5)
// BENCH_QUERY="?vec4=1" A/Bs a shader-variant flag: appended to /zero-tvm.html,
// skips the WebLLM half and does NOT touch results.json (A/B runs are for
// comparing flags, not for updating the headline numbers).
const QUERY = process.env.BENCH_QUERY ?? ''
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

function attachDiagnostics(page, label) {
  page.on('pageerror', (e) => console.error(`[${label}:pageerror] ${e.message}`))
  page.on('console', (m) => {
    const t = m.text()
    if (
      m.type() === 'error' ||
      /\[bench\]|median=|\[summary\]|webgpu|adapter|gpu|f16|weight|shader|error|fail/i.test(t)
    )
      console.log(`[${label}] ${t}`)
  })
}

async function bootReady(page, path) {
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 60_000 })

  // Up-front WebGPU sanity: is an adapter (with shader-f16) actually there?
  const gpu = await page
    .evaluate(async () => {
      if (!('gpu' in navigator)) return { gpu: false }
      const probe = async (opts) => {
        try {
          const a = await navigator.gpu.requestAdapter(opts)
          if (!a) return { ok: false }
          return {
            ok: true,
            f16: !!a.features?.has('shader-f16'),
            info: a.info ? a.info.vendor + '/' + (a.info.description || a.info.architecture) : null,
          }
        } catch (e) {
          return { err: String(e) }
        }
      }
      return {
        gpu: true,
        default: await probe(),
        highPerf: await probe({ powerPreference: 'high-performance' }),
      }
    })
    .catch((e) => ({ evalError: String(e) }))
  console.log(`[webgpu] ${JSON.stringify(gpu)}`)

  // Click the download gate if present; cached-skip path won't show it.
  try {
    await page.waitForSelector('#start-btn', { timeout: 8_000 })
    await page.click('#start-btn')
  } catch {
    /* auto-boot */
  }

  // Poll the badge; log every transition, bail fast on an error state.
  const start = Date.now()
  let last = ''
  while (Date.now() - start < READY_MS) {
    const badge = await page
      .evaluate(() => document.getElementById('badge')?.textContent || '')
      .catch(() => '')
    if (badge && badge !== last) {
      console.log(`[boot] badge: ${badge}`)
      last = badge
    }
    if (badge === 'Ready') return
    if (/error|fail/i.test(badge)) throw new Error(`boot failed — badge="${badge}"`)
    await new Promise((r) => setTimeout(r, 1500))
  }
  throw new Error(`boot timed out — last badge="${last}"`)
}

let viteLog = ''
const vite = spawn(
  resolve(ROOT, 'node_modules/.bin/vite'),
  ['--port', String(PORT), '--strictPort', '--host', HOST],
  { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] },
)
vite.stdout.on('data', (b) => { viteLog += b })
vite.stderr.on('data', (b) => { viteLog += b; process.stderr.write(`[vite] ${b}`) })

let browser
let ok = false
try {
  await waitForUrl(`${BASE}/zero-tvm.html`, 120_000).catch((e) => {
    console.error('\n[vite never became reachable] recent vite output:\n' + viteLog.slice(-2000))
    throw e
  })
  // Headless Linux + a real NVIDIA GPU (container / Colab) needs ANGLE-over-
  // Vulkan with no display surface — per Chrome's headless-GPU docs
  // (developer.chrome.com/docs/web-platform/webgpu/colab-headless). The desktop
  // path (headless:false, e.g. macOS/Metal) keeps the simpler flag set.
  const gpuArgs =
    HEADLESS === 'new'
      ? ['--use-angle=vulkan', '--enable-features=Vulkan', '--disable-vulkan-surface', '--ignore-gpu-blocklist']
      : ['--enable-features=Vulkan']
  browser = await puppeteer.launch({
    headless: HEADLESS,
    args: [
      '--no-sandbox', // required when running as root in a container
      '--enable-unsafe-webgpu',
      // disable_adapter_blocklist: Dawn (Chrome's WebGPU) blocklists some NVIDIA
      // drivers by default, so requestAdapter() returns null on a real T4.
      '--enable-dawn-features=allow_unsafe_apis,disable_adapter_blocklist',
      ...gpuArgs,
    ],
    protocolTimeout: READY_MS,
  })

  // 1) Zero-TVM decode median via the engine's built-in bench().
  const ztPage = await browser.newPage()
  attachDiagnostics(ztPage, 'zt')
  await bootReady(ztPage, `/zero-tvm.html${QUERY}`)
  const zt = await ztPage.evaluate((n, runs) => window.bench(n, runs), N_TOKENS, N_RUNS)
  console.log(`→ Zero-TVM median: ${zt.median.toFixed(2)} tok/s${QUERY ? `  (${QUERY})` : ''}`)

  if (QUERY) {
    // A/B mode: one engine config, no WebLLM half, no results.json.
    console.log(`[a/b] runs: ${zt.runs?.map((x) => x.tokPerS.toFixed(2)).join(', ')}`)
    ok = true
  } else {
    // 2) WebLLM decode median — webllm-bench.html runs on start and sets window.webllmResult.
    const wlPage = await browser.newPage()
    attachDiagnostics(wlPage, 'wl')
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
  }
} finally {
  await browser?.close().catch(() => {})
  vite.kill('SIGTERM')
}

if (ok && !QUERY) spawnSync('node', [resolve(ROOT, 'bench/sync-docs.mjs'), '--write'], { cwd: ROOT, stdio: 'inherit' })
