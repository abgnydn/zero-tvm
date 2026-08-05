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
//
// BENCH_QUERY="?model=qwen3" is the cross-ENGINE A/B: it runs BOTH halves —
// zero-tvm.html with the query AND webllm-bench.html with the same model
// param — and prints both medians + the gap, but still does NOT touch
// results.json (that file is the Phi-3 headline artifact; Qwen numbers live
// in bench/results/*.json and BENCH.md's Qwen sections).
const QUERY = process.env.BENCH_QUERY ?? ''
const AB_MODEL = QUERY ? new URLSearchParams(QUERY.replace(/^\?/, '')).get('model') : null
// BENCH_BASELINE picks which engine runs as the second half.
//   webllm (default) — @mlc-ai/web-llm on the SAME q4f16_1 bytes we load.
//                      A true runtime-only A/B. Unchanged behaviour.
//   wllama           — llama.cpp WASM+WebGPU on GGUF Q4_K_M. NOT the same
//                      bytes: different quantization of the same base model,
//                      so the gap it prints is runtime + quantization. Never
//                      writes results.json (that file's `webllmDecode` field
//                      feeds bench/sync-docs.mjs into the published numbers,
//                      and a wllama figure must not land there unlabelled).
//   tjs              — @huggingface/transformers (ONNX Runtime Web, WebGPU EP)
//                      on ONNX q4f16. Also NOT the same bytes, though the
//                      confound is smaller than GGUF's: same 4-bit width and
//                      same fp16 activations, different block layout. Same
//                      results.json embargo as wllama, for the same reason.
const BASELINES = {
  webllm: { page: '/webllm-bench.html', name: 'WebLLM', global: 'webllmResult', label: 'wl' },
  wllama: { page: '/wllama-bench.html', name: 'wllama (llama.cpp WebGPU, GGUF)', global: 'wllamaResult', label: 'wllama' },
  tjs:    { page: '/tjs-bench.html',    name: 'transformers.js (ORT Web WebGPU, ONNX q4f16)', global: 'tjsResult', label: 'tjs' },
}
const BASELINE = BASELINES[process.env.BENCH_BASELINE] ? process.env.BENCH_BASELINE : 'webllm'
const BASE_PAGE = BASELINES[BASELINE].page
const BASE_NAME = BASELINES[BASELINE].name
const BASE_GLOBAL = BASELINES[BASELINE].global
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
  // visible: true matters — #start-btn lives inside a closed <dialog> until
  // chat.ts's async cache check calls showModal(), so an existence-only wait
  // clicks a hidden node and the click is silently lost.
  try {
    await page.waitForSelector('#start-btn', { visible: true, timeout: 8_000 })
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

/** Boot the selected baseline page and read its window.<engine>Result. */
async function runBaseline(browser, query) {
  const page = await browser.newPage()
  attachDiagnostics(page, BASELINES[BASELINE].label)
  await bootReady(page, `${BASE_PAGE}${query}`)
  const res = await page
    .waitForFunction((k) => window[k], { timeout: READY_MS, polling: 1000 }, BASE_GLOBAL)
    .then((h) => h.jsonValue())
  if (BASELINE === 'tjs') {
    // Same class of check as the wllama half. `device: 'webgpu'` is a request,
    // not a guarantee — ORT Web falls back to the WASM EP silently, and the
    // only symptom is a slow number. The page proves it by counting GPU queue
    // submits during a real generation, not by reading navigator.gpu.
    console.log(`[tjs] transformers.js=${res.transformersJs} backend=${res.backend} dtype=${res.dtype} adapter=${res.adapterInfo ?? 'none'}`)
    console.log(`[tjs] gpu: devices=${res.gpuCounters?.requestDeviceCalls} pipelines=${res.gpuCounters?.computePipelines} submits=${res.gpuCounters?.queueSubmits} · weights=${res.weightsFrom} · repo=${res.repo}`)
    if (!res.webgpuProven) console.log('[tjs] *** WEBGPU NOT PROVEN — do not publish this as a WebGPU baseline ***')
    if (res.mode === 'ttft-sweep') {
      // Sweep mode has no median; it produces the curve instead.
      for (const p of res.points ?? [])
        console.log(`[tjs:ttft] prompt=${p.promptTokens} ttft=${p.ttftMs.toFixed(0)}ms (${p.ttftMsRuns.map((x) => x.toFixed(0)).join(', ')}) decode=${p.decodeTokPerS.toFixed(2)} tok/s`)
      return res
    }
    if (!res.contentAccountingOk)
      console.log(`[tjs] *** TOKEN ACCOUNTING BROKEN — runs: ${res.runs?.map((r) => `${r.tokens}tok`).join(', ')} — median is meaningless, do not publish it ***`)
  }
  if (BASELINE === 'wllama') {
    // A wllama run that silently fell back to CPU/WASM is not a baseline at
    // all — say so loudly rather than letting the pair get quoted.
    console.log(`[wllama] backend=${res.backend} layers_on_gpu=${res.layersOnGpu}/${res.layersTotal} adapter=${res.adapterInfo ?? 'none'}`)
    console.log(`[wllama] threads=${res.nThreads} multithread=${res.multithread} gguf=${res.gguf} (${res.quant})`)
    if (!res.webgpuProven) console.log('[wllama] *** WEBGPU NOT PROVEN — this is a CPU/WASM number, do not publish it ***')
    // A run that counted 0 content tokens (thinking-mode tokens arriving as
    // reasoning_content) still yields a numeric median, so check explicitly.
    if (!res.contentAccountingOk)
      console.log(`[wllama] *** TOKEN ACCOUNTING BROKEN — runs: ${res.runs?.map((r) => `${r.tokens}tok/${r.reasoningChunks}reasoning`).join(', ')} — median is meaningless, do not publish it ***`)
  }
  return res
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

  if (QUERY && AB_MODEL) {
    // Cross-engine A/B (?model=…): run the baseline half too, same model, same
    // session — but never write results.json (Phi-3 headline artifact).
    console.log(`[a/b] zt runs: ${zt.runs?.map((x) => x.tokPerS.toFixed(2)).join(', ')}`)
    const wl = await runBaseline(browser, `?model=${AB_MODEL}`)
    console.log(`→ ${BASE_NAME} median: ${wl.median.toFixed(2)} tok/s  (model=${AB_MODEL})`)
    console.log(`[a/b] ${BASELINES[BASELINE].label} runs: ${wl.runs?.map((x) => x.tokPerS.toFixed(2)).join(', ')}`)
    const gap = ((zt.median - wl.median) / wl.median) * 100
    console.log(
      `\n[a/b] model=${AB_MODEL} same-session pair: ` +
      `Zero-TVM ${zt.median.toFixed(2)} vs ${BASE_NAME} ${wl.median.toFixed(2)} tok/s — ` +
      `Zero-TVM is ${gap >= 0 ? '+' : ''}${gap.toFixed(1)}% vs ${BASE_NAME} ` +
      `(results.json NOT written)`,
    )
    if (wl.caveat) console.log(`[a/b] CAVEAT: ${wl.caveat}`)
    ok = true
  } else if (QUERY) {
    // A/B mode: one engine config, no WebLLM half, no results.json.
    console.log(`[a/b] runs: ${zt.runs?.map((x) => x.tokPerS.toFixed(2)).join(', ')}`)
    ok = true
  } else {
    // 2) Baseline decode median — the page runs on start and sets its own global.
    const wl = await runBaseline(browser, '')
    console.log(`→ ${BASE_NAME} median: ${wl.median.toFixed(2)} tok/s`)

    if (BASELINE !== 'webllm') {
      // Phi-3 headline pair, but against a different-quantization baseline:
      // print it and stop. results.json is the same-bytes WebLLM artifact that
      // sync-docs.mjs publishes; a different-quantization number cannot go in
      // it. Applies to wllama (GGUF) and tjs (ONNX q4f16) alike.
      const gap = ((zt.median - wl.median) / wl.median) * 100
      console.log(
        `\n[pair] same-session: Zero-TVM ${zt.median.toFixed(2)} vs ${BASE_NAME} ` +
        `${wl.median.toFixed(2)} tok/s — Zero-TVM is ${gap >= 0 ? '+' : ''}${gap.toFixed(1)}% ` +
        `(results.json NOT written — BENCH_BASELINE=${BASELINE})`,
      )
      console.log(`[pair] CAVEAT: ${wl.caveat}`)
    } else {
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
    }
    ok = true
  }
} finally {
  await browser?.close().catch(() => {})
  vite.kill('SIGTERM')
}

// Docs sync reads bench/results.json, which only the same-bytes WebLLM run writes.
if (ok && !QUERY && BASELINE === 'webllm')
  spawnSync('node', [resolve(ROOT, 'bench/sync-docs.mjs'), '--write'], { cwd: ROOT, stdio: 'inherit' })
