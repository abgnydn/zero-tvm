/**
 * WebLLM head-to-head bench.
 *
 * Loads @mlc-ai/web-llm against the EXACT same Phi-3-mini-q4f16_1 weights
 * Zero-TVM uses, served from our local mirror at /local-weights/*. This is
 * a true apples-to-apples: identical bytes on disk, identical prompt,
 * identical hardware (M2 Pro, 19-core). Only the inference engine differs.
 *
 * The WASM `model_lib` (compiled Relax compute graph) still comes from
 * WebLLM's CDN — it's architecture-matched for Phi-3-mini-4k-instruct.
 */

import * as webllm from '@mlc-ai/web-llm'

const $ = (id: string) => document.getElementById(id)!
function log(msg: string) {
  const el = $('log') as HTMLPreElement
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
  console.log(msg)
}
function setBadge(text: string, loading = false) {
  const b = $('badge')
  b.textContent = text
  b.className = loading ? 'badge loading' : 'badge'
}
function setStats(text: string) {
  $('stats').textContent = text
}

const MODEL_ID = 'Phi-3-mini-4k-instruct-q4f16_1-MLC-local'

const BENCH_PROMPT = 'Write a four-sentence explanation of how photosynthesis works.'
const TARGET_TOKENS = 120
const NUM_RUNS = 3

// Point WebLLM at our already-on-disk Phi-3-mini-q4f16_1 mirror. The
// `resolve/main/` suffix matches HF-hub shape so WebLLM's cleanModelUrl()
// leaves it alone; the vite middleware strips it server-side.
const LOCAL_MODEL_URL = `${location.origin}/local-weights/resolve/main/`
const WASM_URL =
  webllm.modelLibURLPrefix +
  webllm.modelVersion +
  '/Phi-3-mini-4k-instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm'

const appConfig: webllm.AppConfig = {
  useIndexedDBCache: true,
  model_list: [
    {
      model: LOCAL_MODEL_URL,
      model_id: MODEL_ID,
      model_lib: WASM_URL,
      vram_required_MB: 3672.07,
      low_resource_required: false,
      overrides: { context_window_size: 4096 },
    },
  ],
}

async function main() {
  setBadge('Initializing WebLLM...', true)
  log(`[webllm-bench] model: ${MODEL_ID}`)
  log(`[webllm-bench] weights: ${LOCAL_MODEL_URL}`)
  log(`[webllm-bench] wasm:    ${WASM_URL}`)
  log(`[webllm-bench] prompt:  ${BENCH_PROMPT}`)
  log(`[webllm-bench] target tokens: ${TARGET_TOKENS}`)
  log(`[webllm-bench] runs: ${NUM_RUNS} (first is warmup)`)
  log('')

  const engine = await webllm.CreateMLCEngine(MODEL_ID, {
    appConfig,
    initProgressCallback: (p) => {
      setBadge(`Loading ${Math.floor(p.progress * 100)}%...`, true)
      const txt = p.text.replace(/\s+/g, ' ').trim()
      if (txt) log(`[progress] ${txt}`)
    },
  })

  setBadge('Ready', false)
  log('[webllm-bench] engine ready, running benches')
  log('')

  const runs: { tokens: number; seconds: number; tokPerS: number }[] = []
  for (let r = 0; r <= NUM_RUNS; r++) {
    const label = r === 0 ? 'warmup' : `run${r}/${NUM_RUNS}`
    const t0 = performance.now()
    const reply = await engine.chat.completions.create({
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: BENCH_PROMPT },
      ],
      max_tokens: TARGET_TOKENS,
      temperature: 0,
    })
    const seconds = (performance.now() - t0) / 1000
    const usage = reply.usage
    // WebLLM reports completion_tokens; use its own count for decode rate
    const completionTokens = usage?.completion_tokens ?? 0
    const totalTokens = usage?.total_tokens ?? 0
    const tokPerS = completionTokens / seconds
    log(`[bench] ${label}: ${completionTokens} gen tok / ${seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s (total ${totalTokens})`)
    // Log extra info from WebLLM's own stats if present on first non-warmup run
    const extra = (usage as unknown as { extra?: Record<string, number> } | undefined)?.extra
    if (extra && r === 1) {
      log(`[bench]   webllm extra: ${JSON.stringify(extra)}`)
    }
    if (r > 0) runs.push({ tokens: completionTokens, seconds, tokPerS })
  }

  const sorted = runs.map(r => r.tokPerS).sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]
  const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
  const min = sorted[0]
  const max = sorted[sorted.length - 1]
  log('')
  log(`[summary] WebLLM on ${MODEL_ID}`)
  log(`  median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s`)
  log('')
  const ztvm = 42.14 // bench:zt — Zero-TVM decode median (tok/s), updated by `npm run bench`
  log(`[comparison] Zero-TVM (this repo) median: ${ztvm} tok/s on Phi-3-mini q4f16_1 (M2 Pro, 19-core)`)
  const delta = ((median - ztvm) / ztvm) * 100
  log(`  → Zero-TVM is ${delta < 0 ? '+' : '-'}${Math.abs(delta).toFixed(1)}% ${delta < 0 ? 'faster' : 'slower'} than WebLLM on this machine`)

  setStats(`WebLLM ${median.toFixed(1)} tok/s · Zero-TVM ${ztvm} tok/s`)

  ;(window as Window & typeof globalThis & { webllmResult?: unknown }).webllmResult = {
    model: MODEL_ID,
    runs, median, mean, min, max,
  }
}

main().catch(e => {
  log(`[error] ${e?.message || e}`)
  console.error(e)
  setBadge('Error', false)
})
