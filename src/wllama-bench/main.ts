/**
 * wllama head-to-head bench — llama.cpp (WASM + WebGPU backend) as a THIRD
 * baseline next to WebLLM.
 *
 * ┌───────────────────────────────────────────────────────────────────────┐
 * │ ⚠ THIS IS NOT A SAME-BYTES COMPARISON.                                │
 * │                                                                       │
 * │ The WebLLM head-to-head (src/webllm-bench/main.ts) feeds both engines │
 * │ the EXACT same q4f16_1 tensors off the same mirror — only the runtime │
 * │ differs. wllama cannot do that: llama.cpp reads GGUF, so this page    │
 * │ loads Q4_K_M/Q4 GGUF builds of the same BASE models. Different        │
 * │ quantization format, different group sizes, different rounding, and   │
 * │ (for Phi-3) a different file size on disk.                            │
 * │                                                                       │
 * │ So any number produced here is a RUNTIME+QUANTIZATION comparison, not │
 * │ a runtime comparison. State that wherever it is published. A true     │
 * │ same-bytes race against llama.cpp needs GGUF support in Zero-TVM,     │
 * │ which is a separate project.                                          │
 * └───────────────────────────────────────────────────────────────────────┘
 *
 * Why bother: llama.cpp's WebGPU backend (ngxson/wllama PR #215, shipped in
 * v3.1) is the strong browser baseline — LlamaWeb (arXiv 2605.20706) reports
 * it 45-69% faster decode than WebLLM across 16 devices. Every Zero-TVM
 * number published so far was measured against WebLLM only.
 *
 * URL params:
 *   ?model=phi3|qwen3|qwen35   which GGUF to load (default phi3)
 *   ?ngl=<n>                   override n_gpu_layers. `?ngl=0` is the
 *                              CPU-only control: run it and compare the
 *                              backend banner + tok/s to prove the default
 *                              run really is on the GPU.
 *
 * Weights come from the local dev mirror at /local-gguf/* (vite.config.ts,
 * localGgufPlugin) so the bench is offline and doesn't re-pull ~7.6 GB from
 * HuggingFace.
 */

import { Wllama } from '@wllama/wllama/esm/index.js'
import type { ChatCompletionChunk, ResultTimings, WllamaLogger } from '@wllama/wllama/esm/index.js'
import wllamaWasmUrl from '@wllama/wllama/esm/wasm/wllama.wasm?url'

const params = new URLSearchParams(location.search)

const GGUF = {
  phi3: {
    file: 'phi3-q4.gguf',
    label: 'Phi-3-mini-4k-instruct',
    quant: 'Q4 · microsoft/Phi-3-mini-4k-instruct-gguf',
  },
  qwen3: {
    file: 'qwen3-4b-q4km.gguf',
    label: 'Qwen3-4B',
    quant: 'Q4_K_M · Qwen/Qwen3-4B-GGUF',
  },
  qwen35: {
    file: 'qwen35-4b-q4km.gguf',
    label: 'Qwen3.5-4B',
    quant: 'Q4_K_M · unsloth/Qwen3.5-4B-GGUF',
  },
} as const

const MODEL_KEY = (params.get('model') ?? 'phi3') as keyof typeof GGUF
const SEL = GGUF[MODEL_KEY] ?? GGUF.phi3
const MODEL_ID = `${SEL.label}-GGUF-local`
const GGUF_URL = `${location.origin}/local-gguf/${SEL.file}`

// Same protocol as the other two halves (src/webllm-bench/main.ts and
// src/zero-tvm/bench-console.ts): identical prompt, identical target token
// count, one warmup then NUM_RUNS measured runs.
const BENCH_PROMPT = 'Write a four-sentence explanation of how photosynthesis works.'
const TARGET_TOKENS = 120
const NUM_RUNS = 3
const N_CTX = 4096 // matches webllm-bench's overrides.context_window_size

// wllama v3.1+ offloads every layer by default (n_gpu_layers = 99999).
const N_GPU_LAYERS = params.has('ngl') ? Number(params.get('ngl')) : 99999

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

// ── native llama.cpp log capture ──────────────────────────────────────────
// wllama pipes the WASM module's stdout/stderr through WllamaConfig.logger
// (see node_modules/@wllama/wllama/src/worker.ts, onRecvMsg). Those lines are
// the ONLY trustworthy evidence of which ggml backend actually took the
// weights, so capture all of them and grep afterwards.
const nativeLines: string[] = []
const INTERESTING = /webgpu|offload|buffer size|backend|n_gpu_layers|adapter|device|llama_context|graph splits/i
const fmtArgs = (args: unknown[]) =>
  args.map((a) => (typeof a === 'string' ? a : String(a))).join(' ')
const capture = (level: string, args: unknown[]) => {
  const line = fmtArgs(args).replace(/\n+$/, '')
  if (!line) return
  nativeLines.push(line)
  if (INTERESTING.test(line)) log(`[llama.cpp] ${line}`)
  else if (level === 'error') log(`[llama.cpp:error] ${line}`)
}
const logger: WllamaLogger = {
  debug: (...a: unknown[]) => capture('debug', a),
  log: (...a: unknown[]) => capture('log', a),
  warn: (...a: unknown[]) => capture('warn', a),
  error: (...a: unknown[]) => capture('error', a),
}

interface BenchRun {
  tokens: number
  seconds: number
  tokPerS: number
  ttftMs: number
  decodeTokPerS: number
  selfDecode: number
  selfPrefill: number
  /** Chunks that arrived as reasoning_content; must be 0 for a comparable run. */
  reasoningChunks: number
}

const medianOf = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]

async function main() {
  setBadge('Initializing wllama...', true)
  log('╔════════════════════════════════════════════════════════════════════╗')
  log('║ NOT A SAME-BYTES COMPARISON: wllama reads GGUF, Zero-TVM and       ║')
  log('║ WebLLM read MLC q4f16_1. Same BASE model, DIFFERENT quantization.  ║')
  log('║ Numbers from this page are runtime + quantization, not runtime.    ║')
  log('╚════════════════════════════════════════════════════════════════════╝')
  log('')
  log(`[wllama-bench] model:   ${MODEL_ID}`)
  log(`[wllama-bench] gguf:    ${GGUF_URL}`)
  log(`[wllama-bench] quant:   ${SEL.quant}  (vs q4f16_1 on the other two halves)`)
  log(`[wllama-bench] libllama: ${Wllama.getLibllamaVersion()}`)
  log(`[wllama-bench] prompt:  ${BENCH_PROMPT}`)
  log(`[wllama-bench] target tokens: ${TARGET_TOKENS}`)
  log(`[wllama-bench] runs: ${NUM_RUNS} (plus one warmup)`)
  log(`[wllama-bench] n_gpu_layers: ${N_GPU_LAYERS}${N_GPU_LAYERS === 0 ? '  ← CPU-ONLY CONTROL RUN' : ''}`)
  log('')

  const wllama = new Wllama({ default: wllamaWasmUrl }, { logger })
  // Compat mode exists for Safari/Firefox (no JSPI) and pulls its worker+wasm
  // from a CDN — disable it so this page is fully offline and so a silent
  // slide into the degraded compat path can't be mistaken for a real result.
  wllama.setCompat(null)

  await wllama.loadModelFromUrl(GGUF_URL, {
    n_ctx: N_CTX,
    n_gpu_layers: N_GPU_LAYERS,
    progressCallback: ({ loaded, total }) => {
      setBadge(`Loading ${total ? Math.floor((loaded / total) * 100) : 0}%...`, true)
    },
  })

  // ── backend proof ───────────────────────────────────────────────────────
  // navigator.gpu existing proves nothing (wllama.isSupportWebGPU() is a bare
  // `!!navigator.gpu`). What proves it is ggml-webgpu.cpp's own load-time
  // output: the adapter_info line is only printed after a GPUAdapter AND
  // GPUDevice were successfully acquired, and "offloaded N/M layers to GPU"
  // only counts layers whose tensors landed in a WebGPU buffer.
  const joined = nativeLines.join('\n')
  const adapterInfo = /ggml_webgpu:\s*adapter_info:\s*(.+)/i.exec(joined)?.[1]?.trim() ?? null
  const offloadMatch = /offloaded\s+(\d+)\/(\d+)\s+layers to GPU/i.exec(joined)
  const layersOnGpu = offloadMatch ? Number(offloadMatch[1]) : 0
  const layersTotal = offloadMatch ? Number(offloadMatch[2]) : 0
  const gpuBufferLine = /^.*WebGPU.*buffer size.*$/im.exec(joined)?.[0]?.trim() ?? null
  const webgpuProven = !!adapterInfo && layersOnGpu > 0
  const backend = webgpuProven ? 'webgpu' : 'wasm-cpu'

  log('')
  log('════════════════════ BACKEND CHECK ════════════════════')
  log(`  navigator.gpu present:      ${wllama.isSupportWebGPU()}`)
  log(`  JSPI (WebAssembly.Suspending): ${!!(WebAssembly as unknown as { Suspending?: unknown }).Suspending}`)
  log(`  crossOriginIsolated:        ${globalThis.crossOriginIsolated}`)
  log(`  multi-thread WASM:          ${wllama.isMultithread()} (${wllama.getNumThreads()} threads)`)
  log(`  n_gpu_layers requested:     ${N_GPU_LAYERS}`)
  log(`  ggml_webgpu adapter_info:   ${adapterInfo ?? '*** NOT FOUND ***'}`)
  log(`  layers offloaded to GPU:    ${offloadMatch ? `${layersOnGpu}/${layersTotal}` : '*** NOT FOUND ***'}`)
  log(`  WebGPU buffer line:         ${gpuBufferLine ?? '(none)'}`)
  log(`  ⇒ ACTIVE BACKEND: ${backend.toUpperCase()}`)
  if (!webgpuProven) {
    log('  *** WebGPU is NOT proven active. These are CPU/WASM numbers and are')
    log('  *** NOT a valid baseline for the Zero-TVM comparison.')
  }
  log('═══════════════════════════════════════════════════════')
  log('')

  setBadge('Ready', false)
  log('[wllama-bench] model loaded, running benches')
  log('')

  const runs: BenchRun[] = []
  for (let r = 0; r <= NUM_RUNS; r++) {
    const label = r === 0 ? 'warmup' : `run${r}/${NUM_RUNS}`
    let count = 0
    let tFirst = 0
    let text = ''
    let reasoningChunks = 0
    let timings: ResultTimings | undefined
    const t0 = performance.now()
    await wllama.createChatCompletion({
      messages: [
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: BENCH_PROMPT },
      ],
      max_tokens: TARGET_TOKENS,
      temperature: 0,
      // Every run must pay a FULL prefill, exactly like the Zero-TVM half's
      // resetKVTracking() and WebLLM's fresh chat completion. llama.cpp's
      // prompt cache defaults to ON, which would let runs 2..N skip prefill
      // and silently inflate wllama's wall-clock tok/s.
      cache_prompt: false,
      // Qwen3/Qwen3.5 GGUF chat templates enable thinking BY DEFAULT, and with
      // it on llama.cpp streams every token as `delta.reasoning_content`
      // instead of `delta.content` — so a content-only counter sees 0 tokens
      // and reports 0.00 tok/s while the GPU is in fact decoding at full rate.
      // The Zero-TVM half runs the NON-thinking template too (model-select.ts
      // passes {thinking:false} to buildChatMLPrompt), so switching it off here
      // is what makes the two halves the same task. Phi-3's template ignores
      // the kwarg (verified: unchanged tok/s).
      chat_template_kwargs: { enable_thinking: false },
      // Makes llama.cpp attach its own prefill/decode split to the chunks,
      // the analogue of WebLLM's usage.extra.decode_tokens_per_s.
      timings_per_token: true,
      stream: true,
      onData: (chunk: ChatCompletionChunk) => {
        if (chunk.timings) timings = chunk.timings
        // reasoning_content isn't in wllama's declared delta type, but
        // llama.cpp emits it whenever the template's thinking mode is live.
        // Counted so the guard below can tell "the model produced nothing"
        // apart from "we counted the wrong field" — those are identical in a
        // tok/s number, and one of them is a bug.
        const d = chunk.choices?.[0]?.delta as
          | { content?: string | null; reasoning_content?: string }
          | undefined
        if (d?.reasoning_content) reasoningChunks++
        const delta = d?.content
        if (!delta) return
        if (count === 0) tFirst = performance.now()
        count++
        text += delta
      },
    })
    const tEnd = performance.now()
    const seconds = (tEnd - t0) / 1000
    const tokPerS = count / seconds
    // Identical formulas to bench-console.ts: TTFT covers prefill + the first
    // decoded token; the decode rate excludes both, so the three halves line
    // up like for like.
    const ttftMs = tFirst ? tFirst - t0 : seconds * 1000
    const decodeTokPerS = count > 1 && tFirst ? (count - 1) / ((tEnd - tFirst) / 1000) : tokPerS
    const selfDecode = timings?.predicted_per_second ?? 0
    const selfPrefill = timings?.prompt_per_second ?? 0
    log(
      `[bench] ${label}: ${count} tok / ${seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s total` +
      ` · ttft ${ttftMs.toFixed(0)}ms · decode ${decodeTokPerS.toFixed(2)} tok/s` +
      (selfDecode ? ` · llama.cpp self-reported decode ${selfDecode.toFixed(2)} tok/s · prefill ${selfPrefill.toFixed(2)} tok/s` : '')
    )
    // A run that streamed no content is NOT a slow run — it's an accounting
    // failure, and `count/seconds` turns it into a plausible-looking 0.00 tok/s
    // that then poisons the median. Refuse to let it pass quietly.
    if (count === 0) {
      log(
        `  *** ${label} PRODUCED 0 CONTENT TOKENS in ${seconds.toFixed(2)}s ` +
        `(llama.cpp decoded ${timings?.predicted_n ?? '?'} tokens at ${selfDecode.toFixed(2)} tok/s` +
        `${reasoningChunks ? `, ${reasoningChunks} of them as reasoning_content` : ''}). ` +
        `This is a measurement bug, NOT a performance number — do not publish it.`,
      )
    } else if (reasoningChunks) {
      log(`  *** ${label}: ${reasoningChunks} reasoning_content chunks leaked into a thinking-mode run — token accounting is not comparable.`)
    }
    // GGUF carries its own jinja chat template — a template mismatch shows up
    // as fluent-looking garbage, not as an error, so eyeball the warmup reply.
    if (r === 0) log(`[sample] ${text.replace(/\s+/g, ' ').trim().slice(0, 200)}…`)
    if (r > 0) runs.push({ tokens: count, seconds, tokPerS, ttftMs, decodeTokPerS, selfDecode, selfPrefill, reasoningChunks })
  }

  const tokPerSs = runs.map((x) => x.tokPerS)
  const median = medianOf(tokPerSs)
  const mean = tokPerSs.reduce((a, b) => a + b, 0) / tokPerSs.length
  const min = Math.min(...tokPerSs)
  const max = Math.max(...tokPerSs)
  const medianDecode = medianOf(runs.map((x) => x.decodeTokPerS))
  const medianTtftMs = medianOf(runs.map((x) => x.ttftMs))
  const selfDecodes = runs.map((x) => x.selfDecode).filter((x) => x > 0)
  const medianSelfDecode = selfDecodes.length ? medianOf(selfDecodes) : 0

  log('')
  log(`[summary] wllama (llama.cpp ${Wllama.getLibllamaVersion()}, backend=${backend}) on ${MODEL_ID}`)
  log(`  median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s (wall-clock incl. prefill)`)
  log(`  decode=${medianDecode.toFixed(2)} tok/s · ttft=${medianTtftMs.toFixed(0)}ms`)
  if (medianSelfDecode) log(`  llama.cpp self-reported decode median=${medianSelfDecode.toFixed(2)} tok/s`)
  log('')
  log(`[caveat] ${SEL.quant} vs q4f16_1 on the Zero-TVM / WebLLM halves — runtime AND quantization differ.`)

  setStats(
    `wllama ${median.toFixed(1)} tok/s · ${backend}${webgpuProven ? '' : ' (NOT WebGPU!)'} · GGUF ≠ q4f16_1`,
  )

  ;(window as Window & typeof globalThis & { wllamaResult?: unknown }).wllamaResult = {
    model: MODEL_ID,
    modelKey: MODEL_KEY,
    gguf: SEL.file,
    quant: SEL.quant,
    libllama: Wllama.getLibllamaVersion(),
    backend,
    webgpuProven,
    adapterInfo,
    layersOnGpu,
    layersTotal,
    nGpuLayers: N_GPU_LAYERS,
    multithread: wllama.isMultithread(),
    nThreads: wllama.getNumThreads(),
    runs, median, mean, min, max, medianDecode, medianTtftMs, medianSelfDecode,
    // False if any measured run streamed no content or leaked reasoning_content.
    // The medians above are only meaningful when this is true.
    contentAccountingOk: runs.every((r) => r.tokens > 0 && r.reasoningChunks === 0),
    sameBytes: false,
    caveat:
      'GGUF (Q4_K_M/Q4) vs MLC q4f16_1 — different quantization of the same base model. ' +
      'This is a runtime+quantization comparison, NOT the same-bytes A/B the WebLLM baseline is.',
  }
}

main().catch((e) => {
  log(`[error] ${e?.message || e}`)
  console.error(e)
  setBadge('Error', false)
})
