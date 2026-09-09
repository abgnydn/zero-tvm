/**
 * transformers.js head-to-head bench — HuggingFace's own browser runtime
 * (ONNX Runtime Web / WebGPU EP) as a FOURTH baseline.
 *
 * ┌───────────────────────────────────────────────────────────────────────┐
 * │ ⚠ THIS IS NOT A SAME-BYTES COMPARISON.                                │
 * │                                                                       │
 * │ transformers.js reads ONNX `q4f16` — MatMulNBits 4-bit blocks with    │
 * │ fp16 activations, as exported by onnx-community. Zero-TVM and the     │
 * │ WebLLM baseline read MLC `q4f16_1`. Both are "4-bit weights, fp16     │
 * │ compute" and the names look alike, but the block layout, group size   │
 * │ and rounding are different and the files are not byte-identical.      │
 * │ So every number here is a RUNTIME+QUANTIZATION result, in the same    │
 * │ sense the wllama/GGUF page is — just with a much smaller confound     │
 * │ than GGUF Q4_K_M, since the bit width and activation dtype match.     │
 * └───────────────────────────────────────────────────────────────────────┘
 *
 * Why this baseline: transformers.js is the reference browser runtime for the
 * HuggingFace ecosystem, and the interesting open question it carries is
 * huggingface/transformers.js#1599 — Qwen3.5-4B q4f16 reported at 14.6 tok/s
 * with a **16,152 ms TTFT** on an M2 Pro, against Qwen3-4B's 22.3 tok/s /
 * 828 ms. A ~19x TTFT regression with only a ~1.5x decode regression is a
 * prefill-stage signature, so this page measures TTFT as a function of prompt
 * length (`?ttft=1`) rather than only at one prompt.
 *
 * URL params:
 *   ?model=phi3|qwen3|qwen35   which ONNX repo to load (default phi3)
 *   ?ttft=1                    run the TTFT-vs-prompt-length sweep instead of
 *                              the standard throughput protocol
 *   ?lengths=64,256            override the prompt lengths swept. Long points
 *                              are slow enough (minutes each) that running
 *                              them one page-load at a time is the practical
 *                              way to collect the curve.
 *   ?runs=2                    runs per sweep point (default 3)
 *   ?device=wasm               CPU control run. The analogue of the wllama
 *                              page's `?ngl=0`: run it and compare the GPU
 *                              dispatch counters + tok/s to prove the default
 *                              run really is on the GPU.
 *   ?remote=1                  fetch weights from huggingface.co instead of
 *                              the local mirror (see below)
 *
 * Weights come from the local dev mirror at /local-onnx/* (vite.config.ts,
 * localOnnxPlugin) so the bench is offline and the ~8 GB of ONNX is pulled
 * once. transformers.js is pointed at it with env.localModelPath, which is a
 * supported, documented path — no patching required.
 */

import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers'
import { BENCH_PROMPT, buildPromptOfLength, TTFT_PROMPT_LENGTHS } from '../bench-prompts.js'

const params = new URLSearchParams(location.search)

const REPOS = {
  phi3: {
    repo: 'Phi-3-mini-4k-instruct-ONNX',
    label: 'Phi-3-mini-4k-instruct',
    quant: 'ONNX q4f16 · onnx-community/Phi-3-mini-4k-instruct-ONNX',
  },
  qwen3: {
    repo: 'Qwen3-4B-ONNX',
    label: 'Qwen3-4B',
    quant: 'ONNX q4f16 · onnx-community/Qwen3-4B-ONNX',
  },
  qwen35: {
    repo: 'Qwen3.5-4B-ONNX',
    label: 'Qwen3.5-4B',
    quant: 'ONNX q4f16 · onnx-community/Qwen3.5-4B-ONNX',
  },
} as const

const MODEL_KEY = (params.get('model') ?? 'phi3') as keyof typeof REPOS
const SEL = REPOS[MODEL_KEY] ?? REPOS.phi3
const REMOTE = params.get('remote') === '1'
const TTFT_MODE = params.get('ttft') === '1'
// 'wasm' is the CPU control. Anything else is the real run.
const DEVICE = (params.get('device') ?? 'webgpu') as 'webgpu' | 'wasm'
// q4f16 is what onnx-community ships for all three of these and what #1599
// was filed against. Qwen3-4B's config.json even names it as the repo's
// transformers.js default dtype.
const DTYPE = 'q4f16'

const MODEL_ID = REMOTE ? `onnx-community/${SEL.repo}` : SEL.repo

// Same protocol as the other three halves (src/webllm-bench/main.ts,
// src/wllama-bench/main.ts and src/zero-tvm/bench-console.ts): identical
// prompt, identical target token count, one warmup then NUM_RUNS measured.
const TARGET_TOKENS = 120
const NUM_RUNS = 3

// Non-thinking, to match every other half. Qwen3/Qwen3.5's chat templates
// take this kwarg; Phi-3's ignores it (apply_chat_template forwards unknown
// kwargs straight into the jinja context, so passing it is inert there).
const TEMPLATE_KWARGS = { enable_thinking: false }

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

// ── WebGPU proof, installed BEFORE any model load ─────────────────────────
// `navigator.gpu` existing proves nothing, and neither does passing
// `device: 'webgpu'` — ORT Web silently falls back to the WASM EP when the
// WebGPU EP cannot take a graph, and the only visible symptom is a slow
// number. So instrument the WebGPU device itself: count compute pipelines
// created and queue submits issued. A run that really executed a 4B decode
// on the GPU cannot have zero of either, and the CPU control (`?device=wasm`)
// gives the falsification baseline.
const gpuCounters = {
  requestAdapterCalls: 0,
  requestDeviceCalls: 0,
  computePipelines: 0,
  shaderModules: 0,
  queueSubmits: 0,
  /** Submits counted from the start of the first measured run onward. */
  submitsAtRunStart: 0,
}
let capturedAdapterInfo: string | null = null
let capturedDeviceLabel: string | null = null

function instrumentWebGPU() {
  if (!('gpu' in navigator)) return
  const gpu = navigator.gpu as GPU
  const origRequestAdapter = gpu.requestAdapter.bind(gpu)
  gpu.requestAdapter = async (opts?: GPURequestAdapterOptions) => {
    gpuCounters.requestAdapterCalls++
    const adapter = await origRequestAdapter(opts)
    if (!adapter) return adapter
    const origRequestDevice = adapter.requestDevice.bind(adapter)
    adapter.requestDevice = async (desc?: GPUDeviceDescriptor) => {
      gpuCounters.requestDeviceCalls++
      const device = await origRequestDevice(desc)
      capturedDeviceLabel = device.label || null
      const info = (device as GPUDevice & { adapterInfo?: GPUAdapterInfo }).adapterInfo ?? adapter.info
      if (info) {
        capturedAdapterInfo = [
          info.vendor && `vendor: ${info.vendor}`,
          info.architecture && `architecture: ${info.architecture}`,
          info.device && `device: ${info.device}`,
          info.description && `description: ${info.description}`,
        ]
          .filter(Boolean)
          .join(' | ')
      }
      const origCreatePipeline = device.createComputePipeline.bind(device)
      device.createComputePipeline = (d: GPUComputePipelineDescriptor) => {
        gpuCounters.computePipelines++
        return origCreatePipeline(d)
      }
      const origCreatePipelineAsync = device.createComputePipelineAsync.bind(device)
      device.createComputePipelineAsync = (d: GPUComputePipelineDescriptor) => {
        gpuCounters.computePipelines++
        return origCreatePipelineAsync(d)
      }
      const origCreateShaderModule = device.createShaderModule.bind(device)
      device.createShaderModule = (d: GPUShaderModuleDescriptor) => {
        gpuCounters.shaderModules++
        return origCreateShaderModule(d)
      }
      const queue = device.queue
      const origSubmit = queue.submit.bind(queue)
      queue.submit = (buffers: Iterable<GPUCommandBuffer>) => {
        gpuCounters.queueSubmits++
        return origSubmit(buffers)
      }
      return device
    }
    return adapter
  }
}

interface BenchRun {
  tokens: number
  seconds: number
  tokPerS: number
  ttftMs: number
  decodeTokPerS: number
}

interface TtftPoint {
  targetTokens: number
  promptTokens: number
  ttftMsRuns: number[]
  ttftMs: number
  decodeTokPerS: number
  /** Set when the point could not be completed (crash, device loss, OOM). */
  error?: string
}

const SWEEP_LENGTHS = params.has('lengths')
  ? params.get('lengths')!.split(',').map(Number).filter((n) => Number.isFinite(n) && n > 0)
  : TTFT_PROMPT_LENGTHS
const SWEEP_RUNS = params.has('runs') ? Math.max(1, Number(params.get('runs'))) : 3

const medianOf = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]

/** Templated ids for a user message — the exact sequence the engine prefills. */
function templatedIds(tokenizer: PreTrainedTokenizer, userMessage: string): number[] {
  const ids = tokenizer.apply_chat_template(
    [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: userMessage },
    ],
    { add_generation_prompt: true, tokenize: true, return_tensor: false, return_dict: false, ...TEMPLATE_KWARGS },
  ) as number[]
  return ids
}

/**
 * One generation. Returns wall-clock split into TTFT and decode, using the
 * same formulas as every other half: TTFT covers prefill + the first decoded
 * token, the decode rate excludes both.
 */
async function generateOnce(
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer,
  userMessage: string,
  maxNewTokens: number,
): Promise<{ tokens: number; seconds: number; ttftMs: number; decodeTokPerS: number; text: string }> {
  const inputs = tokenizer.apply_chat_template(
    [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: userMessage },
    ],
    { add_generation_prompt: true, tokenize: true, return_dict: true, ...TEMPLATE_KWARGS },
  ) as Record<string, unknown>

  let count = 0
  let tFirst = 0
  let text = ''
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (s: string) => {
      text += s
    },
    token_callback_function: () => {
      if (count === 0) tFirst = performance.now()
      count++
    },
  })

  const t0 = performance.now()
  await model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    // Greedy, matching temperature:0 on the WebLLM/wllama halves and the
    // engine's own argmax sampler.
    do_sample: false,
    streamer,
  })
  const tEnd = performance.now()

  const seconds = (tEnd - t0) / 1000
  const ttftMs = tFirst ? tFirst - t0 : seconds * 1000
  const decodeTokPerS =
    count > 1 && tFirst ? (count - 1) / ((tEnd - tFirst) / 1000) : count / seconds
  return { tokens: count, seconds, ttftMs, decodeTokPerS, text }
}

async function main() {
  instrumentWebGPU()

  setBadge('Initializing transformers.js...', true)
  log('╔════════════════════════════════════════════════════════════════════╗')
  log('║ NOT A SAME-BYTES COMPARISON: transformers.js reads ONNX q4f16,     ║')
  log('║ Zero-TVM and WebLLM read MLC q4f16_1. Same bit width and the same  ║')
  log('║ fp16 activations, but different block layout — not the same file.  ║')
  log('╚════════════════════════════════════════════════════════════════════╝')
  log('')

  // Point transformers.js at the local mirror unless ?remote=1. localModelPath
  // + allowLocalModels is the library's own documented offline path; hub.js
  // builds `${localModelPath}${model_id}/${filename}`, which is exactly the
  // shape /local-onnx/* serves.
  if (!REMOTE) {
    env.allowLocalModels = true
    env.allowRemoteModels = false
    env.localModelPath = '/local-onnx/'
  }

  log(`[tjs-bench] transformers.js: ${env.version}`)
  log(`[tjs-bench] model:   ${MODEL_ID}`)
  log(`[tjs-bench] weights: ${REMOTE ? 'huggingface.co (remote)' : `${location.origin}/local-onnx/${SEL.repo}/`}`)
  log(`[tjs-bench] quant:   ${SEL.quant}  (vs q4f16_1 on the Zero-TVM / WebLLM halves)`)
  log(`[tjs-bench] device:  ${DEVICE}${DEVICE === 'wasm' ? '  ← CPU-ONLY CONTROL RUN' : ''}  dtype: ${DTYPE}`)
  log(`[tjs-bench] mode:    ${TTFT_MODE ? 'TTFT vs prompt length sweep' : 'standard throughput protocol'}`)
  if (!TTFT_MODE) {
    log(`[tjs-bench] prompt:  ${BENCH_PROMPT}`)
    log(`[tjs-bench] target tokens: ${TARGET_TOKENS}`)
    log(`[tjs-bench] runs: ${NUM_RUNS} (plus one warmup)`)
  }
  log('')

  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID)
  // ?profile=1 turns on ORT Web's per-kernel profiling. Used to settle WHERE
  // prefill time actually goes rather than inferring it from the graph shape.
  const PROFILE = params.get('profile') === '1'
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    dtype: DTYPE,
    device: DEVICE,
    ...(PROFILE ? { session_options: { enableProfiling: true } } : {}),
    progress_callback: (p: { status: string; progress?: number; file?: string }) => {
      if (p.status === 'progress' && typeof p.progress === 'number') {
        setBadge(`Loading ${Math.floor(p.progress)}%...`, true)
      } else if (p.status === 'ready' || p.status === 'done') {
        setBadge('Compiling...', true)
      }
    },
  })

  const pipelinesAtLoad = gpuCounters.computePipelines
  const submitsAtLoad = gpuCounters.queueSubmits

  setBadge('Ready', false)
  log('[tjs-bench] model loaded')
  log('')

  // ── warmup ──────────────────────────────────────────────────────────────
  // Also the point where shader compilation and any lazily-created pipelines
  // happen, and — under ?remote=1 — where the network is paid. Nothing
  // downstream of here should touch the network.
  const warm = await generateOnce(model, tokenizer, BENCH_PROMPT, TTFT_MODE ? 8 : TARGET_TOKENS)
  log(
    `[bench] warmup: ${warm.tokens} tok / ${warm.seconds.toFixed(2)}s = ${(warm.tokens / warm.seconds).toFixed(2)} tok/s total` +
    ` · ttft ${warm.ttftMs.toFixed(0)}ms · decode ${warm.decodeTokPerS.toFixed(2)} tok/s`,
  )
  log(`[sample] ${warm.text.replace(/\s+/g, ' ').trim().slice(0, 200)}…`)

  // ── backend proof ───────────────────────────────────────────────────────
  const ortEnv = env.backends.onnx as { webgpu?: { adapter?: GPUAdapter } } | undefined
  const ortAdapter = ortEnv?.webgpu?.adapter ?? null
  gpuCounters.submitsAtRunStart = gpuCounters.queueSubmits
  const submitsDuringWarmup = gpuCounters.queueSubmits - submitsAtLoad
  // Both conditions must hold: a device was actually created AND real compute
  // work went through it during a generation. Either alone is satisfiable by
  // an ORT probe that then falls back to WASM.
  const webgpuProven =
    DEVICE === 'webgpu' && gpuCounters.requestDeviceCalls > 0 && submitsDuringWarmup > 0
  const backend = webgpuProven ? 'webgpu' : DEVICE === 'wasm' ? 'wasm-cpu' : 'unproven'

  log('')
  log('════════════════════ BACKEND CHECK ════════════════════')
  log(`  navigator.gpu present:        ${'gpu' in navigator}`)
  log(`  device requested:             ${DEVICE}`)
  log(`  GPUAdapter requests:          ${gpuCounters.requestAdapterCalls}`)
  log(`  GPUDevice creations:          ${gpuCounters.requestDeviceCalls}`)
  log(`  adapter info (from device):   ${capturedAdapterInfo ?? '*** NOT FOUND ***'}`)
  log(`  ort.env.webgpu.adapter set:   ${!!ortAdapter}`)
  log(`  GPU device label:             ${capturedDeviceLabel || '(none)'}`)
  log(`  shader modules created:       ${gpuCounters.shaderModules}`)
  log(`  compute pipelines created:    ${gpuCounters.computePipelines} (${pipelinesAtLoad} by end of load)`)
  log(`  queue submits during warmup:  ${submitsDuringWarmup}`)
  log(`  ⇒ ACTIVE BACKEND: ${backend.toUpperCase()}`)
  if (!webgpuProven && DEVICE === 'webgpu') {
    log('  *** WebGPU is NOT proven active. These are unusable as a WebGPU')
    log('  *** baseline — do not publish them as one.')
  }
  log('═══════════════════════════════════════════════════════')
  log('')

  const common = {
    model: MODEL_ID,
    modelKey: MODEL_KEY,
    repo: `onnx-community/${SEL.repo}`,
    quant: SEL.quant,
    transformersJs: env.version,
    dtype: DTYPE,
    deviceRequested: DEVICE,
    backend,
    webgpuProven,
    adapterInfo: capturedAdapterInfo,
    ortAdapterSet: !!ortAdapter,
    gpuCounters: { ...gpuCounters },
    weightsFrom: REMOTE ? 'huggingface.co' : '/local-onnx/',
    sameBytes: false,
    caveat:
      'ONNX q4f16 (MatMulNBits int4 + fp16) vs MLC q4f16_1 — same bit width and activation dtype, ' +
      'different block layout, not byte-identical. Runtime+quantization comparison, not runtime-only.',
  }

  if (TTFT_MODE) {
    await runTtftSweep(model, tokenizer, common)
    return
  }

  // ── standard throughput protocol ────────────────────────────────────────
  const runs: BenchRun[] = []
  for (let r = 1; r <= NUM_RUNS; r++) {
    const g = await generateOnce(model, tokenizer, BENCH_PROMPT, TARGET_TOKENS)
    const tokPerS = g.tokens / g.seconds
    log(
      `[bench] run${r}/${NUM_RUNS}: ${g.tokens} tok / ${g.seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s total` +
      ` · ttft ${g.ttftMs.toFixed(0)}ms · decode ${g.decodeTokPerS.toFixed(2)} tok/s`,
    )
    if (g.tokens === 0) {
      log(
        `  *** run${r} PRODUCED 0 TOKENS in ${g.seconds.toFixed(2)}s. This is a measurement ` +
        `bug, NOT a performance number — do not publish it.`,
      )
    }
    runs.push({ tokens: g.tokens, seconds: g.seconds, tokPerS, ttftMs: g.ttftMs, decodeTokPerS: g.decodeTokPerS })
  }

  const tokPerSs = runs.map((x) => x.tokPerS)
  const median = medianOf(tokPerSs)
  const mean = tokPerSs.reduce((a, b) => a + b, 0) / tokPerSs.length
  const min = Math.min(...tokPerSs)
  const max = Math.max(...tokPerSs)
  const medianDecode = medianOf(runs.map((x) => x.decodeTokPerS))
  const medianTtftMs = medianOf(runs.map((x) => x.ttftMs))
  const promptTokens = templatedIds(tokenizer, BENCH_PROMPT).length

  log('')
  log(`[summary] transformers.js ${env.version} (backend=${backend}, dtype=${DTYPE}) on ${MODEL_ID}`)
  log(`  median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s (wall-clock incl. prefill)`)
  log(`  decode=${medianDecode.toFixed(2)} tok/s · ttft=${medianTtftMs.toFixed(0)}ms · prompt=${promptTokens} tok`)
  log('')
  log(`[caveat] ${SEL.quant} vs MLC q4f16_1 on the Zero-TVM / WebLLM halves — runtime AND quantization differ.`)

  setStats(
    `transformers.js ${median.toFixed(1)} tok/s · ${backend}${webgpuProven ? '' : ' (NOT WebGPU!)'} · ONNX q4f16 ≠ q4f16_1`,
  )

  ;(window as Window & typeof globalThis & { tjsResult?: unknown }).tjsResult = {
    ...common,
    promptTokens,
    runs,
    median,
    mean,
    min,
    max,
    medianDecode,
    medianTtftMs,
    generatedTokens: runs[0]?.tokens ?? 0,
    contentAccountingOk: runs.every((r) => r.tokens > 0),
    sample: warm.text.replace(/\s+/g, ' ').trim().slice(0, 300),
  }
}

/**
 * THE KEY EXPERIMENT — TTFT as a function of prompt length.
 *
 * #1599's signature is a ~19x TTFT regression against a ~1.5x decode
 * regression, which localises the problem to prefill. A single prompt length
 * cannot distinguish "prefill has a large constant overhead" from "prefill
 * scales badly with length"; a sweep can. Only 8 new tokens are generated per
 * point because everything after the first token is decode, which the
 * standard protocol already measures.
 */
async function runTtftSweep(
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer,
  common: Record<string, unknown>,
) {
  const RUNS_PER_POINT = SWEEP_RUNS
  const GEN_TOKENS = 8

  log('════════════ TTFT vs PROMPT LENGTH ════════════')
  log(`  lengths: ${SWEEP_LENGTHS.join(', ')} tokens (nominal)`)
  log(`  ${RUNS_PER_POINT} runs per point, ${GEN_TOKENS} new tokens each, median reported`)
  log('')

  const points: TtftPoint[] = []
  for (const target of SWEEP_LENGTHS) {
    const built = buildPromptOfLength(target, (m) => templatedIds(tokenizer, m))
    const ttfts: number[] = []
    const decodes: number[] = []
    // One unmeasured run at this length first: a new sequence length can
    // trigger fresh shader specialisation in ORT, and that one-off compile is
    // not the prefill cost being measured.
    await generateOnce(model, tokenizer, built.userMessage, GEN_TOKENS)
    for (let r = 0; r < RUNS_PER_POINT; r++) {
      const g = await generateOnce(model, tokenizer, built.userMessage, GEN_TOKENS)
      ttfts.push(g.ttftMs)
      decodes.push(g.decodeTokPerS)
      log(
        `[ttft] prompt=${built.actualTokens} tok (target ${target}) run${r + 1}: ` +
        `ttft ${g.ttftMs.toFixed(0)}ms · ${g.tokens} gen tok · decode ${g.decodeTokPerS.toFixed(2)} tok/s`,
      )
    }
    points.push({
      targetTokens: target,
      promptTokens: built.actualTokens,
      ttftMsRuns: ttfts,
      ttftMs: medianOf(ttfts),
      decodeTokPerS: medianOf(decodes),
    })
  }

  log('')
  log('  prompt tok |  TTFT ms |  ms/prompt-token |  vs shortest')
  log('  -----------+----------+------------------+-------------')
  const base = points[0]
  for (const p of points) {
    log(
      `  ${String(p.promptTokens).padStart(10)} | ${p.ttftMs.toFixed(0).padStart(8)} | ` +
      `${(p.ttftMs / p.promptTokens).toFixed(3).padStart(16)} | ` +
      `${(p.ttftMs / base.ttftMs).toFixed(2).padStart(11)}x`,
    )
  }
  log('')
  log('  Linear prefill ⇒ ms/prompt-token flat and "vs shortest" tracking the')
  log('  length ratio. Superlinear ⇒ ms/prompt-token rising with length.')

  setStats(`transformers.js TTFT sweep · ${points.map((p) => `${p.promptTokens}:${p.ttftMs.toFixed(0)}ms`).join(' · ')}`)

  ;(window as Window & typeof globalThis & { tjsResult?: unknown }).tjsResult = {
    ...common,
    mode: 'ttft-sweep',
    queueSubmitsTotal: gpuCounters.queueSubmits,
    computePipelines: gpuCounters.computePipelines,
    runsPerPoint: RUNS_PER_POINT,
    genTokensPerPoint: GEN_TOKENS,
    points,
  }
}

main().catch((e) => {
  log(`[error] ${e?.message || e}`)
  console.error(e)
  setBadge('Error', false)
})
