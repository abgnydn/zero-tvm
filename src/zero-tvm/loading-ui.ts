/**
 * SHARED LOADING UI + BOOT FLOW
 *
 * Booting an engine means: request the GPU, load the tokenizer, download the
 * weights with a visible progress bar, allocate the KV cache, and compile the
 * shaders. This module owns that pipeline. Today only validate.ts boots
 * through bootEngine — chat.ts still has its own inline boot flow (moving it
 * here is part of the planned chat/engine-core unification).
 *
 * Pages must include the standard progress markup (see zero-tvm.html /
 * validate.html): #badge, #progress-wrap, #progress-status, #progress-bar,
 * #progress-detail, #progress-log.
 */

import { loadWeights, LoadedWeights } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt, Tokenizer } from './tokenizer.js'
import { allocKVPages, buildDecodeEngine, type DecodeEngine } from './engine-core.js'
import { PHI3 } from '../compiler/compiler.js'

// ============================================================
// DOM helpers — safe to call even if a target element is missing
// ============================================================

export function log(msg: string): void {
  const el = document.getElementById('progress-log')
  if (el) {
    el.textContent += msg + '\n'
    el.scrollTop = el.scrollHeight
  }
}

export type BadgeState = 'idle' | 'loading' | 'ready' | 'error'

export function setBadge(text: string, state: BadgeState = 'idle'): void {
  const badge = document.getElementById('badge')
  if (!badge) return
  badge.textContent = text
  badge.className = `badge ${state}`
}

export function setProgress(pct: number, status?: string, detail?: string): void {
  const bar = document.getElementById('progress-bar')
  const statusEl = document.getElementById('progress-status')
  const detailEl = document.getElementById('progress-detail')
  if (bar) (bar as HTMLElement).style.width = `${Math.min(100, pct)}%`
  if (status && statusEl) statusEl.textContent = status
  if (detail !== undefined && detailEl) detailEl.textContent = detail
}

export function showProgress(): void {
  document.getElementById('progress-wrap')?.classList.add('active')
}

export function hideProgress(): void {
  document.getElementById('progress-wrap')?.classList.remove('active')
}

export function formatBytes(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(1) + ' GB'
  if (b >= 1e6) return (b / 1e6).toFixed(0) + ' MB'
  if (b >= 1e3) return (b / 1e3).toFixed(0) + ' KB'
  return b + ' B'
}

// ============================================================
// Boot pipeline
// ============================================================

export interface BootedEngine {
  device: GPUDevice
  tokenizer: Tokenizer
  weights: LoadedWeights
  engine: DecodeEngine
}

/** Result of bootEngine when something user-recoverable went wrong (e.g. no
 * WebGPU). The page is responsible for surfacing this. */
export interface BootError {
  ok: false
  reason: string
}
export type BootResult = ({ ok: true } & BootedEngine) | BootError

/**
 * Run the full boot sequence with progress reporting wired into the standard
 * markup. Returns either the live engine or a structured failure.
 *
 * The percent budget mirrors what chat.ts originally used: GPU+tokenizer take
 * the first 10%, weight download takes 80% (the long pole), and KV alloc +
 * shader compile take the last 10%.
 */
export async function bootEngine(): Promise<BootResult> {
  if (!navigator.gpu) {
    setBadge('No WebGPU', 'error')
    setProgress(0, 'No WebGPU available')
    return { ok: false, reason: 'WebGPU is not available in this browser' }
  }

  showProgress()
  setBadge('Initializing...', 'loading')
  setProgress(0, 'Requesting GPU access...')

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) {
    setBadge('No GPU', 'error')
    setProgress(0, 'No GPU adapter found')
    return { ok: false, reason: 'No GPU adapter found' }
  }

  let device: GPUDevice
  try {
    device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16' as GPUFeatureName],
    })
  } catch {
    setBadge('shader-f16 missing', 'error')
    setProgress(0, 'GPU does not support shader-f16 — Chrome or Edge required')
    return { ok: false, reason: 'shader-f16 unsupported — Chrome/Edge required' }
  }
  log('GPU: shader-f16 enabled')
  // Surface device loss (driver reset, OOM, GPU process crash) instead of
  // letting every later submit fail silently. No retry — a reload is the fix.
  void device.lost.then((info) => {
    if (info.reason === 'destroyed') return  // intentional teardown
    setBadge('GPU lost', 'error')
    setProgress(0, `GPU device lost: ${info.message || info.reason} — reload the page to recover`)
    log(`ERROR: GPU device lost (${info.reason}): ${info.message}`)
  })
  setProgress(5, 'Loading tokenizer...')

  setBadge('Tokenizer...', 'loading')
  let tokenizer: Tokenizer
  try {
    tokenizer = await loadTokenizer((m) => log(m))
  } catch (e) {
    setBadge('Tokenizer failed', 'error')
    setProgress(5, `Tokenizer error: ${e}`)
    return { ok: false, reason: `Tokenizer load failed: ${e}` }
  }
  log('Tokenizer loaded')
  setProgress(10, 'Loading model weights...')

  setBadge('Downloading...', 'loading')
  let weights: LoadedWeights
  try {
    weights = await loadWeights(device, (m) => {
      log(m)
      // Coarse progress: our weight-loader emits "Loading layer N/32" messages.
      const match = /Loading layer (\d+)/.exec(m)
      if (match) {
        const layer = parseInt(match[1], 10)
        const pct = (layer / 32) * 100
        setProgress(10 + pct * 0.8, `Loading layer ${layer} / 32`)
        setBadge(`${Math.round(pct)}%`, 'loading')
      }
    })
  } catch (e) {
    setBadge('Download failed', 'error')
    setProgress(10, `Weight load error: ${e}`)
    log(`ERROR: ${e}`)
    return { ok: false, reason: `Weight load failed: ${e}` }
  }
  setProgress(92, 'Allocating KV cache...')

  log(`Allocating KV cache (${PHI3.LAYERS} layers × ${PHI3.MAX_PAGES} pages)`)
  const kvPages = allocKVPages(device)
  setProgress(96, 'Compiling shaders...')

  log('Compiling 27 WGSL files (10 kernel roles)')
  const engine = buildDecodeEngine(device, weights, kvPages)

  // Pipeline warmup. createComputePipeline only registers shaders — Chrome's
  // Dawn backend lazily JITs them on first dispatch, which empirically costs
  // 5–10s on the first forward pass (measured: cold ~5147ms, warm ~189ms on
  // a 5-token prompt). Pay it now, behind the progress bar, so chat's first
  // message and validate's first prompt are both honest steady-state.
  setProgress(98, 'Warming up GPU pipelines...')
  const warmupT0 = performance.now()
  const warmupIds = buildChatPrompt(
    [{ role: 'user', content: 'Hi.' }],
    tokenizer
  )
  await engine.forwardLogits(warmupIds)
  log(`Warmup forward: ${Math.round(performance.now() - warmupT0)} ms`)

  setProgress(100, 'Ready')
  log('Ready. Zero TVM. 10 kernels across 27 WGSL files.')
  setBadge('Ready', 'ready')

  return { ok: true, device, tokenizer, weights, engine }
}
