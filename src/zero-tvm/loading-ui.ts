/**
 * SHARED LOADING UI + BOOT FLOW
 *
 * Booting an engine means: request the GPU, load the tokenizer, download the
 * weights with a visible progress bar, allocate the KV cache, and compile the
 * shaders. This module owns that pipeline. Both shipped pages boot through
 * bootEngine: validate.ts takes the defaults (reference engine, blocking
 * generate), chat.ts passes BootEngineOptions to request extra GPU features
 * (subgroups + timestamp-query), probe the subgroup size, and build the fused
 * variant-selected engine.
 *
 * Pages must include the standard progress markup (see zero-tvm.html /
 * validate.html): #badge, #progress-wrap, #progress-status, #progress-bar,
 * #progress-detail, #progress-log.
 */

import { loadWeights, formatEta, LoadedWeights } from './weight-loader.js'
import { Tokenizer } from './tokenizer.js'
import { loadTokenizerFor, buildChatPromptFor } from './model-select.js'
import { allocKVPages, buildDecodeEngine, type DecodeEngine } from './engine-core.js'
import { SCALAR_VARIANTS } from './variants.js'
import { PHI3, type ModelSpec } from '../compiler/compiler.js'

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
  // The chat page nests the label in #badge-text (keeping a status dot span);
  // validate's badge is a bare element. Support both.
  const txt = document.getElementById('badge-text')
  if (txt) txt.textContent = text
  else badge.textContent = text
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
// Subgroup-size probe
// ============================================================

/**
 * Our _sg shaders assume subgroup size == 32 (full 32-thread workgroup in one
 * subgroup). Apple M-series and most shipping WebGPU subgroup backends report
 * 32; the JS limit is unreliable across Chromium versions so we probe via a
 * one-shot compute dispatch. Requires the 'subgroups' feature on `device`.
 */
async function probeSubgroupSize(device: GPUDevice): Promise<boolean> {
  try {
    const probeBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const probeRb  = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const probeMod = device.createShaderModule({ code: `enable subgroups;
      @group(0) @binding(0) var<storage, read_write> out : array<u32>;
      @compute @workgroup_size(32, 1, 1)
      fn probe(@builtin(local_invocation_id) tid : vec3<u32>, @builtin(subgroup_size) s : u32) {
        if (tid.x == 0u) { out[0] = s; }
      }` })
    const probePipe = device.createComputePipeline({ layout: 'auto', compute: { module: probeMod, entryPoint: 'probe' } })
    const probeBg = device.createBindGroup({ layout: probePipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: probeBuf } }] })
    const pe = device.createCommandEncoder()
    const pp = pe.beginComputePass(); pp.setPipeline(probePipe); pp.setBindGroup(0, probeBg); pp.dispatchWorkgroups(1); pp.end()
    pe.copyBufferToBuffer(probeBuf, 0, probeRb, 0, 16)
    device.queue.submit([pe.finish()])
    await probeRb.mapAsync(GPUMapMode.READ)
    const sg = new Uint32Array(probeRb.getMappedRange().slice(0))[0]
    probeRb.unmap()
    const sgSizeOk = sg === 32
    log(`subgroups feature: yes · probed size: ${sg} · path: ${sgSizeOk ? '_sg' : 'scalar'}`)
    probeBuf.destroy(); probeRb.destroy()
    return sgSizeOk
  } catch (e) {
    log(`subgroups probe failed: ${e} · falling back to scalar`)
    return false
  }
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

export interface BootEngineOptions {
  /** Model to boot. Default: PHI3. Selects the weights repo, tokenizer
   * pipeline, chat template and engine shape (see model-select.ts). */
  spec?: ModelSpec
  /** GPU features to request in addition to the required 'shader-f16', each
   * only if the adapter supports it (e.g. 'subgroups', 'timestamp-query'). */
  optionalFeatures?: GPUFeatureName[]
  /** Probe the device's subgroup size with a one-shot dispatch (no-op unless
   * 'subgroups' was granted). The result is passed to buildEngine. */
  probeSubgroups?: boolean
  /** Called on unexpected device loss, in addition to the standard badge /
   * progress / log surfaces (which fire regardless). */
  onDeviceLost?: (info: GPUDeviceLostInfo) => void
  /** Allocate the KV cache and build the engine. Default: f16 KV pages +
   * buildDecodeEngine defaults (unfused reference path, scalar shaders). */
  buildEngine?: (ctx: { device: GPUDevice; weights: LoadedWeights; sgSizeOk: boolean; spec: ModelSpec }) => DecodeEngine
  /** Load only ONE PIPELINE STAGE's weights (see loadWeights). The caller's
   *  buildEngine must pass the same range to buildDecodeEngine. */
  layerRange?: { start: number; end: number }
  /** Warm the lazily-JIT'd pipelines. Default: one forwardLogits pass. */
  warmup?: (engine: DecodeEngine, tokenizer: Tokenizer, log?: (msg: string) => void) => Promise<void>
}

/**
 * Run the full boot sequence with progress reporting wired into the standard
 * markup. Returns either the live engine or a structured failure.
 *
 * The percent budget: GPU+tokenizer take the first 10%, weight download takes
 * 80% (the long pole), and KV alloc + shader compile take the last 10%.
 */
export async function bootEngine(opts: BootEngineOptions = {}): Promise<BootResult> {
  const spec = opts.spec ?? PHI3
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

  const wantFeatures: GPUFeatureName[] = ['shader-f16' as GPUFeatureName]
  for (const f of opts.optionalFeatures ?? []) {
    if (adapter.features.has(f)) wantFeatures.push(f)
  }
  // Lift the storage-binding/buffer ceilings to whatever the adapter offers
  // (same as tests/kernels/gpu.mjs — RESEARCH_STANDARDS.md §13). The default
  // 128 MiB maxStorageBufferBindingSize is smaller than Qwen3-4B's tied
  // embedding/LM-head q_weight (151936 × 320 u32 ≈ 194 MB); binding it fails
  // validation, which invalidates every submit that touches it and reads back
  // as silent all-zero logits. Requesting the adapter's own limits is always
  // valid per spec.
  const requiredLimits: Record<string, number> = {}
  if (adapter.limits) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize
  }
  let device: GPUDevice
  try {
    device = await adapter.requestDevice({ requiredFeatures: wantFeatures, requiredLimits })
  } catch {
    setBadge('shader-f16 missing', 'error')
    setProgress(0, 'GPU does not support shader-f16 — Chrome or Edge required')
    return { ok: false, reason: 'shader-f16 unsupported — Chrome/Edge required' }
  }
  log(`GPU ready — features: ${wantFeatures.join(', ')}`)
  // Surface device loss (driver reset, OOM, GPU process crash) instead of
  // letting every later submit fail silently. No retry — a reload is the fix.
  void device.lost.then((info) => {
    if (info.reason === 'destroyed') return  // intentional teardown
    setBadge('GPU lost', 'error')
    setProgress(0, `GPU device lost: ${info.message || info.reason} — reload the page to recover`)
    log(`ERROR: GPU device lost (${info.reason}): ${info.message}`)
    opts.onDeviceLost?.(info)
  })

  let sgSizeOk = false
  // MoE specs force the probe: moe_router_topk and the expert matmul are
  // subgroup-only, so without it the default boot below cannot build at all.
  if (opts.probeSubgroups || spec.moe) {
    if ((device.features as ReadonlySet<string>).has('subgroups')) {
      setProgress(3, 'Probing GPU subgroups...')
      sgSizeOk = await probeSubgroupSize(device)
    } else {
      log('subgroups feature: no · path: scalar')
    }
    // A MoE spec that cannot build must refuse HERE, at 3%, not at 96%.
    // moe_router_topk and the grid-z expert matmul have no scalar variant, so
    // buildDecodeEngine throws — but that throw used to land after the full
    // download, in a log line inside a collapsed <details>: the visitor paid
    // 16.4 GB for a progress bar frozen at "Compiling shaders…" with no
    // visible error. The probe already ran; use its answer before any byte.
    if (spec.moe && !sgSizeOk) {
      setBadge('GPU unsupported', 'error')
      setProgress(0, `${spec.id} is a sparse MoE and needs GPU subgroups (a Chromium WebGPU `
        + 'feature) — this GPU/browser does not expose them. The dense models on the landing page still run.')
      return { ok: false, reason: 'MoE spec requires the subgroups feature' }
    }
  }
  setProgress(5, 'Loading tokenizer...')

  setBadge('Tokenizer...', 'loading')
  let tokenizer: Tokenizer
  try {
    tokenizer = await loadTokenizerFor(spec, (m) => log(m))
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
    weights = await loadWeights(device, (m) => log(m), (s) => {
      // Real byte-level progress from the loader's own stats.
      const frac = s.totalBytes > 0 ? Math.min(1, s.bytesLoaded / s.totalBytes) : 0
      const pct = Math.round(frac * 100)
      setProgress(
        10 + frac * 80,
        `Downloading weights — shard ${s.shardsLoaded} / ${s.totalShards}`,
        `${formatBytes(s.bytesLoaded)} / ${formatBytes(s.totalBytes)} · ` +
          `${s.mbPerSec.toFixed(0)} MB/s` +
          (s.etaSec > 0 ? ` · ~${formatEta(s.etaSec)} left` : ''),
      )
      setBadge(`${pct}%`, 'loading')
    }, spec, opts.layerRange)
  } catch (e) {
    setBadge('Download failed', 'error')
    setProgress(10, `Weight load error: ${e}`)
    log(`ERROR: ${e}`)
    return { ok: false, reason: `Weight load failed: ${e}` }
  }
  setProgress(92, 'Allocating KV cache...')

  let engine: DecodeEngine
  if (opts.buildEngine) {
    // The page allocates its own KV cache (f16 or int8) and picks the engine
    // mode + shader variants. It may call log() for its own alloc messages.
    setProgress(96, 'Compiling shaders...')
    engine = opts.buildEngine({ device, weights, sgSizeOk, spec })
  } else {
    log(`Allocating KV cache (${spec.layers} layers × ${spec.maxPages} pages)`)
    const kvPages = allocKVPages(device, spec)
    setProgress(96, 'Compiling shaders...')
    log('Compiling WGSL kernels (10 roles)')
    // A MoE spec has no scalar path — moe_router_topk and the grid-z expert
    // matmul are subgroup-only — so the default boot picks the subgroup
    // variants when the probe passed. If it did not, buildDecodeEngine's
    // guard throws, which is the correct answer on such a device.
    const variants = spec.moe && sgSizeOk
      ? { ...SCALAR_VARIANTS, subgroups: true, matmul: 'tiled' as const }
      : undefined
    engine = buildDecodeEngine(device, weights, kvPages, { spec, variants })
  }

  // Pipeline warmup. createComputePipeline only registers shaders — Chrome's
  // Dawn backend lazily JITs them on first dispatch, which empirically costs
  // 5–10s on the first forward pass (measured: cold ~5147ms, warm ~189ms on
  // a 5-token prompt). Pay it now, behind the progress bar, so chat's first
  // message and validate's first prompt are both honest steady-state.
  setProgress(98, 'Warming up GPU pipelines...')
  const warmupT0 = performance.now()
  if (opts.warmup) {
    await opts.warmup(engine, tokenizer, log)
  } else {
    const warmupIds = buildChatPromptFor(
      spec,
      [{ role: 'user', content: 'Hi.' }],
      tokenizer
    )
    await engine.forwardLogits(warmupIds)
  }
  log(`Warmup forward: ${Math.round(performance.now() - warmupT0)} ms`)

  setProgress(100, 'Ready')
  log('Ready. Zero TVM. 10 hand-written WGSL kernel roles.')
  setBadge('Ready', 'ready')

  return { ok: true, device, tokenizer, weights, engine }
}
