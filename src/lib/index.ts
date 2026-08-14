/**
 * LIBRARY ENTRY POINT — this repo used as a package instead of as a site:
 * list the models that ship, boot one, chat with it. Nothing else.
 *
 * THE IMPORT-TIME RULE, which is why this file is shaped the way it is:
 * weight-loader.ts reads `GPUBufferUsage` at module scope, so importing it —
 * or anything that imports it, which is model-select.ts, engine-core.ts and
 * loading-ui.ts — throws `GPUBufferUsage is not defined` on any runtime
 * without WebGPU: Node, a bundler's SSR/prerender pass, a browser with WebGPU
 * off. An entry point that throws when it is merely imported cannot sit in
 * anyone's module graph, so everything GPU-touching is behind a dynamic
 * import inside createEngine(). Only model-registry.ts is imported eagerly —
 * it is kept dependency-free for exactly this reason (see its header).
 * share.ts's runHost() does the same thing for the same reason, and
 * tests/unit/lib-entry.test.ts is the guard that keeps it true here.
 *
 * NO DOM: loading-ui.ts's bootEngine is the page boot path — it reports
 * progress by writing into page markup (#progress-bar, #progress-log) and
 * has no callback to hand a library caller instead. The boot below is that
 * same sequence with the DOM removed and an onProgress callback in its place.
 *
 * NO SAMPLING KNOBS: the decode loop is argmax — greedy, always. There is no
 * temperature, top_p or seed to expose, and accepting parameters the engine
 * ignores would be a lie that only shows up as "why does temperature do
 * nothing". When sampling lands, it can be added here honestly.
 */

import { SHIPPED_MODELS, modelBranding } from '../zero-tvm/model-registry.js'
import type { ModelSpec } from '../compiler/model-spec.js'
import type { ChatMessage } from '../zero-tvm/model-select.js'

export type { ChatMessage }

/** One row of the shipped-model table (src/zero-tvm/model-registry.ts). */
export interface ModelInfo {
  /** What to pass as `createEngine({ model })`. '' is the default, Phi-3. */
  param: string
  /** Stable spec id (also the weight-cache directory suffix). */
  id: string
  name: string
  /** Parameter-count line, e.g. '4B hybrid (DeltaNet)'. */
  params: string
  /** Approximate download size, e.g. '~2.6 GB'. */
  sizeLabel: string
  /** Measured decode rate where a protocol-quality number exists, else ''. */
  rateLabel: string
  /** Present when the model needs more memory than a typical machine has. */
  ramNote?: string
}

export type LoadStage = 'gpu' | 'tokenizer' | 'weights' | 'engine' | 'warmup' | 'ready'

export interface LoadProgress {
  stage: LoadStage
  message: string
  /** Weight download only — the loader's own byte counters. */
  bytesLoaded?: number
  bytesTotal?: number
}

export interface CreateEngineOptions {
  /** A `param` from listModels(). Default '' (Phi-3). */
  model?: string
  onProgress?: (p: LoadProgress) => void
  /** Which chunk-prefill GEMM to build. Exposed so the native host can A/B
   *  kernels against one weight load — an isolated-kernel TF number is not a
   *  prefill rate, and the only honest comparison is two engines over the same
   *  buffers in one process. Default: the engine's own ladder. */
  chunkGemm?: import('../zero-tvm/engine-core.js').DecodeEngineOptions['chunkGemm']
  /** Capture per-layer router choices for scripts/moe-trace.mjs. */
  traceMoe?: boolean
  /** Expert slots held per MoE layer instead of all of them — see
   *  DecodeEngineOptions.expertPool. MoE models only; below top-K + 1 the
   *  engine throws. Off by default. Passed to the LOADER as well as the
   *  engine: the stacked expert tensors are then never allocated, which is
   *  where the memory saving comes from. */
  expertPool?: number
  /** Run the next MoE layer's router a layer early and prefetch what it names
   *  (DecodeEngineOptions.expertSpeculate). Needs expertPool. */
  expertSpeculate?: boolean
  /** Speculative router width for the coverage measurement (engine specWidth). */
  specWidth?: number
  /** Chunked prefill on/off (DecodeEngineOptions.chunkedPrefill — the chat
   *  page's ?chunk=0, which variantQuery does NOT carry). Chunked prefill is
   *  empirically token-identical to per-token, not bit-equal, so a harness
   *  comparing token identity across engines must pin BOTH arms to one
   *  prefill mode. moe-pool-test learned this the hard way: its unpooled
   *  reference arm chunked while the pooled arm structurally cannot, and the
   *  epsilon the chunked prefill leaves in KV flipped thin-margin argmaxes
   *  hundreds of tokens later — which was then reported as a pooling bug. */
  chunkedPrefill?: boolean
  /** Chunk capacity in tokens (DecodeEngineOptions.chunkCap; default 256 with
   *  the matrix unit, 64 without). Exposed for the CHUNK_CAP sweep. */
  chunkCap?: number
  /** Shader-variant flags as a query string, e.g. '?vec4=0'. Same parser the
   *  chat page uses, so a kernel A/B run here selects what users get. */
  variantQuery?: string
}

export interface ChatOptions {
  /** Cap on generated tokens. Default and ceiling: the KV room the prompt
   *  leaves behind (see chat()). */
  maxTokens?: number
  /**
   * Called once per token with the FULL text generated so far, not the new
   * piece. The whole id sequence is re-decoded each time deliberately: a
   * multi-byte character spans tokens, and decoding ids one at a time splits
   * it into replacement characters.
   */
  onToken?: (text: string) => void
}

export interface ZeroTvmEngine {
  /** Which model this engine actually loaded. */
  readonly model: ModelInfo
  /** Render `messages` with the model's chat template and generate a reply. */
  chat(messages: ChatMessage[], opts?: ChatOptions): Promise<string>
}

function infoFor(param: string, spec: ModelSpec): ModelInfo {
  const b = modelBranding(spec)
  return {
    param, id: spec.id, name: b.name, params: b.params,
    sizeLabel: b.sizeLabel, rateLabel: b.rateLabel, ramNote: b.ramNote,
  }
}

/** Every model this build ships, in registry order. Safe anywhere — no GPU,
 *  no network, no DOM. */
export function listModels(): ModelInfo[] {
  return SHIPPED_MODELS.map((m) => infoFor(m.param, m.spec))
}

/**
 * Our _sg shaders assume a 32-lane subgroup (one full 32-thread workgroup per
 * subgroup) and the JS-reported limit is unreliable across Chromium versions,
 * so the size is measured with a one-shot dispatch. Guessing wrong is not
 * slow, it is WRONG: the _sg kernels would reduce across lanes that do not
 * exist. Copied from loading-ui.ts's probeSubgroupSize, which is private to
 * the module that owns the DOM boot path.
 */
async function probeSubgroupSize(device: GPUDevice): Promise<boolean> {
  try {
    const buf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
    const rb = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const mod = device.createShaderModule({ code: `enable subgroups;
      @group(0) @binding(0) var<storage, read_write> out : array<u32>;
      @compute @workgroup_size(32, 1, 1)
      fn probe(@builtin(local_invocation_id) tid : vec3<u32>, @builtin(subgroup_size) s : u32) {
        if (tid.x == 0u) { out[0] = s; }
      }` })
    const pipe = device.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'probe' } })
    const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] })
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass(); pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1); pass.end()
    enc.copyBufferToBuffer(buf, 0, rb, 0, 16)
    device.queue.submit([enc.finish()])
    await rb.mapAsync(GPUMapMode.READ)
    const size = new Uint32Array(rb.getMappedRange().slice(0))[0]
    rb.unmap()
    buf.destroy(); rb.destroy()
    return size === 32
  } catch {
    return false   // no subgroups path — scalar shaders are always correct
  }
}

/**
 * Load a model and return something you can chat with. Downloads gigabytes on
 * a cold start (cached in OPFS afterwards) and holds them on the GPU for the
 * lifetime of the returned engine.
 */
/**
 * RAW boot for embedding hosts (the Dawn-native agent server) — everything
 * createEngine assembles, before the chat() wrapper narrows it. `ctx`
 * rebuilds the spec through specWithCtx (same knob as `?ctx=`). No DOM
 * anywhere on this path; the caller supplies navigator.gpu (a browser, or
 * dawn.node behind scripts/native/shims.mjs).
 */
export async function createEngineRaw(opts: CreateEngineOptions & { ctx?: number } = {}): Promise<{
  engine: import('../zero-tvm/engine-core.js').DecodeEngine
  tokenizer: Awaited<ReturnType<typeof import('../zero-tvm/model-select.js')['loadTokenizerFor']>>
  spec: ModelSpec
  variants: import('../zero-tvm/variants.js').VariantFlags
  info: ModelInfo
  /** Was already returned and merely undeclared. A caller that has to render a
   *  prompt with the model's own template — any bench comparing us to another
   *  runtime — needs it, and reaching for an undeclared runtime property is how
   *  a harness silently renders the wrong template. */
  buildChatPromptFor: typeof import('../zero-tvm/model-select.js')['buildChatPromptFor']
}> {
  const built = await bootShared(opts)
  return built
}

async function bootShared(opts: CreateEngineOptions & { ctx?: number }): Promise<{
  engine: import('../zero-tvm/engine-core.js').DecodeEngine
  tokenizer: Awaited<ReturnType<typeof import('../zero-tvm/model-select.js')['loadTokenizerFor']>>
  spec: ModelSpec
  variants: import('../zero-tvm/variants.js').VariantFlags
  info: ModelInfo
  buildChatPromptFor: typeof import('../zero-tvm/model-select.js')['buildChatPromptFor']
}> {
  const param = opts.model ?? ''
  const hit = SHIPPED_MODELS.find((m) => m.param === param)
  // specForParam() falls back to Phi-3 for anything unknown, which is right
  // for a URL (?model=typo must still boot something) and wrong here: a
  // caller asking for a model this build does not ship would silently
  // download a different one and never learn it happened.
  if (!hit) {
    throw new Error(
      `unknown model ${JSON.stringify(param)} — listModels() ships: ` +
      SHIPPED_MODELS.map((m) => JSON.stringify(m.param)).join(', '),
    )
  }
  let spec = hit.spec
  const ctxWanted = (opts as { ctx?: number }).ctx
  if (ctxWanted) {
    const { specWithCtx } = await import('../zero-tvm/model-registry.js')
    spec = specWithCtx(spec, ctxWanted)
  }
  const report = opts.onProgress ?? ((): void => {})

  // Checked BEFORE the dynamic imports below, so a runtime without WebGPU
  // gets this sentence instead of the weight loader's `GPUBufferUsage is not
  // defined` from ten frames down.
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('WebGPU is not available in this environment')
  }

  // The lazy import the header is about. Everything reachable from here reads
  // GPUBufferUsage (or imports something that does) at module scope.
  const [{ loadWeights }, { loadTokenizerFor, buildChatPromptFor }, { allocKVPages, buildDecodeEngine }, { parseVariantFlags }] =
    await Promise.all([
      import('../zero-tvm/weight-loader.js'),
      import('../zero-tvm/model-select.js'),
      import('../zero-tvm/engine-core.js'),
      import('../zero-tvm/variants.js'),
    ])

  report({ stage: 'gpu', message: 'Requesting GPU access' })
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new Error('No GPU adapter found')
  const features: GPUFeatureName[] = ['shader-f16' as GPUFeatureName]
  if (adapter.features.has('subgroups')) features.push('subgroups' as GPUFeatureName)
  // The matrix unit (experimental Chrome/Dawn). compile() creates the sgmat
  // GEMM only when the device carries it; absent, the ladder degrades.
  if (adapter.features.has('chromium-experimental-subgroup-matrix')) {
    features.push('chromium-experimental-subgroup-matrix' as GPUFeatureName)
  }
  // Enables engine.profileStep() (per-kernel timings) — OPT-IN via
  // ZTVM_PROFILE=1 / ?profile=1, never default: requesting timestamp-query on
  // the device is suspected of serializing Metal command execution.
  const wantProfile = typeof process !== 'undefined' && process.env?.ZTVM_PROFILE === '1'
  if (wantProfile && adapter.features.has('timestamp-query')) {
    features.push('timestamp-query' as GPUFeatureName)
  }
  // Lift the storage-binding/buffer ceilings to whatever the adapter offers
  // (always valid per spec). The default 128 MiB maxStorageBufferBindingSize
  // is smaller than a 4B model's tied embedding/LM-head weight; binding it
  // fails validation, which invalidates every submit that touches it and
  // reads back as silent all-zero logits rather than an error.
  const requiredLimits: Record<string, number> = {}
  if (adapter.limits) {
    requiredLimits.maxStorageBufferBindingSize = adapter.limits.maxStorageBufferBindingSize
    requiredLimits.maxBufferSize = adapter.limits.maxBufferSize
  }
  let device: GPUDevice
  try {
    device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits })
  } catch {
    throw new Error('GPU does not support shader-f16 — Chrome or Edge required')
  }

  const hasSubgroupsFeature = (device.features as ReadonlySet<string>).has('subgroups')
  const sgSizeOk = hasSubgroupsFeature ? await probeSubgroupSize(device) : false
  // An empty search string gives the chat page's DEFAULT variant selection
  // without any of its ?sg= / ?matmul= A/B toggles — the library exposes no
  // shader knobs, and this keeps the defaults in one place (variants.ts).
  // Same query string the chat page's URL flags produce, so a native A/B and a
  // browser A/B select kernels through ONE code path. Two resolvers would let
  // an arm measured here be a kernel users never run.
  const variants = parseVariantFlags(opts.variantQuery ?? '', { hasSubgroupsFeature, sgSizeOk })

  report({ stage: 'tokenizer', message: 'Loading tokenizer' })
  const tokenizer = await loadTokenizerFor(spec, (m) => report({ stage: 'tokenizer', message: m }))

  report({ stage: 'weights', message: 'Loading weights' })
  const weights = await loadWeights(
    device,
    (m) => report({ stage: 'weights', message: m }),
    (s) => report({
      stage: 'weights',
      message: `shard ${s.shardsLoaded} / ${s.totalShards}`,
      bytesLoaded: s.bytesLoaded,
      bytesTotal: s.totalBytes,
    }),
    spec,
    // No pipeline split here, and the pool has to reach the LOADER: the engine
    // alone can only pool beside the full stacks, which costs memory instead of
    // saving it. The two numbers must agree, and buildDecodeEngine checks.
    undefined,
    opts.expertPool,
  )

  report({ stage: 'engine', message: 'Compiling shaders' })
  // qkNorm specs and MLX-affine checkpoints take the UNFUSED QKV path:
  // qkv_fused folds RoPE+append into the projection with no per-head norm
  // hook, and dequantises symmetric group-32 inline. Same rule chat.ts
  // applies; the failure mode is not a crash, it is wrong logits.
  const fused = !spec.qkNorm && spec.weightFormat !== 'mlx-safetensors'
  const engine = buildDecodeEngine(device, weights, allocKVPages(device, spec), {
    spec, variants, fused, ...(opts.chunkGemm ? { chunkGemm: opts.chunkGemm } : {}),
    ...(opts.traceMoe ? { traceMoe: true } : {}),
    ...(opts.expertPool ? { expertPool: opts.expertPool } : {}),
    ...(opts.expertSpeculate ? { expertSpeculate: true } : {}),
    ...(opts.specWidth ? { specWidth: opts.specWidth } : {}),
    ...(opts.chunkedPrefill === false ? { chunkedPrefill: false } : {}),
    ...(opts.chunkCap ? { chunkCap: opts.chunkCap } : {}),
  })

  // Dawn JITs each pipeline on its FIRST dispatch (measured ~5 s cold vs
  // ~190 ms warm), so pay it here rather than inside the caller's first
  // chat(). MoE specs warm on one token only: chunked prefill is off for
  // them, so a full warmup prompt costs minutes. Same rule as chat.ts.
  report({ stage: 'warmup', message: 'Warming up GPU pipelines' })
  const warmupIds = buildChatPromptFor(spec, [{ role: 'user', content: 'Hi.' }], tokenizer)
  await engine.generatePipelined(spec.moe ? warmupIds.slice(0, 1) : warmupIds, 1, () => {})
  report({ stage: 'ready', message: 'Ready' })

  return { engine, tokenizer, spec, variants, info: infoFor(param, spec), buildChatPromptFor }
}

export async function createEngine(opts: CreateEngineOptions = {}): Promise<ZeroTvmEngine> {
  const { engine, tokenizer, spec, info, buildChatPromptFor } = await bootShared(opts)
  let generating = false
  return {
    model: info,
    async chat(messages: ChatMessage[], chatOpts: ChatOptions = {}): Promise<string> {
      // The engine is single-stream (batch = 1) and carries KV state across
      // the call, so two overlapping chats would interleave submissions into
      // one cache and both would come back wrong. share.ts serializes with a
      // queue because its callers are remote; a local caller can await.
      if (generating) {
        throw new Error('this engine is single-stream — await the previous chat() before starting another')
      }
      const promptIds = buildChatPromptFor(spec, messages, tokenizer)
      // The reply cap is the KV room the prompt leaves behind, never a
      // constant: a fixed 1024 once cut a long answer off mid-sentence.
      const budget = spec.maxContext - promptIds.length
      if (budget < 16) {
        throw new Error(`prompt is ${promptIds.length} tokens — over ${spec.id}'s ${spec.maxContext}-token context`)
      }
      const maxTokens = Math.min(chatOpts.maxTokens ?? budget, budget)
      const onToken = chatOpts.onToken
      const ids: number[] = []
      generating = true
      try {
        // Stop tokens are consumed by the loop, never emitted.
        await engine.generatePipelined(promptIds, maxTokens, (id) => {
          ids.push(id)
          // Only decode per token when someone is listening — it is O(n²)
          // work for a caller that just awaits the return value.
          if (onToken) onToken(tokenizer.decode(ids))
        })
      } finally {
        generating = false
      }
      return tokenizer.decode(ids)
    },
  }
}

// ---- host surface (the Dawn-native agent server rides these) --------------
// LAZY, deliberately: a static re-export would evaluate model-select →
// weight-loader at module scope, which reads GPUBufferUsage and would crash
// every non-GPU import of this library — the exact failure the lazy-import
// design at the top of createEngine exists to prevent. Hosts call this after
// installing their shims.
export async function hostSurface() {
  const [ms, tc, pool] = await Promise.all([
    import('../zero-tvm/model-select.js'),
    import('../zero-tvm/tool-calls.js'),
    import('../zero-tvm/kv-pool.js'),
  ])
  return {
    buildChatPromptFor: ms.buildChatPromptFor,
    withTools: tc.withTools,
    parseToolCalls: tc.parseToolCalls,
    renderAssistantCalls: tc.renderAssistantCalls,
    poolSave: pool.poolSave,
    poolTryRestore: pool.poolTryRestore,
  }
}
