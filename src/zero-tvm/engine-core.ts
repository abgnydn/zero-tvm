/**
 * ZERO-TVM ENGINE CORE
 *
 * Pure GPU pipeline: takes a loaded device + weights + KV cache, returns THE
 * DecodeEngine. No DOM, no UI. Both shipped pages drive this one engine:
 *
 *   - validate.ts (via loading-ui.ts's bootEngine) runs the default
 *     configuration — unfused reference path (9 dispatches/layer), scalar
 *     shaders, blocking generate()/forwardLogits() with deterministic
 *     per-token positions and logits access.
 *   - chat.ts runs the throughput configuration — fused QKV+RoPE+KV-append
 *     (7 dispatches/layer, 8 with int8 KV), URL-flag shader variants
 *     (src/zero-tvm/variants.ts), and generatePipelined()'s readback ring
 *     with on-GPU argmax→inputIds chaining.
 *
 * Both paths share the same buffers, uniforms, bind-group construction and
 * recordForward() dispatch recorder; they differ only in the per-layer QKV
 * stage (mode flags) and in how tokens are read back.
 */

import { LoadedWeights } from './weight-loader.js'
import { compile, PHI3 } from '../compiler/compiler.js'
import { SCALAR_VARIANTS, resolveVariantPipelines, type VariantFlags } from './variants.js'

// ============================================================
// GPU helpers
// ============================================================

function createBuf(device: GPUDevice, size: number, usage: number, label?: string): GPUBuffer {
  return device.createBuffer({ size: Math.max(size, 4), usage, label })
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function makeBuf(device: GPUDevice, size: number, label: string): GPUBuffer {
  return createBuf(device, size, STORAGE, label)
}

function uniformBuf(device: GPUDevice, data: (number | ArrayBuffer)[]): GPUBuffer {
  const parts: ArrayBuffer[] = data.map(d =>
    d instanceof ArrayBuffer ? d : (() => { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, d, true); return a })()
  )
  const size = parts.reduce((s, p) => s + p.byteLength, 0)
  const padded = Math.ceil(size / 16) * 16
  const buf = device.createBuffer({ size: Math.max(padded, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const arr = new Uint8Array(padded)
  let off = 0
  for (const p of parts) { arr.set(new Uint8Array(p), off); off += p.byteLength }
  device.queue.writeBuffer(buf, 0, arr)
  return buf
}

function u32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, v, true); return a }
function i32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setInt32(0, v, true); return a }

function bg(device: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
}

// Profile state is closed over by buildDecodeEngine's `dispatch` helper.
// When active, every compute pass gets begin/end timestamp writes into a
// shared GPUQuerySet and the label is recorded alongside the slot pair.
interface ProfileState {
  querySet: GPUQuerySet
  capacity: number
  labels: string[]       // parallel to slot index; length === labels.length
  nextSlot: number       // next available slot (each pass consumes 2)
}

// ============================================================
// KV cache allocation (one pages buffer per layer)
// ============================================================

export function allocKVPages(device: GPUDevice): GPUBuffer[] {
  const bytesPerPage = 98304 * 2  // 32 heads * 16 slots * 96 dims * 2 (K+V) * 2 bytes
  const pages = PHI3.MAX_PAGES * bytesPerPage  // 257 * 196608 ≈ 50MB
  return Array.from({ length: PHI3.LAYERS }, (_, i) =>
    makeBuf(device, pages, `kvPages_${i}`)
  )
}

// int8 layout: 32 heads × 16 slots × 2 sides × 24 u32-words = 24576 u32 per page.
// Halves KV memory from ~1.6GB to ~800MB for 4K context, at the cost of one
// extra dispatch per layer (quantize). Opt-in (?kv8=1 on the chat page) —
// validate output parity against the f16 baseline on your target hardware.
export function allocKVPagesInt8(device: GPUDevice): { pages: GPUBuffer[]; scales: GPUBuffer[] } {
  const wordsPerPage = 32 * 16 * 2 * 24     // 24576
  const bytesPerPage = wordsPerPage * 4     // 96 KB
  const scalesPerPage = 32 * 16 * 2         // 1024 f16 per page
  const pagesBytes = PHI3.MAX_PAGES * bytesPerPage
  const scalesBytes = PHI3.MAX_PAGES * scalesPerPage * 2
  return {
    pages: Array.from({ length: PHI3.LAYERS }, (_, i) =>
      makeBuf(device, pagesBytes, `kvPagesI8_${i}`)
    ),
    scales: Array.from({ length: PHI3.LAYERS }, (_, i) =>
      makeBuf(device, scalesBytes, `kvScales_${i}`)
    ),
  }
}

// ============================================================
// Decode engine
// ============================================================

// Quantization layout constants (Q4f16_1):
//   8 int4 values packed into one u32
//   group size of 32 weights shares one f16 scale
const PACK = 8
const GROUP = 32

// Per-matmul (K, M) shape uniforms compute K_PACKED = K/8 and SCALES = K/32
const QKV_K_PACKED   = PHI3.D / PACK            // 384  (K=3072)
const QKV_SCALES     = PHI3.D / GROUP           // 96
const FFN_DN_K_PACKED = PHI3.FFN / PACK         // 1024 (K=8192)
const FFN_DN_SCALES   = PHI3.FFN / GROUP        // 256

// Workgroup-count constants — derived from each shader's @workgroup_size:
//   embedding/rms_norm/add_norm/kv_append/rope: WG_SIZE_D = 256, one wg per 256 hidden units
//   matmul shaders: one wg per output element (M), divided by the variant's rows/WG
const WG_SIZE_D = 256
const D_WGS     = PHI3.D / WG_SIZE_D            // 12   (3072/256)
const QKV_WGS   = PHI3.QKV_DIM / WG_SIZE_D      // 36   (9216/256)

export interface KernelProfile {
  kernels: Array<{ label: string; totalMs: number; calls: number; pctOfTotal: number }>
  totalMs: number
}

export interface BatchedBenchResult {
  msBatched: number
  msTiledTotal: number
  msPerMBatched: number
  msPerMTiled: number
  speedup: number
}

export interface DecodeEngine {
  /**
   * Blocking generate: one submit + one readback per token, deterministic
   * per-token positions, KV reuse via `startPos`. The validation harness path.
   */
  generate(
    promptIds: number[],
    startPos: number,
    maxTokens: number,
    onToken: (id: number) => void
  ): Promise<number[]>
  /**
   * Pipelined generate: readback-free prefill, then PIPELINE_DEPTH tokens of
   * work kept in flight with argmax→inputIds chained on-GPU. The chat path.
   * `shouldStop` is polled between pipeline submissions (cooperative stop).
   */
  generatePipelined(
    promptIds: number[],
    maxTokens: number,
    onToken: (id: number) => void,
    shouldStop?: () => boolean,
  ): Promise<number[]>
  /** Run a forward pass through prefill of `promptIds` and return f32 logits at the final position. */
  forwardLogits(promptIds: number[]): Promise<Float32Array>
  /** Reset KV-cache invalidation tracking (call when starting a fresh conversation). */
  resetKVTracking(): void
  /** Hard KV ceiling in tokens: PHI3.MAX_PAGES × PHI3.PAGE_SIZE. */
  maxContext: number
  /** Timestamp-query profile of one steady-state decode step (null if the feature is off). */
  profileStep(warmupIds: number[]): Promise<KernelProfile | null>
  benchBatchedFfnDown(
    M: number,
    iters: number,
    target?: 'ffnDown' | 'oproj',
  ): Promise<BatchedBenchResult | null>
}

export interface DecodeEngineOptions {
  /** Resolved shader-variant flags (see variants.ts). Default: SCALAR_VARIANTS. */
  variants?: VariantFlags
  /**
   * true  → fused QKV+RoPE+KV-append (qkv_fused writes kvPages directly;
   *         7 dispatches/layer, 8 with int8 KV) — the chat path.
   * false → unfused reference: qkv matmul → rope → kv_append (9/layer).
   */
  fused?: boolean
  /** int8 KV cache. Requires `fused` and a kv argument carrying `scales`. */
  int8KV?: boolean
}

export function buildDecodeEngine(
  device: GPUDevice,
  weights: LoadedWeights,
  kv: GPUBuffer[] | { pages: GPUBuffer[]; scales?: GPUBuffer[] },
  opts: DecodeEngineOptions = {},
): DecodeEngine {
  const variants = opts.variants ?? SCALAR_VARIANTS
  const fused = opts.fused ?? false
  const int8Mode = opts.int8KV ?? false
  const kvPages = Array.isArray(kv) ? kv : kv.pages
  const kvScales = Array.isArray(kv) ? undefined : kv.scales
  if (int8Mode && (!fused || !kvScales)) {
    throw new Error('buildDecodeEngine: int8KV requires fused mode and a KV cache with scales buffers')
  }

  const { pipelines } = compile(device, { subgroups: variants.subgroups })
  const P = pipelines
  const R = resolveVariantPipelines(variants, P)
  // Split-K attention (?splitk=N) exists only for the f16 KV layout;
  // attention_int8 has no split-K variant.
  if (int8Mode && R.splitK) {
    console.warn('[engine] ?splitk ignored with int8 KV (attention_int8 has no split-K variant)')
  }
  const splitK = int8Mode ? 0 : R.splitK
  console.log(
    `[engine] attention=${R.attentionLabel} argmax=${R.argmaxLabel} qkv=${R.qkvLabel} ` +
    `ffn=${R.ffnLabel} matmul=${R.matmulLabel} rowsPerWG=${R.matmulRowsPerWG} ` +
    `mode=${fused ? (int8Mode ? 'fused+int8KV' : 'fused') : 'unfused'}`
  )

  // Activation buffers. The per-layer QKV stage differs by mode:
  //   unfused    — qkv matmul → qkvOut, rope splits it into qOut/kOut/vOut,
  //                kv_append copies kOut/vOut into kvPages
  //   fused f16  — qkv_fused writes qOut + kvPages[L] directly
  //   fused int8 — qkv_fused_scratch writes qOut + kSlot/vSlot, kv_quantize
  //                packs them into int8 pages + f16 scales
  const B = {
    residual:  makeBuf(device, PHI3.D * 2, 'residual'),      // running residual (ping)
    residual2: makeBuf(device, PHI3.D * 2, 'residual2'),     // running residual (pong)
    hidden1:   makeBuf(device, PHI3.D * 2, 'hidden1'),       // normed scratch
    hidden2:   makeBuf(device, PHI3.D * 2, 'hidden2'),       // matmul output scratch
    qOut:      makeBuf(device, PHI3.D * 2, 'qOut'),
    attnOut:   makeBuf(device, PHI3.D * 2, 'attnOut'),
    ffnOut:    makeBuf(device, PHI3.FFN * 2, 'ffnOut'),
    logits:    makeBuf(device, PHI3.VOCAB * 4, 'logits'),
    tokenOut:  makeBuf(device, 4, 'tokenOut'),
    inputIds:  makeBuf(device, 4, 'inputIds'),
    posMap:    makeBuf(device, 4, 'posMap'),
    pageIndptr: makeBuf(device, 8, 'pageIndptr'),
    pageValues: makeBuf(device, PHI3.MAX_PAGES * 4, 'pageValues'),
    lengthInfo: makeBuf(device, 12, 'lengthInfo'),
    // Mode-conditional entries (see above)
    qkvOut: null as GPUBuffer | null,
    kOut:   null as GPUBuffer | null,
    vOut:   null as GPUBuffer | null,
    kSlot:  null as GPUBuffer | null,
    vSlot:  null as GPUBuffer | null,
    // ?fuseprologue=1: ffnDown writes here so add3_norm can still see the
    // o-proj delta in hidden2 (addNorm1's residual sum never materializes).
    hidden3: null as GPUBuffer | null,
    // ?splitk=N: per-(head, partition) online-softmax partials
    // [m, d, o[HEAD_DIM]] in f32, merged by attention_combine.
    attnPartials: null as GPUBuffer | null,
  }
  if (!fused) {
    B.qkvOut = makeBuf(device, PHI3.QKV_DIM * 2, 'qkvOut')
    B.kOut   = makeBuf(device, PHI3.D * 2, 'kOut')
    B.vOut   = makeBuf(device, PHI3.D * 2, 'vOut')
  }
  if (int8Mode) {
    B.kSlot = makeBuf(device, PHI3.D * 2, 'kSlot')
    B.vSlot = makeBuf(device, PHI3.D * 2, 'vSlot')
  }
  if (R.ffnPrologue) {
    B.hidden3 = makeBuf(device, PHI3.D * 2, 'hidden3')
  }
  if (splitK) {
    B.attnPartials = makeBuf(device, PHI3.HEADS * splitK * (PHI3.HEAD_DIM + 2) * 4, 'attnPartials')
  }

  // Static uniforms — matmul shapes (K_packed, scales_per_row, M)
  const qkvU   = fused ? null : uniformBuf(device, [u32(QKV_K_PACKED), u32(QKV_SCALES), u32(PHI3.QKV_DIM)])
  const oProjU = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(PHI3.D)])
  const ffnDnU = uniformBuf(device, [u32(FFN_DN_K_PACKED), u32(FFN_DN_SCALES), u32(PHI3.D)])
  const lmHdU  = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(PHI3.VOCAB)])
  const embU   = uniformBuf(device, [u32(1), u32(D_WGS)])
  const normU  = uniformBuf(device, [u32(1)])
  const ffnU   = uniformBuf(device, [u32(PHI3.FFN)])
  const argmaxU = uniformBuf(device, [u32(PHI3.VOCAB)])

  // Hoisted per-layer uniforms — all fields are constant across tokens.
  // Unfused: rope + kv_append. Fused: qkv_fused (f16) or scratch + quantize (int8).
  const ropeU  = fused ? null : uniformBuf(device, [i32(1), i32(0), i32(1), u32(QKV_WGS)])
  const kvAppU = fused ? null : uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(D_WGS)])
  const qkvFusedU = (fused && !int8Mode) ? uniformBuf(device, [i32(0), i32(0), u32(4608)]) : null      // f16-KV mode
  const qkvFusedScratchU = int8Mode ? uniformBuf(device, [i32(0), u32(4608)]) : null                   // int8-KV mode
  const kvQuantU = int8Mode ? uniformBuf(device, [i32(0), i32(0), i32(0), u32(64)]) : null             // pos_off, pages_off, scales_off, 64 WG

  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)

  // Hard KV ceiling — writing a slot at or past this position would run off
  // the last page and silently corrupt the cache.
  const MAX_CONTEXT = PHI3.MAX_PAGES * PHI3.PAGE_SIZE

  // Attention uniform (f16-KV mode): 9 fields, 36 bytes, padded to 48.
  // nnz_pages at offset 8 is the only per-token field; the rest are constant.
  const attnU = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'attnU',
  })
  {
    const init = new ArrayBuffer(48)
    const dv = new DataView(init)
    dv.setInt32(0, 1, true)                    // batch
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_num_pages
    // offset 8: nnz_pages — updated per token via writeBuffer
    dv.setInt32(12, 0, true)                   // pages_elem_offset
    dv.setInt32(16, 0, true)                   // page_indptr_elem_offset
    dv.setInt32(20, 0, true)                   // page_values_elem_offset
    dv.setInt32(24, 0, true)                   // length_info_elem_offset
    dv.setFloat32(28, SM_SCALE, true)          // sm_scale
    dv.setUint32(32, 1, true)                  // packGridDimX
    device.queue.writeBuffer(attnU, 0, init)
  }

  // Split-K attention uniforms (?splitk=N): the partial pass mirrors the
  // f16 attention PODArgs with packGridDimX replaced by num_splits (same
  // nnz_pages slot at byte offset 8, updated per token alongside attnU);
  // the combine pass needs only num_splits.
  let attnSkU: GPUBuffer | null = null
  let combineU: GPUBuffer | null = null
  if (splitK) {
    attnSkU = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'attnSkU',
    })
    const init = new ArrayBuffer(48)
    const dv = new DataView(init)
    dv.setInt32(0, 1, true)                    // B
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_num_pages
    // offset 8: nnz_pages — updated per token via writeBuffer
    dv.setInt32(12, 0, true)                   // pages_elem_offset
    dv.setInt32(16, 0, true)                   // page_indptr_elem_offset
    dv.setInt32(20, 0, true)                   // page_values_elem_offset
    dv.setInt32(24, 0, true)                   // length_info_elem_offset
    dv.setFloat32(28, SM_SCALE, true)          // sm_scale
    dv.setUint32(32, splitK, true)             // num_splits
    device.queue.writeBuffer(attnSkU, 0, init)
    combineU = uniformBuf(device, [u32(splitK)])
  }

  // Attention uniform (int8-KV mode): extra scales_elem_offset field; 10 fields,
  // 40 bytes, padded to 48.
  let attnI8U: GPUBuffer | null = null
  if (int8Mode) {
    attnI8U = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'attnI8U' })
    const tmpl = new ArrayBuffer(48)
    const dv = new DataView(tmpl)
    dv.setInt32(0, 1, true)                    // B
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_num_pages
    // offset 8: nnz_pages — rewritten per token
    dv.setInt32(12, 0, true)                   // pages_elem_offset
    dv.setInt32(16, 0, true)                   // page_indptr_elem_offset
    dv.setInt32(20, 0, true)                   // page_values_elem_offset
    dv.setInt32(24, 0, true)                   // length_info_elem_offset
    dv.setInt32(28, 0, true)                   // scales_elem_offset
    dv.setFloat32(32, SM_SCALE, true)          // sm_scale
    dv.setUint32(36, 1, true)                  // packGridDimX
    device.queue.writeBuffer(attnI8U, 0, tmpl)
  }
  const nnzPagesScratch = new Uint32Array(1)   // reused per token for writeBuffer

  // Token readback buffer for the blocking path — allocated once, reused
  // every decode step.
  const readBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'tokenReadback',
  })

  // Logit readback buffer — used by forwardLogits() for the validation harness only.
  // Allocated lazily on first call.
  let logitsReadBuf: GPUBuffer | null = null

  // Initialize identity page table (page i → physical page i)
  {
    const pageVals = new Int32Array(PHI3.MAX_PAGES)
    for (let i = 0; i < PHI3.MAX_PAGES; i++) pageVals[i] = i
    device.queue.writeBuffer(B.pageValues, 0, pageVals)
  }

  // ============================================================
  // Pre-computed bind groups
  //
  // Bind group contents are deterministic (same buffers, same layout) so we
  // hoist them out of the hot loop — ~10 per layer × 32 layers otherwise.
  //
  // The residual ping-pong is also deterministic: every layer reads `residual`
  // and writes `residual2` in addNorm1 (post-attention), then reads `residual2`
  // and writes `residual` in addNorm2 (post-FFN). Two swaps per layer return to
  // the same state, so the bind groups are identical for every layer.
  // ============================================================

  // Global (non-per-layer) bind groups
  const bgEmbedding = bg(device, P.embedding, [
    B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU,
  ])
  const bgInitNorm = bg(device, P.rmsNorm, [
    B.hidden1, B.residual, weights.layers[0].normGamma1, normU,
  ])
  const bgRope = fused ? null : bg(device, P.rope, [
    B.qOut, B.kOut!, B.vOut!, B.qkvOut!, B.posMap, ropeU!,
  ])
  const bgLmHead = bg(device, R.matmulF32, [
    B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU,
  ])
  const bgArgmax = bg(device, R.argmax, [
    B.logits, B.tokenOut, argmaxU,
  ])
  // Split-K combine reads the shared partials scratch and writes attnOut —
  // identical for every layer, so one bind group serves all 32.
  const bgAttnCombine = splitK
    ? bg(device, R.attentionCombine!, [B.attnPartials!, B.attnOut, combineU!])
    : null

  // Per-layer bind groups
  interface LayerBG {
    qkv: GPUBindGroup           // unfused: qkv matmul | fused: qkv_fused / qkv_fused_scratch
    kvApp?: GPUBindGroup        // unfused only
    kvQuantize?: GPUBindGroup   // int8 mode only
    attn: GPUBindGroup          // splitK: the partial pass (writes attnPartials)
    oProj: GPUBindGroup
    addNorm1?: GPUBindGroup     // post-attention: reads residual, writes residual2 (absent w/ ?fuseprologue=1)
    ffn: GPUBindGroup           // prologue mode: reads (rIn, hidden2, gamma) instead of hidden1
    ffnDown: GPUBindGroup       // prologue mode: writes hidden3 (hidden2 must survive for add3_norm)
    addNorm2: GPUBindGroup      // post-FFN: reads residual2, writes residual (prologue mode: add3_norm)
  }
  const layerBGs: LayerBG[] = []
  for (let L = 0; L < PHI3.LAYERS; L++) {
    const lw = weights.layers[L]
    const nextGamma = L < PHI3.LAYERS - 1
      ? weights.layers[L + 1].normGamma1
      : weights.finalNormGamma

    let qkvBG: GPUBindGroup
    let attnBG: GPUBindGroup
    let kvAppBG: GPUBindGroup | undefined
    let kvQuantizeBG: GPUBindGroup | undefined
    // Split-K partial pass replaces the single-WG-per-head attention bind
    // group in the f16-KV modes (int8 has no split-K variant).
    const attnF16BG = () => splitK
      ? bg(device, R.attentionSplitK!, [
          B.qOut, B.pageIndptr, B.pageValues, kvPages[L], B.lengthInfo, B.attnPartials!, attnSkU!,
        ])
      : bg(device, R.attention, [
          B.qOut, B.pageIndptr, B.pageValues, kvPages[L], B.lengthInfo, B.attnOut, attnU,
        ])
    if (!fused) {
      qkvBG = bg(device, R.matmul, [
        B.qkvOut!, B.hidden1, lw.qkvScales, lw.qkvWeights, qkvU!,
      ])
      kvAppBG = bg(device, P.kvAppend, [
        B.kOut!, B.vOut!, kvPages[L], B.posMap, kvAppU!,
      ])
      attnBG = attnF16BG()
    } else if (int8Mode) {
      qkvBG = bg(device, P.qkvFusedScratch, [
        B.qOut, B.kSlot!, B.vSlot!, B.hidden1, lw.qkvScales, lw.qkvWeights, B.posMap, qkvFusedScratchU!,
      ])
      kvQuantizeBG = bg(device, P.kvQuantizeInt8, [
        B.kSlot!, B.vSlot!, kvPages[L], kvScales![L], B.posMap, kvQuantU!,
      ])
      attnBG = bg(device, P.attentionInt8, [
        B.qOut, B.pageIndptr, B.pageValues, kvPages[L], kvScales![L],
        B.lengthInfo, B.attnOut, attnI8U!,
      ])
    } else {
      qkvBG = bg(device, R.qkvFused, [
        B.qOut, kvPages[L], B.hidden1, lw.qkvScales, lw.qkvWeights, B.posMap, qkvFusedU!,
      ])
      attnBG = attnF16BG()
    }

    if (R.ffnPrologue) {
      // ?fuseprologue=1: addNorm1 is gone, so there is only ONE residual
      // hand-off per layer and the ping-pong alternates by layer parity
      // (two swaps per layer collapse to one). The FFN kernel computes
      // rIn + hidden2 and its RMSNorm into shared memory itself; add3_norm
      // reconstructs the residual sum (rIn + oproj + ffnDown) at the tail —
      // which is why ffnDown must write hidden3, not hidden2.
      const rIn  = L % 2 === 0 ? B.residual : B.residual2
      const rOut = L % 2 === 0 ? B.residual2 : B.residual
      layerBGs.push({
        qkv: qkvBG,
        kvApp: kvAppBG,
        kvQuantize: kvQuantizeBG,
        attn: attnBG,
        oProj: bg(device, R.matmul, [B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU]),
        ffn: bg(device, R.ffn, [B.ffnOut, rIn, B.hidden2, lw.normGamma2, lw.ffnScales, lw.ffnWeights, ffnU]),
        ffnDown: bg(device, R.matmul, [B.hidden3!, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU]),
        addNorm2: bg(device, P.add3Norm, [rIn, B.hidden2, B.hidden3!, nextGamma, B.hidden1, rOut, normU]),
      })
      continue
    }

    layerBGs.push({
      qkv: qkvBG,
      kvApp: kvAppBG,
      kvQuantize: kvQuantizeBG,
      attn: attnBG,
      oProj: bg(device, R.matmul, [B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU]),
      addNorm1: bg(device, P.addNorm, [B.hidden2, B.residual, lw.normGamma2, B.hidden1, B.residual2, normU]),
      ffn: bg(device, R.ffn, [B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU]),
      ffnDown: bg(device, R.matmul, [B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU]),
      addNorm2: bg(device, P.addNorm, [B.hidden2, B.residual2, nextGamma, B.hidden1, B.residual, normU]),
    })
  }

  // Profile handle — null unless profileStep() is currently running.
  // `dispatch()` reads this on every call; when set it instruments the pass.
  let profile: ProfileState | null = null

  function dispatch(
    enc: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    wgX: number, wgY = 1, wgZ = 1,
    label?: string,
  ): void {
    let desc: GPUComputePassDescriptor | undefined
    if (profile && label) {
      const begin = profile.nextSlot
      const end = begin + 1
      if (end < profile.capacity) {
        profile.labels[begin] = label
        profile.nextSlot = end + 1
        desc = {
          timestampWrites: {
            querySet: profile.querySet,
            beginningOfPassWriteIndex: begin,
            endOfPassWriteIndex: end,
          },
        }
      }
    }
    const pass = enc.beginComputePass(desc)
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(wgX, wgY, wgZ)
    pass.end()
  }

  /**
   * Record one full forward pass (embedding → 32 layers → LM head → argmax)
   * into `enc`. Both generate styles and profileStep share this recorder, so
   * dispatch order is identical everywhere.
   */
  function recordForward(enc: GPUCommandEncoder): void {
    // --- EMBEDDING → B.residual (ping) ---
    dispatch(enc, P.embedding, bgEmbedding, D_WGS, 1, 1, 'embedding')
    // --- INITIAL RMSNORM: B.residual → B.hidden1 (layer 0's normGamma1) ---
    dispatch(enc, P.rmsNorm, bgInitNorm, 1, 1, 1, 'rmsNorm_init')

    // --- 32 TRANSFORMER LAYERS ---
    // Residual ping-pong is encoded into the cached bind groups: addNorm1 reads
    // residual / writes residual2; addNorm2 reads residual2 / writes residual.
    // Unfused: 9 dispatches/layer. Fused f16: 7. Fused int8: 8 (adds a
    // kv_quantize pass between qkv_fused_scratch and attention_int8).
    for (let L = 0; L < PHI3.LAYERS; L++) {
      const blk = layerBGs[L]

      // Attention on the f16 KV layout — split-K (?splitk=N) turns the one
      // WG-per-head dispatch into a partial pass over N partitions per head
      // plus a per-head combine over the partials scratch.
      const attentionF16 = () => {
        if (splitK) {
          dispatch(enc, R.attentionSplitK!, blk.attn, splitK, PHI3.HEADS, 1, 'attention')
          dispatch(enc, R.attentionCombine!, bgAttnCombine!, 1, PHI3.HEADS, 1, 'attnCombine')
        } else {
          dispatch(enc, R.attention, blk.attn, 1, PHI3.HEADS, 1, 'attention')
        }
      }

      if (!fused) {
        // QKV matmul: B.hidden1 → B.qkvOut
        dispatch(enc, R.matmul, blk.qkv, PHI3.QKV_DIM / R.matmulRowsPerWG, 1, 1, 'qkvMatmul')
        // RoPE: B.qkvOut → B.qOut, B.kOut, B.vOut
        dispatch(enc, P.rope, bgRope!, QKV_WGS, 1, 1, 'rope')
        // KV append: kOut, vOut → kvPages[L]
        dispatch(enc, P.kvAppend, blk.kvApp!, D_WGS, 1, 1, 'kvAppend')
        // Attention: Q + kvPages[L] → B.attnOut
        attentionF16()
      } else if (int8Mode) {
        dispatch(enc, P.qkvFusedScratch, blk.qkv, 4608, 1, 1, 'qkvFused')
        dispatch(enc, P.kvQuantizeInt8, blk.kvQuantize!, 64, 1, 1, 'kvQuantize')
        dispatch(enc, P.attentionInt8, blk.attn, 1, PHI3.HEADS, 1, 'attention')
      } else {
        // Fused QKV+RoPE+KV-append: B.hidden1 → B.qOut + kvPages[L]
        dispatch(enc, R.qkvFused, blk.qkv, 4608 / R.qkvPairsPerWG, 1, 1, 'qkvFused')
        attentionF16()
      }

      // O projection: B.attnOut → B.hidden2
      dispatch(enc, R.matmul, blk.oProj, PHI3.D / R.matmulRowsPerWG, 1, 1, 'oproj')
      if (R.ffnPrologue) {
        // ?fuseprologue=1: no addNorm1 dispatch. The FFN kernel computes
        // rIn + hidden2 and its RMSNorm in its own prologue; ffnDown lands
        // in hidden3; add3_norm merges rIn + hidden2 + hidden3 at the tail.
        dispatch(enc, R.ffn, blk.ffn, PHI3.FFN / R.ffnRowsPerWG, 1, 1, 'fusedFfn')
        dispatch(enc, R.matmul, blk.ffnDown, PHI3.D / R.matmulRowsPerWG, 1, 1, 'ffnDown')
        dispatch(enc, P.add3Norm, blk.addNorm2, 1, 1, 1, 'addNorm2')
      } else {
        // AddNorm (attention): residual += hidden2; hidden1 = RMSNorm(residual)
        dispatch(enc, P.addNorm, blk.addNorm1!, 1, 1, 1, 'addNorm1')
        // Fused FFN gate+up+SiLU: B.hidden1 → B.ffnOut
        dispatch(enc, R.ffn, blk.ffn, PHI3.FFN / R.ffnRowsPerWG, 1, 1, 'fusedFfn')
        // FFN down: B.ffnOut → B.hidden2
        dispatch(enc, R.matmul, blk.ffnDown, PHI3.D / R.matmulRowsPerWG, 1, 1, 'ffnDown')
        // AddNorm (FFN): residual += hidden2; hidden1 = RMSNorm(residual)
        //   For last layer the bind group binds finalNormGamma instead of next
        //   layer's normGamma1, so hidden1 is ready for the LM head.
        dispatch(enc, P.addNorm, blk.addNorm2, 1, 1, 1, 'addNorm2')
      }
    }

    // --- LM HEAD: B.hidden1 (already normalized with model.norm) → B.logits ---
    // PHI3.VOCAB=32064 is divisible by 4, so rowsPerWG=4 works exactly.
    dispatch(enc, R.matmulF32, bgLmHead, PHI3.VOCAB / R.matmulRowsPerWG, 1, 1, 'lmHead')
    // --- ARGMAX: B.logits → B.tokenOut ---
    dispatch(enc, R.argmax, bgArgmax, 1, 1, 1, 'argmax')
  }

  /**
   * Write the per-token state buffers. Queue-ordered with the following
   * submit; the GPU reads these during compute, and any later writeBuffer for
   * the same buffer is serialized after the current submit — so this is safe
   * even while previous submits are still in flight.
   */
  function writeStepState(inputId: number | null, position: number): void {
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1
    if (inputId !== null) {
      device.queue.writeBuffer(B.inputIds, 0, new Int32Array([inputId]))
    }
    device.queue.writeBuffer(B.posMap, 0, new Int32Array([position]))
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    // length_info: total tokens in sequence = position + 1
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))
    // nnz_pages lives at byte offset 8 in both f16 and int8 attention uniforms.
    nnzPagesScratch[0] = nnzPages
    device.queue.writeBuffer(int8Mode ? attnI8U! : attnU, 8, nnzPagesScratch)
    // The split-K partial-pass uniform mirrors the same layout (offset 8).
    if (attnSkU) device.queue.writeBuffer(attnSkU, 8, nnzPagesScratch)
  }

  // ============================================================
  // Blocking path — decodeToken / forwardLogits (validation harness)
  // ============================================================

  async function decodeToken(tokenId: number, position: number): Promise<number> {
    if (position < 0 || position >= MAX_CONTEXT) {
      throw new Error(
        `zero-tvm: context overflow — position ${position} exceeds max context ` +
        `${MAX_CONTEXT} tokens (PHI3.MAX_PAGES=${PHI3.MAX_PAGES} × PAGE_SIZE=${PHI3.PAGE_SIZE}). ` +
        `Shorten the prompt or raise MAX_PAGES in src/compiler/compiler.ts (costs ~${Math.round(PHI3.LAYERS * 196608 / (1024 * 1024))} MB of KV cache per page block).`
      )
    }
    writeStepState(tokenId, position)

    const enc = device.createCommandEncoder()
    recordForward(enc)
    // Fold the argmax readback into the same command encoder → one submit per token.
    enc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
    device.queue.submit([enc.finish()])

    await readBuf.mapAsync(GPUMapMode.READ)
    const result = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    return result
  }

  /**
   * Run the same forward pass as decodeToken but read back the f32 logits buffer
   * instead of the argmax token. Used by the validation harness to compare
   * Zero-TVM logits against WebLLM logits at every token position.
   */
  async function readLogits(tokenId: number, position: number): Promise<Float32Array> {
    // Reuse decodeToken's GPU work — easiest way is to call it (it already
    // submits the command encoder) and then issue a separate readback of B.logits.
    // The argmax dispatch is harmless extra work; we ignore its output.
    await decodeToken(tokenId, position)

    if (!logitsReadBuf) {
      logitsReadBuf = device.createBuffer({
        size: PHI3.VOCAB * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'logitsReadback',
      })
    }
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.logits, 0, logitsReadBuf, 0, PHI3.VOCAB * 4)
    device.queue.submit([enc.finish()])

    await logitsReadBuf.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(logitsReadBuf.getMappedRange().slice(0))
    logitsReadBuf.unmap()
    return out
  }

  const STOP = new Set([2, 32000, 32007])

  async function generate(
    promptIds: number[],
    startPos: number,
    maxTokens: number,
    onToken: (id: number) => void
  ): Promise<number[]> {
    const tokens: number[] = []

    // Prefill from startPos to populate KV cache for the new tokens.
    // KV slots [0, startPos) already contain valid entries from the previous
    // turn (caller guarantees prompt[0..startPos] matches what was prefilled
    // before). The last call's return is the argmax over the final prefill
    // step's logits — that *is* the first generated token.
    let tokenId = 0
    if (startPos >= promptIds.length) {
      // The new prompt is a strict prefix of the previous one (or identical).
      // No new tokens to prefill — but we still need a valid `tokenId` to
      // start decoding from. Run the last prompt token through decodeToken at
      // its existing position to re-read the logits.
      tokenId = await decodeToken(promptIds[promptIds.length - 1], promptIds.length - 1)
    } else {
      for (let i = startPos; i < promptIds.length; i++) {
        tokenId = await decodeToken(promptIds[i], i)
      }
    }

    // Decode loop. Each emitted token is fed back at the next free KV slot:
    // the first generated token decodes at position promptIds.length (the
    // slot right after the prompt), then the position advances by one.
    let pos = promptIds.length
    for (let i = 0; i < maxTokens; i++) {
      if (tokenId < 0 || tokenId >= PHI3.VOCAB || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken(tokenId)
      tokenId = await decodeToken(tokenId, pos)
      pos++
    }

    return tokens
  }

  /**
   * Forward pass for validation. Always prefills from position 0 (no KV reuse)
   * and returns the f32 logits at the final prompt position. The argmax of
   * these logits is the model's next-token prediction for the prompt.
   */
  async function forwardLogits(promptIds: number[]): Promise<Float32Array> {
    if (promptIds.length === 0) throw new Error('forwardLogits: empty prompt')
    // Prefill all but the last token.
    for (let i = 0; i < promptIds.length - 1; i++) {
      await decodeToken(promptIds[i], i)
    }
    // Final token — read back logits instead of argmax.
    return readLogits(promptIds[promptIds.length - 1], promptIds.length - 1)
  }

  function resetKVTracking(): void {
    // The KV cache is just a buffer — there is no metadata to clear here.
    // Callers track their own prefix length and pass startPos to generate().
    // This method exists so callers can be explicit about conversation resets.
  }

  // ============================================================
  // Pipelined path — submitStep / generatePipelined (chat)
  // ============================================================

  // Persistent readback ring (size = decode pipeline depth).
  // Each slot is a 4-byte MAP_READ buffer receiving a copy of B.tokenOut.
  const PIPELINE_DEPTH = 2
  const readRing: GPUBuffer[] = []
  for (let i = 0; i < PIPELINE_DEPTH; i++) {
    readRing.push(device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }))
  }
  let readCursor = 0

  /**
   * Submit one forward pass.
   *
   * If `writeInputId` is non-null, it's written to B.inputIds[0] before the pass
   * (prefill path). Otherwise the pass reads whatever argmax from the previous
   * submission wrote there (decode chain).
   *
   * If `wantReadback` is true, a copy of B.tokenOut is routed to a readback
   * slot and a Promise resolving to the token ID is returned. Otherwise the
   * pass is fire-and-forget (the GPU chain still runs).
   *
   * Single submit per call: compute + argmax + GPU chain + (optional) readback
   * copy are all in one command encoder.
   */
  function submitStep(
    writeInputId: number | null,
    position: number,
    wantReadback: boolean
  ): Promise<number> | null {
    writeStepState(writeInputId, position)

    const enc = device.createCommandEncoder()
    recordForward(enc)
    // GPU chain: next decode step reads inputIds[0] without a CPU round-trip.
    enc.copyBufferToBuffer(B.tokenOut, 0, B.inputIds, 0, 4)

    if (wantReadback) {
      const slot = readRing[readCursor]
      readCursor = (readCursor + 1) % readRing.length
      enc.copyBufferToBuffer(B.tokenOut, 0, slot, 0, 4)
      device.queue.submit([enc.finish()])
      return slot.mapAsync(GPUMapMode.READ).then(() => {
        const id = new DataView(slot.getMappedRange()).getInt32(0, true)
        slot.unmap()
        return id
      })
    }

    device.queue.submit([enc.finish()])
    return null
  }

  async function generatePipelined(
    promptIds: number[],
    maxTokens: number,
    onToken: (id: number) => void,
    shouldStop?: () => boolean
  ): Promise<number[]> {
    const tokens: number[] = []
    if (promptIds.length === 0 || maxTokens <= 0) return tokens
    // Refuse oversized prompts up front — prefilling at position >= MAX_CONTEXT
    // would write past the last KV page and silently corrupt the cache.
    if (promptIds.length >= MAX_CONTEXT) {
      throw new Error(
        `zero-tvm: prompt (${promptIds.length} tokens) exceeds max context ` +
        `${MAX_CONTEXT} (PHI3.MAX_PAGES=${PHI3.MAX_PAGES} × PAGE_SIZE=${PHI3.PAGE_SIZE})`
      )
    }

    // --- Prefill ---
    // Fire all but the last step without readback. The argmax of intermediate
    // prefill steps is not useful (we overwrite inputIds on the next step), so
    // skipping readback removes N-1 CPU syncs.
    for (let i = 0; i < promptIds.length - 1; i++) {
      submitStep(promptIds[i], i, false)
    }
    // Last prefill step: readback to get the first generated token.
    const firstTokenPromise = submitStep(
      promptIds[promptIds.length - 1],
      promptIds.length - 1,
      true,
    )!
    const firstToken = await firstTokenPromise
    if (STOP.has(firstToken) || firstToken < 0 || firstToken >= PHI3.VOCAB) return tokens
    tokens.push(firstToken)
    onToken(firstToken)
    if (tokens.length >= maxTokens) return tokens

    // --- Pipelined decode ---
    // Keep PIPELINE_DEPTH tokens of work in flight so the GPU never waits on
    // a CPU round-trip between tokens. argmax → inputIds[0] is chained on-GPU.
    const inFlight: Promise<number>[] = []
    let pos = promptIds.length

    try {
      for (let k = 0; k < PIPELINE_DEPTH && tokens.length + k < maxTokens && pos < MAX_CONTEXT; k++) {
        inFlight.push(submitStep(null, pos, true)!)
        pos++
      }

      while (inFlight.length > 0) {
        const tok = await inFlight.shift()!
        if (STOP.has(tok) || tok < 0 || tok >= PHI3.VOCAB) break
        tokens.push(tok)
        onToken(tok)
        if (tokens.length >= maxTokens) break
        // Cooperative stop: never unwind mid-pipeline — just stop submitting;
        // the finally-drain below leaves the readback ring clean.
        if (shouldStop?.()) break
        // Keep the pipeline full, but don't overshoot maxTokens or the KV window.
        if (tokens.length + inFlight.length < maxTokens && pos < MAX_CONTEXT) {
          inFlight.push(submitStep(null, pos, true)!)
          pos++
        }
      }
    } finally {
      // Drain pending readbacks (frees ring slots for the next generation —
      // mapAsync on a still-pending slot buffer would be a validation error).
      while (inFlight.length > 0) {
        await inFlight.shift()!.catch(() => {})
      }
    }

    return tokens
  }

  // ============================================================
  // Profiling + bench primitives
  // ============================================================

  // Instrument a single forward pass with timestamp-query. Runs the warmup
  // prefill without instrumentation (so we're measuring a steady-state decode
  // step, not cold-compile time), then records one more step with a QuerySet
  // and reads back begin/end timestamps per dispatch.
  //
  // Returns null if the 'timestamp-query' feature wasn't enabled at device
  // creation — the caller prints a hint in that case.
  async function profileStep(warmupIds: number[]): Promise<KernelProfile | null> {
    if (!(device.features as ReadonlySet<string>).has('timestamp-query')) return null

    // Warmup: run prefill + a handful of decode steps without profile, so
    // shader compilation, texture caches, etc. are warm when we measure.
    for (let i = 0; i < warmupIds.length; i++) {
      submitStep(warmupIds[i], i, false)
    }
    // Two decode warmup steps, then await so the GPU is idle before we
    // allocate the query set.
    const warmA = submitStep(null, warmupIds.length, true)!
    const warmB = submitStep(null, warmupIds.length + 1, true)!
    await warmA; await warmB
    const nextPos = warmupIds.length + 2

    const CAPACITY = 600  // 2 slots × (~228 fused / ~260 int8 / ~292 unfused passes)
    const querySet = device.createQuerySet({ type: 'timestamp', count: CAPACITY })
    const resolveBuf = device.createBuffer({
      size: CAPACITY * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    const profReadBuf = device.createBuffer({
      size: CAPACITY * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })

    profile = { querySet, capacity: CAPACITY, labels: new Array(CAPACITY), nextSlot: 0 }
    try {
      writeStepState(null, nextPos)

      const enc = device.createCommandEncoder()
      recordForward(enc)
      const usedSlots = profile.nextSlot
      enc.resolveQuerySet(querySet, 0, usedSlots, resolveBuf, 0)
      enc.copyBufferToBuffer(resolveBuf, 0, profReadBuf, 0, usedSlots * 8)
      device.queue.submit([enc.finish()])

      await profReadBuf.mapAsync(GPUMapMode.READ)
      const raw = new BigInt64Array(profReadBuf.getMappedRange().slice(0))
      profReadBuf.unmap()

      // Aggregate: slot 2k = begin, 2k+1 = end, label at labels[2k].
      // Very short passes can read identical begin/end timestamps (duration
      // below the GPU timer resolution) or — rarely — negative on
      // counter-domain quirks. Clamp to zero rather than drop, so the call
      // count still reflects every dispatch we issued.
      const totals = new Map<string, { ns: bigint; calls: number }>()
      let totalNs = 0n
      for (let i = 0; i + 1 < usedSlots; i += 2) {
        const label = profile.labels[i]
        if (!label) continue
        const raw_dur = raw[i + 1] - raw[i]
        const dur = raw_dur > 0n ? raw_dur : 0n
        const entry = totals.get(label) ?? { ns: 0n, calls: 0 }
        entry.ns += dur
        entry.calls += 1
        totals.set(label, entry)
        totalNs += dur
      }
      const totalMs = Number(totalNs) / 1e6
      const kernels = [...totals.entries()]
        .map(([label, v]) => ({
          label,
          totalMs: Number(v.ns) / 1e6,
          calls: v.calls,
          pctOfTotal: totalMs > 0 ? (Number(v.ns) / Number(totalNs)) * 100 : 0,
        }))
        .sort((a, b) => b.totalMs - a.totalMs)
      return { kernels, totalMs }
    } finally {
      profile = null
      querySet.destroy()
      resolveBuf.destroy()
      profReadBuf.destroy()
    }
  }

  // Isolated falsifiability test for the batched int4 matmul primitive before
  // we invest in porting 4 more shaders (fused_ffn, qkv_fused, attention,
  // argmax). Runs two timed passes on real ffnDown weights:
  //
  //   batched:  `iters` dispatches of int4MatmulBatchedM4 (each produces M×N)
  //   tiled×M:  `iters * M` dispatches of the current tiled matmul (each 1×N)
  //
  // If weight reuse lands, batched should approach 1× the cost of a single
  // tiled dispatch — i.e. speedup → M. If Apple can't actually cache the
  // reused weight tile across 16 f32 accumulators × 4 batch rows, speedup
  // collapses to ~1× and the batched-forward plan in RESEARCH.md dies here.
  async function benchBatchedFfnDown(
    M: number = 4,
    iters: number = 500,
    target: 'ffnDown' | 'oproj' = 'ffnDown',
  ): Promise<BatchedBenchResult | null> {
    if (!P.int4MatmulBatchedM4) {
      console.warn('[batched-bench] int4MatmulBatchedM4 unavailable (subgroups off)')
      return null
    }
    if (M !== 4) {
      console.warn(`[batched-bench] M=${M} unsupported by batched shader (TILE_M=4 hardcoded); forcing M=4`)
      M = 4
    }
    // Target matmul: ffnDown has K=8192, oproj has K=3072.
    const isFfnDown = target === 'ffnDown'
    const K = isFfnDown ? 8192 : 3072
    const N = 3072
    const uniformBufForTarget = isFfnDown ? ffnDnU : oProjU
    const scalesBuf = isFfnDown ? weights.layers[0].ffnDownScales : weights.layers[0].oProjScales
    const weightsBuf = isFfnDown ? weights.layers[0].ffnDownWeights : weights.layers[0].oProjWeights
    const batchedPipeline = P.int4MatmulBatchedM4

    const inBuf  = makeBuf(device, M * K * 2, 'batchedBench.in')
    const outBuf = makeBuf(device, M * N * 2, 'batchedBench.out')
    const tiledOutBuf = makeBuf(device, N * 2, 'batchedBench.tiledOut')

    const inf16 = new Uint16Array(M * K)
    for (let i = 0; i < inf16.length; i++) inf16[i] = 0x2a00 | (i & 0x03ff)
    device.queue.writeBuffer(inBuf, 0, inf16)

    const batchedBG = bg(device, batchedPipeline,
      [outBuf, inBuf, scalesBuf, weightsBuf, uniformBufForTarget])
    const tiledBG = bg(device, R.matmul,
      [tiledOutBuf, inBuf, scalesBuf, weightsBuf, uniformBufForTarget])

    const batchedWGs = N / 4
    const tiledWGs   = N / R.matmulRowsPerWG

    {
      const enc = device.createCommandEncoder()
      for (let i = 0; i < 20; i++) {
        const p1 = enc.beginComputePass()
        p1.setPipeline(batchedPipeline); p1.setBindGroup(0, batchedBG)
        p1.dispatchWorkgroups(batchedWGs); p1.end()
        const p2 = enc.beginComputePass()
        p2.setPipeline(R.matmul); p2.setBindGroup(0, tiledBG)
        p2.dispatchWorkgroups(tiledWGs); p2.end()
      }
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }

    const tB0 = performance.now()
    {
      const enc = device.createCommandEncoder()
      for (let i = 0; i < iters; i++) {
        const pass = enc.beginComputePass()
        pass.setPipeline(batchedPipeline); pass.setBindGroup(0, batchedBG)
        pass.dispatchWorkgroups(batchedWGs); pass.end()
      }
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }
    const msBatched = performance.now() - tB0

    const tT0 = performance.now()
    {
      const enc = device.createCommandEncoder()
      for (let i = 0; i < iters * M; i++) {
        const pass = enc.beginComputePass()
        pass.setPipeline(R.matmul); pass.setBindGroup(0, tiledBG)
        pass.dispatchWorkgroups(tiledWGs); pass.end()
      }
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
    }
    const msTiledTotal = performance.now() - tT0

    const msPerMBatched = msBatched / iters
    const msPerMTiled   = msTiledTotal / iters
    const speedup = msPerMTiled / msPerMBatched

    // Correctness: dispatch once, read back first 32 f16 of outBuf, compute a
    // byte-sum. If the shader returns early or fails silently, sum will be 0.
    // A proper A/B against the default variant would match byte-exactly.
    const sampleReadBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(batchedPipeline); pass.setBindGroup(0, batchedBG)
      pass.dispatchWorkgroups(batchedWGs); pass.end()
      enc.copyBufferToBuffer(outBuf, 0, sampleReadBuf, 0, 64)
      device.queue.submit([enc.finish()])
    }
    await sampleReadBuf.mapAsync(GPUMapMode.READ)
    const sample = new Uint16Array(sampleReadBuf.getMappedRange().slice(0))
    sampleReadBuf.unmap()
    let sum = 0
    let nonZero = 0
    for (const v of sample) { sum += v; if (v !== 0) nonZero++ }
    const sampleHex = Array.from(sample).slice(0, 8).map(v => v.toString(16).padStart(4, '0')).join(' ')

    console.log(`[batched-bench] ${target} M=${M} iters=${iters}`)
    console.log(`  batched:  ${msBatched.toFixed(1)} ms total  → ${msPerMBatched.toFixed(3)} ms per M-pack`)
    console.log(`  tiled×M:  ${msTiledTotal.toFixed(1)} ms total → ${msPerMTiled.toFixed(3)} ms per M sequential`)
    console.log(`  speedup:  ${speedup.toFixed(2)}× (>1 means weight reuse amortizes; ceiling is M=${M})`)
    console.log(`  output[0..8] f16 hex: ${sampleHex}`)
    console.log(`  output[0..32] non-zero count: ${nonZero}/32, byte-sum: ${sum}`)

    sampleReadBuf.destroy()
    inBuf.destroy(); outBuf.destroy(); tiledOutBuf.destroy()
    return { msBatched, msTiledTotal, msPerMBatched, msPerMTiled, speedup }
  }

  return {
    generate,
    generatePipelined,
    forwardLogits,
    resetKVTracking,
    maxContext: MAX_CONTEXT,
    profileStep,
    benchBatchedFfnDown,
  }
}
