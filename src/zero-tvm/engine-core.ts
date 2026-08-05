/**
 * ZERO-TVM ENGINE CORE
 *
 * Pure GPU pipeline: takes a loaded device + weights + KV cache, returns THE
 * DecodeEngine. No DOM, no UI. Both shipped pages drive this one engine:
 *
 *   - validate.ts (via loading-ui.ts's bootEngine) runs the default
 *     configuration — unfused reference path (9 dispatches/layer; 10 for
 *     qkNorm specs like Qwen3, which insert a per-head Q/K RMSNorm between
 *     the QKV matmul and RoPE), scalar shaders, blocking
 *     generate()/forwardLogits() with deterministic per-token positions and
 *     logits access.
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
import { compile, PHI3, type ModelSpec } from '../compiler/compiler.js'
import { SCALAR_VARIANTS, resolveVariantPipelines, resolveMatmul, type VariantFlags } from './variants.js'

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

// Each entry is a whole buffer or a { buffer, offset, size } region view
// (offset must respect minStorageBufferOffsetAlignment — the GDN packed-
// projection regions the engine binds are all 256-aligned by construction).
type BindEntry = GPUBuffer | { buffer: GPUBuffer; offset: number; size: number }

function bg(device: GPUDevice, pipeline: GPUComputePipeline, bufs: BindEntry[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({
      binding: i,
      resource: 'buffer' in b ? (b as GPUBufferBinding) : { buffer: b as GPUBuffer },
    })),
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

export function allocKVPages(device: GPUDevice, spec: ModelSpec = PHI3): GPUBuffer[] {
  const bytesPerPage = spec.kvPageStride * 2  // kvHeads * pageSize slots * headDim * 2 (K+V) * 2 bytes
  const pages = spec.maxPages * bytesPerPage  // Phi-3: 257 * 196608 ≈ 50MB
  // One pages buffer per ATTENTION layer: hybrid specs (Qwen3.5) have KV only
  // on their 'attn' layers and the engine indexes by attention-layer ordinal.
  // Pure-attention specs have layerKinds all 'attn' → one per layer, as before.
  const attnLayers = spec.layerKinds.filter((k) => k === 'attn').length
  return Array.from({ length: attnLayers }, (_, i) =>
    makeBuf(device, pages, `kvPages_${i}`)
  )
}

// int8 layout: kvHeads × pageSize slots × 2 sides × headDim/4 u32-words per page
// (Phi-3: 24576 u32). Halves KV memory from ~1.6GB to ~800MB for 4K context, at
// the cost of one extra dispatch per layer (quantize). Opt-in (?kv8=1 on the
// chat page) — validate output parity against the f16 baseline on your target
// hardware.
export function allocKVPagesInt8(device: GPUDevice, spec: ModelSpec = PHI3): { pages: GPUBuffer[]; scales: GPUBuffer[] } {
  const bytesPerPage = spec.kvI8PageWords * 4       // Phi-3: 96 KB
  const scalesPerPage = spec.kvScalesPerPage        // Phi-3: 1024 f16 per page
  const pagesBytes = spec.maxPages * bytesPerPage
  const scalesBytes = spec.maxPages * scalesPerPage * 2
  return {
    pages: Array.from({ length: spec.layers }, (_, i) =>
      makeBuf(device, pagesBytes, `kvPagesI8_${i}`)
    ),
    scales: Array.from({ length: spec.layers }, (_, i) =>
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

// Workgroup width shared by the elementwise shaders — derived from their
// @workgroup_size: embedding/rms_norm/add_norm/kv_append/rope use 256
// threads, one wg per 256 hidden units. Matmul shaders get one wg per
// output element (M), divided by the variant's rows/WG.
const WG_SIZE_D = 256

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
  /**
   * Debug assertion for cross-turn prefix reuse: compute the reusable prefix
   * for `promptIds` against the engine's absorbed-token record, run the
   * REUSED-prefix prefill and read the final-position logits, then run a
   * FRESH full prefill of the same prompt and diff the two logit vectors.
   * Both paths are deterministic dispatch-for-dispatch, so the expected diff
   * is exactly 0. Blocking path; leaves the engine state at end-of-prompt.
   */
  debugCompareReuse(promptIds: number[]): Promise<{
    startPos: number
    promptLen: number
    maxAbsDiff: number
    meanAbsDiff: number
  }>
  /** Stats from the most recent generatePipelined prefill (reuse + chunking). */
  getLastPrefill(): { promptLen: number; reused: number; chunks: number } | null
  /** Hard KV ceiling in tokens: spec.maxPages × spec.pageSize. */
  maxContext: number
  /** The ModelSpec this engine was built for. */
  spec: ModelSpec
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
   *         Incompatible with qkNorm specs (Qwen3) — throws.
   * false → unfused reference: qkv matmul → [qk_norm →] rope → kv_append
   *         (9/layer; 10 with qkNorm).
   */
  fused?: boolean
  /** int8 KV cache. Requires `fused` and a kv argument carrying `scales`. */
  int8KV?: boolean
  /** Model shape to build for. Default: PHI3. Must match the loaded weights. */
  spec?: ModelSpec
  /**
   * Cross-turn prefix reuse in generatePipelined (default true; ?reuse=0 on
   * the chat page disables). The engine tracks the exact (position, token)
   * record of every submitted forward pass; a new prompt that extends the
   * absorbed prefix prefills only the delta. Trust rules: pure-attention
   * specs reuse the longest common prefix (KV rewrite is idempotent); hybrid
   * specs additionally require the GDN state boundary to sit exactly at the
   * end of the absorbed record (the recurrence is not rewindable) and fall
   * back to a full re-prefill otherwise.
   */
  prefixReuse?: boolean
  /**
   * Chunked GDN prefill (hybrid specs; default true; ?chunk=0 disables).
   * Requires the subgroups pipelines (int4_matmul_batched_dyn). Prompt
   * tokens before the last are processed in chunks of up to 64: batched
   * projections + one gdn_recur dispatch per layer per chunk.
   */
  chunkedPrefill?: boolean
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
  const S = opts.spec ?? PHI3
  const kvPages = Array.isArray(kv) ? kv : kv.pages
  const kvScales = Array.isArray(kv) ? undefined : kv.scales
  if (int8Mode && (!fused || !kvScales)) {
    throw new Error('buildDecodeEngine: int8KV requires fused mode and a KV cache with scales buffers')
  }
  // Hybrid specs (Qwen3.5): layerKinds mixes 'gdn' and 'attn' layers. GDN
  // layers run the GatedDeltaNet chain; attention layers run the unfused
  // gated-attention chain (c_attn → split → qk_norm → rope → append →
  // attention → sigmoid gate). KV pages exist only for 'attn' layers,
  // indexed by attention-layer ordinal.
  const hybrid = S.layerKinds.includes('gdn')
  if (hybrid && (fused || int8Mode)) {
    throw new Error('buildDecodeEngine: hybrid (GDN) specs require the unfused f16-KV composition')
  }
  const kvIndex: number[] = []
  {
    let ord = 0
    for (const k of S.layerKinds) kvIndex.push(k === 'attn' ? ord++ : -1)
  }
  // QK-norm specs (Qwen3) must run the unfused composition: qkv_fused folds
  // RoPE+KV-append into the projection, leaving nowhere to normalize Q/K
  // between the matmul and the rotation. Pages gate `fused` off per spec.
  if (S.qkNorm && fused) {
    throw new Error(
      'buildDecodeEngine: fused QKV is incompatible with qkNorm specs (Qwen3) — ' +
      'the per-head Q/K RMSNorm must run between the QKV matmul and RoPE. Use unfused mode.'
    )
  }
  // Fused qk_norm+RoPE+KV-append (?fuseqk, default on via parseVariantFlags):
  // on the unfused qkNorm path the 3 post-matmul dispatches collapse into 1
  // (10 → 8 per layer). Pure-attention specs only — the hybrid gated-attention
  // chain keeps the reference composition (its rope input comes from
  // gated_qkv_split and the win is 2 of 12 dispatches on 8 of 32 layers).
  const fuseQk = S.qkNorm && !hybrid && !fused && variants.fuseQkNorm

  // Per-matmul (K, M) shape uniforms compute K_PACKED = K/8 and
  // SCALES = K/QGROUP. K = d for the QKV/gate_up projections, qDim for o_proj
  // (== d when heads == kvHeads), ffn for down_proj.
  //
  // QGROUP, not the module's GROUP: MLC quantises in groups of 32, MLX-affine
  // in groups of 64. The kernels read SCALES_PER_ROW straight out of these
  // uniforms and index the scale array with it, so a 32 here against 64-wide
  // groups reads the WRONG SCALE for everything past the first group. It runs
  // clean, every value is finite, and the logits are nonsense.
  const QGROUP = S.weightFormat === 'mlx-safetensors' ? 64 : GROUP
  const QKV_K_PACKED    = S.d / PACK          // Phi-3: 384  (K=3072)
  const QKV_SCALES      = S.d / QGROUP        // Phi-3: 96
  const OPROJ_K_PACKED  = S.qDim / PACK       // Phi-3: 384 (qDim == d)
  const OPROJ_SCALES    = S.qDim / QGROUP     // Phi-3: 96
  const FFN_DN_K_PACKED = S.ffn / PACK        // Phi-3: 1024 (K=8192)
  const FFN_DN_SCALES   = S.ffn / QGROUP      // Phi-3: 256

  // Elementwise workgroup counts (one wg per WG_SIZE_D elements).
  const D_WGS   = S.d / WG_SIZE_D             // Phi-3: 12   (3072/256)
  const QKV_WGS = S.qkvDim / WG_SIZE_D        // Phi-3: 36   (9216/256)
  const KV_WGS  = S.kvDim / WG_SIZE_D         // kv_append grid — Phi-3: 12, Qwen3: 4

  const { pipelines } = compile(device, { subgroups: variants.subgroups }, S)
  const P = pipelines
  // The MoE expert matmul is compiled at rowsPerWG 4 and only that.
  const MOE_RPW = 4
  // A sparse-MoE spec has no scalar path. moe_router_topk.wgsl is 32 lanes ×
  // 8 experts held in registers and the expert matmul is {subgroups,
  // rowsPerWG:4}; neither has a non-subgroup sibling and neither is planned.
  // Without this the pipelines are null, the MoE branch is skipped, and the
  // engine runs the DENSE FFN against weights that are not laid out for it —
  // which produces tokens, just not the model's. buildDecodeEngine defaults to
  // SCALAR_VARIANTS, so this is the configuration validate.ts uses.
  // Which expert-matmul serves this spec is a property of the CHECKPOINT
  // (MoeDims.bits), resolved once — bind groups and dispatches must use the
  // same object, since layout:'auto' pipelines each own a distinct layout.
  const moeMM = S.moe?.bits === 3 ? P.moeMatmulQ3 : P.moeMatmul
  if (S.moe && !(P.moeRouterTopk && moeMM)) {
    throw new Error(
      'buildDecodeEngine: a MoE spec requires the subgroups feature — moe_router_topk '
      + 'and the grid-z expert matmul have no scalar variant. Pass { subgroups: true } to compile().',
    )
  }
  const R = resolveVariantPipelines(variants, P, S)
  // Split-K attention (?splitk=N) exists only for the f16 KV layout;
  // attention_int8 has no split-K variant.
  if (int8Mode && R.splitK) {
    console.warn('[engine] ?splitk ignored with int8 KV (attention_int8 has no split-K variant)')
  }
  const splitK = int8Mode ? 0 : R.splitK
  console.log(
    `[engine] attention=${R.attentionLabel} argmax=${R.argmaxLabel} qkv=${R.qkvLabel} ` +
    `ffn=${R.ffnLabel} matmul=${R.matmulLabel} rowsPerWG=${R.matmulRowsPerWG} ` +
    `mode=${fused ? (int8Mode ? 'fused+int8KV' : 'fused') : hybrid ? 'hybrid-unfused' : 'unfused'}` +
    (S.qkNorm && !hybrid ? ` fuseqk=${fuseQk ? 'on' : 'off'}` : '')
  )

  // Activation buffers. The per-layer QKV stage differs by mode:
  //   unfused    — qkv matmul → qkvOut, rope splits it into qOut/kOut/vOut,
  //                kv_append copies kOut/vOut into kvPages
  //   fused f16  — qkv_fused writes qOut + kvPages[L] directly
  //   fused int8 — qkv_fused_scratch writes qOut + kSlot/vSlot, kv_quantize
  //                packs them into int8 pages + f16 scales
  const B = {
    residual:  makeBuf(device, S.d * 2, 'residual'),      // running residual (ping)
    residual2: makeBuf(device, S.d * 2, 'residual2'),     // running residual (pong)
    hidden1:   makeBuf(device, S.d * 2, 'hidden1'),       // normed scratch
    hidden2:   makeBuf(device, S.d * 2, 'hidden2'),       // matmul output scratch
    qOut:      makeBuf(device, S.qDim * 2, 'qOut'),
    attnOut:   makeBuf(device, S.qDim * 2, 'attnOut'),
    ffnOut:    makeBuf(device, S.ffn * 2, 'ffnOut'),
    logits:    makeBuf(device, S.vocab * 4, 'logits'),
    tokenOut:  makeBuf(device, 4, 'tokenOut'),
    inputIds:  makeBuf(device, 4, 'inputIds'),
    posMap:    makeBuf(device, 4, 'posMap'),
    pageIndptr: makeBuf(device, 8, 'pageIndptr'),
    pageValues: makeBuf(device, S.maxPages * 4, 'pageValues'),
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
    // Hybrid (Qwen3.5) activation scratch — allocated only for hybrid specs.
    cAttnOut:   null as GPUBuffer | null,  // c_attn projection [C_ATTN_DIM f16]
    attnGateRaw: null as GPUBuffer | null, // raw per-head gate rows [Q_DIM f16]
    // Fused GDN input projection output [GDN_PROJ_ROWS f16]: regions
    // [qkv_raw | z | a | b] at row offsets 0 / gdnQkvDim / +gdnVDim / +vHeads.
    // gdn_conv reads the qkv region (offset 0, binds the whole buffer);
    // gdn_norm_out binds the z region and gdn_gates the [a|b] pair via
    // 256-aligned bind-group offsets — no per-kernel copies.
    gdnProjOut: null as GPUBuffer | null,
    gdnConvOut: null as GPUBuffer | null,  // conv+SiLU output [GDN_QKV_DIM f16]
    gdnGates:   null as GPUBuffer | null,  // [exp(g) | beta] [2*GDN_V_HEADS f32]
    gdnRecurOut: null as GPUBuffer | null, // recurrence readout [GDN_V_DIM f32]
    gdnNormed:  null as GPUBuffer | null,  // gated-norm output [GDN_V_DIM f16]
    // Sparse MoE scratch — shared across layers, the block is stateless.
    routerLogits: null as GPUBuffer | null,  // [experts+1] f32
    moeIds:       null as GPUBuffer | null,  // [slots] u32 — expert per slot
    moeScores:    null as GPUBuffer | null,  // [slots] f32
    moeGateUp:    null as GPUBuffer | null,  // [slots][2*ffn] f16
    moeH:         null as GPUBuffer | null,  // [slots][ffn] f16
    moeDown:      null as GPUBuffer | null,  // [slots][d] f16
  }
  if (!fused) {
    B.qkvOut = makeBuf(device, S.qkvDim * 2, 'qkvOut')
    B.kOut   = makeBuf(device, S.kvDim * 2, 'kOut')
    B.vOut   = makeBuf(device, S.kvDim * 2, 'vOut')
  }
  if (int8Mode) {
    B.kSlot = makeBuf(device, S.kvDim * 2, 'kSlot')
    B.vSlot = makeBuf(device, S.kvDim * 2, 'vSlot')
  }
  if (R.ffnPrologue) {
    B.hidden3 = makeBuf(device, S.d * 2, 'hidden3')
  }
  if (splitK) {
    B.attnPartials = makeBuf(device, S.heads * splitK * (S.headDim + 2) * 4, 'attnPartials')
  }
  // Per-GDN-layer persistent state: conv ring ((convK-1) × GDN_QKV_DIM f16)
  // and the recurrent state matrix (GDN_V_HEADS × headK × headV f32 ≈ 2 MB).
  // Both are zeroed on the GPU whenever a forward pass runs at position 0
  // (fresh conversation / fresh prefill) — see clearGdnState().
  const gdnConvState: (GPUBuffer | null)[] = []
  const gdnRecurState: (GPUBuffer | null)[] = []
  const gdnStateBufs: GPUBuffer[] = []
  if (hybrid) {
    B.cAttnOut   = makeBuf(device, S.cAttnDim * 2, 'cAttnOut')
    B.attnGateRaw = makeBuf(device, S.qDim * 2, 'attnGateRaw')
    B.gdnProjOut = makeBuf(device, S.gdnProjRows * 2, 'gdnProjOut')
    B.gdnConvOut = makeBuf(device, S.gdnQkvDim * 2, 'gdnConvOut')
    B.gdnGates   = makeBuf(device, 2 * S.gdnVHeads * 4, 'gdnGates')
    B.gdnRecurOut = makeBuf(device, S.gdnVDim * 4, 'gdnRecurOut')
    B.gdnNormed  = makeBuf(device, S.gdnVDim * 2, 'gdnNormed')
    for (let L = 0; L < S.layers; L++) {
      if (S.layerKinds[L] === 'gdn') {
        const conv = makeBuf(device, (S.gdnConvK - 1) * S.gdnQkvDim * 2, `gdnConvState_${L}`)
        const recur = makeBuf(device, S.gdnVHeads * S.gdnStatePerHead * 4, `gdnRecurState_${L}`)
        gdnConvState.push(conv)
        gdnRecurState.push(recur)
        gdnStateBufs.push(conv, recur)
      } else {
        gdnConvState.push(null)
        gdnRecurState.push(null)
      }
    }
  }

  // A MoE block is STATELESS — unlike the GDN recurrence above, nothing carries
  // between tokens — so every buffer here is shared across all 40 layers.
  // moeH and moeDown are allocated fresh rather than reusing B.ffnOut (ffn*2 =
  // 1 KB, but 9 slots need 9 KB) or B.hidden2 (d*2 = 4 KB against 36 KB): a
  // stray dense dispatch should fail loudly, not corrupt slot 0.
  if (S.moe) {
    const M = S.moe
    B.routerLogits = makeBuf(device, (M.experts + 1) * 4, 'routerLogits')
    B.moeIds       = makeBuf(device, S.moeSlots * 4, 'moeIds')
    B.moeScores    = makeBuf(device, S.moeSlots * 4, 'moeScores')
    B.moeGateUp    = makeBuf(device, S.moeSlots * 2 * S.ffn * 2, 'moeGateUp')
    B.moeH         = makeBuf(device, S.moeSlots * S.ffn * 2, 'moeH')
    B.moeDown      = makeBuf(device, S.moeSlots * S.d * 2, 'moeDown')
  }

  // Static uniforms — matmul shapes (K_packed, scales_per_row, M)
  const qkvU   = fused ? null : uniformBuf(device, [u32(QKV_K_PACKED), u32(QKV_SCALES), u32(S.qkvDim)])
  const oProjU = uniformBuf(device, [u32(OPROJ_K_PACKED),  u32(OPROJ_SCALES),  u32(S.d)])
  // MoE uniforms. The matmul PODArgs is FIVE u32 here, not three: the moe
  // variant adds IN_SLOT_STRIDE and OUT_SLOT_STRIDE. IN_SLOT_STRIDE 0 means
  // every slot reads the SAME activation (gate/up); down strides both.
  // Affine groups are 64, so scales-per-row is K/64, not K/32.
  const moeRouterU = S.moe ? uniformBuf(device, [u32(S.d), u32(S.moe.experts + 1)]) : null
  const moeTopkU = S.moe
    ? uniformBuf(device, [u32(S.moe.experts), u32(S.moe.topK), u32(S.moe.normTopkProb ? 1 : 0)])
    : null
  // K_PACKED is words per weight row: K*bits/32 (4-bit: K/8, 3-bit: 3K/32).
  const moeBits = S.moe?.bits ?? 4
  const moeGateU = S.moe
    ? uniformBuf(device, [u32((S.d * moeBits) / 32), u32(S.d / 64), u32(S.ffn), u32(0), u32(2 * S.ffn)])
    : null
  const moeDownU = S.moe
    ? uniformBuf(device, [u32((S.ffn * moeBits) / 32), u32(S.ffn / 64), u32(S.d), u32(S.ffn), u32(S.d)])
    : null
  const moeSiluU = S.moe
    ? uniformBuf(device, [i32(S.moeSlots), i32(Math.ceil(S.moeSlots * S.ffn / 256))])
    : null
  const moeCombU = S.moe ? uniformBuf(device, [u32(S.d), u32(S.moeSlots)]) : null
  const ffnDnU = uniformBuf(device, [u32(FFN_DN_K_PACKED), u32(FFN_DN_SCALES), u32(S.d)])
  const lmHdU  = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(S.vocab)])
  const embU   = uniformBuf(device, [u32(1), u32(D_WGS)])
  const normU  = uniformBuf(device, [u32(1)])
  const ffnU   = uniformBuf(device, [u32(S.ffn)])
  const argmaxU = uniformBuf(device, [u32(S.vocab)])

  // Hoisted per-layer uniforms — all fields are constant across tokens.
  // Unfused: rope + kv_append. Fused: qkv_fused (f16) or scratch + quantize (int8).
  const ropeU  = fused ? null : uniformBuf(device, [i32(1), i32(0), i32(1), u32(QKV_WGS)])
  const kvAppU = fused ? null : uniformBuf(device, [i32(1), i32(S.maxPages), i32(0), i32(0), u32(KV_WGS)])
  // Qwen3 per-head Q/K RMSNorm — one 32-thread WG per (token, head), heads
  // ordered Q then K: grid = seq_len × (HEADS + KV_HEADS). seq_len is 1 on
  // the decode path, so the uniform is fully constant.
  const QK_NORM_WGS = S.heads + S.kvHeads
  const qkNormU = S.qkNorm ? uniformBuf(device, [i32(1), u32(QK_NORM_WGS)]) : null
  // Fused qk_norm+RoPE+append (?fuseqk): one 32-thread WG per (token, head)
  // over Q, K AND V heads. seq_len is 1 on the decode path — fully constant.
  const QK_FUSE_WGS = S.heads + 2 * S.kvHeads
  const qkFuseU = fuseQk ? uniformBuf(device, [i32(1), u32(QK_FUSE_WGS)]) : null
  const qkvFusedU = (fused && !int8Mode) ? uniformBuf(device, [i32(0), i32(0), u32(S.qkvPairs)]) : null      // f16-KV mode
  const qkvFusedScratchU = int8Mode ? uniformBuf(device, [i32(0), u32(S.qkvPairs)]) : null                   // int8-KV mode
  const kvQuantU = int8Mode ? uniformBuf(device, [i32(0), i32(0), i32(0), u32(S.kvHeads * 2)]) : null  // pos_off, pages_off, scales_off, kvHeads*2 WG

  // Hybrid (Qwen3.5) uniforms + grids. All static except gdnConvU.pos, which
  // writeStepState rewrites per token (queue-ordered like attnU.nnz_pages).
  const GDN_CONV_WGS = hybrid ? Math.ceil(S.gdnQkvDim / WG_SIZE_D) : 0
  const C_ATTN_WGS   = hybrid ? Math.ceil(S.cAttnDim / WG_SIZE_D) : 0
  const ATTN_GATE_WGS = hybrid ? Math.ceil(S.qDim / WG_SIZE_D) : 0
  // Fused GDN input projection: one 12352-row (Qwen3.5) K=d matmul replaces
  // the 4 separate in_proj_qkv/z/a/b dispatches (weights packed by the loader).
  const gdnProjU = hybrid ? uniformBuf(device, [u32(S.dPacked), u32(S.d / QGROUP), u32(S.gdnProjRows)]) : null
  const gdnOutU  = hybrid ? uniformBuf(device, [u32(S.gdnVDim / PACK), u32(S.gdnVDim / QGROUP), u32(S.d)]) : null
  const cAttnU   = hybrid ? uniformBuf(device, [u32(S.dPacked), u32(S.d / QGROUP), u32(S.cAttnDim)]) : null
  const gdnConvU = hybrid ? uniformBuf(device, [i32(0), u32(GDN_CONV_WGS)]) : null  // pos rewritten per token
  // gdn_gates / gdn_norm_out are seq-capable (chunked prefill): the decode
  // uniforms pin seq_len=1 with tightly-packed strides (the region views the
  // decode bind groups carry are single-token).
  const gdnGatesU = hybrid ? uniformBuf(device, [i32(1), i32(2 * S.gdnVHeads), u32(1)]) : null
  const gdnRecurU = hybrid ? uniformBuf(device, [i32(1), u32(S.gdnVHeads)]) : null  // seq_len = 1 (decode/step prefill)
  const gdnNormU  = hybrid ? uniformBuf(device, [i32(1), i32(S.gdnVDim), u32(S.gdnVHeads)]) : null
  const gatedSplitU = hybrid ? uniformBuf(device, [i32(1), u32(C_ATTN_WGS)]) : null
  const attnGateU = hybrid ? uniformBuf(device, [u32(ATTN_GATE_WGS)]) : null
  // GDN out_proj is a K = GDN_V_DIM matmul instance — resolve its own
  // pipeline so the vec4 K-divisibility gate applies to the right K
  // (== R.matmulOProj on Qwen3.5, where gdnVDim == qDim == 4096).
  // Same affine gate as resolveVariantPipelines — this is the fourth K instance
  // and it is resolved here rather than there, so it has to repeat the test.
  const matmulGdnOut = hybrid
    ? resolveMatmul(variants.matmul, P, variants.vec4, S.gdnVDim, variants.vec4Half,
                    S.weightFormat === 'mlx-safetensors').pipeline
    : null

  const SM_SCALE = 1.0 / Math.sqrt(S.headDim)

  // Hard KV ceiling — writing a slot at or past this position would run off
  // the last page and silently corrupt the cache.
  const MAX_CONTEXT = S.maxContext

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
    dv.setInt32(4, S.maxPages, true)           // max_num_pages
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
    dv.setInt32(4, S.maxPages, true)           // max_num_pages
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
    dv.setInt32(4, S.maxPages, true)           // max_num_pages
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
  const gdnPosScratch = new Int32Array(1)      // reused per token for gdnConvU.pos

  // Token readback buffer for the blocking path — allocated once, reused
  // every decode step.
  const readBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'tokenReadback',
  })

  // ── GDN state-position tracker (hybrid specs) ─────────────────────────────
  // Number of tokens the persistent GDN state (conv rings + recurrent S)
  // currently has absorbed — i.e. the next position a forward pass may run at
  // without corrupting the recurrence. Every submitted forward pass at
  // position p advances it to p+1 (a pass at position 0 first zeroes the
  // state, so the counter is exact from a fresh prefill onward).
  //
  // This is what lets the blocking generate() skip the prompt replay: unlike
  // the KV cache (idempotent — rewriting a slot with the same K/V is
  // harmless), a GDN step MUTATES S and rotates the conv ring, so re-running
  // any already-absorbed token double-applies it. generate() therefore only
  // trusts `startPos` when gdnStatePos proves the state actually sits at that
  // boundary; anything else falls back to a full replay from 0 (which
  // re-zeroes the state). The pipelined path may leave gdnStatePos past the
  // consumed sequence (up to PIPELINE_DEPTH-1 speculative steps submitted
  // after a stop token) — those extra mutations are harmless precisely
  // because no later call trusts state it can't match: pipelined generation
  // always re-prefills from 0, and the blocking path checks this counter.
  let gdnStatePos = 0

  // ── Absorbed-token record (cross-turn prefix reuse) ───────────────────────
  // absorbed[p] is the token id whose forward pass was the LAST submitted at
  // position p — i.e. what KV slot p currently encodes and (for hybrid) what
  // the GDN state absorbed at step p. Every submission path maintains it:
  // blocking decodeToken and prefill submitSteps note tokens directly;
  // generatePipelined patches the chained-argmax positions from its readbacks
  // (the input of the step at position p is the readback of step p-1), so the
  // record stays exact even for the ≤ PIPELINE_DEPTH-1 overrun steps
  // submitted after a stop token. A readback failure (device loss) marks the
  // record invalid, which disables reuse until resetKVTracking().
  let absorbed: number[] = []
  let absorbedValid = true
  const prefixReuse = opts.prefixReuse ?? true
  let lastPrefill: { promptLen: number; reused: number; chunks: number } | null = null

  function noteAbsorbed(position: number, id: number): void {
    if (position > absorbed.length) { absorbedValid = false; return }
    absorbed.length = position
    absorbed.push(id)
  }

  /**
   * Longest reusable prefix of `promptIds` given the absorbed record.
   * Pure attention: min(LCP, len-1) — KV slots are idempotent, stale slots
   * past the LCP are overwritten by the delta prefill in order, and at least
   * the final prompt token always runs (to produce the first argmax).
   * Hybrid: the GDN recurrence is non-rewindable, so reuse requires the new
   * prompt to extend EVERY absorbed token (LCP == absorbed.length — in normal
   * chat the ≤1-token stop overrun is the <|im_end|> stop id, which IS the
   * next prompt token) with the state boundary exactly there; else 0 (full
   * re-prefill, which re-zeroes the GDN state at position 0).
   */
  function computeReuseStart(promptIds: number[]): number {
    if (!prefixReuse || !absorbedValid || promptIds.length === 0) return 0
    const max = Math.min(absorbed.length, promptIds.length)
    let lcp = 0
    while (lcp < max && absorbed[lcp] === promptIds[lcp]) lcp++
    if (!hybrid) return Math.min(lcp, promptIds.length - 1)
    if (gdnStatePos === absorbed.length && lcp === absorbed.length && lcp <= promptIds.length - 1) {
      return lcp
    }
    return 0
  }

  // Logit readback buffer — used by forwardLogits() for the validation harness only.
  // Allocated lazily on first call.
  let logitsReadBuf: GPUBuffer | null = null

  // Initialize identity page table (page i → physical page i)
  {
    const pageVals = new Int32Array(S.maxPages)
    for (let i = 0; i < S.maxPages; i++) pageVals[i] = i
    device.queue.writeBuffer(B.pageValues, 0, pageVals)
  }

  // ============================================================
  // Pre-computed bind groups
  //
  // Bind group contents are deterministic (same buffers, same layout) so we
  // hoist them out of the hot loop — ~10 per layer × spec.layers otherwise.
  //
  // The residual ping-pong is also deterministic: every layer reads `residual`
  // and writes `residual2` in addNorm1 (post-attention), then reads `residual2`
  // and writes `residual` in addNorm2 (post-FFN). Two swaps per layer return to
  // the same state, so the bind groups are identical for every layer.
  // ============================================================

  // Affine (MLX) matmuls bind a per-group bias at @binding(5); the symmetric
  // kernels have no such binding. Appending it exactly when the spec is affine
  // is what keeps ONE bind-group construction serving both families — and a
  // mismatch is a loud bind-group validation error ("5 entries, expected 6"),
  // not a wrong number, which is the only reason this is safe to centralise.
  const AFFINE = S.weightFormat === 'mlx-safetensors'
  const withBias = (entries: BindEntry[], bias: GPUBuffer | undefined, what: string): BindEntry[] => {
    if (!AFFINE) return entries
    if (!bias) throw new Error(`buildDecodeEngine: ${S.id} is affine but ${what} has no bias buffer`)
    return [...entries, bias]
  }

  // Global (non-per-layer) bind groups
  // embedding.wgsl is hard-symmetric — (nibble-7)*scale over groups of 32, no
  // bias — so an affine table needs the other kernel, not just an extra entry.
  // One name for both the bind group and the dispatch: with layout:'auto' each
  // pipeline owns a DISTINCT layout object even when structurally identical, so
  // binding against one and dispatching the other is a validation error.
  const embeddingPipeline = AFFINE ? P.embeddingAffine : P.embedding
  const bgEmbedding = bg(device, embeddingPipeline, withBias(
    [B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU],
    weights.embdBiases, 'embed_tokens'))
  const bgInitNorm = bg(device, P.rmsNorm, [
    B.hidden1, B.residual, weights.layers[0].normGamma1, normU,
  ])
  const bgRope = fused ? null : bg(device, P.rope, [
    B.qOut, B.kOut!, B.vOut!, B.qkvOut!, B.posMap, ropeU!,
  ])
  const bgLmHead = bg(device, R.matmulF32, withBias(
    [B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU],
    weights.lmHeadBiases, 'lm_head'))
  const bgArgmax = bg(device, R.argmax, [
    B.logits, B.tokenOut, argmaxU,
  ])
  // Split-K combine reads the shared partials scratch and writes attnOut —
  // identical for every layer, so one bind group serves every layer.
  const bgAttnCombine = splitK
    ? bg(device, R.attentionCombine!, [B.attnPartials!, B.attnOut, combineU!])
    : null

  // Per-layer bind groups
  interface LayerBG {
    qkv?: GPUBindGroup          // unfused: qkv matmul | fused: qkv_fused / qkv_fused_scratch | hybrid attn: c_attn matmul
    qkNorm?: GPUBindGroup       // qkNorm specs (Qwen3/Qwen3.5) only — in-place Q/K RMSNorm on qkvOut
    qkFused?: GPUBindGroup      // ?fuseqk (qkNorm specs): fused qk_norm+RoPE+append replaces qkNorm/rope/kvApp
    gatedSplit?: GPUBindGroup   // hybrid attn layers: c_attn → qkvOut + attnGateRaw unpack
    attnGate?: GPUBindGroup     // hybrid attn layers: attnOut *= sigmoid(gate), in place
    kvApp?: GPUBindGroup        // unfused only
    kvQuantize?: GPUBindGroup   // int8 mode only
    attn?: GPUBindGroup         // splitK: the partial pass (writes attnPartials)
    // Hybrid GDN layers: the whole GatedDeltaNet chain replaces qkv/attn.
    gdn?: {
      proj: GPUBindGroup        // fused in_proj qkv‖z‖a‖b matmul → gdnProjOut
      conv: GPUBindGroup        // causal conv + SiLU (ring state; reads the qkv region)
      gates: GPUBindGroup       // exp(g) decay + beta (reads the [a|b] region)
      recur: GPUBindGroup       // gated delta rule (f32 state)
      normOut: GPUBindGroup     // gated RMSNorm · silu(z region) → gdnNormed
    }
    oProj: GPUBindGroup         // attn: o_proj | gdn: out_proj (both → hidden2)
    addNorm1?: GPUBindGroup     // post-attention: reads residual, writes residual2 (absent w/ ?fuseprologue=1)
    ffn?: GPUBindGroup          // dense FFN; absent on a MoE spec
    ffnDown?: GPUBindGroup      // prologue mode: writes hidden3 (hidden2 must survive for add3_norm)
    /** The MoE block's seven bind groups, present iff spec.moe. Replaces
     *  ffn/ffnDown between addNorm1 and addNorm2 — the surrounding residual
     *  chain is identical, which is why the block is a drop-in. */
    moe?: {
      routerLogits: GPUBindGroup
      routerTopk: GPUBindGroup
      gate: GPUBindGroup
      up: GPUBindGroup
      silu: GPUBindGroup
      down: GPUBindGroup
      combine: GPUBindGroup
    }
    addNorm2: GPUBindGroup      // post-FFN: reads residual2, writes residual (prologue mode: add3_norm)
  }
  if (S.moe && R.ffnPrologue) {
    throw new Error('buildDecodeEngine: ?fuseprologue=1 folds add_norm into the DENSE FFN; a MoE spec has no dense FFN')
  }
  if (hybrid && R.ffnPrologue) {
    throw new Error('buildDecodeEngine: ?fuseprologue=1 is not supported for hybrid (GDN) specs')
  }
  const layerBGs: LayerBG[] = []
  for (let L = 0; L < S.layers; L++) {
    const lw = weights.layers[L]
    const isGdn = S.layerKinds[L] === 'gdn'
    const nextGamma = L < S.layers - 1
      ? weights.layers[L + 1].normGamma1
      : weights.finalNormGamma

    let qkvBG: GPUBindGroup | undefined
    let attnBG: GPUBindGroup | undefined
    let qkNormBG: GPUBindGroup | undefined
    let qkFusedBG: GPUBindGroup | undefined
    let gatedSplitBG: GPUBindGroup | undefined
    let attnGateBG: GPUBindGroup | undefined
    let kvAppBG: GPUBindGroup | undefined
    let kvQuantizeBG: GPUBindGroup | undefined
    let gdnBG: LayerBG['gdn']
    let oProjBG: GPUBindGroup
    // KV pages for this layer's attention-layer ordinal (== L for
    // pure-attention specs).
    const kvL = () => kvPages[kvIndex[L]]
    // Split-K partial pass replaces the single-WG-per-head attention bind
    // group in the f16-KV modes (int8 has no split-K variant).
    const attnF16BG = () => splitK
      ? bg(device, R.attentionSplitK!, [
          B.qOut, B.pageIndptr, B.pageValues, kvL(), B.lengthInfo, B.attnPartials!, attnSkU!,
        ])
      : bg(device, R.attention, [
          B.qOut, B.pageIndptr, B.pageValues, kvL(), B.lengthInfo, B.attnOut, attnU,
        ])
    const qkNormBGFor = () => {
      // q_norm/k_norm gammas are HEAD_DIM f16 vectors in the MLC layout —
      // same dtype as every other norm gamma, uploaded raw by the loader.
      if (!lw.qNormGamma || !lw.kNormGamma) {
        throw new Error(`buildDecodeEngine: spec ${S.id} sets qkNorm but layer ${L} has no q_norm/k_norm weights`)
      }
      return bg(device, P.qkNorm, [B.qkvOut!, lw.qNormGamma, lw.kNormGamma, qkNormU!])
    }
    if (isGdn) {
      const gw = lw.gdn
      if (!gw) throw new Error(`buildDecodeEngine: spec ${S.id} marks layer ${L} 'gdn' but the loader has no GDN weights for it`)
      // Region views into the packed projection output [qkv | z | a | b].
      // Offsets are f16-element row offsets × 2 bytes; both are multiples of
      // 256 (z at 16384, a|b at 24576 for Qwen3.5) so they satisfy
      // minStorageBufferOffsetAlignment on every conformant device.
      const zRegion  = { buffer: B.gdnProjOut!, offset: S.gdnQkvDim * 2, size: S.gdnVDim * 2 }
      const abRegion = { buffer: B.gdnProjOut!, offset: (S.gdnQkvDim + S.gdnVDim) * 2, size: 2 * S.gdnVHeads * 2 }
      gdnBG = {
        proj: bg(device, R.matmul, withBias(
          [B.gdnProjOut!, B.hidden1, gw.projScales, gw.projWeights, gdnProjU!], gw.projBiases, 'gdn in_proj')),
        // gdn_conv's qkv_raw binding sees the whole packed buffer; it only
        // reads channels < GDN_QKV_DIM (the qkv region at offset 0).
        conv: bg(device, P.gdnConv, [B.gdnConvOut!, B.gdnProjOut!, gdnConvState[L]!, gw.convWeight, gdnConvU!]),
        gates: bg(device, P.gdnGates, [B.gdnGates!, abRegion, gw.aLog, gw.dtBias, gdnGatesU!]),
        recur: bg(device, P.gdnRecur, [B.gdnRecurOut!, B.gdnConvOut!, B.gdnGates!, gdnRecurState[L]!, gdnRecurU!]),
        normOut: bg(device, P.gdnNormOut, [B.gdnNormed!, B.gdnRecurOut!, gw.normGamma, zRegion, gdnNormU!]),
      }
      oProjBG = bg(device, matmulGdnOut!, withBias(
        [B.hidden2, B.gdnNormed!, gw.outScales, gw.outWeights, gdnOutU!], gw.outBiases, 'gdn out_proj'))
    } else if (hybrid) {
      // Gated attention layer: c_attn packs per-head [Q|gate] before K‖V.
      qkvBG = bg(device, R.matmul, withBias(
        [B.cAttnOut!, B.hidden1, lw.qkvScales!, lw.qkvWeights!, cAttnU!], lw.qkvBiases, 'c_attn'))
      gatedSplitBG = bg(device, P.gatedQkvSplit, [B.qkvOut!, B.attnGateRaw!, B.cAttnOut!, gatedSplitU!])
      if (S.qkNorm) qkNormBG = qkNormBGFor()
      kvAppBG = bg(device, P.kvAppend, [
        B.kOut!, B.vOut!, kvL(), B.posMap, kvAppU!,
      ])
      attnBG = attnF16BG()
      attnGateBG = bg(device, P.attnGate, [B.attnOut, B.attnGateRaw!, attnGateU!])
      oProjBG = bg(device, R.matmulOProj, withBias(
        [B.hidden2, B.attnOut, lw.oProjScales!, lw.oProjWeights!, oProjU], lw.oProjBiases, 'o_proj'))
    } else if (!fused) {
      qkvBG = bg(device, R.matmul, [
        B.qkvOut!, B.hidden1, lw.qkvScales!, lw.qkvWeights!, qkvU!,
      ])
      if (fuseQk) {
        // ?fuseqk: one fused kernel replaces qk_norm → rope → kv_append.
        if (!lw.qNormGamma || !lw.kNormGamma) {
          throw new Error(`buildDecodeEngine: spec ${S.id} sets qkNorm but layer ${L} has no q_norm/k_norm weights`)
        }
        qkFusedBG = bg(device, P.qkNormRopeAppend, [
          B.qOut, kvL(), B.qkvOut!, lw.qNormGamma, lw.kNormGamma, B.posMap, qkFuseU!,
        ])
      } else {
        if (S.qkNorm) qkNormBG = qkNormBGFor()
        kvAppBG = bg(device, P.kvAppend, [
          B.kOut!, B.vOut!, kvL(), B.posMap, kvAppU!,
        ])
      }
      attnBG = attnF16BG()
      oProjBG = bg(device, R.matmulOProj, [B.hidden2, B.attnOut, lw.oProjScales!, lw.oProjWeights!, oProjU])
    } else if (int8Mode) {
      qkvBG = bg(device, P.qkvFusedScratch, [
        B.qOut, B.kSlot!, B.vSlot!, B.hidden1, lw.qkvScales!, lw.qkvWeights!, B.posMap, qkvFusedScratchU!,
      ])
      kvQuantizeBG = bg(device, P.kvQuantizeInt8, [
        B.kSlot!, B.vSlot!, kvL(), kvScales![L], B.posMap, kvQuantU!,
      ])
      attnBG = bg(device, P.attentionInt8, [
        B.qOut, B.pageIndptr, B.pageValues, kvL(), kvScales![L],
        B.lengthInfo, B.attnOut, attnI8U!,
      ])
      oProjBG = bg(device, R.matmulOProj, [B.hidden2, B.attnOut, lw.oProjScales!, lw.oProjWeights!, oProjU])
    } else {
      qkvBG = bg(device, R.qkvFused, [
        B.qOut, kvL(), B.hidden1, lw.qkvScales!, lw.qkvWeights!, B.posMap, qkvFusedU!,
      ])
      attnBG = attnF16BG()
      oProjBG = bg(device, R.matmulOProj, [B.hidden2, B.attnOut, lw.oProjScales!, lw.oProjWeights!, oProjU])
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
        qkNorm: qkNormBG,
        qkFused: qkFusedBG,
        kvApp: kvAppBG,
        kvQuantize: kvQuantizeBG,
        attn: attnBG,
        oProj: oProjBG,
        // ?fuseprologue is dense-only — buildDecodeEngine throws for S.moe.
        ffn: bg(device, R.ffn, [B.ffnOut, rIn, B.hidden2, lw.normGamma2, lw.ffnScales!, lw.ffnWeights!, ffnU]),
        ffnDown: bg(device, R.matmulFfnDown, [B.hidden3!, B.ffnOut, lw.ffnDownScales!, lw.ffnDownWeights!, ffnDnU]),
        addNorm2: bg(device, P.add3Norm, [rIn, B.hidden2, B.hidden3!, nextGamma, B.hidden1, rOut, normU]),
      })
      continue
    }

    // The MoE block reads B.hidden1 (what addNorm1 just produced) and writes
    // B.hidden2 (what addNorm2 consumes) — exactly the dense FFN's contract,
    // which is why nothing around it changes.
    const m = lw.moe
    const moeBG = S.moe && m
      ? {
          routerLogits: bg(device, P.moeRouterLogits,
            [B.routerLogits!, B.hidden1, m.routerWeights, m.routerScales, m.routerBiases, moeRouterU!]),
          routerTopk: bg(device, P.moeRouterTopk!, [B.moeIds!, B.moeScores!, B.routerLogits!, moeTopkU!]),
          // gate and up write interleaved halves of one [slot][2*ffn] buffer:
          // same kernel, `up` bound ffn*2 bytes in, which is the layout silu_mul
          // reads. The bind OFFSET must stay 256-aligned — ffn*2 = 1024 here.
          gate: bg(device, moeMM!, [
            { buffer: B.moeGateUp!, offset: 0, size: S.moeSlots * 2 * S.ffn * 2 },
            B.hidden1, m.gateScales, m.gateWeights, moeGateU!, m.gateBiases, B.moeIds!]),
          up: bg(device, moeMM!, [
            { buffer: B.moeGateUp!, offset: S.ffn * 2, size: S.moeSlots * 2 * S.ffn * 2 - S.ffn * 2 },
            B.hidden1, m.upScales, m.upWeights, moeGateU!, m.upBiases, B.moeIds!]),
          silu: bg(device, P.siluMul, [B.moeH!, B.moeGateUp!, moeSiluU!]),
          down: bg(device, moeMM!, [
            { buffer: B.moeDown!, offset: 0, size: S.moeSlots * S.d * 2 },
            B.moeH!, m.downScales, m.downWeights, moeDownU!, m.downBiases, B.moeIds!]),
          combine: bg(device, P.moeCombine, [B.hidden2, B.moeDown!, B.moeScores!, moeCombU!]),
        }
      : undefined

    layerBGs.push({
      qkv: qkvBG,
      qkNorm: qkNormBG,
      qkFused: qkFusedBG,
      gatedSplit: gatedSplitBG,
      attnGate: attnGateBG,
      kvApp: kvAppBG,
      kvQuantize: kvQuantizeBG,
      attn: attnBG,
      gdn: gdnBG,
      oProj: oProjBG,
      addNorm1: bg(device, P.addNorm, [B.hidden2, B.residual, lw.normGamma2, B.hidden1, B.residual2, normU]),
      ffn: S.moe ? undefined : bg(device, R.ffn, [B.ffnOut, B.hidden1, lw.ffnScales!, lw.ffnWeights!, ffnU]),
      ffnDown: S.moe ? undefined
        : bg(device, R.matmulFfnDown, [B.hidden2, B.ffnOut, lw.ffnDownScales!, lw.ffnDownWeights!, ffnDnU]),
      moe: moeBG,
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
   * Record one full forward pass (embedding → all layers → LM head → argmax)
   * into `enc`. Both generate styles and profileStep share this recorder, so
   * dispatch order is identical everywhere.
   */
  function recordForward(enc: GPUCommandEncoder): void {
    // --- EMBEDDING → B.residual (ping) ---
    dispatch(enc, embeddingPipeline, bgEmbedding, D_WGS, 1, 1, 'embedding')
    // --- INITIAL RMSNORM: B.residual → B.hidden1 (layer 0's normGamma1) ---
    dispatch(enc, P.rmsNorm, bgInitNorm, 1, 1, 1, 'rmsNorm_init')

    // --- TRANSFORMER LAYERS ---
    // Residual ping-pong is encoded into the cached bind groups: addNorm1 reads
    // residual / writes residual2; addNorm2 reads residual2 / writes residual.
    // Unfused: 9 dispatches/layer (10 for qkNorm specs — Qwen3 inserts the
    // per-head Q/K RMSNorm between qkv matmul and rope; 8 with ?fuseqk, which
    // fuses qkNorm+rope+kvAppend into one pass). Fused f16: 7. Fused int8: 8
    // (adds a kv_quantize pass between qkv_fused_scratch and attention_int8).
    for (let L = 0; L < S.layers; L++) {
      const blk = layerBGs[L]

      // Attention on the f16 KV layout — split-K (?splitk=N) turns the one
      // WG-per-head dispatch into a partial pass over N partitions per head
      // plus a per-head combine over the partials scratch.
      const attentionF16 = () => {
        if (splitK) {
          dispatch(enc, R.attentionSplitK!, blk.attn!, splitK, S.heads, 1, 'attention')
          dispatch(enc, R.attentionCombine!, bgAttnCombine!, 1, S.heads, 1, 'attnCombine')
        } else {
          dispatch(enc, R.attention, blk.attn!, 1, S.heads, 1, 'attention')
        }
      }

      if (blk.gdn) {
        // GatedDeltaNet layer (Qwen3.5 linear_attn): ONE fused input
        // projection (qkv‖z‖a‖b rows packed at load time — replaces the 4
        // separate matmul dispatches), then conv → gates → recurrence →
        // gated norm.
        const g = blk.gdn
        dispatch(enc, R.matmul, g.proj, Math.ceil(S.gdnProjRows / R.matmulRowsPerWG), 1, 1, 'gdnProjMatmul')
        dispatch(enc, P.gdnConv, g.conv, GDN_CONV_WGS, 1, 1, 'gdnConv')
        dispatch(enc, P.gdnGates, g.gates, 1, 1, 1, 'gdnGates')
        dispatch(enc, P.gdnRecur, g.recur, S.gdnVHeads, 1, 1, 'gdnRecur')
        dispatch(enc, P.gdnNormOut, g.normOut, S.gdnVHeads, 1, 1, 'gdnNormOut')
        // out_proj: B.gdnNormed → B.hidden2 (K = GDN_V_DIM instance)
        dispatch(enc, matmulGdnOut!, blk.oProj, S.d / R.matmulRowsPerWG, 1, 1, 'gdnOutProj')
      } else if (hybrid) {
        // Gated attention layer (Qwen3.5): c_attn → per-head [Q|gate] split →
        // qk_norm → partial RoPE → KV append → attention → sigmoid gate.
        dispatch(enc, R.matmul, blk.qkv!, S.cAttnDim / R.matmulRowsPerWG, 1, 1, 'cAttnMatmul')
        dispatch(enc, P.gatedQkvSplit, blk.gatedSplit!, C_ATTN_WGS, 1, 1, 'gatedQkvSplit')
        if (S.qkNorm) dispatch(enc, P.qkNorm, blk.qkNorm!, QK_NORM_WGS, 1, 1, 'qkNorm')
        dispatch(enc, P.rope, bgRope!, QKV_WGS, 1, 1, 'rope')
        dispatch(enc, P.kvAppend, blk.kvApp!, KV_WGS, 1, 1, 'kvAppend')
        attentionF16()
        dispatch(enc, P.attnGate, blk.attnGate!, ATTN_GATE_WGS, 1, 1, 'attnGate')
        dispatch(enc, R.matmulOProj, blk.oProj, S.d / R.matmulRowsPerWG, 1, 1, 'oproj')
      } else if (!fused) {
        // QKV matmul: B.hidden1 → B.qkvOut
        dispatch(enc, R.matmul, blk.qkv!, S.qkvDim / R.matmulRowsPerWG, 1, 1, 'qkvMatmul')
        if (blk.qkFused) {
          // ?fuseqk: fused qk_norm+RoPE+append — B.qkvOut → B.qOut + kvPages[L]
          // in one dispatch (replaces the qkNorm/rope/kvAppend chain below).
          dispatch(enc, P.qkNormRopeAppend, blk.qkFused, QK_FUSE_WGS, 1, 1, 'qkNormRopeAppend')
        } else {
          // QK-norm (Qwen3): per-head Q/K RMSNorm in place on B.qkvOut, pre-RoPE
          if (S.qkNorm) dispatch(enc, P.qkNorm, blk.qkNorm!, QK_NORM_WGS, 1, 1, 'qkNorm')
          // RoPE: B.qkvOut → B.qOut, B.kOut, B.vOut
          dispatch(enc, P.rope, bgRope!, QKV_WGS, 1, 1, 'rope')
          // KV append: kOut, vOut → kvPages[L] (grid covers KV_DIM elements)
          dispatch(enc, P.kvAppend, blk.kvApp!, KV_WGS, 1, 1, 'kvAppend')
        }
        // Attention: Q + kvPages[L] → B.attnOut
        attentionF16()
        dispatch(enc, R.matmulOProj, blk.oProj, S.d / R.matmulRowsPerWG, 1, 1, 'oproj')
      } else if (int8Mode) {
        dispatch(enc, P.qkvFusedScratch, blk.qkv!, S.qkvPairs, 1, 1, 'qkvFused')
        dispatch(enc, P.kvQuantizeInt8, blk.kvQuantize!, S.kvHeads * 2, 1, 1, 'kvQuantize')
        dispatch(enc, P.attentionInt8, blk.attn!, 1, S.heads, 1, 'attention')
        dispatch(enc, R.matmulOProj, blk.oProj, S.d / R.matmulRowsPerWG, 1, 1, 'oproj')
      } else {
        // Fused QKV+RoPE+KV-append: B.hidden1 → B.qOut + kvPages[L]
        dispatch(enc, R.qkvFused, blk.qkv!, S.qkvPairs / R.qkvPairsPerWG, 1, 1, 'qkvFused')
        attentionF16()
        dispatch(enc, R.matmulOProj, blk.oProj, S.d / R.matmulRowsPerWG, 1, 1, 'oproj')
      }
      if (R.ffnPrologue) {
        // ?fuseprologue=1: no addNorm1 dispatch. The FFN kernel computes
        // rIn + hidden2 and its RMSNorm in its own prologue; ffnDown lands
        // in hidden3; add3_norm merges rIn + hidden2 + hidden3 at the tail.
        dispatch(enc, R.ffn, blk.ffn!, S.ffn / R.ffnRowsPerWG, 1, 1, 'fusedFfn')
        dispatch(enc, R.matmulFfnDown, blk.ffnDown!, S.d / R.matmulRowsPerWG, 1, 1, 'ffnDown')
        dispatch(enc, P.add3Norm, blk.addNorm2, 1, 1, 1, 'addNorm2')
      } else {
        // AddNorm (attention): residual += hidden2; hidden1 = RMSNorm(residual)
        dispatch(enc, P.addNorm, blk.addNorm1!, 1, 1, 1, 'addNorm1')
        if (blk.moe) {
          // Sparse MoE, seven dispatches, B.hidden1 → B.hidden2. Every slot —
          // the top-K routed experts AND the shared one at index E — rides in
          // grid z, so the expert count costs dispatches, not the top-k.
          const M = S.moe!
          dispatch(enc, P.moeRouterLogits, blk.moe.routerLogits, M.experts + 1, 1, 1, 'moeRouterLogits')
          dispatch(enc, P.moeRouterTopk!, blk.moe.routerTopk, 1, 1, 1, 'moeRouterTopk')
          const rows = S.ffn / MOE_RPW
          dispatch(enc, moeMM!, blk.moe.gate, rows, 1, S.moeSlots, 'moeGate')
          dispatch(enc, moeMM!, blk.moe.up, rows, 1, S.moeSlots, 'moeUp')
          dispatch(enc, P.siluMul, blk.moe.silu, Math.ceil(S.moeSlots * S.ffn / 256), 1, 1, 'moeSilu')
          dispatch(enc, moeMM!, blk.moe.down, S.d / MOE_RPW, 1, S.moeSlots, 'moeDown')
          dispatch(enc, P.moeCombine, blk.moe.combine, Math.ceil(S.d / 256), 1, 1, 'moeCombine')
        } else {
        // Fused FFN gate+up+SiLU: B.hidden1 → B.ffnOut
        dispatch(enc, R.ffn, blk.ffn!, S.ffn / R.ffnRowsPerWG, 1, 1, 'fusedFfn')
        // FFN down: B.ffnOut → B.hidden2 (K = ffn instance)
        dispatch(enc, R.matmulFfnDown, blk.ffnDown!, S.d / R.matmulRowsPerWG, 1, 1, 'ffnDown')
        }
        // AddNorm (FFN): residual += hidden2; hidden1 = RMSNorm(residual)
        //   For last layer the bind group binds finalNormGamma instead of next
        //   layer's normGamma1, so hidden1 is ready for the LM head.
        dispatch(enc, P.addNorm, blk.addNorm2, 1, 1, 1, 'addNorm2')
      }
    }

    // --- LM HEAD: B.hidden1 (already normalized with model.norm) → B.logits ---
    // vocab (Phi-3: 32064, Qwen3: 151936) is divisible by 4, so rowsPerWG=4
    // works exactly. Grids past maxComputeWorkgroupsPerDimension (65535 —
    // Qwen3's 151936-row scalar case) are folded into z; the kernels index by
    // blockIdx.z * gridDim.x + blockIdx.x and guard on packGridDimX.
    const lmWGs = S.vocab / R.matmulRowsPerWG
    if (lmWGs > 65535) {
      const lmX = 16384
      dispatch(enc, R.matmulF32, bgLmHead, lmX, 1, Math.ceil(lmWGs / lmX), 'lmHead')
    } else {
      dispatch(enc, R.matmulF32, bgLmHead, lmWGs, 1, 1, 'lmHead')
    }
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
    const nnzPages = Math.floor(position / S.pageSize) + 1
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
    // gdn_conv selects its ring slots from the absolute position (pos at
    // byte offset 0 of its PODArgs).
    if (gdnConvU) {
      gdnPosScratch[0] = position
      device.queue.writeBuffer(gdnConvU, 0, gdnPosScratch)
    }
  }

  /**
   * Zero every GDN layer's conv ring + recurrent state. Recorded into the
   * step's own command encoder whenever a forward pass runs at position 0, so
   * a fresh prefill (new conversation, validation prompt, replay) never sees
   * stale state — and the clear is queue-ordered before the pass that reads it.
   */
  function clearGdnState(enc: GPUCommandEncoder): void {
    for (const b of gdnStateBufs) enc.clearBuffer(b)
  }

  // ============================================================
  // Blocking path — decodeToken / forwardLogits (validation harness)
  // ============================================================

  async function decodeToken(tokenId: number, position: number): Promise<number> {
    if (position < 0 || position >= MAX_CONTEXT) {
      throw new Error(
        `zero-tvm: context overflow — position ${position} exceeds max context ` +
        `${MAX_CONTEXT} tokens (maxPages=${S.maxPages} × pageSize=${S.pageSize}). ` +
        `Shorten the prompt or raise maxPages in src/compiler/model-spec.ts (costs ~${Math.round(S.layers * S.kvPageStride * 2 / (1024 * 1024))} MB of KV cache per page block).`
      )
    }
    writeStepState(tokenId, position)

    const enc = device.createCommandEncoder()
    if (position === 0) clearGdnState(enc)
    recordForward(enc)
    // Fold the argmax readback into the same command encoder → one submit per token.
    enc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
    device.queue.submit([enc.finish()])
    gdnStatePos = position + 1
    noteAbsorbed(position, tokenId)

    await readBuf.mapAsync(GPUMapMode.READ)
    const result = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    return result
  }

  /**
   * Read back the argmax the LAST submitted forward pass left in B.tokenOut —
   * no new forward pass. Used by the hybrid generate() when the GDN state
   * already sits exactly at end-of-prompt (e.g. right after forwardLogits):
   * the pure-attention path re-runs the final prompt token idempotently to
   * recover this value, but a GDN re-run would double-apply it to S.
   */
  async function readLastToken(): Promise<number> {
    const enc = device.createCommandEncoder()
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
        size: S.vocab * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'logitsReadback',
      })
    }
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(B.logits, 0, logitsReadBuf, 0, S.vocab * 4)
    device.queue.submit([enc.finish()])

    await logitsReadBuf.mapAsync(GPUMapMode.READ)
    const out = new Float32Array(logitsReadBuf.getMappedRange().slice(0))
    logitsReadBuf.unmap()
    return out
  }

  const STOP = new Set(S.stops)

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
    if (hybrid && startPos >= promptIds.length && gdnStatePos === promptIds.length) {
      // GDN state + KV sit exactly at end-of-prompt (the caller — e.g.
      // forwardLogits — already ran every prompt token). Re-running the last
      // token, as the pure-attention branch below does, would double-apply
      // it to the non-idempotent recurrent state; instead read back the
      // argmax that final pass already computed. Zero prompt passes.
      tokenId = await readLastToken()
    } else if (hybrid && startPos < promptIds.length && startPos > 0 && gdnStatePos === startPos) {
      // The state provably sits at startPos — extend incrementally, exactly
      // one forward pass per NEW prompt token (same contract as the
      // pure-attention KV reuse below).
      for (let i = startPos; i < promptIds.length; i++) {
        tokenId = await decodeToken(promptIds[i], i)
      }
    } else if (hybrid) {
      // GDN layers are stateful and NOT idempotent: re-running a token
      // double-applies it to the recurrent state and rotates the conv ring
      // past it. When gdnStatePos can't prove the state matches startPos
      // (fresh prompt, or state left mid-sequence by a previous generation),
      // replay the whole prompt from position 0 — which also re-zeroes the
      // GDN state (clearGdnState fires at position 0). Within-call decode
      // below is always incremental: one forward pass per generated token.
      for (let i = 0; i < promptIds.length; i++) {
        tokenId = await decodeToken(promptIds[i], i)
      }
    } else if (startPos >= promptIds.length) {
      // The new prompt is a strict prefix of the previous one (or identical).
      // No new tokens to prefill — but we still need a valid `tokenId` to
      // start decoding from. Run the last prompt token through decodeToken at
      // its existing position to re-read the logits (idempotent for pure
      // attention: the KV slot is simply rewritten with the same values).
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
      if (tokenId < 0 || tokenId >= S.vocab || STOP.has(tokenId)) break
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

  /**
   * Opt-in debug assertion for prefix reuse (?checkreuse=1 / window.checkReuse):
   * runs the REUSED-prefix prefill of `promptIds` and reads the final-position
   * logits, then a FRESH full prefill of the same prompt, and diffs the two
   * f32 logit vectors. Every dispatch is deterministic and the reused prefix's
   * KV/GDN state is bit-identical to what the fresh replay recomputes, so the
   * expected maxAbsDiff is exactly 0. Blocking path (scalar-config dispatch
   * chain); leaves KV + GDN state at end-of-prompt, absorbed record intact.
   */
  async function debugCompareReuse(promptIds: number[]): Promise<{
    startPos: number
    promptLen: number
    maxAbsDiff: number
    meanAbsDiff: number
  }> {
    if (promptIds.length < 2) throw new Error('debugCompareReuse: prompt too short')
    const startPos = computeReuseStart(promptIds)
    // Reused-prefix pass: prefill only the delta, then read logits.
    for (let i = startPos; i < promptIds.length - 1; i++) {
      await decodeToken(promptIds[i], i)
    }
    const reused = await readLogits(promptIds[promptIds.length - 1], promptIds.length - 1)
    // Fresh pass: full prefill from 0 (re-zeroes GDN state at position 0).
    for (let i = 0; i < promptIds.length - 1; i++) {
      await decodeToken(promptIds[i], i)
    }
    const fresh = await readLogits(promptIds[promptIds.length - 1], promptIds.length - 1)
    let maxAbs = 0
    let sumAbs = 0
    for (let i = 0; i < fresh.length; i++) {
      const d = Math.abs(reused[i] - fresh[i])
      if (d > maxAbs) maxAbs = d
      sumAbs += d
    }
    return { startPos, promptLen: promptIds.length, maxAbsDiff: maxAbs, meanAbsDiff: sumAbs / fresh.length }
  }

  function getLastPrefill(): { promptLen: number; reused: number; chunks: number } | null {
    return lastPrefill
  }

  function resetKVTracking(): void {
    // Drop the absorbed-token record so the next generatePipelined performs a
    // full prefill (the KV pages themselves need no clearing — stale slots
    // are overwritten in order, and a from-0 prefill re-zeroes GDN state).
    // Blocking-path callers additionally track their own prefix length and
    // pass startPos to generate().
    absorbed = []
    absorbedValid = true
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

  // ============================================================
  // Chunked GDN prefill (hybrid specs, subgroups path)
  //
  // Prompt tokens before the last are processed in chunks of up to CHUNK_CAP:
  // every projection (fused GDN in_proj, out_proj, c_attn, o_proj, gate_up,
  // down) becomes ONE int4_matmul_batched_dyn dispatch with M = chunk length
  // (4× weight-traffic amortization from the m=4 register block), the GDN
  // conv/gates/norm run batched over the chunk, and the recurrence — already
  // seq-capable — runs as ONE gdn_recur dispatch per layer per chunk.
  // Attention layers run fully batched too: rope/kv_append are seq-capable,
  // and attention_prefill enforces causality with per-token kv_len. One
  // submit per chunk; per-chunk uniforms are rewritten between submits
  // (queue-ordered). ~394 dispatches per 64-token chunk vs 340/token before.
  // ============================================================

  const CHUNK_CAP = 64   // chunk capacity in tokens (buffer sizing)
  const CHUNK_MIN = 8    // below this, the per-token path is not worth the uniform churn

  interface ChunkPrefill { record(promptIds: number[], start: number, seqLen: number): void }

  function buildChunkPrefill(): ChunkPrefill {
    const C = CHUNK_CAP
    const dyn = P.int4MatmulBatchedDyn!
    // Batched activation buffers ([C, dim] row-major).
    const CB = {
      inputIds: makeBuf(device, C * 4, 'c.inputIds'),
      posMap:   makeBuf(device, C * 4, 'c.posMap'),
      residual: makeBuf(device, C * S.d * 2, 'c.residual'),
      residual2: makeBuf(device, C * S.d * 2, 'c.residual2'),
      hidden1:  makeBuf(device, C * S.d * 2, 'c.hidden1'),
      hidden2:  makeBuf(device, C * S.d * 2, 'c.hidden2'),
      cAttnOut: makeBuf(device, C * S.cAttnDim * 2, 'c.cAttnOut'),
      qkvOut:   makeBuf(device, C * S.qkvDim * 2, 'c.qkvOut'),
      gateRaw:  makeBuf(device, C * S.qDim * 2, 'c.gateRaw'),
      qOut:     makeBuf(device, C * S.qDim * 2, 'c.qOut'),
      kOut:     makeBuf(device, C * S.kvDim * 2, 'c.kOut'),
      vOut:     makeBuf(device, C * S.kvDim * 2, 'c.vOut'),
      attnOut:  makeBuf(device, C * S.qDim * 2, 'c.attnOut'),
      gateUp:   makeBuf(device, C * 2 * S.ffn * 2, 'c.gateUp'),
      ffnOut:   makeBuf(device, C * S.ffn * 2, 'c.ffnOut'),
      gdnProjOut: makeBuf(device, C * S.gdnProjRows * 2, 'c.gdnProjOut'),
      gdnConvOut: makeBuf(device, C * S.gdnQkvDim * 2, 'c.gdnConvOut'),
      gdnGates:   makeBuf(device, C * 2 * S.gdnVHeads * 4, 'c.gdnGates'),
      gdnRecurOut: makeBuf(device, C * S.gdnVDim * 4, 'c.gdnRecurOut'),
      gdnNormed:  makeBuf(device, C * S.gdnVDim * 2, 'c.gdnNormed'),
    }

    // Elementwise grids per token (packGridDimX values scale with seq_len).
    const CATTN_WGS = S.cAttnDim / WG_SIZE_D
    const FFN_WGS = S.ffn / WG_SIZE_D
    const CONV_COMMIT_WGS = ((S.gdnConvK - 1) * S.gdnQkvDim) / WG_SIZE_D

    // Per-chunk uniforms — contents rewritten before each chunk's submit.
    // int4_matmul_batched_dyn PODArgs: {K_PACKED, SCALES_PER_ROW, N, M_ROWS};
    // M_ROWS (offset 12) is the per-chunk field.
    const cU = {
      emb:   uniformBuf(device, [i32(0), u32(0)]),
      norm:  uniformBuf(device, [u32(0)]),
      gdnProj: uniformBuf(device, [u32(S.dPacked), u32(S.d / QGROUP), u32(S.gdnProjRows), u32(0)]),
      gdnOut:  uniformBuf(device, [u32(S.gdnVDim / PACK), u32(S.gdnVDim / QGROUP), u32(S.d), u32(0)]),
      cAttn:   uniformBuf(device, [u32(S.dPacked), u32(S.d / QGROUP), u32(S.cAttnDim), u32(0)]),
      oProj:   uniformBuf(device, [u32(S.qDim / PACK), u32(S.qDim / QGROUP), u32(S.d), u32(0)]),
      gateUp:  uniformBuf(device, [u32(S.dPacked), u32(S.d / QGROUP), u32(2 * S.ffn), u32(0)]),
      ffnDown: uniformBuf(device, [u32(S.ffn / PACK), u32(S.ffn / QGROUP), u32(S.d), u32(0)]),
      gatedSplit: uniformBuf(device, [i32(0), u32(0)]),
      qkNorm: uniformBuf(device, [i32(0), u32(0)]),
      rope:   uniformBuf(device, [i32(1), i32(0), i32(0), u32(0)]),
      kvApp:  uniformBuf(device, [i32(0), i32(S.maxPages), i32(0), i32(0), u32(0)]),
      attn:   uniformBuf(device, [i32(0), i32(0), (() => { const a = new ArrayBuffer(4); new DataView(a).setFloat32(0, SM_SCALE, true); return a })()]),
      attnGate: uniformBuf(device, [u32(0)]),
      conv:   uniformBuf(device, [i32(0), i32(0), i32(S.gdnProjRows), u32(0)]),
      convCommit: uniformBuf(device, [i32(0), i32(0), i32(S.gdnProjRows), u32(CONV_COMMIT_WGS)]),
      gates:  uniformBuf(device, [i32(0), i32(S.gdnProjRows), u32(0)]),
      recur:  uniformBuf(device, [i32(0), u32(S.gdnVHeads)]),
      normOut: uniformBuf(device, [i32(0), i32(S.gdnProjRows), u32(0)]),
      silu:   uniformBuf(device, [i32(0), u32(0)]),
    }

    // Bind groups (buffers are fixed; only uniform contents change per chunk).
    const cbgEmb = bg(device, P.embedding, [CB.residual, CB.inputIds, weights.embdScales, weights.embdWeights, cU.emb])
    const cbgInitNorm = bg(device, P.rmsNorm, [CB.hidden1, CB.residual, weights.layers[0].normGamma1, cU.norm])

    interface ChunkLayerBG {
      // attention layers
      cAttn?: GPUBindGroup
      gatedSplit?: GPUBindGroup
      qkNorm?: GPUBindGroup
      rope?: GPUBindGroup
      kvApp?: GPUBindGroup
      attn?: GPUBindGroup
      attnGate?: GPUBindGroup
      // GDN layers
      gdnProj?: GPUBindGroup
      conv?: GPUBindGroup
      convCommit?: GPUBindGroup
      gates?: GPUBindGroup
      recur?: GPUBindGroup
      normOut?: GPUBindGroup
      // shared
      oProj: GPUBindGroup
      addNorm1: GPUBindGroup
      gateUp: GPUBindGroup
      silu: GPUBindGroup
      ffnDown: GPUBindGroup
      addNorm2: GPUBindGroup
    }
    const cLayers: ChunkLayerBG[] = []
    for (let L = 0; L < S.layers; L++) {
      const lw = weights.layers[L]
      const isGdn = S.layerKinds[L] === 'gdn'
      const nextGamma = L < S.layers - 1 ? weights.layers[L + 1].normGamma1 : weights.finalNormGamma
      const common = {
        addNorm1: bg(device, P.addNorm, [CB.hidden2, CB.residual, lw.normGamma2, CB.hidden1, CB.residual2, cU.norm]),
        // Chunked prefill is dense-only — buildDecodeEngine sets chunkPrefill
        // null for a MoE spec, so ffn* are present wherever this runs.
        gateUp: bg(device, dyn, [CB.gateUp, CB.hidden1, lw.ffnScales!, lw.ffnWeights!, cU.gateUp]),
        silu: bg(device, P.siluMul, [CB.ffnOut, CB.gateUp, cU.silu]),
        ffnDown: bg(device, dyn, [CB.hidden2, CB.ffnOut, lw.ffnDownScales!, lw.ffnDownWeights!, cU.ffnDown]),
        addNorm2: bg(device, P.addNorm, [CB.hidden2, CB.residual2, nextGamma, CB.hidden1, CB.residual, cU.norm]),
      }
      if (isGdn) {
        const gw = lw.gdn!
        // Region views into the BATCHED packed projection [C, qkv|z|a|b]:
        // token t's z / [a|b] rows sit at t·gdnProjRows + region offset — the
        // kernels stride by gdnProjRows (z_stride / ab_stride uniforms).
        const zRegion = {
          buffer: CB.gdnProjOut, offset: S.gdnQkvDim * 2,
          size: (C - 1) * S.gdnProjRows * 2 + S.gdnVDim * 2,
        }
        const abRegion = {
          buffer: CB.gdnProjOut, offset: (S.gdnQkvDim + S.gdnVDim) * 2,
          size: (C - 1) * S.gdnProjRows * 2 + 2 * S.gdnVHeads * 2,
        }
        cLayers.push({
          gdnProj: bg(device, dyn, [CB.gdnProjOut, CB.hidden1, gw.projScales, gw.projWeights, cU.gdnProj]),
          conv: bg(device, P.gdnConvSeq, [CB.gdnConvOut, CB.gdnProjOut, gdnConvState[L]!, gw.convWeight, cU.conv]),
          convCommit: bg(device, P.gdnConvCommit, [gdnConvState[L]!, CB.gdnProjOut, cU.convCommit]),
          gates: bg(device, P.gdnGates, [CB.gdnGates, abRegion, gw.aLog, gw.dtBias, cU.gates]),
          recur: bg(device, P.gdnRecur, [CB.gdnRecurOut, CB.gdnConvOut, CB.gdnGates, gdnRecurState[L]!, cU.recur]),
          normOut: bg(device, P.gdnNormOut, [CB.gdnNormed, CB.gdnRecurOut, gw.normGamma, zRegion, cU.normOut]),
          oProj: bg(device, dyn, [CB.hidden2, CB.gdnNormed, gw.outScales, gw.outWeights, cU.gdnOut]),
          ...common,
        })
      } else {
        cLayers.push({
          cAttn: bg(device, dyn, [CB.cAttnOut, CB.hidden1, lw.qkvScales!, lw.qkvWeights!, cU.cAttn]),
          gatedSplit: bg(device, P.gatedQkvSplit, [CB.qkvOut, CB.gateRaw, CB.cAttnOut, cU.gatedSplit]),
          qkNorm: bg(device, P.qkNorm, [CB.qkvOut, lw.qNormGamma!, lw.kNormGamma!, cU.qkNorm]),
          rope: bg(device, P.rope, [CB.qOut, CB.kOut, CB.vOut, CB.qkvOut, CB.posMap, cU.rope]),
          kvApp: bg(device, P.kvAppend, [CB.kOut, CB.vOut, kvPages[kvIndex[L]], CB.posMap, cU.kvApp]),
          attn: bg(device, P.attentionPrefill, [CB.qOut, B.pageValues, kvPages[kvIndex[L]], CB.attnOut, cU.attn]),
          attnGate: bg(device, P.attnGate, [CB.attnOut, CB.gateRaw, cU.attnGate]),
          oProj: bg(device, dyn, [CB.hidden2, CB.attnOut, lw.oProjScales!, lw.oProjWeights!, cU.oProj]),
          ...common,
        })
      }
    }

    function record(promptIds: number[], start: number, seqLen: number): void {
      const n = seqLen
      // Per-chunk state + uniform writes — queue-ordered before the submit.
      const ids = new Int32Array(n)
      const posv = new Int32Array(n)
      for (let t = 0; t < n; t++) { ids[t] = promptIds[start + t]; posv[t] = start + t }
      device.queue.writeBuffer(CB.inputIds, 0, ids)
      device.queue.writeBuffer(CB.posMap, 0, posv)
      device.queue.writeBuffer(cU.emb, 0, new Int32Array([n, n * D_WGS]))
      device.queue.writeBuffer(cU.norm, 0, new Uint32Array([n]))
      const m = new Uint32Array([n])
      for (const u of [cU.gdnProj, cU.gdnOut, cU.cAttn, cU.oProj, cU.gateUp, cU.ffnDown]) {
        device.queue.writeBuffer(u, 12, m)
      }
      device.queue.writeBuffer(cU.gatedSplit, 0, new Int32Array([n, n * CATTN_WGS]))
      device.queue.writeBuffer(cU.qkNorm, 0, new Int32Array([n, n * QK_NORM_WGS]))
      device.queue.writeBuffer(cU.rope, 8, new Int32Array([n, n * QKV_WGS]))
      device.queue.writeBuffer(cU.kvApp, 0, new Int32Array([n]))
      device.queue.writeBuffer(cU.kvApp, 16, new Uint32Array([n * KV_WGS]))
      device.queue.writeBuffer(cU.attn, 0, new Int32Array([n, start + 1]))
      device.queue.writeBuffer(cU.attnGate, 0, new Uint32Array([n * ATTN_GATE_WGS]))
      device.queue.writeBuffer(cU.conv, 0, new Int32Array([start, n]))
      device.queue.writeBuffer(cU.conv, 12, new Uint32Array([n * GDN_CONV_WGS]))
      device.queue.writeBuffer(cU.convCommit, 0, new Int32Array([start, n]))
      device.queue.writeBuffer(cU.gates, 0, new Int32Array([n]))
      device.queue.writeBuffer(cU.gates, 8, new Uint32Array([n]))
      device.queue.writeBuffer(cU.recur, 0, new Int32Array([n]))
      device.queue.writeBuffer(cU.normOut, 0, new Int32Array([n]))
      device.queue.writeBuffer(cU.normOut, 8, new Uint32Array([n * S.gdnVHeads]))
      device.queue.writeBuffer(cU.silu, 0, new Int32Array([n, n * FFN_WGS]))

      const dynGrid = (rows: number) => Math.ceil(rows / 4)
      const enc = device.createCommandEncoder()
      if (start === 0) clearGdnState(enc)
      dispatch(enc, P.embedding, cbgEmb, n * D_WGS, 1, 1, 'cEmbedding')
      dispatch(enc, P.rmsNorm, cbgInitNorm, n, 1, 1, 'cRmsNormInit')
      for (let L = 0; L < S.layers; L++) {
        const blk = cLayers[L]
        if (blk.gdnProj) {
          dispatch(enc, dyn, blk.gdnProj, dynGrid(S.gdnProjRows), 1, 1, 'cGdnProj')
          dispatch(enc, P.gdnConvSeq, blk.conv!, n * GDN_CONV_WGS, 1, 1, 'cGdnConv')
          dispatch(enc, P.gdnConvCommit, blk.convCommit!, CONV_COMMIT_WGS, 1, 1, 'cGdnConvCommit')
          dispatch(enc, P.gdnGates, blk.gates!, n, 1, 1, 'cGdnGates')
          dispatch(enc, P.gdnRecur, blk.recur!, S.gdnVHeads, 1, 1, 'cGdnRecur')
          dispatch(enc, P.gdnNormOut, blk.normOut!, n * S.gdnVHeads, 1, 1, 'cGdnNormOut')
          dispatch(enc, dyn, blk.oProj, dynGrid(S.d), 1, 1, 'cGdnOutProj')
        } else {
          dispatch(enc, dyn, blk.cAttn!, dynGrid(S.cAttnDim), 1, 1, 'cCAttn')
          dispatch(enc, P.gatedQkvSplit, blk.gatedSplit!, n * CATTN_WGS, 1, 1, 'cGatedSplit')
          if (S.qkNorm) dispatch(enc, P.qkNorm, blk.qkNorm!, n * QK_NORM_WGS, 1, 1, 'cQkNorm')
          dispatch(enc, P.rope, blk.rope!, n * QKV_WGS, 1, 1, 'cRope')
          dispatch(enc, P.kvAppend, blk.kvApp!, n * KV_WGS, 1, 1, 'cKvAppend')
          dispatch(enc, P.attentionPrefill, blk.attn!, n, S.heads, 1, 'cAttention')
          dispatch(enc, P.attnGate, blk.attnGate!, n * ATTN_GATE_WGS, 1, 1, 'cAttnGate')
          dispatch(enc, dyn, blk.oProj, dynGrid(S.d), 1, 1, 'cOProj')
        }
        dispatch(enc, P.addNorm, blk.addNorm1, n, 1, 1, 'cAddNorm1')
        dispatch(enc, dyn, blk.gateUp, dynGrid(2 * S.ffn), 1, 1, 'cGateUp')
        dispatch(enc, P.siluMul, blk.silu, n * FFN_WGS, 1, 1, 'cSiluMul')
        dispatch(enc, dyn, blk.ffnDown, dynGrid(S.d), 1, 1, 'cFfnDown')
        dispatch(enc, P.addNorm, blk.addNorm2, n, 1, 1, 'cAddNorm2')
      }
      device.queue.submit([enc.finish()])
      gdnStatePos = start + n
      for (let t = 0; t < n; t++) noteAbsorbed(start + t, promptIds[start + t])
    }

    return { record }
  }

  // MoE specs opt OUT of chunked prefill, deliberately and for two independent
  // reasons — and Qwen3.6 IS hybrid, so without this it would turn on by
  // default and quietly record the dense gate_up/silu/ffn_down chain:
  //   1. buildChunkPrefill dispatches int4_matmul_batched_dyn for every
  //      projection, and int4_matmul.gen.ts forbids affine with mDyn.
  //   2. the MoE ids[] buffer is indexed by SLOT with no token dimension, so
  //      batching a chunk would apply one token's expert choice to all of them.
  // The cost is per-token prefill; correctness is not negotiable for a speedup.
  const chunkPrefill: ChunkPrefill | null =
    hybrid && !S.moe && (opts.chunkedPrefill ?? true) && P.int4MatmulBatchedDyn
      ? buildChunkPrefill()
      : null
  if (hybrid) {
    const why = chunkPrefill ? `on (cap ${CHUNK_CAP})`
      : S.moe ? 'off (per-token — MoE: affine has no batched_dyn, and ids[] has no token dim)'
      : 'off (per-token)'
    console.log(`[engine] chunked prefill: ${why}`)
  }

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
    if (position === 0) clearGdnState(enc)
    recordForward(enc)
    // GPU chain: next decode step reads inputIds[0] without a CPU round-trip.
    enc.copyBufferToBuffer(B.tokenOut, 0, B.inputIds, 0, 4)

    if (wantReadback) {
      const slot = readRing[readCursor]
      readCursor = (readCursor + 1) % readRing.length
      enc.copyBufferToBuffer(B.tokenOut, 0, slot, 0, 4)
      device.queue.submit([enc.finish()])
      gdnStatePos = position + 1
      if (writeInputId !== null) noteAbsorbed(position, writeInputId)
      return slot.mapAsync(GPUMapMode.READ).then(() => {
        const id = new DataView(slot.getMappedRange()).getInt32(0, true)
        slot.unmap()
        return id
      })
    }

    device.queue.submit([enc.finish()])
    gdnStatePos = position + 1
    if (writeInputId !== null) noteAbsorbed(position, writeInputId)
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
        `${MAX_CONTEXT} (maxPages=${S.maxPages} × pageSize=${S.pageSize})`
      )
    }

    // --- Prefill ---
    // Cross-turn prefix reuse: skip prompt tokens the engine has provably
    // already absorbed (KV slots + GDN state) — see computeReuseStart.
    const startPos = computeReuseStart(promptIds)
    const last = promptIds.length - 1
    let prefillPos = startPos
    let chunks = 0
    // Chunked prefill (hybrid + subgroups): all but the last prompt token run
    // in chunks — batched projections, one gdn_recur dispatch per layer per
    // chunk. Short tails fall through to the per-token path below.
    if (chunkPrefill) {
      while (last - prefillPos >= CHUNK_MIN) {
        const s = Math.min(CHUNK_CAP, last - prefillPos)
        chunkPrefill.record(promptIds, prefillPos, s)
        prefillPos += s
        chunks++
      }
    }
    // Per-token tail: fire without readback. The argmax of intermediate
    // prefill steps is not useful (we overwrite inputIds on the next step), so
    // skipping readback removes the CPU syncs.
    for (; prefillPos < last; prefillPos++) {
      submitStep(promptIds[prefillPos], prefillPos, false)
    }
    lastPrefill = { promptLen: promptIds.length, reused: startPos, chunks }
    if (startPos > 0 || chunks > 0) {
      console.log(
        `[engine] prefill: ${promptIds.length} tokens, reused prefix ${startPos}` +
        (chunks ? `, ${chunks} chunk${chunks > 1 ? 's' : ''} of ≤${CHUNK_CAP}` : ''),
      )
    }
    // Last prefill step: readback to get the first generated token.
    const firstTokenPromise = submitStep(promptIds[last], last, true)!
    // Readback bookkeeping for the absorbed record: the input of the step at
    // position p is the readback of the step at position p-1.
    const rbByPos = new Map<number, Promise<number>>()
    rbByPos.set(last, firstTokenPromise)
    const firstToken = await firstTokenPromise
    if (STOP.has(firstToken) || firstToken < 0 || firstToken >= S.vocab) return tokens
    tokens.push(firstToken)
    onToken(firstToken)
    if (tokens.length >= maxTokens) return tokens

    // --- Pipelined decode ---
    // Keep PIPELINE_DEPTH tokens of work in flight so the GPU never waits on
    // a CPU round-trip between tokens. argmax → inputIds[0] is chained on-GPU.
    //
    // Hybrid (GDN) note — why the 2-deep ring is safe with non-idempotent
    // state: the on-GPU chain never *mispredicts* a token (each step consumes
    // the previous step's real argmax, not a guess), so the only steps whose
    // state mutations are ever discarded are the ≤ PIPELINE_DEPTH-1 already
    // submitted when we break (stop token / maxTokens / shouldStop). Those
    // run positions past the end of the emitted sequence and mutate S/conv
    // rings there — harmless, because generation ends and no later call
    // trusts that state: this path always prefills from 0 (position 0 clears
    // GDN state) and the blocking generate() verifies gdnStatePos before
    // reusing state (falling back to a full replay on mismatch).
    const inFlight: Promise<number>[] = []
    let pos = promptIds.length
    const submitChained = (): void => {
      const p = submitStep(null, pos, true)!
      rbByPos.set(pos, p)
      inFlight.push(p)
      pos++
    }

    try {
      for (let k = 0; k < PIPELINE_DEPTH && tokens.length + k < maxTokens && pos < MAX_CONTEXT; k++) {
        submitChained()
      }

      while (inFlight.length > 0) {
        const tok = await inFlight.shift()!
        if (STOP.has(tok) || tok < 0 || tok >= S.vocab) break
        tokens.push(tok)
        onToken(tok)
        if (tokens.length >= maxTokens) break
        // Cooperative stop: never unwind mid-pipeline — just stop submitting;
        // the finally-drain below leaves the readback ring clean.
        if (shouldStop?.()) break
        // Keep the pipeline full, but don't overshoot maxTokens or the KV window.
        if (tokens.length + inFlight.length < maxTokens && pos < MAX_CONTEXT) {
          submitChained()
        }
      }
    } finally {
      // Drain pending readbacks (frees ring slots for the next generation —
      // mapAsync on a still-pending slot buffer would be a validation error).
      while (inFlight.length > 0) {
        await inFlight.shift()!.catch(() => {})
      }
      // Patch the absorbed record for the chained-argmax steps: every decode
      // step at position p (last+1 .. pos-1) consumed the readback of the
      // step at p-1 — including the ≤ PIPELINE_DEPTH-1 overrun steps past the
      // emitted text (in normal chat the first overrun input is the stop id,
      // which the next turn's prompt also contains). All these promises are
      // settled by the drain above.
      try {
        for (let p = promptIds.length; p < pos; p++) {
          noteAbsorbed(p, await rbByPos.get(p - 1)!)
        }
      } catch {
        absorbedValid = false
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

    // 2 slots per pass; worst case is the hybrid gated-attention path
    // (12/layer; GDN layers are 10 with the fused input projection) plus
    // embedding, init norm, LM head, argmax and a little headroom.
    const CAPACITY = 2 * (S.layers * 14 + 8)
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
    const K = isFfnDown ? S.ffn : S.qDim
    const N = S.d
    const uniformBufForTarget = isFfnDown ? ffnDnU : oProjU
    const oprojLayer = weights.layers.find((l) => l.oProjScales)  // first attention layer (hybrid: layer 0 may be GDN)
    if (!isFfnDown && !oprojLayer) {
      console.warn('[batched-bench] no attention layer with o_proj weights')
      return null
    }
    // The ffn_down target has no meaning on a MoE spec — there is no dense down
    // projection to bench. Say so instead of dereferencing an absent buffer.
    if (isFfnDown && !weights.layers[0].ffnDownScales) {
      console.warn('[bench] ffn_down target: this spec has a sparse MoE FFN, no dense down projection')
      return null
    }
    const scalesBuf = isFfnDown ? weights.layers[0].ffnDownScales! : oprojLayer!.oProjScales!
    const weightsBuf = isFfnDown ? weights.layers[0].ffnDownWeights! : oprojLayer!.oProjWeights!
    const batchedPipeline = P.int4MatmulBatchedM4
    // Per-instance vec4 gating: compare against the pipeline the engine
    // actually dispatches for this K (identical to R.matmul on Phi-3).
    const tiledPipeline = isFfnDown ? R.matmulFfnDown : R.matmulOProj

    const inBuf  = makeBuf(device, M * K * 2, 'batchedBench.in')
    const outBuf = makeBuf(device, M * N * 2, 'batchedBench.out')
    const tiledOutBuf = makeBuf(device, N * 2, 'batchedBench.tiledOut')

    const inf16 = new Uint16Array(M * K)
    for (let i = 0; i < inf16.length; i++) inf16[i] = 0x2a00 | (i & 0x03ff)
    device.queue.writeBuffer(inBuf, 0, inf16)

    const batchedBG = bg(device, batchedPipeline,
      [outBuf, inBuf, scalesBuf, weightsBuf, uniformBufForTarget])
    const tiledBG = bg(device, tiledPipeline,
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
        p2.setPipeline(tiledPipeline); p2.setBindGroup(0, tiledBG)
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
        pass.setPipeline(tiledPipeline); pass.setBindGroup(0, tiledBG)
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
    debugCompareReuse,
    getLastPrefill,
    maxContext: MAX_CONTEXT,
    spec: S,
    profileStep,
    benchBatchedFfnDown,
  }
}
