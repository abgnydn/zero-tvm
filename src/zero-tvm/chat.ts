/**
 * ZERO-TVM CHAT — Phi-3 inference. No WebLLM. No TVM.
 *
 * Loads weights from browser cache (cached by a prior WebLLM session)
 * or fetches from HuggingFace. Uses 10 hand-written WGSL kernel roles
 * (across 27 .wgsl files counting tiled/subgroup variants).
 *
 * Correct buffer layout:
 *   B.residual  — persistent running residual (updated in-place by add_norm)
 *   B.hidden1   — normed scratch (matmul input)
 *   B.hidden2   — matmul output scratch (OProj, FFN down)
 */

import { loadWeights, LoadedWeights } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt, Tokenizer } from './tokenizer.js'
import { summarize as summarizePLD } from './spec-sim.js'
import { compile, PHI3 } from '../compiler/compiler.js'

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

// Map a matmul variant name to the int4 + int4-f32 pipelines and the
// rows-per-workgroup count the caller needs for dispatch. Falls back to
// scalar when a requested _sg/_tiled pipeline isn't compiled (subgroups
// feature disabled or variant name unknown).
function resolveMatmul(
  variant: 'tiled8' | 'tiled' | 'sg' | 'scalar',
  P: ReturnType<typeof compile>['pipelines'],
): { pipeline: GPUComputePipeline; pipelineF32: GPUComputePipeline; rowsPerWG: number; label: string } {
  if (variant === 'tiled8' && P.int4MatmulTiled8 && P.int4MatmulF32Tiled8) {
    return { pipeline: P.int4MatmulTiled8, pipelineF32: P.int4MatmulF32Tiled8, rowsPerWG: 8, label: 'tiled8' }
  }
  if (variant === 'tiled' && P.int4MatmulTiled && P.int4MatmulF32Tiled) {
    return { pipeline: P.int4MatmulTiled, pipelineF32: P.int4MatmulF32Tiled, rowsPerWG: 4, label: 'tiled' }
  }
  if (variant === 'sg' && P.int4MatmulSg && P.int4MatmulF32Sg) {
    return { pipeline: P.int4MatmulSg, pipelineF32: P.int4MatmulF32Sg, rowsPerWG: 1, label: 'sg' }
  }
  return { pipeline: P.int4Matmul, pipelineF32: P.lmHead, rowsPerWG: 1, label: 'scalar' }
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
// KV cache config + allocation
// ============================================================

// Experimental: int8 KV cache with per-(head, slot, K|V) f16 scale.
// Halves KV memory from ~1.6GB to ~800MB for 4K context, at the cost of one
// extra dispatch per layer (quantize). Leave false unless you've validated
// output parity against the f16 baseline on your target hardware.
const USE_INT8_KV = false

function allocKVPages(device: GPUDevice): GPUBuffer[] {
  const bytesPerPage = 98304 * 2  // 32 heads * 16 slots * 96 dims * 2 (K+V) * 2 bytes
  const pages = PHI3.MAX_PAGES * bytesPerPage  // 257 * 196608 ≈ 50MB
  return Array.from({ length: PHI3.LAYERS }, (_, i) =>
    makeBuf(device, pages, `kvPages_${i}`)
  )
}

// int8 layout: 32 heads × 16 slots × 2 sides × 24 u32-words = 24576 u32 per page.
function allocKVPagesInt8(device: GPUDevice): { pages: GPUBuffer[]; scales: GPUBuffer[] } {
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

interface KernelProfile {
  kernels: Array<{ label: string; totalMs: number; calls: number; pctOfTotal: number }>
  totalMs: number
}

interface BatchedBenchResult {
  msBatched: number
  msTiledTotal: number
  msPerMBatched: number
  msPerMTiled: number
  speedup: number
}

interface DecodeEngine {
  generate(promptIds: number[], maxTokens: number, onToken: (id: number) => void): Promise<number[]>
  profileStep(warmupIds: number[]): Promise<KernelProfile | null>
  benchBatchedFfnDown(
    M: number,
    iters: number,
    target?: 'ffnDown' | 'oproj',
  ): Promise<BatchedBenchResult | null>
}

function buildDecodeEngine(
  device: GPUDevice,
  weights: LoadedWeights,
  kvPages: GPUBuffer[],
  kvInt8: { pages: GPUBuffer[]; scales: GPUBuffer[] } | null = null,
  opts: { sgSizeOk?: boolean } = {},
): DecodeEngine {
  // URL toggles for bisecting the subgroup path.
  //   ?sg=0       disable all subgroup shaders
  //   ?sgattn=0   disable subgroup attention only (keep _sg argmax)
  //   ?sgargmax=0 disable subgroup argmax only (keep _sg attention)
  const q = new URLSearchParams(location.search)
  const sgAll     = q.get('sg') !== '0'
  const sgAttn    = sgAll && q.get('sgattn') !== '0'
  const sgArgmax  = sgAll && q.get('sgargmax') !== '0'
  const sgQkv     = sgAll && q.get('sgqkv') !== '0'
  // Opt-in: tiled qkv (4 pairs/WG + input cache). Default off until A/B verified.
  const qkvTile   = sgAll && q.get('qkvtile') === '1'
  // Opt-in: tiled2sg qkv (2 pairs/WG, 64 threads, input cache). Default off until A/B verified.
  const qkvTile2  = sgAll && q.get('qkvtile2') === '1'
  const sgFfn     = sgAll && q.get('sgffn') !== '0'
  // _sg shaders assume subgroup size == 32 (full 32-thread workgroup in one
  // subgroup). If the device reports otherwise, fall back to scalar.
  const hasSgFeature = (device.features as ReadonlySet<string>).has('subgroups')
  const subgroups = sgAll && hasSgFeature && !!opts.sgSizeOk
  const { pipelines } = compile(device, { subgroups })
  const P = pipelines
  // Select subgroup variants when available. Bindings are identical to the
  // scalar versions, so callers can swap the pipeline reference and keep the
  // same bind group.
  // Matmul variants:
  //   tiled  — 4 output rows per workgroup, shares input reads across rows (default on Apple sg32)
  //   tiled8 — 8 rows per WG; measured ~5% slower than tiled4 on M-series (register pressure /
  //            reduced occupancy wins out over the halved input-vector DRAM traffic). Kept compiled
  //            so `?matmul=tiled8` can re-test on other GPUs.
  //   sg     — naive 32-thread + subgroupAdd (experimental, slower than scalar on Apple)
  //   scalar — original 64-thread tree reduction
  type MatmulVariant = 'tiled8' | 'tiled' | 'sg' | 'scalar'
  const requestedVariant = (q.get('matmul') ?? (sgAll ? 'tiled' : 'scalar')) as MatmulVariant
  const { pipeline: PMatmul, pipelineF32: PMatmulF32, rowsPerWG: matmulRowsPerWG, label: matmulLabel } =
    resolveMatmul(requestedVariant, P)
  const PAttn   = (sgAttn   && P.attentionSg) ? P.attentionSg : P.attention
  const PArgmax = (sgArgmax && P.argmaxSg)    ? P.argmaxSg    : P.argmax
  const PQkvFused = (qkvTile2 && P.qkvFusedTiled2Sg) ? P.qkvFusedTiled2Sg
                  : (qkvTile && P.qkvFusedTiledSg) ? P.qkvFusedTiledSg
                  : (sgQkv && P.qkvFusedSg) ? P.qkvFusedSg
                  : P.qkvFused
  // 1 pair/WG for scalar and _sg variants; 2 pairs/WG for tiled variants.
  const qkvPairsPerWG = (PQkvFused === P.qkvFusedTiledSg || PQkvFused === P.qkvFusedTiled2Sg) ? 2 : 1
  const qkvLabel = PQkvFused === P.qkvFusedTiled2Sg ? 'tiled2_sg'
                 : PQkvFused === P.qkvFusedTiledSg ? 'tiled_sg'
                 : PQkvFused === P.qkvFusedSg ? '_sg'
                 : 'scalar'
  const PFfn = (sgFfn && P.fusedFfnTiledSg) ? P.fusedFfnTiledSg : P.fusedFfn
  const ffnRowsPerWG = PFfn === P.fusedFfnTiledSg ? 4 : 1
  console.log(`[engine] attention=${PAttn === P.attentionSg ? '_sg' : 'scalar'} argmax=${PArgmax === P.argmaxSg ? '_sg' : 'scalar'} qkv=${qkvLabel} ffn=${PFfn === P.fusedFfnTiledSg ? 'tiled_sg' : 'scalar'} matmul=${matmulLabel} rowsPerWG=${matmulRowsPerWG}`)

  // Activation buffers
  const B = {
    residual:  makeBuf(device, PHI3.D * 2, 'residual'),      // running residual (ping)
    residual2: makeBuf(device, PHI3.D * 2, 'residual2'),     // running residual (pong)
    hidden1:   makeBuf(device, PHI3.D * 2, 'hidden1'),       // normed scratch
    hidden2:   makeBuf(device, PHI3.D * 2, 'hidden2'),       // matmul output scratch
    qOut:      makeBuf(device, PHI3.D * 2, 'qOut'),
    kSlot:     makeBuf(device, PHI3.D * 2, 'kSlot'),   // int8-mode scratch (unused otherwise)
    vSlot:     makeBuf(device, PHI3.D * 2, 'vSlot'),   // int8-mode scratch (unused otherwise)
    attnOut:   makeBuf(device, PHI3.D * 2, 'attnOut'),
    ffnOut:    makeBuf(device, PHI3.FFN * 2, 'ffnOut'),
    logits:    makeBuf(device, PHI3.VOCAB * 4, 'logits'),
    tokenOut:  makeBuf(device, 4, 'tokenOut'),
    inputIds:  makeBuf(device, 4, 'inputIds'),
    posMap:    makeBuf(device, 4, 'posMap'),
    pageIndptr: makeBuf(device, 8, 'pageIndptr'),
    pageValues: makeBuf(device, PHI3.MAX_PAGES * 4, 'pageValues'),
    lengthInfo: makeBuf(device, 12, 'lengthInfo'),
  }

  // Static uniforms (matmul dimensions, vocab sizes, etc.)
  const oProjU = uniformBuf(device, [u32(384), u32(96), u32(3072)])
  const ffnDnU = uniformBuf(device, [u32(1024), u32(256), u32(3072)])
  const lmHdU  = uniformBuf(device, [u32(384), u32(96), u32(PHI3.VOCAB)])
  const embU   = uniformBuf(device, [u32(1), u32(12)])
  const normU  = uniformBuf(device, [u32(1)])
  const ffnU   = uniformBuf(device, [u32(PHI3.FFN)])
  const argmaxU = uniformBuf(device, [u32(PHI3.VOCAB)])

  // Per-layer static uniforms (same values every token, every layer)
  const qkvFusedU = uniformBuf(device, [i32(0), i32(0), u32(4608)])         // f16-KV mode
  const qkvFusedScratchU = uniformBuf(device, [i32(0), u32(4608)])          // int8-KV mode
  const kvQuantU = uniformBuf(device, [i32(0), i32(0), i32(0), u32(64)])    // pos_off, pages_off, scales_off, 64 WG

  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)

  // Attention uniform (f16-KV mode): 9 fields, 36 bytes, padded to 48.
  // nnz_pages at offset 8 is the only per-token field; the rest are constant.
  const attnU = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const tmpl = new ArrayBuffer(48)
    const dv = new DataView(tmpl)
    dv.setInt32(0, 1, true)                    // B
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_num_pages
    // offset 8: nnz_pages — rewritten per token
    dv.setInt32(12, 0, true)                   // pages_elem_offset
    dv.setInt32(16, 0, true)                   // page_indptr_elem_offset
    dv.setInt32(20, 0, true)                   // page_values_elem_offset
    dv.setInt32(24, 0, true)                   // length_info_elem_offset
    dv.setFloat32(28, SM_SCALE, true)          // sm_scale
    dv.setUint32(32, 1, true)                  // packGridDimX
    device.queue.writeBuffer(attnU, 0, tmpl)
  }

  // Attention uniform (int8-KV mode): extra scales_elem_offset field; 10 fields,
  // 40 bytes, padded to 48.
  const attnI8U = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const tmpl = new ArrayBuffer(48)
    const dv = new DataView(tmpl)
    dv.setInt32(0, 1, true)                    // B
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_num_pages
    dv.setInt32(12, 0, true)                   // pages_elem_offset
    dv.setInt32(16, 0, true)                   // page_indptr_elem_offset
    dv.setInt32(20, 0, true)                   // page_values_elem_offset
    dv.setInt32(24, 0, true)                   // length_info_elem_offset
    dv.setInt32(28, 0, true)                   // scales_elem_offset
    dv.setFloat32(32, SM_SCALE, true)          // sm_scale
    dv.setUint32(36, 1, true)                  // packGridDimX
    device.queue.writeBuffer(attnI8U, 0, tmpl)
  }

  // Identity page table (page i → physical page i)
  {
    const pageVals = new Int32Array(PHI3.MAX_PAGES)
    for (let i = 0; i < PHI3.MAX_PAGES; i++) pageVals[i] = i
    device.queue.writeBuffer(B.pageValues, 0, pageVals)
  }

  // Pre-built bind groups (buffer refs are stable across decode tokens)
  const embedBG   = bg(device, P.embedding, [B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU])
  const initNormBG = bg(device, P.rmsNorm, [B.hidden1, B.residual, weights.layers[0].normGamma1, normU])
  const lmHeadBG  = bg(device, PMatmulF32, [B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU])
  const argmaxBG  = bg(device, PArgmax,   [B.logits, B.tokenOut, argmaxU])

  const int8Mode = kvInt8 !== null

  interface LayerBG {
    qkvFused: GPUBindGroup      // f16 mode: fused QKV+RoPE+KV-append | int8 mode: QKV+RoPE to scratch
    kvQuantize?: GPUBindGroup   // int8 mode only
    attn: GPUBindGroup
    oproj: GPUBindGroup
    addNorm1: GPUBindGroup
    ffn: GPUBindGroup
    ffnDown: GPUBindGroup
    addNorm2: GPUBindGroup
  }

  const layerBGs: LayerBG[] = []
  for (let L = 0; L < PHI3.LAYERS; L++) {
    const lw = weights.layers[L]
    const nextGamma = L < PHI3.LAYERS - 1
      ? weights.layers[L + 1].normGamma1
      : weights.finalNormGamma

    let qkvFusedBG: GPUBindGroup
    let attnBG: GPUBindGroup
    let kvQuantizeBG: GPUBindGroup | undefined
    if (int8Mode) {
      qkvFusedBG = bg(device, P.qkvFusedScratch, [
        B.qOut, B.kSlot, B.vSlot, B.hidden1, lw.qkvScales, lw.qkvWeights, B.posMap, qkvFusedScratchU,
      ])
      kvQuantizeBG = bg(device, P.kvQuantizeInt8, [
        B.kSlot, B.vSlot, kvInt8!.pages[L], kvInt8!.scales[L], B.posMap, kvQuantU,
      ])
      attnBG = bg(device, P.attentionInt8, [
        B.qOut, B.pageIndptr, B.pageValues, kvInt8!.pages[L], kvInt8!.scales[L],
        B.lengthInfo, B.attnOut, attnI8U,
      ])
    } else {
      qkvFusedBG = bg(device, PQkvFused, [
        B.qOut, kvPages[L], B.hidden1, lw.qkvScales, lw.qkvWeights, B.posMap, qkvFusedU,
      ])
      attnBG = bg(device, PAttn, [
        B.qOut, B.pageIndptr, B.pageValues, kvPages[L], B.lengthInfo, B.attnOut, attnU,
      ])
    }

    layerBGs.push({
      qkvFused: qkvFusedBG,
      kvQuantize: kvQuantizeBG,
      attn: attnBG,
      oproj: bg(device, PMatmul, [B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU]),
      addNorm1: bg(device, P.addNorm, [B.hidden2, B.residual, lw.normGamma2, B.hidden1, B.residual2, normU]),
      ffn: bg(device, PFfn, [B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU]),
      ffnDown: bg(device, PMatmul, [B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU]),
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

  function recordForward(enc: GPUCommandEncoder): void {
    // Embedding + initial norm
    dispatch(enc, P.embedding, embedBG, 12, 1, 1, 'embedding')
    dispatch(enc, P.rmsNorm, initNormBG, 1, 1, 1, 'rmsNorm_init')

    // 32 transformer layers. f16 KV: 7 dispatches/layer. int8 KV: 8/layer
    // (adds a kv_quantize pass between qkv_fused_scratch and attention_int8).
    for (let L = 0; L < PHI3.LAYERS; L++) {
      const lbg = layerBGs[L]
      if (int8Mode) {
        dispatch(enc, P.qkvFusedScratch, lbg.qkvFused, 4608, 1, 1, 'qkvFused')
        dispatch(enc, P.kvQuantizeInt8, lbg.kvQuantize!, 64, 1, 1, 'kvQuantize')
        dispatch(enc, P.attentionInt8, lbg.attn, 1, PHI3.HEADS, 1, 'attention')
      } else {
        dispatch(enc, PQkvFused, lbg.qkvFused, 4608 / qkvPairsPerWG, 1, 1, 'qkvFused')
        dispatch(enc, PAttn, lbg.attn, 1, PHI3.HEADS, 1, 'attention')
      }
      dispatch(enc, PMatmul, lbg.oproj, 3072 / matmulRowsPerWG, 1, 1, 'oproj')
      dispatch(enc, P.addNorm, lbg.addNorm1, 1, 1, 1, 'addNorm1')
      dispatch(enc, PFfn, lbg.ffn, PHI3.FFN / ffnRowsPerWG, 1, 1, 'fusedFfn')
      dispatch(enc, PMatmul, lbg.ffnDown, 3072 / matmulRowsPerWG, 1, 1, 'ffnDown')
      dispatch(enc, P.addNorm, lbg.addNorm2, 1, 1, 1, 'addNorm2')
    }

    // LM head + on-GPU argmax
    // PHI3.VOCAB=32064 is divisible by 4, so rowsPerWG=4 works exactly.
    dispatch(enc, PMatmulF32, lmHeadBG, PHI3.VOCAB / matmulRowsPerWG, 1, 1, 'lmHead')
    dispatch(enc, PArgmax, argmaxBG, 1, 1, 1, 'argmax')
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
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1

    // Per-token state. Queue-ordered with the submit below; the GPU reads these
    // during compute, and any later writeBuffer for the same buffer is serialized
    // after the current submit.
    if (writeInputId !== null) {
      device.queue.writeBuffer(B.inputIds, 0, new Int32Array([writeInputId]))
    }
    device.queue.writeBuffer(B.posMap, 0, new Int32Array([position]))
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))
    // nnz_pages lives at byte offset 8 in both f16 and int8 attention uniforms.
    device.queue.writeBuffer(int8Mode ? attnI8U : attnU, 8, new Int32Array([nnzPages]))

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

    const CAPACITY = 600  // 2 slots × (~228 f16 passes or ~260 int8 passes)
    const querySet = device.createQuerySet({ type: 'timestamp', count: CAPACITY })
    const resolveBuf = device.createBuffer({
      size: CAPACITY * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    })
    const readBuf = device.createBuffer({
      size: CAPACITY * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    })

    profile = { querySet, capacity: CAPACITY, labels: new Array(CAPACITY), nextSlot: 0 }
    try {
      const nnzPages = Math.floor(nextPos / PHI3.PAGE_SIZE) + 1
      device.queue.writeBuffer(B.posMap, 0, new Int32Array([nextPos]))
      device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
      device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([nextPos + 1, 0, 0]))
      device.queue.writeBuffer(int8Mode ? attnI8U : attnU, 8, new Int32Array([nnzPages]))

      const enc = device.createCommandEncoder()
      recordForward(enc)
      const usedSlots = profile.nextSlot
      enc.resolveQuerySet(querySet, 0, usedSlots, resolveBuf, 0)
      enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, usedSlots * 8)
      device.queue.submit([enc.finish()])

      await readBuf.mapAsync(GPUMapMode.READ)
      const raw = new BigInt64Array(readBuf.getMappedRange().slice(0))
      readBuf.unmap()

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
      readBuf.destroy()
    }
  }

  const STOP = new Set([2, 32000, 32007])

  async function generate(
    promptIds: number[],
    maxTokens: number,
    onToken: (id: number) => void
  ): Promise<number[]> {
    const tokens: number[] = []
    if (promptIds.length === 0 || maxTokens <= 0) return tokens

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

    for (let k = 0; k < PIPELINE_DEPTH && tokens.length + k < maxTokens; k++) {
      inFlight.push(submitStep(null, pos, true)!)
      pos++
    }

    while (inFlight.length > 0) {
      const tok = await inFlight.shift()!
      if (STOP.has(tok) || tok < 0 || tok >= PHI3.VOCAB) {
        // Drain the rest without emitting (frees readback slots for next generation).
        while (inFlight.length > 0) await inFlight.shift()!
        break
      }
      tokens.push(tok)
      onToken(tok)
      if (tokens.length >= maxTokens) {
        while (inFlight.length > 0) await inFlight.shift()!
        break
      }
      // Keep the pipeline full, but don't overshoot maxTokens.
      if (tokens.length + inFlight.length < maxTokens) {
        inFlight.push(submitStep(null, pos, true)!)
        pos++
      }
    }

    return tokens
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
    const tiledBG = bg(device, PMatmul,
      [tiledOutBuf, inBuf, scalesBuf, weightsBuf, uniformBufForTarget])

    const batchedWGs = N / 4
    const tiledWGs   = N / matmulRowsPerWG

    {
      const enc = device.createCommandEncoder()
      for (let i = 0; i < 20; i++) {
        const p1 = enc.beginComputePass()
        p1.setPipeline(batchedPipeline); p1.setBindGroup(0, batchedBG)
        p1.dispatchWorkgroups(batchedWGs); p1.end()
        const p2 = enc.beginComputePass()
        p2.setPipeline(PMatmul); p2.setBindGroup(0, tiledBG)
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
        pass.setPipeline(PMatmul); pass.setBindGroup(0, tiledBG)
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
    const readBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(batchedPipeline); pass.setBindGroup(0, batchedBG)
      pass.dispatchWorkgroups(batchedWGs); pass.end()
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, 64)
      device.queue.submit([enc.finish()])
    }
    await readBuf.mapAsync(GPUMapMode.READ)
    const sample = new Uint16Array(readBuf.getMappedRange().slice(0))
    readBuf.unmap()
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

    readBuf.destroy()
    inBuf.destroy(); outBuf.destroy(); tiledOutBuf.destroy()
    return { msBatched, msTiledTotal, msPerMBatched, msPerMTiled, speedup }
  }

  return { generate, profileStep, benchBatchedFfnDown }
}

// ============================================================
// UI helpers
// ============================================================

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T | null

function log(msg: string) {
  const el = document.getElementById('progress-log') as HTMLPreElement | null
  if (!el) { console.log(msg); return }
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

type BadgeState = 'idle' | 'loading' | 'ready' | 'error'
function setBadge(text: string, state: BadgeState = 'idle') {
  const badge = $('badge')
  const txt = $('badge-text')
  if (txt) txt.textContent = text
  else if (badge) badge.textContent = text
  if (badge) badge.className = `badge ${state}`
}

function setStats(text: string) {
  const el = $('stats')
  if (el) el.textContent = text
}

function setProgress(pct: number, status?: string, detail?: string) {
  const bar = $('progress-bar')
  if (bar) bar.style.width = `${Math.min(100, Math.max(0, pct))}%`
  if (status !== undefined) {
    const s = $('progress-status'); if (s) s.textContent = status
  }
  const d = $('progress-detail')
  if (d) d.textContent = detail ?? `${Math.round(pct)}%`
}

function hideLoadingOverlay() { $('loading-overlay')?.classList.remove('active') }
function showLoadingError(msg: string) {
  const el = $('loading-error')
  if (!el) return
  el.textContent = msg
  el.classList.add('visible')
}

function hideWelcome() {
  const w = $('welcome')
  if (w && !w.classList.contains('hidden')) w.classList.add('hidden')
}

function autoGrow(ta: HTMLTextAreaElement) {
  ta.style.height = 'auto'
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
}

function scrollToBottom(force = false) {
  const main = $('chat-main')
  if (!main) return
  const distance = main.scrollHeight - main.scrollTop - main.clientHeight
  if (force || distance < 120) main.scrollTop = main.scrollHeight
}

let scrollFabWired = false
function wireScrollFab() {
  if (scrollFabWired) return
  const main = $('chat-main'); const fab = $('scroll-fab')
  if (!main || !fab) return
  scrollFabWired = true
  const update = () => {
    const distance = main.scrollHeight - main.scrollTop - main.clientHeight
    fab.classList.toggle('visible', distance > 140)
  }
  main.addEventListener('scroll', update, { passive: true })
  fab.addEventListener('click', () => scrollToBottom(true))
  update()
}

const ICON_COPY    = '<svg class="icon" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>'
const ICON_CHECK   = '<svg class="icon" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>'
const ICON_REFRESH = '<svg class="icon" viewBox="0 0 24 24"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'

/** Wire a copy button that shows a 1.5s "Copied" confirmation on success. */
function wireCopyButton(btn: HTMLButtonElement, getText: () => string): void {
  btn.innerHTML = `${ICON_COPY}<span>Copy</span>`
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(getText())
      btn.classList.add('copied')
      btn.innerHTML = `${ICON_CHECK}<span>Copied</span>`
      setTimeout(() => {
        btn.classList.remove('copied')
        btn.innerHTML = `${ICON_COPY}<span>Copy</span>`
      }, 1500)
    } catch { /* no-op */ }
  })
}

// ---------- Minimal markdown renderer ----------
// Phi-3 emits headings, numbered/bulleted lists, fenced code blocks, inline
// code and bold. A hand-rolled parser keeps the bundle small; it runs on each
// streamed token via requestAnimationFrame (see AiMsgHandle.render).
function renderInline(text: string, into: Node): void {
  // Tokens: `code`, **bold**, *italic*, [text](url). Longest-match ordered.
  const re = /(```)|(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|\[([^\]]+)\]\(([^)\s]+)\)/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m[1]) { continue /* ``` inline is unusual; leave it */ }
    if (m.index > last) into.appendChild(document.createTextNode(text.slice(last, m.index)))
    if (m[2]) {
      const c = document.createElement('code'); c.textContent = m[2].slice(1, -1); into.appendChild(c)
    } else if (m[3]) {
      const b = document.createElement('strong'); b.textContent = m[3].slice(2, -2); into.appendChild(b)
    } else if (m[4]) {
      const em = document.createElement('em'); em.textContent = m[4].slice(1, -1); into.appendChild(em)
    } else if (m[5] && m[6]) {
      const a = document.createElement('a')
      a.href = m[6]; a.textContent = m[5]
      a.target = '_blank'; a.rel = 'noopener noreferrer'
      into.appendChild(a)
    }
    last = m.index + m[0].length
  }
  if (last < text.length) into.appendChild(document.createTextNode(text.slice(last)))
}

function makeCodeBlock(code: string, lang: string): HTMLElement {
  const wrap = document.createElement('div')
  wrap.className = 'code-block'
  const head = document.createElement('div')
  head.className = 'code-head'
  const langEl = document.createElement('span')
  langEl.className = 'code-lang'
  langEl.textContent = lang || 'code'
  head.appendChild(langEl)
  const btn = document.createElement('button')
  btn.type = 'button'
  btn.className = 'code-copy'
  wireCopyButton(btn, () => code)
  head.appendChild(btn)
  wrap.appendChild(head)
  const pre = document.createElement('pre')
  const codeEl = document.createElement('code')
  if (lang) codeEl.className = `lang-${lang}`
  codeEl.textContent = code
  pre.appendChild(codeEl)
  wrap.appendChild(pre)
  return wrap
}

function renderMarkdown(src: string): DocumentFragment {
  const frag = document.createDocumentFragment()
  const lines = src.split('\n')
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    // Skip pure-blank lines between blocks.
    if (line.trim() === '') { i++; continue }

    // Fenced code block
    const fence = /^\s*```(.*)$/.exec(line)
    if (fence) {
      const lang = fence[1].trim()
      i++
      const buf: string[] = []
      while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i++ }
      if (i < lines.length) i++ // consume closing fence
      frag.appendChild(makeCodeBlock(buf.join('\n'), lang))
      continue
    }

    // Horizontal rule
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      frag.appendChild(document.createElement('hr'))
      i++
      continue
    }

    // Heading
    const hm = /^(#{1,6})\s+(.*)$/.exec(line)
    if (hm) {
      const level = Math.min(6, hm[1].length)
      const h = document.createElement(`h${level}`) as HTMLElement
      renderInline(hm[2].trim(), h)
      frag.appendChild(h)
      i++
      continue
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const bq = document.createElement('blockquote')
      const buf: string[] = []
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      renderInline(buf.join(' '), bq)
      frag.appendChild(bq)
      continue
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = document.createElement('ul')
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const li = document.createElement('li')
        renderInline(lines[i].replace(/^\s*[-*]\s+/, ''), li)
        ul.appendChild(li)
        i++
      }
      frag.appendChild(ul)
      continue
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const ol = document.createElement('ol')
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const li = document.createElement('li')
        renderInline(lines[i].replace(/^\s*\d+\.\s+/, ''), li)
        ol.appendChild(li)
        i++
      }
      frag.appendChild(ol)
      continue
    }

    // Paragraph — absorb contiguous non-blank, non-block lines.
    const para: string[] = []
    while (i < lines.length && lines[i].trim() !== '' &&
           !/^\s*```/.test(lines[i]) &&
           !/^#{1,6}\s+/.test(lines[i]) &&
           !/^\s*>\s?/.test(lines[i]) &&
           !/^\s*[-*]\s+/.test(lines[i]) &&
           !/^\s*\d+\.\s+/.test(lines[i])) {
      para.push(lines[i])
      i++
    }
    if (para.length > 0) {
      const p = document.createElement('p')
      renderInline(para.join(' '), p)
      frag.appendChild(p)
    }
  }
  return frag
}

// ---------- Message construction ----------

function addUserMsg(text: string): void {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg user'
  const bubble = document.createElement('div')
  bubble.className = 'bubble'
  bubble.textContent = text
  wrap.appendChild(bubble)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)
}

interface AiMsgHandle {
  wrap: HTMLElement
  body: HTMLElement
  cursor: HTMLElement
  /** Replace the role tag with a thinking indicator (before first token). */
  showThinking(): void
  /** Re-render the full text (Markdown) into body on next animation frame. */
  render(text: string): void
  /** Remove cursor, add copy + stats + regenerate actions, stop streaming. */
  finish(opts: { fullText: string; tokens: number; tokPerS: number; onRegenerate?: () => void }): void
}

function addAiMsg(): AiMsgHandle {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg ai'
  wrap.innerHTML = `
    <div class="role">
      <span class="role-dot">Z</span>
      <span class="role-name">Phi-3-mini</span>
      <span class="model-tag">q4f16_1</span>
    </div>
    <div class="body"></div>
    <div class="actions"></div>
  `
  const body = wrap.querySelector('.body') as HTMLElement
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  body.appendChild(cursor)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)

  let rafPending = false
  let latestText = ''
  const doRender = () => {
    rafPending = false
    const frag = renderMarkdown(latestText)
    body.replaceChildren(frag, cursor)
    scrollToBottom(false)
  }

  return {
    wrap, body, cursor,
    showThinking() {
      // Replace the empty body (just cursor) with a thinking indicator.
      body.replaceChildren()
      const t = document.createElement('span')
      t.className = 'thinking'
      t.innerHTML = '<span></span><span></span><span></span>'
      body.appendChild(t)
    },
    render(text: string) {
      latestText = text
      if (!rafPending) {
        rafPending = true
        requestAnimationFrame(doRender)
      }
    },
    finish({ fullText, tokens, tokPerS, onRegenerate }) {
      // Ensure final markdown render without cursor.
      const frag = renderMarkdown(fullText)
      body.replaceChildren(frag)
      const actions = wrap.querySelector('.actions') as HTMLElement | null
      if (!actions) return
      actions.replaceChildren()

      const copy = document.createElement('button')
      copy.type = 'button'
      copy.className = 'action-btn'
      wireCopyButton(copy, () => fullText)
      actions.appendChild(copy)

      if (onRegenerate) {
        const regen = document.createElement('button')
        regen.type = 'button'
        regen.className = 'action-btn'
        regen.innerHTML = `${ICON_REFRESH}<span>Regenerate</span>`
        regen.addEventListener('click', onRegenerate)
        actions.appendChild(regen)
      }

      const stats = document.createElement('span')
      stats.className = 'msg-stats'
      stats.textContent = `${tokens} tok · ${tokPerS.toFixed(1)} tok/s`
      actions.appendChild(stats)
    },
  }
}

// ============================================================
// Boot helpers — SW registration, cache probe, download-gate UI
// (replaces the legacy <dialog>-based bootstrap; SW gives a single
// shared weight cache across /zero-tvm chat + compiler-chat + webllm-bench)
// ============================================================

async function registerWeightsSW(): Promise<void> {
  if (!('serviceWorker' in navigator)) return
  try {
    await navigator.serviceWorker.register('/weights-cache-sw.js', { scope: '/' })
    // Wait for the active SW so first weight fetch is intercepted (and
    // shared with WebLLM in compiler-chat.html / webllm-bench.html).
    await navigator.serviceWorker.ready
  } catch (e) {
    // Best-effort — chat still works, just no shared cache.
    console.warn('[zero-tvm] weight-cache SW registration failed:', e)
  }
}

async function hasWeightsCached(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return false
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('zero-tvm-weights')
    // ndarray-cache.json is the smallest sentinel — its presence means at
    // least the manifest has been fetched (and shards followed in-session).
    await dir.getFileHandle('ndarray-cache.json')
    return true
  } catch {
    return false
  }
}

function showDownloadGate(): Promise<void> {
  return new Promise((resolve) => {
    const chat = document.getElementById('chat')
    if (!chat) { resolve(); return }
    // Same #start-screen / #start-btn IDs as compiler-chat.html so existing
    // e2e harness (which clicks #start-btn) works on both pages uniformly.
    chat.innerHTML = `
      <div id="start-screen" class="start-screen" style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:1.25rem;text-align:center;padding:2rem">
        <div class="title" style="font-size:1.1rem;font-weight:600;color:#ccc">Phi-3-mini, in your browser</div>
        <div class="desc" style="font-size:0.85rem;color:#888;max-width:440px;line-height:1.6">
          First load downloads ~1.8 GB of Phi-3-mini-q4f16_1 weights and caches them
          locally (OPFS). Every subsequent visit — including the WebLLM comparison
          page — is instant; weights are served from the same shared browser cache.
        </div>
        <button id="start-btn" class="start-btn" style="background:#10b981;color:#000;font-weight:600;font-size:0.95rem;padding:0.7rem 2rem;border:none;border-radius:8px;cursor:pointer">Download &amp; Start</button>
        <div class="start-hint" style="font-size:0.68rem;color:#444">Phi-3-mini-4k-instruct-q4f16_1 · ~1.8 GB · cached after first load</div>
      </div>`
    setBadge('Waiting')
    const btn = document.getElementById('start-btn') as HTMLButtonElement | null
    btn?.addEventListener('click', () => {
      btn.disabled = true
      btn.textContent = 'Starting...'
      chat.innerHTML = ''
      resolve()
    }, { once: true })
  })
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  setBadge('Initializing…', 'loading')
  setProgress(2, 'Requesting GPU access…')
  log('Zero-TVM Phi-3 — No WebLLM, No TVM')
  log('10 hand-written WGSL kernel roles across 27 files')
  log('')

  if (!navigator.gpu) {
    setBadge('No WebGPU', 'error')
    showLoadingError('WebGPU is not available in this browser. Try Chrome 113+ or Edge 113+.')
    log('ERROR: WebGPU not supported')
    return
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) {
    setBadge('No GPU adapter', 'error')
    showLoadingError('No GPU adapter found.')
    log('ERROR: No GPU adapter')
    return
  }

  const wantFeatures: GPUFeatureName[] = ['shader-f16' as GPUFeatureName]
  const hasSubgroupsFeature = adapter.features.has('subgroups' as GPUFeatureName)
  if (hasSubgroupsFeature) wantFeatures.push('subgroups' as GPUFeatureName)
  if (adapter.features.has('timestamp-query' as GPUFeatureName)) {
    wantFeatures.push('timestamp-query' as GPUFeatureName)
  }
  let device: GPUDevice
  try {
    device = await adapter.requestDevice({ requiredFeatures: wantFeatures })
  } catch (e) {
    setBadge('shader-f16 missing', 'error')
    showLoadingError(`GPU request failed: ${e}. shader-f16 is required — Chrome/Edge 113+.`)
    log(`ERROR: ${e}`)
    return
  }
  log(`GPU ready — features: ${wantFeatures.join(', ')}`)
  setProgress(5, 'Probing GPU subgroups…')
  // Our _sg shaders assume subgroup size == 32 (full 32-thread workgroup in
  // one subgroup). Apple M-series and most shipping WebGPU subgroup backends
  // report 32; the JS limit is unreliable across Chromium versions so we
  // probe via a one-shot compute dispatch.
  let sgSizeOk = false
  if (hasSubgroupsFeature) {
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
      sgSizeOk = sg === 32
      log(`subgroups feature: yes · probed size: ${sg} · path: ${sgSizeOk ? '_sg' : 'scalar'}`)
      probeBuf.destroy(); probeRb.destroy()
    } catch (e) {
      log(`subgroups probe failed: ${e} · falling back to scalar`)
    }
  } else {
    log('subgroups feature: no · path: scalar')
  }

  // Tokenizer
  setBadge('Loading tokenizer…', 'loading')
  setProgress(8, 'Loading tokenizer…')
  let tokenizer: Tokenizer
  try {
    tokenizer = await loadTokenizer((msg) => log(msg))
  } catch (e) {
    setBadge('Tokenizer failed', 'error')
    showLoadingError(`Tokenizer load failed: ${e}`)
    log(`ERROR: ${e}`)
    return
  }

  // Weights
  setBadge('Downloading…', 'loading')
  setProgress(10, 'Loading model weights…')
  log('')
  log('Loading weights (checks browser cache first)…')
  let weights: LoadedWeights
  try {
    weights = await loadWeights(device, (msg) => {
      log(msg)
      const m = /Loading layer (\d+)/.exec(msg)
      if (m) {
        const layer = parseInt(m[1], 10)
        const pct = (layer / PHI3.LAYERS) * 100
        setProgress(10 + pct * 0.82, `Loading layer ${layer} / ${PHI3.LAYERS}`, `${Math.round(pct)}%`)
        setBadge(`${Math.round(pct)}%`, 'loading')
      }
    })
  } catch (e) {
    setBadge('Download failed', 'error')
    showLoadingError(`Weight load failed: ${e}`)
    log(`ERROR loading weights: ${e}`)
    return
  }

  // KV cache: int8 halves memory at the cost of one extra dispatch per layer.
  setProgress(94, 'Allocating KV cache…')
  log(USE_INT8_KV ? 'Allocating int8 KV cache (~800 MB)…' : 'Allocating KV cache (~1.6 GB)…')
  const kvPages = USE_INT8_KV ? [] : allocKVPages(device)
  const kvInt8 = USE_INT8_KV ? allocKVPagesInt8(device) : null

  // Build engine
  setProgress(98, 'Compiling shaders…')
  log('Building decode engine…')
  const engine = buildDecodeEngine(device, weights, kvPages, kvInt8, { sgSizeOk })

  setProgress(100, 'Ready')
  log('')
  log('Ready. Zero TVM. 10 WGSL kernel roles.')
  setBadge('Ready', 'ready')
  hideLoadingOverlay()

  // ─────────── Wire chat UI ───────────
  const form     = $('composer') as HTMLFormElement | null
  const inp      = $('inp') as HTMLTextAreaElement | null
  const sendBtn  = $('btn') as HTMLButtonElement | null
  const stopBtn  = $('stop-btn') as HTMLButtonElement | null
  const newBtn   = $('new-chat-btn') as HTMLButtonElement | null
  const suggest  = $('suggest-grid')
  const ctxHint  = $('ctx-hint')
  const welcome  = $('welcome')
  if (!form || !inp || !sendBtn || !stopBtn) {
    console.error('[zero-tvm] chat UI elements missing'); return
  }

  const SYSTEM_PROMPT = 'You are a helpful, concise assistant. Use Markdown (numbered lists, **bold**, and fenced ```code``` blocks with a language tag) when it clarifies the answer.'
  const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: SYSTEM_PROMPT },
  ]
  let generating = false
  let stopRequested = false

  function updateSendEnabled() {
    if (!inp || !sendBtn) return
    sendBtn.disabled = generating || inp.value.trim().length === 0
  }
  function setBusy(busy: boolean) {
    if (!inp || !sendBtn || !stopBtn || !newBtn) return
    generating = busy
    sendBtn.hidden = busy
    stopBtn.hidden = !busy
    stopBtn.disabled = !busy
    inp.disabled = busy
    newBtn.disabled = busy
    inp.placeholder = busy ? 'Generating…' : 'Ask Phi-3 anything…'
  }

  function updateCtxHint(nTokens?: number) {
    if (!ctxHint) return
    if (!nTokens) {
      ctxHint.textContent = 'Zero TVM · 10 WGSL kernels · 228 dispatches/token'
    } else {
      const pct = Math.round((nTokens / PHI3.MAX_SEQ) * 100)
      ctxHint.textContent = `Context ${nTokens} / ${PHI3.MAX_SEQ} tokens (${pct}%) · 228 dispatches/token`
    }
  }

  function resetChat() {
    if (generating) return
    history.length = 0
    history.push({ role: 'system', content: SYSTEM_PROMPT })
    const msgs = $('messages')
    if (msgs) msgs.replaceChildren()
    welcome?.classList.remove('hidden')
    setStats('')
    updateCtxHint()
    inp?.focus()
  }

  // Initial enabled state
  inp.disabled = false
  inp.placeholder = 'Ask Phi-3 anything…'
  newBtn && (newBtn.disabled = false)
  autoGrow(inp)
  updateSendEnabled()
  updateCtxHint()
  wireScrollFab()
  inp.focus()

  inp.addEventListener('input', () => { autoGrow(inp); updateSendEnabled() })
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!sendBtn.disabled) form.requestSubmit()
    }
  })
  form.addEventListener('submit', (e) => {
    e.preventDefault()
    if (!sendBtn.disabled) void send()
  })
  stopBtn.addEventListener('click', () => {
    stopRequested = true
    stopBtn.disabled = true
  })
  newBtn?.addEventListener('click', resetChat)
  suggest?.addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest('.suggest') as HTMLElement | null
    if (!target) return
    const prompt = target.dataset.prompt
    if (!prompt) return
    inp.value = prompt
    autoGrow(inp); updateSendEnabled()
    form.requestSubmit()
  })

  async function runGeneration(ai: AiMsgHandle, promptIds: number[]): Promise<string> {
    log(`Prompt: ${promptIds.length} tokens`)
    updateCtxHint(promptIds.length)
    ai.showThinking()
    const t0 = performance.now()
    let count = 0
    let firstToken = true
    const allIds: number[] = []
    let fullResponse = ''

    try {
      await engine.generate(promptIds, 500, (id) => {
        if (stopRequested) throw new Error('__stopped__')
        if (firstToken) {
          // Swap thinking indicator for the real body on first token.
          ai.body.replaceChildren(ai.cursor)
          firstToken = false
        }
        count++
        allIds.push(id)
        fullResponse = tokenizer.decode(allIds)
        ai.render(fullResponse)
        const elapsed = (performance.now() - t0) / 1000
        setStats(`${count} tok · ${(count / elapsed).toFixed(1)} tok/s`)
      })
    } catch (e) {
      if (e instanceof Error && e.message === '__stopped__') {
        fullResponse = tokenizer.decode(allIds)
      } else {
        fullResponse += `\n\n_[Error: ${e}]_`
        log(`Error: ${e}`)
      }
    }
    const totalSec = (performance.now() - t0) / 1000
    const tokPerS = count / Math.max(totalSec, 0.001)
    ai.finish({
      fullText: fullResponse,
      tokens: count,
      tokPerS,
      onRegenerate: () => { void regenerate() },
    })
    // Reflect the true KV position count (prompt + generated) in the context hint.
    updateCtxHint(promptIds.length + count)
    return fullResponse
  }

  /** Shared bracketing around a single decode turn: busy flags, prefill
   * tokenization, run, commit response to history, re-enable UI. */
  async function runTurn(): Promise<void> {
    setBusy(true); stopRequested = false
    const ai = addAiMsg()
    const promptIds = buildChatPrompt(history, tokenizer)
    const fullResponse = await runGeneration(ai, promptIds)
    history.push({ role: 'assistant', content: fullResponse })
    setBusy(false); stopRequested = false
    updateSendEnabled()
    inp?.focus()
  }

  async function send(): Promise<void> {
    if (generating || !inp) return
    const text = inp.value.trim()
    if (!text) return
    inp.value = ''
    autoGrow(inp)
    addUserMsg(text)
    history.push({ role: 'user', content: text })
    await runTurn()
  }

  async function regenerate(): Promise<void> {
    if (generating) return
    if (history.length === 0 || history[history.length - 1].role !== 'assistant') return
    history.pop()
    $('messages')?.querySelector('.msg.ai:last-child')?.remove()
    await runTurn()
  }

  // Benchmark harness: `await window.bench(nTokens=128, nRuns=3)`.
  // Uses a fixed one-turn prompt and runs `nRuns + 1` decodes; the first is
  // discarded as warmup. Reports per-run and aggregate tok/s (min/median/mean/max).
  // Run from devtools / preview_eval; does not touch the chat UI.
  ;(window as Window & typeof globalThis & { bench?: unknown }).bench =
    async (nTokens: number = 128, nRuns: number = 3, profile: boolean = false): Promise<unknown> => {
      if (generating) { console.warn('[bench] decode already in flight — skipping'); return null }
      setBusy(true)
      try {
        const benchPrompt = 'Write a four-sentence explanation of how photosynthesis works.'
        const benchHist: Array<{ role: 'system' | 'user'; content: string }> = [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: benchPrompt },
        ]
        const promptIds = buildChatPrompt(benchHist, tokenizer)

        if (profile) {
          const prof = await engine.profileStep(promptIds)
          if (!prof) {
            console.warn('[bench] profile requested but timestamp-query feature unavailable')
            return null
          }
          console.log(`[profile] total = ${prof.totalMs.toFixed(3)} ms/token across ${prof.kernels.reduce((s, k) => s + k.calls, 0)} dispatches`)
          console.table(prof.kernels.map(k => ({
            kernel: k.label,
            calls: k.calls,
            'ms/token': +k.totalMs.toFixed(3),
            'ms/call': +(k.totalMs / k.calls).toFixed(4),
            '% total': +k.pctOfTotal.toFixed(1),
          })))
          return prof
        }

        const runs: { tokens: number; seconds: number; tokPerS: number }[] = []
        for (let r = 0; r <= nRuns; r++) {
          const label = r === 0 ? 'warmup' : `run${r}/${nRuns}`
          const t0 = performance.now()
          let count = 0
          await engine.generate(promptIds, nTokens, () => { count++ })
          const seconds = (performance.now() - t0) / 1000
          const tokPerS = count / seconds
          console.log(`[bench] ${label}: ${count} tok / ${seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s`)
          if (r > 0) runs.push({ tokens: count, seconds, tokPerS })
        }
        const sorted = runs.map(r => r.tokPerS).sort((a, b) => a - b)
        const median = sorted[Math.floor(sorted.length / 2)]
        const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
        const min = sorted[0]
        const max = sorted[sorted.length - 1]
        const summary = { runs, median, mean, min, max }
        console.log(`[bench] summary: median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s`)
        return summary
      } finally {
        setBusy(false)
        updateSendEnabled()
      }
    }
  ;(window as Window & typeof globalThis & { benchBatched?: unknown }).benchBatched =
    async (
      M: number = 4,
      iters: number = 500,
      target: 'ffnDown' | 'oproj' = 'ffnDown',
    ): Promise<unknown> => {
      if (generating) { console.warn('[batched-bench] decode in flight — skipping'); return null }
      return engine.benchBatchedFfnDown(M, iters, target)
    }
  ;(window as Window & typeof globalThis & { specSim?: unknown }).specSim =
    async (nTokens: number = 160, N: number = 3, K: number = 3): Promise<unknown> => {
      if (generating) { console.warn('[specSim] decode in flight — skipping'); return null }
      setBusy(true)
      try {
        const prompts: Array<{ name: string; messages: Array<{ role: 'system' | 'user'; content: string }> }> = [
          {
            name: 'prose',
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: 'Write a four-sentence explanation of how photosynthesis works.' },
            ],
          },
          {
            name: 'code',
            messages: [
              { role: 'system', content: 'You are a helpful coding assistant. Reply with only code, no prose.' },
              { role: 'user', content: 'Write a TypeScript function that computes the nth Fibonacci number using memoization. Include a short test block that prints fib(10), fib(20), and fib(30).' },
            ],
          },
          {
            name: 'summary',
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: 'List the top five considerations when designing a REST API, with a one-sentence explanation for each.' },
            ],
          },
        ]

        const results = []
        for (const p of prompts) {
          const promptIds = buildChatPrompt(p.messages, tokenizer)
          const generated: number[] = []
          await engine.generate(promptIds, nTokens, (id) => { generated.push(id) })
          // Concatenate prompt + generated; PLD keys search over the full
          // sequence (prompt is the richest lookup source).
          const full = [...promptIds, ...generated]
          const sim = summarizePLD(p.name, full, promptIds.length, N, K)
          results.push({ ...sim, generatedTokens: generated.length })
          console.log(
            `[specSim] ${p.name}: ${generated.length} gen tok, ` +
            `hit=${(sim.hitRate * 100).toFixed(0)}%, ` +
            `α=${(sim.acceptanceRate * 100).toFixed(0)}%, ` +
            `accepted/step=${sim.meanAcceptedPerStep.toFixed(2)}/${K}, ` +
            `theoretical speedup=${sim.theoreticalSpeedup.toFixed(2)}×`
          )
        }
        console.table(results.map(r => ({
          prompt: r.name,
          gen: r.generatedTokens,
          'hit %': +(r.hitRate * 100).toFixed(1),
          'accept α': +(r.acceptanceRate * 100).toFixed(1),
          'accepted/step': +r.meanAcceptedPerStep.toFixed(2),
          speedup: +r.theoreticalSpeedup.toFixed(2),
        })))
        return results
      } finally {
        setBusy(false)
        updateSendEnabled()
      }
    }
  console.log('[bench] harness ready — call `await bench(128, 3)` or `await bench(0, 0, true)` for per-kernel profile')
  console.log('[bench] batched primitive: `await benchBatched(4, 500)` — ffnDown weight-reuse falsifiability test')
  console.log('[bench] PLD acceptance sim: `await specSim(160, 3, 3)` — measures spec-decode upside without GPU changes')
}
async function boot(): Promise<void> {
  // Register the weight-cache SW first so it intercepts the very first
  // network fetch (from main()'s loadWeights call) — and any future
  // fetch from WebLLM in compiler-chat / webllm-bench.
  await registerWeightsSW()
  // Skip the gate when weights are already in OPFS — returning visitors
  // (or visitors who started from compiler-chat) should boot straight in.
  if (!(await hasWeightsCached())) await showDownloadGate()
  await main()
}
boot().catch((e) => { console.error(e); log(`FATAL: ${e}`) })
