/**
 * ZERO-TVM ENGINE CORE
 *
 * Pure GPU pipeline: takes a loaded device + weights, returns a DecodeEngine
 * that exposes generate / forwardLogits / resetKVTracking. No DOM, no UI.
 *
 * validate.ts wires this into the validation page (via loading-ui.ts's
 * bootEngine). chat.ts does NOT use this module yet — it carries its own
 * forked copy of the decode loop (see CLAUDE.md "Known gaps"); unification is
 * planned. Until then, keep inner-loop edits in sync across both files.
 */

import { LoadedWeights } from './weight-loader.js'
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

function dispatch(
  enc: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  wgX: number, wgY = 1, wgZ = 1
): void {
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(wgX, wgY, wgZ)
  pass.end()
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
//   matmul shaders: one wg per output element (M)
const WG_SIZE_D = 256
const D_WGS     = PHI3.D / WG_SIZE_D            // 12   (3072/256)
const QKV_WGS   = PHI3.QKV_DIM / WG_SIZE_D      // 36   (9216/256)

export interface DecodeEngine {
  generate(
    promptIds: number[],
    startPos: number,
    maxTokens: number,
    onToken: (id: number) => void
  ): Promise<number[]>
  /** Run a forward pass through prefill of `promptIds` and return f32 logits at the final position. */
  forwardLogits(promptIds: number[]): Promise<Float32Array>
  /** Reset KV-cache invalidation tracking (call when starting a fresh conversation). */
  resetKVTracking(): void
}

export function buildDecodeEngine(
  device: GPUDevice,
  weights: LoadedWeights,
  kvPages: GPUBuffer[]
): DecodeEngine {
  const { pipelines } = compile(device)
  const P = pipelines

  // Activation buffers
  const B = {
    residual:  makeBuf(device, PHI3.D * 2, 'residual'),      // running residual (ping)
    residual2: makeBuf(device, PHI3.D * 2, 'residual2'),     // running residual (pong)
    hidden1:   makeBuf(device, PHI3.D * 2, 'hidden1'),       // normed scratch
    hidden2:   makeBuf(device, PHI3.D * 2, 'hidden2'),       // matmul output scratch
    qkvOut:    makeBuf(device, PHI3.QKV_DIM * 2, 'qkvOut'),
    qOut:      makeBuf(device, PHI3.D * 2, 'qOut'),
    kOut:      makeBuf(device, PHI3.D * 2, 'kOut'),
    vOut:      makeBuf(device, PHI3.D * 2, 'vOut'),
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

  // Static uniforms — matmul shapes (K_packed, scales_per_row, M)
  const qkvU   = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(PHI3.QKV_DIM)])
  const oProjU = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(PHI3.D)])
  const ffnDnU = uniformBuf(device, [u32(FFN_DN_K_PACKED), u32(FFN_DN_SCALES), u32(PHI3.D)])
  const lmHdU  = uniformBuf(device, [u32(QKV_K_PACKED),    u32(QKV_SCALES),    u32(PHI3.VOCAB)])
  const embU   = uniformBuf(device, [u32(1), u32(D_WGS)])
  const normU  = uniformBuf(device, [u32(1)])
  const ffnU   = uniformBuf(device, [u32(PHI3.FFN)])
  const argmaxU = uniformBuf(device, [u32(PHI3.VOCAB)])

  // Hoisted per-layer uniforms. Previously these were allocated inside the
  // per-layer hot loop on every token, leaking ~96 uniform buffers per token.
  // rope, kv_append: all fields are constant across tokens.
  const ropeU  = uniformBuf(device, [i32(1), i32(0), i32(1), u32(QKV_WGS)])
  const kvAppU = uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(D_WGS)])

  // attention: same layout as before, but allocated once. Only field that
  // changes per token is nnz_pages at byte offset 8 — we writeBuffer it.
  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)
  const attnU = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: 'attnU',
  })
  {
    const init = new ArrayBuffer(48)
    const dv = new DataView(init)
    dv.setInt32(0, 1, true)                    // batch
    dv.setInt32(4, PHI3.MAX_PAGES, true)       // max_pages
    // offset 8: nnz_pages — updated per token via writeBuffer
    dv.setInt32(12, 0, true)
    dv.setInt32(16, 0, true)
    dv.setInt32(20, 0, true)
    dv.setInt32(24, 0, true)
    dv.setFloat32(28, SM_SCALE, true)          // sm_scale
    dv.setUint32(32, 1, true)
    device.queue.writeBuffer(attnU, 0, init)
  }
  const nnzPagesScratch = new Uint32Array(1)   // reused per token for writeBuffer

  // Token readback buffer — allocated once, reused every decode step.
  // Previously a new 4-byte GPUBuffer was created and destroyed per token.
  const readBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'tokenReadback',
  })

  // Logit readback buffer — used by forwardLogits() for the validation harness only.
  // Allocated lazily on first call.
  let logitsReadBuf: GPUBuffer | null = null

  // Initialize identity page table (page i → physical page i)
  const pageVals = new Int32Array(PHI3.MAX_PAGES)
  for (let i = 0; i < PHI3.MAX_PAGES; i++) pageVals[i] = i
  device.queue.writeBuffer(B.pageValues, 0, pageVals)

  const MAX_CONTEXT = PHI3.MAX_PAGES * PHI3.PAGE_SIZE

  // ============================================================
  // Pre-computed bind groups
  //
  // Previously each per-token decode rebuilt ~10 bind groups per layer × 32 layers
  // = ~320 createBindGroup calls per token. Bind group contents are deterministic
  // (same buffers, same layout) so we hoist them out of the hot loop.
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
  const bgRope = bg(device, P.rope, [
    B.qOut, B.kOut, B.vOut, B.qkvOut, B.posMap, ropeU,
  ])
  const bgLmHead = bg(device, P.lmHead, [
    B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU,
  ])
  const bgArgmax = bg(device, P.argmax, [
    B.logits, B.tokenOut, argmaxU,
  ])

  // Per-layer bind groups
  interface LayerBG {
    qkv: GPUBindGroup
    kvApp: GPUBindGroup
    attn: GPUBindGroup
    oProj: GPUBindGroup
    addNorm1: GPUBindGroup  // post-attention: reads residual, writes residual2
    fusedFfn: GPUBindGroup
    ffnDown: GPUBindGroup
    addNorm2: GPUBindGroup  // post-FFN: reads residual2, writes residual
  }
  const layerBGs: LayerBG[] = []
  for (let L = 0; L < PHI3.LAYERS; L++) {
    const lw = weights.layers[L]
    const nextGamma = L < PHI3.LAYERS - 1
      ? weights.layers[L + 1].normGamma1
      : weights.finalNormGamma
    layerBGs.push({
      qkv: bg(device, P.qkvMatmul, [
        B.qkvOut, B.hidden1, lw.qkvScales, lw.qkvWeights, qkvU,
      ]),
      kvApp: bg(device, P.kvAppend, [
        B.kOut, B.vOut, kvPages[L], B.posMap, kvAppU,
      ]),
      attn: bg(device, P.attention, [
        B.qOut, B.pageIndptr, B.pageValues, kvPages[L],
        B.lengthInfo, B.attnOut, attnU,
      ]),
      oProj: bg(device, P.oProjMatmul, [
        B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU,
      ]),
      addNorm1: bg(device, P.addNorm, [
        B.hidden2, B.residual, lw.normGamma2, B.hidden1, B.residual2, normU,
      ]),
      fusedFfn: bg(device, P.fusedFfn, [
        B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU,
      ]),
      ffnDown: bg(device, P.ffnDownMatmul, [
        B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU,
      ]),
      addNorm2: bg(device, P.addNorm, [
        B.hidden2, B.residual2, nextGamma, B.hidden1, B.residual, normU,
      ]),
    })
  }

  async function decodeToken(tokenId: number, position: number): Promise<number> {
    if (position < 0 || position >= MAX_CONTEXT) {
      throw new Error(
        `zero-tvm: context overflow — position ${position} exceeds max context ` +
        `${MAX_CONTEXT} tokens (PHI3.MAX_PAGES=${PHI3.MAX_PAGES} × PAGE_SIZE=${PHI3.PAGE_SIZE}). ` +
        `Shorten the prompt or raise MAX_PAGES in src/compiler/compiler.ts (costs ~${Math.round(PHI3.LAYERS * 196608 / (1024 * 1024))} MB of KV cache per page block).`
      )
    }
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1

    // --- Write per-token state ---
    device.queue.writeBuffer(B.inputIds, 0, new Int32Array([tokenId]))
    device.queue.writeBuffer(B.posMap, 0, new Int32Array([position]))
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    // length_info: total tokens in sequence = position + 1
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))
    // attnU.nnz_pages lives at byte offset 8 — the only field that varies per token
    nnzPagesScratch[0] = nnzPages
    device.queue.writeBuffer(attnU, 8, nnzPagesScratch)

    const enc = device.createCommandEncoder()

    // --- EMBEDDING → B.residual (ping) ---
    dispatch(enc, P.embedding, bgEmbedding, D_WGS)

    // --- INITIAL RMSNORM: B.residual → B.hidden1 (layer 0's normGamma1) ---
    dispatch(enc, P.rmsNorm, bgInitNorm, 1)

    // --- 32 TRANSFORMER LAYERS ---
    // Residual ping-pong is encoded into the cached bind groups: addNorm1 reads
    // residual / writes residual2; addNorm2 reads residual2 / writes residual.
    // Two swaps per layer return to the same state, so layer L+1's addNorm1
    // sees the residual updated by layer L's addNorm2.
    for (let L = 0; L < PHI3.LAYERS; L++) {
      const blk = layerBGs[L]

      // [0] QKV matmul: B.hidden1 → B.qkvOut
      dispatch(enc, P.qkvMatmul, blk.qkv, PHI3.QKV_DIM)

      // [1] RoPE: B.qkvOut → B.qOut, B.kOut, B.vOut
      dispatch(enc, P.rope, bgRope, QKV_WGS)

      // [2] KV append: kOut, vOut → kvPages[L]
      dispatch(enc, P.kvAppend, blk.kvApp, D_WGS)

      // [3] Attention: Q + kvPages[L] → B.attnOut
      dispatch(enc, P.attention, blk.attn, 1, PHI3.HEADS)

      // [4] O projection: B.attnOut → B.hidden2
      dispatch(enc, P.oProjMatmul, blk.oProj, PHI3.D)

      // [5] AddNorm (attention): residual += hidden2; hidden1 = RMSNorm(residual)
      dispatch(enc, P.addNorm, blk.addNorm1, 1)

      // [6] Fused FFN gate+up+SiLU: B.hidden1 → B.ffnOut
      dispatch(enc, P.fusedFfn, blk.fusedFfn, PHI3.FFN)

      // [7] FFN down: B.ffnOut → B.hidden2
      dispatch(enc, P.ffnDownMatmul, blk.ffnDown, PHI3.D)

      // [8] AddNorm (FFN): residual += hidden2; hidden1 = RMSNorm(residual)
      //   For last layer the bind group binds finalNormGamma instead of next
      //   layer's normGamma1, so hidden1 is ready for the LM head.
      dispatch(enc, P.addNorm, blk.addNorm2, 1)
    }

    // --- LM HEAD: B.hidden1 (already normalized with model.norm) → B.logits ---
    dispatch(enc, P.lmHead, bgLmHead, PHI3.VOCAB)

    // --- ARGMAX: B.logits → B.tokenOut ---
    dispatch(enc, P.argmax, bgArgmax, 1)

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

  return { generate, forwardLogits, resetKVTracking }
}
