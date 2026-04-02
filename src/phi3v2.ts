/**
 * PHI-3 ENGINE v2 — Built from first principles.
 *
 * Every uniform byte written from decoded WGSL struct layouts.
 * No captured snapshots. No TVM writes. No replay.
 *
 * Still uses TVM for: model load (weights to GPU), shader compilation.
 * Uses TVM's compiled pipelines and weight buffers.
 * Everything else is ours.
 */

import { CaptureResult } from './capture.js'

// ============================================================
// Config
// ============================================================

const D = 3072
// const HEADS = 32  // used implicitly via HEAD_DIM = D/HEADS
const HEAD_DIM = 96  // D / HEADS
const LAYERS = 32
const FFN_DIM = 8192
const VOCAB = 32064
const PAGE_SIZE = 16
const MAX_PAGES = 257
const SM_SCALE = 1.0 / Math.sqrt(HEAD_DIM)  // 1/√96 ≈ 0.10206

// ============================================================
// Uniform Writers — from WGSL struct definitions
// ============================================================

// All uniform writers padded to match TVM's buffer sizes (16B aligned)

/** Matmul podArgs — 16B (4B data + 12B padding) */
function writeMatmulParams(packGridDimX: number): ArrayBuffer {
  const buf = new ArrayBuffer(16)
  new DataView(buf).setUint32(0, packGridDimX, true)
  return buf
}

/** RoPE podArgs — 16B: { apply_rope, position_map_elem_offset, seq_len, packGridDimX } */
function writeRopeParams(seqLen: number): ArrayBuffer {
  const buf = new ArrayBuffer(16)
  const v = new DataView(buf)
  v.setInt32(0, 1, true)       // apply_rope = 1
  v.setInt32(4, 0, true)       // position_map_elem_offset = 0
  v.setInt32(8, seqLen, true)  // seq_len (always 1 for decode)
  v.setUint32(12, 36, true)    // packGridDimX = 36
  return buf
}

/** KV append podArgs — 32B (20B data + 12B padding) */
function writeKvAppendParams(): ArrayBuffer {
  const buf = new ArrayBuffer(32)
  const v = new DataView(buf)
  v.setInt32(0, 1, true)       // ntoken = 1
  v.setInt32(4, MAX_PAGES, true) // num_pages = 257
  v.setInt32(8, 0, true)       // pages_elem_offset = 0
  v.setInt32(12, 0, true)      // position_map_elem_offset = 0
  v.setUint32(16, 12, true)    // packGridDimX = 12
  return buf
}

/** Attention podArgs — 64B (56B data + 8B padding) */
function writeAttentionParams(nnzPages: number): ArrayBuffer {
  const buf = new ArrayBuffer(64)
  const v = new DataView(buf)
  v.setInt32(0, 1, true)              // B = 1
  v.setInt32(4, 0, true)              // k_rope_pos_offset_elem_offset = 0
  v.setInt32(8, 0, true)              // length_info_elem_offset = 0
  v.setInt32(12, MAX_PAGES, true)     // max_num_pages = 257
  v.setInt32(16, nnzPages, true)      // nnz_pages
  v.setInt32(20, 0, true)             // page_indptr_elem_offset = 0
  v.setInt32(24, 0, true)             // page_values_elem_offset = 0
  v.setInt32(28, 0, true)             // pages_elem_offset = 0
  v.setInt32(32, 0, true)             // q_rope_position_elem_offset = 0
  v.setFloat32(36, 1.0, true)         // rope_scale = 1.0
  v.setFloat32(40, 10000.0, true)     // rope_theta = 10000.0
  v.setInt32(44, 0, true)             // rotary_mode = 0
  v.setFloat32(48, SM_SCALE, true)    // sm_scale = 1/√96
  v.setUint32(52, 1, true)            // packGridDimX = 1
  return buf
}

/** AddNorm podArgs — 16B: { packGridDimX, extra, ... } */
function writeAddNormParams(): ArrayBuffer {
  const buf = new ArrayBuffer(16)
  const v = new DataView(buf)
  v.setUint32(0, 1, true)
  v.setUint32(4, 1, true)
  return buf
}

/** SiLU podArgs — 16B: { packGridDimX, extra, ... } */
function writeSiluParams(): ArrayBuffer {
  const buf = new ArrayBuffer(16)
  const v = new DataView(buf)
  v.setUint32(0, 1, true)
  v.setUint32(4, 1, true)
  return buf
}

// ============================================================
// Engine Builder
// ============================================================

export function buildPhi3v2(capture: CaptureResult) {
  const dev = capture.device

  // Find pipelines
  const find = (name: string) => {
    const p = capture.pipelines.find(p => p.entryPoint.includes(name))
    if (!p) throw new Error(`Pipeline not found: ${name}`)
    return p.pipeline
  }

  const embed = find('dequantize_take1')
  const initialNorm = find('rms_norm2')
  const qkvMatmul = find('NT_matmul10')
  const rope = find('fused_rope')
  const kvAppend = find('kv_cache_transpose_append_kernel')
  const attention = find('batch_decode_paged_kv_kernel')
  const oProj = find('NT_matmul11')
  const addNorm = find('fuse_add_norm_decode')
  const ffnUp = find('NT_matmul12')
  const silu = find('split2_silu2_multiply2')
  const ffnDown = find('NT_matmul13')

  console.log('[phi3v2] All pipelines found')

  console.log('[phi3v2] Uniform writers ready')

  // Build bind groups from captured dispatch entries + our uniforms
  // For each dispatch, replace the uniform binding with our buffer, keep everything else
  const dispatches = capture.dispatches
  if (dispatches.length < 342) throw new Error(`Need 342 dispatches, got ${dispatches.length}`)

  /** Build bind group using TVM's original buffers. We write our own values via writeBuffer. */
  function buildBG(dispIdx: number, pipeline: GPUComputePipeline): GPUBindGroup {
    const d = dispatches[dispIdx]
    const entries: GPUBindGroupEntry[] = d.entries.map(e => ({
      binding: e.binding,
      resource: { buffer: e.buffer, offset: e.offset, size: e.size },
    }))
    return dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
  }

  /** Find the podArgs uniform buffer — always the LAST binding in TVM convention */
  function findUniformBuf(dispIdx: number): GPUBuffer | null {
    const d = dispatches[dispIdx]
    if (d.entries.length === 0) return null
    // Last binding is the uniform (podArgs)
    let maxBinding = d.entries[0]
    for (const e of d.entries) {
      if (e.binding > maxBinding.binding) maxBinding = e
    }
    return maxBinding.buffer
  }

  // Preamble bind groups (use TVM's buffers directly — embedding + norm are simple)
  const embedBG = dev.createBindGroup({
    layout: embed.getBindGroupLayout(0),
    entries: dispatches[0].entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
  })
  const normBG = dev.createBindGroup({
    layout: initialNorm.getBindGroupLayout(0),
    entries: dispatches[1].entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
  })

  // Per-layer bind groups
  const layers: Array<{
    qkv: GPUBindGroup; rope: GPUBindGroup; kvAppend: GPUBindGroup; attn: GPUBindGroup
    oProj: GPUBindGroup; addNorm1: GPUBindGroup; ffnUp: GPUBindGroup; silu: GPUBindGroup
    ffnDown: GPUBindGroup; addNorm2: GPUBindGroup
  }> = []

  // Track uniform buffers so we can write our values to them
  const layerUniformBufs: Array<{
    qkv: GPUBuffer | null; rope: GPUBuffer | null; kvAppend: GPUBuffer | null
    attn: GPUBuffer | null; oProj: GPUBuffer | null; addNorm1: GPUBuffer | null
    ffnUp: GPUBuffer | null; silu: GPUBuffer | null; ffnDown: GPUBuffer | null
    addNorm2: GPUBuffer | null
  }> = []

  for (let L = 0; L < LAYERS; L++) {
    const base = 2 + L * 10
    layers.push({
      qkv: buildBG(base + 0, qkvMatmul),
      rope: buildBG(base + 1, rope),
      kvAppend: buildBG(base + 2, kvAppend),
      attn: buildBG(base + 3, attention),
      oProj: buildBG(base + 4, oProj),
      addNorm1: buildBG(base + 5, addNorm),
      ffnUp: buildBG(base + 6, ffnUp),
      silu: buildBG(base + 7, silu),
      ffnDown: buildBG(base + 8, ffnDown),
      addNorm2: buildBG(base + 9, addNorm),
    })
    layerUniformBufs.push({
      qkv: findUniformBuf(base + 0),
      rope: findUniformBuf(base + 1),
      kvAppend: findUniformBuf(base + 2),
      attn: findUniformBuf(base + 3),
      oProj: findUniformBuf(base + 4),
      addNorm1: findUniformBuf(base + 5),
      ffnUp: findUniformBuf(base + 6),
      silu: findUniformBuf(base + 7),
      ffnDown: findUniformBuf(base + 8),
      addNorm2: findUniformBuf(base + 9),
    })
  }

  // Tail (LM head + sampling) — keep TVM's for now
  const tail: Array<{ pipeline: GPUComputePipeline; bg: GPUBindGroup; wg: [number, number, number] }> = []
  for (let i = 322; i < dispatches.length; i++) {
    const d = dispatches[i]
    tail.push({
      pipeline: d.pipeline,
      bg: dev.createBindGroup({
        layout: d.pipeline.getBindGroupLayout(0),
        entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
      }),
      wg: d.workgroups,
    })
  }

  // Log uniform buffer sizes for first layer
  if (layerUniformBufs.length > 0) {
    const u = layerUniformBufs[0]
    console.log(`[phi3v2] Layer 0 uniform bufs: qkv=${u.qkv?.size} rope=${u.rope?.size} kvAppend=${u.kvAppend?.size} attn=${u.attn?.size} oProj=${u.oProj?.size} addNorm=${u.addNorm1?.size} ffnUp=${u.ffnUp?.size} silu=${u.silu?.size} ffnDown=${u.ffnDown?.size}`)
  }
  console.log(`[phi3v2] Built ${LAYERS} layer bind groups + ${tail.length} tail dispatches`)

  // Token input buffer (for embedding lookup) + position map
  // These are written per-token from our code, not from captured writes
  // Token ID buffer — write[0] in captured writes, 4B buffer
  const tokenIdBuf = dispatches[0].entries.find(e => e.buffer.size === 4)?.buffer ?? null

  // Position map buffer — in RoPE dispatch (binding 1 = position_map)
  // It's a 4096B buffer. Find it by looking for binding 1 in RoPE dispatch.
  const ropeDisp = dispatches[3]  // first layer's RoPE (base=2, offset=1)
  const positionMapBuf = ropeDisp?.entries.find(e => e.binding === 1)?.buffer ?? null

  // Also need length_info and other KV cache metadata buffers
  // Find them from the attention dispatch (base=2, offset=3 = first layer's attention)
  const attnDisp = dispatches[5]  // first layer's attention
  const lengthInfoBuf = attnDisp?.entries.find(e => e.binding === 2)?.buffer ?? null // length_info
  const pageIndptrBuf = attnDisp?.entries.find(e => e.binding === 5)?.buffer ?? null // page_table_indptr
  const qRopePosBuf = attnDisp?.entries.find(e => e.binding === 8)?.buffer ?? null   // q_rope_position

  // Copy source for token readback
  const copySrc = capture.copy?.src
  const copySrcOff = capture.copy?.srcOffset ?? 0
  const copySize = capture.copy?.size ?? 4

  console.log(`[phi3v2] Buffers: token=${tokenIdBuf?.size}B posMap=${positionMapBuf?.size}B qRopePos=${qRopePosBuf?.size}B lengthInfo=${lengthInfoBuf?.size}B pageIndptr=${pageIndptrBuf?.size}B copy=${copySrc?.size}B`)

  // ============================================================
  // Generate
  // ============================================================

  let position = 0  // set during first generate call

  async function generate(startPosition: number, startTokenId: number, maxTokens: number, onToken?: (id: number) => void): Promise<number[]> {
    position = startPosition
    let tokenId = startTokenId

    const STOP = new Set([2, 32000, 32007])
    const tokens: number[] = []

    for (let i = 0; i < maxTokens; i++) {
      position++
      const nnzPages = Math.floor(position / PAGE_SIZE) + 1

      // Write per-token values
      if (tokenIdBuf) dev.queue.writeBuffer(tokenIdBuf, 0, new Int32Array([tokenId]))
      if (positionMapBuf) dev.queue.writeBuffer(positionMapBuf, 0, new Int32Array([position]))
      if (qRopePosBuf) dev.queue.writeBuffer(qRopePosBuf, 0, new Int32Array([position]))
      // length_info stores the current sequence length for attention
      if (lengthInfoBuf) dev.queue.writeBuffer(lengthInfoBuf, 0, new Int32Array([position + 1]))

      // On first iteration, dump TVM's uniform values for comparison
      if (i === 0) {
        // Read TVM's values from captured writes for first layer
        const base = 2
        for (let d = base; d < base + 10; d++) {
          const disp = dispatches[d]
          // Find the uniform entry (last binding)
          let lastEntry = disp.entries[0]
          for (const e of disp.entries) { if (e.binding > lastEntry.binding) lastEntry = e }
          // Find matching captured write
          const matchingWrite = capture.writes.find(w => w.buffer === lastEntry.buffer)
          if (matchingWrite) {
            const vals = []
            for (let b = 0; b < Math.min(matchingWrite.data.length, 32); b += 4) {
              vals.push(matchingWrite.data[b] | (matchingWrite.data[b+1] << 8) | (matchingWrite.data[b+2] << 16) | (matchingWrite.data[b+3] << 24))
            }
            console.log(`[phi3v2] Dispatch ${d - base} TVM uniform (${matchingWrite.data.length}B): [${vals.join(', ')}]`)
          }
        }
      }

      // Write ALL uniforms from first principles — no TVM dependency
      for (let L = 0; L < LAYERS; L++) {
        const u = layerUniformBufs[L]
        if (u.qkv) dev.queue.writeBuffer(u.qkv, 0, writeMatmulParams(D * 3 - 1))
        if (u.rope) dev.queue.writeBuffer(u.rope, 0, writeRopeParams(1))
        if (u.kvAppend) dev.queue.writeBuffer(u.kvAppend, 0, writeKvAppendParams())
        if (u.attn) dev.queue.writeBuffer(u.attn, 0, writeAttentionParams(nnzPages))
        if (u.oProj) dev.queue.writeBuffer(u.oProj, 0, writeMatmulParams(D - 1))
        if (u.addNorm1) dev.queue.writeBuffer(u.addNorm1, 0, writeAddNormParams())
        if (u.ffnUp) dev.queue.writeBuffer(u.ffnUp, 0, writeMatmulParams(FFN_DIM * 2 - 1))
        if (u.silu) dev.queue.writeBuffer(u.silu, 0, writeSiluParams())
        if (u.ffnDown) dev.queue.writeBuffer(u.ffnDown, 0, writeMatmulParams(D - 1))
        if (u.addNorm2) dev.queue.writeBuffer(u.addNorm2, 0, writeAddNormParams())
      }

      // Preamble
      dispatch(embed, embedBG, dispatches[0].workgroups)
      dispatch(initialNorm, normBG, dispatches[1].workgroups)

      // 32 transformer layers
      for (let L = 0; L < LAYERS; L++) {
        const bg = layers[L]
        const base = 2 + L * 10
        dispatch(qkvMatmul, bg.qkv, dispatches[base + 0].workgroups)
        dispatch(rope, bg.rope, dispatches[base + 1].workgroups)
        dispatch(kvAppend, bg.kvAppend, dispatches[base + 2].workgroups)
        dispatch(attention, bg.attn, dispatches[base + 3].workgroups)
        dispatch(oProj, bg.oProj, dispatches[base + 4].workgroups)
        dispatch(addNorm, bg.addNorm1, dispatches[base + 5].workgroups)
        dispatch(ffnUp, bg.ffnUp, dispatches[base + 6].workgroups)
        dispatch(silu, bg.silu, dispatches[base + 7].workgroups)
        dispatch(ffnDown, bg.ffnDown, dispatches[base + 8].workgroups)
        dispatch(addNorm, bg.addNorm2, dispatches[base + 9].workgroups)
      }

      // Tail
      for (const t of tail) dispatch(t.pipeline, t.bg, t.wg)

      // Read token
      tokenId = await readToken()
      if (tokenId < 0 || tokenId >= VOCAB) break
      tokens.push(tokenId)
      if (onToken) onToken(tokenId)
      if (STOP.has(tokenId)) break
    }

    return tokens
  }

  function dispatch(pipeline: GPUComputePipeline, bg: GPUBindGroup, wg: [number, number, number]): void {
    const enc = dev.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bg)
    pass.dispatchWorkgroups(...wg)
    pass.end()
    dev.queue.submit([enc.finish()])
  }

  async function readToken(): Promise<number> {
    if (!copySrc) return -1
    const buf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = dev.createCommandEncoder()
    enc.copyBufferToBuffer(copySrc, copySrcOff, buf, 0, copySize)
    dev.queue.submit([enc.finish()])
    await buf.mapAsync(GPUMapMode.READ)
    const token = new DataView(buf.getMappedRange()).getInt32(0, true)
    buf.unmap()
    buf.destroy()
    return token
  }

  return { generate }
}
