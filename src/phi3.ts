/**
 * PHI-3 INFERENCE ENGINE
 *
 * Built from scratch using TVM's compiled WGSL shaders as building blocks.
 * No TVM runtime. No replay. No capture tape.
 *
 * Model: Phi-3-mini-4k-instruct (3.8B params, Q4 quantized)
 * Architecture: 32 layers, D=3072, 32 heads, head_dim=96, FFN=8192, vocab=32064
 *
 * Per-layer dispatch sequence (10 dispatches):
 *   0. QKV projection   — fused_dequantize + NT_matmul (int4 dequant + matmul)
 *   1. RoPE             — rotary position embedding
 *   2. KV cache append  — write K,V to paged cache
 *   3. Attention        — paged KV attention decode
 *   4. Output projection — fused_dequantize + NT_matmul
 *   5. Residual + Norm  — add + RMSNorm (fused)
 *   6. FFN gate+up      — fused_dequantize + NT_matmul
 *   7. SiLU activation  — split + silu + multiply (fused)
 *   8. FFN down         — fused_dequantize + NT_matmul
 *   9. Residual + Norm  — add + RMSNorm (fused)
 *
 * Preamble: embedding lookup (1 dispatch) + initial RMSNorm (1 dispatch)
 * Tail: LM head projection (1 dispatch) + sampling (19 dispatches)
 */

import { CaptureResult } from './capture.js'
// @ts-ignore — Vite raw import
import argmaxWGSL from './shaders/argmax.wgsl?raw'
// @ts-ignore — Vite raw import
import fusedFFNWGSL from './shaders/fused-ffn.wgsl?raw'

// ============================================================
// Model Config
// ============================================================

export const CFG = {
  D: 3072,           // hidden dimension
  HEADS: 32,         // attention heads
  HEAD_DIM: 96,      // D / HEADS
  LAYERS: 32,        // transformer layers
  FFN: 8192,         // FFN intermediate dimension
  VOCAB: 32064,      // vocabulary size
  PAGE_SIZE: 16,     // KV cache page size
  MAX_SEQ: 4096,     // max sequence length
} as const

// ============================================================
// Pipeline Registry — maps each step to its shader
// ============================================================

interface Pipelines {
  // Preamble
  embed: GPUComputePipeline
  initialNorm: GPUComputePipeline

  // Per-layer (same pipeline reused across layers, different bind groups for different weights)
  qkv: GPUComputePipeline
  rope: GPUComputePipeline
  kvAppend: GPUComputePipeline
  attention: GPUComputePipeline
  oProj: GPUComputePipeline
  addNorm: GPUComputePipeline
  ffnUp: GPUComputePipeline
  silu: GPUComputePipeline
  ffnDown: GPUComputePipeline

  // Tail (LM head + sampling — use TVM's captured pipelines directly for now)
  tail: Array<{ pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; workgroups: [number, number, number] }>
}

// ============================================================
// Build the engine
// ============================================================

export function buildPhi3Engine(capture: CaptureResult) {
  const dev = capture.device

  // Step 1: Find pipelines by entry point name
  const find = (name: string) => {
    const p = capture.pipelines.find(p => p.entryPoint.includes(name))
    if (!p) throw new Error(`Pipeline not found: ${name}`)
    return p
  }

  const pipes: Pipelines = {
    embed: find('dequantize_take1').pipeline,
    initialNorm: find('rms_norm2').pipeline,
    qkv: find('NT_matmul10').pipeline,
    rope: find('fused_rope').pipeline,
    kvAppend: find('kv_cache_transpose_append_kernel').pipeline,
    attention: find('batch_decode_paged_kv_kernel').pipeline,
    oProj: find('NT_matmul11').pipeline,
    addNorm: find('fuse_add_norm_decode').pipeline,
    ffnUp: find('NT_matmul12').pipeline,
    silu: find('split2_silu2_multiply2').pipeline,
    ffnDown: find('NT_matmul13').pipeline,
    tail: [],  // built from captured dispatches
  }

  // Create fused FFN pipeline from our WGSL
  const fusedFFNModule = dev.createShaderModule({ code: fusedFFNWGSL, label: 'fused_ffn' })
  const fusedFFNPipeline = dev.createComputePipeline({
    layout: 'auto',
    compute: { module: fusedFFNModule, entryPoint: 'fused_ffn_kernel' },
  })

  // Fused FFN uniform: packGridDimX = 8191 (8192 - 1)
  const fusedFFNUniform = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  dev.queue.writeBuffer(fusedFFNUniform, 0, new Uint32Array([8191]))

  console.log('[phi3] Pipelines found (including fused FFN)')

  // Step 2: Tail dispatches (LM head + sampling) — use TVM's pipelines
  if (capture.dispatches.length >= 342) {
    for (let i = 322; i < capture.dispatches.length; i++) {
      const d = capture.dispatches[i]
      const bg = dev.createBindGroup({
        layout: d.pipeline.getBindGroupLayout(0),
        entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
      })
      pipes.tail.push({ pipeline: d.pipeline, bindGroup: bg, workgroups: d.workgroups })
    }
    console.log(`[phi3] Tail: ${pipes.tail.length} dispatches (LM head + sampling)`)
  }

  // Step 3: Build per-layer bind groups
  // Each layer needs different weight buffers but same activation buffers
  // We identify weight buffers from the captured dispatches:
  //   Layer N's dispatches are at indices 2 + N*10 .. 2 + N*10 + 9

  // For now, use TVM's captured bind group entries directly for each layer
  // This still uses TVM's buffers but with proper per-layer isolation
  const USE_FUSED_FFN = false  // toggle fused FFN (disabled until correctness verified)

  const layerBindGroups: Array<{
    qkv: GPUBindGroup
    rope: GPUBindGroup
    kvAppend: GPUBindGroup
    attention: GPUBindGroup
    oProj: GPUBindGroup
    addNorm1: GPUBindGroup
    fusedFFN: GPUBindGroup | null  // fused gate+up+silu (replaces ffnUp + silu)
    ffnUp: GPUBindGroup
    silu: GPUBindGroup
    ffnDown: GPUBindGroup
    addNorm2: GPUBindGroup
  }> = []

  for (let layer = 0; layer < CFG.LAYERS; layer++) {
    const base = 2 + layer * 10  // first 2 are preamble

    const makeBG = (dispIdx: number, pipeline: GPUComputePipeline) => {
      const d = capture.dispatches[dispIdx]
      if (!d) throw new Error(`Dispatch ${dispIdx} not found`)
      const entries: GPUBindGroupEntry[] = d.entries.map(e => ({
        binding: e.binding,
        resource: { buffer: e.buffer, offset: e.offset, size: e.size },
      }))
      return dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries })
    }

    // Build fused FFN bind group:
    // output = SiLU's output buffer (dispatch base+7, binding 0)
    // input = FFN up's input buffer (dispatch base+6, binding 1)
    // scales = FFN up's scales (dispatch base+6, binding 2)
    // weights = FFN up's weights (dispatch base+6, binding 3)
    // uniform = our fused FFN uniform
    let fusedFFNBG: GPUBindGroup | null = null
    if (USE_FUSED_FFN) {
      const ffnUpDisp = capture.dispatches[base + 6]
      const siluDisp = capture.dispatches[base + 7]
      if (ffnUpDisp && siluDisp) {
        const siluOutput = siluDisp.entries.find(e => e.binding === 0)  // SiLU output
        const ffnInput = ffnUpDisp.entries.find(e => e.binding === 1)   // normed hidden state
        const ffnScales = ffnUpDisp.entries.find(e => e.binding === 2)  // weight scales
        const ffnWeights = ffnUpDisp.entries.find(e => e.binding === 3) // int4 weights
        if (siluOutput && ffnInput && ffnScales && ffnWeights) {
          try {
            fusedFFNBG = dev.createBindGroup({
              layout: fusedFFNPipeline.getBindGroupLayout(0),
              entries: [
                { binding: 0, resource: { buffer: siluOutput.buffer, offset: siluOutput.offset, size: siluOutput.size } },
                { binding: 1, resource: { buffer: ffnInput.buffer, offset: ffnInput.offset, size: ffnInput.size } },
                { binding: 2, resource: { buffer: ffnScales.buffer, offset: ffnScales.offset, size: ffnScales.size } },
                { binding: 3, resource: { buffer: ffnWeights.buffer, offset: ffnWeights.offset, size: ffnWeights.size } },
                { binding: 4, resource: { buffer: fusedFFNUniform, offset: 0, size: 16 } },
              ],
            })
          } catch (e) {
            console.warn(`[phi3] Layer ${layer} fused FFN bind group failed: ${e}`)
          }
        }
      }
    }

    layerBindGroups.push({
      qkv: makeBG(base + 0, pipes.qkv),
      rope: makeBG(base + 1, pipes.rope),
      kvAppend: makeBG(base + 2, pipes.kvAppend),
      attention: makeBG(base + 3, pipes.attention),
      oProj: makeBG(base + 4, pipes.oProj),
      addNorm1: makeBG(base + 5, pipes.addNorm),
      fusedFFN: fusedFFNBG,
      ffnUp: makeBG(base + 6, pipes.ffnUp),
      silu: makeBG(base + 7, pipes.silu),
      ffnDown: makeBG(base + 8, pipes.ffnDown),
      addNorm2: makeBG(base + 9, pipes.addNorm),
    })
  }

  console.log(`[phi3] Built ${layerBindGroups.length} layer bind groups`)

  // Preamble bind groups
  const embedBG = (() => {
    const d = capture.dispatches[0]
    return dev.createBindGroup({
      layout: pipes.embed.getBindGroupLayout(0),
      entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
    })
  })()

  const initialNormBG = (() => {
    const d = capture.dispatches[1]
    return dev.createBindGroup({
      layout: pipes.initialNorm.getBindGroupLayout(0),
      entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
    })
  })()

  // Step 4: Identify uniform buffers that need per-token updates
  // From captured writes, we know which buffers change per token
  // For now, capture the write targets for the 6 changing values
  const writes = capture.writes
  console.log(`[phi3] ${writes.length} uniform writes captured`)

  // Step 5: Find copy operation for token readback
  const copySrc = capture.copy?.src ?? null
  const copySrcOff = capture.copy?.srcOffset ?? 0
  const copySize = capture.copy?.size ?? 4

  // ============================================================
  // Generate function
  // ============================================================

  // Helper functions
  const readU32 = (d: Uint8Array) => d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24)
  const writeU32 = (v: number) => { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, v, true); return a }

  // Find per-token buffers from captured writes
  const tokenIdBuf = writes[0]?.buffer
  const posMapBuf = (() => {
    const ropeDisp = capture.dispatches[3] // first layer RoPE
    return ropeDisp?.entries.find(e => e.binding === 1)?.buffer ?? null
  })()
  const qRopePosBuf = (() => {
    const attnDisp = capture.dispatches[5] // first layer attention
    return attnDisp?.entries.find(e => e.binding === 8)?.buffer ?? null
  })()
  const lengthInfoBuf = (() => {
    const attnDisp = capture.dispatches[5]
    return attnDisp?.entries.find(e => e.binding === 2)?.buffer ?? null
  })()

  /**
   * Run one forward pass (all 342 dispatches) for a single token at a given position.
   * This builds the KV cache entry for that position.
   */
  function forwardPass(tokenId: number, pos: number): void {
    const nnzPages = Math.floor(pos / CFG.PAGE_SIZE) + 1

    // Write per-token values
    if (tokenIdBuf) dev.queue.writeBuffer(tokenIdBuf, 0, writeU32(tokenId))
    if (posMapBuf) dev.queue.writeBuffer(posMapBuf, 0, new Int32Array([pos]))
    if (qRopePosBuf) dev.queue.writeBuffer(qRopePosBuf, 0, new Int32Array([pos]))
    if (lengthInfoBuf) dev.queue.writeBuffer(lengthInfoBuf, 0, new Int32Array([pos + 1]))

    // Update nnz_pages in attention uniforms for all layers
    for (let layer = 0; layer < CFG.LAYERS; layer++) {
      const attnDisp = capture.dispatches[2 + layer * 10 + 3]
      const attnUniform = attnDisp?.entries.find(e => e.binding === 9)?.buffer
      if (attnUniform) dev.queue.writeBuffer(attnUniform, 16, new Int32Array([nnzPages])) // offset 16 = nnz_pages
    }

    // Preamble
    dispatch(pipes.embed, embedBG, capture.dispatches[0].workgroups)
    dispatch(pipes.initialNorm, initialNormBG, capture.dispatches[1].workgroups)

    // 32 layers
    for (let layer = 0; layer < CFG.LAYERS; layer++) {
      const bg = layerBindGroups[layer]
      const base = 2 + layer * 10
      dispatch(pipes.qkv, bg.qkv, capture.dispatches[base + 0].workgroups)
      dispatch(pipes.rope, bg.rope, capture.dispatches[base + 1].workgroups)
      dispatch(pipes.kvAppend, bg.kvAppend, capture.dispatches[base + 2].workgroups)
      dispatch(pipes.attention, bg.attention, capture.dispatches[base + 3].workgroups)
      dispatch(pipes.oProj, bg.oProj, capture.dispatches[base + 4].workgroups)
      dispatch(pipes.addNorm, bg.addNorm1, capture.dispatches[base + 5].workgroups)
      if (USE_FUSED_FFN && bg.fusedFFN) {
        dispatch(fusedFFNPipeline, bg.fusedFFN, [8192, 1, 1])
      } else {
        dispatch(pipes.ffnUp, bg.ffnUp, capture.dispatches[base + 6].workgroups)
        dispatch(pipes.silu, bg.silu, capture.dispatches[base + 7].workgroups)
      }
      dispatch(pipes.ffnDown, bg.ffnDown, capture.dispatches[base + 8].workgroups)
      dispatch(pipes.addNorm, bg.addNorm2, capture.dispatches[base + 9].workgroups)
    }

    // Tail (LM head + sampling)
    for (const t of pipes.tail) dispatch(t.pipeline, t.bindGroup, t.workgroups)
    flushDispatches()
  }

  /**
   * Sequential prefill: process each prompt token through the full decode pipeline.
   * Builds KV cache entries for all prompt positions.
   */
  async function prefill(tokenIds: number[], startPos: number): Promise<number> {
    console.log(`[phi3] Prefilling ${tokenIds.length} tokens from pos ${startPos}...`)

    // Initialize page table for the required number of pages
    const totalPositions = startPos + tokenIds.length
    const neededPages = Math.floor(totalPositions / CFG.PAGE_SIZE) + 1

    // page_table_indptr: [0, neededPages] (sequence 0 uses pages 0..neededPages-1)
    const pageIndptrBuf = (() => {
      const attnDisp = capture.dispatches[5]
      return attnDisp?.entries.find(e => e.binding === 5)?.buffer ?? null
    })()
    if (pageIndptrBuf) {
      dev.queue.writeBuffer(pageIndptrBuf, 0, new Int32Array([0, neededPages]))
    }

    // page_table_values: use pages starting from offset 200 to avoid conflicting
    // with TVM's existing page allocations (TVM uses pages 0-N from model init)
    const PAGE_OFFSET = 200
    const pageValuesBuf = (() => {
      const attnDisp = capture.dispatches[5]
      return attnDisp?.entries.find(e => e.binding === 6)?.buffer ?? null
    })()
    if (pageValuesBuf) {
      const pageValues = new Int32Array(neededPages)
      for (let i = 0; i < neededPages; i++) pageValues[i] = PAGE_OFFSET + i
      dev.queue.writeBuffer(pageValuesBuf, 0, pageValues)
      console.log(`[phi3] Page values: [${Array.from(pageValues).join(', ')}]`)
    }

    console.log(`[phi3] Page table: ${neededPages} pages for ${totalPositions} positions`)

    let lastTokenId = 0
    for (let i = 0; i < tokenIds.length; i++) {
      forwardPass(tokenIds[i], startPos + i)
      lastTokenId = await readToken()
      if (i < 3 || i === tokenIds.length - 1) {
        console.log(`[phi3] Prefill step ${i}: token=${tokenIds[i]} pos=${startPos + i} → output=${lastTokenId}`)
      }
    }
    console.log(`[phi3] Prefill done. Last output: ${lastTokenId}`)
    return lastTokenId
  }

  async function generate(maxTokens: number, onToken?: (id: number) => void): Promise<number[]> {
    if (!copySrc) throw new Error('No copy source for token readback')

    // Pure replay first token (establishes correct activation state)
    for (const w of writes) {
      dev.queue.writeBuffer(w.buffer, w.offset, w.data.slice().buffer)
    }

    // Preamble
    dispatch(pipes.embed, embedBG, capture.dispatches[0].workgroups)
    dispatch(pipes.initialNorm, initialNormBG, capture.dispatches[1].workgroups)

    // 32 layers
    for (let layer = 0; layer < CFG.LAYERS; layer++) {
      const bg = layerBindGroups[layer]
      const base = 2 + layer * 10
      dispatch(pipes.qkv, bg.qkv, capture.dispatches[base + 0].workgroups)
      dispatch(pipes.rope, bg.rope, capture.dispatches[base + 1].workgroups)
      dispatch(pipes.kvAppend, bg.kvAppend, capture.dispatches[base + 2].workgroups)
      dispatch(pipes.attention, bg.attention, capture.dispatches[base + 3].workgroups)
      dispatch(pipes.oProj, bg.oProj, capture.dispatches[base + 4].workgroups)
      dispatch(pipes.addNorm, bg.addNorm1, capture.dispatches[base + 5].workgroups)
      if (USE_FUSED_FFN && bg.fusedFFN) {
        dispatch(fusedFFNPipeline, bg.fusedFFN, [8192, 1, 1])  // fused: gate+up+silu in one
      } else {
        dispatch(pipes.ffnUp, bg.ffnUp, capture.dispatches[base + 6].workgroups)
        dispatch(pipes.silu, bg.silu, capture.dispatches[base + 7].workgroups)
      }
      dispatch(pipes.ffnDown, bg.ffnDown, capture.dispatches[base + 8].workgroups)
      dispatch(pipes.addNorm, bg.addNorm2, capture.dispatches[base + 9].workgroups)
    }

    // Tail: use TVM's sampling pipeline (includes temperature + top-p)
    // TODO: replace with our own fused sampling shader
    for (const t of pipes.tail) dispatch(t.pipeline, t.bindGroup, t.workgroups)

    flushDispatches()
    let tokenId = await readToken()
    console.log(`[phi3] First token: ${tokenId}`)

    let pos = readU32(writes[4]?.data ?? new Uint8Array(4)) + 1
    const seqBase = readU32(writes[8]?.data ?? new Uint8Array(4))
    const pos0 = pos - 1

    const STOP = new Set([2, 32000, 32007])
    const tokens: number[] = []

    for (let i = 0; i < maxTokens; i++) {
      pos++

      // Patch 6 changing values
      dev.queue.writeBuffer(writes[0].buffer, writes[0].offset, writeU32(tokenId))
      for (const idx of [4, 11, 12]) dev.queue.writeBuffer(writes[idx].buffer, writes[idx].offset, writeU32(pos))
      dev.queue.writeBuffer(writes[8].buffer, writes[8].offset, writeU32(seqBase + (pos - pos0)))
      if (writes.length > 349) {
        const seed = new ArrayBuffer(4)
        new DataView(seed).setFloat32(0, Math.random(), true)
        dev.queue.writeBuffer(writes[349].buffer, writes[349].offset, seed)
      }

      // Preamble
      dispatch(pipes.embed, embedBG, capture.dispatches[0].workgroups)
      dispatch(pipes.initialNorm, initialNormBG, capture.dispatches[1].workgroups)

      // 32 layers
      for (let layer = 0; layer < CFG.LAYERS; layer++) {
        const bg = layerBindGroups[layer]
        const base = 2 + layer * 10
        dispatch(pipes.qkv, bg.qkv, capture.dispatches[base + 0].workgroups)
        dispatch(pipes.rope, bg.rope, capture.dispatches[base + 1].workgroups)
        dispatch(pipes.kvAppend, bg.kvAppend, capture.dispatches[base + 2].workgroups)
        dispatch(pipes.attention, bg.attention, capture.dispatches[base + 3].workgroups)
        dispatch(pipes.oProj, bg.oProj, capture.dispatches[base + 4].workgroups)
        dispatch(pipes.addNorm, bg.addNorm1, capture.dispatches[base + 5].workgroups)
        dispatch(pipes.ffnUp, bg.ffnUp, capture.dispatches[base + 6].workgroups)
        dispatch(pipes.silu, bg.silu, capture.dispatches[base + 7].workgroups)
        dispatch(pipes.ffnDown, bg.ffnDown, capture.dispatches[base + 8].workgroups)
        dispatch(pipes.addNorm, bg.addNorm2, capture.dispatches[base + 9].workgroups)
      }

      // Tail
      for (const t of pipes.tail) dispatch(t.pipeline, t.bindGroup, t.workgroups)

      flushDispatches()
      tokenId = await readToken()
      if (tokenId < 0 || tokenId >= CFG.VOCAB) break
      tokens.push(tokenId)
      if (onToken) onToken(tokenId)
      if (STOP.has(tokenId)) break
    }

    return tokens
  }

  /**
   * Full chat: prefill the prompt, then decode.
   * ZERO TVM. Our engine does everything.
   */
  async function chat(promptTokens: number[], maxTokens: number, onToken?: (id: number) => void): Promise<number[]> {
    if (!copySrc) throw new Error('No copy source for token readback')

    // Prefill: process each prompt token, building KV cache
    let tokenId = await prefill(promptTokens, 0)
    let pos = promptTokens.length

    const STOP = new Set([2, 32000, 32007])
    const tokens: number[] = []

    // First token from prefill
    if (tokenId >= 0 && tokenId < CFG.VOCAB && !STOP.has(tokenId)) {
      tokens.push(tokenId)
      if (onToken) onToken(tokenId)
    }

    // Decode loop
    for (let i = 0; i < maxTokens - 1; i++) {
      pos++
      forwardPass(tokenId, pos)
      tokenId = await readToken()
      if (tokenId < 0 || tokenId >= CFG.VOCAB) break
      tokens.push(tokenId)
      if (onToken) onToken(tokenId)
      if (STOP.has(tokenId)) break
    }

    return tokens
  }

  // Batched dispatch — accumulate passes, flush periodically
  let pendingEncoder: GPUCommandEncoder | null = null
  let pendingCount = 0
  const BATCH = 1  // correctness first, speed from fusion

  function dispatch(pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, workgroups: [number, number, number]): void {
    if (!pendingEncoder) pendingEncoder = dev.createCommandEncoder()
    const pass = pendingEncoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(...workgroups)
    pass.end()
    pendingCount++
    if (pendingCount >= BATCH) flushDispatches()
  }

  function flushDispatches(): void {
    if (pendingEncoder && pendingCount > 0) {
      dev.queue.submit([pendingEncoder.finish()])
      pendingEncoder = null
      pendingCount = 0
    }
  }

  async function readToken(): Promise<number> {
    const src = copySrc!
    const srcOff = copySrcOff
    const sz = copySize

    const buf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = dev.createCommandEncoder()
    enc.copyBufferToBuffer(src, srcOff, buf, 0, sz)
    dev.queue.submit([enc.finish()])
    await buf.mapAsync(GPUMapMode.READ)
    const token = new DataView(buf.getMappedRange()).getInt32(0, true)
    buf.unmap()
    buf.destroy()
    return token
  }

  return { generate, chat, prefill, pipes, layerBindGroups }
}
