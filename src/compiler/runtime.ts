/**
 * RUNTIME — Prefill + decode loop. Zero TVM.
 *
 * Uses pipelines and buffers from compiler.ts.
 * Drives the full transformer forward pass:
 *   embed → norm → 32×[qkv→rope→kv→attn→oproj→addnorm→ffn→ffndown→addnorm] → norm → lmhead → argmax
 *
 * Decode: 1 token at a time, ~100 dispatches per token (vs TVM's 342).
 * Prefill: N tokens at once (TODO: batch matmul shaders).
 */

import { PHI3, CompiledModel } from './compiler.js'

// ============================================================
// Uniform helpers
// ============================================================

function u32(v: number): ArrayBuffer {
  const a = new ArrayBuffer(4)
  new DataView(a).setUint32(0, v, true)
  return a
}

function i32(v: number): ArrayBuffer {
  const a = new ArrayBuffer(4)
  new DataView(a).setInt32(0, v, true)
  return a
}

function f32(v: number): ArrayBuffer {
  const a = new ArrayBuffer(4)
  new DataView(a).setFloat32(0, v, true)
  return a
}

function uniformBuf(device: GPUDevice, data: ArrayBuffer[]): GPUBuffer {
  const totalSize = data.reduce((s, d) => s + d.byteLength, 0)
  const padded = Math.ceil(totalSize / 16) * 16  // WebGPU 16-byte alignment
  const buf = device.createBuffer({
    size: Math.max(padded, 16),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const combined = new Uint8Array(padded)
  let offset = 0
  for (const d of data) {
    combined.set(new Uint8Array(d), offset)
    offset += d.byteLength
  }
  device.queue.writeBuffer(buf, 0, combined)
  return buf
}

// ============================================================
// Bind group helpers
// ============================================================

function bg(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  entries: GPUBuffer[]
): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: entries.map((buf, i) => ({
      binding: i,
      resource: { buffer: buf, offset: 0, size: buf.size },
    })),
  })
}

// ============================================================
// Forward pass — single decode token
// ============================================================

export interface Runtime {
  decodeToken(tokenId: number, position: number): Promise<number>
  generate(promptTokenIds: number[], maxTokens: number, onToken?: (id: number) => void): Promise<number[]>
}

export function createRuntime(model: CompiledModel): Runtime {
  const { device, pipelines, buffers, layerWeights } = model
  const P = pipelines
  const B = buffers

  // Pre-create uniform buffers for matmul dimensions
  const qkvUniform = uniformBuf(device, [u32(384), u32(96), u32(9216)])     // K=3072: K_PACKED=384, SCALES=96, N=9216
  const oProjUniform = uniformBuf(device, [u32(384), u32(96), u32(3072)])   // K=3072→3072
  const ffnDownUniform = uniformBuf(device, [u32(1024), u32(256), u32(3072)]) // K=8192: K_PACKED=1024, SCALES=256, N=3072
  const lmHeadUniform = uniformBuf(device, [u32(384), u32(96), u32(32064)])  // K=3072→32064
  const fusedFfnUniform = uniformBuf(device, [u32(8192)])                    // packGridDimX=8192
  const argmaxUniform = uniformBuf(device, [u32(PHI3.VOCAB)])

  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)  // 0.10206

  function dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    wgX: number, wgY = 1, wgZ = 1
  ): void {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(wgX, wgY, wgZ)
    pass.end()
  }

  async function decodeToken(tokenId: number, position: number): Promise<number> {
    const enc = device.createCommandEncoder()

    // Write token ID and position
    device.queue.writeBuffer(B.inputIds, 0, i32(tokenId))
    device.queue.writeBuffer(B.positionMap, 0, i32(position))

    // Update page table
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    const pageVals = new Int32Array(nnzPages)
    for (let i = 0; i < nnzPages; i++) pageVals[i] = i
    device.queue.writeBuffer(B.pageValues, 0, pageVals)
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))

    // --- EMBEDDING ---
    const embdUniform = uniformBuf(device, [u32(1), u32(12)])  // seq_len=1, packGridDimX=12
    dispatch(enc, P.embedding, bg(device, P.embedding, [
      B.hidden1, B.inputIds, B.embdScales, B.embdWeights, embdUniform
    ]), 12)

    // --- INITIAL RMSNORM ---
    const normUniform = uniformBuf(device, [u32(1)])
    dispatch(enc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.hidden2, B.hidden1, B.initNormGamma, normUniform
    ]), 1)

    // --- 32 TRANSFORMER LAYERS ---
    // hidden2 = normed input, hidden1 = residual from embedding
    // After each layer: hidden2 = normed output for next layer

    let normIn = B.hidden2   // normed hidden for matmul input
    let resIn = B.hidden1    // residual connection

    for (let L = 0; L < PHI3.LAYERS; L++) {
      const lw = layerWeights[L]

      // QKV matmul: normIn → qkvOut (9216)
      dispatch(enc, P.qkvMatmul, bg(device, P.qkvMatmul, [
        B.qkvOut, normIn, lw.qkvScales, lw.qkvWeights, qkvUniform
      ]), 9216)

      // RoPE + QKV split: qkvOut → qOut, kOut, vOut
      const ropeUniform = uniformBuf(device, [i32(1), i32(0), i32(1), u32(36)])
      dispatch(enc, P.rope, bg(device, P.rope, [
        B.kOut, B.positionMap, B.qOut, B.qkvOut, B.vOut, ropeUniform
      ]), 36)

      // KV append: kOut, vOut → kvPages[L]
      const kvUniform = uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(12)])
      dispatch(enc, P.kvAppend, bg(device, P.kvAppend, [
        B.kOut, B.vOut, lw.kvPages, B.positionMap, kvUniform
      ]), 12)

      // Attention: Q + KV pages → attnOut
      const attnUniform = uniformBuf(device, [
        i32(1),             // B
        i32(PHI3.MAX_PAGES), // max_num_pages
        i32(nnzPages),      // nnz_pages
        i32(0),             // pages_elem_offset
        i32(0),             // page_indptr_elem_offset
        i32(0),             // page_values_elem_offset
        i32(0),             // length_info_elem_offset
        f32(SM_SCALE),      // sm_scale
        u32(1),             // packGridDimX
      ])
      dispatch(enc, P.attention, bg(device, P.attention, [
        B.qOut, B.pageIndptr, B.pageValues, lw.kvPages,
        B.lengthInfo, B.attnOut, attnUniform
      ]), 1, PHI3.HEADS)

      // O projection: attnOut → hidden2 (reuse for temp)
      dispatch(enc, P.oProjMatmul, bg(device, P.oProjMatmul, [
        B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjUniform
      ]), 3072)

      // AddNorm1: residual + oproj → new residual, normed for FFN
      const addNormUniform = uniformBuf(device, [u32(1)])
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, resIn, lw.normGamma2, normIn, B.residual, addNormUniform
      ]), 1)

      // Fused FFN: normIn → ffnOut (gate+up matmul + SiLU)
      dispatch(enc, P.fusedFfn, bg(device, P.fusedFfn, [
        B.ffnOut, normIn, lw.ffnScales, lw.ffnWeights, fusedFfnUniform
      ]), 8192)

      // FFN down: ffnOut → hidden2
      dispatch(enc, P.ffnDownMatmul, bg(device, P.ffnDownMatmul, [
        B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDownUniform
      ]), 3072)

      // AddNorm2: residual + ffndown → new residual, normed for next layer
      const addNorm2Uniform = uniformBuf(device, [u32(1)])
      const nextNormGamma = L < PHI3.LAYERS - 1 ? layerWeights[L + 1].normGamma1 : B.initNormGamma
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, B.residual, nextNormGamma, normIn, resIn, addNorm2Uniform
      ]), 1)

      // Ping-pong: for next layer, normIn and resIn swap roles
      const tmp = normIn
      normIn = resIn
      resIn = tmp
    }

    // --- FINAL NORM + LM HEAD ---
    // After 32 layers, normIn has the normed final hidden state
    dispatch(enc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.hidden1, normIn, B.initNormGamma, normUniform
    ]), 1)

    // LM head matmul: hidden1 → logits (f32)
    dispatch(enc, P.lmHead, bg(device, P.lmHead, [
      B.logits, B.hidden1, B.lmHeadScales, B.lmHeadWeights, lmHeadUniform
    ]), 32064)

    // Argmax: logits → tokenResult
    dispatch(enc, P.argmax, bg(device, P.argmax, [
      B.logits, B.tokenResult, argmaxUniform
    ]), 1)

    // Submit all dispatches in one command buffer
    device.queue.submit([enc.finish()])

    // Readback token from GPU
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const readEnc = device.createCommandEncoder()
    readEnc.copyBufferToBuffer(B.tokenResult, 0, readBuf, 0, 4)
    device.queue.submit([readEnc.finish()])

    await readBuf.mapAsync(GPUMapMode.READ)
    const result = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    readBuf.destroy()

    return result
  }

  const STOP_TOKENS = new Set([2, 32000, 32007])

  async function generate(
    promptTokenIds: number[],
    maxTokens: number,
    onToken?: (id: number) => void
  ): Promise<number[]> {
    const tokens: number[] = []

    // TODO: proper prefill with batch matmul
    // For now: sequential prefill using decode (one token at a time)
    console.log(`[runtime] Prefilling ${promptTokenIds.length} tokens...`)
    for (let i = 0; i < promptTokenIds.length; i++) {
      await decodeToken(promptTokenIds[i], i)
    }

    // Decode loop
    console.log(`[runtime] Decoding up to ${maxTokens} tokens...`)
    let pos = promptTokenIds.length
    let lastToken = promptTokenIds[promptTokenIds.length - 1]

    // Get first generated token
    lastToken = await decodeToken(lastToken, pos - 1)

    for (let i = 0; i < maxTokens; i++) {
      if (lastToken < 0 || lastToken >= PHI3.VOCAB || STOP_TOKENS.has(lastToken)) break

      tokens.push(lastToken)
      onToken?.(lastToken)

      pos++
      lastToken = await decodeToken(lastToken, pos)
    }

    console.log(`[runtime] Generated ${tokens.length} tokens`)
    return tokens
  }

  return { decodeToken, generate }
}
