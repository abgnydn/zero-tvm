/**
 * MODEL LOADER — loads Phi-3 weights from WebLLM's cache.
 *
 * Strategy: use WebLLM to download + cache the model,
 * then intercept the weight uploads to GPU buffers.
 * We steal TVM's weight buffers directly (they're already on GPU).
 *
 * Alternative: parse the cached model files ourselves.
 * For now, we use the interception approach since the model
 * is already cached from WebLLM.
 */

import { PHI3, compile, allocateLayerWeights, CompiledModel, LayerWeights } from './compiler.js'
import { createRuntime, Runtime } from './runtime.js'

// ============================================================
// Weight extraction from TVM's captured buffers
// ============================================================

/**
 * Maps TVM's buffer landscape to our compiler's weight layout.
 *
 * From the reverse engineering dump:
 *   Per layer (10 dispatches):
 *     [QKV]     @2: scales (1.77MB), @3: weights (14.16MB)
 *     [OProj]   @2: scales (589KB),  @3: weights (4.72MB)
 *     [AddNorm] @2: gamma (6.1KB)
 *     [FFNUp]   @2: scales (3.15MB), @3: weights (25.17MB)
 *     [FFNDown] @2: scales (1.57MB), @3: weights (12.58MB)
 *
 *   Global:
 *     BUF#0:  embedding output (6.29MB) — also initial norm gamma
 *     BUF#2:  embedding scales (6.16MB)
 *     BUF#3:  embedding weights (49.25MB)
 *     BUF#694: LM head scales (6.16MB)
 *     BUF#695: LM head weights (49.25MB)
 */

export interface CapturedWeights {
  device: GPUDevice

  // Global
  embdWeights: GPUBuffer
  embdScales: GPUBuffer
  lmHeadWeights: GPUBuffer
  lmHeadScales: GPUBuffer
  initNormGamma: GPUBuffer

  // Per-layer
  layers: Array<{
    qkvWeights: GPUBuffer
    qkvScales: GPUBuffer
    oProjWeights: GPUBuffer
    oProjScales: GPUBuffer
    normGamma1: GPUBuffer   // attention norm
    normGamma2: GPUBuffer   // FFN norm
    ffnWeights: GPUBuffer
    ffnScales: GPUBuffer
    ffnDownWeights: GPUBuffer
    ffnDownScales: GPUBuffer
  }>
}

/**
 * Extract weights from captured TVM dispatch data.
 * This reads the bind group entries from the captured dispatches
 * and maps buffers to our weight layout.
 */
export function extractWeightsFromCapture(cap: {
  dispatches: Array<{
    pipeline: GPUComputePipeline
    entries: Array<{ binding: number; buffer: GPUBuffer; offset: number; size: number }>
    workgroups: [number, number, number]
  }>
  prefillDispatches: typeof cap.dispatches
  device: GPUDevice
}): CapturedWeights {
  const dispatches = cap.dispatches
  const device = cap.device

  // Embedding: dispatch[0] — fused_dequantize_take1_kernel
  //   @0: output (6.29MB), @1: input_ids (4B), @2: scales, @3: weights
  const embdScales = dispatches[0].entries[2].buffer
  const embdWeights = dispatches[0].entries[3].buffer

  // Initial norm gamma: dispatch[1] — rms_norm2_kernel
  //   @0: output, @1: input (embedding output), @2: gamma
  const initNormGamma = dispatches[1].entries[2].buffer

  // Per-layer extraction (10 dispatches per layer, starting at dispatch[2])
  const layers: CapturedWeights['layers'] = []

  for (let L = 0; L < PHI3.LAYERS; L++) {
    const base = 2 + L * 10

    // [0] QKV matmul: @2=scales, @3=weights
    const qkvScales = dispatches[base + 0].entries[2].buffer
    const qkvWeights = dispatches[base + 0].entries[3].buffer

    // [4] KV append: @1=KV pages (50.53MB) — but we need it from attention
    // [3] Attention (batch_decode_paged_kv): @7=KV pages
    // Actually for our layout, we'll create new KV pages

    // [4] OProj: @2=scales (589KB), @3=weights (4.72MB)
    const oProjScales = dispatches[base + 4].entries[2].buffer
    const oProjWeights = dispatches[base + 4].entries[3].buffer

    // [5] AddNorm1: @2=gamma
    const normGamma1 = dispatches[base + 5].entries[2].buffer

    // [6] FFNUp: @2=scales (3.15MB), @3=weights (25.17MB)
    const ffnScales = dispatches[base + 6].entries[2].buffer
    const ffnWeights = dispatches[base + 6].entries[3].buffer

    // [8] FFNDown: @2=scales (1.57MB), @3=weights (12.58MB)
    const ffnDownScales = dispatches[base + 8].entries[2].buffer
    const ffnDownWeights = dispatches[base + 8].entries[3].buffer

    // [9] AddNorm2: @2=gamma
    const normGamma2 = dispatches[base + 9].entries[2].buffer

    layers.push({
      qkvWeights, qkvScales,
      oProjWeights, oProjScales,
      normGamma1, normGamma2,
      ffnWeights, ffnScales,
      ffnDownWeights, ffnDownScales,
    })
  }

  // LM head: dispatch[322] — fused_dequantize5_fused_NT_matmul14_cast2_kernel
  //   @1=scales (6.16MB), @2=weights (49.25MB)
  const tailStart = 2 + PHI3.LAYERS * 10  // dispatch index 322
  const lmHeadScales = dispatches[tailStart].entries[1].buffer
  const lmHeadWeights = dispatches[tailStart].entries[2].buffer

  console.log(`[model] Extracted weights: ${layers.length} layers + embedding + LM head`)

  return {
    device,
    embdWeights, embdScales,
    lmHeadWeights, lmHeadScales,
    initNormGamma,
    layers,
  }
}

/**
 * Build a complete CompiledModel from captured TVM data.
 * Uses TVM's weight buffers directly (zero copy).
 * Creates our own activation/uniform buffers and pipelines.
 */
export function buildFromCapture(cap: {
  dispatches: Array<{
    pipeline: GPUComputePipeline
    entries: Array<{ binding: number; buffer: GPUBuffer; offset: number; size: number }>
    workgroups: [number, number, number]
  }>
  prefillDispatches: typeof cap.dispatches
  device: GPUDevice
}): { model: CompiledModel; runtime: Runtime } {
  const weights = extractWeightsFromCapture(cap)
  const { pipelines, buffers } = compile(weights.device)

  // Point shared buffers to TVM's weight buffers (zero copy)
  buffers.embdWeights = weights.embdWeights
  buffers.embdScales = weights.embdScales
  buffers.lmHeadWeights = weights.lmHeadWeights
  buffers.lmHeadScales = weights.lmHeadScales
  buffers.initNormGamma = weights.initNormGamma

  // Build per-layer weight references (zero copy from TVM)
  const layerWeights: LayerWeights[] = weights.layers.map((lw) => ({
    qkvWeights: lw.qkvWeights,
    qkvScales: lw.qkvScales,
    oProjWeights: lw.oProjWeights,
    oProjScales: lw.oProjScales,
    normGamma1: lw.normGamma1,
    normGamma2: lw.normGamma2,
    ffnWeights: lw.ffnWeights,
    ffnScales: lw.ffnScales,
    ffnDownWeights: lw.ffnDownWeights,
    ffnDownScales: lw.ffnDownScales,
    kvPages: allocateLayerWeights(weights.device).kvPages,  // fresh KV pages
  }))

  const model: CompiledModel = {
    device: weights.device,
    pipelines,
    buffers,
    layerWeights,
  }

  const runtime = createRuntime(model)

  console.log(`[model] Built model: ${PHI3.LAYERS} layers, 10 shaders, ~100 dispatches/token`)

  return { model, runtime }
}
