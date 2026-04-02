/**
 * STANDALONE ENGINE — Our own WebGPU inference engine.
 *
 * Uses TVM's compiled WGSL shaders but creates its own:
 * - GPU buffers (activations, KV cache, uniforms)
 * - Bind groups (from captured pipeline layouts)
 * - Token loop (dispatch, read, repeat)
 *
 * No TVM runtime in the hot path. No shared buffers. No interleaving issues.
 */

import { CaptureResult, CapturedPipeline } from './capture.js'

// ============================================================
// Config
// ============================================================

export const PHI3 = {
  D: 3072,
  HEADS: 32,
  HEAD_DIM: 96,
  LAYERS: 32,
  FFN: 8192,
  VOCAB: 32064,
  PAGE_SIZE: 16,
  MAX_PAGES: 257,
  EPS: 1e-5,
} as const

// ============================================================
// Decode Token Sequence
// ============================================================

// Per-layer dispatch pattern (10 dispatches × 32 layers)
// from profiler: shader indices [6, 40, 26, 29, 46, 22, 47, 19, 27, 22]
// Entry points: QKV, RoPE, KV_append, Attention, O_proj, AddNorm, FFN_up, SiLU, FFN_down, AddNorm

interface LayerPipelines {
  qkv: CapturedPipeline        // fused_dequantize1_NT_matmul10
  rope: CapturedPipeline        // fused_rope
  kvAppend: CapturedPipeline    // tir_kv_cache_transpose_append
  attention: CapturedPipeline   // batch_decode_paged_kv
  oProj: CapturedPipeline       // fused_dequantize2_NT_matmul11
  addNorm: CapturedPipeline     // fuse_add_norm_decode
  ffnUp: CapturedPipeline       // fused_dequantize3_NT_matmul12
  silu: CapturedPipeline        // fused_split2_silu2_multiply2
  ffnDown: CapturedPipeline     // fused_dequantize4_NT_matmul13
}

// ============================================================
// Find pipelines by entry point name
// ============================================================

function findPipeline(pipelines: CapturedPipeline[], name: string): CapturedPipeline {
  const p = pipelines.find(p => p.entryPoint.includes(name))
  if (!p) throw new Error(`Pipeline not found: ${name}`)
  return p
}

function findLayerPipelines(pipelines: CapturedPipeline[]): LayerPipelines {
  return {
    qkv: findPipeline(pipelines, 'NT_matmul10'),
    rope: findPipeline(pipelines, 'fused_rope'),
    kvAppend: findPipeline(pipelines, 'kv_cache_transpose_append_kernel'),
    attention: findPipeline(pipelines, 'batch_decode_paged_kv_kernel'),
    oProj: findPipeline(pipelines, 'NT_matmul11'),
    addNorm: findPipeline(pipelines, 'fuse_add_norm_decode'),
    ffnUp: findPipeline(pipelines, 'NT_matmul12'),
    silu: findPipeline(pipelines, 'split2_silu2_multiply2'),
    ffnDown: findPipeline(pipelines, 'NT_matmul13'),
  }
}

// ============================================================
// Build the standalone engine
// ============================================================

export interface StandaloneEngine {
  generate(maxTokens: number, onToken?: (id: number) => void): Promise<number[]>
}

export function buildStandaloneEngine(capture: CaptureResult): StandaloneEngine {
  const { pipelines } = capture

  // Find the per-layer pipelines
  const layerPipelines = findLayerPipelines(pipelines)

  // Find preamble pipelines
  const embedPipeline = findPipeline(pipelines, 'dequantize_take1')
  const initialNormPipeline = findPipeline(pipelines, 'rms_norm2')

  // Find LM head pipeline
  const lmHeadPipeline = findPipeline(pipelines, 'NT_matmul14')

  console.log(`[standalone] Found pipelines:`)
  console.log(`  embed: ${embedPipeline.entryPoint}`)
  console.log(`  norm: ${initialNormPipeline.entryPoint}`)
  console.log(`  layer: ${Object.values(layerPipelines).map(p => p.entryPoint).join(', ')}`)
  console.log(`  lm_head: ${lmHeadPipeline.entryPoint}`)
  console.log(`  Total pipelines: ${pipelines.length}`)

  // Phase 2: Analyze buffer landscape from captured dispatches
  if (capture.dispatches.length > 0) {
    const allBuffers = new Map<GPUBuffer, { size: number; bindings: Set<string>; dispatchCount: number }>()

    for (const d of capture.dispatches) {
      const pName = capture.pipelines[d.pipelineIndex]?.entryPoint ?? '?'
      for (const e of d.entries) {
        let info = allBuffers.get(e.buffer)
        if (!info) {
          info = { size: e.buffer.size, bindings: new Set(), dispatchCount: 0 }
          allBuffers.set(e.buffer, info)
        }
        info.bindings.add(`${pName}@${e.binding}`)
        info.dispatchCount++
      }
    }

    // Classify by size
    const weights: GPUBuffer[] = []
    const activations: GPUBuffer[] = []
    const uniforms: GPUBuffer[] = []
    const kvCache: GPUBuffer[] = []

    for (const [buf, info] of allBuffers) {
      if (info.size > 1_000_000) weights.push(buf)         // >1MB = weights
      else if (info.size > 10_000) kvCache.push(buf)        // >10KB = KV cache pages
      else if (info.size <= 64) uniforms.push(buf)           // <=64B = uniforms
      else activations.push(buf)                              // 65B-10KB = activations
    }

    console.log(`[standalone] Buffer landscape:`)
    console.log(`  Weights: ${weights.length} (${weights.reduce((s, b) => s + b.size, 0) / 1e6 | 0}MB)`)
    console.log(`  KV cache: ${kvCache.length} (${kvCache.reduce((s, b) => s + b.size, 0) / 1e6 | 0}MB)`)
    console.log(`  Activations: ${activations.length} (${activations.reduce((s, b) => s + b.size, 0) / 1e3 | 0}KB)`)
    console.log(`  Uniforms: ${uniforms.length} (${uniforms.reduce((s, b) => s + b.size, 0)}B)`)
    console.log(`  Total unique buffers: ${allBuffers.size}`)
  } else {
    console.log(`[standalone] No dispatches captured yet — send a chat message first`)
  }

  return {
    async generate(_maxTokens: number, _onToken?: (id: number) => void): Promise<number[]> {
      throw new Error('Not implemented yet — need dispatch capture (send a chat first)')
    }
  }
}
