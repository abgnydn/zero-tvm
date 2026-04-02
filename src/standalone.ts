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

  // TODO Phase 2: Allocate own buffers
  // TODO Phase 3: Build bind groups using pipeline.getBindGroupLayout(0)
  // TODO Phase 4: Implement decode loop

  return {
    async generate(_maxTokens: number, _onToken?: (id: number) => void): Promise<number[]> {
      throw new Error('Not implemented yet — need Phase 2-4')
    }
  }
}
