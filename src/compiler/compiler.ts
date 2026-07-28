/**
 * COMPILER — Creates GPU pipelines from our WGSL shaders.
 *
 * No TVM. We create every pipeline, buffer, and bind group ourselves.
 *
 * The model shape comes from a ModelSpec (src/compiler/model-spec.ts);
 * compile() defaults to PHI3:
 *   D=3072, HEADS=32, HEAD_DIM=96, LAYERS=32
 *   FFN=8192, VOCAB=32064
 *   Q4: group_size=32, zero_point=7
 *   KV cache: page_size=16, max_pages=257
 */

// Shader sources imported as strings by Vite; the int4_matmul family is
// generated (see shaders/int4_matmul.gen.ts). Every source is piped through
// withPrelude() so model-shape constants come from one place (shader-prelude.ts).
import { withPrelude, PHI3, type ModelSpec } from './shader-prelude'
import { int4MatmulWGSL } from './shaders/int4_matmul.gen'
import rmsNormSrc from './shaders/rms_norm.wgsl?raw'
import addNormSrc from './shaders/add_norm.wgsl?raw'
import ropeSrc from './shaders/rope.wgsl?raw'
import kvAppendSrc from './shaders/kv_append.wgsl?raw'
import qkvFusedSrc from './shaders/qkv_fused.wgsl?raw'
import qkvFusedSgSrc from './shaders/qkv_fused_sg.wgsl?raw'
import qkvFusedSgVec4Src from './shaders/qkv_fused_sg_vec4.wgsl?raw'
import qkvFusedTiledSgSrc from './shaders/qkv_fused_tiled_sg.wgsl?raw'
import qkvFusedTiled2SgSrc from './shaders/qkv_fused_tiled2sg.wgsl?raw'
import qkvFusedScratchSrc from './shaders/qkv_fused_scratch.wgsl?raw'
import kvQuantizeInt8Src from './shaders/kv_quantize_int8.wgsl?raw'
import attentionInt8Src from './shaders/attention_int8.wgsl?raw'
import attentionSrc from './shaders/attention.wgsl?raw'
import attentionSgSrc from './shaders/attention_sg.wgsl?raw'
import attentionSplitkSrc from './shaders/attention_splitk.wgsl?raw'
import attentionSplitkSgSrc from './shaders/attention_splitk_sg.wgsl?raw'
import attentionCombineSrc from './shaders/attention_combine.wgsl?raw'
import fusedFfnSrc from './shaders/fused_ffn.wgsl?raw'
import fusedFfnTiledSgSrc from './shaders/fused_ffn_tiled_sg.wgsl?raw'
import fusedFfnPrologueSrc from './shaders/fused_ffn_prologue.wgsl?raw'
import fusedFfnTiledSgPrologueSrc from './shaders/fused_ffn_tiled_sg_prologue.wgsl?raw'
import add3NormSrc from './shaders/add3_norm.wgsl?raw'
import embeddingSrc from './shaders/embedding.wgsl?raw'
import argmaxSrc from './shaders/argmax.wgsl?raw'
import argmaxSgSrc from './shaders/argmax_sg.wgsl?raw'

// ============================================================
// Constants
// ============================================================

// PHI3 / QWEN3_4B live in model-spec.ts (single source of truth for TS and
// WGSL, via shader-prelude.ts); re-exported here so existing importers keep
// working.
export { PHI3, QWEN3_4B } from './shader-prelude'
export type { ModelSpec } from './shader-prelude'

// ============================================================
// Types
// ============================================================

export interface Pipelines {
  embedding: GPUComputePipeline
  rmsNorm: GPUComputePipeline
  qkvMatmul: GPUComputePipeline      // int4 matmul, K=3072→9216
  int4Matmul: GPUComputePipeline     // alias to the scalar int4_matmul (shared across QKV/O/FFN-down)
  int4MatmulSg: GPUComputePipeline | null  // subgroup variant, null if feature absent
  int4MatmulTiled: GPUComputePipeline | null  // tiled subgroup variant (4 rows/WG)
  int4MatmulTiled8: GPUComputePipeline | null // wider tile (8 rows/WG)
  int4MatmulF32Sg: GPUComputePipeline | null  // subgroup variant of LM head matmul
  int4MatmulF32Tiled: GPUComputePipeline | null  // tiled subgroup variant of LM head matmul
  int4MatmulF32Tiled8: GPUComputePipeline | null // wider tile variant of LM head matmul
  int4MatmulBatchedM4: GPUComputePipeline | null // M=4 batched GEMM (for spec decode / prefill chunks)
  // vec4-load variants — default ON since 2026-07-25 (+7.1% with the qkv
  // sibling on M2 Max, BENCH.md); ?vec4=0 opts out:
  int4MatmulSgVec4: GPUComputePipeline | null       // vec4-load _sg variant
  int4MatmulTiledVec4: GPUComputePipeline | null    // vec4-load tiled variant (4 rows/WG)
  int4MatmulF32SgVec4: GPUComputePipeline | null    // vec4-load _sg LM-head variant
  int4MatmulF32TiledVec4: GPUComputePipeline | null // vec4-load tiled LM-head variant
  rope: GPUComputePipeline
  kvAppend: GPUComputePipeline
  qkvFused: GPUComputePipeline       // decode-path fusion: QKV matmul + RoPE + KV append
  qkvFusedSg: GPUComputePipeline | null  // subgroup variant of qkvFused
  qkvFusedSgVec4: GPUComputePipeline | null  // ?vec4qkv=1 vec4-load variant (32-thread)
  qkvFusedTiledSg: GPUComputePipeline | null  // tiled+subgroup variant (input cache, 4 pairs/WG)
  qkvFusedTiled2Sg: GPUComputePipeline | null  // 2-subgroup, 2-pair tile (keeps 64 threads/WG)
  qkvFusedScratch: GPUComputePipeline  // int8-KV variant: writes K,V to scratch f16
  kvQuantizeInt8: GPUComputePipeline   // quantize scratch K,V → int8 pages + f16 scales
  attentionInt8: GPUComputePipeline    // int8-KV variant of paged attention
  attention: GPUComputePipeline
  attentionSg: GPUComputePipeline | null  // subgroup variant; null if `subgroups` feature absent
  // ?splitk=N experiment (measured ~+3% at short context on M2 Max,
  // 2026-07-25; opt-in until a long-context A/B — BENCH.md):
  attentionSplitK: GPUComputePipeline     // split-K partial pass (tree reduction)
  attentionSplitKSg: GPUComputePipeline | null  // split-K partial pass (subgroupAdd)
  attentionCombine: GPUComputePipeline    // merges split-K partials per head
  oProjMatmul: GPUComputePipeline     // int4 matmul, K=3072→3072
  addNorm: GPUComputePipeline
  fusedFfn: GPUComputePipeline        // gate+up+SiLU fused
  fusedFfnTiledSg: GPUComputePipeline | null  // tiled subgroup variant (4 rows/WG)
  // ?fuseprologue=1 experiment (measured −13.7% on M2 Max, 2026-07-25 —
  // falsified there; kept for A/B on other GPUs — BENCH.md):
  fusedFfnPrologue: GPUComputePipeline          // add_norm folded into the FFN's phase 1
  fusedFfnTiledSgPrologue: GPUComputePipeline | null  // same, on the tiled_sg kernel
  add3Norm: GPUComputePipeline        // 3-way residual merge + norm (layer tail)
  ffnDownMatmul: GPUComputePipeline   // int4 matmul, K=8192→3072
  lmHead: GPUComputePipeline          // int4 matmul f32 output
  argmax: GPUComputePipeline
  argmaxSg: GPUComputePipeline | null // subgroup variant; null if `subgroups` feature absent
}

/** Per-layer weight buffers */
export interface LayerWeights {
  qkvWeights: GPUBuffer    // 14.16MB (9216 × 384 u32)
  qkvScales: GPUBuffer     // 1.77MB  (9216 × 96 f16)
  oProjWeights: GPUBuffer  // 4.72MB  (3072 × 384 u32)
  oProjScales: GPUBuffer   // 589KB   (3072 × 96 f16)
  normGamma1: GPUBuffer    // 6.1KB   (3072 f16) — attention norm
  normGamma2: GPUBuffer    // 6.1KB   (3072 f16) — FFN norm
  ffnWeights: GPUBuffer    // 25.17MB (16384 × 384 u32) — gate+up
  ffnScales: GPUBuffer     // 3.15MB  (16384 × 96 f16)
  ffnDownWeights: GPUBuffer // 12.58MB (3072 × 1024 u32)
  ffnDownScales: GPUBuffer  // 1.57MB  (3072 × 256 f16)
  kvPages: GPUBuffer        // 50.53MB per layer (257 pages × 98304 f16)
}

// ============================================================
// Compile
// ============================================================

function createPipelineFor(device: GPUDevice, src: string, entry: string, spec: ModelSpec): GPUComputePipeline {
  const module = device.createShaderModule({ code: withPrelude(src, spec) })
  return device.createComputePipeline({
    layout: 'auto',
    compute: { module, entryPoint: entry },
  })
}

function createBuf(device: GPUDevice, size: number, usage: number, label?: string): GPUBuffer {
  return device.createBuffer({
    size: Math.max(size, 4), // WebGPU minimum
    usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label,
  })
}

const STORAGE = GPUBufferUsage.STORAGE

export function compile(
  device: GPUDevice,
  opts: { subgroups?: boolean } = {},
  spec: ModelSpec = PHI3,
): { pipelines: Pipelines } {
  console.log(`[compiler] Creating pipelines (${spec.id})...`)
  const subgroups = !!opts.subgroups
  if (subgroups) console.log('[compiler] subgroups feature enabled — compiling _sg variants')

  // Bind createPipeline to this spec so the big literal below stays readable.
  const createPipeline = (device: GPUDevice, src: string, entry: string) =>
    createPipelineFor(device, src, entry, spec)

  const pipelines: Pipelines = {
    embedding: createPipeline(device, embeddingSrc, 'embedding'),
    rmsNorm: createPipeline(device, rmsNormSrc, 'rms_norm'),
    qkvMatmul: createPipeline(device, int4MatmulWGSL(), 'int4_matmul'),
    int4Matmul: createPipeline(device, int4MatmulWGSL(), 'int4_matmul'),
    int4MatmulSg: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true }), 'int4_matmul_sg') : null,
    int4MatmulTiled: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true, rowsPerWG: 4 }), 'int4_matmul_tiled') : null,
    int4MatmulTiled8: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true, rowsPerWG: 8 }), 'int4_matmul_tiled8') : null,
    int4MatmulF32Sg: subgroups ? createPipeline(device, int4MatmulWGSL({ outF32: true, subgroups: true }), 'int4_matmul_f32_sg') : null,
    int4MatmulF32Tiled: subgroups ? createPipeline(device, int4MatmulWGSL({ outF32: true, subgroups: true, rowsPerWG: 4 }), 'int4_matmul_f32_tiled') : null,
    int4MatmulF32Tiled8: subgroups ? createPipeline(device, int4MatmulWGSL({ outF32: true, subgroups: true, rowsPerWG: 8 }), 'int4_matmul_f32_tiled8') : null,
    int4MatmulBatchedM4: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, m: 4 }), 'int4_matmul_batched_m4') : null,
    int4MatmulSgVec4: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true, vec4: true }), 'int4_matmul_sg_vec4') : null,
    int4MatmulTiledVec4: subgroups ? createPipeline(device, int4MatmulWGSL({ subgroups: true, rowsPerWG: 4, vec4: true }), 'int4_matmul_tiled_vec4') : null,
    int4MatmulF32SgVec4: subgroups ? createPipeline(device, int4MatmulWGSL({ outF32: true, subgroups: true, vec4: true }), 'int4_matmul_f32_sg_vec4') : null,
    int4MatmulF32TiledVec4: subgroups ? createPipeline(device, int4MatmulWGSL({ outF32: true, subgroups: true, rowsPerWG: 4, vec4: true }), 'int4_matmul_f32_tiled_vec4') : null,
    rope: createPipeline(device, ropeSrc, 'rope_kernel'),
    kvAppend: createPipeline(device, kvAppendSrc, 'kv_append'),
    qkvFused: createPipeline(device, qkvFusedSrc, 'qkv_fused'),
    qkvFusedSg: subgroups ? createPipeline(device, qkvFusedSgSrc, 'qkv_fused_sg') : null,
    qkvFusedSgVec4: subgroups ? createPipeline(device, qkvFusedSgVec4Src, 'qkv_fused_sg_vec4') : null,
    qkvFusedTiledSg: subgroups ? createPipeline(device, qkvFusedTiledSgSrc, 'qkv_fused_tiled_sg') : null,
    qkvFusedTiled2Sg: subgroups ? createPipeline(device, qkvFusedTiled2SgSrc, 'qkv_fused_tiled2sg') : null,
    qkvFusedScratch: createPipeline(device, qkvFusedScratchSrc, 'qkv_fused_scratch'),
    kvQuantizeInt8: createPipeline(device, kvQuantizeInt8Src, 'kv_quantize_int8'),
    attentionInt8: createPipeline(device, attentionInt8Src, 'attention_int8'),
    attention: createPipeline(device, attentionSrc, 'attention'),
    attentionSg: subgroups ? createPipeline(device, attentionSgSrc, 'attention_sg') : null,
    attentionSplitK: createPipeline(device, attentionSplitkSrc, 'attention_splitk'),
    attentionSplitKSg: subgroups ? createPipeline(device, attentionSplitkSgSrc, 'attention_splitk_sg') : null,
    attentionCombine: createPipeline(device, attentionCombineSrc, 'attention_combine'),
    oProjMatmul: createPipeline(device, int4MatmulWGSL(), 'int4_matmul'),
    addNorm: createPipeline(device, addNormSrc, 'add_norm'),
    fusedFfn: createPipeline(device, fusedFfnSrc, 'fused_ffn_kernel'),
    fusedFfnTiledSg: subgroups ? createPipeline(device, fusedFfnTiledSgSrc, 'fused_ffn_tiled_sg') : null,
    fusedFfnPrologue: createPipeline(device, fusedFfnPrologueSrc, 'fused_ffn_prologue'),
    fusedFfnTiledSgPrologue: subgroups ? createPipeline(device, fusedFfnTiledSgPrologueSrc, 'fused_ffn_tiled_sg_prologue') : null,
    add3Norm: createPipeline(device, add3NormSrc, 'add3_norm'),
    ffnDownMatmul: createPipeline(device, int4MatmulWGSL(), 'int4_matmul'),
    lmHead: createPipeline(device, int4MatmulWGSL({ outF32: true }), 'int4_matmul_f32'),
    argmax: createPipeline(device, argmaxSrc, 'argmax_kernel'),
    argmaxSg: subgroups ? createPipeline(device, argmaxSgSrc, 'argmax_sg') : null,
  }

  console.log(`[compiler] Done: ${Object.keys(pipelines).length} pipelines`)

  // Activation buffers are NOT allocated here — the engine
  // (src/zero-tvm/engine-core.ts) allocates its own, sized per mode
  // (fused/unfused, f16/int8 KV). compile() returns only the pipelines.
  return { pipelines }
}

/** Allocate per-layer weight buffers */
export function allocateLayerWeights(device: GPUDevice, spec: ModelSpec = PHI3): LayerWeights {
  const f16 = 2
  const S = spec
  return {
    // Weight rows are (out_dim × K/8) u32 words; scales are (out_dim × K/32) f16.
    // K = d for qkv/gate_up, K = qDim for o_proj, K = ffn for down_proj.
    qkvWeights: createBuf(device, S.qkvDim * S.dPacked * 4, STORAGE),
    qkvScales: createBuf(device, S.qkvDim * S.dScales * f16, STORAGE),
    oProjWeights: createBuf(device, S.d * (S.qDim / 8) * 4, STORAGE),
    oProjScales: createBuf(device, S.d * (S.qDim / 32) * f16, STORAGE),
    normGamma1: createBuf(device, S.d * f16, STORAGE),
    normGamma2: createBuf(device, S.d * f16, STORAGE),
    ffnWeights: createBuf(device, 2 * S.ffn * S.dPacked * 4, STORAGE),
    ffnScales: createBuf(device, 2 * S.ffn * S.dScales * f16, STORAGE),
    ffnDownWeights: createBuf(device, S.d * (S.ffn / 8) * 4, STORAGE),
    ffnDownScales: createBuf(device, S.d * (S.ffn / 32) * f16, STORAGE),
    kvPages: createBuf(device, S.maxPages * S.kvPageStride * f16, STORAGE),
  }
}
