/**
 * COMPILER — Creates GPU pipelines from our WGSL shaders.
 *
 * No TVM. We create every pipeline, buffer, and bind group ourselves.
 *
 * Phi-3 config:
 *   D=3072, HEADS=32, HEAD_DIM=96, LAYERS=32
 *   FFN=8192, VOCAB=32064
 *   Q4: group_size=32, zero_point=7
 *   KV cache: page_size=16, max_pages=257
 */

// Shader sources imported as strings by Vite
import int4MatmulSrc from './shaders/int4_matmul.wgsl?raw'
import int4MatmulF32Src from './shaders/int4_matmul_f32.wgsl?raw'
import rmsNormSrc from './shaders/rms_norm.wgsl?raw'
import addNormSrc from './shaders/add_norm.wgsl?raw'
import ropeSrc from './shaders/rope.wgsl?raw'
import kvAppendSrc from './shaders/kv_append.wgsl?raw'
import attentionSrc from './shaders/attention.wgsl?raw'
import fusedFfnSrc from './shaders/fused_ffn.wgsl?raw'
import embeddingSrc from './shaders/embedding.wgsl?raw'
import argmaxSrc from './shaders/argmax.wgsl?raw'

// ============================================================
// Constants
// ============================================================

export const PHI3 = {
  D: 3072,
  HEADS: 32,
  HEAD_DIM: 96,
  LAYERS: 32,
  FFN: 8192,
  VOCAB: 32064,
  QKV_DIM: 9216,     // 3 * 32 * 96
  PAGE_SIZE: 16,
  MAX_PAGES: 257,
  MAX_SEQ: 4096,
} as const

// ============================================================
// Types
// ============================================================

export interface CompiledModel {
  device: GPUDevice
  pipelines: Pipelines
  buffers: Buffers
  layerWeights: LayerWeights[]
}

interface Pipelines {
  embedding: GPUComputePipeline
  rmsNorm: GPUComputePipeline
  qkvMatmul: GPUComputePipeline      // int4 matmul, K=3072→9216
  rope: GPUComputePipeline
  kvAppend: GPUComputePipeline
  attention: GPUComputePipeline
  oProjMatmul: GPUComputePipeline     // int4 matmul, K=3072→3072
  addNorm: GPUComputePipeline
  fusedFfn: GPUComputePipeline        // gate+up+SiLU fused
  ffnDownMatmul: GPUComputePipeline   // int4 matmul, K=8192→3072
  lmHead: GPUComputePipeline          // int4 matmul f32 output
  argmax: GPUComputePipeline
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

/** Shared buffers (not per-layer) */
interface Buffers {
  // Activation scratch (ping-pong)
  hidden1: GPUBuffer       // 3072 f16 = 6KB
  hidden2: GPUBuffer       // 3072 f16 = 6KB
  residual: GPUBuffer      // 3072 f16 = 6KB

  // QKV / attention intermediates
  qkvOut: GPUBuffer        // 9216 f16 = 18KB
  qOut: GPUBuffer          // 3072 f16 = 6KB
  kOut: GPUBuffer          // 3072 f16 = 6KB
  vOut: GPUBuffer          // 3072 f16 = 6KB
  attnOut: GPUBuffer       // 3072 f16 = 6KB

  // FFN intermediates
  ffnOut: GPUBuffer        // 8192 f16 = 16KB (SiLU output)

  // Logits + sampling
  logits: GPUBuffer        // 32064 f32 = 125KB
  tokenResult: GPUBuffer   // 1 i32 = 4B

  // Embedding + position
  inputIds: GPUBuffer      // MAX_SEQ i32 = 16KB
  positionMap: GPUBuffer   // MAX_SEQ i32 = 16KB

  // KV cache page tables
  pageTable: GPUBuffer     // indptr + values
  pageIndptr: GPUBuffer    // (B+1) i32
  pageValues: GPUBuffer    // max_pages i32
  lengthInfo: GPUBuffer    // 3*B i32

  // Embedding weights
  embdWeights: GPUBuffer
  embdScales: GPUBuffer

  // LM head weights
  lmHeadWeights: GPUBuffer
  lmHeadScales: GPUBuffer

  // Initial norm gamma
  initNormGamma: GPUBuffer
}

// ============================================================
// Compile
// ============================================================

function createPipeline(device: GPUDevice, src: string, entry: string): GPUComputePipeline {
  const module = device.createShaderModule({ code: src })
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

export function compile(device: GPUDevice): { pipelines: Pipelines; buffers: Buffers } {
  console.log('[compiler] Creating pipelines...')

  const pipelines: Pipelines = {
    embedding: createPipeline(device, embeddingSrc, 'embedding'),
    rmsNorm: createPipeline(device, rmsNormSrc, 'rms_norm'),
    qkvMatmul: createPipeline(device, int4MatmulSrc, 'int4_matmul'),
    rope: createPipeline(device, ropeSrc, 'rope_kernel'),
    kvAppend: createPipeline(device, kvAppendSrc, 'kv_append'),
    attention: createPipeline(device, attentionSrc, 'attention'),
    oProjMatmul: createPipeline(device, int4MatmulSrc, 'int4_matmul'),
    addNorm: createPipeline(device, addNormSrc, 'add_norm'),
    fusedFfn: createPipeline(device, fusedFfnSrc, 'fused_ffn_kernel'),
    ffnDownMatmul: createPipeline(device, int4MatmulSrc, 'int4_matmul'),
    lmHead: createPipeline(device, int4MatmulF32Src, 'int4_matmul_f32'),
    argmax: createPipeline(device, argmaxSrc, 'argmax_kernel'),
  }

  console.log('[compiler] Allocating buffers...')

  const f16 = 2  // bytes per f16
  const D = PHI3.D

  const buffers: Buffers = {
    // Activations
    hidden1: createBuf(device, D * f16, STORAGE, 'hidden1'),
    hidden2: createBuf(device, D * f16, STORAGE, 'hidden2'),
    residual: createBuf(device, D * f16, STORAGE, 'residual'),

    qkvOut: createBuf(device, PHI3.QKV_DIM * f16, STORAGE, 'qkvOut'),
    qOut: createBuf(device, D * f16, STORAGE, 'qOut'),
    kOut: createBuf(device, D * f16, STORAGE, 'kOut'),
    vOut: createBuf(device, D * f16, STORAGE, 'vOut'),
    attnOut: createBuf(device, D * f16, STORAGE, 'attnOut'),

    ffnOut: createBuf(device, PHI3.FFN * f16, STORAGE, 'ffnOut'),

    logits: createBuf(device, PHI3.VOCAB * 4, STORAGE, 'logits'),
    tokenResult: createBuf(device, 4, STORAGE, 'tokenResult'),

    inputIds: createBuf(device, PHI3.MAX_SEQ * 4, STORAGE, 'inputIds'),
    positionMap: createBuf(device, PHI3.MAX_SEQ * 4, STORAGE, 'positionMap'),

    pageTable: createBuf(device, PHI3.MAX_PAGES * 4, STORAGE, 'pageTable'),
    pageIndptr: createBuf(device, 8, STORAGE, 'pageIndptr'),       // [0, nnz_pages]
    pageValues: createBuf(device, PHI3.MAX_PAGES * 4, STORAGE, 'pageValues'),
    lengthInfo: createBuf(device, 12, STORAGE, 'lengthInfo'),      // 3 × B=1

    // These will be populated by model.ts
    embdWeights: createBuf(device, 4, STORAGE, 'embdWeights_placeholder'),
    embdScales: createBuf(device, 4, STORAGE, 'embdScales_placeholder'),
    lmHeadWeights: createBuf(device, 4, STORAGE, 'lmHeadWeights_placeholder'),
    lmHeadScales: createBuf(device, 4, STORAGE, 'lmHeadScales_placeholder'),
    initNormGamma: createBuf(device, D * f16, STORAGE, 'initNormGamma'),
  }

  console.log(`[compiler] Done: ${Object.keys(pipelines).length} pipelines, ${Object.keys(buffers).length} buffers`)

  return { pipelines, buffers }
}

/** Allocate per-layer weight buffers */
export function allocateLayerWeights(device: GPUDevice): LayerWeights {
  const f16 = 2
  return {
    qkvWeights: createBuf(device, 9216 * 384 * 4, STORAGE),
    qkvScales: createBuf(device, 9216 * 96 * f16, STORAGE),
    oProjWeights: createBuf(device, 3072 * 384 * 4, STORAGE),
    oProjScales: createBuf(device, 3072 * 96 * f16, STORAGE),
    normGamma1: createBuf(device, 3072 * f16, STORAGE),
    normGamma2: createBuf(device, 3072 * f16, STORAGE),
    ffnWeights: createBuf(device, 16384 * 384 * 4, STORAGE),
    ffnScales: createBuf(device, 16384 * 96 * f16, STORAGE),
    ffnDownWeights: createBuf(device, 3072 * 1024 * 4, STORAGE),
    ffnDownScales: createBuf(device, 3072 * 256 * f16, STORAGE),
    kvPages: createBuf(device, 257 * 98304 * f16, STORAGE),
  }
}
