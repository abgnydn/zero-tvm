/**
 * MODEL SPEC — single source of truth for a model's shape and identity.
 *
 * A ModelSpec carries the base dimensions plus every derived product the
 * engine, the shader prelude, and the weight loader need, so no file
 * re-derives (or hardcodes) address math. `makeModelSpec()` computes the
 * derived fields once and freezes the object.
 *
 * PHI3 reproduces the historical constants exactly (for Phi-3
 * heads == kvHeads and qDim == d, so e.g. qkvDim = 3*d = 9216 still holds
 * under the general GQA derivation qDim + 2*kvDim).
 *
 * QWEN3_4B exists so later phases can compile/run against GQA dims; no
 * kernel consumes it yet.
 *
 * This module is dependency-free on purpose: tests/kernels/*.mjs import it
 * directly under Node (type stripping), where Vite `?raw` imports are
 * unavailable. Keep it to erasable-TS syntax (interfaces + plain functions).
 */

// ============================================================
// Weight-file parameter naming (MLC ndarray-cache record names)
// ============================================================

export interface LayerParamNames {
  qkvWeight: string[]
  qkvScale: string[]
  oProjWeight: string[]
  oProjScale: string[]
  norm1: string[]        // input_layernorm (pre-attention)
  norm2: string[]        // post_attention_layernorm (pre-FFN)
  ffnWeight: string[]    // gate_up fused
  ffnScale: string[]
  ffnDownWeight: string[]
  ffnDownScale: string[]
  qNorm?: string[]       // per-head RMSNorm gamma over head_dim (Qwen3)
  kNorm?: string[]
}

/** Candidate-name tables for each weight the engine binds. Each entry is an
 *  ordered candidate list (first match wins) — same contract as the old
 *  weight-loader `find()` calls. */
export interface ParamNaming {
  embedWeight: string[]
  embedScale: string[]
  lmHeadWeight: string[]   // for tiedEmbeddings models, same records as embed*
  lmHeadScale: string[]
  finalNorm: string[]
  layer: (L: number) => LayerParamNames
}

// ============================================================
// ModelSpec
// ============================================================

export interface ModelSpecBase {
  /** Short stable id — used for per-model cache-dir suffixes. */
  id: string
  d: number          // hidden dim
  layers: number
  heads: number      // query heads
  kvHeads: number    // KV heads (== heads for MHA, < heads for GQA)
  headDim: number
  ffn: number        // FFN intermediate dim (gate→up row stride)
  vocab: number
  pageSize: number   // KV-cache slots per page
  maxPages: number   // KV-cache page budget
  maxSeq: number     // model's trained context length
  ropeTheta: number
  rmsEps: number
  tiedEmbeddings: boolean  // lm_head reuses the quantized embedding matrix
  qkNorm: boolean          // per-head q_norm/k_norm RMSNorm before RoPE
  stops: readonly number[] // stop token ids for the decode loops
  chatTemplateId: 'phi3' | 'chatml'
  hfRepo: string           // HuggingFace repo with the MLC q4f16_1 layout
  paramNaming: ParamNaming
}

export interface ModelSpec extends ModelSpecBase {
  // Derived — computed by makeModelSpec, never set by hand.
  qDim: number          // heads * headDim (== d for Phi-3, ≠ d for Qwen3)
  kvDim: number         // kvHeads * headDim
  qkvDim: number        // qDim + 2*kvDim (fused QKV projection rows)
  gqaGroup: number      // heads / kvHeads
  halfHeadDim: number   // headDim / 2 (RoPE pair distance)
  dPacked: number       // d / 8  (u32 words per K=d weight row)
  dScales: number       // d / 32 (int4 scales per K=d weight row)
  qkvGroupPairs: number // heads * halfHeadDim (RoPE pairs in the Q group)
  qkvPairs: number      // qkvDim / 2 (total QKV pairs — the qkv_fused grid)
  headPageStride: number // pageSize * headDim (f16 per head per page)
  kvPageStride: number   // kvHeads * pageSize * headDim * 2 (f16 per page, K+V)
  vPageOffset: number    // kvHeads * pageSize * headDim (V region start in a page)
  maxContext: number     // maxPages * pageSize (hard KV ceiling in tokens)
  // int8-KV layout products
  kvI8RowWords: number     // headDim / 4 (u32 words per (head,slot,side) row)
  kvI8SlotWords: number    // 2 * kvI8RowWords (K+V)
  kvI8HeadWords: number    // pageSize * kvI8SlotWords
  kvI8PageWords: number    // kvHeads * kvI8HeadWords
  kvScalesPerSlot: number  // K + V
  kvScalesPerHead: number  // pageSize * kvScalesPerSlot
  kvScalesPerPage: number  // kvHeads * kvScalesPerHead
}

export function makeModelSpec(base: ModelSpecBase): ModelSpec {
  if (base.heads % base.kvHeads !== 0) throw new Error(`${base.id}: heads not divisible by kvHeads`)
  if (base.headDim % 2 !== 0) throw new Error(`${base.id}: headDim must be even (RoPE pairs)`)
  if (base.d % 32 !== 0) throw new Error(`${base.id}: d must be divisible by 32 (int4 scale groups)`)
  const qDim = base.heads * base.headDim
  const kvDim = base.kvHeads * base.headDim
  const kvI8RowWords = base.headDim / 4
  return Object.freeze({
    ...base,
    qDim,
    kvDim,
    qkvDim: qDim + 2 * kvDim,
    gqaGroup: base.heads / base.kvHeads,
    halfHeadDim: base.headDim / 2,
    dPacked: base.d / 8,
    dScales: base.d / 32,
    qkvGroupPairs: base.heads * (base.headDim / 2),
    qkvPairs: (qDim + 2 * kvDim) / 2,
    headPageStride: base.pageSize * base.headDim,
    kvPageStride: 2 * base.kvHeads * base.pageSize * base.headDim,
    vPageOffset: base.kvHeads * base.pageSize * base.headDim,
    maxContext: base.maxPages * base.pageSize,
    kvI8RowWords,
    kvI8SlotWords: 2 * kvI8RowWords,
    kvI8HeadWords: base.pageSize * 2 * kvI8RowWords,
    kvI8PageWords: base.kvHeads * base.pageSize * 2 * kvI8RowWords,
    kvScalesPerSlot: 2,
    kvScalesPerHead: base.pageSize * 2,
    kvScalesPerPage: base.kvHeads * base.pageSize * 2,
  })
}

// ============================================================
// Phi-3-mini — values reproduce the historical PHI3 constants exactly
// ============================================================

export const PHI3: ModelSpec = makeModelSpec({
  id: 'phi3-mini',
  d: 3072,
  layers: 32,
  heads: 32,
  kvHeads: 32,   // MHA: heads == kvHeads
  headDim: 96,
  ffn: 8192,
  vocab: 32064,
  pageSize: 16,
  maxPages: 257,
  maxSeq: 4096,
  ropeTheta: 10000,
  rmsEps: 1e-5,
  tiedEmbeddings: false,
  qkNorm: false,
  stops: [2, 32000, 32007],  // </s>, <|endoftext|>, <|end|>
  chatTemplateId: 'phi3',
  hfRepo: 'mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC',
  paramNaming: {
    embedWeight: ['transformer.embd.q_weight', 'embed_tokens.q_weight', 'model.embed_tokens.q_weight'],
    embedScale: ['transformer.embd.q_scale', 'embed_tokens.q_scale', 'model.embed_tokens.q_scale'],
    lmHeadWeight: ['lm_head.q_weight', 'model.lm_head.q_weight'],
    lmHeadScale: ['lm_head.q_scale', 'model.lm_head.q_scale'],
    finalNorm: ['transformer.norm.weight', 'model.norm.weight', 'norm.weight'],
    layer: (L: number) => {
      const h = `transformer.h.${L}`   // MLC prefix
      const p = `model.layers.${L}`    // HF prefix fallback
      return {
        qkvWeight: [`${h}.mixer.qkv_proj.q_weight`, `${p}.self_attn.qkv_proj.q_weight`],
        qkvScale: [`${h}.mixer.qkv_proj.q_scale`, `${p}.self_attn.qkv_proj.q_scale`],
        oProjWeight: [`${h}.mixer.out_proj.q_weight`, `${p}.self_attn.o_proj.q_weight`],
        oProjScale: [`${h}.mixer.out_proj.q_scale`, `${p}.self_attn.o_proj.q_scale`],
        norm1: [`${h}.ln.weight`, `${p}.input_layernorm.weight`],
        norm2: [`${h}.post_attention_layernorm.weight`, `${p}.post_attention_layernorm.weight`],
        ffnWeight: [`${h}.mlp.gate_up_proj.q_weight`, `${p}.mlp.gate_up_proj.q_weight`],
        ffnScale: [`${h}.mlp.gate_up_proj.q_scale`, `${p}.mlp.gate_up_proj.q_scale`],
        ffnDownWeight: [`${h}.mlp.down_proj.q_weight`, `${p}.mlp.down_proj.q_weight`],
        ffnDownScale: [`${h}.mlp.down_proj.q_scale`, `${p}.mlp.down_proj.q_scale`],
      }
    },
  },
})

// ============================================================
// Qwen3-4B (q4f16_1) — GQA 4:1, qDim ≠ d, tied embeddings, q/k-norm.
// No kernel consumes this yet: it exists so the GQA port can compile and
// size against real dims.
// ============================================================

export const QWEN3_4B: ModelSpec = makeModelSpec({
  id: 'qwen3-4b',
  d: 2560,
  layers: 36,
  heads: 32,
  kvHeads: 8,      // GQA 4:1
  headDim: 128,    // qDim = 4096 ≠ d
  ffn: 9728,       // gate_up fused rows = 19456
  vocab: 151936,
  pageSize: 16,
  maxPages: 257,   // same page budget as Phi-3 for now (model supports 40960)
  maxSeq: 40960,
  ropeTheta: 1e6,
  rmsEps: 1e-6,
  tiedEmbeddings: true,   // lm_head = quantized embed matrix
  qkNorm: true,           // per-head RMSNorm over head_dim before RoPE
  stops: [151645, 151643], // <|im_end|>, <|endoftext|>
  chatTemplateId: 'chatml',
  hfRepo: 'mlc-ai/Qwen3-4B-q4f16_1-MLC',
  paramNaming: {
    embedWeight: ['model.embed_tokens.q_weight'],
    embedScale: ['model.embed_tokens.q_scale'],
    // Tied embeddings: MLC ships no separate lm_head records.
    lmHeadWeight: ['model.embed_tokens.q_weight'],
    lmHeadScale: ['model.embed_tokens.q_scale'],
    finalNorm: ['model.norm.weight'],
    layer: (L: number) => {
      const p = `model.layers.${L}`
      return {
        qkvWeight: [`${p}.self_attn.c_attn.q_weight`, `${p}.self_attn.qkv_proj.q_weight`],
        qkvScale: [`${p}.self_attn.c_attn.q_scale`, `${p}.self_attn.qkv_proj.q_scale`],
        oProjWeight: [`${p}.self_attn.o_proj.q_weight`],
        oProjScale: [`${p}.self_attn.o_proj.q_scale`],
        norm1: [`${p}.input_layernorm.weight`],
        norm2: [`${p}.post_attention_layernorm.weight`],
        ffnWeight: [`${p}.mlp.gate_up_proj.q_weight`],
        ffnScale: [`${p}.mlp.gate_up_proj.q_scale`],
        ffnDownWeight: [`${p}.mlp.down_proj.q_weight`],
        ffnDownScale: [`${p}.mlp.down_proj.q_scale`],
        qNorm: [`${p}.self_attn.q_norm.weight`],
        kNorm: [`${p}.self_attn.k_norm.weight`],
      }
    },
  },
})
