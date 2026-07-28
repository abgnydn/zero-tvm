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
  // GatedDeltaNet (Qwen3.5 linear_attn) records — present only on specs with
  // `gdn`; a layer resolves either the self_attn fields above or these,
  // according to layerKinds[L].
  gdnQkvWeight?: string[]   // in_proj_qkv [2*gdnKDim+gdnVDim, d] int4
  gdnQkvScale?: string[]
  gdnZWeight?: string[]     // in_proj_z (output gate) [gdnVDim, d] int4
  gdnZScale?: string[]
  gdnAWeight?: string[]     // in_proj_a (decay input) [vHeads, d] int4
  gdnAScale?: string[]
  gdnBWeight?: string[]     // in_proj_b (beta input) [vHeads, d] int4
  gdnBScale?: string[]
  gdnALog?: string[]        // A_log [vHeads] f32
  gdnDtBias?: string[]      // dt_bias [vHeads] f32
  gdnConvWeight?: string[]  // conv1d_weight [conv dim, 1, convK] f16
  gdnNormWeight?: string[]  // gated RMSNorm gamma [headV] f16 (no +1 offset)
  gdnOutWeight?: string[]   // out_proj [d, gdnVDim] int4
  gdnOutScale?: string[]
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

/** Gated-DeltaNet (linear attention) dimensions — Qwen3.5 `linear_attn`. */
export interface GdnDims {
  kHeads: number   // linear_num_key_heads (Q and K share these heads)
  vHeads: number   // linear_num_value_heads (state + gates are per v-head)
  headK: number    // linear_key_head_dim
  headV: number    // linear_value_head_dim
  convK: number    // linear_conv_kernel_dim (causal depthwise conv width)
}

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
  tokenizerKind: 'spm' | 'byteLevel'  // which tokenizer pipeline tokenizer.json needs
  hfRepo: string           // HuggingFace repo with the MLC q4f16_1 layout
  /** Weight-manifest filename in the repo. Older MLC repos ship
   *  ndarray-cache.json (the default); repos built with newer MLC ship the
   *  renamed tensor-cache.json (Qwen3.5). */
  manifestName?: string
  paramNaming: ParamNaming
  // ── Qwen3.5 hybrid-architecture fields (all optional; absent = pure attention)
  /** Every Nth layer is full attention, the rest are GDN. HF layer_types:
   *  layer i is attention iff (i+1) % interval == 0 (Qwen3.5: 4 → layers
   *  3,7,…,31 attention, verified against tensor-cache.json self_attn records).
   *  Requires `gdn` when set. */
  fullAttnInterval?: number
  /** GatedDeltaNet dims (Qwen3.5 linear_attn layers). */
  gdn?: GdnDims
  /** Fraction of headDim rotated by RoPE (Qwen3.5: 0.25 → 64 of 256 dims).
   *  Default 1 (full rotation). */
  partialRotaryFactor?: number
  /** Gated attention (Qwen3.5): the fused attention projection packs per-head
   *  [Q|gate] pairs before K and V — cAttnDim = 2*qDim + 2*kvDim — and
   *  sigmoid(gate) multiplies the attention output before o_proj. */
  attnGate?: boolean
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
  // Qwen3.5 hybrid derivations. For specs without gdn/attnGate these collapse
  // to the plain-attention values (rotaryDim == headDim, cAttnDim == qkvDim,
  // layerKinds all 'attn', gdn* mirror the attention dims) so the shader
  // prelude can emit every const unconditionally and all kernels compile
  // under every spec.
  layerKinds: ReadonlyArray<'gdn' | 'attn'>  // per-layer block kind for the engine
  rotaryDim: number     // headDim * partialRotaryFactor (RoPE-rotated dims per head)
  halfRotary: number    // rotaryDim / 2 (RoPE pair distance in the rotary slice)
  cAttnDim: number      // fused attention projection rows: qkvDim (+qDim gate rows when attnGate)
  gdnKHeads: number     // GDN key/query heads
  gdnVHeads: number     // GDN value heads (GVA: >= kHeads)
  gdnHeadK: number      // GDN key/query head dim
  gdnHeadV: number      // GDN value head dim
  gdnConvK: number      // causal-conv kernel width
  gdnGvaGroup: number   // vHeads / kHeads (v-heads sharing one k/q head)
  gdnKDim: number       // kHeads * headK (Q rows == K rows in in_proj_qkv)
  gdnVDim: number       // vHeads * headV (V rows; also z-gate / out_proj width)
  gdnQkvDim: number     // 2*gdnKDim + gdnVDim (in_proj_qkv rows == conv channels)
  gdnStatePerHead: number // headK * headV (f32 recurrent-state cells per v-head)
}

export function makeModelSpec(base: ModelSpecBase): ModelSpec {
  if (base.heads % base.kvHeads !== 0) throw new Error(`${base.id}: heads not divisible by kvHeads`)
  if (base.headDim % 2 !== 0) throw new Error(`${base.id}: headDim must be even (RoPE pairs)`)
  if (base.d % 32 !== 0) throw new Error(`${base.id}: d must be divisible by 32 (int4 scale groups)`)
  if (base.fullAttnInterval && !base.gdn)
    throw new Error(`${base.id}: fullAttnInterval requires gdn dims`)
  const qDim = base.heads * base.headDim
  const kvDim = base.kvHeads * base.headDim
  const kvI8RowWords = base.headDim / 4
  // GDN fallback for pure-attention specs: mirror the attention dims so the
  // prelude consts stay valid (nonzero) and every shader compiles; no GDN
  // kernel is ever dispatched for these specs (layerKinds is all 'attn').
  const g = base.gdn ?? {
    kHeads: base.heads,
    vHeads: base.heads,
    headK: base.headDim,
    headV: base.headDim,
    convK: 4,
  }
  if (g.vHeads % g.kHeads !== 0) throw new Error(`${base.id}: gdn vHeads not divisible by kHeads`)
  const interval = base.fullAttnInterval ?? 0
  // HF Qwen3_5Config: layer i is "linear_attention" iff (i+1) % interval != 0.
  const layerKinds = Object.freeze(
    Array.from({ length: base.layers }, (_, i): 'gdn' | 'attn' =>
      interval && (i + 1) % interval !== 0 ? 'gdn' : 'attn',
    ),
  )
  const rotaryDim = Math.round(base.headDim * (base.partialRotaryFactor ?? 1))
  if (rotaryDim % 2 !== 0) throw new Error(`${base.id}: rotaryDim must be even (RoPE pairs)`)
  return Object.freeze({
    ...base,
    layerKinds,
    rotaryDim,
    halfRotary: rotaryDim / 2,
    cAttnDim: qDim + 2 * kvDim + (base.attnGate ? qDim : 0),
    gdnKHeads: g.kHeads,
    gdnVHeads: g.vHeads,
    gdnHeadK: g.headK,
    gdnHeadV: g.headV,
    gdnConvK: g.convK,
    gdnGvaGroup: g.vHeads / g.kHeads,
    gdnKDim: g.kHeads * g.headK,
    gdnVDim: g.vHeads * g.headV,
    gdnQkvDim: 2 * g.kHeads * g.headK + g.vHeads * g.headV,
    gdnStatePerHead: g.headK * g.headV,
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
  tokenizerKind: 'spm',
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
  tokenizerKind: 'byteLevel',
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

// ============================================================
// Qwen3.5-4B (q4f16_1) — hybrid GatedDeltaNet + gated attention.
//
// Verified against mlc-ai/Qwen3.5-4B-q4f16_1-MLC (mlc-chat-config.json +
// tensor-cache.json) and the mlc-llm qwen35 model/loader sources:
//   - 32 layers, full_attention_interval=4: layers 3,7,…,31 are self_attn
//     (8 attention layers), the other 24 are linear_attn (GDN).
//   - GDN: in_proj_qkv [8192,d] rows = Q(16×128) | K(16×128) | V(32×128)
//     (HF Qwen3_5GatedDeltaNet splits [key_dim, key_dim, value_dim]);
//     conv1d_weight [8192,1,4]; in_proj_z [4096,d]; in_proj_a/in_proj_b
//     [32,d]; A_log/dt_bias [32] f32; norm.weight [128]; out_proj [d,4096].
//   - Attention: c_attn [10240,d] = q_proj(16 heads × [256 Q | 256 gate]
//     interleaved per head) ‖ K(4×256) ‖ V(4×256) (qwen35_loader.py:
//     concat(q_proj, k_proj, v_proj)); partial RoPE rotary_dim=64, theta 1e7;
//     q/k_norm[256] (+1.0 offset pre-baked by the MLC loader); output gated
//     by sigmoid(gate) before o_proj.
//   - lm_head tied to the 248320-row embedding.
// The GDN kernel family is pinned against tests/kernels/compile-qwen35.mjs;
// the engine's hybrid path (engine-core.ts) dispatches per layerKinds[L].
// ============================================================

export const QWEN35_4B: ModelSpec = makeModelSpec({
  id: 'qwen35-4b',
  d: 2560,
  layers: 32,
  heads: 16,
  kvHeads: 4,      // GQA 4:1 on the 8 attention layers
  headDim: 256,
  ffn: 9216,       // gate_up fused rows = 18432
  vocab: 248320,
  pageSize: 16,
  maxPages: 257,   // same page budget as the other specs (model supports 262144)
  maxSeq: 262144,
  ropeTheta: 1e7,
  rmsEps: 1e-6,
  tiedEmbeddings: true,
  qkNorm: true,
  // <|im_end|>, <|endoftext|> — resolved from the repo's OWN tokenizer.json
  // added_tokens (Qwen3.5 renumbered the specials for the 248320 vocab:
  // <|endoftext|>=248044, <|im_start|>=248045, <|im_end|>=248046). The
  // mlc-chat-config stop_token_ids [151643, 151645] are stale Qwen3 ids that
  // map to ORDINARY BPE tokens in this vocab — do not use them.
  stops: [248046, 248044],
  chatTemplateId: 'chatml',
  tokenizerKind: 'byteLevel',
  hfRepo: 'mlc-ai/Qwen3.5-4B-q4f16_1-MLC',
  manifestName: 'tensor-cache.json',  // MLC renamed ndarray-cache.json; tensor-cache-b16.json also exists — ignore it
  fullAttnInterval: 4,
  gdn: { kHeads: 16, vHeads: 32, headK: 128, headV: 128, convK: 4 },
  partialRotaryFactor: 0.25,  // rotary_dim = 64 of 256 dims per head
  attnGate: true,
  paramNaming: {
    embedWeight: ['model.embed_tokens.q_weight'],
    embedScale: ['model.embed_tokens.q_scale'],
    // Tied embeddings: MLC ships no separate lm_head records.
    lmHeadWeight: ['model.embed_tokens.q_weight'],
    lmHeadScale: ['model.embed_tokens.q_scale'],
    finalNorm: ['model.norm.weight'],
    // Every layer gets both name sets; layerKinds[L] says which resolves
    // (self_attn records exist only on attention layers, linear_attn records
    // only on GDN layers — verified in tensor-cache.json).
    layer: (L: number) => {
      const p = `model.layers.${L}`
      const g = `${p}.linear_attn`
      return {
        qkvWeight: [`${p}.self_attn.c_attn.q_weight`],
        qkvScale: [`${p}.self_attn.c_attn.q_scale`],
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
        gdnQkvWeight: [`${g}.in_proj_qkv.q_weight`],
        gdnQkvScale: [`${g}.in_proj_qkv.q_scale`],
        gdnZWeight: [`${g}.in_proj_z.q_weight`],
        gdnZScale: [`${g}.in_proj_z.q_scale`],
        gdnAWeight: [`${g}.in_proj_a.q_weight`],
        gdnAScale: [`${g}.in_proj_a.q_scale`],
        gdnBWeight: [`${g}.in_proj_b.q_weight`],
        gdnBScale: [`${g}.in_proj_b.q_scale`],
        gdnALog: [`${g}.A_log`],
        gdnDtBias: [`${g}.dt_bias`],
        gdnConvWeight: [`${g}.conv1d_weight`],
        gdnNormWeight: [`${g}.norm.weight`],
        gdnOutWeight: [`${g}.out_proj.q_weight`],
        gdnOutScale: [`${g}.out_proj.q_scale`],
      }
    },
  },
})
