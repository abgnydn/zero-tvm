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
  /**
   * Affine (MLX) checkpoints carry a per-group BIAS next to every scale —
   * `w = scale·q + bias` — which the symmetric MLC layout has no analogue for.
   * Given a weight record name this returns its bias record.
   *
   * One function rather than a `*Bias` list beside every `*Weight`/`*Scale`
   * pair: MLX names are exactly `<base>.weight` / `.scales` / `.biases`, so
   * fourteen more fields would carry no information the suffix does not.
   * Absent = symmetric weights, no bias tensor.
   */
  biasFor?: (weightRecord: string) => string
}

/** Sparse-MoE FFN (Qwen3.6). Absent = the dense gate_up/down FFN.
 *
 *  Two facts the engine assumes rather than parameterises (add-model's
 *  constraint check enforces both from config.json):
 *    - the router is 8-bit quantized — moe_router_logits.wgsl hardcodes the
 *      8-bit unpack, there is no 4-bit router variant;
 *    - shared_expert_intermediate_size == moe_intermediate_size (`ffn`) —
 *      the shared expert is STACKED as expert index E of every expert
 *      tensor, which only works if its rows are the same width. */
export interface MoeDims {
  experts: number             // routed experts (num_experts)
  topK: number                // experts per token (num_experts_per_tok)
  normTopkProb: boolean       // renormalise the top-K probabilities to sum to 1
  /** Expert-stack precision. 3 selects the q3 bitstream kernels and a
   *  checkpoint whose switch_mlp/shared_expert were requantised to 3 bits
   *  (scripts/convert-q3-experts.py). Default 4. */
  bits?: 3 | 4
}

// ============================================================
// ModelSpec
// ============================================================

/** llama3-type RoPE frequency scaling (HF config `rope_scaling`,
 *  rope_type 'llama3' — Llama-3.1/3.2). The piecewise remapping itself lives
 *  in ropeInvFreqTable() below; kernels never see these numbers, only the
 *  precomputed table. yarn/longrope are different formulas and stay
 *  unsupported (constraints.ts keeps them red). */
export interface RopeScaling {
  ropeType: 'llama3'
  factor: number
  lowFreqFactor: number
  highFreqFactor: number
  originalMaxPositionEmbeddings: number
}

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
  /**
   * KV-cache page budget — the hard context ceiling is maxPages × pageSize.
   *
   * Sized per model against a ~1 GiB KV VRAM budget, NOT copied between
   * specs: `kvBytesPerToken` (below) varies ~12× across our three models, so
   * one page count buys wildly different context. The rule is
   * `maxPages = min(1 GiB / (kvBytesPerToken × pageSize), maxSeq / pageSize)`
   * — except Phi-3, which is already at its trained 4k window and would LOSE
   * a third of it for no user benefit if squeezed into 1 GiB (see PHI3).
   */
  maxPages: number
  maxSeq: number     // model's trained context length
  ropeTheta: number
  /** llama3 frequency remapping (Llama-3.1/3.2). Absent = plain RoPE. */
  ropeScaling?: RopeScaling
  rmsEps: number
  tiedEmbeddings: boolean  // lm_head reuses the quantized embedding matrix
  qkNorm: boolean          // per-head q_norm/k_norm RMSNorm before RoPE
  stops: readonly number[] // stop token ids for the decode loops
  chatTemplateId: 'phi3' | 'chatml' | 'llama3'
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
  /** Sparse-MoE FFN (Qwen3.6). When set, `ffn` is moe_intermediate_size — the
   *  per-expert width — and every layer runs the MoE block instead of the dense
   *  gate_up/down pair. */
  moe?: MoeDims
  /**
   * On-disk weight layout. 'mlc' is the ndarray/tensor-cache shard format with
   * symmetric q4f16_1 (group 32); 'mlx-safetensors' is a HuggingFace
   * safetensors checkpoint quantised MLX-affine (group 64, with biases).
   * Default 'mlc'.
   */
  weightFormat?: 'mlc' | 'mlx-safetensors'
  /**
   * MLX checkpoints only: the record-name prefix before `model.*` / `lm_head`.
   * '' for a text-only checkpoint (`model.layers.0…`, `lm_head.weight`);
   * 'language_model.' for Qwen3.6's multimodal root
   * (`language_model.model.layers.0…`). mlx-weights.ts derives every record
   * name from this — the loader has no other prefix knowledge.
   */
  mlxPrefix?: string
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
  /** f16 KV bytes ONE context position costs, summed over every attention
   *  layer: 2 (K+V) × kvHeads × headDim × 2 B × attnLayers. This is what sets
   *  the VRAM price of maxPages, and it differs by ~12× across our specs
   *  (Qwen3.5 keeps KV on 8 of 32 layers, Phi-3 on all 32 with MHA), so a
   *  shared page budget is never the right answer. `maxContext ×
   *  kvBytesPerToken` is the total KV allocation (halved by ?kv8=1). */
  kvBytesPerToken: number
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
  /** Fused GDN input projection: in_proj_qkv ‖ in_proj_z ‖ in_proj_a ‖
   *  in_proj_b rows concatenated into ONE K=d int4 matmul (Qwen3.5:
   *  8192+4096+32+32 = 12352). The loader packs the four weight/scale
   *  records into one buffer pair; the engine runs one dispatch and the
   *  downstream kernels read the regions at fixed offsets
   *  (z at gdnQkvDim, a at gdnQkvDim+gdnVDim, b right after a). */
  gdnProjRows: number
  // ── MoE derivations (1 / 0 / -1 for dense specs, so consumers can branch on
  // `spec.moe` and still read these unconditionally)
  /** Expert slots one token runs: the top-K routed experts PLUS the shared
   *  expert, which the loader stacks as expert index `sharedExpertIndex` so it
   *  is not a special case in any kernel. */
  moeSlots: number
  /** Index of the shared expert within every stacked expert tensor, and of its
   *  gate within the router — both are appended at the end, so both are
   *  `moe.experts`. -1 for dense specs. */
  sharedExpertIndex: number
}

/**
 * The standard MLX record-name table for a checkpoint under `prefix`
 * (= the spec's `mlxPrefix`: '' text-only, 'language_model.' multimodal).
 *
 * MLX names every quantized tensor `<base>.weight/.scales/.biases` and ships
 * projections unfused, so the whole table is mechanical — generated specs call
 * this instead of restating forty lines of names. The entries name the records
 * the engine binds by; concatenation (c_attn = q++k++v etc.) is
 * mlx-weights.ts's business, not the spec's. QWEN36_35B_A3B predates this
 * helper and spells the same names inline.
 */
export function mlxParamNaming(prefix: string): ParamNaming {
  const m = `${prefix}model.`
  return {
    embedWeight: [`${m}embed_tokens.weight`],
    embedScale: [`${m}embed_tokens.scales`],
    lmHeadWeight: [`${prefix}lm_head.weight`],
    lmHeadScale: [`${prefix}lm_head.scales`],
    finalNorm: [`${m}norm.weight`],
    biasFor: (w: string) => w.replace(/\.weight$/, '.biases'),
    layer: (L: number) => {
      const p = `${m}layers.${L}`
      const g = `${p}.linear_attn`
      return {
        qkvWeight: [`${p}.self_attn.q_proj.weight`],
        qkvScale: [`${p}.self_attn.q_proj.scales`],
        oProjWeight: [`${p}.self_attn.o_proj.weight`],
        oProjScale: [`${p}.self_attn.o_proj.scales`],
        norm1: [`${p}.input_layernorm.weight`],
        norm2: [`${p}.post_attention_layernorm.weight`],
        ffnWeight: [`${p}.mlp.gate_proj.weight`],
        ffnScale: [`${p}.mlp.gate_proj.scales`],
        ffnDownWeight: [`${p}.mlp.down_proj.weight`],
        ffnDownScale: [`${p}.mlp.down_proj.scales`],
        qNorm: [`${p}.self_attn.q_norm.weight`],
        kNorm: [`${p}.self_attn.k_norm.weight`],
        gdnQkvWeight: [`${g}.in_proj_qkv.weight`],
        gdnQkvScale: [`${g}.in_proj_qkv.scales`],
        gdnZWeight: [`${g}.in_proj_z.weight`],
        gdnZScale: [`${g}.in_proj_z.scales`],
        gdnAWeight: [`${g}.in_proj_a.weight`],
        gdnAScale: [`${g}.in_proj_a.scales`],
        gdnBWeight: [`${g}.in_proj_b.weight`],
        gdnBScale: [`${g}.in_proj_b.scales`],
        gdnALog: [`${g}.A_log`],
        gdnDtBias: [`${g}.dt_bias`],
        gdnConvWeight: [`${g}.conv1d.weight`],
        gdnNormWeight: [`${g}.norm.weight`],
        gdnOutWeight: [`${g}.out_proj.weight`],
        gdnOutScale: [`${g}.out_proj.scales`],
      }
    },
  }
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
    gdnProjRows: (2 * g.kHeads * g.headK + g.vHeads * g.headV) + g.vHeads * g.headV + 2 * g.vHeads,
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
    kvBytesPerToken: 2 * base.kvHeads * base.headDim * 2
      * layerKinds.filter((k) => k === 'attn').length,
    kvI8RowWords,
    kvI8SlotWords: 2 * kvI8RowWords,
    kvI8HeadWords: base.pageSize * 2 * kvI8RowWords,
    kvI8PageWords: base.kvHeads * base.pageSize * 2 * kvI8RowWords,
    kvScalesPerSlot: 2,
    kvScalesPerHead: base.pageSize * 2,
    kvScalesPerPage: base.kvHeads * base.pageSize * 2,
    moeSlots: base.moe ? base.moe.topK + 1 : 1,
    sharedExpertIndex: base.moe ? base.moe.experts : -1,
  })
}

// ============================================================
// RoPE inverse-frequency table
// ============================================================

/**
 * Per-pair RoPE inverse frequencies — the f32 table rope.wgsl binds
 * (binding 6, HALF_ROTARY entries). Computed on the CPU so llama3-style
 * rope_scaling is a table swap, not a kernel variant: for plain specs
 * entry i is theta^(-2i/rotaryDim) (what the kernel used to compute with
 * pow() inline); with `ropeScaling` set, the llama3 piecewise remapping from
 * HF transformers' _compute_llama3_parameters is applied verbatim —
 * wavelengths shorter than orig_ctx/high_freq_factor keep their frequency,
 * longer than orig_ctx/low_freq_factor divide by `factor`, and the band
 * between interpolates smoothly.
 */
export function ropeInvFreqTable(spec: ModelSpec): Float32Array<ArrayBuffer> {
  const out = new Float32Array(spec.halfRotary)
  const rs = spec.ropeScaling
  for (let i = 0; i < spec.halfRotary; i++) {
    let inv = Math.pow(spec.ropeTheta, -(2 * i) / spec.rotaryDim)
    if (rs) {
      const lowFreqWavelen = rs.originalMaxPositionEmbeddings / rs.lowFreqFactor
      const highFreqWavelen = rs.originalMaxPositionEmbeddings / rs.highFreqFactor
      const wavelen = (2 * Math.PI) / inv
      if (wavelen > lowFreqWavelen) {
        inv = inv / rs.factor
      } else if (!(wavelen < highFreqWavelen)) {
        // medium band — smooth_factor interpolation, boundaries inclusive
        // (torch.where order in _compute_llama3_parameters)
        const smooth = (rs.originalMaxPositionEmbeddings / wavelen - rs.lowFreqFactor)
          / (rs.highFreqFactor - rs.lowFreqFactor)
        inv = (1 - smooth) * (inv / rs.factor) + smooth * inv
      }
    }
    out[i] = inv
  }
  return out
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
  // 4112 tokens. MHA over 32 layers is our most expensive KV by far —
  // 384 KiB/token, so this costs 1542 MiB (771 MiB with ?kv8=1). Left alone:
  // the model is only trained to 4096, and a 1 GiB budget would cut the
  // window to 2720. The KV bill here is the price of MHA, not of the ceiling.
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
  // 7168 tokens. GQA 8:1 over 36 layers = 144 KiB/token → 1008 MiB, just
  // inside the 1 GiB budget. (The model itself would go to 40960; that would
  // cost 5.6 GiB of KV, which is not a sane default alongside 2.3 GB of
  // weights.)
  maxPages: 448,
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
  // 32768 tokens — 8× Phi-3's window for 2/3 of its VRAM. The hybrid only
  // keeps KV on its 8 attention layers (the 24 GDN layers carry a fixed-size
  // recurrent state that does not grow with context), so a position costs
  // 32 KiB and 32768 of them land on exactly 1024 MiB — 128 MiB per layer
  // buffer, which is also the WebGPU floor for maxStorageBufferBindingSize.
  // (maxSeq is 262144; that would be 8 GiB of KV.)
  maxPages: 2048,
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


// ============================================================
// Qwen3.6-35B-A3B — the first MoE spec, and the first MLX-affine checkpoint
// ============================================================

/**
 * Same architecture as Qwen3.5, scaled: 40 layers instead of 32, a sparse MoE
 * FFN instead of a dense one, and an untied lm_head. The GDN dims are
 * IDENTICAL (kHeads 16, vHeads 32, headK/V 128, convK 4), which is why every
 * gdn_* kernel runs unchanged — verified against mlx_lm's own modules on the
 * real checkpoint by tests/kernels/real-weights.mjs.
 *
 * `ffn` is moe_intermediate_size (512), the PER-EXPERT width, not a dense FFN
 * width: it is what silu_mul strides by between slots.
 *
 * maxPages: one position costs 2 (K+V) × 2 kvHeads × 256 headDim × 2 B × 10
 * attention layers = 20 KiB. 384 pages = 6144 tokens on a 126 MiB KV budget —
 * deliberately small, because the MODEL is 19.5 GiB resident and decode dies
 * the moment the total working set exceeds physical RAM (measured 2026-08-05:
 * the GPU process is killed mid-prefill at 0.1 GB free; every hundred MiB of
 * headroom is worth more than context length here). maxSeq is 262144; even
 * the old 3072-page budget (49k tokens, 960 MiB) was 1/5th of that.
 *
 * The checkpoint is MULTIMODAL — every text record sits under
 * `language_model.`, and `vision_tower.*` (0.89 GB) is never read for text.
 */
export const QWEN36_35B_A3B: ModelSpec = makeModelSpec({
  id: 'qwen36-35b-a3b',
  d: 2048,
  layers: 40,
  heads: 16,
  kvHeads: 2,      // GQA 8:1 on the 10 attention layers
  headDim: 256,
  ffn: 512,        // moe_intermediate_size — per-expert, not a dense FFN width
  vocab: 248320,
  pageSize: 16,
  maxPages: 384,
  maxSeq: 262144,
  ropeTheta: 1e7,
  rmsEps: 1e-6,
  tiedEmbeddings: false,   // lm_head is its own 248320 x 2048 tensor (0.29 GB int4)
  qkNorm: true,
  // Same ids as Qwen3.5 — confirmed against THIS repo's tokenizer.json
  // added_tokens, not assumed from the shared 248320 vocab size.
  stops: [248046, 248044],   // <|im_end|>, <|endoftext|>
  chatTemplateId: 'chatml',
  tokenizerKind: 'byteLevel',
  hfRepo: 'lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit',
  manifestName: 'model.safetensors.index.json',
  weightFormat: 'mlx-safetensors',
  mlxPrefix: 'language_model.',  // multimodal checkpoint — text tower under language_model.
  fullAttnInterval: 4,       // layers 3, 7, ... 39 are full attention (10 of 40)
  gdn: { kHeads: 16, vHeads: 32, headK: 128, headV: 128, convK: 4 },
  partialRotaryFactor: 0.25, // rotary_dim = 64 of 256 dims per head
  attnGate: true,
  moe: { experts: 256, topK: 8, normTopkProb: true },
  // MLX names every quantized tensor <base>.weight / .scales / .biases, and
  // ships the projections UNFUSED — src/zero-tvm/mlx-weights.ts holds the
  // concatenation plan (c_attn = q++k++v, gdn proj = qkv++z++a++b, expert
  // stacks with the shared expert appended). These entries name the records
  // the rest of the engine binds by; the unfused parts are the loader's
  // business, not the spec's.
  paramNaming: {
    embedWeight: ['language_model.model.embed_tokens.weight'],
    embedScale: ['language_model.model.embed_tokens.scales'],
    lmHeadWeight: ['language_model.lm_head.weight'],   // NOT tied — its own tensor
    lmHeadScale: ['language_model.lm_head.scales'],
    finalNorm: ['language_model.model.norm.weight'],
    biasFor: (w: string) => w.replace(/\.weight$/, '.biases'),
    layer: (L: number) => {
      const p = `language_model.model.layers.${L}`
      const g = `${p}.linear_attn`
      return {
        qkvWeight: [`${p}.self_attn.q_proj.weight`],   // ++ k_proj ++ v_proj (see mlx-weights.ts)
        qkvScale: [`${p}.self_attn.q_proj.scales`],
        oProjWeight: [`${p}.self_attn.o_proj.weight`],
        oProjScale: [`${p}.self_attn.o_proj.scales`],
        norm1: [`${p}.input_layernorm.weight`],
        norm2: [`${p}.post_attention_layernorm.weight`],
        ffnWeight: [`${p}.mlp.switch_mlp.gate_proj.weight`],
        ffnScale: [`${p}.mlp.switch_mlp.gate_proj.scales`],
        ffnDownWeight: [`${p}.mlp.switch_mlp.down_proj.weight`],
        ffnDownScale: [`${p}.mlp.switch_mlp.down_proj.scales`],
        qNorm: [`${p}.self_attn.q_norm.weight`],
        kNorm: [`${p}.self_attn.k_norm.weight`],
        gdnQkvWeight: [`${g}.in_proj_qkv.weight`],
        gdnQkvScale: [`${g}.in_proj_qkv.scales`],
        gdnZWeight: [`${g}.in_proj_z.weight`],
        gdnZScale: [`${g}.in_proj_z.scales`],
        gdnAWeight: [`${g}.in_proj_a.weight`],
        gdnAScale: [`${g}.in_proj_a.scales`],
        gdnBWeight: [`${g}.in_proj_b.weight`],
        gdnBScale: [`${g}.in_proj_b.scales`],
        gdnALog: [`${g}.A_log`],
        gdnDtBias: [`${g}.dt_bias`],
        gdnConvWeight: [`${g}.conv1d.weight`],
        gdnNormWeight: [`${g}.norm.weight`],
        gdnOutWeight: [`${g}.out_proj.weight`],
        gdnOutScale: [`${g}.out_proj.scales`],
      }
    },
  },
})

/**
 * The 3-bit-expert build of QWEN36_35B_A3B — same architecture, same router,
 * same attention/GDN/lm_head precision; only the expert stacks (and the shared
 * expert inside them) are 3-bit.
 *
 * WHY IT EXISTS: the 4-bit build is 19.7 GB resident and a 32 GB Mac kills the
 * GPU process the moment the working set exceeds physical RAM (measured
 * 2026-08-05, mid-prefill, 0.1 GB free). 3-bit experts bring resident to
 * ~15.7 GB — real headroom. The cost is measured, not guessed: block-output
 * cosine 0.936 vs the 4-bit block (2-bit measured 0.79 and is not offered).
 *
 * The checkpoint is produced locally by scripts/convert-q3-experts.py; the
 * hfRepo below is where an upload WOULD live and what names the dev-mirror
 * directory — until it is uploaded, this spec only works where the converted
 * checkpoint exists on disk.
 */
export const QWEN36_35B_A3B_Q3: ModelSpec = makeModelSpec({
  ...QWEN36_35B_A3B,
  id: 'qwen36-35b-a3b-q3',
  hfRepo: 'abgunaydin/Qwen3.6-35B-A3B-MLX-q3exp',
  moe: { ...QWEN36_35B_A3B.moe!, bits: 3 },
})

// ============================================================
// Generated specs — scripts/add-model.mjs appends below this marker after its
// constraint check passes. Hand-edits above survive; below, expect churn.
// ADD-MODEL:SPECS

// mlx-community/Llama-3.2-1B-Instruct-4bit — generated by scripts/add-model.mjs (2026-08-06)
export const LLAMA_3_2_1B_INSTRUCT_4BIT: ModelSpec = makeModelSpec({
  id: 'llama-3-2-1b-instruct-4bit',
  d: 2048,
  layers: 16,
  heads: 32,
  kvHeads: 8,
  headDim: 64,
  ffn: 8192,
  vocab: 128256,
  pageSize: 16,
  // 32768 tokens on the 1 GiB KV budget rule (32 KiB/token × 16 attn layers).
  maxPages: 2048,
  maxSeq: 131072,
  ropeTheta: 500000,
  // llama3 frequency remapping — ropeInvFreqTable() precomputes the table.
  ropeScaling: { ropeType: 'llama3', factor: 32, lowFreqFactor: 1, highFreqFactor: 4, originalMaxPositionEmbeddings: 8192 },
  rmsEps: 0.00001,
  tiedEmbeddings: true,
  qkNorm: false,
  // Resolved from THIS repo's tokenizer.json added_tokens by name, not assumed.
  stops: [128009, 128001],
  chatTemplateId: 'llama3',
  tokenizerKind: 'byteLevel',
  hfRepo: 'mlx-community/Llama-3.2-1B-Instruct-4bit',
  manifestName: 'model.safetensors.index.json',
  weightFormat: 'mlx-safetensors',
  paramNaming: mlxParamNaming(""),
})

// mlx-community/Qwen3-4B-4bit — generated by scripts/add-model.mjs (2026-08-06)
export const QWEN3_4B_MLX: ModelSpec = makeModelSpec({
  id: 'qwen3-4b-mlx',
  d: 2560,
  layers: 36,
  heads: 32,
  kvHeads: 8,
  headDim: 128,
  ffn: 9728,
  vocab: 151936,
  pageSize: 16,
  // 7168 tokens on the 1 GiB KV budget rule (144 KiB/token × 36 attn layers).
  maxPages: 448,
  maxSeq: 40960,
  ropeTheta: 1000000,
  rmsEps: 0.000001,
  tiedEmbeddings: true,
  qkNorm: true,
  // Resolved from THIS repo's tokenizer.json added_tokens by name, not assumed.
  stops: [151645, 151643],
  chatTemplateId: 'chatml',
  tokenizerKind: 'byteLevel',
  hfRepo: 'mlx-community/Qwen3-4B-4bit',
  manifestName: 'model.safetensors.index.json',
  weightFormat: 'mlx-safetensors',
  paramNaming: mlxParamNaming(""),
})
// ============================================================
