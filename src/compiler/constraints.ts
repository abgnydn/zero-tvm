/**
 * CONSTRAINTS — the machine-readable answer to "can this engine run that
 * checkpoint?", used by scripts/add-model.mjs to turn a HuggingFace repo into
 * either a generated ModelSpec (green) or a report naming exactly which kernel
 * or loader feature is missing (red).
 *
 * Everything here was extracted from the kernels and the engine, not from
 * intent: a rule exists because some shader's address math, workgroup size or
 * dequant loop actually breaks without it. When a rule names a kernel, that is
 * the file to change to lift it.
 *
 * The checks and the human-readable SUPPORT_MATRIX live side by side in this
 * one file ON PURPOSE — docs/COMPAT.md is generated from the matrix, so the
 * only way the docs drift from the checks is within these ~300 lines.
 *
 * Dependency-free, erasable-TS syntax: scripts/add-model.mjs imports this
 * directly under Node type stripping, exactly like model-spec.ts.
 */

// ============================================================
// What the probe extracts from a checkpoint
// ============================================================

export interface DetectedQuant {
  bits: number
  groupSize: number
  /** Per-tensor-path overrides from config.json's quantization block
   *  (path fragments, e.g. 'mlp.gate' -> {bits: 8, group_size: 64}). */
  overrides: Record<string, { bits: number; groupSize: number }>
}

export interface DetectedMoe {
  /** config scoring_func — 'softmax' is the only one the top-k kernel does. */
  scoringFunc?: string | null
  /** False when mlp.gate ships unquantized (DeepSeek); the router kernels read
   *  affine rows, so a full-precision router is a third unpack, not a bonus. */
  routerQuantized?: boolean
  /** Layers running a DENSE FFN instead of the expert block. A non-empty list
   *  (Qwen's mlp_only_layers, DeepSeek's first_k_dense_replace) means the stack
   *  is not uniform, which ModelSpec has no way to express. */
  denseLayers: number[]
  /** Every Nth layer is MoE (Qwen's decoder_sparse_step). 1 = all of them. */
  sparseStep: number
  experts: number
  topK: number
  /** Checkpoint ships a shared_expert (Qwen3.5/3.6 style). The engine STACKS
   *  it as expert index E, so its absence is structural, not cosmetic. */
  sharedExpert: boolean
  /** shared_expert_intermediate_size — must equal `ffn` for the stacking. */
  sharedFfn: number | null
  normTopkProb: boolean
  /** Router quantization width: an 8-bit per-tensor override when the
   *  checkpoint has one, otherwise the model's base bits. */
  routerBits: number
}

export interface DetectedGdn {
  kHeads: number
  vHeads: number
  headK: number
  headV: number
  convK: number
  fullAttnInterval: number
}

/** Normalised view of config.json + the safetensors index. Numeric fields are
 *  null when the config did not yield them (unmapped family). */
/** Multi-head Latent Attention (DeepSeek V2/V3/V4). K and V are not cached per
 *  head at all: one `kvLoraRank`-wide latent per token, plus a single
 *  `qkRopeHeadDim` RoPE key shared across heads, and kv_b_proj up-projects on
 *  demand. The reason to care is the cache — V2-Lite stores 576 values a token
 *  where its 16x128 MHA equivalent would store 4096. */
export interface DetectedMla {
  kvLoraRank: number
  qLoraRank: number | null    // null = q is one plain projection (V2-Lite)
  qkNopeHeadDim: number
  qkRopeHeadDim: number
  vHeadDim: number
}

export interface DetectedModel {
  mla?: DetectedMla | null
  /** Detected as an embedding model — no LM head is dispatched. */
  embeddingOnly?: boolean
  /** yarn parameters when ropeScalingType is 'yarn'. */
  ropeYarn?: {
    factor: number | null
    betaFast: number | null
    betaSlow: number | null
    originalMaxPositionEmbeddings: number | null
    mscale: number | null
    mscaleAllDim: number | null
  } | null
  /** Top-level config.json fields matching neither CONFIG_KEYS_READ nor
   *  CONFIG_KEYS_IGNORED — see the `config` rule in checkModel. */
  unknownConfigKeys?: string[]
  repo: string
  family: string              // config model_type ('' when unreadable)
  multimodal: boolean         // dims read from text_config / language_model root
  d: number | null
  layers: number | null
  heads: number | null
  kvHeads: number | null
  headDim: number | null
  ffn: number | null          // MoE: moe_intermediate_size
  vocab: number | null
  ropeTheta: number | null
  ropeScalingType: string | null   // rope_scaling.rope_type, null = plain RoPE
  /** The llama3 remapping parameters, when ropeScalingType is 'llama3' —
   *  everything ropeInvFreqTable() needs. Other scaling types stay null
   *  (and red below). */
  ropeScaling: {
    factor: number | null
    lowFreqFactor: number | null
    highFreqFactor: number | null
    originalMaxPositionEmbeddings: number | null
  } | null
  rmsEps: number | null
  maxSeq: number | null
  tied: boolean
  qkNorm: boolean             // q_norm/k_norm records present in the index
  attnBias: boolean           // config attention_bias / any *.bias records
  mlpBias?: boolean           // config mlp_bias — an additive bias on the FFN projections
  slidingWindow: boolean
  hiddenAct: string | null    // 'silu' expected
  quant: DetectedQuant | null // null = unquantised (f16/bf16) checkpoint
  moe: DetectedMoe | null
  gdn: DetectedGdn | null
  attnGate: boolean           // gated attention (per-head [Q|gate] in q_proj)
  partialRotaryFactor: number | null
  /** config `layer_types`, verbatim — a POSITIVE per-layer block claim.
   *  null when the config has none (uniform attention assumed). */
  layerTypes: string[] | null
  hasIndex: boolean           // model.safetensors.index.json exists
  tokenizer: 'byteLevel' | 'spm' | 'unknown'
  chatTemplate: 'chatml' | 'phi3' | 'llama3' | 'unknown'
}

export interface Failure {
  rule: string
  detail: string
  /** What would have to be built to lift this — kernel file or loader feature. */
  needs: string
}

export interface CheckResult {
  ok: boolean
  failures: Failure[]
  /** True facts worth knowing that are NOT blockers (flag gates, perf modes). */
  notes: string[]
}

// ============================================================
// The checks
// ============================================================

/** Quant override paths that are allowed to deviate from 4-bit/group-64:
 *  the router ships 8-bit (moe_router_logits.wgsl hardcodes the unpack) and
 *  expert stacks may be our own 3-bit conversion (int4_matmul.gen.ts q3). */
const ROUTER_PATH = /(^|\.)mlp\.(gate|shared_expert_gate)$/
const EXPERT_PATH = /(switch_mlp|shared_expert)\.(gate_proj|up_proj|down_proj)$/

/**
 * Config keys the detector READS, plus keys that carry no inference meaning.
 * Anything in a config.json outside these two sets is machinery we would be
 * silently dropping — see the `config` rule in checkModel for why that is
 * refused rather than ignored.
 */
export const CONFIG_KEYS_READ: ReadonlySet<string> = new Set([
  'model_type', 'text_config', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
  'num_key_value_heads', 'head_dim', 'vocab_size', 'intermediate_size', 'moe_intermediate_size',
  'num_experts', 'num_experts_per_tok', 'shared_expert_intermediate_size', 'norm_topk_prob',
  // rope_parameters is the newer transformers spelling and absorbs rope_theta
  // and partial_rotary_factor; add-model.mjs normalises it back into these.
  'rope_theta', 'rope_scaling', 'rope_parameters',
  'rms_norm_eps', 'max_position_embeddings', 'tie_word_embeddings',
  'attention_bias', 'sliding_window', 'use_sliding_window', 'hidden_act', 'hidden_activation',
  'quantization', 'quantization_config', 'partial_rotary_factor', 'layer_types',
  'linear_num_key_heads', 'linear_num_value_heads', 'linear_key_head_dim', 'linear_value_head_dim',
  'linear_conv_kernel_dim', 'full_attention_interval',
  'bos_token_id', 'eos_token_id',
  // Read because their non-default values ARE architecture: mlp_bias adds an
  // FFN bias epilogue, and decoder_sparse_step / mlp_only_layers say which
  // layers are dense instead of MoE. max_window_layers only means anything
  // when sliding_window is enabled, which is refused on its own.
  'mlp_bias', 'decoder_sparse_step', 'mlp_only_layers', 'max_window_layers',
  // DeepSeek's spellings. Read so the family is DESCRIBED rather than guessed
  // at — the rules below refuse what we cannot run, which is most of it.
  'kv_lora_rank', 'q_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
  'n_routed_experts', 'n_shared_experts', 'first_k_dense_replace', 'moe_layer_freq',
  'scoring_func', 'topk_method', 'routed_scaling_factor', 'n_group', 'topk_group',
])

/** Bookkeeping, training-time, or serialization fields — no effect on a
 *  forward pass, so ignoring them is safe rather than merely convenient. */
export const CONFIG_KEYS_IGNORED: ReadonlySet<string> = new Set([
  'architectures', 'auto_map', 'torch_dtype', 'dtype', 'transformers_version', 'use_cache',
  'initializer_range', 'attention_dropout', 'hidden_dropout', 'pretraining_tp', 'pad_token_id',
  'unk_token_id', '_name_or_path', 'output_router_logits', 'router_aux_loss_coef',
  'aux_loss_alpha', 'seq_aux', 'output_attentions', 'output_hidden_states', 'return_dict',
  'tokenizer_class', 'is_causal', 'attn_implementation', 'label2id', 'id2label',
])

export function checkModel(m: DetectedModel): CheckResult {
  const failures: Failure[] = []
  const notes: string[] = []
  const fail = (rule: string, detail: string, needs: string) => failures.push({ rule, detail, needs })

  // ── container ──────────────────────────────────────────────
  if (!m.hasIndex) {
    fail('index', 'model.safetensors.index.json is missing from the repo',
      'the loader resolves every tensor by ranged reads through the index — generate one (mlx_lm writes it on convert)')
  }

  // ── quantization ───────────────────────────────────────────
  if (!m.quant) {
    fail('quant', 'checkpoint is not quantized (f16/bf16 weights)',
      'every matmul kernel dequantises int4 inline — an f16-weight matmul family does not exist')
  } else {
    if (m.quant.bits !== 4 || m.quant.groupSize !== 64) {
      fail('quant', `base quantization is ${m.quant.bits}-bit group ${m.quant.groupSize}`,
        'int4_matmul.gen.ts affine kernels read 4-bit nibbles at group 64 (3-bit exists for MoE expert stacks only)')
    }
    for (const [path, q] of Object.entries(m.quant.overrides)) {
      // 16 = the checkpoint marked this tensor as NOT quantized (add-model
      // records `false` that way). Refusing it here contradicted routerBits 16,
      // which the f16 router kernel and loader plan both support.
      const routerOk = ROUTER_PATH.test(path)
        && ((q.bits === 8 && q.groupSize === 64) || q.bits === 16)
      const expertOk = EXPERT_PATH.test(path) && (q.bits === 3 || q.bits === 4) && q.groupSize === 64
      if (!routerOk && !expertOk && !(q.bits === 4 && q.groupSize === 64)) {
        fail('quant', `'${path}' is ${q.bits}-bit group ${q.groupSize}`,
          'only the MoE router may be 8-bit and only expert stacks 3-bit; everything else needs 4-bit group 64')
      }
    }
  }

  // ── architecture blocks ────────────────────────────────────
  if (m.hiddenAct !== null && m.hiddenAct !== 'silu') {
    fail('activation', `hidden_act is '${m.hiddenAct}'`,
      'silu_mul.wgsl and fused_ffn.wgsl hardcode SiLU — a GeGLU/other-activation variant of both')
  }
  // llama3 and yarn are both green: rope.wgsl binds a precomputed inv_freq
  // table and ropeInvFreqTable() implements each remapping (yarn checked
  // against DeepSeek's own helpers in tests/unit/rope-yarn.test.ts). longrope
  // and anything newer stay red until someone writes them.
  if (m.ropeScalingType !== null && m.ropeScalingType !== 'llama3' && m.ropeScalingType !== 'yarn') {
    fail('rope', `rope_scaling '${m.ropeScalingType}' (longrope-style frequency remapping)`,
      'a frequency formula in model-spec.ts ropeInvFreqTable() — rope.wgsl already reads the table')
  }
  if (m.ropeScalingType === 'yarn') {
    const y = m.ropeYarn
    const missing = !y || [y.factor, y.betaFast, y.betaSlow, y.originalMaxPositionEmbeddings]
      .some((v) => v === null || v === undefined)
    if (missing) {
      fail('rope', 'rope_scaling is yarn but the config omits factor / beta_fast / beta_slow / original_max_position_embeddings',
        'all four decide the table; defaulting any of them would silently mis-stretch long context')
    }
  }
  if (m.ropeScalingType === 'llama3') {
    const rs = m.ropeScaling
    const missing = !rs || [rs.factor, rs.lowFreqFactor, rs.highFreqFactor, rs.originalMaxPositionEmbeddings]
      .some((v) => typeof v !== 'number')
    if (missing) {
      fail('rope', 'rope_scaling llama3 without factor/low_freq_factor/high_freq_factor/original_max_position_embeddings',
        'the full parameter set in config.json — ropeInvFreqTable() cannot remap without it')
    }
  }
  if (m.mlpBias) {
    fail('bias', 'FFN biases (mlp_bias)', 'the matmul kernels have no additive-bias epilogue')
  }
  if (m.attnBias) {
    fail('bias', 'linear-layer biases (attention_bias / *.bias records)',
      'the matmul kernels have no additive-bias epilogue (the affine "biases" tensors are per-GROUP dequant offsets, not per-row biases)')
  }
  if (m.slidingWindow) {
    fail('attention', 'sliding-window attention',
      'attention.wgsl walks every KV page unconditionally — windowed variants plus page eviction')
  }
  // layer_types is a POSITIVE per-layer block claim, and the checker must
  // refuse what it does not recognise: LFM2's 'conv' blocks would otherwise
  // detect as plain attention (nothing above knows conv exists) and pass every
  // dimension rule — a fake green that only dies at logits time.
  // 'linear_attention' is known only when the GDN dims resolved; the same
  // string on a non-GDN family would run a kernel chain that does not match.
  if (m.layerTypes) {
    const known = new Set(['full_attention', 'attention'])
    if (m.gdn) known.add('linear_attention')
    for (const t of new Set(m.layerTypes.filter((t) => !known.has(t)))) {
      fail('blocks', `layer_types contains unknown block type '${t}'`,
        `a kernel family for '${t}' blocks — only full attention and GatedDeltaNet linear attention exist here`)
    }
  }

  // ── dimensions ─────────────────────────────────────────────
  const dims = { d: m.d, layers: m.layers, heads: m.heads, kvHeads: m.kvHeads, headDim: m.headDim, ffn: m.ffn, vocab: m.vocab }
  const missing = Object.entries(dims).filter(([, v]) => v === null).map(([k]) => k)
  if (missing.length) {
    fail('config', `could not resolve ${missing.join(', ')} from config.json (family '${m.family || '?'}')`,
      'a family mapping in scripts/add-model.mjs detect() — the HF config omits fields it considers defaults')
  } else {
    const d = m.d!, heads = m.heads!, kvHeads = m.kvHeads!, headDim = m.headDim!, ffn = m.ffn!, vocab = m.vocab!
    const qDim = heads * headDim
    const kvDim = kvHeads * headDim
    const qkvDim = qDim + 2 * kvDim
    if (heads % kvHeads !== 0) fail('dims', `heads ${heads} % kvHeads ${kvHeads} != 0`, 'GQA group must be integral (attention.wgsl maps q-heads to kv-heads by division)')
    if (headDim % 32 !== 0) fail('dims', `headDim ${headDim} % 32 != 0`,
      'attention/qk_norm kernels load rows 32 lanes at a time — headDim 80/96-style models need a tail path (Phi-3\'s 96 works via the MLC layout only)')
    if (headDim > 256) fail('dims', `headDim ${headDim} > 256`, 'per-head shared-memory sizing in attention.wgsl')
    if (d % 256 !== 0) fail('dims', `d ${d} % 256 != 0`, 'add_norm/embedding grids are d/256 workgroups with no tail')
    if (qkvDim % 256 !== 0) fail('dims', `qkvDim ${qkvDim} % 256 != 0`, 'rope grid is qkvDim/256 workgroups with no tail')
    if (kvDim % 256 !== 0) fail('dims', `kvDim ${kvDim} % 256 != 0`, 'kv_append grid is kvDim/256 workgroups with no tail')
    // Only when an LM head actually runs. An embedding model's output is the
    // pooled hidden state; its lm_head is never dispatched, so refusing it here
    // would be the checker wrong in the expensive direction.
    if (vocab % 4 !== 0 && !m.embeddingOnly) {
      fail('dims', `vocab ${vocab} % 4 != 0`, 'lm_head runs rowsPerWG=4 — the remainder rows would silently drop')
    }
    // Every K the matmuls reduce over: unfused projections (K=d), o_proj
    // (K=qDim), ffn down (K=ffn), GDN out (K=vHeads*headV).
    const ks: [string, number][] = [['d', d], ['qDim', qDim], ['ffn', ffn]]
    if (m.gdn) ks.push(['gdnVDim', m.gdn.vHeads * m.gdn.headV])
    // The general int4 matmul strides K by the workgroup width and stops on a
    // bound, so a ragged K keeps its tail — only the scale grouping constrains
    // the shape now. The _vec4 / _vec4h siblings still want K % 1024 / % 512,
    // but resolveMatmul() gates them per INSTANCE and falls through to the
    // general pipeline, so they never force a whole model out.
    for (const [name, k] of ks) {
      if (k % 64 !== 0) fail('dims', `matmul K=${name} (${k}) % 64 != 0`, 'affine scale groups are 64 wide')
    }
  }

  // ── UNRECOGNISED CONFIG ────────────────────────────────────
  // The detector reads Qwen-shaped field names. A family that spells things
  // differently does not fail — it produces a PLAUSIBLE WRONG picture, which
  // is far worse. DeepSeek-V2-Lite reported as dense 16/16x128 MHA when it is
  // actually MLA (kv_lora_rank 512) with a 64-expert MoE (n_routed_experts);
  // fixing the two things this checker did flag would have turned it GREEN and
  // it would have generated fluent nonsense.
  //
  // So an unknown key is a refusal. It over-refuses by design: adding a key to
  // CONFIG_KEYS_IGNORED is a deliberate act that says "I read this and it does
  // not change the forward pass", which is exactly the review that was missing.
  if (m.unknownConfigKeys?.length) {
    fail('config', `config.json carries ${m.unknownConfigKeys.length} field(s) this checker does not read: ${m.unknownConfigKeys.join(', ')}`,
      'each one is either architecture we would silently ignore or a field to add to CONFIG_KEYS_IGNORED in constraints.ts after reading it')
  }

  // ── MLA ────────────────────────────────────────────────────
  if (m.mla) {
    const cached = m.mla.kvLoraRank + m.mla.qkRopeHeadDim
    const mha = m.kvHeads !== null && m.headDim !== null ? m.kvHeads * m.headDim * 2 : 0
    fail('attention', `multi-head latent attention (kv_lora_rank ${m.mla.kvLoraRank}`
      + `, qk ${m.mla.qkNopeHeadDim}+${m.mla.qkRopeHeadDim}, v ${m.mla.vHeadDim})`,
      `a latent KV cache of ${cached} values per token${mha ? ` instead of ${mha}` : ''} and an attention kernel that scores `
      + 'against it: q_nope through kv_b_proj into latent space, one shared RoPE key, output re-expanded through kv_b_proj\'s V half')
  }

  // ── MoE ────────────────────────────────────────────────────
  if (m.moe) {
    // A stack that MIXES dense and MoE layers. ModelSpec carries one block kind
    // for every layer, so a checkpoint whose layer 0 is dense would run the
    // expert block over dense weights — fluent nonsense, no error. Both
    // spellings are in the wild (Qwen's mlp_only_layers, DeepSeek's
    // first_k_dense_replace).
    // ModelSpec can now SAY this (moe.denseLayers + denseFfn -> ffnKinds) and
    // the loader plans the right records per layer. What is still missing is
    // the engine: buffer sizing and dispatch shapes come from a single S.ffn,
    // and a stack with two widths needs both sets built and chosen per layer.
    if (m.moe.denseLayers.length) {
      fail('moe', `layers [${m.moe.denseLayers.join(', ')}] run a dense FFN while the rest are MoE`,
        'engine-core sizes the FFN buffers from one width — a mixed stack needs both, chosen per layer '
        + '(ModelSpec.ffnKinds/ffnWidthAt and the loader plans already handle it)')
    }
    // Routing that is not plain softmax + top-k.
    if (m.moe.scoringFunc && m.moe.scoringFunc !== 'softmax') {
      fail('moe', `router scoring '${m.moe.scoringFunc}' (not softmax)`,
        'moe_router_topk.wgsl softmaxes the routed logits — a scoring branch there')
    }
    // An unquantized router is GREEN: moe_router_logits_f16.wgsl reads the bare
    // f16 rows, planLayer plans a single weight instead of a quantized trio,
    // and the engine binds four buffers rather than six. routerBits carries
    // which of the three it is — see the MoeDims note on why guessing is fatal.
    if (m.moe.sparseStep !== 1) {
      fail('moe', `only every ${m.moe.sparseStep}th layer is MoE (decoder_sparse_step)`,
        'a per-layer block kind in ModelSpec, like layer_types for the hybrids')
    }
    // No shared expert is fine: the loader stops appending index E, the router
    // drops to E rows, and moe_router_topk emits K slots instead of K+1. The
    // WIDTH check below still applies when there IS one, because that is what
    // stacking it into the expert tensors requires.
    if (m.moe.sharedExpert && m.moe.sharedFfn !== null && m.ffn !== null && m.moe.sharedFfn !== m.ffn) {
      fail('moe', `shared expert width ${m.moe.sharedFfn} != moe_intermediate ${m.ffn}`,
        'stacking the shared expert into the expert tensors requires equal row width')
    }
    // The router width decides which moe_router_logits entry point runs. Only
    // 4 and 8 have one; anything else would silently read the row at the wrong
    // stride, which is noise, not an error.
    if (m.moe.routerBits !== 4 && m.moe.routerBits !== 8 && m.moe.routerBits !== 16) {
      fail('moe', `router is ${m.moe.routerBits}-bit`,
        'the router kernels cover 4-bit, 8-bit and unquantized f16 rows, and no other width')
    }
    if (m.moe.experts > 256) fail('moe', `${m.moe.experts} experts > 256`,
      'moe_router_topk.wgsl holds expert scores in 32 lanes × 8 registers')
    if (m.moe.topK > 32) fail('moe', `top-${m.moe.topK} > 32`, 'one router slot per lane in moe_router_topk.wgsl')
    notes.push('MoE requires WebGPU subgroups — there is no scalar router/expert path')
  }

  // ── GDN (linear attention) ─────────────────────────────────
  if (m.gdn) {
    if (m.gdn.vHeads % m.gdn.kHeads !== 0) fail('gdn', `vHeads ${m.gdn.vHeads} % kHeads ${m.gdn.kHeads} != 0`, 'GVA grouping in gdn_recur.wgsl')
    if (m.gdn.headV % 32 !== 0 || m.gdn.headV > 256) fail('gdn', `headV ${m.gdn.headV}`, 'gdn_recur/gdn_norm_out workgroup sizing (32-lane rows, ≤256)')
    if (m.gdn.convK !== 4) fail('gdn', `conv kernel width ${m.gdn.convK} != 4`, 'gdn_conv.wgsl ring state is sized for width 4')
    notes.push('GDN layers force the unfused f16-KV composition (no ?kv8, no fused QKV)')
  }

  // ── tokenizer & template ───────────────────────────────────
  if (m.tokenizer === 'unknown') {
    fail('tokenizer', 'tokenizer.json is neither SentencePiece nor byte-level BPE',
      'a third tokenizer pipeline in src/zero-tvm/ (tokenizer.ts covers SPM, tokenizer-bpe.ts byte-level BPE)')
  }
  if (m.chatTemplate === 'unknown') {
    fail('template', 'chat template is none of Phi-3, ChatML, Llama-3 or DeepSeek',
      'a renderer branch in model-select.ts buildChatPromptFor (phi3, chatml non-thinking, llama3 and deepseek exist)')
  }

  // ── notes that gate flags, not support ─────────────────────
  if (m.headDim !== null && m.headDim > 128) notes.push('int8 KV (?kv8=1) unavailable — kv_quantize_int8 packs headDim ≤ 128')
  if (m.quant) notes.push('MLX-affine runs the unfused composition (no fused QKV, 3-dispatch dense FFN); hybrid chunked prefill is off (no affine batched_dyn)')

  return { ok: failures.length === 0, failures, notes }
}

// ============================================================
// Human-readable support matrix — docs/COMPAT.md is generated from this
// ============================================================

export interface MatrixRow {
  area: string
  supported: string
  not: string
  /** What lifting the limit takes (kernel/loader work). '' = nothing pending. */
  needs: string
}

export const SUPPORT_MATRIX: MatrixRow[] = [
  { area: 'Weight format', supported: 'MLC q4f16_1 shards (group 32, symmetric); MLX safetensors (group 64, affine, 4-bit; 8-bit router; 3-bit expert stacks via convert-q3-experts)', not: 'f16/bf16 unquantised, GPTQ/AWQ, other group sizes', needs: 'new dequant paths in int4_matmul.gen.ts' },
  { area: 'Attention', supported: 'MHA and GQA, headDim %32 ≤256, full RoPE or partial (rotary fraction), qk-norm optional, gated attention (Qwen3.5/3.6), paged f16 KV; int8 KV for headDim ≤128', not: 'sliding window, MLA, ALiBi, softcap, attention biases', needs: 'windowed/MLA variants of attention.wgsl; bias epilogue in matmuls' },
  { area: 'RoPE', supported: 'plain theta (any base), partial rotary factor, llama3 and yarn frequency scaling (precomputed inv_freq table; yarn also scales attention logits by mscale^2)', not: 'longrope frequency scaling', needs: 'a frequency formula in model-spec.ts ropeInvFreqTable() — rope.wgsl already reads the table' },
  { area: 'FFN', supported: 'SwiGLU dense (fused kernel for MLC, matmul+silu_mul chain for MLX-affine)', not: 'GeGLU / ReLU / non-gated FFN, FFN biases', needs: 'activation-parameterised fused_ffn.wgsl + silu_mul.wgsl' },
  { area: 'Norm', supported: 'RMSNorm with plain gamma', not: 'LayerNorm (beta), Gemma-style (1+gamma), post-norm sandwiches', needs: 'variants of rms_norm/add_norm.wgsl' },
  { area: 'MoE', supported: '≤256 routed experts, top-K ≤32, with OR without a shared expert (when present it stacks as index E and must equal moe_intermediate in width), router at 4-bit, 8-bit or unquantized f16, norm_topk_prob either way; subgroups required', not: 'grouped/expert-parallel routing, a stack mixing dense and MoE layers', needs: 'a routing layout where experts are sharded across dispatches rather than stacked; a per-layer block kind for mixed stacks' },
  { area: 'Linear attention', supported: 'GatedDeltaNet (Qwen3.5/3.6): GVA, headV %32 ≤256, conv width 4', not: 'Mamba/S4, RWKV, conv hybrids (LFM2 layer_types \'conv\'), other conv widths', needs: 'new recurrence kernels' },
  { area: 'Embedding / head', supported: 'quantised embedding (symmetric or affine), tied or untied lm_head, vocab %4', not: 'unquantised embedding tables', needs: 'f16 gather path' },
  { area: 'Tokenizer', supported: 'SentencePiece (Phi-3), byte-level BPE (Qwen/Llama-style tokenizer.json)', not: 'tekken, WordPiece, custom pipelines', needs: 'new pipeline beside tokenizer-bpe.ts' },
  { area: 'Chat template', supported: 'Phi-3, ChatML (non-thinking), Llama-3 header template, DeepSeek prose turns', not: 'Gemma turns, Mistral [INST], thinking-mode rendering', needs: 'renderer branch in model-select.ts — the single highest-leverage gap in the survey (docs/PORTING.md): it blocks 29 of 51 refused repos and is the SOLE blocker on 5' },
  { area: 'Decoding', supported: 'greedy argmax (default), seeded temperature / top-p / min-p sampling, streaming, cross-turn prefix reuse', not: 'top-k, repetition/presence penalties, beam search, batch > 1', needs: 'a rank selection pass beside sampler.wgsl\'s mass threshold; a per-sequence token-count buffer for penalties' },
]
