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
  experts: number
  topK: number
  /** Checkpoint ships a shared_expert (Qwen3.5/3.6 style). The engine STACKS
   *  it as expert index E, so its absence is structural, not cosmetic. */
  sharedExpert: boolean
  /** shared_expert_intermediate_size — must equal `ffn` for the stacking. */
  sharedFfn: number | null
  normTopkProb: boolean
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
export interface DetectedModel {
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
  rmsEps: number | null
  maxSeq: number | null
  tied: boolean
  qkNorm: boolean             // q_norm/k_norm records present in the index
  attnBias: boolean           // config attention_bias / any *.bias records
  slidingWindow: boolean
  hiddenAct: string | null    // 'silu' expected
  quant: DetectedQuant | null // null = unquantised (f16/bf16) checkpoint
  moe: DetectedMoe | null
  gdn: DetectedGdn | null
  attnGate: boolean           // gated attention (per-head [Q|gate] in q_proj)
  partialRotaryFactor: number | null
  hasIndex: boolean           // model.safetensors.index.json exists
  tokenizer: 'byteLevel' | 'spm' | 'unknown'
  chatTemplate: 'chatml' | 'phi3' | 'unknown'
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
      const routerOk = ROUTER_PATH.test(path) && q.bits === 8 && q.groupSize === 64
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
  if (m.ropeScalingType !== null) {
    fail('rope', `rope_scaling '${m.ropeScalingType}' (llama3/yarn-style frequency remapping)`,
      'rope.wgsl computes theta^(-2i/d) inline — scaled RoPE needs a precomputed frequency table binding')
  }
  if (m.attnBias) {
    fail('bias', 'linear-layer biases (attention_bias / *.bias records)',
      'the matmul kernels have no additive-bias epilogue (the affine "biases" tensors are per-GROUP dequant offsets, not per-row biases)')
  }
  if (m.slidingWindow) {
    fail('attention', 'sliding-window attention',
      'attention.wgsl walks every KV page unconditionally — windowed variants plus page eviction')
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
    if (vocab % 4 !== 0) fail('dims', `vocab ${vocab} % 4 != 0`, 'lm_head runs rowsPerWG=4 — the remainder rows would silently drop')
    // Every K the matmuls reduce over: unfused projections (K=d), o_proj
    // (K=qDim), ffn down (K=ffn), GDN out (K=vHeads*headV).
    const ks: [string, number][] = [['d', d], ['qDim', qDim], ['ffn', ffn]]
    if (m.gdn) ks.push(['gdnVDim', m.gdn.vHeads * m.gdn.headV])
    for (const [name, k] of ks) {
      if (k % 512 !== 0) fail('dims', `matmul K=${name} (${k}) % 512 != 0`,
        'the scalar int4 matmul unrolls K in 512-element strides (validate.html\'s reference path)')
      if (k % 64 !== 0) fail('dims', `matmul K=${name} (${k}) % 64 != 0`, 'affine scale groups are 64 wide')
    }
  }

  // ── MoE ────────────────────────────────────────────────────
  if (m.moe) {
    if (!m.moe.sharedExpert) {
      fail('moe', 'MoE without a shared expert (Mixtral / Qwen3-MoE style)',
        'the loader stacks the shared expert as index E and the router carries its gate as row E — an optional-shared-slot layout does not exist yet')
    } else if (m.moe.sharedFfn !== null && m.ffn !== null && m.moe.sharedFfn !== m.ffn) {
      fail('moe', `shared expert width ${m.moe.sharedFfn} != moe_intermediate ${m.ffn}`,
        'stacking the shared expert into the expert tensors requires equal row width')
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
    fail('template', 'chat template is neither Phi-3 nor ChatML',
      'a renderer branch in model-select.ts buildChatPromptFor (only phi3 and chatml non-thinking exist)')
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
  { area: 'RoPE', supported: 'plain theta (any base), partial rotary factor', not: 'llama3 / yarn / longrope frequency scaling', needs: 'precomputed frequency-table binding in rope.wgsl' },
  { area: 'FFN', supported: 'SwiGLU dense (fused kernel for MLC, matmul+silu_mul chain for MLX-affine)', not: 'GeGLU / ReLU / non-gated FFN, FFN biases', needs: 'activation-parameterised fused_ffn.wgsl + silu_mul.wgsl' },
  { area: 'Norm', supported: 'RMSNorm with plain gamma', not: 'LayerNorm (beta), Gemma-style (1+gamma), post-norm sandwiches', needs: 'variants of rms_norm/add_norm.wgsl' },
  { area: 'MoE', supported: '≤256 routed experts, top-K ≤32, stacked shared expert (width == moe_intermediate), 8-bit router, norm_topk_prob; subgroups required', not: 'MoE without shared expert (Mixtral, Qwen3-MoE), grouped/expert-parallel routing', needs: 'optional-shared-slot layout in loader + moe_router_topk.wgsl' },
  { area: 'Linear attention', supported: 'GatedDeltaNet (Qwen3.5/3.6): GVA, headV %32 ≤256, conv width 4', not: 'Mamba/S4, RWKV, other conv widths', needs: 'new recurrence kernels' },
  { area: 'Embedding / head', supported: 'quantised embedding (symmetric or affine), tied or untied lm_head, vocab %4', not: 'unquantised embedding tables', needs: 'f16 gather path' },
  { area: 'Tokenizer', supported: 'SentencePiece (Phi-3), byte-level BPE (Qwen/Llama-style tokenizer.json)', not: 'tekken, WordPiece, custom pipelines', needs: 'new pipeline beside tokenizer-bpe.ts' },
  { area: 'Chat template', supported: 'Phi-3, ChatML (non-thinking)', not: 'Llama-3 header template, Gemma turns, thinking-mode rendering', needs: 'renderer branch in model-select.ts' },
  { area: 'Decoding', supported: 'greedy argmax, streaming, cross-turn prefix reuse', not: 'sampling (temperature/top-p), batch > 1', needs: 'sampling kernel after logits' },
]
