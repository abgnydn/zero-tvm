/**
 * SHADER PRELUDE — renders a WGSL const block from a ModelSpec.
 *
 * `shaderPrelude(spec)` renders a WGSL const-declaration block (default spec:
 * PHI3); `withPrelude(src, spec)` injects it into a shader source right after
 * any `enable` directives (which must precede all other declarations).
 * WGSL const-expressions fold at shader-compile time — zero runtime cost.
 *
 * The integer consts are emitted WITHOUT a `u`/`i` suffix, i.e. as
 * AbstractInt, so the same name coerces cleanly in both i32 and u32 address
 * math. ROPE_THETA / RMS_EPS are emitted as floats.
 *
 * Model shape lives in model-spec.ts (re-exported here); this module is
 * dependency-free beyond that on purpose: tests/kernels/run.mjs imports it
 * directly under Node (type stripping), where Vite `?raw` imports —
 * and therefore compiler.ts — are unavailable.
 */

// NOTE: the explicit .ts extension is load-bearing — tests/kernels/*.mjs
// import this file under Node type stripping, which does no extension
// searching. tsconfig sets allowImportingTsExtensions +
// rewriteRelativeImportExtensions so tsc emit stays correct.
import { PHI3, type ModelSpec } from './model-spec.ts'

export { PHI3, QWEN3_4B, QWEN35_4B, makeModelSpec } from './model-spec.ts'
export type { ModelSpec, ModelSpecBase, ParamNaming, LayerParamNames, GdnDims } from './model-spec.ts'

/** Emit a float literal that is valid WGSL AbstractFloat (always has a dot or exponent). */
function f(v: number): string {
  const s = String(v)
  return /[.e]/i.test(s) ? s : `${s}.0`
}

/** WGSL const block: model shape + every named product used in address math. */
export function shaderPrelude(spec: ModelSpec = PHI3): string {
  const P = spec
  return `// ── injected prelude (src/compiler/shader-prelude.ts) — ${P.id} shape ──
const D               = ${P.d};      // hidden dim
const HEADS           = ${P.heads};        // attention (query) heads
const HEAD_DIM        = ${P.headDim};        // dims per head
const FFN_DIM         = ${P.ffn};      // FFN intermediate dim (also gate→up row stride)
const VOCAB           = ${P.vocab};     // vocabulary size
const PAGE_SIZE       = ${P.pageSize};        // KV-cache slots per page
const MAX_PAGES       = ${P.maxPages};       // KV-cache page budget
const QKV_DIM         = ${P.qkvDim};      // Q_DIM + 2*KV_DIM (Q,K,V stacked projection rows)
const HALF_HEAD_DIM   = ${P.halfHeadDim};        // HEAD_DIM / 2 (RoPE pair distance)
const D_PACKED        = ${P.dPacked};       // D / 8  (u32 words per K=D weight row)
const D_SCALES        = ${P.dScales};        // D / 32 (int4 scales per K=D weight row)
const QKV_GROUP_PAIRS = ${P.qkvGroupPairs};      // HEADS * HALF_HEAD_DIM (RoPE pairs in the Q group)
const HEAD_PAGE_STRIDE = ${P.headPageStride};     // PAGE_SIZE * HEAD_DIM (f16 per head per page)
const KV_PAGE_STRIDE  = ${P.kvPageStride};     // KV_HEADS * PAGE_SIZE * HEAD_DIM * 2 (f16 per page, K+V)
const V_PAGE_OFFSET   = ${P.vPageOffset};     // KV_HEADS * PAGE_SIZE * HEAD_DIM (V region start within a page)
const KV_I8_ROW_WORDS   = ${P.kvI8RowWords};      // HEAD_DIM / 4  (u32 words per int8 (head,slot,side) row)
const KV_I8_SLOT_WORDS  = ${P.kvI8SlotWords};      // 2 * KV_I8_ROW_WORDS (K+V)
const KV_I8_HEAD_WORDS  = ${P.kvI8HeadWords};     // PAGE_SIZE * KV_I8_SLOT_WORDS
const KV_I8_PAGE_WORDS  = ${P.kvI8PageWords};   // KV_HEADS * KV_I8_HEAD_WORDS
const KV_SCALES_PER_SLOT = ${P.kvScalesPerSlot};       // K + V
const KV_SCALES_PER_HEAD = ${P.kvScalesPerHead};      // PAGE_SIZE * KV_SCALES_PER_SLOT
const KV_SCALES_PER_PAGE = ${P.kvScalesPerPage};    // KV_HEADS * KV_SCALES_PER_HEAD
// GQA-aware names (== the MHA values for Phi-3; consumed by the GQA-ported shaders)
const KV_HEADS        = ${P.kvHeads};        // KV heads (== HEADS for MHA)
const Q_DIM           = ${P.qDim};      // HEADS * HEAD_DIM (== D for Phi-3)
const KV_DIM          = ${P.kvDim};      // KV_HEADS * HEAD_DIM
const GQA_GROUP       = ${P.gqaGroup};         // HEADS / KV_HEADS (query heads per KV head)
const ROPE_THETA      = ${f(P.ropeTheta)};  // RoPE base frequency
const RMS_EPS         = ${f(P.rmsEps)};     // RMSNorm epsilon
// Qwen3.5 hybrid consts (collapse to plain-attention values for other specs
// — ROTARY_DIM == HEAD_DIM, C_ATTN_DIM == QKV_DIM, GDN_* mirror the attention
// dims — so every kernel compiles under every spec; only Qwen3.5 dispatches
// the gdn_* / gated-attention kernels)
const ROTARY_DIM      = ${P.rotaryDim};        // RoPE-rotated dims per head (partial RoPE)
const HALF_ROTARY     = ${P.halfRotary};        // ROTARY_DIM / 2 (pair distance in the rotary slice)
const ATTN_GATE       = ${P.attnGate ? 1 : 0};         // 1: c_attn packs per-head [Q|gate], sigmoid-gated output
const C_ATTN_DIM      = ${P.cAttnDim};      // fused attention projection rows (QKV_DIM + gate rows)
const GDN_K_HEADS     = ${P.gdnKHeads};        // GDN key/query heads
const GDN_V_HEADS     = ${P.gdnVHeads};        // GDN value heads (GVA)
const GDN_HEAD_K      = ${P.gdnHeadK};       // GDN key/query head dim
const GDN_HEAD_V      = ${P.gdnHeadV};       // GDN value head dim
const GDN_CONV_K      = ${P.gdnConvK};         // causal-conv kernel width
const GDN_GVA_GROUP   = ${P.gdnGvaGroup};         // GDN_V_HEADS / GDN_K_HEADS
const GDN_K_DIM       = ${P.gdnKDim};      // GDN_K_HEADS * GDN_HEAD_K (Q rows == K rows)
const GDN_V_DIM       = ${P.gdnVDim};      // GDN_V_HEADS * GDN_HEAD_V (V / z-gate / out_proj width)
const GDN_QKV_DIM     = ${P.gdnQkvDim};      // 2*GDN_K_DIM + GDN_V_DIM (in_proj_qkv rows == conv channels)
const GDN_STATE_PER_HEAD = ${P.gdnStatePerHead};   // GDN_HEAD_K * GDN_HEAD_V (f32 state cells per v-head)
// ── end injected prelude ──
`
}

/** Insert the prelude after any `enable` directives (they must come first). */
export function withPrelude(src: string, spec: ModelSpec = PHI3): string {
  const lines = src.split('\n')
  let insertAt = 0
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*enable\s/.test(lines[i])) insertAt = i + 1
  }
  lines.splice(insertAt, 0, shaderPrelude(spec))
  return lines.join('\n')
}
