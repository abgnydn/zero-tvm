/**
 * SHADER PRELUDE — single source of truth for the Phi-3 model shape.
 *
 * `shaderPrelude()` renders a WGSL const-declaration block from PHI3;
 * `withPrelude(src)` injects it into a shader source right after any
 * `enable` directives (which must precede all other declarations).
 * WGSL const-expressions fold at shader-compile time — zero runtime cost.
 *
 * The consts are emitted WITHOUT a `u`/`i` suffix, i.e. as AbstractInt,
 * so the same name coerces cleanly in both i32 and u32 address math.
 *
 * This module is dependency-free on purpose: tests/kernels/run.mjs imports
 * it directly under Node (type stripping), where Vite `?raw` imports —
 * and therefore compiler.ts — are unavailable.
 */

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

/** WGSL const block: model shape + every named product used in address math. */
export function shaderPrelude(): string {
  const P = PHI3
  const HALF_HEAD_DIM = P.HEAD_DIM / 2
  const KV_I8_ROW_WORDS = P.HEAD_DIM / 4
  return `// ── injected prelude (src/compiler/shader-prelude.ts) — Phi-3 shape ──
const D               = ${P.D};      // hidden dim
const HEADS           = ${P.HEADS};        // attention heads
const HEAD_DIM        = ${P.HEAD_DIM};        // dims per head
const FFN_DIM         = ${P.FFN};      // FFN intermediate dim (also gate→up row stride)
const VOCAB           = ${P.VOCAB};     // vocabulary size
const PAGE_SIZE       = ${P.PAGE_SIZE};        // KV-cache slots per page
const MAX_PAGES       = ${P.MAX_PAGES};       // KV-cache page budget
const QKV_DIM         = ${3 * P.D};      // 3 * D (Q,K,V stacked projection rows)
const HALF_HEAD_DIM   = ${HALF_HEAD_DIM};        // HEAD_DIM / 2 (RoPE pair distance)
const D_PACKED        = ${P.D / 8};       // D / 8  (u32 words per K=D weight row)
const D_SCALES        = ${P.D / 32};        // D / 32 (int4 scales per K=D weight row)
const QKV_GROUP_PAIRS = ${P.HEADS * HALF_HEAD_DIM};      // HEADS * HALF_HEAD_DIM (RoPE pairs per Q/K/V group)
const HEAD_PAGE_STRIDE = ${P.PAGE_SIZE * P.HEAD_DIM};     // PAGE_SIZE * HEAD_DIM (f16 per head per page)
const KV_PAGE_STRIDE  = ${2 * P.HEADS * P.PAGE_SIZE * P.HEAD_DIM};     // HEADS * PAGE_SIZE * HEAD_DIM * 2 (f16 per page, K+V)
const V_PAGE_OFFSET   = ${P.HEADS * P.PAGE_SIZE * P.HEAD_DIM};     // HEADS * PAGE_SIZE * HEAD_DIM (V region start within a page)
const KV_I8_ROW_WORDS   = ${KV_I8_ROW_WORDS};      // HEAD_DIM / 4  (u32 words per int8 (head,slot,side) row)
const KV_I8_SLOT_WORDS  = ${2 * KV_I8_ROW_WORDS};      // 2 * KV_I8_ROW_WORDS (K+V)
const KV_I8_HEAD_WORDS  = ${P.PAGE_SIZE * 2 * KV_I8_ROW_WORDS};     // PAGE_SIZE * KV_I8_SLOT_WORDS
const KV_I8_PAGE_WORDS  = ${P.HEADS * P.PAGE_SIZE * 2 * KV_I8_ROW_WORDS};   // HEADS * KV_I8_HEAD_WORDS
const KV_SCALES_PER_SLOT = ${2};       // K + V
const KV_SCALES_PER_HEAD = ${P.PAGE_SIZE * 2};      // PAGE_SIZE * KV_SCALES_PER_SLOT
const KV_SCALES_PER_PAGE = ${P.HEADS * P.PAGE_SIZE * 2};    // HEADS * KV_SCALES_PER_HEAD
// ── end injected prelude ──
`
}

/** Insert the prelude after any `enable` directives (they must come first). */
export function withPrelude(src: string): string {
  const lines = src.split('\n')
  let insertAt = 0
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*enable\s/.test(lines[i])) insertAt = i + 1
  }
  lines.splice(insertAt, 0, shaderPrelude())
  return lines.join('\n')
}
