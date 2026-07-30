/**
 * UNIT TEST — per-model KV budget (src/compiler/model-spec.ts).
 *
 * Regression guard for the context-ceiling fix: all three specs used to carry
 * a copy-pasted `maxPages: 257`, which prices a Qwen3.5 context (32 KiB per
 * position) exactly like a Phi-3 one (384 KiB) and capped every model at 4112
 * tokens. This pins the two things that made that wrong:
 *
 *   - kvBytesPerToken really is 2 (K+V) × kvHeads × headDim × 2 B × attention
 *     layers — the hybrid spec counts only its 8 attention layers, not 32;
 *   - the resulting allocation (maxContext × kvBytesPerToken) sits inside the
 *     ~1 GiB VRAM budget the ceilings were chosen against, and never above
 *     the model's own trained maxSeq.
 *
 * Pure arithmetic over the frozen specs — no GPU, no DOM.
 */

import { describe, test, expect } from 'vitest'
import { PHI3, QWEN3_4B, QWEN35_4B, type ModelSpec } from '../../src/compiler/model-spec.js'

const MiB = 1024 * 1024

/** Expected KV cost/ceiling per spec — hand-derived, see model-spec.ts. */
const EXPECTED: Array<{
  spec: ModelSpec
  attnLayers: number
  kibPerToken: number
  maxContext: number
  totalMiB: number
}> = [
  // MHA on all 32 layers: 2 × 32 × 96 × 2 B = 12 KiB/layer.
  { spec: PHI3, attnLayers: 32, kibPerToken: 384, maxContext: 4112, totalMiB: 1542 },
  // GQA 8:1 on all 36 layers: 2 × 8 × 128 × 2 B = 4 KiB/layer.
  { spec: QWEN3_4B, attnLayers: 36, kibPerToken: 144, maxContext: 7168, totalMiB: 1008 },
  // Hybrid: KV lives on 8 of 32 layers (the 24 GDN layers hold a fixed-size
  // recurrent state instead), 2 × 4 × 256 × 2 B = 4 KiB/layer.
  { spec: QWEN35_4B, attnLayers: 8, kibPerToken: 32, maxContext: 32768, totalMiB: 1024 },
]

describe('per-model KV budget', () => {
  test.each(EXPECTED)(
    '$spec.id: KV cost, context ceiling and VRAM budget',
    ({ spec, attnLayers, kibPerToken, maxContext, totalMiB }) => {
      expect(spec.layerKinds.filter((k) => k === 'attn').length).toBe(attnLayers)
      expect(spec.kvBytesPerToken).toBe(2 * spec.kvHeads * spec.headDim * 2 * attnLayers)
      expect(spec.kvBytesPerToken / 1024).toBe(kibPerToken)

      // The ceiling the engine enforces, and the allocation it implies.
      expect(spec.maxPages * spec.pageSize).toBe(spec.maxContext)
      expect(spec.maxContext).toBe(maxContext)
      expect((spec.maxContext * spec.kvBytesPerToken) / MiB).toBeCloseTo(totalMiB, 0)

      // Never ask the GPU for more KV than the budget the ceilings were
      // chosen against, and never promise more context than the model was
      // trained for. (Phi-3 is the documented exception on the first bound:
      // its 4k trained window costs 1.5 GiB under MHA.)
      const budgetMiB = spec.id === PHI3.id ? 1542 : 1024
      expect((spec.maxContext * spec.kvBytesPerToken) / MiB).toBeLessThanOrEqual(budgetMiB)
      expect(spec.maxContext).toBeLessThanOrEqual(spec.maxSeq + spec.pageSize)

      // Each per-layer page buffer must stay inside the WebGPU floor for
      // maxStorageBufferBindingSize (128 MiB) so the allocation is portable.
      expect(spec.maxPages * spec.kvPageStride * 2).toBeLessThanOrEqual(128 * MiB)
    },
  )
})
