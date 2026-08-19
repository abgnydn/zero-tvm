// What the entrance tells you a context window COSTS.
//
// This is the number someone reads to decide whether a model fits on their
// machine, so the only acceptable error direction is pessimistic. It was
// optimistic: the int8 term counted one f16 scale per (token, kv head, side)
// and the engine allocates one scales buffer PER ATTENTION LAYER, so Phi-3
// displayed 771.5 MB against 787.1 MB actually allocated.
//
// Asserted against allocKVPagesInt8's own arithmetic rather than against a
// recorded constant: a constant would agree with a wrong formula the moment
// someone changed the layout, which is exactly how this drifted.

import { describe, expect, it } from 'vitest'
import { specForParam, SHIPPED_MODELS, kvBytesPerTokenShown as shown } from '../../src/zero-tvm/model-registry.ts'
import type { ModelSpec } from '../../src/compiler/model-spec.ts'

// Imported, NOT re-implemented. The first version of this file copied the
// formula out of landing.ts to test it, and the mutation gate caught that at
// once: breaking the real one left the copy green. A test that restates its
// subject asserts only that the author can type it twice.

/** What allocKVPagesInt8 actually reserves, per token of context. */
const allocated = (s: ModelSpec): number => {
  const attn = s.layerKinds.filter((k) => k === 'attn').length
  const pages = s.maxPages * s.kvI8PageWords * 4 * attn
  const scales = s.maxPages * s.kvScalesPerPage * 2 * attn
  return (pages + scales) / s.maxContext
}

describe('the KV figure on the model sheet', () => {
  it('matches what the engine allocates, for every model that can use int8', () => {
    for (const { param, spec } of SHIPPED_MODELS) {
      if (spec.mla || spec.embeddingOnly) continue
      const a = allocated(spec)
      const b = shown(spec, true)
      // Exact, not approximate. Both are integer byte counts over the same
      // page geometry; a tolerance here would hide the layer-count bug that
      // motivated this file, which was only 2%.
      expect(b, `${param}: sheet shows ${b} B/token, engine allocates ${a}`).toBe(a)
    }
  })

  it('never UNDERSTATES — the only safe error is pessimistic', () => {
    for (const { param, spec } of SHIPPED_MODELS) {
      if (spec.mla || spec.embeddingOnly) continue
      expect(shown(spec, true), `${param} understates`).toBeGreaterThanOrEqual(allocated(spec))
    }
  })

  it('counts the scales once per ATTENTION layer, not once per model', () => {
    // The bug, stated as a property: on a hybrid, attention layers are a
    // minority, so dropping the multiplier is a small error that hides in a
    // rounded GB figure. Phi-3 is all-attention and shows it least.
    const s = specForParam('qwen35')
    const attn = s.layerKinds.filter((k) => k === 'attn').length
    expect(attn).toBeLessThan(s.layers)
    const wrong = s.kvBytesPerToken / 2 + 4 * s.kvHeads
    expect(shown(s, true) - wrong).toBe(4 * s.kvHeads * (attn - 1))
  })

  it('int8 is meaningfully smaller than f16, and MLA opts out', () => {
    const s = specForParam('qwen38')
    expect(shown(s, true)).toBeLessThan(shown(s, false))
    expect(shown(s, true) / shown(s, false)).toBeCloseTo(0.5, 1)
    const mla = SHIPPED_MODELS.map((m) => m.spec).find((x) => x.mla)
    if (mla) expect(shown(mla, true)).toBe(mla.kvBytesPerToken)
  })
})
