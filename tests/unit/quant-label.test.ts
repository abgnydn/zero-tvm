// What the entrance tells you about how a model is quantised.
//
// This exists because the sheet used to say NOTHING for most of the roster. The
// only place quantisation appeared was the variant picker, and a picker needs a
// choice — so the two 35B builds showed "3-bit experts / full 4-bit" and the
// other five groups showed nothing at all, which a reader takes as "not
// quantised". Every model here is 4-bit; constraints.ts refuses unquantised
// weights outright.
//
// Derived from the spec rather than authored, so the assertions below are about
// the DERIVATION. A hand-written label would drift from the checkpoint the first
// time a variant was added, and this is a claim about what someone is about to
// download.

import { describe, expect, it } from 'vitest'
import { SHIPPED_MODELS, quantLabel, specForParam } from '../../src/zero-tvm/model-registry.ts'

describe('quantLabel', () => {
  it('every shipped model states a quantisation — none may be silent', () => {
    for (const { param, spec } of SHIPPED_MODELS) {
      const label = quantLabel(spec)
      expect(label, `${param} has no quantisation label`).toBeTruthy()
      expect(label, `${param} does not say how many bits`).toMatch(/-bit/)
    }
  })

  it('names the two weight formats by what the checkpoint actually is', () => {
    expect(quantLabel(specForParam('qwen35'))).toBe('4-bit · MLC q4f16_1 · group 32')
    expect(quantLabel(specForParam('qwen38'))).toBe('4-bit · MLX affine · group 64')
  })

  it('the 3-bit build says which parts are 3-bit, and which are not', () => {
    // "3-bit" flat would overstate it: only the expert stacks were requantised,
    // and the +10.4% perplexity that costs is attributed to exactly that.
    const q3 = quantLabel(specForParam('qwen36q3'))
    expect(q3).toBe('3-bit experts, 4-bit elsewhere · MLX affine · group 64')
    expect(q3).toContain('4-bit elsewhere')
  })

  it('the two 35B builds are distinguishable from the label alone', () => {
    // They sit in one group behind a picker, so the sheet must not render the
    // same string for both — that is the bug this row exists to prevent.
    expect(quantLabel(specForParam('qwen36q3'))).not.toBe(quantLabel(specForParam('qwen36')))
  })
})
