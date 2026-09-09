// The entrance must name a model the SAME WAY everywhere, and offer it in an
// order a visitor can trust. Both were wrong on main, and both were reported as
// something else — which is why they are worth pinning.
//
// The naming bug read as a MISSING MODEL. The stage printed the branding name
// in full while the roster card ran it through `.replace(/-Instruct.*$/, '')`.
// Exactly one shipped checkpoint carried that suffix, so exactly one card
// disagreed with the stage above it: the arrows landed on "Llama-3.2-1B-Instruct"
// while the rail said "Llama-3.2-1B", and the model on stage looked like one the
// roster did not carry. Nobody reports "two surfaces disagree about a string";
// they report "the arrows show a model that isn't there", and then you go
// looking for a filter that does not exist.
//
// The rule that prevents the whole class: a branding name is the ONE name. No
// surface may derive a different one from it. A test cannot see CSS, but it can
// refuse the suffix that made the two derivations differ.
//
// The ORDER bug was quieter. SHIPPED_MODELS led with Phi-3, so the strongest
// model this project runs sat below four weaker ones on the character select.
// Reordering it is a one-line change with a trap underneath: `specForParam`
// resolves an unknown or absent ?model= by FALLING THROUGH to Phi-3, and it
// must keep doing that whatever position Phi-3 holds in the list — every
// pre-registry URL depends on it.
import { describe, expect, it } from 'vitest'
import { PHI3 } from '../../src/compiler/model-spec.js'
import { SHIPPED_MODELS, modelBranding, specForParam } from '../../src/zero-tvm/model-registry.js'

describe('entrance identity', () => {
  it('gives every shipped model exactly one name', () => {
    // -Instruct is the specific suffix the roster stripped and the stage kept.
    // It also distinguishes nothing here: every chat model this project ships
    // is instruct-tuned, so the suffix is noise on the one card that had it.
    for (const { param, spec } of SHIPPED_MODELS) {
      const { name } = modelBranding(spec)
      expect(name, `${param || '(default)'} carries a suffix one surface will strip`)
        .not.toMatch(/-Instruct/i)
    }
  })

  it('keeps two builds of one character distinguishable inside its card', () => {
    // Sharing a branding name is DELIBERATE: buildGroups() keys groups by it,
    // which is how qwen36q3 and qwen36 become one "Qwen3.6-35B-A3B" card with
    // two quantisation chips. (An earlier version of this test asserted names
    // were unique and failed on exactly that pair — the rule it encoded was
    // not the rule the entrance has.)
    //
    // What must hold is the level below: the chips come from the part of
    // `params` after ' · ', so two builds under one name need DIFFERENT
    // variant labels or the card offers two chips reading the same thing.
    const byName = new Map<string, string[]>()
    for (const { spec } of SHIPPED_MODELS) {
      if (spec.embeddingOnly) continue
      const b = modelBranding(spec)
      const i = b.params.indexOf(' · ')
      const variant = i < 0 ? b.params : b.params.slice(i + 3)
      byName.set(b.name, [...(byName.get(b.name) ?? []), variant])
    }
    for (const [name, variants] of byName) {
      expect(new Set(variants).size, `"${name}" offers ${variants.length} builds `
        + `but only ${new Set(variants).size} distinct labels: ${variants.join(', ')}`)
        .toBe(variants.length)
    }
  })

  it('leads the roster with the strongest model, not the smallest', () => {
    // The character select reads top-down. Phi-3 leading it made the weakest
    // shipped model the project's first impression.
    expect(SHIPPED_MODELS[0].param).toBe('qwen38')
    expect(SHIPPED_MODELS[0].spec.id).not.toBe(PHI3.id)
  })

  it('still falls through to Phi-3 for an absent or unknown ?model=', () => {
    // The trap under the reorder: this must not depend on Phi-3's position.
    expect(specForParam(null).id).toBe(PHI3.id)
    expect(specForParam('').id).toBe(PHI3.id)
    expect(specForParam('not-a-model').id).toBe(PHI3.id)
    // And the empty param must still be Phi-3's own entry, not a coincidence
    // of the fall-through — ?model= with no value is a real pre-registry URL.
    const empty = SHIPPED_MODELS.find((m) => m.param === '')
    expect(empty?.spec.id).toBe(PHI3.id)
  })
})
