// Every shipped model must be DISPOSITIONED on every surface that reads the
// registry — asserted by iterating the live registry, not a hand-kept list.
//
// The failure this exists to catch, in FreeToken's words
// (tests/server/test_parser_auto_selection.py): "Add a model family and forget
// the cascade, and the model serves perfectly: it just stops emitting tool
// calls and stops separating out its thinking, BECAUSE IT SILENTLY FELL THROUGH
// TO THE GENERIC DEFAULT. Every other parser test still passes, since they all
// start from a parser name that is already correct."
//
// The emphasised clause was elided from an earlier draft of this file, and it
// is the half that matters here: `toolDialectFor` returns 'chatml-json' for any
// id it does not recognise, and `modelBranding` returns Phi-3's card for any id
// it does not know. Both are silent fall-throughs to a generic default, so the
// first version of this file asserted `toolDialectFor(...)` was one of three
// legal values — which no input can fail — and that branding "renders", which
// an unbranded spec does, as Phi-3.
//
// That is the shape here — but state the gap accurately, because an earlier
// draft of this header did not. FIVE test files already iterate SHIPPED_MODELS
// on main: chatml-generations, kv-alloc-matches-flags, kv-figure, lib-entry and
// quant-label. Each checks ONE property across every spec, and they do it well.
//
// What did not exist is a single place asking "is a new spec dispositioned
// everywhere the registry is read" — so a twelfth could land with no BRANDINGS
// entry (rendering Phi-3's card), a chatTemplateId no renderer branches on
// (silently taking the chatml-json dialect), or dimensions the ingest checker
// would have refused, and the suite would stay green.
//
// The quant-label and KV-cost assertions below OVERLAP quant-label.test.ts and
// kv-figure.test.ts deliberately: this file's job is to be the one list a new
// spec must satisfy. They are not new coverage and are not claimed as such.
//
// So: a new spec landing here IS the bug this file exists to catch. When it goes
// red, the fix is to disposition the model, not to add it to an exception list.
//
// Deliberately NOT asserted: anything needing a GPU, a network fetch, or a
// checkpoint on disk. Those belong to validate-model.mjs and the kernel suites.
// This is the cheap always-runnable half.

import { describe, expect, it } from 'vitest'
import { SHIPPED_MODELS, specForParam, modelBranding, quantLabel, opfsDirFor, kvBytesPerTokenShown } from '../../src/zero-tvm/model-registry.ts'
import { PHI3 } from '../../src/compiler/model-spec.ts'

/** Chat templates a renderer actually branches on. `toolDialectFor` defaults to
 *  'chatml-json' for anything else, silently, so this list is the only thing
 *  standing between a new spec and the wrong dialect. Add a branch first. */
const KNOWN_CHAT_TEMPLATES = ['chatml', 'chatml-q35', 'chatml-q38', 'llama3', 'phi3', 'deepseek']
const PHI3_ID = PHI3.id
const PHI3_BRAND = modelBranding(PHI3)

const entries = SHIPPED_MODELS.map((m) => ({ ...m, label: m.param || '(default)' }))

describe('the registry itself', () => {
  it('is not empty, and this file is not vacuous', () => {
    // Guards the guard: if the import ever resolves to an empty array, the
    // describe.each below silently runs zero cases and the file reports green.
    expect(entries.length).toBeGreaterThanOrEqual(11)
  })

  it('every ?model= param is unique', () => {
    const seen = new Map<string, string[]>()
    for (const e of entries) seen.set(e.param, [...(seen.get(e.param) ?? []), e.spec.id])
    const dupes = [...seen].filter(([, ids]) => ids.length > 1)
    expect(dupes, `duplicate ?model= params: ${dupes.map(([p, ids]) => `${p || '(default)'} → ${ids.join(' + ')}`).join('; ')}. `
      + 'specForParam returns the FIRST match, so the later spec is unreachable by URL.').toEqual([])
  })

  it('every spec id is unique', () => {
    const ids = entries.map((e) => e.spec.id)
    expect(new Set(ids).size, `spec ids collide: ${ids.join(', ')}`).toBe(ids.length)
  })

  it('every OPFS weight directory is unique', () => {
    // Honest scope: opfsDirFor is currently injective in spec.id, so given the
    // id-uniqueness test above this cannot fail independently TODAY. It is here
    // for the change that would break that — a truncation, a slug, a
    // lower-casing, or a revision suffix that collapses two ids. A collision
    // means the second model to load reads the first's tensors: fluent wrong
    // output, the failure mode docs/VERIFICATION.md opens with.
    const dirs = entries.map((e) => ({ dir: opfsDirFor(e.spec), id: e.spec.id }))
    const byDir = new Map<string, string[]>()
    for (const d of dirs) byDir.set(d.dir, [...(byDir.get(d.dir) ?? []), d.id])
    const clashes = [...byDir].filter(([, ids]) => ids.length > 1)
    expect(clashes, `OPFS directory collision: ${clashes.map(([d, ids]) => `${d} ← ${ids.join(' + ')}`).join('; ')}. `
      + 'Two models sharing a weight cache means one loads the other tensors.').toEqual([])
  })
})

describe.each(entries)('?model=$label', ({ param, spec }) => {
  it('round-trips through specForParam', () => {
    // Every param must resolve to its OWN spec. specForParam falls back to
    // Phi-3 on a miss, so a typo'd param silently serves the default rather
    // than failing — that fallback is the reason this assertion exists.
    expect(specForParam(param).id).toBe(spec.id)
  })

  it('has its OWN branding, not the Phi-3 fallback', () => {
    // modelBranding returns BRANDINGS[spec.id] ?? BRANDINGS[PHI3.id], so an
    // unbranded spec renders Phi-3's name, size and rate on its card and passes
    // any "is it truthy" check. Comparing against Phi-3's own branding is how
    // the fallback is detected without exporting the table.
    const b = modelBranding(spec)
    expect(b.name, `${spec.id} has no display name`).toBeTruthy()
    expect(b.params, `${spec.id} has no parameter-count line`).toBeTruthy()
    expect(b.sizeLabel, `${spec.id} has no size label; the download figure is the whole decision on that card`).toBeTruthy()
    if (spec.id !== PHI3_ID) {
      expect(b, `${spec.id} has no BRANDINGS entry — it is rendering Phi-3's card `
        + '(name, size, rate) on the roster and in the switcher').not.toEqual(PHI3_BRAND)
    }
  })

  it('has a quantisation label', () => {
    // Derived from the spec, never typed: the 3-bit build was labelled the same
    // as the 4-bit one once, and the mutation gate still reinstates that.
    expect(quantLabel(spec), `${spec.id} renders no quant label`).toBeTruthy()
  })

  it('reports a finite KV cost per token, both cache modes', () => {
    for (const int8 of [false, true]) {
      const v = kvBytesPerTokenShown(spec, int8)
      expect(Number.isFinite(v), `${spec.id} kvBytesPerToken(int8=${int8}) is not finite`).toBe(true)
      expect(v, `${spec.id} kvBytesPerToken(int8=${int8}) is not positive`).toBeGreaterThan(0)
    }
    // int8 must be cheaper than f16, or the label is lying about the trade.
    expect(kvBytesPerTokenShown(spec, true))
      .toBeLessThan(kvBytesPerTokenShown(spec, false))
  })

  it('declares a chat template the renderer actually knows', () => {
    // NOT `expect([...legal dialects]).toContain(toolDialectFor(id))` — that is
    // a tautology, because toolDialectFor returns 'chatml-json' for every id it
    // does not recognise. A twelfth spec with chatTemplateId 'gemma3' would
    // have passed it while silently getting the Qwen3-era JSON dialect.
    expect(KNOWN_CHAT_TEMPLATES, `${spec.id} declares chatTemplateId '${spec.chatTemplateId}', `
      + 'which no renderer branches on — it will silently take the chatml-json default. '
      + 'Add a branch and list it here, or use an existing template.')
      .toContain(spec.chatTemplateId)
  })

  it('declares dimensions the kernels can tile', () => {
    // The rules docs/COMPAT.md and constraints.ts state, at the moduli they
    // actually state them. An earlier draft asserted `d % 64` where the rule is
    // `% 256`, and asserted `qDim === heads * headDim`, which makeModelSpec
    // COMPUTES — a tautology. Two others (heads % kvHeads, headDim) cannot fail
    // here either, because makeModelSpec throws on them at import, so they are
    // left out rather than left in looking like coverage.
    //
    // add-model.mjs checks these for a CANDIDATE. Nothing re-checked the specs
    // already in the tree, which is what this is for.
    expect(spec.headDim, `${spec.id}: headDim ${spec.headDim} > 256 — per-head shared memory in attention.wgsl`).toBeLessThanOrEqual(256)
    expect(spec.d % 256, `${spec.id}: d ${spec.d} % 256 — add_norm/embedding grids are d/256 workgroups with no tail`).toBe(0)
    expect(spec.qkvDim % 256, `${spec.id}: qkvDim ${spec.qkvDim} % 256 — the rope grid has no tail`).toBe(0)
    expect(spec.kvDim % 256, `${spec.id}: kvDim ${spec.kvDim} % 256 — the kv_append grid has no tail`).toBe(0)
    // Scoped exactly as constraints.ts scopes it: only where an LM head runs.
    // An embedding model's output is the pooled hidden state and its lm_head is
    // never dispatched, so qwen3-embedding's vocab of 151669 is not a defect —
    // asserting it flatly made this test red against a correct spec, which is
    // the checker being wrong in the expensive direction.
    if (!spec.embeddingOnly) {
      expect(spec.vocab % 4, `${spec.id}: vocab ${spec.vocab} % 4 — lm_head runs rowsPerWG=4 and the remainder rows drop silently`).toBe(0)
    }
    // Every matmul K: affine scale groups are 64 wide.
    for (const [name, k] of [['d', spec.d], ['qDim', spec.qDim], ['ffn', spec.ffn],
      ...(spec.gdn ? [['gdnVDim', spec.gdnVDim] as const] : [])] as const) {
      expect(k % 64, `${spec.id}: matmul K ${name}=${k} % 64 — affine scale groups are 64 wide`).toBe(0)
    }
  })

  it('declares a hybrid GDN shape only when it has GDN layers, and vice versa', () => {
    // gdnProjRows is DERIVED for every spec, including pure-attention ones, so
    // reading it without checking layerKinds is how a non-hybrid gets treated
    // as a hybrid. Recorded because that exact confusion reached a PR body.
    const hasGdnLayers = spec.layerKinds.includes('gdn')
    expect(hasGdnLayers, `${spec.id}: gdn dims declared but no gdn layer kinds (or the reverse)`)
      .toBe(spec.gdn != null)
  })
})
