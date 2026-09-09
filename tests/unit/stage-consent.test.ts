// A STAGE IS NOT A SIZE-FREE PASS.
//
// stage-range.ts bounded `?layers=` against the checkpoint it names, which
// stopped `0-64` on a 64-layer model. It did not touch what a bounded range is
// allowed to BUY, and the two consent gates suppressed the whole-checkpoint RAM
// note for ANY non-null stage. Measured in a real Chrome (the red-before run of
// tests/e2e/stage-consent-holds.mjs, which is where these three came from):
//
//   /share.html?model=qwen38                 #gate-ram "needs ~18 GB free RAM"
//   /share.html?model=qwen38&layers=0-64     #gate-ram "needs ~18 GB free RAM"
//                                                      (the bound catches it)
//   /share.html?model=qwen38&layers=0-63     #gate-ram HIDDEN
//        #gate-weights "Layers 0-63 of 64 download once — a slice of the full
//                       ~14.1 GB, not all of it — …"
//   /share.html?model=qwen38&layers=0-8      the same sentence, word for word
//
// `0-63 of 64` is a LEGITIMATE host stage in a two-machine split, so a tighter
// `end` bound is not the answer — the SUPPRESSION RULE was. ~13.9 of 14.1 GB
// read exactly like a twelfth of it, and the one limit a visitor cannot undo by
// waiting was the half that went missing.
//
// THE RULING, and it is one rule on both surfaces because both call one
// function: a stage STATES ITS SHARE of the checkpoint's layers, and the
// whole-checkpoint RAM figure is never deleted — it is LABELLED as the whole
// checkpoint's. That keeps the fix that made the suppression look right (the
// iPhone told to approve "~14.1 GB" for one layer, 2026-08-29): nothing here
// claims a per-stage byte count, which is unknowable before the safetensors
// headers are read. It states the checkpoint's own two figures, says what
// fraction of it this device is taking, and lets the reader divide.
//
// The second rule is the one `stageFor` inherited neither of. landing.ts's
// `splitFor` refuses a split on a checkpoint the loader cannot CUT
// (`canSplitAcrossDevices`) and refuses a host stage that does not start at
// layer 0; share.html's routing point had neither, so `?model=&layers=0-31`
// painted a whole consent screen for a split Phi-3 cannot do, and
// `?model=qwen38&layers=8-40` painted one and then threw raw engine prose out
// of runHost.
//
// What is NOT here: that the rendered gates actually show this. share.ts cannot
// be imported headlessly (`signalEnv()` reads `location` at module scope), so
// its half is a source check, and both gates were read in a browser by
// tests/e2e/stage-consent-holds.mjs.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { GROUPS, type Selection, bootPlanFor, entranceIntent, gateCopy } from '../../src/landing.js'
import { stageFor, stageGateCopy } from '../../src/zero-tvm/stage-range.js'
import { canSplitAcrossDevices, specForParam } from '../../src/zero-tvm/model-registry.js'

const ROOT = join(import.meta.dirname, '../..')
const share = (): string => readFileSync(join(ROOT, 'src/zero-tvm/share.ts'), 'utf8')

/** qwen38 — 64 layers, ~14.1 GB, and the only kind of model that has a RAM
 *  note to lose. The checkpoint every measurement above was taken on. */
const SEL: Selection = (() => {
  const s = GROUPS.flatMap((g, gi) => g.variants.map((_, vi) => ({ gi, vi, mi: 0, xi: 0 })))
    .find((x) => bootPlanFor(x, null).spec.id === specForParam('qwen38').id)
  if (!s) throw new Error('no roster slot for qwen38')
  return s
})()
const PLAN = bootPlanFor(SEL, null)
const L = PLAN.spec.layers

const copy = (stage: { start: number; end: number } | null, cached = false) =>
  gateCopy(PLAN, { room: false, cached, stage, int8: true })

describe('the premise', () => {
  it('the model under test carries both whole-checkpoint figures', () => {
    // Without these the assertions below would pass while checking nothing.
    expect(PLAN.ramNote).toBeTruthy()
    expect(PLAN.sizeLabel).toBeTruthy()
    expect(L).toBeGreaterThan(2)
  })
})

describe('a stage states its share, and never buys silence on RAM', () => {
  it('the whole model is unchanged', () => {
    const c = copy(null)
    expect(c.what).toContain(`(${PLAN.sizeLabel})`)
    expect(c.cost).toContain(PLAN.ramNote)
  })

  it('KEEPS the RAM limit for a stage that is nearly the whole checkpoint', () => {
    // ?layers=0-63 of 64. The bound lets this through and must: it is the
    // host's half of a two-machine split. What it may not do is delete the
    // ~18 GB.
    const c = copy({ start: 0, end: L - 1 })
    expect(c.cost).toContain(PLAN.ramNote)
  })

  it('keeps it for a SMALL stage too — labelled, not deleted', () => {
    // The phone holding one layer is not told it needs 18 GB. It is told what
    // the whole checkpoint needs, said as the whole checkpoint's figure, next
    // to the fraction this device is taking.
    const c = copy({ start: 0, end: 1 })
    expect(c.cost).toContain(PLAN.ramNote)
    expect(c.cost).toContain('whole checkpoint')
  })

  it('a 98% stage does not READ like a 13% one', () => {
    // 63/64 and 8/64 — the two URLs measured in the browser, which produced
    // one sentence between them.
    const big = copy({ start: 0, end: L - 1 })
    const small = copy({ start: 0, end: 8 })
    expect(big.what).not.toBe(small.what)
    expect(big.what).toContain('98%')
    expect(small.what).toContain('13%')
  })

  it('states no per-stage BYTE figure — the one thing that is not knowable yet', () => {
    // The size that appears is the CHECKPOINT's, and it is named as the
    // checkpoint's. A stage's own bytes come from the safetensors headers,
    // read after consent.
    const c = copy({ start: 0, end: 8 })
    const gb = c.what.match(/[\d.]+\s*GB/g) ?? []
    expect(gb).toEqual([PLAN.sizeLabel.replace('~', '')])
    expect(c.what).toContain('full')
  })

  it('being cached still changes the WORDING, not whether the share is stated', () => {
    const warm = copy({ start: 0, end: L - 1 }, true)
    expect(warm.what).toContain('already cached on this device')
    expect(warm.what).toContain('98%')
    expect(warm.cost).toContain(PLAN.ramNote)
  })
})

describe('the two gates cannot drift: one generator, both surfaces', () => {
  it('the rule rounds the share off the layer counts it prints', () => {
    const o = { layers: 64, cached: false, sizeLabel: '~14.1 GB', ramNote: 'needs ~18 GB free RAM' }
    expect(stageGateCopy({ ...o, stage: { start: 0, end: 63 } }).weights).toContain('98%')
    expect(stageGateCopy({ ...o, stage: { start: 0, end: 8 } }).weights).toContain('13%')
    expect(stageGateCopy({ ...o, stage: { start: 32, end: 64 } }).weights).toContain('50%')
    // A helper legitimately holding the whole thing says so.
    expect(stageGateCopy({ ...o, stage: { start: 0, end: 64 } }).weights).toContain('100%')
  })

  it('no stage is the whole-model wording, RAM note intact', () => {
    const o = { layers: 64, stage: null, sizeLabel: '~14.1 GB', ramNote: 'needs ~18 GB free RAM' }
    expect(stageGateCopy({ ...o, cached: false }).weights).toContain('(~14.1 GB)')
    expect(stageGateCopy({ ...o, cached: false }).ram).toBe('needs ~18 GB free RAM')
    expect(stageGateCopy({ ...o, cached: true }).weights).toContain('already cached')
  })

  it('a checkpoint with no RAM note gets no invented one', () => {
    const o = { layers: 32, cached: false, sizeLabel: '~2 GB', ramNote: '' }
    expect(stageGateCopy({ ...o, stage: null }).ram).toBe('')
    expect(stageGateCopy({ ...o, stage: { start: 0, end: 8 } }).ram).toBe('')
  })

  it('share.html renders its gate through the same function', () => {
    // The entrance's half is exercised directly above (gateCopy is imported).
    // share.ts cannot be imported here, so what is checked is that its gate
    // has no second copy of the rule: `brand.ramNote` is READ ONCE, and that
    // read is the argument to the shared generator.
    const src = share()
    expect(src).toMatch(/import\s*\{[^}]*\bstageGateCopy\b[^}]*\}\s*from\s*'\.\/stage-range\.js'/)
    expect(src).toMatch(/stageGateCopy\(/)
    expect(src.match(/brand\.ramNote/g) ?? []).toHaveLength(1)
    // …and the RAM paragraph is no longer hidden on the mere presence of one.
    expect(src).not.toMatch(/if\s*\(!stage\s*&&\s*brand\.ramNote\)/)
  })
})

describe('a range on a checkpoint the loader cannot CUT is no range', () => {
  it('the entrance already refuses one, and still does', () => {
    // `?model=&split=0,16,32&stage=0` — Phi-3 ships MLC shards; loadWeights
    // refuses a layerRange on it. splitFor has had this rule since it was
    // written, and it is the sibling of the one share.html was missing.
    expect(canSplitAcrossDevices(specForParam(''))).toBe(false)
    expect(entranceIntent('?model=&split=0,16,32&stage=0', '').split).toBeNull()
  })

  it('share.html asks the same registry predicate before it routes a range', () => {
    // Measured before this: /share.html?model=&layers=0-31 painted a whole
    // consent screen for a split Phi-3 cannot do — title "Phi-3-mini · layers
    // 0–31 of 32", the whole model sold as "a slice … not all of it", the RAM
    // note gone, and a hosting-a-split paragraph — then died in the loader.
    const src = share()
    expect(src).toMatch(/import\s*\{[^}]*\bcanSplitAcrossDevices\b[^}]*\}\s*from\s*'\.\/model-registry\.js'/)
    expect(src).toMatch(/canSplitAcrossDevices\(/)
  })

  it('the rule itself: no cuttable layers, no stage', () => {
    // `null` layers is how a caller says "this checkpoint cannot be cut at
    // all", so the refusal is the module's and not the caller's to remember.
    expect(stageFor({ start: 0, end: 31 }, null, 'host')).toBeNull()
    expect(stageFor({ start: 16, end: 32 }, null, 'helper')).toBeNull()
  })
})

describe('a HOST stage holds the embedding, so it starts at layer 0', () => {
  it('refuses a host range that skips the start of the model', () => {
    // share.ts threw for this — after painting the scene with the stage it was
    // about to refuse. splitFor's ruling is the one that belongs here: a
    // malformed range is NO range, and the whole model with the honest
    // whole-model gate is the fallback.
    expect(stageFor({ start: 8, end: 40 }, L, 'host')).toBeNull()
    expect(stageFor({ start: 1, end: 2 }, L, 'host')).toBeNull()
  })

  it('a HELPER still holds the far half of a split', () => {
    expect(stageFor({ start: 32, end: L }, L, 'helper')).toEqual({ start: 32, end: L })
  })

  it('share.html no longer keeps its own copy of the rule', () => {
    expect(share()).not.toMatch(/a hosting stage must start at layer 0/)
  })
})
