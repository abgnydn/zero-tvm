// `share.html?layers=` WAS UNBOUNDED, SO A CRAFTED LINK LIED ABOUT THE
// DOWNLOAD AND DELETED THE RAM WARNING.
//
// `stageRangeFrom` (room-url.ts) is a PARSER: it accepts any `\d+-\d+` because
// it has no spec to check a range against. share.ts checked one thing —
// `start !== 0` — and nothing at all bounded `end`. Measured on qwen38
// (64 layers), before the fix:
//
//   share.html?model=qwen38
//     #gate-weights  "The weights download once (~14.1 GB)…"
//     #gate-ram      "needs ~18 GB free RAM"
//
//   share.html?model=qwen38&layers=0-64
//     #gate-weights  "Layers 0-64 of 64 download once — a slice of the full
//                     ~14.1 GB, NOT ALL OF IT"
//     #gate-ram      HIDDEN
//
//   share.html?model=qwen38&layers=0-9999
//     #gate-weights  the same false sentence, for a range past the checkpoint
//     #gate-ram      HIDDEN
//
// 0-64 IS the whole model. The link understated a 14.1 GB download as a slice
// of itself and removed the one warning a visitor cannot undo by waiting —
// `confirmDownload` suppresses `brand.ramNote` whenever a stage is set, because
// a whole-checkpoint RAM figure is false FOR A REAL STAGE. And 0-9999 made
// `planModel` emit plans past `spec.layers`: the loader fetched all 64 layers
// and then died.
//
// Same class as the `?split=`/`?stage=` hardening the entrance just received —
// this is the sibling key on the surface the entrance's own room CTA links to,
// and the ruling is the one splitFor already made: A MALFORMED RANGE IS NO
// RANGE. Boot the whole model with the honest whole-model gate; do not boot a
// broken range.
//
// What this file holds and what it does not: the RULE is pure and is asserted
// directly. That share.ts actually routes through it — rather than keeping its
// own unbounded copy — is a source check, because share.ts cannot be imported
// headlessly (`signalEnv()` reads `location` at module scope). The rendered
// gate copy for the three URLs above was checked against a real Chrome.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { stageFor } from '../../src/zero-tvm/stage-range.ts'

const ROOT = join(import.meta.dirname, '../..')
const share = (): string => readFileSync(join(ROOT, 'src/zero-tvm/share.ts'), 'utf8')

/** qwen38's layer count — the checkpoint the defect was measured on. */
const L = 64

describe('a hosting stage is a SLICE, never the whole checkpoint', () => {
  it('keeps an honest slice', () => {
    expect(stageFor({ start: 0, end: 32 }, L, 'host')).toEqual({ start: 0, end: 32 })
  })

  it('keeps the largest honest slice, one layer short of the model', () => {
    expect(stageFor({ start: 0, end: L - 1 }, L, 'host')).toEqual({ start: 0, end: L - 1 })
  })

  it('REFUSES a range that is the whole model — the reported link', () => {
    // ?layers=0-64 on a 64-layer checkpoint. Null means "no stage": the whole
    // model, quoted at its real size, with its RAM note intact.
    expect(stageFor({ start: 0, end: L }, L, 'host')).toBeNull()
  })

  it('REFUSES a range past the end of the checkpoint', () => {
    expect(stageFor({ start: 0, end: 9999 }, L, 'host')).toBeNull()
    expect(stageFor({ start: 0, end: L + 1 }, L, 'host')).toBeNull()
  })
})

describe('a helper legitimately ends the model', () => {
  it('keeps the far half of a split', () => {
    // ?layers=32-64#room is the whole point of a helper: it holds the layers
    // the host does not. Bounding it at `< layers` would break every split.
    expect(stageFor({ start: 32, end: L }, L, 'helper')).toEqual({ start: 32, end: L })
  })

  it('still refuses a range past the end', () => {
    expect(stageFor({ start: 32, end: L + 1 }, L, 'helper')).toBeNull()
    expect(stageFor({ start: 8, end: 9999 }, L, 'helper')).toBeNull()
  })
})

describe('degenerate ranges are no range at all', () => {
  it('no ?layers= is no stage', () => {
    expect(stageFor(null, L, 'host')).toBeNull()
    expect(stageFor(null, L, 'helper')).toBeNull()
  })

  it('empty and descending ranges', () => {
    expect(stageFor({ start: 8, end: 8 }, L, 'helper')).toBeNull()
    expect(stageFor({ start: 32, end: 8 }, L, 'helper')).toBeNull()
  })

  it('non-integers and negatives', () => {
    // stageRangeFrom's own `\d+-\d+` cannot produce these today. The rule does
    // not lean on that: it is the validator, and a validator that is only
    // correct for one caller's regex is a check waiting to be wrong.
    expect(stageFor({ start: 0.5, end: 32 }, L, 'host')).toBeNull()
    expect(stageFor({ start: 0, end: 32.5 }, L, 'host')).toBeNull()
    expect(stageFor({ start: -1, end: 32 }, L, 'host')).toBeNull()
    expect(stageFor({ start: 0, end: Infinity }, L, 'host')).toBeNull()
  })
})

describe('share.html routes through the bound', () => {
  it('imports the rule', () => {
    expect(share()).toMatch(/import\s*\{[^}]*\bstageFor\b[^}]*\}\s*from\s*'\.\/stage-range\.js'/)
  })

  it('hands the ROLES a bounded range, never the parser\'s raw one', () => {
    // The three consumers of a range on this page — the title and sheet, the
    // consent gate, and the loader's layerRange — all read whatever runHost and
    // runHelper are handed. Bounding it at the routing point is what makes them
    // agree; a check further in would leave the sheet already painted with a
    // range the loader then refuses.
    const src = share()
    expect(src).toMatch(/stageFor\(/)
    // The raw parse must not reach a role directly any more.
    expect(src).not.toMatch(/runHost\(room,\s*stageRangeFrom\(/)
    expect(src).not.toMatch(/runHelper\(room!,\s*stage!\)/)
  })
})
