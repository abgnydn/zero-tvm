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
// directly, and WHAT REACHES A ROLE is asserted by RUNNING share.ts's own
// routing block. share.ts cannot be imported headlessly (`signalEnv()` reads
// `location` at module scope), so the block is lifted out and evaluated the way
// stage-honesty.test.ts's `renderedAt` evaluates each surface's call site. The
// rendered gate copy for the three URLs above was checked against a real Chrome.
//
// IT USED TO BE THREE REGEXES OVER THE SOURCE — `toMatch(/stageFor\(/)`,
// `not.toMatch(/runHost\(room,\s*stageRangeFrom\(/)`,
// `not.toMatch(/runHelper\(room!,\s*stage!\)/)` — and all three are satisfied by
// `void runHost(room, raw)`, which hands the role the parser's UNBOUNDED range.
// Re-measured here on 2026-09-02 with exactly that mutation applied to
// share.ts: all THREE old assertions PASS and `npm run typecheck` is clean,
// while the assertions below go red on three cases. (The review that found it
// also drove the mutated build in Chrome and reproduced the defect sentence
// verbatim — "Layers 0-9999 of 64 … a slice of the full ~14.1 GB, not all of
// it", RAM note gone; that half is its measurement, not this file's.)
// A regex over a call site cannot see what the call site passes.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import ts from 'typescript'
import { describe, expect, it } from 'vitest'
import { canSplitAcrossDevices, specForParam } from '../../src/zero-tvm/model-registry.ts'
import { roleFor, roomIdFrom, stageRangeFrom } from '../../src/zero-tvm/room-url.ts'
import { stageFor } from '../../src/zero-tvm/stage-range.ts'

const ROOT = join(import.meta.dirname, '../..')
const share = (): string => readFileSync(join(ROOT, 'src/zero-tvm/share.ts'), 'utf8')

/** qwen38's layer count — the checkpoint the defect was measured on. Checked
 *  against the registry below, so a spec change breaks the test rather than
 *  quietly re-pointing every bound in this file at a number that moved. */
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

/**
 * SHARE.HTML'S OWN ROUTING BLOCK, PULLED OUT AND RUN.
 *
 * From `const room = roomIdFrom(` to the end of the `runHost(` call that closes
 * the three-way dispatch — the one region where a parsed range becomes a role's
 * ARGUMENT. Everything downstream (the title, the sheet's Layers row, the
 * consent paragraph, the download gate, the loader's layerRange, the helper
 * links) reads what runHost/runHelper are handed, so this is the value the whole
 * page agrees on.
 *
 * The end anchor stops at `runHost(`, deliberately, and the argument list is
 * taken by BALANCED COUNTING rather than matched: a mutated argument must be
 * RUN here, not fail to be found. `void runHost(room, raw)` is what this exists
 * to catch, and a regex that included `, stage)` would simply stop matching and
 * report "no routing block" — a different failure, and one a reader could
 * mistake for a refactor.
 */
const BEGIN = 'const room = roomIdFrom('
const END = 'void runHost('

function routingBlock(): string {
  const src = share()
  const from = src.indexOf(BEGIN)
  expect(from, `share.ts no longer starts routing with \`${BEGIN}…\``).toBeGreaterThan(-1)
  const call = src.indexOf(END, from)
  expect(call, `share.ts no longer dispatches with \`${END}…\``).toBeGreaterThan(from)
  let depth = 0
  let i = call + END.length - 1
  for (; i < src.length; i++) {
    if (src[i] === '(') depth++
    else if (src[i] === ')' && --depth === 0) break
  }
  expect(depth, `unbalanced \`${END}\` in share.ts`).toBe(0)
  return src.slice(from, i + 1)
}

type Stage = { start: number; end: number } | null
type Routed = { role: 'host' | 'helper' | 'guest'; stage: Stage }

/**
 * Where one URL lands, and with WHAT. Every free name in the block is bound to
 * the real function share.ts imports — only the three roles are stubs, because
 * they are what the assertion reads. The block carries `room!`, so it goes
 * through the actual TypeScript transpiler rather than a hand-rolled strip: a
 * `!` regex that mangled an unrelated `!==` would be a second bug hiding in the
 * test that is supposed to find one.
 */
function route(url: string): Routed {
  const u = new URL(url, 'http://localhost/share.html')
  const js = ts.transpileModule(routingBlock(), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext },
  }).outputText
  const seen: Routed[] = []
  const run = new Function(
    'location', 'roomIdFrom', 'roleFor', 'stageRangeFrom', 'specForParam',
    'canSplitAcrossDevices', 'stageFor', 'runHost', 'runHelper', 'runGuest', js)
  run(
    { search: u.search, hash: u.hash },
    roomIdFrom, roleFor, stageRangeFrom, specForParam, canSplitAcrossDevices, stageFor,
    (_room: string | null, stage: Stage) => { seen.push({ role: 'host', stage }) },
    (_room: string, stage: Stage) => { seen.push({ role: 'helper', stage }) },
    () => { seen.push({ role: 'guest', stage: null }) },
  )
  expect(seen.length, `${url} reached ${seen.length} roles, not exactly 1`).toBe(1)
  return seen[0]
}

describe('share.html hands the ROLE a bounded range', () => {
  /** A real room-id shape: 22 base64url characters, inside ROOM_ID's 16-64. */
  const ROOM = '#Zm9vYmFyYmF6cXV4MDE'

  it('the premises — qwen38 really has L layers, and the parser really is unbounded', () => {
    // Without the first, every bound below is checked against the wrong number.
    // Without the second, `stage: null` could mean the parser stopped producing
    // the range rather than the bound refusing it.
    expect(specForParam('qwen38').layers).toBe(L)
    expect(canSplitAcrossDevices(specForParam('qwen38'))).toBe(true)
    expect(stageRangeFrom('?model=qwen38&layers=0-9999')).toEqual({ start: 0, end: 9999 })
  })

  it('THE REPORTED LINK: ?layers=0-9999 reaches the host as NO stage', () => {
    expect(route('/share.html?model=qwen38&layers=0-9999')).toEqual({ role: 'host', stage: null })
  })

  it('?layers=0-64 — the whole checkpoint — reaches the host as NO stage', () => {
    expect(route(`/share.html?model=qwen38&layers=0-${L}`)).toEqual({ role: 'host', stage: null })
  })

  it('an honest slice reaches the host intact', () => {
    expect(route('/share.html?model=qwen38&layers=0-32'))
      .toEqual({ role: 'host', stage: { start: 0, end: 32 } })
  })

  it('no ?layers= is the whole model with the whole-model gate', () => {
    expect(route('/share.html?model=qwen38')).toEqual({ role: 'host', stage: null })
  })

  it('a helper is handed the far half — the one range that MAY end the model', () => {
    expect(route(`/share.html?model=qwen38&layers=32-${L}${ROOM}`))
      .toEqual({ role: 'helper', stage: { start: 32, end: L } })
  })

  it('a helper whose range does not bound joins as a GUEST, holding nothing', () => {
    // Not the whole model in someone else's room: that is a larger commitment
    // than the link asked for. A guest needs no WebGPU and downloads nothing.
    expect(route(`/share.html?model=qwen38&layers=32-9999${ROOM}`))
      .toEqual({ role: 'guest', stage: null })
  })

  it('a checkpoint the loader cannot cut is handed no stage at all', () => {
    // `?model=` is Phi-3, an MLC checkpoint: canSplitAcrossDevices is false, so
    // the layer count arrives as null. This used to paint a whole consent screen
    // for a split that cannot exist and then die inside loadWeights.
    expect(canSplitAcrossDevices(specForParam(''))).toBe(false)
    expect(route('/share.html?model=&layers=0-16')).toEqual({ role: 'host', stage: null })
  })

  it('and it is THIS module the page routes through, not a local copy', () => {
    // `route` injects the real stageFor, so a share.ts that grew its own broken
    // one would still pass above. This is the half that cannot be run.
    expect(share()).toMatch(/import\s*\{[^}]*\bstageFor\b[^}]*\}\s*from\s*'\.\/stage-range\.js'/)
  })
})
