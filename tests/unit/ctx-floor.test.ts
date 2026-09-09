// `?ctx=` HAD NO FLOOR, AND THE ENTRANCE HAD NO MIDDLE.
//
// Two defects at opposite ends of the same key, both measured in a browser
// (tests/e2e/gate-holds.mjs):
//
//   share.html?model=qwen38&ctx=0.5   →  "Context | 16 tokens"
//   zero-tvm.html?model=qwen38&ctx=0.5 → "0K CONTEXT"
//
//       ctxFrom's guard was `n > 0`, which admits fractions and sub-page
//       values. specWithCtx's floor is ONE PAGE, so nothing errored: both
//       surfaces built a 16-token KV cache and then fetched 14.1 GB of weights
//       for a window that cannot hold a prompt. landing-swarm.ts's own context
//       field has clamped to >= 256 since it was written; the URL reader,
//       which is where the same number arrives from a LINK, did not.
//
//   /?model=qwen38&ctx=5000            →  "Context 16k tokens", href drops it
//
//       The entrance honoured `?ctx=` only on an exact match to one of
//       ctxModesOf's three enumerated builds. landing-room.ts writes room
//       chips carrying any ctx in [256, maxContext] and hands the SAME ?ctx=
//       to the other machines — so `?ctx=5000` booted this tab at the spec
//       default while every other machine ran 5000. Two tabs on different KV
//       budgets for one conversation, and neither prints the number: exactly
//       the failure ctxFrom's own docstring exists to prevent, arriving
//       through the one surface that was not reading ctxFrom.
//
// Both are fixed in ctxFrom and in ctxModesOf's `linked` argument, so every
// surface inherits the floor and the entrance stops being a second reader.

import { describe, expect, it } from 'vitest'
import { PHI3 } from '../../src/compiler/model-spec.js'
import { SHIPPED_MODELS, specForParam, specWithCtx } from '../../src/zero-tvm/model-registry.js'
import { MIN_CTX, ctxFrom } from '../../src/zero-tvm/room-url.js'
import { GROUPS, bootPlanFor } from '../../src/landing.js'

const QWEN38 = specForParam('qwen38')
const slotFor = (id: string): { gi: number; vi: number } => {
  for (let g = 0; g < GROUPS.length; g++) {
    const v = GROUPS[g].variants.findIndex((x) => x.spec.id === id)
    if (v >= 0) return { gi: g, vi: v }
  }
  throw new Error(`${id} is not on the roster`)
}

describe('ctxFrom has a floor, and it is a whole number of tokens', () => {
  it('refuses the measured sub-page values', () => {
    // The exact strings from the two surfaces above, plus the neighbours a
    // fix that only special-cased 0.5 would leave open.
    for (const q of ['?ctx=0.5', '?ctx=0.9', '?ctx=1', '?ctx=15', '?ctx=16',
      '?ctx=255', '?ctx=255.9', '?ctx=1e-3', '?ctx=.5']) {
      expect(ctxFrom(q), q).toBeNull()
    }
  })

  it('refuses a fraction even when it is large', () => {
    // `n > 0` let 8192.5 through, and specWithCtx rounds it to a page — so the
    // link and the engine disagreed about the number by design.
    for (const q of ['?ctx=8192.5', '?ctx=32768.0001', '?ctx=1024.5']) {
      expect(ctxFrom(q), q).toBeNull()
    }
    // 32768.0 IS an integer as far as Number is concerned, and that is fine:
    // it is the same budget written differently, not a different budget.
    expect(ctxFrom('?ctx=32768.0')).toBe(32768)
  })

  it('still refuses everything it refused before', () => {
    for (const q of ['', '?ctx=', '?ctx=0', '?ctx=-1', '?ctx=-4096', '?ctx=junk',
      '?ctx=Infinity', '?ctx=NaN', '?model=qwen35']) {
      expect(ctxFrom(q), q).toBeNull()
    }
  })

  it('still reads every budget a link legitimately carries', () => {
    expect(ctxFrom(`?ctx=${MIN_CTX}`)).toBe(MIN_CTX)
    expect(ctxFrom('?ctx=8192')).toBe(8192)
    expect(ctxFrom('?model=qwen3mlx&ctx=262144&layers=0-18')).toBe(262144)
    // The compiled default of every shipped spec, which is what room-host.ts
    // writes into the links it hands out. A floor that clipped one of these
    // would silently drop the host's window on every helper.
    for (const { param, spec } of SHIPPED_MODELS) {
      expect(ctxFrom(`?ctx=${spec.maxContext}`), param || '(default)').toBe(spec.maxContext)
    }
  })

  it('REFUSES rather than clamps — a malformed budget is no budget', () => {
    // Returning MIN_CTX for `?ctx=0.5` would be this module inventing a number
    // the link never asked for, and null is what "use your own default" means
    // downstream. Same rule the entrance applies to a malformed ?split=.
    expect(ctxFrom('?ctx=0.5')).not.toBe(MIN_CTX)
    expect(ctxFrom('?ctx=0.5')).toBeNull()
  })
})

describe('the entrance honours any in-range ?ctx=, not just the three it lists', () => {
  it('the measured link: ?ctx=5000 on qwen38', () => {
    // 5000 is inside [256, 262144] and is not Standard (16384), Long (65536)
    // or Full (262144). KV pages round UP — 16-token pages, so ceil(5000/16)
    // = 313 pages = 5008 tokens — and the EFFECTIVE window is what the chip
    // quotes and what the link carries onward, because that is the number the
    // engine allocates.
    const plan = bootPlanFor({ ...slotFor(QWEN38.id), mi: 0, xi: 0 }, null)
    expect(plan.ctxTokens).toBe(QWEN38.maxContext)      // no link → the default

    const linked = specWithCtx(QWEN38, 5000).maxContext
    expect(linked).toBe(5008)
    // The picker gains the link's window as a fourth build; the entrance opens
    // on it, so xi is its index — the last one, since it is appended and never
    // sorted in (go() resets to index 0 and index 0 must stay the default).
    const withLink = bootPlanFor({ ...slotFor(QWEN38.id), mi: 0, xi: 3 }, 5000)
    expect(withLink.ctxTokens).toBe(5008)
    expect(withLink.query).toContain('ctx=5008')
    expect(ctxFrom(withLink.query)).toBe(5008)
  })

  it('index 0 stays the compiled default, link or no link', () => {
    // go() resets the picker to 0 on every roster move. If the link's window
    // sorted in ahead of Standard, walking the roster would land on a window
    // nobody chose.
    for (const g of GROUPS.keys()) {
      for (const v of GROUPS[g].variants.keys()) {
        const spec = GROUPS[g].variants[v].spec
        const plan = bootPlanFor({ gi: g, vi: v, mi: 0, xi: 0 }, 5000)
        expect(plan.ctxTokens, `${spec.id}`).toBe(spec.maxContext)
        expect(plan.query, spec.id).not.toContain('ctx=')
      }
    }
  })

  it('a link naming a window the roster already lists adds no duplicate', () => {
    // ?ctx=16384 on qwen38 IS Standard. A fourth chip reading the same number
    // would be a rendering fault, and xi=3 would fall off the end.
    const sel = { ...slotFor(QWEN38.id), mi: 0, xi: 3 }
    expect(bootPlanFor(sel, QWEN38.maxContext).ctxTokens).toBe(QWEN38.maxContext)
    expect(bootPlanFor(sel, QWEN38.maxSeq).ctxTokens).toBe(QWEN38.maxContext)
  })

  it('a link over the trained window lands where the engine would put it', () => {
    // specWithCtx clamps to floor(maxSeq / pageSize) pages. The entrance runs
    // the link through the SAME function, so it cannot quote a window the
    // engine will not build — which is how it would come to disagree with
    // zero-tvm.html on the same URL all over again.
    const over = QWEN38.maxSeq * 4
    expect(specWithCtx(QWEN38, over).maxContext).toBe(QWEN38.maxSeq)
    const sel = { ...slotFor(QWEN38.id), mi: 0, xi: 3 }
    expect(bootPlanFor(sel, over).ctxTokens).toBe(QWEN38.maxContext)   // deduped into Full
  })

  it('the default model, whose window is ABOVE its trained length, is unmoved', () => {
    // Phi-3 is the one spec where maxContext (4112, 257 pages of 16) exceeds
    // maxSeq (4096). It lists ONE context build, so a link's window is the
    // second — and the compiled default must survive at index 0 either way.
    const sel = { ...slotFor(PHI3.id), mi: 0, xi: 0 }
    expect(bootPlanFor(sel, null).ctxTokens).toBe(4112)
    expect(bootPlanFor(sel, 1000).ctxTokens).toBe(4112)
    expect(bootPlanFor({ ...sel, xi: 1 }, 1000).ctxTokens).toBe(specWithCtx(PHI3, 1000).maxContext)
  })
})
