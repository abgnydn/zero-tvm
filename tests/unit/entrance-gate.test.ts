// THE CONSENT GATE HAS TO BE A CONTROL, NOT A PICTURE OF ONE.
//
// `?chat=1` is a link a stranger can send. It used to click ENTER for you, so
// landing.ts grew `openUrlGate` — and the first version hid the two verbs with
// `show('.cs-verbs', false)`, which sets `display:none`. That hides pixels. It
// is not a control, and three ways past it were measured in a browser
// (tests/e2e/gate-holds.mjs, which is the red-before evidence):
//
//   /?model=qwen38&chat=1 — focus #model-browser, press Enter
//       chatting: true, booted "Qwen3.8-27B", gate still on screen, ?chat=1
//       still in the URL. `#model-browser` has tabIndex = 0, so ANY click on
//       the scene focuses it — including the click that dismisses the splash —
//       and the keydown handler synthesised a click on the hidden CTA without
//       ever asking whether the gate was open. That path also skipped the
//       history.replaceState that strips chat/room, so the "⟨ Roster cannot
//       escape" bug from the round before came back on it.
//
//   /?model=llama32&chat=1 — ArrowDown x2, then the gate's OWN button
//       agreed: Llama-3.2-1B (~0.6 GB, ~528 MB KV).  booted: Qwen3.6-35B-A3B
//       (~16.4 GB, ~20 GB of free RAM). The gate snapshotted its text at show
//       time and never repainted; the accept synthesised a click on `.mb-cta`,
//       which resolved the LIVE roster all over again. Agreement and action
//       were two different reads of two different states.
//
//   same, but clicking a ROSTER CARD — no keyboard involved at all. Any fix
//   that only guards the keydown handler leaves this one open.
//
// WHAT IS PINNED HERE, AND WHAT IS NOT.
//
// The rules below are the two the fix turns on, extracted so they are
// decidable without a GPU or a DOM — the same move the round before made with
// `entranceIntent`:
//
//   keyIntent()     the scene's keyboard as a DECISION rather than as a
//                   listener body. Refusing while gated is the mechanism that
//                   makes the roster unable to move under a dialog.
//   bootPlanFor()   the ONE resolver from a Selection to what boots. The
//                   gate's sentence (gateCopy) and the boot (enter()) both
//                   take a BootPlan, so they cannot name different models.
//
// What these CANNOT hold is the wiring: that `enter()` is really the only
// call site, that showModal() really makes the document inert, that the
// verbs really come back on decline. Those are DOM facts and they are checked
// in a real browser by tests/e2e/gate-holds.mjs.

import { describe, expect, it } from 'vitest'
import { GROUPS, type Selection, bootPlanFor, gateCopy, keyIntent } from '../../src/landing.js'
import { specForParam } from '../../src/zero-tvm/model-registry.js'
import { ctxFrom } from '../../src/zero-tvm/room-url.js'

const specOf = (i: { gi: number; vi: number }) => GROUPS[i.gi].variants[i.vi].spec
/** Every roster slot, as the Selection the scene would hold on it. */
const SLOTS: Selection[] = GROUPS.flatMap((g, gi) =>
  g.variants.map((_, vi) => ({ gi, vi, mi: 0, xi: 0 })))
const groupOf = (name: string): number => {
  const i = GROUPS.findIndex((g) => g.name === name)
  if (i < 0) throw new Error(`no roster group named ${name}`)
  return i
}
const OPEN = { chatting: false, gated: false, swarm: false }

describe('the scene refuses every key while the gate is up', () => {
  it('Enter does not reach the CTA — the measured HIGH-1 bypass', () => {
    // The exact sequence: the splash click focused #model-browser (a DIV, and
    // tabIndex = 0 makes it focusable), then one Enter. Nothing else changed.
    expect(keyIntent('Enter', 'DIV', OPEN)).toBe('enter')
    expect(keyIntent('Enter', 'DIV', { ...OPEN, gated: true })).toBe('ignore')
  })

  it('the arrow keys do not walk the roster under a dialog someone is reading', () => {
    // These are what a visitor presses to SCROLL a full-screen scene, which is
    // why this was reachable by accident and not only on purpose.
    for (const k of ['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight']) {
      expect(keyIntent(k, 'DIV', OPEN), k).not.toBe('ignore')
      expect(keyIntent(k, 'DIV', { ...OPEN, gated: true }), k).toBe('ignore')
    }
  })

  it('refuses EVERY key, not a list of the ones that were reported', () => {
    // A blocklist of the three keys in the bug report would pass the two tests
    // above and leave the next key someone wires straight through.
    for (const k of ['Enter', ' ', 'Escape', 'Home', 'End', 'PageDown', 'a', 'Tab',
      'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']) {
      expect(keyIntent(k, 'DIV', { ...OPEN, gated: true }), k).toBe('ignore')
      expect(keyIntent(k, 'BUTTON', { ...OPEN, gated: true }), `BUTTON ${k}`).toBe('ignore')
    }
  })

  it('gated outranks swarm — Escape belongs to the dialog, not to the mode', () => {
    // The dialog's own `cancel` event is the decline path. If this handler ate
    // Escape first it would exit the swarm behind an open consent dialog and
    // leave the dialog up over a scene that had changed.
    expect(keyIntent('Escape', 'DIV', { ...OPEN, swarm: true })).toBe('exit-swarm')
    expect(keyIntent('Escape', 'DIV', { ...OPEN, swarm: true, gated: true })).toBe('ignore')
  })

  it('still does its job when no gate is up', () => {
    // The guard must not be a way of switching the scene's keyboard off.
    expect(keyIntent('ArrowDown', 'DIV', OPEN)).toBe('next')
    expect(keyIntent('ArrowUp', 'DIV', OPEN)).toBe('prev')
    expect(keyIntent('Enter', 'DIV', OPEN)).toBe('enter')
    // Pre-existing rules, kept: the composer owns the keyboard in chat mode,
    // a text field owns it anywhere (the swarm sheet has one), and Enter on a
    // BUTTON or an A is that control's own activation, not the scene's.
    expect(keyIntent('ArrowDown', 'DIV', { ...OPEN, chatting: true })).toBe('ignore')
    expect(keyIntent('ArrowDown', 'INPUT', OPEN)).toBe('ignore')
    expect(keyIntent('ArrowDown', 'TEXTAREA', OPEN)).toBe('ignore')
    expect(keyIntent('Enter', 'BUTTON', OPEN)).toBe('ignore')
    expect(keyIntent('Enter', 'A', OPEN)).toBe('ignore')
    expect(keyIntent('Escape', 'DIV', OPEN)).toBe('ignore')
  })
})

describe('the gate says what it will boot', () => {
  it('quotes ITS OWN plan and no other roster slot', () => {
    // The bug was not that the sentence was wrong for some model — it was
    // right for Llama-3.2-1B. It was that the sentence and the boot read two
    // different states. One BootPlan feeds both now, so the sweep below asks
    // the only question left: can the copy name anything the plan does not?
    const names = new Set(GROUPS.map((g) => g.name))
    for (const sel of SLOTS) {
      const plan = bootPlanFor(sel, null)
      const copy = gateCopy(plan, { room: false, cached: false, stage: null, int8: true })
      expect(plan.name, `slot ${sel.gi}/${sel.vi}`).toBe(GROUPS[sel.gi].name)
      expect(copy.what).toContain(plan.name)
      expect(copy.what).toContain(plan.sizeLabel)
      expect(copy.title).toContain(plan.name)
      for (const other of names) {
        // Substring, not equality: "Qwen3-4B" is inside "Qwen3-4B" only, but
        // "Qwen3.6-35B-A3B" would swallow a sloppier check. Skip the names
        // that legitimately contain this one.
        if (other === plan.name || plan.name.includes(other) || other.includes(plan.name)) continue
        expect(copy.what, `${plan.name} must not mention ${other}`).not.toContain(other)
      }
    }
  })

  it('prices the window the plan actually allocates', () => {
    const sel: Selection = { gi: groupOf('Llama-3.2-1B'), vi: 0, mi: 0, xi: 0 }
    const plan = bootPlanFor(sel, null)
    const copy = gateCopy(plan, { room: false, cached: false, stage: null, int8: true })
    // The measured sentence, whose second half is the eager KV allocation.
    expect(copy.what).toContain('This link asks to run Llama-3.2-1B')
    expect(copy.cost).toContain('32k context')
    expect(copy.cost).toMatch(/allocated at boot/)
    expect(plan.ctxTokens).toBe(specOf(sel).maxContext)
  })

  it('a STAGE is priced as a slice, and says how big a slice', () => {
    // Quoting the whole checkpoint's size for a stage is the figure that once
    // promised a phone 14.1 GB for a fraction of it. Dropping the RAM note for
    // any stage was the overcorrection: `?split=0,63,64&stage=0` fetched ~13.9
    // of 14.1 GB and read exactly like this 12% one. The rule is stage-range.ts's
    // now, and tests/unit/stage-consent.test.ts holds it — what is pinned here
    // is that the entrance's gate goes through it.
    const sel = SLOTS.find((s) => bootPlanFor(s, null).ramNote !== '')!
    const plan = bootPlanFor(sel, null)
    const whole = gateCopy(plan, { room: false, cached: false, stage: null, int8: true })
    const slice = gateCopy(plan, { room: false, cached: false, stage: { start: 0, end: 8 }, int8: true })
    expect(whole.cost).toContain(plan.ramNote)
    expect(slice.cost).toContain(plan.ramNote)
    expect(slice.cost).toContain('whole checkpoint')
    expect(slice.what).toContain('Layers 0–8')
    expect(slice.what).toMatch(/\d+% of this checkpoint's layers/)
  })

  it('being cached changes the WORDING, never whether the question is put', () => {
    const plan = bootPlanFor(SLOTS[0], null)
    const cold = gateCopy(plan, { room: false, cached: false, stage: null, int8: true })
    const warm = gateCopy(plan, { room: false, cached: true, stage: null, int8: true })
    expect(cold.go).toBe('Download & enter →')
    expect(warm.go).toBe('Enter chat →')
    expect(warm.what).toContain('already cached on this device')
    // Both still say what is being agreed to, and both still have a button.
    for (const c of [cold, warm]) expect(c.what).toContain(plan.name)
    const room = gateCopy(plan, { room: true, cached: false, stage: null, int8: true })
    expect(room.go).toBe('Download & open a room →')
  })
})

describe('a plan is the same model wherever it is read', () => {
  it('the query it writes resolves back to the spec it holds', () => {
    // `plan.query` is three things at once: the CTA's href, the link the room
    // verb opens, and where a failed in-place mount navigates. All three land
    // on a page with NO ROSTER — zero-tvm.html or share.html — so the reader
    // to ask is `specForParam`, which is what those pages ask. If it answered
    // a different model, a middle-click and an in-place ENTER on the same card
    // would boot different checkpoints: the gate's defect, one URL further out.
    //
    // Note which reader this is NOT. Running plan.query back through
    // `entranceIntent` fails for the default model, correctly: Phi-3's param
    // is '', so the plan writes no `model=` key, and an ABSENT key is the
    // select screen's own question ("which character does it open on"), which
    // the roster answers with its lead. Two readers, two questions — the split
    // entrance-url-grammar.test.ts exists to keep from being collapsed.
    for (const sel of SLOTS) {
      const plan = bootPlanFor(sel, null)
      const named = new URLSearchParams(plan.query).get('model')
      expect(specForParam(named).id, plan.query).toBe(plan.spec.id)
    }
  })

  it('never writes a ?ctx= the one reader of that key would drop', () => {
    // ctxFrom refuses non-integers and anything under MIN_CTX. A link the
    // entrance emits that its own reader discards is exactly the "two tabs
    // disagreeing about a number neither prints" failure.
    for (const sel of SLOTS) {
      for (let xi = 0; xi < 4; xi++) {
        const plan = bootPlanFor({ ...sel, xi }, null)
        const read = ctxFrom(plan.query)
        if (read === null) expect(plan.query, `xi=${xi}`).not.toContain('ctx=')
        else expect(read, plan.query).toBe(plan.ctxTokens)
      }
    }
  })
})
