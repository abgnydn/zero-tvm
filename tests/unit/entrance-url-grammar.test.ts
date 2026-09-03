// THE ENTRANCE READS ITS OWN URL, AND EVERY ANSWER IT GAVE WAS A LIABILITY.
//
// `/` is a link people send each other. Four of the questions a link can ask
// it were answered wrong, and each wrong answer cost bytes, a room, or a way
// back out:
//
//   ?model=      An unknown value fell through to roster slot 0. That used to
//                be Phi-3 and was harmless; the roster reorder made slot 0 the
//                14.1 GB flagship, so `/?model=not-a-model` started a 14.1 GB
//                download while `/zero-tvm.html?model=not-a-model` — which
//                asks the registry — booted Phi-3. Two surfaces, one registry,
//                opposite answers.
//
// ── TWO RULES FOR ?model=, AND THEY MUST STAY TWO ────────────────────────────
//
// The obvious repair is one rule: "always answer specForParam". It is wrong,
// and a later reader WILL try it, so here is why it is wrong.
//
// The two cases are different QUESTIONS:
//
//   NO `model` key at all      "which character does the select screen OPEN
//   (`/`, `/?chat=1`)          on" — a presentation choice this page owns
//                              alone. /zero-tvm.html and share.html have no
//                              roster, so specForParam has no opinion about
//                              it. The answer is the roster's FIRST CARD, the
//                              strongest model this project runs. Phi-3 led
//                              the roster once and b37e849 reordered it
//                              precisely because the weakest shipped model was
//                              the project's first impression; answering
//                              specForParam(null) here would silently put
//                              Phi-3 back on stage and undo that.
//
//   `model` key PRESENT but    "what does an unrecognised ?model= mean" — a
//   NOT ON THE ROSTER          COMPATIBILITY question, and every pre-registry
//   (`?model=bogus`,           URL depends on the fall-through. zero-tvm.html
//    `?model=`, and also       already answers Phi-3, so the entrance must
//    `?model=embed`)           too, and the two surfaces may not disagree for
//                              any value that resolves to a ROSTERED spec.
//
// `?model=embed` is in that list for a different reason and the difference
// matters. It is not unresolvable: `specForParam('embed')` returns the
// embedding spec, and /zero-tvm.html boots it. The ENTRANCE shows Phi-3 for
// it, so the two surfaces genuinely DO disagree there — deliberately. A
// character-select screen for conversation cannot put on stage a model that
// returns a vector and does not speak, so the roster excludes it and the
// entrance falls back rather than promoting whichever model leads the roster
// this month. Anyone "fixing that inconsistency" deletes the exclusion; the
// test below pins the disagreement so they cannot do it quietly.
//
// The split is therefore on PRESENCE of the key, never on whether its value
// resolves. Both halves are pinned below; collapsing them into one rule breaks
// whichever test is not being looked at.
//
//   ?split=      The guard checked only that the last boundary equalled the
//                layer count. `split=4,8,16&stage=0` therefore served layers
//                4-8 from a HOST stage with no embedding (share.ts throws on
//                exactly that); `stage=0.5` indexed the bounds with a
//                fraction and the room plan read "layers undefined-undefined";
//                `split=0,999,16` died inside the weight loader reading
//                'normGamma1' of undefined, before engine-core's own range
//                guard could fire; and nothing checked the checkpoint could be
//                cut at all, so `?model=&split=0,16,32` sent Phi-3 into
//                serving mode to die on "layerRange needs an MLX checkpoint".
//
//   ?chat=1      Clicked ENTER for you. Any link a stranger sent started a
//                multi-gigabyte download and an eager KV allocation with no
//                click anywhere. share.html grew a consent gate for precisely
//                this incident (confirmDownload in share.ts); the entrance,
//                which is the surface people actually link to, had none.
//
//   #swarm       README publishes https://zerotvm.com/#swarm. Only a CLICK on
//                `a[href="#swarm"]` was wired, and render() hides the #swarm
//                fallback section — so the published link opened nothing.
//
// The grammar lives in one pure exported function now, so this file can hold
// all of it. What it CANNOT hold is the wiring: that the gate is what the
// scene actually shows, that the hash actually reaches openSwarm(). Those are
// checked in a browser. What is pinned here is the DECISION each URL produces,
// which is where all four bugs lived.

import { describe, expect, it } from 'vitest'
import { PHI3 } from '../../src/compiler/model-spec.js'
import { SHIPPED_MODELS, canSplitAcrossDevices, specForParam } from '../../src/zero-tvm/model-registry.js'
import { GROUPS, entranceIntent, urlAfterEnter } from '../../src/landing.js'

/** The spec a decision resolves to — the thing that actually boots. */
const specOf = (i: { gi: number; vi: number }) => GROUPS[i.gi].variants[i.vi].spec
const onRoster = (id: string): boolean =>
  GROUPS.some((g) => g.variants.some((v) => v.spec.id === id))

/** A `?model=` value, encoded the way a link carries it. */
const search = (param: string | null, rest = ''): string =>
  `${param === null ? '' : `?model=${encodeURIComponent(param)}`}${rest ? (param === null ? '?' : '&') + rest : ''}`

describe('a ?model= that is PRESENT is the registry\'s question', () => {
  it('resolves every present ?model= the way specForParam does', () => {
    // specForParam IS the answer — /zero-tvm.html, share.html and validate all
    // ask it. The entrance is the only surface that used to answer for itself.
    // Strings only: `null` means the key is ABSENT and is the other rule.
    const present: string[] = [
      '', 'not-a-model', 'qwen38', 'phi3', '0', 'undefined',
      ...SHIPPED_MODELS.map((m) => m.param),
    ]
    for (const param of present) {
      const registry = specForParam(param)
      // The roster deliberately does not carry every shipped spec (the
      // embedding model answers nothing a visitor typed; a pending build is
      // not yet numerics-validated). The entrance cannot put one of those on
      // stage, so it takes the registry's OWN fallback — never whichever model
      // happens to lead the roster this month.
      const want = onRoster(registry.id) ? registry : specForParam(null)
      expect(specOf(entranceIntent(search(param), '')).id, `?model=${param}`).toBe(want.id)
    }
  })

  it('boots Phi-3 for an unresolvable ?model=, exactly like /zero-tvm.html', () => {
    // The measured URLs. `specForParam` cannot name a spec for any of these,
    // so both surfaces take the registry's fallback and agree.
    for (const s of [
      '?model=not-a-model&chat=1',
      '?model=&chat=1',
      '?model=not-a-model',
    ]) {
      expect(specForParam(new URLSearchParams(s).get('model')).id, s).toBe(PHI3.id)
      expect(specOf(entranceIntent(s, '')).id, s).toBe(PHI3.id)
    }
  })

  it('DISAGREES with /zero-tvm.html on ?model=embed, and that is the point', () => {
    // `embed` RESOLVES — it is a real registry entry, and /zero-tvm.html boots
    // it. The entrance does not, because a character-select screen for
    // conversation cannot put on stage a model that returns a vector and does
    // not speak; the roster excludes it (buildGroups skips embeddingOnly) and
    // the entrance takes the registry's own fallback rather than the roster's
    // lead. Both halves are asserted so that "fixing the inconsistency" —
    // answering specForParam here like the other surfaces do — fails loudly
    // instead of quietly deleting the exclusion.
    expect(specForParam('embed').id).toBe('qwen3-embedding-0-6b-4bit-dwq')
    expect(specForParam('embed').embeddingOnly).toBe(true)
    expect(onRoster('qwen3-embedding-0-6b-4bit-dwq')).toBe(false)
    for (const s of ['?model=embed', '?model=embed&chat=1']) {
      expect(specOf(entranceIntent(s, '')).id, s).toBe(PHI3.id)
      expect(specOf(entranceIntent(s, '')).id, s).not.toBe(specForParam('embed').id)
    }
  })
})

describe('an ABSENT ?model= is the select screen\'s own question', () => {
  // NOT specForParam. This is "which character does the roster open on", the
  // one part of the grammar that has no counterpart on /zero-tvm.html — and
  // the reorder in b37e849 exists because Phi-3 leading the roster made the
  // weakest shipped model the project's first impression. A "simplification"
  // that answers specForParam(null) here puts it straight back.
  const lead = GROUPS[0].variants[0].spec

  it('opens on the roster\'s first card, which is not Phi-3', () => {
    expect(lead.id).not.toBe(PHI3.id)
    expect(SHIPPED_MODELS[0].spec.id).toBe(lead.id)
  })

  it('opens on the roster lead for every URL that names no model', () => {
    for (const s of ['', '?chat=1', '?ctx=32768', '?chat=1&room=1', '?kv8=0']) {
      expect(specOf(entranceIntent(s, '')).id, s || '(bare /)').toBe(lead.id)
    }
  })

  it('is decided by the KEY, not by whether a value resolves', () => {
    // The whole rule in two lines: same unresolvable-ness, opposite answers,
    // because one URL asked for a model and the other did not.
    expect(specOf(entranceIntent('?model=', '')).id).toBe(PHI3.id)
    expect(specOf(entranceIntent('', '')).id).toBe(lead.id)
  })
})

// A splittable checkpoint and its layer count, read from the registry rather
// than typed — a hand-copied 36 here would rot the day a checkpoint changes.
const CUTTABLE = SHIPPED_MODELS.find((m) => canSplitAcrossDevices(m.spec) && onRoster(m.spec.id))!
const L = CUTTABLE.spec.layers
const MID = Math.floor(L / 2)

describe('a malformed split is NO split, never a broken one', () => {
  it('accepts a well-formed first stage of a checkpoint that can be cut', () => {
    const i = entranceIntent(search(CUTTABLE.param, `split=0,${MID},${L}&stage=0`), '')
    expect(i.split).toEqual({ bounds: [0, MID, L], index: 0 })
  })

  it('opens the room strip for a split — the other stages need links', () => {
    const i = entranceIntent(search(CUTTABLE.param, `split=0,${MID},${L}&stage=0&chat=1`), '')
    expect(i.enter?.room).toBe(true)
  })

  const bad: [string, string][] = [
    ['a split that does not start at layer 0 — `split=4,8,16` on a 16-layer '
      + 'checkpoint served layers 4-8 from a stage with no embedding', `split=1,${MID},${L}&stage=0`],
    ['a stage that does not start the model — share.ts throws on it, and the '
      + 'room plan hands out a machine-1 link with no room fragment', `split=0,${MID},${L}&stage=1`],
    ['a fractional stage index — bounds[0.5] is undefined', `split=0,${MID},${L}&stage=0.5`],
    ['a negative stage index', `split=0,${MID},${L}&stage=-1`],
    ['a stage index past the last stage', `split=0,${MID},${L}&stage=2`],
    ['a non-numeric stage index', `split=0,${MID},${L}&stage=first`],
    ['boundaries that do not ascend', `split=0,${L + 999},${L}&stage=0`],
    ['an empty stage', `split=0,0,${L}&stage=0`],
    ['fractional boundaries', `split=0,${MID}.5,${L}&stage=0`],
    ['a boundary that is not a number', `split=0,eight,${L}&stage=0`],
    ['one stage, which is not a split', `split=0,${L}&stage=0`],
    ['a last boundary that is not the layer count', `split=0,${MID},${L - 1}&stage=0`],
  ]
  for (const [why, rest] of bad) {
    it(`refuses ${why}`, () => {
      expect(entranceIntent(search(CUTTABLE.param, rest), '').split).toBeNull()
    })
  }

  it('refuses a split of a checkpoint the loader cannot cut', () => {
    // canSplitAcrossDevices exists so the swarm link builder cannot hand out a
    // ?layers= URL that throws at boot. This reader was a second builder that
    // could: Phi-3 entered serving mode and died on "loadWeights: layerRange
    // needs an MLX checkpoint; phi3-mini ships MLC shards".
    const mlc = SHIPPED_MODELS.find((m) => !canSplitAcrossDevices(m.spec) && onRoster(m.spec.id))!
    const n = mlc.spec.layers
    const i = entranceIntent(search(mlc.param, `split=0,${Math.floor(n / 2)},${n}&stage=0`), '')
    expect(canSplitAcrossDevices(specOf(i))).toBe(false)
    expect(i.split).toBeNull()
  })

  it('ignores a split written for a different character', () => {
    // The bounds belong to the model the URL NAMED. Nothing else may wear them.
    expect(entranceIntent(`?split=0,${MID},${L}&stage=0`, '').split).toBeNull()
  })
})

describe('a link is not consent', () => {
  // share.html asks before it spends bytes, and asks EVEN WHEN CACHED — being
  // cached changes the wording, not whether the question is put. The entrance
  // is the surface people link to and it asked nothing at all.
  const links = [
    '?chat=1',
    '?chat=1&room=1',
    '?model=qwen38&chat=1',
    '?model=&chat=1&ctx=32768',
    search(CUTTABLE.param, `split=0,${MID},${L}&stage=0&chat=1`),
  ]
  for (const s of links) {
    it(`asks first on ${s}`, () => {
      expect(entranceIntent(s, '').enter?.act).toBe('consent')
    })
  }

  it('asks for nothing when the URL asks for nothing', () => {
    expect(entranceIntent('', '').enter).toBeNull()
    expect(entranceIntent('?model=qwen38', '').enter).toBeNull()
    expect(entranceIntent('?chat=0', '').enter).toBeNull()
  })

  it('carries the room verb through the gate', () => {
    expect(entranceIntent('?chat=1&room=1', '').enter).toEqual({ room: true, act: 'consent' })
    expect(entranceIntent('?chat=1', '').enter).toEqual({ room: false, act: 'consent' })
  })
})

describe('#swarm is a published link', () => {
  it('opens the mode on a cold load, not only on a click', () => {
    expect(entranceIntent('', '#swarm').swarm).toBe(true)
  })

  it('leaves every other hash alone', () => {
    expect(entranceIntent('', '').swarm).toBe(false)
    expect(entranceIntent('', '#models').swarm).toBe(false)
    expect(entranceIntent('', '#swarming').swarm).toBe(false)
  })
})

describe('the way out of a link that entered', () => {
  // "⟨ Roster" and the /roster command both location.reload(), which keeps the
  // query — so on any ?chat=1 URL the way out walked straight back in. Every
  // link the room plan writes carries chat=1.
  it('drops the keys that mean "act without a click"', () => {
    expect(urlAfterEnter('?model=llama32&chat=1')).toBe('?model=llama32')
    expect(urlAfterEnter('?chat=1')).toBe('')
    expect(urlAfterEnter('?chat=1&room=1')).toBe('')
  })

  it('keeps the character and its build, so the roster comes back as it was', () => {
    const q = new URLSearchParams(urlAfterEnter('?model=qwen3mlx&ctx=32768&pool=64&chat=1&room=1'))
    expect(q.get('model')).toBe('qwen3mlx')
    expect(q.get('ctx')).toBe('32768')
    expect(q.get('pool')).toBe('64')
    expect(q.has('chat')).toBe(false)
    expect(q.has('room')).toBe(false)
  })

  it('leaves a URL that asked for nothing alone', () => {
    expect(urlAfterEnter('?model=llama32')).toBe('?model=llama32')
    expect(urlAfterEnter('')).toBe('')
  })
})
