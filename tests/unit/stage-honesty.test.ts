// A TAB THAT HOLDS ONE STAGE MUST NOT DESCRIBE ITSELF AS THE WHOLE MODEL.
//
// The entrance can now host ONE STAGE of a split in place — a tab holding, say,
// layers 0-8 of 62. Several surfaces were still written for the only case that
// used to exist, where a serving tab held everything.
//
// The size figure is a REGRESSION of a bug already fixed once, in share.ts's
// download gate: "the iPhone that held one layer of the 27B was asked to approve
// '~14.1 GB' for a slice worth a fraction of that (real device, 2026-08-29)".
// The resolution share.ts chose is the one pinned here: state the RANGE, never
// invent a per-stage byte figure (it is not knowable before the safetensors
// headers are read), and SUPPRESS the whole-model RAM note rather than
// substituting a guess.
//
// What is and is not checkable here:
//   - `panelMarkup` is a pure string function, so the boot card's two lines are
//     asserted directly. That is the item that carried real numbers.
//   - The room's consent PARAGRAPH is prose. A test cannot judge whether a
//     sentence is true, so it pins what is mechanical: the paragraph BRANCHES on
//     holding a stage, names the range, and drops the two promises that are
//     false under a split — and the entrance's copy and share.html's copy stay
//     byte-identical, since they are two generators of one fact.
//   - The composer's `hidden` is a CSS specificity bug, so the gate is over the
//     stylesheet: anything this repo hides with the attribute needs a [hidden]
//     rule, because a class `display` beats the UA sheet.
//   - The guest surface's copy (share.ts `reveal`/`onHostMsg`) and the
//     peer-weights offer are NOT covered. The guest cannot know whether the room
//     is split — room-host.ts's `info` frame carries no stage or chain field —
//     so the only correct copy is copy that is true either way, and "is this
//     sentence true in both cases" is a reading, not an assertion.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { QWEN3_8_27B_4BIT } from '../../src/compiler/model-spec.ts'
import { panelMarkup } from '../../src/landing-chat.ts'
import { roomConsentCopy } from '../../src/landing-room.ts'
import { modelBranding } from '../../src/zero-tvm/model-registry.ts'

const ROOT = join(import.meta.dirname, '../..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

const SPEC = QWEN3_8_27B_4BIT
const BRAND = modelBranding(SPEC)
const STAGE = { start: 0, end: 8 }

describe('a stage does not quote the whole checkpoint', () => {
  // The premise. If the registry ever stops giving this model a size label and
  // a RAM note, the assertions below would pass while checking nothing.
  it('the model under test really does carry both whole-model figures', () => {
    expect(BRAND.sizeLabel).toBeTruthy()
    expect(BRAND.ramNote).toBeTruthy()
  })

  it('the whole-model panel still shows the download size and the RAM note', () => {
    const html = panelMarkup(SPEC, BRAND, BRAND.params)
    expect(html).toContain(BRAND.sizeLabel)
    expect(html).toContain(BRAND.ramNote as string)
  })

  it('a stage panel states the layer range instead of the checkpoint size', () => {
    const html = panelMarkup(SPEC, BRAND, BRAND.params, STAGE)
    expect(html).not.toContain(BRAND.sizeLabel)
    expect(html).toContain(`${STAGE.start}–${STAGE.end} of ${SPEC.layers}`)
  })

  it('a stage panel drops the whole-model RAM note and puts nothing in its place', () => {
    const html = panelMarkup(SPEC, BRAND, BRAND.params, STAGE)
    expect(html).not.toContain(BRAND.ramNote as string)
    // Not "no number that happens to differ" — NO invented per-stage figure at
    // all. Nothing on this card may claim a GB count for a slice whose real size
    // is only known once the safetensors headers are read.
    expect(html).not.toMatch(/[\d.]+\s*GB/)
  })

  it('a chosen memory build is still named on a stage — that one IS true here', () => {
    const html = panelMarkup(SPEC, BRAND, '8 experts · 32k ctx', STAGE)
    expect(html).toContain('8 experts · 32k ctx')
  })
})

describe('the room consent paragraph', () => {
  it('says something different when this tab holds only a stage', () => {
    const whole = roomConsentCopy(BRAND.name, null, SPEC.layers)
    const stage = roomConsentCopy(BRAND.name, STAGE, SPEC.layers)
    expect(stage).not.toBe(whole)
    expect(stage).toContain(`layers ${STAGE.start}–${STAGE.end} of ${SPEC.layers}`)
  })

  it("does not offer a stage host's cache as a model a guest can run", () => {
    // The whole-model copy makes that offer and it is true there. From a stage
    // host a guest gets one layer range: loadWeights writes only that range into
    // the model's OPFS dir and peer-weights.ts streams that dir as it stands.
    expect(roomConsentCopy(BRAND.name, null, SPEC.layers)).toContain('run it locally')
    expect(roomConsentCopy(BRAND.name, STAGE, SPEC.layers)).not.toContain('run it locally')
  })

  it('never claims a split model runs on THIS machine', () => {
    // room-chain.ts's stepAll runs this tab's layers and then awaits every
    // downstream stage on OTHER machines, so the whole-model sentence is false.
    expect(roomConsentCopy(BRAND.name, null, SPEC.layers)).toContain('THIS machine')
    expect(roomConsentCopy(BRAND.name, STAGE, SPEC.layers)).not.toContain('THIS machine')
  })
})

describe('the entrance and share.html host the same room', () => {
  // Two files carry this paragraph. They cannot share one import: share.html's
  // guest needs no WebGPU and downloads nothing, while landing-room.ts reaches
  // model-select → weight-loader. So the fact has two generators, and this is
  // the gate that keeps them one fact. The block is delimited in both files.
  const block = (rel: string): string => {
    const src = read(rel)
    const m = /\/\* CONSENT:BEGIN[\s\S]*?\/\* CONSENT:END \*\//.exec(src)
    expect(m, `${rel} has no CONSENT:BEGIN…CONSENT:END block`).not.toBeNull()
    return (m as RegExpExecArray)[0]
  }

  it('word for word, in both files', () => {
    expect(block('src/zero-tvm/share.ts')).toBe(block('src/landing-room.ts'))
  })

  it('and the block really is the code that renders — not a stale copy', () => {
    // Two identical blocks that no longer hold the live function would compare
    // equal to each other and prove nothing. The phrases are taken FROM the
    // output, so they cannot go stale the way a typed-in expectation would.
    const b = block('src/landing-room.ts')
    expect(b).toContain('export function roomConsentCopy')
    for (const stage of [null, STAGE]) {
      const out = roomConsentCopy('NAME', stage, SPEC.layers)
      expect(b).toContain(out.slice(out.lastIndexOf('Keep this tab')))
    }
  })
})

describe('an element hidden with the attribute is actually hidden', () => {
  // `[hidden] { display: none }` lives in the UA sheet at the lowest priority,
  // so ANY class rule that sets a display beats it. The stage path hid #composer
  // and got a greyed but fully laid-out composer on a tab whose own text says it
  // answers no chat (782x53, checkVisibility() true — measured 2026-09-01).
  //
  // Scoped to the surfaces that render chat-ui.css's markup, the way
  // no-orphan-exports.test.ts is scoped: a sweep wide enough to drown gets muted.
  const SURFACES = [
    'src/landing-chat.ts',
    'src/zero-tvm/share.ts',
    'src/zero-tvm/chat-ui.ts',
    'src/zero-tvm/chat-flow.ts',
    'zero-tvm.html',
    'share.html',
  ]
  const css = read('public/chat-ui.css').replace(/\/\*[\s\S]*?\*\//g, '')

  /** Selector → display, for every rule in the sheet that sets one. */
  function rules(): Array<{ sel: string; display: string }> {
    const flat = css.replace(/@media[^{]*\{/g, '')
    const out: Array<{ sel: string; display: string }> = []
    for (const [, sel, body] of flat.matchAll(/([^{}]+)\{([^{}]*)\}/g)) {
      const d = /(?:^|;)\s*display\s*:\s*([^;]+)/.exec(body)
      if (d) out.push({ sel: sel.trim(), display: d[1].trim() })
    }
    return out
  }

  /** Classes on any element these surfaces give the `hidden` ATTRIBUTE to —
   *  written into the markup, or set on an id whose markup names its classes. */
  function hiddenClasses(): Set<string> {
    const out = new Set<string>()
    const idClasses = new Map<string, string[]>()
    const hiddenIds = new Set<string>()
    for (const rel of SURFACES) {
      const src = read(rel)
      for (const [tag] of src.matchAll(/<[a-zA-Z][^>]*>/g)) {
        const cls = /class="([^"]*)"/.exec(tag)?.[1] ?? ''
        const names = cls.split(/\s+/).filter((c) => c && !c.includes('$'))
        const id = /\bid="([^"${}]*)"/.exec(tag)?.[1]
        if (id) idClasses.set(id, names)
        // Strip every attribute VALUE first, so class="hidden" is not read as
        // the attribute (chat-ui.ts uses a `hidden` CLASS for the welcome box).
        const bare = tag.replace(/=\s*"[^"]*"/g, '=""').replace(/=\s*'[^']*'/g, "=''")
        if (/\bhidden\b/.test(bare.slice(bare.indexOf(' ')))) for (const c of names) out.add(c)
      }
      // Runtime hides, per line: querySelector('#composer')…setAttribute('hidden').
      for (const line of src.split('\n')) {
        if (!/setAttribute\(\s*'hidden'|\.hidden\s*=\s*true/.test(line)) continue
        for (const [, id] of line.matchAll(/['"`]#([\w-]+)['"`]/g)) hiddenIds.add(id)
      }
    }
    for (const id of hiddenIds) for (const c of idClasses.get(id) ?? []) out.add(c)
    return out
  }

  const all = rules()
  const hideable = hiddenClasses()

  it('found the composer — otherwise this test is scanning nothing', () => {
    expect(hideable.has('composer')).toBe(true)
    expect(hideable.has('composer-btn')).toBe(true)
  })

  for (const cls of [...hiddenClasses()].sort()) {
    const setters = all.filter((r) => r.sel === `.${cls}` && r.display !== 'none')
    if (!setters.length) continue
    it(`.${cls} carries a [hidden] guard against its own display: ${setters[0].display}`, () => {
      const guard = all.find((r) => r.sel === `.${cls}[hidden]` && r.display === 'none')
      expect(guard, `.${cls} sets display but nothing restores display:none under [hidden]`)
        .toBeTruthy()
    })
  }
})
