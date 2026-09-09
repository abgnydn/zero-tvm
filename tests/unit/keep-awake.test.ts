// THE THROTTLING EXEMPTION IS ONE GENERATOR, NOT ONE PER SURFACE.
//
// A serving tab that loses the foreground is throttled hard: measured, a
// backgrounded host generated at ~23 tok/s where the focused tab does ~65, and
// served weights at ~1 MB/s. share.html's two serving roles have carried the
// control that buys the exemption — a screen wake-lock plus a silent audio
// track — since they shipped. The entrance's ⟁ Room tool, which this session
// promoted to the PRIMARY hosting path (the swarm's first machine boots in
// place via onHostHere rather than opening share.html), took a best-effort
// `wakeLock.request('screen')` and nothing else: no audio track, so no
// exemption, so the recommended flow was the throttled one.
//
// The fix is not a second copy of the wiring. A hand-copied fact drifts, and
// two implementations of "how a tab escapes background throttling" is exactly
// that fact copied. So this file pins BOTH halves:
//
//   - STRUCTURE: the wiring exists once, in a module that imports nothing (it
//     has to be safe for share.html's guest role, whose defining property is
//     that it needs no WebGPU and downloads nothing), and both surfaces call
//     that one function rather than rolling their own wake lock.
//   - BEHAVIOUR: the toggle actually does the four things it claims. There is
//     no DOM in this suite, so the button and the two browser APIs are stubs;
//     what is asserted is what wireKeepAwake DOES with them — which is the part
//     that regressed, and the part a source-text check cannot see.
//
// What is NOT covered here, and why: that the entrance's button is inserted in
// the chat head, that it renders in `.cs-chat-tool` chrome, and that
// `.cs-tool-live` lights it. Those are cascade and layout, and this suite has
// no browser. They were checked against a real Chrome instead.

import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { KEEP_AWAKE_NOTE, wireKeepAwake } from '../../src/zero-tvm/keep-awake.ts'

const ROOT = join(import.meta.dirname, '../..')
const read = (rel: string): string => readFileSync(join(ROOT, rel), 'utf8')

/** Source with comments removed. The checks below are about what the code
 *  DOES; a comment that names the wake lock in order to explain why this file
 *  no longer takes one is the opposite of the defect, and must not read as it. */
const code = (rel: string): string => read(rel)
  .replace(/\/\*[\s\S]*?\*\//g, '')
  .replace(/^\s*\/\/.*$/gm, '')

const MODULE = 'src/zero-tvm/keep-awake.ts'
const SHARE = 'src/zero-tvm/share.ts'
const ENTRANCE = 'src/landing-room.ts'

function sources(dir: string, out: string[] = []): string[] {
  for (const e of readdirSync(join(ROOT, dir))) {
    const rel = `${dir}/${e}`
    if (statSync(join(ROOT, rel)).isDirectory()) sources(rel, out)
    else if (e.endsWith('.ts')) out.push(rel)
  }
  return out
}

// ────────────────────────────────────────────────────────────────────────────
// STRUCTURE — one generator, and one that either surface can import
// ────────────────────────────────────────────────────────────────────────────

describe('one generator', () => {
  it('the wiring is defined exactly once in src/', () => {
    const defines = sources('src').filter((f) => /function wireKeepAwake\b/.test(code(f)))
    expect(defines).toEqual([MODULE])
  })

  it('nothing outside the module requests a wake lock of its own', () => {
    const owners = sources('src').filter((f) => /wakeLock\s*\??\.\s*request\(/.test(code(f)))
    expect(owners).toEqual([MODULE])
  })

  it('both serving surfaces import the one wiring and call it', () => {
    expect(code(SHARE)).toMatch(/import\s*\{[^}]*\bwireKeepAwake\b[^}]*\}\s*from\s*'\.\/keep-awake\.js'/)
    expect(code(ENTRANCE)).toMatch(/import\s*\{[^}]*\bwireKeepAwake\b[^}]*\}\s*from\s*'\.\/zero-tvm\/keep-awake\.js'/)
    expect(code(SHARE)).toMatch(/wireKeepAwake\(/)
    expect(code(ENTRANCE)).toMatch(/wireKeepAwake\(/)
  })

  it('the honest sentence is one copy, referenced by both surfaces', () => {
    // The prose cannot be judged by a test. What is mechanical is that there is
    // one of it: the surface that prints it under the button and the surface
    // that carries it on the button both name the constant.
    expect(read(SHARE)).toContain('KEEP_AWAKE_NOTE')
    expect(read(ENTRANCE)).toContain('KEEP_AWAKE_NOTE')
    const literal = 'silent audio track, so the browser throttles'
    const copies = sources('src').filter((f) => read(f).includes(literal))
    expect(copies).toEqual([MODULE])
  })

  it('the sentence still says the thing that makes it honest', () => {
    // The whole defence of running an inaudible oscillator is that the browser
    // tells the operator about it. If that clause ever leaves the copy, the
    // control stops being honest and this test is the tripwire.
    expect(KEEP_AWAKE_NOTE).toContain('audio indicator')
  })

  it('the module imports nothing, so the guest surface stays clean', () => {
    // share.html's guest needs no WebGPU and downloads nothing. Anything on its
    // import path that reaches model-select → weight-loader breaks that. The
    // cheapest possible guarantee is that this module has no import at all.
    expect(code(MODULE)).not.toMatch(/^import\b/m)
    expect(code(MODULE)).not.toMatch(/\bfrom\s*'/)
  })
})

// ────────────────────────────────────────────────────────────────────────────
// BEHAVIOUR — against stubbed DOM + wake-lock + AudioContext
// ────────────────────────────────────────────────────────────────────────────

interface Sentinel { released: boolean; release(): Promise<void> }

function scene(opts: { wakeLock?: boolean } = {}): {
  btn: { classes: Set<string>; attrs: Map<string, string>; click(): void }
  visible(v: boolean): void
  sentinels: Sentinel[]
  audio: { created: number; resumed: number; suspended: number; started: number; gain: number | null }
  wire(): void
  settle(): Promise<void>
} {
  const classes = new Set<string>()
  const attrs = new Map<string, string>()
  let onClick: (() => void) | null = null
  let onVisibility: (() => void) | null = null
  const sentinels: Sentinel[] = []
  const audio = { created: 0, resumed: 0, suspended: 0, started: 0, gain: null as number | null }

  const btnStub = {
    addEventListener: (type: string, fn: () => void): void => { if (type === 'click') onClick = fn },
    setAttribute: (k: string, v: string): void => { attrs.set(k, v) },
    classList: {
      toggle: (c: string, on: boolean): void => { if (on) classes.add(c); else classes.delete(c) },
    },
  }

  const doc = {
    visibilityState: 'visible',
    addEventListener: (type: string, fn: () => void): void => {
      if (type === 'visibilitychange') onVisibility = fn
    },
  }

  const nav = {
    wakeLock: opts.wakeLock === false ? undefined : {
      request: async (): Promise<Sentinel> => {
        const s: Sentinel = { released: false, release: async (): Promise<void> => { s.released = true } }
        sentinels.push(s)
        return s
      },
    },
  }

  class FakeAudioContext {
    constructor() { audio.created++ }
    createOscillator(): { connect(n: unknown): unknown; start(): void } {
      return { connect: (n: unknown): unknown => n, start: (): void => { audio.started++ } }
    }
    createGain(): { gain: { value: number }; connect(n: unknown): unknown } {
      const g = { gain: { value: 1 }, connect: (n: unknown): unknown => n }
      // The oscillator is connected THROUGH the gain, so the value the module
      // sets is read back off this object after wiring.
      Object.defineProperty(g.gain, 'value', {
        get: () => audio.gain ?? 1,
        set: (v: number) => { audio.gain = v },
      })
      return g
    }
    get destination(): unknown { return {} }
    async resume(): Promise<void> { audio.resumed++ }
    async suspend(): Promise<void> { audio.suspended++ }
  }

  const define = (k: string, v: unknown): void => {
    Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true })
  }

  return {
    btn: { classes, attrs, click: (): void => onClick?.() },
    visible: (v: boolean): void => { doc.visibilityState = v ? 'visible' : 'hidden'; onVisibility?.() },
    sentinels,
    audio,
    wire: (): void => {
      define('document', doc)
      define('navigator', nav)
      define('AudioContext', FakeAudioContext)
      wireKeepAwake(btnStub as unknown as HTMLElement)
    },
    // apply() is async and the click handler does not await it.
    settle: async (): Promise<void> => { await new Promise((r) => setTimeout(r, 0)) },
  }
}

const SAVED = {
  document: Object.getOwnPropertyDescriptor(globalThis, 'document'),
  navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator'),
  AudioContext: Object.getOwnPropertyDescriptor(globalThis, 'AudioContext'),
}

afterEach(() => {
  for (const [k, d] of Object.entries(SAVED)) {
    if (d) Object.defineProperty(globalThis, k, d)
    else delete (globalThis as Record<string, unknown>)[k]
  }
})

describe('the toggle', () => {
  it('starts off, and starts NOTHING until it is clicked', async () => {
    const s = scene()
    s.wire()
    await s.settle()
    // A room opening is not a user gesture, and wiring a control is not one
    // either. Nothing may play until the operator says so.
    expect(s.audio.created).toBe(0)
    expect(s.sentinels).toHaveLength(0)
    expect(s.btn.attrs.get('aria-pressed')).toBe('false')
    expect(s.btn.classes.has('cs-tool-live')).toBe(false)
  })

  it('takes both holds on, and lights the control', async () => {
    const s = scene()
    s.wire()
    s.btn.click()
    await s.settle()
    expect(s.sentinels).toHaveLength(1)
    expect(s.sentinels[0].released).toBe(false)
    expect(s.audio.created).toBe(1)
    expect(s.audio.started).toBe(1)
    expect(s.audio.resumed).toBe(1)
    // Inaudible, but "playing" as far as the scheduler cares.
    expect(s.audio.gain).toBeGreaterThan(0)
    expect(s.audio.gain).toBeLessThan(0.01)
    expect(s.btn.attrs.get('aria-pressed')).toBe('true')
    expect(s.btn.classes.has('cs-tool-live')).toBe(true)
  })

  it('releases both off, and unlights', async () => {
    const s = scene()
    s.wire()
    s.btn.click()
    await s.settle()
    s.btn.click()
    await s.settle()
    expect(s.sentinels[0].released).toBe(true)
    expect(s.audio.suspended).toBe(1)
    expect(s.btn.attrs.get('aria-pressed')).toBe('false')
    expect(s.btn.classes.has('cs-tool-live')).toBe(false)
  })

  it('reuses the one audio graph across on/off/on', async () => {
    const s = scene()
    s.wire()
    s.btn.click(); await s.settle()
    s.btn.click(); await s.settle()
    s.btn.click(); await s.settle()
    expect(s.audio.created).toBe(1)
    expect(s.audio.started).toBe(1)
    expect(s.audio.resumed).toBe(2)
  })

  it('re-acquires the wake lock when the tab comes back', async () => {
    // The UA drops a screen wake lock the moment the tab is hidden, and never
    // gives it back on its own — which is the ONLY state that matters here,
    // since a serving tab is a background tab for most of its life.
    const s = scene()
    s.wire()
    s.btn.click()
    await s.settle()
    s.visible(false)
    await s.settle()
    expect(s.sentinels).toHaveLength(1)
    s.visible(true)
    await s.settle()
    expect(s.sentinels).toHaveLength(2)
  })

  it('does not re-acquire when the operator left it off', async () => {
    const s = scene()
    s.wire()
    s.visible(false)
    s.visible(true)
    await s.settle()
    expect(s.sentinels).toHaveLength(0)
    expect(s.audio.created).toBe(0)
  })

  it('still takes the audio track where there is no wake-lock API', async () => {
    // Wake Lock is unsupported in Firefox and was late to Safari. The throttle
    // exemption is the half that carries the measured 23 → 65 tok/s, so it must
    // not be lost to the other half being unavailable.
    const s = scene({ wakeLock: false })
    s.wire()
    s.btn.click()
    await s.settle()
    expect(s.audio.created).toBe(1)
    expect(s.audio.resumed).toBe(1)
    expect(s.btn.classes.has('cs-tool-live')).toBe(true)
  })

  it('wires nothing when the button is absent', () => {
    const s = scene()
    Object.defineProperty(globalThis, 'document', {
      value: { addEventListener: (): void => { throw new Error('listened without a button') } },
      configurable: true,
      writable: true,
    })
    expect(() => wireKeepAwake(null)).not.toThrow()
    void s
  })
})
