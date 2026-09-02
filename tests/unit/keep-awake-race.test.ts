// A CONTROL THAT LIES ABOUT THE MACHINE IS WORSE THAN NO CONTROL.
//
// keep-awake.ts's toggle buys two things a serving tab needs — a screen wake
// lock, and a silent audio track that exempts the tab from background
// throttling — and its whole defence for running an inaudible oscillator is
// that the browser tells the operator about it: the tab shows its audio
// indicator, and the button shows `aria-pressed`.
//
// Those two must agree. `apply()` is async and awaits `wakeLock.request()`
// BEFORE it builds the audio graph, while the click handler flips the visible
// state synchronously. Two clicks in one tick therefore interleave: the OFF
// branch runs first and finds both holds still null — so both releases are
// no-ops — and then the ON branch's await resolves and installs a live
// sentinel and a running context UNDERNEATH a control that reads off. The
// screen stays awake, the tab stays exempt, the audio indicator stays lit, and
// the only truthful thing on the screen is the indicator the module's own
// header offers as proof of honesty.
//
// Not remotely triggerable: it needs the operator's own double-click on their
// own hosting tab. Still a control that lies.
//
// So the invariant this file pins is one sentence, checked after every
// sequence of clicks rather than after a lucky one:
//
//   WHAT THE BUTTON SAYS IS WHAT THE MACHINE IS DOING.
//   aria-pressed true  ⇒ exactly one wake-lock sentinel held, context running.
//   aria-pressed false ⇒ no sentinel held, no context running.
//
// Both halves matter. "No sentinel held" alone would pass a version that leaks
// a second sentinel while showing on; "exactly one" catches that too.
//
// The stubs model the two browser APIs closely enough for the races to be real:
// `wakeLock.request()` can be held pending and can reject with NotAllowedError
// (which is what a hidden tab actually answers), the UA drops granted sentinels
// when the tab hides (which is why the module re-acquires at all), and the
// AudioContext applies resume/suspend in call order through a queue, so a
// suspend issued while a resume is still in flight settles the way the spec
// says it does. No browser is needed to hold any of this.

import { afterEach, describe, expect, it } from 'vitest'
import { wireKeepAwake } from '../../src/zero-tvm/keep-awake.ts'

interface Sentinel { released: boolean; release(): Promise<void> }

interface Report {
  ariaPressed: string | undefined
  lit: boolean
  audioRunning: boolean
  wakeRequests: number
  wakeReleases: number
  sentinelsHeld: number
}

interface Scene {
  click(): void
  visible(v: boolean): void
  flushWake(): void
  openAudio(): void
  audioCreated(): number
  report(): Report
  settle(): Promise<void>
}

/** ON is one held sentinel and a running context; OFF is neither. The counts
 *  the caller passes are the only part that varies with the click sequence. */
function on(wakeRequests: number, wakeReleases: number): Report {
  return { ariaPressed: 'true', lit: true, audioRunning: true, wakeRequests, wakeReleases, sentinelsHeld: 1 }
}
function off(wakeRequests: number, wakeReleases: number, audioRunning = false): Report {
  return { ariaPressed: 'false', lit: false, audioRunning, wakeRequests, wakeReleases, sentinelsHeld: 0 }
}

function scene(opts: { gateWake?: boolean; rejectWake?: boolean; gateAudio?: boolean } = {}): Scene {
  const classes = new Set<string>()
  const attrs = new Map<string, string>()
  let onClick: (() => void) | null = null
  let onVisibility: (() => void) | null = null

  const granted: Sentinel[] = []
  const pendingWake: Array<() => void> = []
  const wake = { requests: 0, releases: 0 }

  // The audio gate models "the context is still starting": every state
  // transition queues behind it, in call order, so a suspend issued while a
  // resume is in flight lands after that resume rather than before it.
  let letAudioThrough = (): void => {}
  const audioGate: Promise<void> = opts.gateAudio
    ? new Promise<void>((r) => { letAudioThrough = r })
    : Promise.resolve()

  let created = 0
  let ctx: { state: string } | null = null

  const btnStub = {
    addEventListener: (type: string, fn: () => void): void => { if (type === 'click') onClick = fn },
    setAttribute: (k: string, v: string): void => { attrs.set(k, v) },
    classList: {
      toggle: (c: string, lit: boolean): void => { if (lit) classes.add(c); else classes.delete(c) },
    },
  }

  const doc = {
    visibilityState: 'visible',
    addEventListener: (type: string, fn: () => void): void => {
      if (type === 'visibilitychange') onVisibility = fn
    },
  }

  const nav = {
    wakeLock: {
      request: (): Promise<Sentinel> => {
        wake.requests++
        return new Promise<Sentinel>((resolve, reject) => {
          const settleIt = (): void => {
            if (opts.rejectWake) {
              // What a hidden tab really answers. The module treats it as
              // "no wake lock available" and keeps the audio half.
              const e = new Error('The requesting page is not visible')
              e.name = 'NotAllowedError'
              reject(e)
              return
            }
            const s: Sentinel = {
              released: false,
              release: async (): Promise<void> => { s.released = true; wake.releases++ },
            }
            granted.push(s)
            resolve(s)
          }
          if (opts.gateWake) pendingWake.push(settleIt)
          else settleIt()
        })
      },
    },
  }

  class FakeAudioContext {
    state = 'suspended'
    private q: Promise<void> = Promise.resolve()
    constructor() { created++; ctx = this }
    createOscillator(): { connect(n: unknown): unknown; start(): void } {
      return { connect: (n: unknown): unknown => n, start: (): void => {} }
    }
    createGain(): { gain: { value: number }; connect(n: unknown): unknown } {
      return { gain: { value: 1 }, connect: (n: unknown): unknown => n }
    }
    get destination(): unknown { return {} }
    resume(): Promise<void> { return this.step('running') }
    suspend(): Promise<void> { return this.step('suspended') }
    private step(s: string): Promise<void> {
      this.q = this.q.then(() => audioGate).then(() => { this.state = s })
      return this.q
    }
  }

  const define = (k: string, v: unknown): void => {
    Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true })
  }
  define('document', doc)
  define('navigator', nav)
  define('AudioContext', FakeAudioContext)
  wireKeepAwake(btnStub as unknown as HTMLElement)

  return {
    click: (): void => onClick?.(),
    visible: (v: boolean): void => {
      // The UA drops every screen wake lock when the tab hides, without asking
      // and without telling the page. That is the ONLY reason the module
      // listens for visibilitychange at all, so the stub has to do it too.
      if (!v) for (const s of granted) s.released = true
      doc.visibilityState = v ? 'visible' : 'hidden'
      onVisibility?.()
    },
    flushWake: (): void => { for (const f of pendingWake.splice(0)) f() },
    openAudio: (): void => { letAudioThrough() },
    audioCreated: (): number => created,
    report: (): Report => ({
      ariaPressed: attrs.get('aria-pressed'),
      lit: classes.has('cs-tool-live'),
      audioRunning: ctx?.state === 'running',
      wakeRequests: wake.requests,
      wakeReleases: wake.releases,
      sentinelsHeld: granted.filter((s) => !s.released).length,
    }),
    // The click handler does not await apply(); one macrotask turn drains
    // every microtask it left behind, however deeply chained.
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

describe('clicks that land in the same tick', () => {
  it('off → on → off leaves nothing running behind an off control', async () => {
    // THE DEFECT. Both clicks land before the first apply() gets past its
    // await, so the OFF branch releases two nulls and the ON branch installs
    // its holds afterwards, under a button that already reads off.
    const s = scene()
    s.click()
    s.click()
    await s.settle()
    expect(s.report()).toEqual(off(1, 1))
    // And nothing was ever built to leave running: the request the operator
    // cancelled before it resolved must not become an audio graph.
    expect(s.audioCreated()).toBe(0)
  })

  it('on → off → on ends on, holding exactly one sentinel', async () => {
    const s = scene()
    s.click()
    await s.settle()
    s.click()
    s.click()
    await s.settle()
    expect(s.report()).toEqual(on(2, 1))
  })

  it('three clicks from off end on, with the abandoned sentinel released', async () => {
    // on, off, on. Two requests go out; the first one's answer arrives to a
    // state that has moved twice and must be dropped, not stacked.
    const s = scene()
    s.click()
    s.click()
    s.click()
    await s.settle()
    expect(s.report()).toEqual(on(2, 1))
  })

  it('four clicks from off end off, holding nothing', async () => {
    const s = scene()
    s.click()
    s.click()
    s.click()
    s.click()
    await s.settle()
    expect(s.report()).toEqual(off(2, 2))
    expect(s.audioCreated()).toBe(0)
  })

  it('six clicks from off end off, holding nothing', async () => {
    const s = scene()
    for (let i = 0; i < 6; i++) s.click()
    await s.settle()
    // Even count from off is off; the odd counts above cover the on side.
    expect(s.report()).toEqual(off(3, 3))
  })
})

describe('clicks while the wake lock is still pending', () => {
  it('a grant that arrives after the operator turned it off is released, not installed', async () => {
    const s = scene({ gateWake: true })
    s.click()
    await s.settle()
    // In flight: the button reads on, and the UA has not answered yet. That
    // half-state is honest; what follows is what must not become dishonest.
    expect(s.report().wakeRequests).toBe(1)
    expect(s.report().sentinelsHeld).toBe(0)
    s.click()                                // off, while the request is in flight
    await s.settle()
    s.flushWake()                            // the UA answers now
    await s.settle()
    expect(s.report()).toEqual(off(1, 1))
    expect(s.audioCreated()).toBe(0)
  })

  it('a REJECTED request that lands after the operator turned it off starts no audio', async () => {
    // A hidden tab answers NotAllowedError. The module deliberately keeps the
    // audio half when the wake half is unavailable — but only for the state
    // that asked for it.
    const s = scene({ gateWake: true, rejectWake: true })
    s.click()
    await s.settle()
    s.click()
    await s.settle()
    s.flushWake()
    await s.settle()
    expect(s.report()).toEqual(off(1, 0))
    expect(s.audioCreated()).toBe(0)
  })

  it('a REJECTED request for the state still wanted keeps the audio half', async () => {
    // The measured 23 → 65 tok/s is carried by the audio exemption, not the
    // wake lock, so losing the wake lock must not cost the exemption.
    const s = scene({ gateWake: true, rejectWake: true })
    s.click()
    await s.settle()
    s.flushWake()
    await s.settle()
    expect(s.report()).toEqual({
      ariaPressed: 'true', lit: true, audioRunning: true,
      wakeRequests: 1, wakeReleases: 0, sentinelsHeld: 0,
    })
  })
})

describe('clicks while the audio context is still starting', () => {
  it('turning off before the context finishes resuming leaves it suspended', async () => {
    const s = scene({ gateAudio: true })
    s.click()
    await s.settle()
    expect(s.audioCreated()).toBe(1)        // built, resume in flight, not running yet
    s.click()                                // off, while it is still starting
    await s.settle()
    s.openAudio()                            // both transitions apply now, in order
    await s.settle()
    expect(s.report()).toEqual(off(1, 1))
  })

  it('a double-click while the context is starting ends off and quiet', async () => {
    const s = scene({ gateAudio: true })
    s.click()
    await s.settle()
    s.click()
    s.click()
    s.click()
    await s.settle()
    s.openAudio()
    await s.settle()
    expect(s.report()).toEqual(off(2, 2))
  })
})

describe('the visibility re-acquire', () => {
  it('re-acquires on return, still holding exactly one sentinel', async () => {
    const s = scene()
    s.click()
    await s.settle()
    s.visible(false)                         // the UA drops the lock here
    await s.settle()
    s.visible(true)
    await s.settle()
    expect(s.report()).toEqual(on(2, 0))     // the UA released the first, not us
  })

  it('cannot resurrect anything after the operator turned it off', async () => {
    // The guard that makes this safe is `if (on && …)`. Without it a hidden
    // tab coming back would restart audio the operator switched off.
    const s = scene()
    s.click()
    await s.settle()
    s.click()
    await s.settle()
    s.visible(false)
    s.visible(true)
    await s.settle()
    expect(s.report()).toEqual(off(1, 1))
  })

  it('a re-acquire in flight loses to a click that turns the toggle off', async () => {
    // The real-world shape of the race: a serving tab spends its life in the
    // background, so the re-acquire is the request most likely to be in flight
    // when the operator finally touches the button.
    const s = scene({ gateWake: true })
    s.click()
    s.flushWake()
    await s.settle()
    s.visible(false)
    s.visible(true)
    await s.settle()                         // re-acquire requested, still pending
    s.click()                                // off
    await s.settle()
    s.flushWake()
    await s.settle()
    // We release twice: the sentinel the UA had already dropped on hide, and
    // the one the abandoned re-acquire brings back.
    expect(s.report()).toEqual(off(2, 2))
  })
})

describe('what already worked', () => {
  it('a single on → off releases the sentinel and suspends the context', async () => {
    const s = scene()
    s.click()
    await s.settle()
    expect(s.report()).toEqual(on(1, 0))
    s.click()
    await s.settle()
    expect(s.report()).toEqual(off(1, 1))
    expect(s.audioCreated()).toBe(1)
  })

  it('reuses the one audio graph across on/off/on', async () => {
    const s = scene()
    s.click(); await s.settle()
    s.click(); await s.settle()
    s.click(); await s.settle()
    expect(s.audioCreated()).toBe(1)
    expect(s.report()).toEqual(on(2, 1))
  })
})
