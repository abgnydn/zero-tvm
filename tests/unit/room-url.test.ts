// The swarm link builder against the router that reads its links.
//
// The entrance hands out URLs for three roles. share.html decides which role a
// URL is by parsing it — and the failure mode is not an error message, it is
// the WRONG ROLE: a query written after the fragment leaves location.search
// empty and location.hash unparseable, so the page opens a brand-new room and
// the machines never meet. A real user lost an hour to exactly that on
// 2026-08-29.
//
// So these do not check the URLs against a second copy of the grammar. They
// feed each generated URL back through roleFor() — the function share.ts's own
// three-way branch calls — and assert the role it was labelled with. A builder
// whose output routes somewhere else fails here.

import { describe, expect, it } from 'vitest'
import {
  ctxFrom, roleFor, roomIdFrom, roomLink, splitBounds, stageRangeFrom, swarmUrls,
} from '../../src/zero-tvm/room-url.ts'
import { SHIPPED_MODELS, canSplitAcrossDevices, modelBranding } from '../../src/zero-tvm/model-registry.ts'

const ORIGIN = 'https://zerotvm.com'
// 22 base64url chars — what room-host.ts makes of 16 random bytes.
const ROOM = 'Zm9vYmFyYmF6cXV4MTIzNA'

/** What share.ts sees when a browser opens the URL. */
const routeOf = (url: string): string => {
  const u = new URL(url)
  return roleFor(u.search, u.hash)
}

describe('the room id', () => {
  it('accepts what room-host.ts generates, with or without the #', () => {
    expect(roomIdFrom(`#${ROOM}`)).toBe(ROOM)
    expect(roomIdFrom(ROOM)).toBe(ROOM)
  })

  it('refuses anything that is not a bare id — a half-copied link is not a room', () => {
    expect(roomIdFrom('#short')).toBeNull()
    expect(roomIdFrom(`#${ROOM}?model=qwen3mlx`)).toBeNull()
    expect(roomIdFrom('#https://zerotvm.com/share.html')).toBeNull()
  })
})

describe('swarmUrls routes every link to the role it is labelled with', () => {
  for (const machines of [2, 3, 4]) {
    it(`${machines}-way split, room known`, () => {
      const stops = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines, room: ROOM })
      expect(stops).toHaveLength(machines + 1)   // one per machine, plus the guest
      for (const s of stops) {
        expect(s.url, `${s.role} link is missing`).not.toBeNull()
        expect(routeOf(s.url!), `${s.url} routes to the wrong role`).toBe(s.role)
      }
    })
  }

  it('the host link starts at layer 0 — share.ts throws on a hosting stage that does not', () => {
    const [host] = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 3, room: ROOM })
    expect(host.range).toEqual({ start: 0, end: 12 })
    expect(stageRangeFrom(new URL(host.url!).search)).toEqual({ start: 0, end: 12 })
  })

  it('every helper starts above 0 — that is the only thing making it a helper', () => {
    const stops = swarmUrls({ origin: ORIGIN, param: 'qwen38', layers: 64, machines: 4, room: ROOM })
    for (const s of stops.filter((x) => x.role === 'helper')) {
      expect(s.range!.start).toBeGreaterThan(0)
    }
  })

  it('withholds the helper and guest links until the room exists', () => {
    // A placeholder fragment would fail roomIdFrom and route to HOST — the
    // silent second-room bug, with a Copy button on it. Nothing is better.
    const stops = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 3, room: null })
    expect(stops[0].url).not.toBeNull()          // the host can be opened first
    expect(routeOf(stops[0].url!)).toBe('host')
    for (const s of stops.slice(1)) expect(s.url).toBeNull()
  })

  it('the guest link carries no ?model= — a guest that hosts is not a guest', () => {
    const guest = swarmUrls({ origin: ORIGIN, param: 'qwen36q3', layers: 40, machines: 2, room: ROOM }).at(-1)!
    expect(guest.role).toBe('guest')
    expect(new URL(guest.url!).search).toBe('')
    expect(guest.range).toBeUndefined()
  })
})

describe('the mistake the builder exists to prevent', () => {
  it('a query written AFTER the fragment routes to host, not helper', () => {
    // Not a hypothetical: this is what a URL bar produces when you paste the
    // room link and then type the query on the end. share.ts sees no search
    // and no room id, takes the host branch, and opens a SECOND room — with
    // no error anywhere, on either machine.
    const wrong = `${ORIGIN}/share.html#${ROOM}?model=qwen3mlx&layers=18-36`
    expect(routeOf(wrong)).toBe('host')

    const right = `${ORIGIN}/share.html?model=qwen3mlx&layers=18-36#${ROOM}`
    expect(routeOf(right)).toBe('helper')
  })

  it('a stage that STARTS the model is a host even inside an existing room', () => {
    // `start > 0` is the whole difference between the two serving roles, and
    // it is not decoration: runHost throws on a hosting stage that does not
    // start at 0, and a helper has no embedding to prefill with. Written the
    // other way round — anything with ?layers= inside a room is a helper — the
    // second machine of a room silently loses its ability to hold the start.
    expect(routeOf(`${ORIGIN}/share.html?model=qwen3mlx&layers=0-18#${ROOM}`)).toBe('host')
    expect(routeOf(`${ORIGIN}/share.html?model=qwen3mlx&layers=18-36#${ROOM}`)).toBe('helper')
  })

  it('dropping ?layers= from a helper link makes it a second full host', () => {
    // Still legal — it is how a guest that copied the weights joins the room
    // as another whole-model host — but it is not a stage, and a chain waiting
    // for those layers waits forever.
    expect(routeOf(`${ORIGIN}/share.html?model=qwen3mlx#${ROOM}`)).toBe('host')
  })
})

describe('splitBounds', () => {
  it('tiles [0, layers] with no gap and no overlap', () => {
    for (const layers of [16, 27, 32, 36, 40, 64]) {
      for (const machines of [2, 3, 4]) {
        const b = splitBounds(layers, machines)
        expect(b).toHaveLength(machines + 1)
        expect(b[0]).toBe(0)
        expect(b.at(-1)).toBe(layers)
        for (let i = 1; i < b.length; i++) expect(b[i]).toBeGreaterThan(b[i - 1])
      }
    }
  })

  it('never asks for more machines than there are layers', () => {
    expect(splitBounds(3, 8)).toEqual([0, 1, 2, 3])
  })
})

describe('the builder only offers models that can actually be split', () => {
  it('MLX checkpoints only — loadWeights refuses a layerRange on an MLC one', () => {
    for (const { param, spec } of SHIPPED_MODELS) {
      expect(canSplitAcrossDevices(spec), `${param || 'default'}`)
        .toBe(spec.weightFormat === 'mlx-safetensors' && !spec.embeddingOnly)
    }
    // The three that must stay out, named so a format change here is loud.
    for (const p of ['', 'qwen3', 'qwen35', 'embed']) {
      const hit = SHIPPED_MODELS.find((m) => m.param === p)!
      expect(canSplitAcrossDevices(hit.spec), `${p || 'default'} must not be offered`).toBe(false)
    }
  })

  it('at least one splittable, non-pending model is left to offer', () => {
    const offered = SHIPPED_MODELS.filter(
      ({ spec }) => canSplitAcrossDevices(spec) && !modelBranding(spec).pending)
    expect(offered.length).toBeGreaterThan(0)
    // Every offered model must survive a 4-way split, or a chip in the picker
    // produces ranges the loader cannot build.
    for (const { param, spec } of offered) {
      expect(spec.layers, `${param} has too few layers to split`).toBeGreaterThanOrEqual(4)
    }
  })
})

// ── the link carries the room's CONFIGURATION ────────────────────────────────
// A link used to say only which room. A guest could not tell what it was
// joining, and a helper link had to be hand-edited out of a guest link — which
// is how a stage ended up on a different ?ctx= than the host, and so a
// differently sized KV cache for the same conversation.

describe('an OLD link keeps working', () => {
  it('/share.html#<room> with no query at all is still a guest link', () => {
    // Every link handed out before links carried config looks like this. If it
    // ever routes anywhere else, every one of them silently opens a new room.
    expect(routeOf(`${ORIGIN}/share.html#${ROOM}`)).toBe('guest')
    expect(roomIdFrom(`#${ROOM}`)).toBe(ROOM)
    expect(stageRangeFrom('')).toBeNull()
    expect(ctxFrom('')).toBeNull()
  })
})

describe('ctxFrom', () => {
  it('reads a positive ?ctx=', () => {
    expect(ctxFrom('?ctx=8192')).toBe(8192)
    expect(ctxFrom('?model=qwen3mlx&ctx=262144&layers=0-18')).toBe(262144)
  })

  it('refuses everything that is not a real budget', () => {
    // specWithCtx treats these as "no override"; ctxFrom must agree, or the
    // link's ctx and the spec's ctx disagree about whether one was given.
    for (const q of ['', '?ctx=', '?ctx=0', '?ctx=-1', '?ctx=junk', '?ctx=Infinity', '?ctx=NaN']) {
      expect(ctxFrom(q), q).toBeNull()
    }
  })
})

describe('an absent ?ctx= stays absent', () => {
  it('is never turned into a number by this module', () => {
    // There used to be a `ctxFor(search, ownDefault)` here that answered an
    // absent key with the caller's own default, and a number always reads
    // downstream as "re-size me". specFromSearch handed it the spec's own
    // maxContext, which then went back through specWithCtx's
    // floor(maxSeq/pageSize) ceiling — a no-op on ten of eleven shipped specs
    // and a silent SHRINK on Phi-3, whose 257 KV pages hold 4112 tokens
    // against a 4096-token window. Every plain page load of the default model
    // lost the 257th page. ctxFor is gone; nothing here may reintroduce a
    // reader that cannot say ABSENT.
    expect(ctxFrom('?model=qwen35')).toBeNull()
    expect(ctxFrom('?model=qwen35&ctx=8192')).toBe(8192)
  })

  // Which ctx wins — the link's, over this device's compiled default — is a
  // property of the CALLER now, and is tested on the shipped function in
  // tests/unit/ctx-override.test.ts. It is not restated here against a
  // hand-rolled `?? ownDefault`, which is the shape that hid the shrink.
})

describe('roomLink', () => {
  it('writes the query BEFORE the fragment, every time', () => {
    // The whole reason this builder exists. Assert on the ROUTE, not on the
    // string: a link that reads right and routes wrong is the actual bug.
    const helper = roomLink({
      origin: ORIGIN, path: '/share.html', room: ROOM,
      model: 'qwen3mlx', layers: { start: 18, end: 36 }, ctx: 8192,
    })
    expect(helper.indexOf('?')).toBeLessThan(helper.indexOf('#'))
    expect(routeOf(helper)).toBe('helper')
    const u = new URL(helper)
    expect(stageRangeFrom(u.search)).toEqual({ start: 18, end: 36 })
    expect(ctxFrom(u.search)).toBe(8192)
    expect(roomIdFrom(u.hash)).toBe(ROOM)
  })

  it('never puts the room id in the query — it is 128 bits of secret', () => {
    // Fragments are not sent over HTTP, so the static host never logs the id.
    // A query would hand it to every access log between here and the origin.
    const link = roomLink({
      origin: ORIGIN, path: '/share.html', room: ROOM, model: 'qwen3mlx', ctx: 4096,
    })
    expect(new URL(link).search).not.toContain(ROOM)
    expect(new URL(link).hash).toBe(`#${ROOM}`)
  })

  it('omits every key it was not given', () => {
    expect(roomLink({ origin: ORIGIN, path: '/share.html', room: ROOM }))
      .toBe(`${ORIGIN}/share.html#${ROOM}`)
    expect(routeOf(roomLink({ origin: ORIGIN, path: '/share.html', room: ROOM }))).toBe('guest')
  })

  it('model + ctx inside a room is a SECOND FULL HOST, not a helper', () => {
    // This is what room-host.ts emits when it holds the whole model, and what
    // a guest that copied the weights opens. It must keep routing to host —
    // `start > 0` is the only thing that makes a link a helper.
    const link = roomLink({
      origin: ORIGIN, path: '/share.html', room: ROOM, model: 'qwen3mlx', ctx: 8192,
    })
    expect(routeOf(link)).toBe('host')
    expect(ctxFrom(new URL(link).search)).toBe(8192)
  })

  it("model:'' is the DEFAULT model, not an absent one", () => {
    // room-host.ts passes the raw `?model=` value, which is '' for Phi-3.
    // Dropping it would emit a "help this room" link that routes to GUEST —
    // the recipient joins, runs nothing, and the room never gains a machine.
    const link = roomLink({ origin: ORIGIN, path: '/share.html', room: ROOM, model: '', ctx: 4096 })
    expect(routeOf(link)).toBe('host')
    expect(new URL(link).searchParams.has('model')).toBe(true)
    // null/undefined still means "no model" — that is the guest link.
    expect(routeOf(roomLink({ origin: ORIGIN, path: '/share.html', room: ROOM, model: null }))).toBe('guest')
  })

  it('a null room writes a link that opens a NEW room', () => {
    const link = roomLink({ origin: ORIGIN, path: '/share.html', room: null, model: 'qwen3mlx', ctx: 8192 })
    expect(link).not.toContain('#')
    expect(routeOf(link)).toBe('host')
  })
})

describe('swarmUrls carries ctx to every SERVING machine', () => {
  it('puts the same ctx on the host and on every helper', () => {
    const stops = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 3, room: ROOM, ctx: 8192 })
    for (const s of stops) {
      expect(routeOf(s.url!), `${s.url} routes to the wrong role`).toBe(s.role)
      if (s.role === 'guest') continue
      expect(ctxFrom(new URL(s.url!).search), `${s.role} lost its ctx`).toBe(8192)
    }
  })

  it('leaves the guest link bare even when a ctx is given', () => {
    // A guest runs nothing locally, so it has no KV cache to size — and every
    // key on a guest link is one more chance to write ?model= into a link that
    // must not host.
    const guest = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 3, room: ROOM, ctx: 8192 }).at(-1)!
    expect(guest.role).toBe('guest')
    expect(new URL(guest.url!).search).toBe('')
    expect(routeOf(guest.url!)).toBe('guest')
  })

  it('omitting ctx writes no ?ctx= at all — each machine keeps its own default', () => {
    const stops = swarmUrls({ origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 2, room: ROOM })
    for (const s of stops) expect(ctxFrom(new URL(s.url!).search)).toBeNull()
  })
})

// ── the room id is validated INSIDE the builder ─────────────────────────────
//
// roomLink concatenated `#${p.room}` raw while every other field went through
// URLSearchParams, and its contract never said the caller had to validate the
// id. That held only by caller discipline: share.ts and landing-swarm.ts both
// run their id through roomIdFrom first, and landing-room.ts never passes
// `existingRoom`, so its id is hostRoom's own crypto.getRandomValues.
//
// `existingRoom` IS a supported hostRoom option, and "join an existing room
// from the entrance" is the obvious next feature — while landing-room.ts's
// paintSplit already writes swarmUrls' output into
// `<input readonly value="${st.url}">`. The day those two meet, an unvalidated
// id is an attribute-injection sink, and this site ships no CSP
// (public/_headers sets COOP/COEP only), so nothing would mitigate it. These
// pin the guarantee to the function rather than to caller discipline.
describe('roomLink validates the room id itself', () => {
  const BAD: [string, string][] = [
    ['a quote closes the attribute', `${ROOM}" onfocus=alert(1) x="`],
    ['an angle bracket opens a tag', `${ROOM}<img src=x onerror=alert(1)>`],
    ['a space ends the attribute value', `${ROOM} onfocus=alert(1)`],
    ['too short to be 128 bits', 'short'],
    ['a whole link, not an id', `https://evil.example/#${ROOM}`],
    ['a second fragment', `${ROOM}#${ROOM}`],
    ['a query smuggled into the fragment', `${ROOM}?model=qwen3mlx`],
  ]
  for (const [why, bad] of BAD) {
    it(`refuses ${why}`, () => {
      expect(() => roomLink({ origin: ORIGIN, path: '/share.html', room: bad })).toThrow(/room id/i)
    })
  }

  it('refuses through swarmUrls too — the sink paintSplit writes into', () => {
    expect(() => swarmUrls({
      origin: ORIGIN, param: 'qwen3mlx', layers: 36, machines: 3,
      room: `${ROOM}" onfocus=alert(1) x="`,
    })).toThrow(/room id/i)
  })

  // One pattern, not two. A second copy of a security regex is how they drift:
  // roomIdFrom would go on refusing what roomLink had quietly started to allow.
  it('accepts exactly what roomIdFrom accepts', () => {
    const CANDIDATES = [
      ROOM, 'A'.repeat(16), 'A'.repeat(64), '-_-_-_-_-_-_-_-_',
      'A'.repeat(15), 'A'.repeat(65), '', 'has space here!!', 'aaaaaaaaaaaaaaaa.',
      `${ROOM}"`, `${ROOM}<`, `${ROOM}%22`, 'aaaaaaaaaaaaaaaa/../..',
    ]
    for (const id of CANDIDATES) {
      const parses = roomIdFrom(`#${id}`) !== null
      let builds = true
      try { roomLink({ origin: ORIGIN, path: '/share.html', room: id }) } catch { builds = false }
      expect(builds, `roomIdFrom and roomLink disagree about ${JSON.stringify(id)}`).toBe(parses)
    }
  })

  it('still writes the links the app actually asks for', () => {
    // The two real sources of an id today: hostRoom's crypto bytes, and a
    // string that already passed roomIdFrom. Neither may start throwing.
    const bytes = new Uint8Array(16).fill(200)
    const crypted = btoa(String.fromCharCode(...bytes))
      .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
    expect(routeOf(roomLink({ origin: ORIGIN, path: '/share.html', room: crypted }))).toBe('guest')
    expect(routeOf(roomLink({ origin: ORIGIN, path: '/share.html', room: null, model: 'qwen3mlx' }))).toBe('host')
  })
})
