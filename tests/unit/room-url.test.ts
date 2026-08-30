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
import { roleFor, roomIdFrom, splitBounds, stageRangeFrom, swarmUrls } from '../../src/zero-tvm/room-url.ts'
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
