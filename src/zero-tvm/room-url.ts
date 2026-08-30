/**
 * ROOM URL GRAMMAR — one parser for the three roles share.html can take, and
 * the builder that writes those URLs for the entrance.
 *
 * The grammar is:
 *   share.html?model=X&layers=0-k      HOST    — opens a new room, holds the
 *                                                first layers
 *   share.html?model=X&layers=k-N#<r>  HELPER  — joins room <r>, holds the rest
 *   share.html#<r>                     GUEST   — chats, runs nothing locally
 *
 * It fails SILENTLY when written wrong. Put the query after the fragment —
 * `share.html#<room>?model=X&layers=18-36`, which is the order a URL bar
 * autocompletes into if you paste the room link and then type — and
 * location.search is empty while location.hash no longer matches a room id.
 * Both tests below fall through to the host branch, so that device opens a
 * BRAND NEW room and the two machines sit waiting for each other with nothing
 * on screen saying why. That cost a real user an hour on 2026-08-29.
 *
 * share.ts routes on the functions here rather than on its own copies, so a
 * URL the entrance emits can be run through roleFor() in a test and checked
 * against the role it was labelled with. A builder that emits a URL share.ts
 * would route somewhere else is the exact bug this file exists to prevent.
 *
 * Dependency-free on purpose: share.ts imports it on the GUEST path, which
 * must run on machines with no WebGPU at all.
 */

/** The 128 random bits room-host.ts puts in the link fragment, base64url. */
export function roomIdFrom(hash: string): string | null {
  const id = hash.replace(/^#/, '')
  return /^[A-Za-z0-9_-]{16,64}$/.test(id) ? id : null
}

/** `?layers=0-20` — this device holds that slice of the model, nothing else. */
export function stageRangeFrom(search: string): { start: number; end: number } | null {
  const m = /^(\d+)-(\d+)$/.exec(new URLSearchParams(search).get('layers') ?? '')
  return m ? { start: Number(m[1]), end: Number(m[2]) } : null
}

export type RoomRole = 'host' | 'helper' | 'guest'

/**
 * Which of the three share.html becomes. A stage that does not start the model
 * cannot serve chat — it has no embedding and no tokenizer role — so it JOINS
 * someone else's room and offers its layers there; everything else follows the
 * grammar above.
 */
export function roleFor(search: string, hash: string): RoomRole {
  const room = roomIdFrom(hash)
  const stage = stageRangeFrom(search)
  if (room && stage && stage.start > 0) return 'helper'
  if (room && !new URLSearchParams(search).has('model')) return 'guest'
  return 'host'
}

/**
 * Layer boundaries for an even split: [0, …, layers], one more entry than
 * there are machines.
 *
 * Even because there is nothing here to weight it by. The first stage also
 * carries the embedding and the last the lm_head, so the ends are slightly
 * heavier — but the sizes of those depend on the checkpoint and a guess
 * dressed as a plan is worse than a straight division.
 */
export function splitBounds(layers: number, machines: number): number[] {
  const n = Math.max(2, Math.min(Math.floor(machines), layers))
  return Array.from({ length: n + 1 }, (_, i) => Math.round((i * layers) / n))
}

export interface SwarmStop {
  role: RoomRole
  /** Layers this machine holds; absent for a guest, which holds none. */
  range?: { start: number; end: number }
  /** null until the room exists — a helper or guest link cannot be written
   *  before the host has opened its room and handed out the id. */
  url: string | null
}

/**
 * Every URL a split needs, in the order they must be opened.
 *
 * The room id does not exist until the host opens its room, so helper and
 * guest URLs are null until one is supplied. Emitting them with a placeholder
 * fragment would be worse than emitting nothing: a placeholder fails
 * roomIdFrom, which routes the link to the host branch, which is the silent
 * new-room failure this module exists to prevent.
 */
export function swarmUrls(o: {
  origin: string
  /** `?model=` value, from the registry — never a hand-typed id. */
  param: string
  layers: number
  machines: number
  room: string | null
}): SwarmStop[] {
  const bounds = splitBounds(o.layers, o.machines)
  const at = (i: number): { start: number; end: number } => ({ start: bounds[i], end: bounds[i + 1] })
  const q = (r: { start: number; end: number }): string =>
    `${o.origin}/share.html?model=${encodeURIComponent(o.param)}&layers=${r.start}-${r.end}`
  const out: SwarmStop[] = [{ role: 'host', range: at(0), url: q(at(0)) }]
  for (let i = 1; i < bounds.length - 1; i++) {
    const range = at(i)
    out.push({ role: 'helper', range, url: o.room ? `${q(range)}#${o.room}` : null })
  }
  out.push({ role: 'guest', url: o.room ? `${o.origin}/share.html#${o.room}` : null })
  return out
}
