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

/**
 * The 128 random bits room-host.ts puts in the link fragment, base64url.
 *
 * ONE definition, read by the PARSER below and by the BUILDER at the bottom of
 * this file. A second copy would be two copies of a security check, and two
 * copies drift: the parser would go on refusing what the builder had quietly
 * started to emit. No `g` flag, so it carries no lastIndex between calls.
 */
const ROOM_ID = /^[A-Za-z0-9_-]{16,64}$/

/** The room id out of a fragment, with or without the leading `#`. */
export function roomIdFrom(hash: string): string | null {
  const id = hash.replace(/^#/, '')
  return ROOM_ID.test(id) ? id : null
}

/** `?layers=0-20` — this device holds that slice of the model, nothing else. */
export function stageRangeFrom(search: string): { start: number; end: number } | null {
  const m = /^(\d+)-(\d+)$/.exec(new URLSearchParams(search).get('layers') ?? '')
  return m ? { start: Number(m[1]), end: Number(m[2]) } : null
}

/**
 * The smallest context a link may ask for.
 *
 * 256 is the figure `landing-swarm.ts`'s context field has clamped to since it
 * was written (`min="256" step="256"`, and a `Math.max(…, 256)` behind it).
 * Anything under it is a window no conversation fits in, and the cost of
 * honouring one is a multi-gigabyte download for an engine that cannot answer.
 * That file still writes its own literal; this is the copy the URL side reads,
 * and it is the one every surface inherits, because every surface reads
 * `?ctx=` through ctxFrom.
 */
export const MIN_CTX = 256

/**
 * `?ctx=N` — the KV budget the room runs at, in tokens. THE ONE READER of that
 * key; every surface that sizes a KV cache from a URL comes through here.
 *
 * CONTEXT PRECEDENCE — the LINK wins over this device's own compiled default.
 *
 * WHY: every stage allocates its own KV cache from spec.maxContext, eagerly,
 * one buffer per attention layer. A helper that fell back to its build's
 * default while the host ran `?ctx=8192` would hold a DIFFERENT number of
 * slots for the same conversation — the host prefills to a position its own
 * cache has room for and the stage behind it does not, so the room dies deep
 * inside a generation instead of at boot, and the two tabs disagree about a
 * number neither of them prints. The host therefore writes its EFFECTIVE
 * maxContext (post-clamp, from the spec it actually built) into the links it
 * hands out, and a stage adopts it verbatim.
 *
 * NULL WHEN THE LINK CARRIES NO ctx, and null must stay a distinct answer from
 * a number. There used to be a `ctxFor(search, ownDefault)` here that collapsed
 * the two, returning the caller's own default for an absent key — and a number
 * always reads downstream as "re-size me". `specFromSearch` passed the spec's
 * own maxContext as that default and fed it back through `specWithCtx`, whose
 * ceiling is floor(maxSeq / pageSize). A no-op on ten of eleven shipped specs,
 * and a silent SHRINK on Phi-3, the default model of zero-tvm.html and
 * validate.html: KV pages round UP, so its 257 pages hold 4112 tokens against
 * a 4096-token window, and every plain page load lost the 257th page (measured
 * 4112 → 4096). Not asking for a budget is not the same as
 * asking for the one you already have, and no signature that cannot say
 * "absent" can keep those apart. ctxFor was deleted with that fix rather than
 * left exported and tested with no caller.
 *
 * A BUDGET IS A WHOLE NUMBER OF TOKENS, AND AT LEAST MIN_CTX OF THEM. The
 * guard was `n > 0`, which admits fractions and sub-page values, and every
 * surface then honoured them: `share.html?model=qwen38&ctx=0.5` rendered
 * "Context | 16 tokens" and `zero-tvm.html?model=qwen38&ctx=0.5` rendered
 * "0K CONTEXT" — both about to fetch 14.1 GB of weights for a window that
 * cannot hold a prompt, let alone a conversation. (specWithCtx's floor is ONE
 * PAGE, so 0.5 does not error anywhere; it quietly builds a 16-token cache.)
 * The swarm's own context field has clamped to >= 256 since it was written;
 * the URL reader, which is where the same number arrives from a link, did not.
 * It does now, so every surface inherits the floor from one place.
 *
 * REFUSED, not clamped: a malformed budget is NO budget, and the caller's own
 * compiled default is a better answer than a number this module invented. It
 * is the same rule the entrance applies to a malformed `?split=`.
 */
export function ctxFrom(search: string): number | null {
  const n = Number(new URLSearchParams(search).get('ctx'))
  return Number.isInteger(n) && n >= MIN_CTX ? n : null
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

/** Clamp a set of interior cut points into a legal ascending split.
 *
 *  An EVEN split is the wrong default for real hardware and was never what a
 *  working swarm looked like: the session this feature came from put 63 layers
 *  on a laptop and ONE on a phone, because a phone cannot hold a sixteenth of
 *  a 27B. So the builder lets each boundary move, and this keeps the result
 *  well-formed — ascending, inside the model, and never an empty stage, which
 *  `offerStage` would refuse as `layers a-a is not a range`. */
export function clampBounds(layers: number, cuts: readonly number[]): number[] {
  const n = cuts.length + 1                       // stages, not cut points
  const out: number[] = [0]
  for (let i = 0; i < cuts.length; i++) {
    // Leave room for every stage still to come, and for this one to be >= 1.
    const lo = out[i] + 1
    const hi = layers - (n - 1 - i)
    out.push(Math.min(Math.max(Math.round(cuts[i]), lo), Math.max(lo, hi)))
  }
  out.push(layers)
  return out
}

export interface SwarmStop {
  role: RoomRole
  /** Layers this machine holds; absent for a guest, which holds none. */
  range?: { start: number; end: number }
  /** null until the room exists — a helper or guest link cannot be written
   *  before the host has opened its room and handed out the id. */
  url: string | null
}

export interface RoomLinkParts {
  /** Scheme + host, no trailing slash. */
  origin: string
  /** Page path, e.g. `/share.html`. */
  path: string
  /** The room id. FRAGMENT, always — it is 128 bits of secret, and browsers
   *  do not send fragments over HTTP, so the static host never logs it. Null
   *  writes a link that opens a NEW room. Anything that is neither null nor a
   *  ROOM_ID makes roomLink THROW — see the note there. */
  room: string | null
  /** `?model=` — LEAVE UNSET (or null) for a link that must route as a guest:
   *  roleFor reads a model inside a room as "this device serves too". The
   *  EMPTY STRING is a real value, not "unset": it is how the default model
   *  is named in this grammar (specForParam('') is Phi-3), and `?model=`
   *  routes to a serving role exactly like a named one. */
  model?: string | null
  /** `?layers=` — the slice the opener holds. */
  layers?: { start: number; end: number } | null
  /** `?ctx=` — see ctxFrom. */
  ctx?: number | null
  /** `?sig=` — dev signaling override, carried so a peer dials the same relay
   *  the host is on rather than the default one. */
  sig?: string | null
}

/**
 * The ONE place a room URL is assembled: query first, fragment last.
 *
 * That order is the whole point. Written the other way round —
 * `#<room>?model=X` — location.search is empty and location.hash no longer
 * parses as a room id, so roleFor takes the host branch and the device opens
 * a brand-new room in silence. Every caller goes through here so no caller
 * can get it wrong by hand.
 *
 * THE ROOM ID IS VALIDATED HERE, not by the callers.
 *
 * It is the only field that does not go through URLSearchParams — it is a
 * fragment, so it is written raw — and the only one that can arrive from
 * outside this build: `hostRoom`'s `existingRoom` option, or a link somebody
 * pasted. Every caller happens to check it today (share.ts and
 * landing-swarm.ts both run theirs through roomIdFrom; landing-room.ts passes
 * no `existingRoom`, so hostRoom's own crypto bytes are the id), but nothing
 * pinned that, and the contract never asked for it. Meanwhile
 * landing-room.ts's `paintSplit` writes what comes out of here straight into
 * `<input readonly value="…">`, and the site ships no CSP. "Join an existing
 * room from the entrance" is one small feature away from turning that into an
 * injection sink. The guarantee belongs to this function.
 *
 * THROW rather than drop the bad id. Dropping it would emit a perfectly
 * well-formed URL with no fragment — which roleFor reads as HOST, so the
 * recipient opens a BRAND NEW room and the two machines sit waiting for each
 * other with nothing on screen saying why. That is the exact silent failure
 * the header of this file exists to prevent, and reaching it through the
 * validation would be worse than reaching it through a typo. A throw cannot
 * strand a user in its place: both real sources of an id (16 bytes of
 * crypto.getRandomValues, or a string that already passed roomIdFrom) satisfy
 * ROOM_ID, so this fires only on a genuinely new bug — and it fires at the
 * builder, with the caller on the stack, instead of inside rendered markup.
 *
 * `origin` and `path` are concatenated raw too, and deliberately: neither is
 * remote-controlled. `origin` is `location.origin` or `''`, browser-normalised
 * to scheme + host; `path` is the literal `/share.html` everywhere except
 * share.ts, which passes `location.pathname` — already percent-encoded by the
 * browser, `<`, `>` and `"` included. Nothing else in this module reaches a
 * URL or markup by concatenation.
 */
export function roomLink(p: RoomLinkParts): string {
  if (p.room != null && !ROOM_ID.test(p.room)) {
    throw new Error(`roomLink: invalid room id ${JSON.stringify(p.room)}`)
  }
  const q = new URLSearchParams()
  // `!= null`, not truthiness: `model: ''` is the DEFAULT model, and dropping
  // it turns a serving link into a guest link that quietly runs nothing.
  if (p.model != null) q.set('model', p.model)
  if (p.layers) q.set('layers', `${p.layers.start}-${p.layers.end}`)
  if (p.ctx) q.set('ctx', String(p.ctx))
  if (p.sig) q.set('sig', p.sig)
  const search = q.toString()
  return `${p.origin}${p.path}${search ? `?${search}` : ''}${p.room ? `#${p.room}` : ''}`
}

/**
 * Every URL a split needs, in the order they must be opened.
 *
 * The room id does not exist until the host opens its room, so helper and
 * guest URLs are null until one is supplied. Emitting them with a placeholder
 * fragment would be worse than emitting nothing: a placeholder fails
 * roomIdFrom, which routes the link to the host branch, which is the silent
 * new-room failure this module exists to prevent.
 *
 * The guest link carries NO query at all — not even ctx. A guest runs nothing
 * locally, so it has no KV cache to size, and every key it could carry is one
 * more chance to write `?model=` into a link that must not host.
 */
export function swarmUrls(o: {
  origin: string
  /** `?model=` value, from the registry — never a hand-typed id. */
  param: string
  layers: number
  machines: number
  room: string | null
  /** `?ctx=` for the serving links, so every stage sizes the same KV cache
   *  (ctxFrom). Omit to let each machine use its build's default. */
  ctx?: number | null
  /** Interior cut points, when the operator has moved them. Omit for an even
   *  split. Clamped by clampBounds, so a caller cannot produce an empty or
   *  descending stage. */
  cuts?: readonly number[] | null
}): SwarmStop[] {
  const bounds = o.cuts && o.cuts.length === o.machines - 1
    ? clampBounds(o.layers, o.cuts)
    : splitBounds(o.layers, o.machines)
  const at = (i: number): { start: number; end: number } => ({ start: bounds[i], end: bounds[i + 1] })
  const serve = (r: { start: number; end: number }, room: string | null): string =>
    roomLink({ origin: o.origin, path: '/share.html', room, model: o.param, layers: r, ctx: o.ctx })
  const out: SwarmStop[] = [{ role: 'host', range: at(0), url: serve(at(0), null) }]
  for (let i = 1; i < bounds.length - 1; i++) {
    const range = at(i)
    out.push({ role: 'helper', range, url: o.room ? serve(range, o.room) : null })
  }
  out.push({
    role: 'guest',
    url: o.room ? roomLink({ origin: o.origin, path: '/share.html', room: o.room }) : null,
  })
  return out
}
