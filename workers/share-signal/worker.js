/**
 * SHARE-SIGNAL — the tiny WebRTC signaling relay behind share.html.
 *
 * One Durable Object instance per room. It relays SDP offers/answers and ICE
 * candidates between the HOSTS (tabs running the model) and the GUESTS, and
 * nothing else: once a DataChannel opens, prompts, tokens and replicated
 * weights flow browser to browser over DTLS and never touch this worker. A
 * room's whole server-side footprint is a few KB of signaling JSON.
 *
 * The room id is 128 bits from crypto.getRandomValues, carried in the link's
 * #fragment (which browsers never send to the static host). Whoever has the
 * link can join — same trust model as an unlisted meet link.
 *
 * A room holds MANY hosts. That is the swarm shape: a guest can copy the
 * weights (peer-weights.ts) and then rejoin the same room as another host, so
 * capacity grows with participants rather than resting on one machine. The DO
 * assigns each guest to the least-loaded host at join time and REASSIGNS its
 * guests when a host disappears — churn is the normal case here, not an edge
 * case, because every host is somebody's browser tab.
 *
 * Message routing:
 *   guest → {…}            to its assigned host as {…, from: <guestId>}
 *   host  → {to: g, …}     to guest g
 *   join/leave/reassign    hosts get {type:'peer-joined'|'peer-left', from};
 *                          guests get {type:'host-changed'} and renegotiate
 *   both sides get {type:'room', hosts, guests} whenever membership changes
 *
 *   npx wrangler dev --port 8787     # local (miniflare) — what the e2e uses
 *   npx wrangler deploy              # production (user-approved only)
 */

// Message types the relay itself originates; clients must not spoof them.
const RELAY_TYPES = new Set(['room', 'host-changed', 'host-left', 'no-host', 'peer-joined', 'peer-left'])

export class Room {
  constructor(state) {
    this.state = state
    this.hosts = new Map()      // hostId -> WebSocket
    this.guests = new Map()     // guestId -> WebSocket
    this.assigned = new Map()   // guestId -> hostId
  }

  fetch(request) {
    if (request.headers.get('Upgrade') !== 'websocket') {
      return new Response('websocket only', { status: 400 })
    }
    const url = new URL(request.url)
    const role = url.searchParams.get('role')
    if (role !== 'host' && role !== 'guest') {
      return new Response('role=host|guest required', { status: 400 })
    }

    const pair = new WebSocketPair()
    const [client, server] = Object.values(pair)
    server.accept()
    // Per-peer session id. Room ids live in the URL path and are created by
    // clients as 128 bits of base64url (22 chars), which already satisfies the
    // 16-64 char range the route accepts; this short id is intentionally not a
    // room id.
    const id = crypto.randomUUID().slice(0, 8)

    if (role === 'host') this.addHost(id, server)
    else this.addGuest(id, server)

    return new Response(null, { status: 101, webSocket: client })
  }

  // ── membership ────────────────────────────────────────────

  addHost(id, ws) {
    this.hosts.set(id, ws)
    ws.addEventListener('message', (e) => this.fromHost(e.data))
    ws.addEventListener('close', () => {
      this.hosts.delete(id)
      // Hand this host's guests to a survivor, or tell them the room is empty.
      for (const [guestId, hostId] of [...this.assigned]) {
        if (hostId !== id) continue
        this.assigned.delete(guestId)
        const guest = this.guests.get(guestId)
        if (!guest) continue
        const next = this.leastLoadedHost()
        if (next) {
          // ORDER MATTERS: the guest must be told to tear down its old
          // connection BEFORE the new host is asked to build one, or the
          // replacement offer can reach a page still wired to the dead peer.
          this.send(guest, { type: 'host-changed' })
          this.assign(guestId, next)
        } else {
          this.send(guest, { type: 'host-left' })
        }
      }
      this.broadcastRoom()
    })
    // A late host adopts anyone still waiting without one — same ordering
    // rule as the takeover path above.
    for (const [guestId] of this.guests) {
      if (!this.assigned.has(guestId)) {
        this.send(this.guests.get(guestId), { type: 'host-changed' })
        this.assign(guestId, id)
      }
    }
    this.broadcastRoom()
  }

  addGuest(id, ws) {
    this.guests.set(id, ws)
    ws.addEventListener('message', (e) => this.fromGuest(id, e.data))
    ws.addEventListener('close', () => {
      const hostId = this.assigned.get(id)
      this.guests.delete(id)
      this.assigned.delete(id)
      const host = hostId && this.hosts.get(hostId)
      if (host) this.send(host, { type: 'peer-left', from: id })
      this.broadcastRoom()
    })
    const host = this.leastLoadedHost()
    // No model on the other end — fail fast instead of a hanging spinner.
    if (!host) this.send(ws, { type: 'no-host' })
    else this.assign(id, host)
    this.broadcastRoom()
  }

  /** Fewest guests wins; ties go to the earliest host (insertion order). */
  leastLoadedHost() {
    let best = null
    let bestLoad = Infinity
    for (const hostId of this.hosts.keys()) {
      let load = 0
      for (const h of this.assigned.values()) if (h === hostId) load++
      if (load < bestLoad) { best = hostId; bestLoad = load }
    }
    return best
  }

  /** Point a guest at a host and ask that host to open the connection.
   *  Re-checks liveness immediately before sending: a host can disappear
   *  between leastLoadedHost() and this call, and a guest must never be
   *  left assigned to a dead host. */
  assign(guestId, hostId) {
    let chosen = this.hosts.has(hostId) ? hostId : this.leastLoadedHost()
    if (chosen && !this.hosts.has(chosen)) chosen = this.leastLoadedHost()
    if (!chosen || !this.hosts.has(chosen)) {
      const guest = this.guests.get(guestId)
      if (guest) this.send(guest, { type: 'no-host' })
      return
    }
    this.assigned.set(guestId, chosen)
    const host = this.hosts.get(chosen)
    if (host) this.send(host, { type: 'peer-joined', from: guestId })
  }

  broadcastRoom() {
    const msg = { type: 'room', hosts: this.hosts.size, guests: this.guests.size }
    for (const ws of this.hosts.values()) this.send(ws, msg)
    for (const ws of this.guests.values()) this.send(ws, msg)
  }

  // ── relaying ──────────────────────────────────────────────

  fromHost(data) {
    let msg
    try { msg = JSON.parse(data) } catch { return }
    if (typeof msg !== 'object' || msg == null) return
    // Hosts must not emit relay-originated control types (e.g. peer-joined,
    // host-changed). Drop them rather than forward.
    if (RELAY_TYPES.has(msg.type)) return
    const ws = this.guests.get(msg.to)
    if (ws) {
      delete msg.to
      this.send(ws, msg)
    }
  }

  fromGuest(id, data) {
    let msg
    try { msg = JSON.parse(data) } catch { return }
    if (typeof msg !== 'object' || msg == null) return
    // Guests must not emit relay-originated control types (e.g. host-changed).
    // Drop them rather than forward.
    if (RELAY_TYPES.has(msg.type)) return
    const host = this.hosts.get(this.assigned.get(id))
    if (host) this.send(host, { ...msg, from: id })
  }

  send(ws, obj) {
    try { ws.send(JSON.stringify(obj)) } catch { /* peer raced a close */ }
  }
}

export default {
  fetch(request, env) {
    const url = new URL(request.url)
    const m = url.pathname.match(/^\/room\/([A-Za-z0-9_-]{16,64})$/)
    if (!m) return new Response('expected /room/<id>', { status: 404 })
    const room = env.ROOMS.get(env.ROOMS.idFromName(m[1]))
    return room.fetch(request)
  },
}
