/**
 * SHARE-SIGNAL — the tiny WebRTC signaling relay behind share.html.
 *
 * One Durable Object instance per room. It relays SDP offers/answers and ICE
 * candidates between the HOST (the tab running the model) and its GUESTS, and
 * nothing else: once the DataChannel opens, prompts and tokens flow browser to
 * browser over DTLS and never touch this worker. A room's whole server-side
 * footprint is a few KB of signaling JSON.
 *
 * The room id IS the capability: 128 bits from crypto.getRandomValues, carried
 * in the link's #fragment (which browsers never send to the static host).
 * Whoever has the link can join — same trust model as an unlisted meet link.
 * (Zero-knowledge signaling — encrypting the SDP blobs with a second fragment
 * secret — is deliberately NOT v1; the relay sees session descriptions, not
 * conversation content.)
 *
 * Message routing:
 *   guest → {…}            relayed to the host as {…, from: <guestId>}
 *   host  → {to: g, …}     relayed to guest g
 *   join/leave             host gets {type:'peer-joined'|'peer-left', from}
 *
 * Multiple guests are allowed — the HOST holds one RTCPeerConnection per
 * guest and serializes generation (the engine is single-stream); the room
 * only routes.
 *
 *   npx wrangler dev --port 8787     # local (miniflare) — what the e2e uses
 *   npx wrangler deploy              # production (user-approved only)
 */

export class Room {
  constructor(state) {
    this.state = state
    this.host = null                 // WebSocket | null
    this.guests = new Map()          // id -> WebSocket
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

    if (role === 'host') {
      // A reconnecting host replaces the old socket (tab reload).
      try { this.host?.close(4000, 'replaced') } catch { /* already gone */ }
      this.host = server
      server.addEventListener('message', (e) => this.fromHost(e.data))
      server.addEventListener('close', () => {
        if (this.host === server) {
          this.host = null
          // Guests learn the model is gone and can show "host left".
          for (const ws of this.guests.values()) this.send(ws, { type: 'host-left' })
        }
      })
      // Late host (guests waiting): announce everyone already here.
      for (const id of this.guests.keys()) this.send(server, { type: 'peer-joined', from: id })
    } else {
      const id = crypto.randomUUID().slice(0, 8)
      if (!this.host) {
        // No model on the other end — fail fast instead of a hanging spinner.
        this.send(server, { type: 'no-host' })
      }
      this.guests.set(id, server)
      server.addEventListener('message', (e) => this.fromGuest(id, e.data))
      server.addEventListener('close', () => {
        this.guests.delete(id)
        if (this.host) this.send(this.host, { type: 'peer-left', from: id })
      })
      if (this.host) this.send(this.host, { type: 'peer-joined', from: id })
    }

    return new Response(null, { status: 101, webSocket: client })
  }

  fromHost(data) {
    let msg
    try { msg = JSON.parse(data) } catch { return }
    const ws = this.guests.get(msg.to)
    if (ws) {
      delete msg.to
      this.send(ws, msg)
    }
  }

  fromGuest(id, data) {
    let msg
    try { msg = JSON.parse(data) } catch { return }
    if (this.host) this.send(this.host, { ...msg, from: id })
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
