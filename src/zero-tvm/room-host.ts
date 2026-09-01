/**
 * ROOM HOST CORE — serve the engine in THIS tab to guests over WebRTC.
 *
 * Extracted from share.ts's runHost so two surfaces can host without drift:
 * share.html (the standalone page, e2e-covered) and the landing entrance's
 * in-place chat (landing-room.ts, where the room is part of the game). Same
 * rule as chat-flow.ts: one host loop, page-specific chrome behind callbacks.
 *
 * DOM-free by contract: everything the host's eyes need arrives through
 * RoomUI. No GPU globals at module scope — the engine comes in as a value,
 * and every import here is proven safe on WebGPU-less machines (share.ts has
 * always imported them statically on the guest path).
 *
 * Protocol, privacy model and the serialized-FIFO rule are unchanged — see
 * share.ts's header. The signaling endpoint constants live HERE now, single-
 * sourced for host, guest and helper.
 */

import type { ModelSpec } from '../compiler/model-spec.js'
import type { DecodeEngine } from './engine-core.js'
import { quantTagFor } from './chat-ui.js'
import { serveWeights } from './peer-weights.js'
import { makeStageClient } from './pipeline-peer.js'
// The chain-assembly rules and the split token loop live in room-chain.ts —
// pure, WebRTC-free, and unit-tested there. This file wires the channels to it.
import { makeRoomChain, type ChainMsg, type StageOffer } from './room-chain.js'

// ── signaling endpoint (single source for host/guest/helper) ──
const DEV = !!(import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV
const PROD_SIGNAL = 'wss://zero-tvm-share-signal.abgunaydin94.workers.dev'  // deployed 2026-08-06
/** `?sig=<port|ws-url>` overrides the relay — DEV ONLY, so a shared link can
 *  never point a guest's signaling at a third party in production. */
export function signalEnv(): { base: string; override: string | null } {
  const override = DEV ? new URLSearchParams(location.search).get('sig') : null
  const base = override
    ? (/^wss?:\/\//.test(override) ? override : `ws://localhost:${override}`)
    : DEV ? 'ws://localhost:8787' : PROD_SIGNAL
  return { base, override }
}

export const ICE: RTCConfiguration = {
  // STUN only, deliberately: same-network and home-NAT paths connect
  // directly. Hostile (corporate) NATs need a TURN relay — a later problem,
  // and a separate consent: TURN routes ciphertext through a third party.
  iceServers: [{ urls: ['stun:stun.cloudflare.com:3478', 'stun:stun.l.google.com:19302'] }],
}

// ── wire protocol ─────────────────────────────────────────────
export interface ChatMsg { role: 'system' | 'user' | 'assistant'; content: string }
export interface ChatReq { type: 'chat'; id: number; messages: ChatMsg[] }
export interface StopReq { type: 'stop'; id: number }
export type { StageOffer }
type GuestMsg = ChatReq | StopReq | StageOffer
export type HostMsg =
  | { type: 'info'; name: string; params: string; rateLabel: string; tag: string; param: string; specId: string }
  | { type: 'text'; id: number; full: string }
  | { type: 'done'; id: number; tokens: number; tps: number }
  | { type: 'busy'; id: number; pos: number }
  | { type: 'error'; id: number; message: string }
  | ChainMsg

// ── the host's eyes ───────────────────────────────────────────
export interface RoomUI {
  /** One request/event row — your machine should not run someone's prompt
   *  without you seeing it. Returns a setter for the row's status cell. */
  row: (who: string, text: string) => (status: string) => void
  /** Membership: signaller counts + how many guests THIS host has channels
   *  to (the number a stage full of mascots draws). */
  onMembers?: (m: { hosts: number; guests: number; connected: number }) => void
  /** Split-model chain state, human-readable ('' when not split). */
  onChain?: (text: string) => void
  /** The chain now tiles the whole model (or stopped doing so). */
  onPaired?: (complete: boolean) => void
  /** One real token generated for a guest — the entrance mascot's mouth. */
  onToken?: () => void
  /** The engine started/stopped serving a guest request. */
  onBusy?: (busy: boolean) => void
}

export interface HostRoomOptions {
  spec: ModelSpec
  brand: { name: string; params: string; rateLabel: string }
  /** ?model= param, sent in the info frame so a guest who copies the weights
   *  can host the same room. */
  param: string
  engine: DecodeEngine
  tokenizer: { decode: (ids: number[]) => string }
  /** Chat template + tokenize — injected so this module never imports the
   *  loader chain. */
  encode: (messages: ChatMsg[]) => number[]
  /** Join an EXISTING room (adds this device as another host). */
  existingRoom?: string | null
  /** This engine holds only layers [start,end) of a split model. */
  stageRange?: { start: number; end: number } | null
  /** Where a guest link points; share.ts passes its own pathname. */
  guestPath?: string
  /** Shared single-owner latch — REQUIRED when another driver (a local chat
   *  surface) runs the same engine. pump() holds it for each generation. */
  lock?: import('./engine-lock.js').EngineLock
  ui: RoomUI
}

export interface RoomHandle {
  roomId: string
  link: string
  close: () => void
}

export function hostRoom(opts: HostRoomOptions): RoomHandle {
  const { spec, brand, param, engine, tokenizer, encode, stageRange, ui } = opts
  const { base: signalBase, override: sigOverride } = signalEnv()

  // Joining an existing room keeps its id (that is what makes this device an
  // ADDITIONAL host rather than a second room nobody has the link to).
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  const roomId = opts.existingRoom
    ?? btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  // Carry a dev signaling override into the guest link, or the guest dials the
  // default relay and the room never forms.
  const link = `${location.origin}${opts.guestPath ?? '/share.html'}`
    + `${sigOverride ? `?sig=${encodeURIComponent(sigOverride)}` : ''}#${roomId}`

  const ws = new WebSocket(`${signalBase}/room/${roomId}?role=host`)
  const peers = new Map<string, { pc: RTCPeerConnection; dc: RTCDataChannel | null }>()
  const pipelines = new Map<string, RTCDataChannel>()

  // Serialized generation: the engine is single-stream. FIFO of pending
  // requests across all guests; queue positions are reported honestly.
  const queue: { guest: string; req: ChatReq }[] = []
  let generating = false
  const stopFlags = new Map<string, number>()   // guest -> request id to stop

  const send = (guest: string, msg: HostMsg): void => {
    const dc = peers.get(guest)?.dc
    if (dc?.readyState === 'open') dc.send(JSON.stringify(msg))
  }

  // ── the rest of a split model ────────────────────────────────────────────
  // Helper peers each announce a layer range and are assembled into an
  // ordered CHAIN. Until the chain reaches the last layer the host has a
  // model it cannot finish, and says so rather than generating from part of
  // a network. Hub-and-spoke on purpose — see share.ts history. The rules and
  // the split token loop are room-chain.ts; this is the wiring to the wires.
  const chain = makeRoomChain({
    spec,
    stageRange: stageRange ?? null,
    engine,
    hasPipe: (guest) => pipelines.has(guest),
    connect: (guest) => {
      const pipe = pipelines.get(guest)
      return pipe ? makeStageClient(pipe) : null
    },
    connected: (guest) => peers.has(guest),
    send,
    row: ui.row,
    announce: () => announce(),
    onPaired: (complete) => ui.onPaired?.(complete),
  })

  async function pump(): Promise<void> {
    if (generating) return
    const next = queue.shift()
    if (!next) return
    generating = true
    const { guest, req } = next
    // The preview slice and the lock acquisition are inside the try; the
    // `generating` flag above is reset by the finally either way. It was not:
    // the preview sat outside, so a guest sending
    // {"type":"chat","id":1,"messages":[{}]} threw on `.content.slice` and
    // wedged the host permanently — every later request from every guest
    // queued forever, with `void pump()` swallowing the rejection and no error
    // reaching anyone. onGuestMessage only validates Array.isArray(messages).
    let st: (s: string) => void = () => {}
    let tok: number | undefined
    try {
      const last = req.messages.at(-1)?.content
      const preview = typeof last === 'string' ? last.slice(0, 80) : ''
      st = ui.row(guest, preview)
      // One engine, one owner: if the host's own chat is mid-reply, this guest
      // waits — visibly — instead of interleaving into the same KV cache.
      if (opts.lock?.held()) st('waiting — the host is chatting…')
      tok = await opts.lock?.acquire()
      const promptIds = encode(req.messages)
      // The per-reply cap is the KV room the prompt leaves behind, not a
      // magic constant — same rule as chat.ts.
      const budget = spec.maxContext - promptIds.length
      if (budget < 16) throw new Error(`prompt is ${promptIds.length} tokens — over this model's ${spec.maxContext}-token context`)
      const allIds: number[] = []
      const t0 = performance.now()
      st('generating…')
      ui.onBusy?.(true)
      const onTok = (id: number) => {
        allIds.push(id)
        ui.onToken?.()
        send(guest, { type: 'text', id: req.id, full: tokenizer.decode(allIds) })
      }
      const stop = () => stopFlags.get(guest) === req.id || !peers.has(guest)
      if (stageRange) await chain.generate(promptIds, budget, onTok, stop)
      else await engine.generatePipelined(promptIds, budget, onTok, stop)
      const secs = (performance.now() - t0) / 1000
      const tps = allIds.length / Math.max(secs, 0.001)
      send(guest, { type: 'done', id: req.id, tokens: allIds.length, tps })
      // The hop SUM is what a token actually waited on — the number that
      // decides whether a long chain is usable at all.
      const hopTotal = chain.stages.reduce((a, s) => a + s.meanHopMs(), 0)
      st(`${allIds.length} tok · ${tps.toFixed(1)} tok/s`
        + (chain.stages.length ? ` · ${hopTotal.toFixed(1)} ms/hop across ${chain.stages.length} stage${chain.stages.length === 1 ? '' : 's'}` : ''))
    } catch (e) {
      send(guest, { type: 'error', id: req.id, message: e instanceof Error ? e.message : String(e) })
      st('error')
    } finally {
      // ONLY if we actually took it. `tok` is undefined when the throw happened
      // above the acquire (ui.row can), and release(undefined) skips the token
      // check — so an unconditional release here hands the engine to a queued
      // waiter while the real holder still believes it holds. That is the exact
      // two-holder interleave this whole change removes, reintroduced by the
      // change itself; caught by a review that ran it rather than read it.
      if (tok !== undefined) opts.lock?.release(tok)
      generating = false
      ui.onBusy?.(false)
      void pump()
    }
  }

  function onGuestMessage(guest: string, raw: string): void {
    let msg: GuestMsg
    try { msg = JSON.parse(raw) as GuestMsg } catch { return }
    if (msg.type === 'stop') { stopFlags.set(guest, msg.id); return }
    if (msg.type === 'stage-offer') { chain.offer(guest, msg); return }
    if (msg.type !== 'chat' || !Array.isArray(msg.messages)) return
    stopFlags.delete(guest)
    queue.push({ guest, req: msg })
    const pos = queue.length - (generating ? 0 : 1)
    if (pos > 0) send(guest, { type: 'busy', id: msg.id, pos })
    void pump()
  }

  // Membership has two sources — the signaller's periodic counts and this
  // host's own peer table — and they must not overwrite each other.
  let roomCounts = { hosts: 1, guests: 0 }
  function announce(): void {
    ui.onMembers?.({ ...roomCounts, connected: peers.size })
    ui.onChain?.(stageRange ? chain.describe() : '')
  }

  function makePeer(guest: string): void {
    // A reassigned guest (its previous host closed) arrives as a fresh
    // peer-joined here; drop any half-built connection under the same id.
    peers.get(guest)?.pc.close()
    const pc = new RTCPeerConnection(ICE)
    const entry = { pc, dc: null as RTCDataChannel | null }
    peers.set(guest, entry)
    const dc = pc.createDataChannel('chat', { ordered: true })
    entry.dc = dc
    // A SECOND channel carries weight replication. Separate because a 2 GB
    // transfer and a token stream must not share a queue: DataChannels are
    // head-of-line blocked, so pieces in flight would stall every reply.
    const weights = pc.createDataChannel('weights', { ordered: true })
    serveWeights(weights, spec, (m) => ui.row(guest, `weights — ${m}`))
    // Third channel: residuals to a peer holding the rest of a split model.
    // Its own queue again — a hand-off must never wait behind a token stream.
    const pipeline = pc.createDataChannel('pipeline', { ordered: true })
    pipeline.binaryType = 'arraybuffer'
    pipelines.set(guest, pipeline)
    dc.onopen = () => {
      send(guest, {
        type: 'info', name: brand.name, params: brand.params, rateLabel: brand.rateLabel,
        tag: quantTagFor(spec), param, specId: spec.id,
      })
      announce()
    }
    dc.onmessage = (e) => onGuestMessage(guest, String(e.data))
    pc.onicecandidate = (e) => {
      if (e.candidate) ws.send(JSON.stringify({ to: guest, type: 'ice', candidate: e.candidate }))
    }
    void pc.createOffer()
      .then(async (offer) => {
        await pc.setLocalDescription(offer)
        ws.send(JSON.stringify({ to: guest, type: 'offer', sdp: offer }))
      })
    announce()
  }

  ws.onmessage = (e) => {
    const msg = JSON.parse(String(e.data)) as {
      type: string; from?: string; hosts?: number; guests?: number
      sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit
    }
    if (msg.type === 'room') {
      roomCounts = { hosts: msg.hosts ?? 1, guests: msg.guests ?? 0 }
      announce()
      return
    }
    const guest = msg.from
    if (!guest) return
    if (msg.type === 'peer-joined') makePeer(guest)
    else if (msg.type === 'peer-left') {
      peers.get(guest)?.pc.close()
      peers.delete(guest)
      pipelines.delete(guest)
      if (stageRange && chain.holds(guest)) {
        ui.row(guest, 'a stage of the model left')
        chain.drop(guest)
      }
      announce()
    }
    else if (msg.type === 'answer' && msg.sdp) void peers.get(guest)?.pc.setRemoteDescription(msg.sdp)
    else if (msg.type === 'ice' && msg.candidate) void peers.get(guest)?.pc.addIceCandidate(msg.candidate)
  }
  const onUnload = (): void => ws.close()
  window.addEventListener('beforeunload', onUnload)
  announce()

  return {
    roomId,
    link,
    close() {
      window.removeEventListener('beforeunload', onUnload)
      ws.close()
      for (const [, p] of peers) p.pc.close()
      peers.clear()
      pipelines.clear()
    },
  }
}
