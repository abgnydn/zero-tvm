/**
 * SHARE — serve the model running in THIS tab to another device, browser to
 * browser, over a WebRTC data channel.
 *
 * Two modes, decided by the URL fragment:
 *   share.html?model=qwen36q3      HOST — boots the engine (same composition
 *                                  as chat.ts), opens a room, serves requests
 *   share.html#<roomId>            GUEST — the chat page's own conversational
 *                                  surface (chat-ui.ts + chat-ui.css) over a
 *                                  DataChannel; needs no WebGPU, downloads
 *                                  nothing, learns the model's identity over
 *                                  the channel
 *
 * Privacy model: the signaling worker (workers/share-signal) relays only
 * SDP/ICE JSON; prompts and tokens travel the DataChannel, which WebRTC
 * always DTLS-encrypts. With a direct P2P path nothing of the conversation
 * touches any server. The room id is 128 random bits carried in the link's
 * #fragment — browsers do not send fragments over HTTP, so the static host
 * never logs it.
 *
 * The engine is single-stream (batch = 1), so guest requests are SERIALIZED:
 * one generates, the rest hold a queue position and are told so. The host tab
 * shows every request as it arrives — your machine should not run someone's
 * prompt without you being able to see it.
 *
 * The wire protocol is TEXT in both directions (guest sends chat messages,
 * host renders the template and tokenizes) — guests never need tokenizer
 * files, and the host's decode loop is chat.ts's: re-decode the whole id
 * sequence each token, send the full text so far. Full-text frames are
 * O(n²) bytes per reply but a 512-token answer is still only ~1 MB across
 * a channel that does orders of magnitude more; dropping mid-token UTF-8
 * boundary bugs is worth far more than the bytes.
 */

import {
  setChatIdentity, quantTagFor, autoGrow, wireScrollFab,
  addUserMsg, addAiMsg, type AiMsgHandle,
} from './chat-ui.js'
import { specForParam } from './model-registry.js'
import { serveWeights, fetchInventory, pullWeights, type Inventory } from './peer-weights.js'
import { serveStage, makeStageClient } from './pipeline-peer.js'

// Signaling endpoint. Dev: `npx wrangler dev --port 8787` in
// workers/share-signal (what scripts/share-e2e.mjs spawns). (Same import.meta
// cast as weight-loader.ts — the repo has no vite/client types.)
const DEV = !!(import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV
const PROD_SIGNAL = 'wss://zero-tvm-share-signal.abgunaydin94.workers.dev'  // deployed 2026-08-06
/** `?sig=<port|ws-url>` overrides the relay — DEV ONLY, so a shared link can
 *  never point a guest's signaling at a third party in production. Two test
 *  drivers run their own wrangler on different ports concurrently. */
const SIG_OVERRIDE = DEV ? new URLSearchParams(location.search).get('sig') : null
const SIGNAL_BASE = SIG_OVERRIDE
  ? (/^wss?:\/\//.test(SIG_OVERRIDE) ? SIG_OVERRIDE : `ws://localhost:${SIG_OVERRIDE}`)
  : DEV ? 'ws://localhost:8787' : PROD_SIGNAL

const ICE: RTCConfiguration = {
  // STUN only, deliberately: same-network and home-NAT paths connect
  // directly. Hostile (corporate) NATs need a TURN relay — a later problem,
  // and a separate consent: TURN routes ciphertext through a third party.
  iceServers: [{ urls: ['stun:stun.cloudflare.com:3478', 'stun:stun.l.google.com:19302'] }],
}

// ── wire protocol ─────────────────────────────────────────────
interface ChatReq { type: 'chat'; id: number; messages: { role: 'system' | 'user' | 'assistant'; content: string }[] }
interface StopReq { type: 'stop'; id: number }
/** A peer that holds the REST of a split model, offering to run it. */
interface StageOffer { type: 'stage-offer'; start: number; end: number; specId: string }
type GuestMsg = ChatReq | StopReq | StageOffer
type HostMsg =
  | { type: 'info'; name: string; params: string; rateLabel: string; tag: string; param: string; specId: string }
  | { type: 'text'; id: number; full: string }
  | { type: 'done'; id: number; tokens: number; tps: number }
  | { type: 'busy'; id: number; pos: number }
  | { type: 'error'; id: number; message: string }
  | { type: 'stage-accept'; start: number; end: number }
  | { type: 'stage-reject'; message: string }

const $ = (id: string) => document.getElementById(id) as HTMLElement

function roomIdFrom(hash: string): string | null {
  const id = hash.replace(/^#/, '')
  return /^[A-Za-z0-9_-]{16,64}$/.test(id) ? id : null
}

// URL grammar, three cases:
//   ?model=X              host a NEW room
//   #<room>               join as a guest (the link you hand out)
//   ?model=X#<room>       serve an EXISTING room from this device too
// The third is what turns a room into a swarm: a guest that copied the
// weights adds ?model= to the link it already has and starts serving.
/** `?layers=0-20` — this device holds that slice of the model, nothing else. */
function stageRangeFrom(search: string): { start: number; end: number } | null {
  const m = /^(\d+)-(\d+)$/.exec(new URLSearchParams(search).get('layers') ?? '')
  return m ? { start: Number(m[1]), end: Number(m[2]) } : null
}

const room = roomIdFrom(location.hash)
const wantsToHost = new URLSearchParams(location.search).has('model')
const stage = stageRangeFrom(location.search)
// A stage that does not start the model cannot serve chat — it has no
// embedding and no tokenizer role — so it JOINS someone else's room and
// offers its layers there. Everything else follows the earlier grammar.
if (room && stage && stage.start > 0) void runHelper(room, stage)
else if (room && !wantsToHost) void runGuest(room)
else void runHost(room, stage)

/**
 * Keep-awake toggle — screen wake-lock plus a silent audio track. The wake
 * lock stops the machine sleeping; the audio track is the standard exemption
 * from background-tab throttling (measured: a backgrounded host generated at
 * ~23 tok/s where the focused tab does ~65, and served weights at ~1 MB/s).
 * Honestly labeled: the browser shows its audio indicator on the tab.
 *
 * Both serving roles need it — a helper stage especially, since it is a
 * background tab for its whole life.
 */
function wireKeepAwake(): void {
  const awake = $('awake') as HTMLInputElement | null
  if (!awake) return
  let wakeLock: WakeLockSentinel | null = null
  let audioCtx: AudioContext | null = null
  const apply = async (on: boolean): Promise<void> => {
    if (on) {
      try { wakeLock = await navigator.wakeLock.request('screen') } catch { /* unsupported / not visible */ }
      if (!audioCtx) {
        audioCtx = new AudioContext()
        const osc = audioCtx.createOscillator()
        const gain = audioCtx.createGain()
        gain.gain.value = 0.0001   // inaudible, but "playing" as far as the scheduler cares
        osc.connect(gain).connect(audioCtx.destination)
        osc.start()
      }
      void audioCtx.resume()
    } else {
      void wakeLock?.release().catch(() => {})
      wakeLock = null
      void audioCtx?.suspend()
    }
  }
  awake.addEventListener('change', () => void apply(awake.checked))
  document.addEventListener('visibilitychange', () => {
    // The UA releases wake locks on hide; re-acquire when we come back.
    if (awake.checked && document.visibilityState === 'visible') void apply(true)
  })
}

// ============================================================
// HOST
// ============================================================

async function runHost(existingRoom: string | null, stageRange: { start: number; end: number } | null): Promise<void> {
  $('host-view').classList.remove('hidden')
  // Everything GPU-touching is imported HERE, not at module scope — the guest
  // path must run on machines without WebGPU, and weight-loader.ts reads
  // GPUBufferUsage the moment it is imported.
  const [{ specFromSearch, modelBranding, buildChatPromptFor }, loadingUi, variantsMod, engineMod] =
    await Promise.all([
      import('./model-select.js'),
      import('./loading-ui.js'),
      import('./variants.js'),
      import('./engine-core.js'),
    ])

  const spec = specFromSearch(location.search)
  const brand = modelBranding(spec)
  $('page-title').textContent = stageRange
    ? `Sharing ${brand.name} — layers ${stageRange.start}-${stageRange.end} here`
    : `Sharing ${brand.name}`
  if (stageRange && stageRange.start !== 0) {
    throw new Error(`share: a hosting stage must start at layer 0 (got ${stageRange.start}); later stages join a room as helpers`)
  }

  wireKeepAwake()

  // Same engine composition as chat.ts — this is the throughput path, not the
  // scalar validation path.
  const boot = await loadingUi.bootEngine({
    spec,
    optionalFeatures: ['subgroups' as GPUFeatureName],
    probeSubgroups: true,
    layerRange: stageRange ?? undefined,
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = variantsMod.parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      const fused = !s.qkNorm && s.weightFormat !== 'mlx-safetensors'
      const kv = engineMod.allocKVPages(device, s)
      return engineMod.buildDecodeEngine(device, weights, kv, {
        variants: flags, fused, spec: s, layerRange: stageRange ?? undefined,
      })
    },
    // A stage cannot run the default warmup (forwardLogits needs the whole
    // model); one pipelineStep at position 0 JITs the same shaders, and the
    // real prefill overwrites position 0 anyway.
    warmup: stageRange ? async (e) => { await e.pipelineStep({ tokenId: 1 }, 0) } : undefined,
  })
  if (!boot.ok) {
    $('loading-error').textContent = boot.reason
    return
  }
  const { engine, tokenizer } = boot
  const enc = (messages: ChatReq['messages']) => buildChatPromptFor(spec, messages, tokenizer)

  // ── room ──
  // Joining an existing room keeps its id (that is what makes this device an
  // ADDITIONAL host rather than a second room nobody has the link to).
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  const roomId = existingRoom
    ?? btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  // Carry a dev signaling override into the guest link, or the guest dials the
  // default relay and the room never forms.
  const link = `${location.origin}${location.pathname}${SIG_OVERRIDE ? `?sig=${encodeURIComponent(SIG_OVERRIDE)}` : ''}#${roomId}`
  if (existingRoom) $('page-title').textContent = `Serving ${brand.name} in a shared room`
  const linkInput = $('share-link') as HTMLInputElement
  linkInput.value = link
  $('copy-link').addEventListener('click', () => {
    void navigator.clipboard.writeText(link)
    $('copy-link').textContent = 'Copied'
    setTimeout(() => { $('copy-link').textContent = 'Copy' }, 1200)
  })

  const ws = new WebSocket(`${SIGNAL_BASE}/room/${roomId}?role=host`)
  const peers = new Map<string, { pc: RTCPeerConnection; dc: RTCDataChannel | null }>()
  const pipelines = new Map<string, RTCDataChannel>()

  // Serialized generation: the engine is single-stream. FIFO of pending
  // requests across all guests; queue positions are reported honestly.
  const queue: { guest: string; req: ChatReq }[] = []
  let generating = false
  const stopFlags = new Map<string, number>()   // guest -> request id to stop

  const logRow = (guest: string, text: string): HTMLElement => {
    $('req-empty')?.remove()
    const li = document.createElement('li')
    li.innerHTML = `<span class="who">${guest}</span> · <span class="body"></span> <span class="st"></span>`
    ;(li.querySelector('.body') as HTMLElement).textContent = text
    $('req-log').prepend(li)
    return li.querySelector('.st') as HTMLElement
  }

  const send = (guest: string, msg: HostMsg): void => {
    const dc = peers.get(guest)?.dc
    if (dc?.readyState === 'open') dc.send(JSON.stringify(msg))
  }

  // ── the other half of a split model ───────────────────────────────────────
  // Set when a helper peer offers exactly the layers this stage lacks. Until
  // then a split host has a model it cannot finish, and says so rather than
  // generating from half a network.
  let downstream: { guest: string; step: (pos: number, residual: ArrayBuffer) => Promise<number>; meanHopMs: () => number } | null = null

  /**
   * The token loop for a SPLIT model — the whole-model generatePipelined
   * cannot run here, because this engine holds only the first layers.
   *
   * Per token: this stage's layers, then a round trip to the peer holding the
   * rest, which returns the argmax. Prefill runs the same way, one position at
   * a time (chunked prefill is a whole-model optimisation).
   */
  async function generateSplit(
    promptIds: number[], budget: number,
    onToken: (id: number) => void, shouldStop: () => boolean,
  ): Promise<number[]> {
    const down = downstream
    if (!down) throw new Error('the other half of this model is not connected')
    const stepBoth = async (tokenId: number, pos: number): Promise<number> => {
      const mid = await engine.pipelineStep({ tokenId }, pos)
      if (!('residual' in mid)) throw new Error('this stage ended the model — nothing to hand on')
      return down.step(pos, mid.residual)
    }
    let tok = 0
    for (let i = 0; i < promptIds.length; i++) tok = await stepBoth(promptIds[i], i)
    const out: number[] = []
    for (let n = 0; n < budget; n++) {
      // Stop ids are consumed, never shown — same contract as generate().
      if (spec.stops.includes(tok) || shouldStop()) break
      out.push(tok)
      onToken(tok)
      tok = await stepBoth(tok, promptIds.length + n)
    }
    return out
  }

  async function pump(): Promise<void> {
    if (generating) return
    const next = queue.shift()
    if (!next) return
    generating = true
    const { guest, req } = next
    const preview = req.messages.at(-1)?.content.slice(0, 80) ?? ''
    const st = logRow(guest, preview)
    try {
      const promptIds = enc(req.messages)
      // The per-reply cap is the KV room the prompt leaves behind, not a magic
      // constant — same rule as chat.ts. (v1 shipped a min(1024, …) here and a
      // guest's long refactor request was cut mid-function at exactly 1024.)
      const budget = spec.maxContext - promptIds.length
      if (budget < 16) throw new Error(`prompt is ${promptIds.length} tokens — over this model's ${spec.maxContext}-token context`)
      const allIds: number[] = []
      const t0 = performance.now()
      st.textContent = 'generating…'
      const onTok = (id: number) => {
        allIds.push(id)
        send(guest, { type: 'text', id: req.id, full: tokenizer.decode(allIds) })
      }
      const stop = () => stopFlags.get(guest) === req.id || !peers.has(guest)
      if (stageRange) await generateSplit(promptIds, budget, onTok, stop)
      else await engine.generatePipelined(promptIds, budget, onTok, stop)
      const secs = (performance.now() - t0) / 1000
      const tps = allIds.length / Math.max(secs, 0.001)
      send(guest, { type: 'done', id: req.id, tokens: allIds.length, tps })
      st.textContent = `${allIds.length} tok · ${tps.toFixed(1)} tok/s`
        + (downstream ? ` · ${downstream.meanHopMs().toFixed(1)} ms/hop to the other stage` : '')
    } catch (e) {
      send(guest, { type: 'error', id: req.id, message: e instanceof Error ? e.message : String(e) })
      st.textContent = 'error'
    } finally {
      generating = false
      void pump()
    }
  }

  function onGuestMessage(guest: string, raw: string): void {
    let msg: GuestMsg
    try { msg = JSON.parse(raw) as GuestMsg } catch { return }
    if (msg.type === 'stop') { stopFlags.set(guest, msg.id); return }
    if (msg.type === 'stage-offer') { acceptStage(guest, msg); return }
    if (msg.type !== 'chat' || !Array.isArray(msg.messages)) return
    stopFlags.delete(guest)
    queue.push({ guest, req: msg })
    const pos = queue.length - (generating ? 0 : 1)
    if (pos > 0) send(guest, { type: 'busy', id: msg.id, pos })
    void pump()
  }

  const param = new URLSearchParams(location.search).get('model') ?? ''

  /**
   * A peer says it holds the rest of the model. Accept only if it is EXACTLY
   * the complement of this stage on the SAME spec — a mismatched range would
   * produce fluent nonsense, which is worse than refusing.
   */
  function acceptStage(guest: string, offer: StageOffer): void {
    const pipe = pipelines.get(guest)
    const ok = !!stageRange && !!pipe && offer.specId === spec.id
      && offer.start === stageRange.end && offer.end === spec.layers
    if (!ok) {
      const why = !stageRange ? 'this host runs the whole model'
        : offer.specId !== spec.id ? `different model (${offer.specId} vs ${spec.id})`
        : !pipe ? 'the pipeline channel is not open'
        : `layers ${offer.start}-${offer.end} do not continue ${stageRange.start}-${stageRange.end}`
      send(guest, { type: 'stage-reject', message: why })
      logRow(guest, `stage offer refused — ${why}`)
      return
    }
    const client = makeStageClient(pipe)
    downstream = { guest, step: client.step, meanHopMs: client.meanHopMs }
    send(guest, { type: 'stage-accept', start: offer.start, end: offer.end })
    $('room-stats').textContent =
      `split model — layers ${stageRange!.start}-${stageRange!.end} here, ${offer.start}-${offer.end} on a peer`
    logRow(guest, `serving layers ${offer.start}-${offer.end} — the model is complete`)
    ;(window as unknown as Record<string, unknown>).__stagePaired = true
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
    serveWeights(weights, spec, (m) => logRow(guest, `weights — ${m}`))
    // Third channel: residuals to a peer holding the rest of a split model.
    // Its own queue again — a hand-off must never wait behind a token stream.
    const pipeline = pc.createDataChannel('pipeline', { ordered: true })
    pipeline.binaryType = 'arraybuffer'
    pipelines.set(guest, pipeline)
    dc.onopen = () => send(guest, {
      type: 'info', name: brand.name, params: brand.params, rateLabel: brand.rateLabel,
      tag: quantTagFor(spec), param, specId: spec.id,
    })
    dc.onmessage = (e) => onGuestMessage(guest, String(e.data))
    pc.onicecandidate = (e) => {
      if (e.candidate) ws.send(JSON.stringify({ to: guest, type: 'ice', candidate: e.candidate }))
    }
    void pc.createOffer()
      .then(async (offer) => {
        await pc.setLocalDescription(offer)
        ws.send(JSON.stringify({ to: guest, type: 'offer', sdp: offer }))
      })
  }

  ws.onmessage = (e) => {
    const msg = JSON.parse(String(e.data)) as {
      type: string; from?: string; hosts?: number; guests?: number
      sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit
    }
    if (msg.type === 'room') {
      const h = msg.hosts ?? 1, g = msg.guests ?? 0
      $('room-stats').textContent =
        `${h} ${h === 1 ? 'machine' : 'machines'} serving · ${g} ${g === 1 ? 'guest' : 'guests'} connected`
      return
    }
    const guest = msg.from
    if (!guest) return
    if (msg.type === 'peer-joined') makePeer(guest)
    else if (msg.type === 'peer-left') {
      peers.get(guest)?.pc.close()
      peers.delete(guest)
      pipelines.delete(guest)
      if (downstream?.guest === guest) {
        downstream = null
        $('room-stats').textContent = `split model — waiting for layers ${stageRange!.end}-${spec.layers} again`
        logRow(guest, 'the other half of the model left')
      }
    }
    else if (msg.type === 'answer' && msg.sdp) void peers.get(guest)?.pc.setRemoteDescription(msg.sdp)
    else if (msg.type === 'ice' && msg.candidate) void peers.get(guest)?.pc.addIceCandidate(msg.candidate)
  }
  window.addEventListener('beforeunload', () => ws.close())

  // e2e hooks
  ;(window as unknown as Record<string, unknown>).__shareLink = link
  ;(window as unknown as Record<string, unknown>).__shareReady = true
}

// ============================================================
// HELPER — this device holds the END of a split model
// ============================================================

/**
 * Join someone else's room and offer the layers they lack.
 *
 * Signalling-wise this is a GUEST: it takes the same offer/answer path and the
 * same three channels, then says "I hold layers k..N" on the chat channel
 * instead of asking a question. Reusing the guest role rather than teaching
 * the relay a third one is what keeps this feature small — the room only ever
 * routes, and what a peer DOES with its channels is between the peers.
 */
async function runHelper(roomId: string, range: { start: number; end: number }): Promise<void> {
  $('host-view').classList.remove('hidden')
  // A helper hands out no link of its own — it joined someone else's room.
  // (The keep-awake card is deliberately a SEPARATE section, so removing this
  // one does not take the toggle with it.)
  $('share-link').closest('section')?.remove()
  wireKeepAwake()
  const [{ specFromSearch, modelBranding }, loadingUi, variantsMod, engineMod] = await Promise.all([
    import('./model-select.js'),
    import('./loading-ui.js'),
    import('./variants.js'),
    import('./engine-core.js'),
  ])
  const spec = specFromSearch(location.search)
  const brand = modelBranding(spec)
  $('page-title').textContent = `Serving ${brand.name} layers ${range.start}-${range.end}`
  const stats = $('room-stats')
  stats.textContent = `loading layers ${range.start}-${range.end} of ${spec.layers}…`

  const boot = await loadingUi.bootEngine({
    spec,
    optionalFeatures: ['subgroups' as GPUFeatureName],
    probeSubgroups: true,
    layerRange: range,
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = variantsMod.parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      return engineMod.buildDecodeEngine(device, weights, engineMod.allocKVPages(device, s), {
        variants: flags, fused: false, spec: s, layerRange: range,
      })
    },
    // A residual of zeros JITs this stage's shaders; the real prefill starts
    // at position 0 again and overwrites it.
    warmup: async (e) => { await e.pipelineStep({ residual: new ArrayBuffer(spec.d * 2) }, 0) },
  })
  if (!boot.ok) { $('loading-error').textContent = boot.reason; return }

  const ws = new WebSocket(`${SIGNAL_BASE}/room/${roomId}?role=guest`)
  const pc = new RTCPeerConnection(ICE)
  let chat: RTCDataChannel | null = null
  pc.onicecandidate = (e) => {
    if (e.candidate) ws.send(JSON.stringify({ type: 'ice', candidate: e.candidate }))
  }
  pc.ondatachannel = (e) => {
    if (e.channel.label === 'pipeline') {
      serveStage(e.channel, boot.engine, (m) => { stats.textContent = `layers ${range.start}-${range.end} · ${m}` })
      return
    }
    if (e.channel.label !== 'chat') return
    chat = e.channel
    chat.onopen = () => chat!.send(JSON.stringify({
      type: 'stage-offer', start: range.start, end: range.end, specId: spec.id,
    } satisfies StageOffer))
    chat.onmessage = (ev) => {
      const msg = JSON.parse(String(ev.data)) as HostMsg
      if (msg.type === 'stage-accept') {
        stats.textContent = `paired — this device runs layers ${range.start}-${range.end}, the host runs the rest`
        ;(window as unknown as Record<string, unknown>).__helperPaired = true
      } else if (msg.type === 'stage-reject') {
        stats.textContent = `the host refused this stage: ${msg.message}`
      }
    }
  }
  ws.onmessage = async (e) => {
    const msg = JSON.parse(String(e.data)) as { type: string; sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit }
    if (msg.type === 'offer' && msg.sdp) {
      await pc.setRemoteDescription(msg.sdp)
      const answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      ws.send(JSON.stringify({ type: 'answer', sdp: answer }))
    } else if (msg.type === 'ice' && msg.candidate) {
      await pc.addIceCandidate(msg.candidate)
    } else if (msg.type === 'no-host' || msg.type === 'host-left') {
      stats.textContent = 'no host in this room to pair with — open the room link on the machine holding the first layers'
    }
  }
  ;(window as unknown as Record<string, unknown>).__helperReady = true
}

// ============================================================
// GUEST — chat-ui.ts's surface over a DataChannel
// ============================================================

async function runGuest(roomId: string): Promise<void> {
  $('guest-view').classList.remove('hidden')
  const setStatus = (t: string) => { $('guest-status').textContent = t }
  const setBadge = (t: string, cls: 'loading' | 'ready' | 'error') => {
    $('guest-badge').className = `badge ${cls}`
    $('guest-badge-text').textContent = t
  }
  setBadge('Connecting', 'loading')
  wireScrollFab()

  const ws = new WebSocket(`${SIGNAL_BASE}/room/${roomId}?role=guest`)
  // The peer connection is REBUILT whenever the room reassigns this guest to
  // another host (the previous one closed its tab). Everything above it — the
  // conversation, the rendered messages — survives, because history lives here
  // and the host is stateless between requests.
  let pc = new RTCPeerConnection(ICE)
  let dc: RTCDataChannel | null = null
  // The weights channel and the `info` frame race: the host opens both
  // channels at once, so info can land on 'chat' before 'weights' has even
  // been announced here. Awaiting a promise removes the race — the offer is
  // built when BOTH have arrived, in whichever order that happens.
  let weightsReady: (dc: RTCDataChannel) => void = () => {}
  let weightsChannel = new Promise<RTCDataChannel>((r) => { weightsReady = r })

  const history: ChatReq['messages'] = []
  let reqId = 0
  let live: AiMsgHandle | null = null
  let liveFull = ''

  const inp = $('inp') as HTMLTextAreaElement
  const sendBtn = $('btn') as HTMLButtonElement
  const stopBtn = $('stop-btn') as HTMLButtonElement
  const setGenerating = (on: boolean): void => {
    sendBtn.hidden = on
    stopBtn.hidden = !on
    sendBtn.disabled = on
  }

  const settle = (): void => {
    live = null
    setGenerating(false)
  }

  const onHostMsg = (msg: HostMsg): void => {
    if (msg.type === 'info') {
      setChatIdentity(msg.name, msg.tag)
      $('guest-model').textContent = `${msg.name} — remote`
      setStatus(`${msg.params}${msg.rateLabel ? ` · ${msg.rateLabel}` : ''} · runs on the host machine; this page holds only the conversation.`)
      setBadge('Ready', 'ready')
      inp.disabled = false
      inp.placeholder = 'Message the remote model…'
      sendBtn.disabled = false
      void offerLocalCopy(msg.param, msg.specId)
    } else if (msg.type === 'text') {
      liveFull = msg.full
      live?.render(liveFull)
    } else if (msg.type === 'done') {
      if (live) {
        live.finish({ fullText: liveFull, tokens: msg.tokens, tokPerS: msg.tps })
        history.push({ role: 'assistant', content: liveFull })
      }
      settle()
    } else if (msg.type === 'busy') {
      setStatus(`Host is generating for someone else — queue position ${msg.pos}.`)
    } else if (msg.type === 'error') {
      if (live) live.body.textContent = `⚠ ${msg.message}`
      settle()
    }
  }

  /** Wire a freshly-created RTCPeerConnection to this page. */
  function wirePeer(): void {
    pc.onicecandidate = (e) => {
      if (e.candidate) ws.send(JSON.stringify({ type: 'ice', candidate: e.candidate }))
    }
    pc.ondatachannel = (e) => {
      if (e.channel.label === 'weights') {
        const ch = e.channel
        ch.binaryType = 'arraybuffer'
        if (ch.readyState === 'open') weightsReady(ch)
        else ch.addEventListener('open', () => weightsReady(ch), { once: true })
        return
      }
      // Match the CHAT channel by name. The host opens three (chat, weights,
      // pipeline) and a plain guest uses one; taking whichever arrived last
      // sent this page's questions down the pipeline channel, where nothing
      // was listening — the request simply vanished.
      if (e.channel.label !== 'chat') return
      dc = e.channel
      dc.onmessage = (ev) => onHostMsg(JSON.parse(String(ev.data)) as HostMsg)
      // Only a channel that is still the CURRENT one may report a disconnect;
      // a reassignment closes the old channel while the new one is opening.
      const closing = dc
      dc.onclose = () => { if (dc === closing) setBadge('Disconnected', 'error') }
    }
  }
  wirePeer()

  ws.onmessage = async (e) => {
    const msg = JSON.parse(String(e.data)) as {
      type: string; hosts?: number; guests?: number
      sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit
    }
    if (msg.type === 'no-host') {
      setBadge('No host', 'error')
      setStatus('Nobody is serving a model in this room — every host tab is closed, or the link expired.')
    } else if (msg.type === 'host-left') {
      setBadge('Host left', 'error')
      setStatus('Every machine serving this room went away. The conversation stays here; it resumes if one comes back.')
      inp.disabled = true
      sendBtn.disabled = true
    } else if (msg.type === 'host-changed') {
      // Another machine in the room took this guest over. Tear the old
      // connection down and wait for the new host's offer — the conversation
      // and everything on screen are untouched.
      setBadge('Switching host', 'loading')
      setStatus('The machine serving you went away; another one in the room is taking over…')
      pc.close()
      pc = new RTCPeerConnection(ICE)
      dc = null
      weightsChannel = new Promise<RTCDataChannel>((r) => { weightsReady = r })
      wirePeer()
      settle()
      inp.disabled = true
      sendBtn.disabled = true
    } else if (msg.type === 'room') {
      const h = msg.hosts ?? 0
      $('room-count').textContent = h > 1 ? `${h} machines serving` : ''
    } else if (msg.type === 'offer' && msg.sdp) {
      await pc.setRemoteDescription(msg.sdp)
      const answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      ws.send(JSON.stringify({ type: 'answer', sdp: answer }))
    } else if (msg.type === 'ice' && msg.candidate) {
      await pc.addIceCandidate(msg.candidate)
    }
  }

  /**
   * Offer to copy the model's weights from the host to THIS device, so it can
   * run locally afterwards instead of chatting through the host — the second
   * machine never re-downloads gigabytes the first one already has.
   *
   * The spec is resolved from the registry by `param` and cross-checked
   * against the host's spec id: the OPFS directory this writes into is chosen
   * LOCALLY, never from a string the host sent.
   */
  async function offerLocalCopy(param: string, specId: string): Promise<void> {
    const panel = $('local-copy')
    const spec = specForParam(param)
    if (spec.id !== specId) return   // this build doesn't know the host's model — offer nothing
    const weightsDc = await Promise.race([
      weightsChannel,
      new Promise<null>((r) => setTimeout(() => r(null), 20_000)),
    ])
    if (!weightsDc) return           // host build predates weight sharing
    const status = $('lc-status')
    const btn = $('lc-btn') as HTMLButtonElement
    let inv: Inventory
    try {
      inv = await fetchInventory(weightsDc)
    } catch {
      return   // host cannot serve weights (nothing cached) — stay quiet
    }
    if (!inv.files) return
    const gb = (inv.bytes / 1e9).toFixed(2)
    panel.classList.remove('hidden')
    status.textContent = `The host has ${gb} GB cached. Copying it here lets this device run ${spec.id} on its own GPU.`
    btn.textContent = `Copy ${gb} GB to this device`
    btn.addEventListener('click', () => {
      btn.disabled = true
      const t = performance.now()
      void pullWeights(weightsDc, spec, (p) => {
        const pct = p.bytesTotal ? Math.round((p.bytesDone / p.bytesTotal) * 100) : 0
        status.textContent = `${pct}% · ${(p.bytesDone / 1e9).toFixed(2)}/${gb} GB · `
          + `${p.filesDone}/${p.filesTotal} files · ${(p.rate / 1e6).toFixed(0)} MB/s`
      }, inv)
        .then((res) => {
          const secs = ((performance.now() - t) / 1000).toFixed(0)
          const p = encodeURIComponent(param)
          const sig = SIG_OVERRIDE ? `&sig=${encodeURIComponent(SIG_OVERRIDE)}` : ''
          status.innerHTML = `Done — ${(res.bytes / 1e9).toFixed(2)} GB in ${secs}s. `
            + `<a href="/zero-tvm.html?model=${p}">Open the chat on this device →</a><br>`
            // The swarm-forming link: this device has the weights now, so it
            // can serve the SAME room instead of only consuming it.
            + `<a href="/share.html?model=${p}${sig}#${roomId}">…or serve this room from here too →</a>`
          btn.remove()
          ;(window as unknown as Record<string, unknown>).__pullDone = res
        })
        .catch((err: Error) => {
          status.textContent = `Copy failed: ${err.message}`
          btn.disabled = false
        })
    })
  }

  const submit = (): void => {
    const text = inp.value.trim()
    if (!text || live || !dc || dc.readyState !== 'open') return
    inp.value = ''
    autoGrow(inp)
    history.push({ role: 'user', content: text })
    addUserMsg(text)
    live = addAiMsg()
    live.showThinking()
    liveFull = ''
    setGenerating(true)
    dc.send(JSON.stringify({ type: 'chat', id: ++reqId, messages: [...history] } satisfies ChatReq))
  }
  ;($('composer') as HTMLFormElement).addEventListener('submit', (e) => { e.preventDefault(); submit() })
  stopBtn.addEventListener('click', () => {
    if (dc?.readyState === 'open') dc.send(JSON.stringify({ type: 'stop', id: reqId } satisfies StopReq))
  })
  inp.addEventListener('input', () => autoGrow(inp))
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() }
  })
}
