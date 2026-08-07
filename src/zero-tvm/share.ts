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
type GuestMsg = ChatReq | StopReq
type HostMsg =
  | { type: 'info'; name: string; params: string; rateLabel: string; tag: string; param: string; specId: string }
  | { type: 'text'; id: number; full: string }
  | { type: 'done'; id: number; tokens: number; tps: number }
  | { type: 'busy'; id: number; pos: number }
  | { type: 'error'; id: number; message: string }

const $ = (id: string) => document.getElementById(id) as HTMLElement

function roomIdFrom(hash: string): string | null {
  const id = hash.replace(/^#/, '')
  return /^[A-Za-z0-9_-]{16,64}$/.test(id) ? id : null
}

const room = roomIdFrom(location.hash)
if (room) void runGuest(room)
else void runHost()

// ============================================================
// HOST
// ============================================================

async function runHost(): Promise<void> {
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
  $('page-title').textContent = `Sharing ${brand.name}`

  // ── keep-awake: screen wake-lock + a silent audio track. The wake lock
  // stops the machine sleeping; the audio track is the standard exemption
  // from background-tab throttling (measured: a backgrounded host generated
  // at ~23 tok/s where the focused tab does ~65). Both are user-toggled and
  // honestly labeled — the browser shows its audio indicator on the tab.
  const awake = $('awake') as HTMLInputElement
  let wakeLock: WakeLockSentinel | null = null
  let audioCtx: AudioContext | null = null
  async function applyAwake(on: boolean): Promise<void> {
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
  awake.addEventListener('change', () => void applyAwake(awake.checked))
  document.addEventListener('visibilitychange', () => {
    // The UA releases wake locks on hide; re-acquire when we come back.
    if (awake.checked && document.visibilityState === 'visible') void applyAwake(true)
  })

  // Same engine composition as chat.ts — this is the throughput path, not the
  // scalar validation path.
  const boot = await loadingUi.bootEngine({
    spec,
    optionalFeatures: ['subgroups' as GPUFeatureName],
    probeSubgroups: true,
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = variantsMod.parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      const fused = !s.qkNorm && s.weightFormat !== 'mlx-safetensors'
      const kv = engineMod.allocKVPages(device, s)
      return engineMod.buildDecodeEngine(device, weights, kv, { variants: flags, fused, spec: s })
    },
  })
  if (!boot.ok) {
    $('loading-error').textContent = boot.reason
    return
  }
  const { engine, tokenizer } = boot
  const enc = (messages: ChatReq['messages']) => buildChatPromptFor(spec, messages, tokenizer)

  // ── room ──
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  const roomId = btoa(String.fromCharCode(...bytes)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  // Carry a dev signaling override into the guest link, or the guest dials the
  // default relay and the room never forms.
  const link = `${location.origin}${location.pathname}${SIG_OVERRIDE ? `?sig=${encodeURIComponent(SIG_OVERRIDE)}` : ''}#${roomId}`
  const linkInput = $('share-link') as HTMLInputElement
  linkInput.value = link
  $('copy-link').addEventListener('click', () => {
    void navigator.clipboard.writeText(link)
    $('copy-link').textContent = 'Copied'
    setTimeout(() => { $('copy-link').textContent = 'Copy' }, 1200)
  })

  const ws = new WebSocket(`${SIGNAL_BASE}/room/${roomId}?role=host`)
  const peers = new Map<string, { pc: RTCPeerConnection; dc: RTCDataChannel | null }>()

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
      await engine.generatePipelined(
        promptIds, budget,
        (id) => {
          allIds.push(id)
          send(guest, { type: 'text', id: req.id, full: tokenizer.decode(allIds) })
        },
        () => stopFlags.get(guest) === req.id || !peers.has(guest),
      )
      const secs = (performance.now() - t0) / 1000
      const tps = allIds.length / Math.max(secs, 0.001)
      send(guest, { type: 'done', id: req.id, tokens: allIds.length, tps })
      st.textContent = `${allIds.length} tok · ${tps.toFixed(1)} tok/s`
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
    if (msg.type !== 'chat' || !Array.isArray(msg.messages)) return
    stopFlags.delete(guest)
    queue.push({ guest, req: msg })
    const pos = queue.length - (generating ? 0 : 1)
    if (pos > 0) send(guest, { type: 'busy', id: msg.id, pos })
    void pump()
  }

  const param = new URLSearchParams(location.search).get('model') ?? ''

  function makePeer(guest: string): void {
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
    const msg = JSON.parse(String(e.data)) as { type: string; from?: string; sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit }
    const guest = msg.from
    if (!guest) return
    if (msg.type === 'peer-joined') makePeer(guest)
    else if (msg.type === 'peer-left') { peers.get(guest)?.pc.close(); peers.delete(guest) }
    else if (msg.type === 'answer' && msg.sdp) void peers.get(guest)?.pc.setRemoteDescription(msg.sdp)
    else if (msg.type === 'ice' && msg.candidate) void peers.get(guest)?.pc.addIceCandidate(msg.candidate)
  }
  window.addEventListener('beforeunload', () => ws.close())

  // e2e hooks
  ;(window as unknown as Record<string, unknown>).__shareLink = link
  ;(window as unknown as Record<string, unknown>).__shareReady = true
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
  const pc = new RTCPeerConnection(ICE)
  let dc: RTCDataChannel | null = null
  // The weights channel and the `info` frame race: the host opens both
  // channels at once, so info can land on 'chat' before 'weights' has even
  // been announced here. Awaiting a promise removes the race — the offer is
  // built when BOTH have arrived, in whichever order that happens.
  let weightsReady: (dc: RTCDataChannel) => void = () => {}
  const weightsChannel = new Promise<RTCDataChannel>((r) => { weightsReady = r })

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

  ws.onmessage = async (e) => {
    const msg = JSON.parse(String(e.data)) as { type: string; sdp?: RTCSessionDescriptionInit; candidate?: RTCIceCandidateInit }
    if (msg.type === 'no-host') {
      setBadge('No host', 'error')
      setStatus('Nobody is sharing a model in this room — the host tab is closed or the link expired.')
    } else if (msg.type === 'host-left') {
      setBadge('Host left', 'error')
      setStatus('The host closed the tab. The conversation stays here; reconnect with a fresh link.')
      inp.disabled = true
      sendBtn.disabled = true
    } else if (msg.type === 'offer' && msg.sdp) {
      await pc.setRemoteDescription(msg.sdp)
      const answer = await pc.createAnswer()
      await pc.setLocalDescription(answer)
      ws.send(JSON.stringify({ type: 'answer', sdp: answer }))
    } else if (msg.type === 'ice' && msg.candidate) {
      await pc.addIceCandidate(msg.candidate)
    }
  }
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
    dc = e.channel
    dc.onmessage = (ev) => onHostMsg(JSON.parse(String(ev.data)) as HostMsg)
    dc.onclose = () => setBadge('Disconnected', 'error')
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
          status.innerHTML = `Done — ${(res.bytes / 1e9).toFixed(2)} GB in ${secs}s. `
            + `<a href="/zero-tvm.html?model=${encodeURIComponent(param)}">Open the chat on this device →</a>`
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
