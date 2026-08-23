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
  setChatIdentity, autoGrow, wireScrollFab,
  addUserMsg, addAiMsg, type AiMsgHandle,
} from './chat-ui.js'
import { specForParam } from './model-registry.js'
import { ENGINE_GPU_FEATURES } from './variants.js'
import type { ModelSpec } from '../compiler/model-spec.js'
import { fetchInventory, pullWeights, type Inventory } from './peer-weights.js'
import { serveStage } from './pipeline-peer.js'
// The host loop, the wire protocol, and the signaling constants live in
// room-host.ts — shared with the landing entrance's in-place room
// (landing-room.ts), so the two hosting surfaces cannot drift.
import {
  hostRoom, signalEnv, ICE,
  type HostMsg, type ChatReq, type StopReq, type StageOffer,
} from './room-host.js'

const { base: SIGNAL_BASE, override: SIG_OVERRIDE } = signalEnv()

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

/**
 * Ask before spending gigabytes.
 *
 * This page booted the engine the moment it opened, so a visitor who clicked
 * "Rooms" in the site nav was already ~2 GB into a download before reading a
 * word about it — no button pressed, no size shown. The chat page has gated
 * this since it shipped; the host path never did.
 *
 * Resolves immediately when the weights are already on this device: there is
 * nothing to consent to then, and a returning host should not be asked twice.
 * cache-probe pulls in the loaders, which read GPUBufferUsage at module scope,
 * so it is imported dynamically like everything else GPU-touching here.
 */
async function confirmDownload(
  spec: ModelSpec,
  brand: { name: string; sizeLabel: string; ramNote?: string },
): Promise<void> {
  const { isModelCached } = await import('./cache-probe.js')
  const cached = await isModelCached(spec)
  // Cached used to mean no dialog at all — so a returning visitor who clicked
  // "Rooms" in the nav was serving a room to anyone with the link the moment
  // the page finished loading. The download question disappears when the
  // weights are local; the HOSTING question never does.

  const dlg = document.createElement('dialog')
  dlg.id = 'share-gate'
  dlg.innerHTML = `
    <h2>Host ${brand.name} from this tab</h2>
    <p>
      Hosting runs the model on THIS machine and serves it to whoever opens
      your room link. ${cached
        ? 'The weights are already cached on this device.'
        : `The weights download once (${brand.sizeLabel}) and are cached locally; every later visit starts from disk.`}
    </p>
    ${brand.ramNote ? `<p class="warn">${brand.ramNote}</p>` : ''}
    <p class="fine">Guests' prompts run on your GPU. You see every request as it arrives.</p>
    <div class="acts">
      <a href="/">Not now</a>
      <button type="button" id="share-gate-go" autofocus>${cached ? 'Start hosting →' : 'Download &amp; host →'}</button>
    </div>`
  document.body.appendChild(dlg)
  dlg.showModal()
  await new Promise<void>((resolve) => {
    dlg.querySelector<HTMLButtonElement>('#share-gate-go')?.addEventListener('click', () => {
      dlg.close()
      dlg.remove()
      resolve()
    }, { once: true })
  })
}

async function runHost(existingRoom: string | null, stageRange: { start: number; end: number } | null): Promise<void> {
  $('host-view').classList.remove('hidden')
  // Before the gate, not after it. Without this a Firefox or old-Safari
  // visitor was shown "Download & host →" for a download whose engine could
  // never boot — the refusal existed, but only on the far side of the consent
  // dialog. The guest path stays open: guests need no WebGPU.
  if (!('gpu' in navigator)) {
    const top = $('loading-error-top')
    top.style.display = 'block'
    top.textContent = 'Hosting needs WebGPU with shader-f16 — Chrome and Edge ship it; Safari from 26. '
      + 'This browser can still JOIN a room as a guest: guests run nothing locally.'
    $('badge').className = 'badge error'
    $('badge-text').textContent = 'No WebGPU'
    return
  }
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

  // Consent BEFORE the first byte, not after.
  await confirmDownload(spec, brand)

  // ?pool= — the same memory-build knob as the chat page, so the entrance's
  // "Enter & open a room" fallback link carries the chosen build honestly.
  const poolSlots = ((): number => {
    if (!spec.moe) return 0
    const v = new URLSearchParams(location.search).get('pool')
    if (!v) return 0
    const E = spec.moe.experts
    const n = v === 'half' ? Math.round(E / 2) : v === 'quarter' ? Math.round(E / 4) : Number(v)
    return Number.isFinite(n) && n > 0 ? n : 0
  })()

  // Same engine composition as chat.ts — this is the throughput path, not the
  // scalar validation path.
  const boot = await loadingUi.bootEngine({
    spec,
    optionalFeatures: ENGINE_GPU_FEATURES,
    probeSubgroups: true,
    layerRange: stageRange ?? undefined,
    ...(poolSlots ? { expertPool: poolSlots } : {}),
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = variantsMod.parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      const fused = !s.qkNorm && s.weightFormat !== 'mlx-safetensors'
      // Both halves of a split model MUST run the same kernels: two stages on
      // different machines that resolve different variants agree for a while
      // and then fork, which reads as a split bug and is not one.
      ;(window as unknown as Record<string, unknown>).__variants = { ...flags, fused }
      const kv = engineMod.allocKVFor(device, s, flags)
      return engineMod.buildDecodeEngine(device, weights, kv, {
        variants: flags, fused, spec: s, layerRange: stageRange ?? undefined,
        ...(poolSlots ? { expertPool: poolSlots } : {}),
      })
    },
    // A stage cannot run the default warmup (forwardLogits needs the whole
    // model); one pipelineStep at position 0 JITs the same shaders, and the
    // real prefill overwrites position 0 anyway.
    warmup: stageRange ? async (e) => { await e.pipelineStep({ tokenId: 1 }, 0) } : undefined,
  })
  if (!boot.ok) {
    // OUTSIDE #progress-wrap: that container is display:none until a download
    // starts, so a boot refused before any download (no WebGPU, MoE without
    // subgroups) wrote its reason into a hidden element and the page just sat
    // there with a "No WebGPU" badge and an empty room card.
    const top = $('loading-error-top')
    top.style.display = 'block'
    top.textContent = boot.reason
    $('loading-error').textContent = boot.reason
    return
  }
  const { engine, tokenizer } = boot

  const logRow = (guest: string, text: string): HTMLElement => {
    $('req-empty')?.remove()
    const li = document.createElement('li')
    // The guest id comes from the RELAY — like everything remote it goes in
    // through textContent, never markup (lens 2026-08-17).
    li.innerHTML = '<span class="who"></span> · <span class="body"></span> <span class="st"></span>'
    ;(li.querySelector('.who') as HTMLElement).textContent = guest
    ;(li.querySelector('.body') as HTMLElement).textContent = text
    $('req-log').prepend(li)
    return li.querySelector('.st') as HTMLElement
  }

  // Membership + chain text must not overwrite each other — one renderer,
  // one source for each half (the lesson from the two-writer #room-stats bug).
  let roomMembers = '1 machine serving \u00b7 0 guests connected'
  let chainText = ''
  function renderStats(): void {
    $('room-stats').textContent = chainText ? `${chainText} \u00b7 ${roomMembers}` : roomMembers
  }

  const room = hostRoom({
    spec,
    // A pooled host must not tell guests the full model's measured rate.
    brand: poolSlots ? { ...brand, rateLabel: '' } : brand,
    param: new URLSearchParams(location.search).get('model') ?? '',
    engine, tokenizer,
    encode: (messages) => buildChatPromptFor(spec, messages, tokenizer),
    existingRoom, stageRange,
    guestPath: location.pathname,
    ui: {
      row: (who, text) => {
        const st = logRow(who, text)
        return (s) => { st.textContent = s }
      },
      onMembers: ({ hosts, guests }) => {
        roomMembers = `${hosts} ${hosts === 1 ? 'machine' : 'machines'} serving \u00b7 `
          + `${guests} ${guests === 1 ? 'guest' : 'guests'} connected`
        renderStats()
      },
      onChain: (text) => { chainText = text; renderStats() },
      onPaired: (ok) => { (window as unknown as Record<string, unknown>).__stagePaired = ok },
    },
  })
  if (existingRoom) $('page-title').textContent = `Serving ${brand.name} in a shared room`
  ;($('share-link') as HTMLInputElement).value = room.link
  $('copy-link').addEventListener('click', () => {
    void navigator.clipboard.writeText(room.link)
    $('copy-link').textContent = 'Copied'
    setTimeout(() => { $('copy-link').textContent = 'Copy' }, 1200)
  })

  // e2e hooks
  ;(window as unknown as Record<string, unknown>).__shareLink = room.link
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

  // Consent BEFORE the first byte — same gate as the host. A helper link in
  // a chat message used to boot the download and enrol the GPU in a
  // stranger's room with no click at all (lens 2026-08-17).
  await confirmDownload(spec, brand)

  const boot = await loadingUi.bootEngine({
    spec,
    optionalFeatures: ENGINE_GPU_FEATURES,
    probeSubgroups: true,
    layerRange: range,
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = variantsMod.parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      ;(window as unknown as Record<string, unknown>).__variants = { ...flags, fused: false }
      return engineMod.buildDecodeEngine(device, weights, engineMod.allocKVFor(device, s, flags), {
        variants: flags, fused: false, spec: s, layerRange: range,
      })
    },
    // A residual of zeros JITs this stage's shaders; the real prefill starts
    // at position 0 again and overwrites it.
    warmup: async (e) => { await e.pipelineStep({ residual: new ArrayBuffer(spec.d * 2) }, 0) },
  })
  if (!boot.ok) {
    const top = $('loading-error-top'); top.style.display = 'block'; top.textContent = boot.reason
    $('loading-error').textContent = boot.reason; return
  }

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
        stats.textContent = `in the chain — this device runs layers ${range.start}-${range.end} of ${spec.layers}`
        ;(window as unknown as Record<string, unknown>).__helperPaired = true
      } else if (msg.type === 'stage-wait') {
        // Not a refusal: this stage fits, the chain just has not reached it.
        stats.textContent = `holding layers ${range.start}-${range.end} — ${msg.message}`
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
      setStatus(`${msg.params}${msg.rateLabel ? ` · ${msg.rateLabel}` : ''} · runs on the host machine — the host can read what you send; this page holds only the conversation.`)
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
