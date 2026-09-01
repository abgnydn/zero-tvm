/**
 * SHARE — serve the model running in THIS tab to another device, browser to
 * browser, over a WebRTC data channel.
 *
 * Three roles, decided by the URL (room-url.ts's grammar):
 *   share.html?model=qwen36q3            HOST   — boots the engine, opens a
 *                                                 room, serves requests
 *   share.html?model=X&layers=k-N#<room> HELPER — holds ONE STAGE of a split
 *                                                 model and answers no chat
 *   share.html#<roomId>                  GUEST  — the conversation, over a
 *                                                 DataChannel; needs no WebGPU
 *                                                 and downloads nothing
 *
 * ── The screen ──
 * This page renders the ENTRANCE'S SCENE, not a second design. index.html's
 * character select (landing.ts) already contains every component a room needs:
 * the plate and the summoning ring for the character, `.mb-panel` for its
 * sheet, `.cs-boot` for the download, `.cs-room-consent` / `.cs-room-live` for
 * the room itself, `.sw-arc` for a machine's place in a chain, `.cs-chat` for a
 * conversation. It used to carry ~130 lines of inline CSS re-implementing all
 * of that, one shade off, under a header bar the game does not have. There is
 * no stylesheet here now: share.html links tokens.css, chat-ui.css and
 * landing.css and nothing else, and what changes between the three roles is
 * which of the entrance's own rows go in the sheet.
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
  setChatIdentity, autoGrow, wireScrollFab, hideWelcome,
  addUserMsg, addAiMsg, type AiMsgHandle,
} from './chat-ui.js'
import { specForParam, modelBranding, quantLabel } from './model-registry.js'
import { ENGINE_GPU_FEATURES } from './variants.js'
import type { ModelSpec } from '../compiler/model-spec.js'
import { fetchInventory, pullWeights, type Inventory } from './peer-weights.js'
import { serveStage } from './pipeline-peer.js'
// The scene's own vocabulary, shared with the entrance: the class sigil, the
// lore line and the lane accent a character wears. All pure functions over a
// spec — no WebGPU is touched until mountMascot actually asks for a device,
// which returns null and hides the canvas on a browser that has none.
import { mascotPalette, mountMascot, type MascotHandle } from '../mascot.js'
import { LANE_SIGIL, laneOf, loreOf } from '../landing-lore.js'
// The swarm builder hands out helper links; a helper that opened one draws the
// same arc on its own ring and repeats the same promise about its job.
import { ROLE_NOTE, drawArcs, lightArcs } from '../landing-swarm.js'
// The URL grammar lives in room-url.ts — shared with the entrance's swarm
// link builder, so a URL that page hands out is routed by the same three-way
// branch that reads it here.
import { roomIdFrom, stageRangeFrom, roleFor, roomLink } from './room-url.js'
// The host loop, the wire protocol, and the signaling constants live in
// room-host.ts — shared with the landing entrance's in-place room
// (landing-room.ts), so the two hosting surfaces cannot drift.
import {
  hostRoom, signalEnv, ICE,
  type HostMsg, type ChatReq, type StopReq, type StageOffer,
} from './room-host.js'

const { base: SIGNAL_BASE, override: SIG_OVERRIDE } = signalEnv()

const $ = (id: string) => document.getElementById(id) as HTMLElement

// The composer/FAB glyphs, the same three the entrance's chat panel draws.
const ICON_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l14-7-7 14-2-5-5-2z"/></svg>'
const ICON_STOP = '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>'
const ICON_DOWN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M19 12l-7 7-7-7"/></svg>'

/**
 * THE REACH CAVEAT, one sentence, one copy.
 *
 * The entrance states it in `.sw-reach-note` and this page shows the same note
 * in the same corner-line register. When a connection stalls, the status line
 * says THIS sentence rather than a second, differently-worded one — a page that
 * explains the same limit twice in two voices has two chances to be wrong.
 */
const REACH_STUN = 'STUN only, no TURN — same network or an ordinary home router; '
  + 'corporate and hotel usually will not'

// URL grammar, four cases (room-url.ts):
//   ?model=X                    host a NEW room
//   ?model=X&layers=0-k         host a new room holding the first layers
//   ?model=X&layers=k-N#<room>  join that room holding the rest
//   #<room>                     join as a guest (the link you hand out)
// `?model=X#<room>` also serves an EXISTING room from this device — that is
// what turns a room into a swarm: a guest that copied the weights adds
// ?model= to the link it already has and starts serving.
const room = roomIdFrom(location.hash)
const stage = stageRangeFrom(location.search)
// The assertions hold by roleFor's own definition: it returns 'helper' only
// when both parses succeeded and 'guest' only when the room did.
const role = roleFor(location.search, location.hash)
if (role === 'helper') void runHelper(room!, stage!)
else if (role === 'guest') void runGuest(room!)
else void runHost(room, stage)

// ============================================================
// THE SCENE — the entrance's own screen, one character on it
// ============================================================

/**
 * Everything landing.ts renders except what this page has no use for: no
 * roster (the model arrived in the URL — there is nothing to pick), no stage
 * arrows, no ENTER slab, no splash.
 *
 * `serving` puts the root in `cs-swarm`, the entrance's stage-mode for
 * "this character is being served across machines": it pins the scene to the
 * viewport and lets the sheet scroll inside it, dims the base ring so a
 * stage's arc reads over it, and shows the reach note in the note row. The
 * guest gets `cs-chatting`, which is the mode a conversation already has.
 */
function mountScene(o: { serving: boolean; sheet: string }): HTMLElement {
  const root = $('share-root')
  root.innerHTML = `
    <div class="cs-spires" aria-hidden="true"></div>
    <div class="cs-col cs-col-l" aria-hidden="true"></div>
    <div class="cs-col cs-col-r" aria-hidden="true"></div>
    <!-- The constellation is a MASKED shape; until paintCharacter gives it a
         lane sigil to be masked by, an unmasked box is a pale rectangle across
         half the scene (which is what a guest saw while connecting). -->
    <div class="cs-sigilbg" aria-hidden="true" hidden></div>
    <div class="cs-fog" aria-hidden="true"><i></i><i></i></div>
    <div class="cs-dust" aria-hidden="true"></div>
    <div class="mb-plate">
      <div class="cs-banner" aria-hidden="true"></div>
      <div class="mb-name"></div>
      <div class="mb-params"><span class="mb-sigil" aria-hidden="true"></span><span class="mb-params-text"></span></div>
      <div class="cs-lore"></div>
    </div>
    <div class="mb-stage">
      <div class="mb-art">
        <div class="mb-pedestal" aria-hidden="true"></div>
        <canvas class="mb-mascot" aria-hidden="true"></canvas>
      </div>
    </div>
    ${o.sheet}
    <div class="sw-reach-note" role="note">
      <span>Reach · ${REACH_STUN}</span>
      <span>Splitting needs an MLX checkpoint · every serving tab has to stay awake</span>
    </div>
    <div class="cs-live" aria-live="polite"></div>
    <div class="cs-wipe" aria-hidden="true"></div>
    <div class="cs-borderline-t" aria-hidden="true"></div>
    <div class="cs-borderline-b" aria-hidden="true"></div>`
  root.classList.add(o.serving ? 'cs-swarm' : 'cs-chatting')
  return root
}

/** The entrance's SELECTION transition — the stage flash, the plate walking in
 *  and the sheet's row stagger, re-armed by yanking `cs-in` off for a frame.
 *  Identical to landing.ts's selectFx; reduced motion disables the keyframes
 *  in the stylesheet, so there is nothing to branch on here. */
function selectFx(root: HTMLElement): void {
  const wipe = root.querySelector<HTMLElement>('.cs-wipe')
  if (wipe) { wipe.classList.remove('cs-go'); void wipe.offsetWidth; wipe.classList.add('cs-go') }
  for (const sel of ['.mb-art', '.mb-plate', '.mb-panel']) {
    const n = root.querySelector<HTMLElement>(sel)
    if (!n) continue
    n.classList.remove('cs-in')
    void n.offsetWidth
    n.classList.add('cs-in')
  }
}

/** Name, class line, lore, constellation and the lane accent — the same five
 *  things landing.ts's paint() writes when a character takes the stage. The
 *  accent lands on the root as `--cs-accent` (the scene) and on the document
 *  as `--accent` (chat-ui's controls), exactly as landing-chat.ts does it. */
function paintCharacter(root: HTMLElement, spec: ModelSpec, name: string, params: string): void {
  const q = <T extends Element>(sel: string): T | null => root.querySelector<T>(sel)
  const sigil = LANE_SIGIL[laneOf(spec)] ?? ''
  const nameEl = q<HTMLElement>('.mb-name')
  if (nameEl) nameEl.textContent = name
  const paramsEl = q<HTMLElement>('.mb-params-text')
  if (paramsEl) paramsEl.textContent = params
  const sigilEl = q<HTMLElement>('.mb-sigil')
  if (sigilEl) sigilEl.innerHTML = sigil
  const lore = q<HTMLElement>('.cs-lore')
  if (lore) lore.textContent = loreOf(spec)
  const bg = q<HTMLElement>('.cs-sigilbg')
  if (bg && sigil) {
    const uri = `url("data:image/svg+xml,${encodeURIComponent(sigil)}")`
    bg.style.webkitMaskImage = uri
    bg.style.maskImage = uri
    bg.hidden = false
  }
  const { accent, accentHi } = mascotPalette(spec)
  root.style.setProperty('--cs-accent', accent)
  root.style.setProperty('--cs-accent-hi', accentHi)
  const st = document.documentElement.style
  st.setProperty('--accent', accent)
  st.setProperty('--accent-hi', accentHi)
  st.setProperty('--accent-2', accentHi)
  st.setProperty('--accent-dim', `${accent}22`)
  st.setProperty('--accent-tint', `${accent}1f`)
}

/** The character itself. Returns null where there is no WebGPU device to draw
 *  it with — a guest's browser routinely has none — and hides the canvas then,
 *  the same fallback landing.ts takes.
 *
 *  `serving` means the weights are on THIS machine: the figure is lit and its
 *  circle is ARMED, the same two states the entrance gives a cached character.
 *  A guest's is neither — nothing of the model is here. */
async function mountFigure(root: HTMLElement, spec: ModelSpec, serving: boolean): Promise<MascotHandle | null> {
  const canvas = root.querySelector<HTMLCanvasElement>('.mb-mascot')
  const art = root.querySelector<HTMLElement>('.mb-art')
  if (!canvas) return null
  const m = await mountMascot(canvas, spec)
  if (!m) { canvas.style.display = 'none'; return null }
  m.setSpec(spec, serving)
  art?.toggleAttribute('data-armed', serving)
  return m
}

/**
 * The panel header: sigil, identity, live badge. The row the page's old `.top`
 * bar was pretending to be.
 *
 * On a serving sheet the identity is the ROLE, not the model's name: the plate
 * carries the name at display size directly above, and the sheet column is
 * 280px on a phone — measured at 390x844, `Llama-3.2-1B-Instruct` set nowrap in
 * here forced the panel 439px wide (`.mb-info` is `align-items: center` under
 * 780px, so the panel sizes to its own min-content) and it hung off both edges
 * of the screen. The guest's head keeps the name: its panel is 420-600px and
 * has no plate to read it off until the room answers.
 */
function chatHead(spec: ModelSpec, who: string, build: string): string {
  return `
    <div class="cs-chat-head sw-row" style="--i:0">
      <span class="cs-chat-sigil" aria-hidden="true">${LANE_SIGIL[laneOf(spec)] ?? ''}</span>
      <div class="cs-chat-id"><b>${who}</b><i>${build}</i></div>
      <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">Waiting</span></span>
    </div>`
}

/** The rite card, verbatim from landing-chat.ts's panel — loading-ui.ts writes
 *  every id in it. Hidden until consent: a progress card sitting at "Preparing…"
 *  above a question nobody has answered yet reads as a stall. */
function bootCard(i: number, title: string, detail: string): string {
  return `
    <div class="cs-boot sw-row" style="--i:${i}" id="progress-wrap" hidden>
      <div class="cs-boot-title" id="loading-title">${title}</div>
      <div class="cs-boot-status" id="progress-status" aria-live="polite">Preparing…</div>
      <div class="cs-boot-track"><i id="progress-bar"></i></div>
      <div class="cs-boot-detail" id="progress-detail">${detail}</div>
      <details class="cs-boot-log"><summary>Rite log</summary><pre id="progress-log"></pre></details>
      <div class="cs-boot-error" id="loading-error"></div>
    </div>`
}

/** A refusal, in the rite card rather than in a hidden container. The old page
 *  wrote boot failures into `#progress-wrap`, which was display:none until a
 *  download started — so a boot refused BEFORE any download (no WebGPU, MoE
 *  without subgroups) left the page sitting silently on a badge. */
function bootFail(reason: string, badge: string): void {
  const wrap = document.getElementById('progress-wrap')
  if (wrap) wrap.hidden = false
  const err = document.getElementById('loading-error')
  if (err) { err.textContent = reason; err.classList.add('visible') }
  const b = document.getElementById('badge')
  if (b) b.className = 'badge error'
  const t = document.getElementById('badge-text')
  if (t) t.textContent = badge
}

/**
 * Keep-awake — screen wake-lock plus a silent audio track. The wake lock stops
 * the machine sleeping; the audio track is the standard exemption from
 * background-tab throttling (measured: a backgrounded host generated at
 * ~23 tok/s where the focused tab does ~65, and served weights at ~1 MB/s).
 * Honestly labeled: the browser shows its audio indicator on the tab.
 *
 * Both serving roles need it — a helper stage especially, since it is a
 * background tab for its whole life. It is a `.cs-chat-tool` toggle now rather
 * than a checkbox, so it wears the scene's own control chrome and lights with
 * `.cs-tool-live` when it is on, the way the entrance's ⟁ Room tool does.
 */
function wireKeepAwake(): void {
  const btn = document.getElementById('awake') as HTMLButtonElement | null
  if (!btn) return
  let on = false
  let wakeLock: WakeLockSentinel | null = null
  let audioCtx: AudioContext | null = null
  const apply = async (want: boolean): Promise<void> => {
    if (want) {
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
  btn.addEventListener('click', () => {
    on = !on
    btn.setAttribute('aria-pressed', String(on))
    btn.classList.toggle('cs-tool-live', on)
    void apply(on)
  })
  document.addEventListener('visibilitychange', () => {
    // The UA releases wake locks on hide; re-acquire when we come back.
    if (on && document.visibilityState === 'visible') void apply(true)
  })
}

/** The keep-awake control and the sentence under it, as sheet rows. */
function keepAwakeRows(i: number): string {
  return `
    <button type="button" class="cs-chat-tool sw-row" style="--i:${i}" id="awake" aria-pressed="false">⟁ Keep this tab awake</button>
    <p class="sw-note sw-row" style="--i:${i + 1}">Screen wake-lock plus a silent audio track, so the browser throttles
      generation and weight serving less while this tab is in the background. The tab shows its audio indicator.</p>`
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
 * The gate is the room's FIRST STEP now, not a dialog over the page: the same
 * `.cs-room-consent` block the entrance's ⟁ Room tool opens on, with the same
 * voice. `#share-gate-go` keeps its id — it is the e2e's handle on this click.
 *
 * The weights sentence resolves against the cache: nothing to consent to when
 * the bytes are already here, and a returning host should not be asked twice
 * about a download. Cached used to mean no gate at all — so a returning
 * visitor who clicked "Rooms" in the nav was serving a room to anyone with the
 * link the moment the page finished loading. The download question disappears
 * when the weights are local; the HOSTING question never does.
 */
async function confirmDownload(
  spec: ModelSpec,
  brand: { name: string; sizeLabel: string; ramNote?: string },
  role: {
    /** 'helper' joins someone else's room and offers layers; 'host' opens one. */
    kind: 'host' | 'helper'
    /** Set when THIS device holds ONE STAGE — either half of a split. */
    stage?: { start: number; end: number }
  },
): Promise<void> {
  const { isModelCached } = await import('./cache-probe.js')
  const { stage } = role
  const cached = await isModelCached(spec, stage)

  // WHAT THIS DEVICE FETCHES — one sentence, both roles, because both can be
  // a stage. The gate used to quote brand.sizeLabel unconditionally: the
  // iPhone that held one layer of the 27B was asked to approve "~14.1 GB" for
  // a slice worth a fraction of that (real device, 2026-08-29). A stage's
  // exact bytes are not knowable here — they come from the safetensors
  // headers, read after consent — so this states the RANGE and leaves the
  // size to the progress panel. It does not guess a number.
  const weightsLine = cached
    ? (stage
        ? `Layers ${stage.start}-${stage.end} are already cached on this device.`
        : 'The weights are already cached on this device.')
    : (stage
        ? `Layers ${stage.start}-${stage.end} of ${spec.layers} download once — a slice of the `
          + `full ${brand.sizeLabel}, not all of it — and are cached locally; the progress panel `
          + 'below shows the real size as it arrives.'
        : `The weights download once (${brand.sizeLabel}) and are cached locally; every later visit starts from disk.`)
  // brand.ramNote is a whole-checkpoint figure ("needs ~20 GB free RAM") and
  // is false for a stage. Nothing replaces it: a per-stage number would have
  // to be invented.
  const ram = document.getElementById('gate-ram')
  if (ram) {
    if (!stage && brand.ramNote) ram.textContent = brand.ramNote
    else ram.hidden = true
  }
  const weights = document.getElementById('gate-weights')
  if (weights) weights.textContent = weightsLine

  const go = document.getElementById('share-gate-go') as HTMLButtonElement
  go.textContent = role.kind === 'helper'
    ? (cached ? 'Serve these layers →' : 'Download the layers & serve →')
    : (cached ? 'Start hosting →' : 'Download & host →')
  go.disabled = false
  await new Promise<void>((resolve) => {
    go.addEventListener('click', () => {
      go.disabled = true
      resolve()
    }, { once: true })
  })
}

async function runHost(existingRoom: string | null, stageRange: { start: number; end: number } | null): Promise<void> {
  // The registry spec is enough to paint the character (name, lane, lore,
  // accent) and it needs no WebGPU — so the scene is on screen before the
  // loader chain is even imported, and stays there if the import is refused.
  // `?ctx=` only moves maxContext, which the sheet reads from the BOOTED spec
  // below rather than from this one.
  const param = new URLSearchParams(location.search).get('model') ?? ''
  const baseSpec = specForParam(param)
  const brand = modelBranding(baseSpec)
  const stageLine = stageRange ? `layers ${stageRange.start}–${stageRange.end} of ${baseSpec.layers}` : ''
  document.title = stageRange
    ? `${brand.name} · ${stageLine} · zero-tvm`
    : `${brand.name} · room · zero-tvm`

  const root = mountScene({
    serving: true,
    sheet: `
    <aside class="mb-info">
      <div class="mb-panel">
        ${chatHead(baseSpec, stageRange ? 'Hosting a stage' : 'Hosting', brand.params)}
        <div class="mb-row-label sw-row" style="--i:1">Room</div>
        <div class="cs-room sw-row" style="--i:2">
          <div class="cs-room-consent">
            <p>Open a room and whoever has the link chats with <b>${brand.name}</b> running on
            THIS machine. Their prompts run on your GPU; every request is listed here as it
            arrives. Guests can also copy the model's cached weights from this machine to
            run it locally. Keep this tab in the foreground while serving.</p>
            <p id="gate-weights"></p>
            <p class="mb-ram" id="gate-ram"></p>
            <button type="button" class="cs-chat-tool" id="share-gate-go" disabled>Checking this device…</button>
          </div>
          <div class="cs-room-live" hidden>
            <div class="cs-room-linkrow">
              <input id="share-link" readonly aria-label="Room link">
              <button type="button" class="cs-chat-tool" id="copy-link">Copy</button>
            </div>
            <div class="cs-room-members" id="room-stats" aria-live="polite"></div>
            <div class="mb-row-label">Link for a machine that will serve too</div>
            <div class="cs-room-linkrow">
              <input id="helper-link" readonly aria-label="Link for another serving machine">
              <button type="button" class="cs-chat-tool" id="copy-helper">Copy</button>
            </div>
            <div class="mb-row-label">Requests</div>
            <ul class="cs-room-log" id="req-log" role="log" aria-live="polite" aria-label="Guest requests"></ul>
          </div>
        </div>
        ${bootCard(3, `Summoning ${brand.name}`,
          stageRange
            ? `${stageLine} · cached after first load`
            : `${brand.sizeLabel} · cached after first load — next visit starts in seconds`)}
        ${keepAwakeRows(4)}
        <div class="mb-row-label sw-row" style="--i:6">This machine</div>
        <dl class="mb-stats sw-row" style="--i:7"></dl>
      </div>
    </aside>`,
  })
  // A stage's nameplate states the SLICE, the same line the helper's does:
  // what this machine holds is the one thing about it that is not the model's.
  paintCharacter(root, baseSpec, brand.name, stageLine || brand.params)
  selectFx(root)
  void mountFigure(root, baseSpec, true)
  wireKeepAwake()

  // Before the gate, not after it. Without this a Firefox or old-Safari
  // visitor was shown "Download & host →" for a download whose engine could
  // never boot — the refusal existed, but only on the far side of the consent
  // dialog. The guest path stays open: guests need no WebGPU.
  if (!('gpu' in navigator)) {
    // Nothing will boot, so the room question and the sheet of figures about
    // what would have run are both moot — the refusal is the whole screen.
    for (const sel of ['.cs-room', '.mb-stats', '.mb-row-label']) {
      for (const n of root.querySelectorAll<HTMLElement>(sel)) n.remove()
    }
    bootFail('Hosting needs WebGPU with shader-f16 — Chrome and Edge ship it; Safari from 26. '
      + 'This browser can still JOIN a room as a guest: guests run nothing locally.', 'No WebGPU')
    return
  }
  // Everything GPU-touching is imported HERE, not at module scope — the guest
  // path must run on machines without WebGPU, and weight-loader.ts reads
  // GPUBufferUsage the moment it is imported.
  const [{ specFromSearch, buildChatPromptFor }, loadingUi, variantsMod, engineMod] =
    await Promise.all([
      import('./model-select.js'),
      import('./loading-ui.js'),
      import('./variants.js'),
      import('./engine-core.js'),
    ])

  const spec = specFromSearch(location.search)
  if (stageRange && stageRange.start !== 0) {
    throw new Error(`share: a hosting stage must start at layer 0 (got ${stageRange.start}); later stages join a room as helpers`)
  }

  // The sheet, from the spec that will actually boot. A stage replaces the
  // Weights row: the download is a SLICE, and quoting the whole checkpoint's
  // size next to "layers 0-13" is the figure that sent an iPhone a 14.1 GB
  // promise for a fraction of it.
  const rows: Array<[string, string]> = [
    stageRange
      ? ['Layers', `${stageRange.start}–${stageRange.end} of ${spec.layers}`]
      : ['Weights', brand.sizeLabel],
    ['Quantisation', quantLabel(spec)],
    ['Context', `${spec.maxContext.toLocaleString()} tokens`],
  ]
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
  // No rate row. The entrance carries measured throughput because that is a
  // screen where you CHOOSE a model; here the model arrived in the URL and a
  // number nobody can act on is decoration.
  root.querySelector<HTMLElement>('.mb-stats')!.innerHTML = rows
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')

  // Consent BEFORE the first byte. A host with ?layers=0-k holds a stage too —
  // it fetches that slice, not the whole checkpoint.
  await confirmDownload(spec, brand, { kind: 'host', ...(stageRange ? { stage: stageRange } : {}) })
  $('progress-wrap').hidden = false
  root.classList.add('cs-summoning')

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
  root.classList.remove('cs-summoning')
  if (!boot.ok) {
    root.classList.add('cs-boot-failed')
    const title = document.getElementById('loading-title')
    if (title) title.textContent = 'The summoning failed'
    bootFail(boot.reason, 'Failed')
    return
  }
  const { engine, tokenizer } = boot
  // Summoned: the rite card folds away exactly as it does in the entrance.
  $('progress-wrap').classList.add('cs-done')

  const logRow = (guest: string, text: string): HTMLElement => {
    const log = $('req-log')
    const li = document.createElement('li')
    // The guest id comes from the RELAY — like everything remote it goes in
    // through textContent, never markup (lens 2026-08-17).
    li.innerHTML = '<b></b><span></span><i></i>'
    ;(li.querySelector('b') as HTMLElement).textContent = guest.slice(0, 8)
    ;(li.querySelector('span') as HTMLElement).textContent = text
    log.prepend(li)
    while (log.children.length > 8) log.lastChild?.remove()
    return li.querySelector('i') as HTMLElement
  }

  // Membership + chain text must not overwrite each other — one renderer,
  // one source for each SEGMENT (the lesson from the two-writer #room-stats
  // bug). `shape` is the third: what this tab actually is, which the room
  // card never said — a host could not read its own model, its own slice or
  // its own context off the screen, and neither could anyone it showed it to.
  // Fixed for the tab's life, so it is a constant rather than a callback.
  const shape = stageRange
    ? `layers ${stageRange.start}-${stageRange.end} of ${spec.layers} here`
    : `all ${spec.layers} layers here`
  let roomMembers = '1 machine serving · 0 guests connected'
  let chainText = ''
  function renderStats(): void {
    $('room-stats').textContent = [shape, chainText, roomMembers].filter(Boolean).join(' · ')
  }

  const roomHandle = hostRoom({
    spec,
    // A pooled host must not tell guests the full model's measured rate.
    brand: poolSlots ? { ...brand, rateLabel: '' } : brand,
    param,
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
        roomMembers = `${hosts} ${hosts === 1 ? 'machine' : 'machines'} serving · `
          + `${guests} ${guests === 1 ? 'guest' : 'guests'} connected`
        renderStats()
      },
      onChain: (text) => { chainText = text; renderStats() },
      onPaired: (ok) => { (window as unknown as Record<string, unknown>).__stagePaired = ok },
    },
  })
  renderStats()
  root.querySelector<HTMLElement>('.cs-room-consent')!.hidden = true
  root.querySelector<HTMLElement>('.cs-room-live')!.hidden = false

  const wireCopy = (btnId: string, inputId: string, value: string): void => {
    ;(document.getElementById(inputId) as HTMLInputElement).value = value
    const btn = $(btnId)
    btn.addEventListener('click', () => {
      void navigator.clipboard.writeText(value)
      btn.textContent = 'Copied'
      setTimeout(() => { btn.textContent = 'Copy' }, 1200)
    })
  }
  wireCopy('copy-link', 'share-link', roomHandle.link)
  // The link a SECOND machine opens to serve this room — model, context and
  // (when this host holds only the first layers) the layers still missing,
  // all in the query, room id in the fragment. It used to be hand-edited out
  // of the guest link above, which is how a stage ended up on a different
  // ?ctx= than the host.
  wireCopy('copy-helper', 'helper-link', roomHandle.helperLink)

  // e2e hooks
  ;(window as unknown as Record<string, unknown>).__shareLink = roomHandle.link
  ;(window as unknown as Record<string, unknown>).__helperLink = roomHandle.helperLink
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
 *
 * The screen says exactly that and nothing more: a helper answers no chat, so
 * there is no composer, no message log and nothing to type into. One stat —
 * the layers it holds — the keep-awake switch it depends on for its whole
 * life, and the one arc it occupies on the ring the swarm builder drew.
 */
async function runHelper(roomId: string, range: { start: number; end: number }): Promise<void> {
  const param = new URLSearchParams(location.search).get('model') ?? ''
  const baseSpec = specForParam(param)
  const brand = modelBranding(baseSpec)
  const layersLine = `layers ${range.start}–${range.end} of ${baseSpec.layers}`
  document.title = `${brand.name} · ${layersLine} · zero-tvm`

  const root = mountScene({
    serving: true,
    sheet: `
    <aside class="mb-info">
      <div class="mb-panel">
        ${chatHead(baseSpec, 'Helper stage', brand.params)}
        <div class="cs-room sw-row" style="--i:1">
          <div class="cs-room-consent">
            <p>This device holds ONE STAGE of <b>${brand.name}</b> — layers ${range.start} to
            ${range.end} of ${baseSpec.layers}, not the whole checkpoint. <span id="gate-weights"></span></p>
            <p>The machine that starts the model sends this stage a hidden state for every
            token and gets one back. It keeps the room and the conversation; this tab holds
            only its layers.</p>
            <p class="mb-ram" id="gate-ram"></p>
            <button type="button" class="cs-chat-tool" id="share-gate-go" disabled>Checking this device…</button>
          </div>
        </div>
        ${bootCard(2, `Summoning ${layersLine}`, `${layersLine} · cached after first load`)}
        <div class="sw-stop sw-row" style="--i:3" data-role="helper">
          <div class="sw-stop-head"><b>${layersLine}</b><span>helper</span></div>
        </div>
        ${keepAwakeRows(4)}
        <div class="cs-room-members sw-row" style="--i:6" id="room-stats" aria-live="polite"></div>
        <p class="sw-note sw-row" style="--i:7">${ROLE_NOTE.helper}</p>
      </div>
    </aside>`,
  })
  paintCharacter(root, baseSpec, brand.name, layersLine)
  selectFx(root)
  void mountFigure(root, baseSpec, true)
  wireKeepAwake()

  // ONE arc on the ring — this machine's place in the chain the swarm builder
  // drew. Lit when the host accepts the stage, which is the only moment this
  // tab is actually part of a model.
  const arcs: HTMLElement[] = []
  const pedestal = root.querySelector<HTMLElement>('.mb-pedestal')
  const art = root.querySelector<HTMLElement>('.mb-art')
  drawArcs(pedestal, arcs, 1)
  const setArc = (lit: boolean): void => lightArcs(art, arcs, [lit])
  setArc(false)

  const stats = $('room-stats')
  // "loading layers X-Y…" set once and never again was the only thing an
  // iPhone helper ever showed for the whole download (2026-08-29): it reads as
  // "loading" whether the tab is working, finished, or dead. Every line below
  // states only what is true at that point, and the rite card carries the rest.
  // The context is the host's, adopted from the link (ctxFor) — printed because
  // a stage silently sizing its KV cache off its own default is exactly the
  // failure that rule prevents, and an unprinted number cannot be checked
  // against the host's.
  stats.textContent = 'not started yet'

  // The same refusal the host path does, and for the same reason: without it
  // the consent step asks a browser that can never boot the engine to approve
  // a download. It matters more here than there — a helper link is something
  // you paste to a phone, and every iOS browser is WebKit, so anything before
  // Safari 26 lands on this branch.
  if (!('gpu' in navigator)) {
    root.querySelector<HTMLElement>('.cs-room')?.remove()
    stats.textContent = 'this browser cannot serve a stage'
    bootFail('Serving a stage needs WebGPU with shader-f16 — Chrome and Edge ship it; Safari from 26. '
      + 'This browser can still JOIN the room as a guest: open the link without the ?layers= part.', 'No WebGPU')
    return
  }
  const [{ specFromSearch }, loadingUi, variantsMod, engineMod] = await Promise.all([
    import('./model-select.js'),
    import('./loading-ui.js'),
    import('./variants.js'),
    import('./engine-core.js'),
  ])
  const spec = specFromSearch(location.search)
  stats.textContent = `${spec.maxContext.toLocaleString()}-token context — not started yet`

  // Consent BEFORE the first byte — same gate as the host. A helper link in
  // a chat message used to boot the download and enrol the GPU in a
  // stranger's room with no click at all (lens 2026-08-17).
  await confirmDownload(spec, brand, { kind: 'helper', stage: range })
  root.querySelector<HTMLElement>('.cs-room-consent')!.hidden = true
  $('progress-wrap').hidden = false
  root.classList.add('cs-summoning')
  stats.textContent = `loading ${layersLine}…`

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
  root.classList.remove('cs-summoning')
  if (!boot.ok) {
    root.classList.add('cs-boot-failed')
    bootFail(boot.reason, 'Failed')
    return
  }
  $('progress-wrap').classList.add('cs-done')
  // Loaded and idle is a real state, and the page used to have no word for it:
  // the line above still read "loading layers 2-6 of 48…" long after the badge
  // said Ready, which on a phone is indistinguishable from a stall.
  stats.textContent = 'loaded — looking for the machine that starts the model'

  const ws = new WebSocket(`${SIGNAL_BASE}/room/${roomId}?role=guest`)
  const pc = new RTCPeerConnection(ICE)
  let chat: RTCDataChannel | null = null
  pc.onicecandidate = (e) => {
    if (e.candidate) ws.send(JSON.stringify({ type: 'ice', candidate: e.candidate }))
  }
  pc.ondatachannel = (e) => {
    if (e.channel.label === 'pipeline') {
      serveStage(e.channel, boot.engine, (m) => { stats.textContent = m })
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
        stats.textContent = 'in the chain — this device runs its layers'
        setArc(true)
        ;(window as unknown as Record<string, unknown>).__helperPaired = true
      } else if (msg.type === 'stage-wait') {
        // Not a refusal: this stage fits, the chain just has not reached it.
        stats.textContent = `holding — ${msg.message}`
      } else if (msg.type === 'stage-reject') {
        setArc(false)
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
      setArc(false)
      // The reach caveat, surfaced where the stall is — the SAME sentence the
      // note in the corner already carries, never a second wording of it.
      stats.textContent = 'no host in this room to pair with — open the room link on the machine '
        + `holding the first layers. Reach: ${REACH_STUN}.`
    }
  }
  ;(window as unknown as Record<string, unknown>).__helperReady = true
}

// ============================================================
// GUEST — the entrance's chat panel over a DataChannel
// ============================================================

async function runGuest(roomId: string): Promise<void> {
  // The stranger who never saw the entrance. The scene mounts IMMEDIATELY,
  // over a connecting state: no model is known yet — the room tells us which
  // character this is — but the room, the ring and the panel are the project's
  // first impression and must not wait on a WebSocket.
  const root = mountScene({
    serving: false,
    sheet: `
    <section class="cs-chat" role="region" aria-label="Chat with the model in this room">
      <div class="cs-chat-head">
        <span class="cs-chat-sigil" id="guest-sigil" aria-hidden="true"></span>
        <div class="cs-chat-id"><b id="guest-model">This room</b><i id="guest-build">remote</i></div>
        <span class="badge loading" id="guest-badge"><span class="dot"></span><span id="guest-badge-text">Connecting</span></span>
        <a class="cs-chat-tool" href="/" title="Back to the character select">⟨ Roster</a>
      </div>
      <!-- Shown only when the host has a weight cache this device could copy.
           The hidden CLASS is kept in step with the hidden ATTRIBUTE: the
           attribute does the hiding, the class is what peer-weights-e2e
           watches for. -->
      <div class="cs-room hidden" id="local-copy" hidden>
        <div class="cs-room-consent">
          <p id="lc-status"></p>
          <button type="button" class="cs-chat-tool" id="lc-btn"></button>
        </div>
      </div>
      <main class="chat-main" id="chat-main">
        <div class="chat-inner">
          <div class="welcome cs-welcome cs-wait" id="welcome">
            <div class="cs-welcome-title" id="welcome-title"></div>
            <div class="cs-welcome-lore" id="welcome-lore"></div>
          </div>
          <div id="messages"></div>
        </div>
        <button class="scroll-fab" id="scroll-fab" title="Scroll to bottom" aria-label="Scroll to bottom">${ICON_DOWN}</button>
      </main>
      <div class="composer-wrap">
        <form class="composer" id="composer">
          <textarea id="inp" rows="1" placeholder="Connecting to the host…" aria-label="Message" disabled></textarea>
          <button class="composer-btn" data-variant="send" id="btn" type="submit" disabled aria-label="Send">${ICON_SEND}</button>
          <button class="composer-btn" data-variant="stop" id="stop-btn" type="button" hidden aria-label="Stop">${ICON_STOP}</button>
        </form>
        <div class="composer-hint">
          <span id="room-count"></span>
          <span id="guest-status">Waiting for the host.</span>
        </div>
      </div>
    </section>`,
  })
  // The nameplate waits with what is actually known — the room, and nothing
  // about the character in it. `info` replaces both lines through the
  // entrance's own plateIn, so the reveal reads as a character arriving.
  root.querySelector<HTMLElement>('.mb-name')!.textContent = 'This room'
  root.querySelector<HTMLElement>('.mb-params-text')!.textContent = 'connecting'

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
  let mascot: MascotHandle | null = null

  const inp = $('inp') as HTMLTextAreaElement
  const sendBtn = $('btn') as HTMLButtonElement
  const stopBtn = $('stop-btn') as HTMLButtonElement
  const setGenerating = (on: boolean): void => {
    sendBtn.hidden = on
    stopBtn.hidden = !on
    sendBtn.disabled = on
    root.classList.toggle('cs-generating', on)
  }

  const settle = (): void => {
    live = null
    setGenerating(false)
    root.classList.remove('cs-thinking')
  }

  /**
   * The room named its character. Paint it with the SAME sequence the entrance
   * runs on selection — charIn on the figure, ringFlash on the circle, plateIn
   * on the nameplate, all off the `cs-in` class — because that is what this is:
   * a character taking the stage, for someone who never saw the roster.
   */
  const reveal = (msg: Extract<HostMsg, { type: 'info' }>): void => {
    setChatIdentity(msg.name, msg.tag)
    document.title = `${msg.name} · room · zero-tvm`
    $('guest-model').textContent = msg.name
    $('guest-build').textContent = `${msg.params} · remote`
    const spec = specForParam(msg.param)
    // The OPFS directory and everything drawn from the spec are chosen LOCALLY;
    // a host whose spec id this build does not know gets the name it sent and
    // no character, rather than someone else's portrait.
    if (spec.id !== msg.specId) {
      $('welcome-title').textContent = `Speak with ${msg.name}`
      return
    }
    $('guest-sigil').innerHTML = LANE_SIGIL[laneOf(spec)] ?? ''
    paintCharacter(root, spec, msg.name, msg.params)
    $('welcome-title').textContent = `Speak with ${msg.name}`
    $('welcome-lore').textContent = `${loreOf(spec)} It runs on the machine that opened this room; `
      + 'this page holds only the conversation.'
    selectFx(root)
    void mountFigure(root, spec, false).then((m) => { mascot = m })
  }

  const onHostMsg = (msg: HostMsg): void => {
    if (msg.type === 'info') {
      reveal(msg)
      // The host's advertised rate is deliberately not shown: a guest cannot
      // act on it, and the one thing a guest DOES need to know about this room
      // is whose machine the words land on.
      setStatus(`${msg.params}${msg.ctx ? ` · ${msg.ctx.toLocaleString()}-token context` : ''}`
        + ' · runs on the host machine — the host can read what you send; this page holds only the conversation.')
      setBadge('Ready', 'ready')
      $('welcome').classList.remove('cs-wait')
      inp.disabled = false
      inp.placeholder = `Message ${msg.name}…`
      sendBtn.disabled = false
      void offerLocalCopy(msg.param, msg.specId, msg.ctx)
    } else if (msg.type === 'text') {
      liveFull = msg.full
      live?.render(liveFull)
      root.classList.remove('cs-thinking')
      mascot?.pulse()
    } else if (msg.type === 'done') {
      if (live) {
        live.finish({ fullText: liveFull, tokens: msg.tokens, tokPerS: msg.tps })
        history.push({ role: 'assistant', content: liveFull })
      }
      mascot?.setMood('idle')
      settle()
    } else if (msg.type === 'busy') {
      setStatus(`Host is generating for someone else — queue position ${msg.pos}.`)
    } else if (msg.type === 'error') {
      if (live) live.body.textContent = `⚠ ${msg.message}`
      settle()
    }
  }

  // e2e/dev hook: the same frames the chat channel delivers, so the guest's
  // revealed state (portrait, plate, ring, welcome) can be driven and looked
  // at without standing up a relay and a host with real weights behind it.
  ;(window as unknown as Record<string, unknown>).__guestFrame = onHostMsg

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
      // The reach caveat where the stall happens, in the note's own words —
      // "nobody is serving" and "your two machines cannot see each other" look
      // identical from here, and only one of them is worth waiting out.
      setStatus('Nobody is serving a model in this room — every host tab is closed, or the link '
        + `expired. Reach: ${REACH_STUN}.`)
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
  async function offerLocalCopy(param: string, specId: string, ctx?: number): Promise<void> {
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
    panel.hidden = false
    status.textContent = `The host has ${gb} GB cached. Copying it here lets this device run ${spec.id} on its own GPU.`
    btn.textContent = `Copy ${gb} GB to this device`
    btn.addEventListener('click', () => {
      btn.disabled = true
      const t = performance.now()
      void pullWeights(weightsDc, spec, (p) => {
        const pct = p.bytesTotal ? Math.round((p.bytesDone / p.bytesTotal) * 100) : 0
        status.textContent = `${pct}% · ${(p.bytesDone / 1e9).toFixed(2)}/${gb} GB · `
          + `${p.filesDone}/${p.filesTotal} files`
      }, inv)
        .then((res) => {
          const secs = ((performance.now() - t) / 1000).toFixed(0)
          const p = encodeURIComponent(param)
          // The swarm-forming link: this device has the weights now, so it
          // can serve the SAME room instead of only consuming it. Built by
          // roomLink so the query lands BEFORE the fragment, and carrying the
          // host's ctx so this machine runs the room's context rather than
          // its own compiled default (ctxFor).
          const serve = roomLink({
            origin: '', path: '/share.html', room: roomId,
            model: param, ctx: ctx ?? null, sig: SIG_OVERRIDE,
          })
          status.innerHTML = `Done — ${(res.bytes / 1e9).toFixed(2)} GB in ${secs}s. `
            + `<a href="/zero-tvm.html?model=${p}">Open the chat on this device →</a><br>`
            + `<a href="${serve}">…or serve this room from here too →</a>`
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
    hideWelcome()
    history.push({ role: 'user', content: text })
    addUserMsg(text)
    live = addAiMsg()
    live.showThinking()
    liveFull = ''
    setGenerating(true)
    // The character on stage waits with you: the ring tightens while the host
    // prefills, then every frame of text is a pulse in its mouth.
    root.classList.add('cs-thinking')
    mascot?.setMood('thinking')
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
