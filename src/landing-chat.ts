/**
 * LANDING CHAT — chat without leaving the entrance.
 *
 * ENTER on the character screen no longer navigates: the roster and the sheet
 * step aside, this module mounts a conversation panel in the right column,
 * and the character on stage stays — summoning while the weights land,
 * heart beating while it thinks, bouncing once per real token while it
 * speaks. The turn loop and boot configuration are chat-flow.ts, shared with
 * zero-tvm.html; the message surface is chat-ui.ts, shared with the rooms
 * guest. This file owns only the panel markup and the stagecraft.
 *
 * Dynamically imported by landing.ts AFTER the `'gpu' in navigator` check —
 * the chat-flow import chain reads GPUBufferUsage at module scope, and the
 * entrance must keep rendering on browsers that cannot run any of this.
 */

import type { ModelSpec } from './compiler/model-spec.js'
import type { MascotHandle } from './mascot.js'
import { mascotPalette } from './mascot.js'
import { modelBranding, specWithCtx } from './zero-tvm/model-registry.js'
import { bootChatEngine, wireChatSurface, showBootError, type ChatPhase } from './zero-tvm/chat-flow.js'
import { LANE_SIGIL, laneOf, loreOf } from './landing-lore.js'
import { recordFeat } from './feats.js'
import { makeEngineLock } from './zero-tvm/engine-lock.js'

export interface EnterChatOptions {
  /** The .cs-root the select screen rendered into. */
  root: HTMLElement
  spec: ModelSpec
  /** ?model= registry param — the room's info frame carries it so a guest
   *  who copies the weights can serve the same room. */
  param: string
  /** Expert slots of the chosen memory build (0 = full model). */
  poolSlots: number
  /** The chosen build's registry label ('' for the full model). */
  poolLabel: string
  /** Context build in tokens (0 = the compiled default). Applied through
   *  specWithCtx — the same knob as ?ctx= on the standalone page. */
  ctxTokens?: number
  /** Arrived via "Enter & open a room" — the room strip opens on its consent
   *  step once the engine is live. Never auto-hosts. */
  openRoom?: boolean
  /** The stage mascot — already showing this character. */
  mascot: MascotHandle | null
  /** Host ONE STAGE of a split here, [start, end). The swarm panel's first
   *  machine is this machine, so it boots in place; only the other stages
   *  need a URL. */
  layerRange?: { start: number; end: number }
  /** The whole split this stage belongs to: [0, …cuts, layers], which stage
   *  this machine is, and the room-wide context. The room strip needs all of
   *  it to write the other machines' links once the room exists. */
  split?: { bounds: number[]; index: number; ctx: number }
}

const ICON_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l14-7-7 14-2-5-5-2z"/></svg>'
const ICON_STOP = '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>'
const ICON_DOWN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M19 12l-7 7-7-7"/></svg>'

function panelMarkup(spec: ModelSpec, brand: ReturnType<typeof modelBranding>, buildLabel: string): string {
  const sigil = LANE_SIGIL[laneOf(spec)] ?? ''
  const size = brand.params.split(/[\s·]/)[0]
  // The one note a visitor cannot undo by waiting: RAM for the full build,
  // the build's own caveat for a chosen one (picking less memory IS the
  // answer to the RAM warning).
  const note = buildLabel !== brand.params
    ? `Build: ${buildLabel}`
    : (brand.ramNote ?? '')
  return `
    <div class="cs-chat-head">
      <span class="cs-chat-sigil" aria-hidden="true">${sigil}</span>
      <div class="cs-chat-id">
        <b>${brand.name}</b>
        <i>${buildLabel}</i>
      </div>
      <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">Summoning</span></span>
      <button class="cs-chat-tool" id="new-chat-btn" type="button" title="New chat" disabled>New chat</button>
      <a class="cs-chat-tool" id="cs-roster-link" href="/" title="Back to the character select">⟨ Roster</a>
    </div>
    <div class="cs-boot" id="progress-wrap">
      <div class="cs-boot-title" id="loading-title">Summoning ${brand.name}</div>
      <div class="cs-boot-status" id="progress-status" aria-live="polite">Preparing…</div>
      <div class="cs-boot-track"><i id="progress-bar"></i></div>
      <div class="cs-boot-detail" id="progress-detail">${brand.sizeLabel} · cached after first load — next visit starts in seconds</div>
      ${note ? `<div class="cs-boot-note">${note}</div>` : ''}
      <details class="cs-boot-log"><summary>Rite log</summary><pre id="progress-log"></pre></details>
      <div class="cs-boot-error" id="loading-error"></div>
    </div>
    <main class="chat-main" id="chat-main">
      <div class="chat-inner">
        <div class="welcome cs-welcome cs-wait" id="welcome">
          <div class="cs-welcome-title">Speak with ${brand.name}</div>
          <div class="cs-welcome-lore">${loreOf(spec)} Prompts, tokens and the KV cache stay in this tab.</div>
          <div class="cs-sug-grid" id="suggest-grid">
            <button class="suggest" data-prompt="What is zero-tvm, and what makes it different from other ways of running an LLM in a browser?">What is zero-tvm?</button>
            <button class="suggest" data-prompt="Explain WebGPU compute shaders in three sentences.">Explain WebGPU compute shaders</button>
            <button class="suggest" data-prompt="Write a TypeScript memoized fibonacci with a short test block that prints fib(10), fib(20), and fib(30).">Write a memoized fibonacci in TypeScript</button>
            <button class="suggest" data-prompt="Summarize how a transformer decode step works, focusing on what the KV cache actually stores.">What does the KV cache actually store?</button>
            <button class="suggest" data-prompt="List four kinds of questions where a ${size} model like ${brand.name} is likely to fall short compared to a frontier LLM, with one-sentence reasons.">Where does a ${size} model fall short?</button>
          </div>
        </div>
        <div id="messages"></div>
      </div>
      <button class="scroll-fab" id="scroll-fab" title="Scroll to bottom" aria-label="Scroll to bottom">${ICON_DOWN}</button>
    </main>
    <div class="composer-wrap">
      <form class="composer" id="composer">
        <textarea id="inp" rows="1" placeholder="Summoning…" aria-label="Message" disabled></textarea>
        <button class="composer-btn" data-variant="send" id="btn" type="submit" disabled aria-label="Send">${ICON_SEND}</button>
        <button class="composer-btn" data-variant="stop" id="stop-btn" type="button" hidden aria-label="Stop">${ICON_STOP}</button>
      </form>
      <div class="composer-hint">
        <span id="ctx-hint">Zero TVM · 10 WGSL kernel roles</span>
        <span class="cs-chat-stats" id="stats"></span>
      </div>
    </div>`
}

export async function enterChat(opts: EnterChatOptions): Promise<void> {
  const { root, mascot } = opts
  // The context build rebuilds the spec (page table + every derived field)
  // BEFORE anything reads it — the panel, the boot, and the turn loop all see
  // the same window. Chat history is keyed by spec.id, which specWithCtx
  // preserves, so a conversation carries across context builds of one model.
  const spec = opts.ctxTokens ? specWithCtx(opts.spec, opts.ctxTokens) : opts.spec
  const brand = modelBranding(spec)
  const buildLabel = [
    opts.poolLabel || brand.params,
    spec.maxContext !== opts.spec.maxContext
      ? `${Math.round(spec.maxContext / 1024)}k ctx`
      : '',
  ].filter(Boolean).join(' · ')

  // The page becomes this character's page: the lane accent the mascot wears
  // is the accent every chat-ui control reads (same move as zero-tvm.html).
  {
    const { accent, accentHi } = mascotPalette(spec)
    const st = document.documentElement.style
    st.setProperty('--accent', accent)
    st.setProperty('--accent-hi', accentHi)
    st.setProperty('--accent-2', accentHi)
    st.setProperty('--accent-dim', `${accent}22`)
    st.setProperty('--accent-tint', `${accent}1f`)
  }
  document.title = `${brand.name} · zero-tvm`

  const panel = document.createElement('section')
  panel.className = 'cs-chat'
  panel.setAttribute('role', 'region')
  panel.setAttribute('aria-label', `Chat with ${brand.name}`)
  panel.innerHTML = panelMarkup(spec, brand, buildLabel)
  root.appendChild(panel)
  root.classList.add('cs-chatting')
  // ENTER just display:none'd itself — without a new focus target the whole
  // boot runs with focus on <body> and nothing announced. The panel takes it
  // (its aria-live status then narrates the summoning).
  panel.tabIndex = -1
  panel.focus()

  // Leaving is a fresh page — the engine and its weights do not tear down
  // mid-session, and OPFS makes the way back in fast.
  panel.querySelector<HTMLAnchorElement>('#cs-roster-link')?.addEventListener('click', (e) => {
    e.preventDefault()
    location.reload()
  })

  // ── Stagecraft ─────────────────────────────────────────────
  // The character ACTS the phase: summoning is a slow heartbeat on the ring;
  // thinking is a face (irises up, pursed mouth, perked ears — setMood, no
  // fake pulse); talking routes every real token's pulse() into the mouth,
  // so the flap envelope is the measured cadence and nothing else.
  let heart = 0
  const stage = (phase: 'summon' | ChatPhase): void => {
    clearInterval(heart)
    root.classList.toggle('cs-summoning', phase === 'summon')
    root.classList.toggle('cs-thinking', phase === 'thinking')
    root.classList.toggle('cs-generating', phase === 'generating')
    mascot?.setHover(phase === 'thinking')
    mascot?.setMood(phase === 'generating' ? 'talking' : phase === 'thinking' ? 'thinking' : 'idle')
    if (phase === 'summon') heart = window.setInterval(() => mascot?.pulse(), 640)
  }
  stage('summon')

  const boot = await bootChatEngine({
    spec,
    poolSlots: opts.poolSlots,
    layerRange: opts.layerRange,
    search: location.search,
    onDeviceLost: (info) => {
      showBootError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
  })
  if (!boot.ok) {
    // Retry re-runs the whole entry; completed shards come straight from OPFS.
    stage('idle')
    root.classList.add('cs-boot-failed')
    // Failure theater: the ring dims (CSS on cs-boot-failed) and the rite
    // card says what happened in the realm's own words. The error line below
    // it keeps the real reason — flavour never replaces the diagnostic.
    const title = panel.querySelector('#loading-title')
    if (title) title.textContent = 'The summoning failed'
    showBootError(boot.reason, () => {
      panel.remove()
      root.classList.remove('cs-chatting', 'cs-boot-failed')
      void enterChat(opts)
    })
    return
  }

  // Summoned: the rite card folds away, the welcome steps forward — and the
  // character GREETS: a held smile with a little bounce, released back to
  // idle unless a fast first message already claimed the face.
  stage('idle')
  root.classList.add('cs-ready')
  panel.querySelector('.cs-boot')?.classList.add('cs-done')
  panel.querySelector('#welcome')?.classList.remove('cs-wait')
  mascot?.setMood('greet')
  mascot?.pulse()
  setTimeout(() => {
    if (!root.classList.contains('cs-thinking') && !root.classList.contains('cs-generating')) {
      mascot?.setMood('idle')
    }
  }, 1600)

  // DEEDS — this summon actually happened; record what kind it was. The
  // lane deeds mirror the roster's own lanes; heavy = the same footprint
  // number the sheet shows (weights + KV), earned at ≥10 GB.
  recordFeat('first-summon')
  const lane = laneOf(spec)
  if (lane === 'dense' || lane === 'hybrid' || lane === 'moe') recordFeat(`lane-${lane}`)
  if (opts.poolSlots) recordFeat('pooled')
  if (opts.ctxTokens) recordFeat('long-ctx')
  {
    const w = /([\d.]+)\s*GB/.exec(opts.poolLabel || brand.sizeLabel)
    const gb = (w ? parseFloat(w[1]) : 0) + (spec.maxContext * spec.kvBytesPerToken) / 2 ** 30
    if (gb >= 10) recordFeat('heavy')
  }

  // ONE engine, one owner: the local turn loop and the room host share this
  // latch — without it their two independent busy flags interleave
  // generations into the same KV cache (lens round 2026-08-17).
  const lock = makeEngineLock()

  // A STAGE does not chat. It holds part of the model, so forwardLogits has
  // nothing to produce — the engine says so itself ("this engine is one
  // pipeline stage — drive it with pipelineStep"). Wiring the turn loop
  // anyway threw on the first message. The guest chats; this tab serves, and
  // the room strip is its whole interface.
  if (opts.layerRange) {
    root.classList.add('cs-serving')
    const w = panel.querySelector<HTMLElement>('#welcome')
    if (w) {
      w.innerHTML = `<div class="cs-welcome-title">Serving ${brand.name}</div>
        <div class="cs-welcome-lore">This tab holds layers
        ${opts.layerRange.start}–${opts.layerRange.end} and answers no chat of its own.
        Hand out the room link below and keep this tab in the foreground.</div>`
    }
    panel.querySelector<HTMLElement>('#composer')?.setAttribute('hidden', '')
  } else {
    wireChatSurface({
      spec,
      tokenizer: boot.tokenizer,
      engine: boot.engine,
      lock,
      onToken: () => mascot?.pulse(),
      onPhase: (p) => stage(p),
      // /roster leaves the chat the same way the header link does.
      commands: { roster: () => location.reload() },
    })
  }

  // The Room tool: serve this booted engine to other machines, from inside
  // the game. Additive — a failure to load it must not touch the chat.
  void import('./landing-room.js').then(({ mountRoomTool }) => mountRoomTool({
    root, panel, spec, param: opts.param,
    engine: boot.engine, tokenizer: boot.tokenizer, mascot,
    lock, poolSlots: opts.poolSlots,
    stageRange: opts.layerRange,
    split: opts.split,
    openStrip: opts.openRoom === true,
  })).catch((e) => console.warn('[landing] room tool failed to mount:', e))

  // LISTENING: typing is real attention — the character turns toward the
  // panel (CSS on cs-listening) and perks its ears (the hover state) while
  // keys are landing, and lets go a beat after they stop. The composer is
  // disabled while busy, so this can never fight the thinking/talking moods.
  {
    const inp = panel.querySelector<HTMLTextAreaElement>('#inp')
    let listenT = 0
    const unlisten = (): void => {
      root.classList.remove('cs-listening')
      if (!root.classList.contains('cs-thinking')) mascot?.setHover(false)
    }
    const listen = (): void => {
      root.classList.add('cs-listening')
      mascot?.setHover(true)
      clearTimeout(listenT)
      listenT = window.setTimeout(unlisten, 1600)
    }
    inp?.addEventListener('input', listen)
    inp?.addEventListener('focus', listen)
    inp?.addEventListener('blur', () => { clearTimeout(listenT); unlisten() })
  }
}
