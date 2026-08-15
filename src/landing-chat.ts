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
import { modelBranding } from './zero-tvm/model-registry.js'
import { bootChatEngine, wireChatSurface, showBootError, type ChatPhase } from './zero-tvm/chat-flow.js'
import { LANE_SIGIL, laneOf, loreOf } from './landing-lore.js'

export interface EnterChatOptions {
  /** The .cs-root the select screen rendered into. */
  root: HTMLElement
  spec: ModelSpec
  /** Expert slots of the chosen memory build (0 = full model). */
  poolSlots: number
  /** The chosen build's registry label ('' for the full model). */
  poolLabel: string
  /** The stage mascot — already showing this character. */
  mascot: MascotHandle | null
}

const ICON_SEND = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12l14-7-7 14-2-5-5-2z"/></svg>'
const ICON_STOP = '<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="7" y="7" width="10" height="10" rx="2"/></svg>'
const ICON_DOWN = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14M19 12l-7 7-7-7"/></svg>'

function panelMarkup(spec: ModelSpec, brand: ReturnType<typeof modelBranding>, poolLabel: string): string {
  const sigil = LANE_SIGIL[laneOf(spec)] ?? ''
  const size = brand.params.split(/[\s·]/)[0]
  // The one note a visitor cannot undo by waiting: RAM for the full build,
  // the build's own caveat for a pooled one (picking less memory IS the
  // answer to the RAM warning).
  const note = poolLabel
    ? `Memory build: ${poolLabel}`
    : (brand.ramNote ?? '')
  return `
    <div class="cs-chat-head">
      <span class="cs-chat-sigil" aria-hidden="true">${sigil}</span>
      <div class="cs-chat-id">
        <b>${brand.name}</b>
        <i>${poolLabel || brand.params}</i>
      </div>
      <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">Summoning</span></span>
      <button class="cs-chat-tool" id="new-chat-btn" type="button" title="New chat" disabled>New chat</button>
      <a class="cs-chat-tool" id="cs-roster-link" href="/" title="Back to the character select">⟨ Roster</a>
    </div>
    <div class="cs-boot" id="progress-wrap">
      <div class="cs-boot-title" id="loading-title">Summoning ${brand.name}</div>
      <div class="cs-boot-status" id="progress-status">Preparing…</div>
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
  const { root, spec, mascot } = opts
  const brand = modelBranding(spec)

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
  panel.innerHTML = panelMarkup(spec, brand, opts.poolLabel)
  root.appendChild(panel)
  root.classList.add('cs-chatting')

  // Leaving is a fresh page — the engine and its weights do not tear down
  // mid-session, and OPFS makes the way back in fast.
  panel.querySelector<HTMLAnchorElement>('#cs-roster-link')?.addEventListener('click', (e) => {
    e.preventDefault()
    location.reload()
  })

  // ── Stagecraft ─────────────────────────────────────────────
  // The character's pulse is real: during the summoning and while thinking it
  // is a steady heartbeat; while generating, every bounce is one emitted
  // token, so the cadence on stage is the measured rate and nothing else.
  let heart = 0
  const stage = (phase: 'summon' | ChatPhase): void => {
    clearInterval(heart)
    root.classList.toggle('cs-summoning', phase === 'summon')
    root.classList.toggle('cs-thinking', phase === 'thinking')
    root.classList.toggle('cs-generating', phase === 'generating')
    mascot?.setHover(phase === 'thinking')
    if (phase === 'summon') heart = window.setInterval(() => mascot?.pulse(), 640)
    if (phase === 'thinking') heart = window.setInterval(() => mascot?.pulse(), 320)
  }
  stage('summon')

  const boot = await bootChatEngine({
    spec,
    poolSlots: opts.poolSlots,
    search: location.search,
    onDeviceLost: (info) => {
      showBootError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
  })
  if (!boot.ok) {
    // Retry re-runs the whole entry; completed shards come straight from OPFS.
    stage('idle')
    root.classList.add('cs-boot-failed')
    showBootError(boot.reason, () => {
      panel.remove()
      root.classList.remove('cs-chatting', 'cs-boot-failed')
      void enterChat(opts)
    })
    return
  }

  // Summoned: the rite card folds away, the welcome steps forward.
  stage('idle')
  root.classList.add('cs-ready')
  panel.querySelector('.cs-boot')?.classList.add('cs-done')
  panel.querySelector('#welcome')?.classList.remove('cs-wait')

  wireChatSurface({
    spec,
    tokenizer: boot.tokenizer,
    engine: boot.engine,
    onToken: () => mascot?.pulse(),
    onPhase: (p) => stage(p),
  })
}
