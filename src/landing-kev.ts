/**
 * LANDING DECIDE — typed decisions without leaving the entrance.
 *
 * ENTER on a decision character (kev: `spec.decisionOnly`) mounts this
 * console in the right column instead of the chat: a state box, a questions
 * box in the TypeSafe request shape, and one probability bar per option out.
 * The character on stage stays. The turn loop is kev.ts's `decide` over the
 * engine's `forwardHiddenPacked` — state prefilled once, every branch in as
 * few passes as fit — with the pointer head applied on the CPU.
 *
 * Dynamically imported by landing.ts AFTER the `'gpu' in navigator` check,
 * for the same reason as the chat: the engine import chain touches WebGPU
 * globals, and the entrance must keep rendering where it cannot run.
 */

import type { ModelSpec } from './compiler/model-spec.js'
import type { MascotHandle } from './mascot.js'
import { mascotPalette } from './mascot.js'
import { modelBranding, specWithCtx } from './zero-tvm/model-registry.js'
import { bootChatEngine, showBootError } from './zero-tvm/chat-flow.js'
import { LANE_SIGIL, laneOf, loreOf } from './landing-lore.js'
import type { KevAnswer, KevQuestion } from './zero-tvm/kev.js'

export interface EnterDecideOptions {
  /** The .cs-root the select screen rendered into. */
  root: HTMLElement
  spec: ModelSpec
  /** ?model= registry param. */
  param: string
  /** Context build in tokens (0 = the compiled default). */
  ctxTokens?: number
  /** The stage mascot — already showing this character. */
  mascot: MascotHandle | null
}

/** Record 1 of tests/fixtures/kev-records.json, transcribed into the
 *  TypeSafe request shape kev.ts takes. The fixture file is the source of
 *  truth; this copy is the demo default, and the comment is the link. */
const DEFAULT_STATE =
  'I was charged twice for the same order and nobody answers my emails. I want my money back now.'
const DEFAULT_QUESTIONS: KevQuestion[] = [
  {
    type: 'choice',
    instructions: 'Which team should handle this ticket?',
    criteria: {
      billing: 'Charges, refunds, invoices',
      'technical support': 'Bugs, outages, login problems',
      sales: null,
      'account management': null,
    },
  },
  {
    type: 'choice',
    instructions: 'Is the customer asking for a refund?',
    criteria: { no: null, yes: null },
  },
  {
    type: 'choice',
    instructions: 'How positive is the sentiment of this message?',
    criteria: { 'very negative': null, negative: null, neutral: null, positive: null, 'very positive': null },
  },
]

const TA_STYLE =
  'width:100%;box-sizing:border-box;background:#17171c;color:inherit;border:1px solid #3a3a44;'
  + 'border-radius:8px;padding:9px 12px;font:12px/1.5 ui-monospace,monospace;resize:vertical'

export function panelMarkup(
  spec: ModelSpec,
  brand: ReturnType<typeof modelBranding>,
  buildLabel: string,
): string {
  const sigil = LANE_SIGIL[laneOf(spec)] ?? ''
  const detail = `${brand.sizeLabel} · cached after first load — next visit starts in seconds`
  const note = buildLabel !== brand.params ? `Build: ${buildLabel}` : (brand.ramNote ?? '')
  return `
    <div class="cs-chat-head">
      <span class="cs-chat-sigil" aria-hidden="true">${sigil}</span>
      <div class="cs-chat-id">
        <b>${brand.name}</b>
        <i>${buildLabel}</i>
      </div>
      <span class="badge" id="badge"><span class="dot"></span><span id="badge-text">Summoning</span></span>
      <a class="cs-chat-tool" id="cs-roster-link" href="/" title="Back to the character select">⟨ Roster</a>
    </div>
    <div class="cs-boot" id="progress-wrap">
      <div class="cs-boot-title" id="loading-title">Summoning ${brand.name}</div>
      <div class="cs-boot-status" id="progress-status" aria-live="polite">Preparing…</div>
      <div class="cs-boot-track"><i id="progress-bar"></i></div>
      <div class="cs-boot-detail" id="progress-detail">${detail}</div>
      ${note ? `<div class="cs-boot-note">${note}</div>` : ''}
      <details class="cs-boot-log"><summary>Rite log</summary><pre id="progress-log"></pre></details>
      <div class="cs-boot-error" id="loading-error"></div>
    </div>
    <main class="chat-main" id="decide-main" hidden>
      <div class="chat-inner">
        <div class="welcome cs-welcome" id="decide-welcome">
          <div class="cs-welcome-title">Ask ${brand.name} to decide</div>
          <div class="cs-welcome-lore">${loreOf(spec)} One state, any number of questions — every branch runs in as few passes as fit.</div>
        </div>
        <label for="kev-state" style="display:block;font-size:12px;color:#9a958c;margin:10px 0 4px">State</label>
        <textarea id="kev-state" rows="3" style="${TA_STYLE}"></textarea>
        <label for="kev-questions" style="display:block;font-size:12px;color:#9a958c;margin:10px 0 4px">Questions (TypeSafe JSON)</label>
        <textarea id="kev-questions" rows="12" style="${TA_STYLE}"></textarea>
        <div style="display:flex;gap:8px;align-items:center;margin:10px 0;flex-wrap:wrap">
          <button class="cs-chat-tool" id="kev-go" type="button" disabled>Decide →</button>
          <span class="cs-chat-stats" id="kev-stats" aria-live="polite"></span>
        </div>
        <div id="kev-out"></div>
      </div>
    </main>`
}

function barRow(label: string, p: number, winner: boolean): string {
  const pct = (p * 100).toFixed(1)
  return `<div style="display:grid;grid-template-columns:minmax(120px,1fr) 90px 52px;gap:8px;align-items:center;margin:3px 0;font-size:12px">`
    + `<span style="${winner ? 'font-weight:700' : ''}">${label}</span>`
    + `<span style="background:#26262e;border-radius:4px;height:8px;position:relative">`
    + `<i style="position:absolute;inset:0 auto 0 0;width:${pct}%;background:var(--accent);border-radius:4px"></i></span>`
    + `<span style="text-align:right;font-variant-numeric:tabular-nums">${pct}%</span></div>`
}

function answerHtml(a: KevAnswer): string {
  if (a.type === 'noul') return `<div>P(yes) = ${a.noul.toFixed(3)}</div>`
  const entries = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1])
  const best = entries[0]?.[0] ?? ''
  const bars = entries.map(([k, p]) => barRow(k, p, k === best)).join('')
  const head = a.type === 'choice'
    ? `<b>${best}</b> <span style="color:#9a958c">confidence ${a.confidence.toFixed(2)}</span>`
    : `<b>score ${a.score.toFixed(2)}</b> <span style="color:#9a958c">confidence ${a.confidence.toFixed(2)}</span>`
  return `<div style="margin:2px 0 8px">${head}</div>${bars}`
}

export async function enterDecide(opts: EnterDecideOptions): Promise<void> {
  const { root, mascot } = opts
  const spec = opts.ctxTokens ? specWithCtx(opts.spec, opts.ctxTokens) : opts.spec
  const brand = modelBranding(spec)
  const buildLabel = [
    brand.params,
    spec.maxContext !== opts.spec.maxContext ? `${Math.round(spec.maxContext / 1024)}k ctx` : '',
  ].filter(Boolean).join(' · ')

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
  panel.setAttribute('aria-label', `Decide with ${brand.name}`)
  panel.innerHTML = panelMarkup(spec, brand, buildLabel)
  root.appendChild(panel)
  root.classList.add('cs-deciding')
  panel.tabIndex = -1
  panel.focus()

  // Leaving is a fresh page — same contract as the chat.
  panel.querySelector<HTMLAnchorElement>('#cs-roster-link')?.addEventListener('click', (e) => {
    e.preventDefault()
    location.reload()
  })

  mascot?.setMood('thinking')
  const boot = await bootChatEngine({
    spec,
    search: location.search,
    onDeviceLost: (info) => {
      showBootError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
  })
  if (!boot.ok) {
    mascot?.setMood('idle')
    root.classList.add('cs-boot-failed')
    const title = panel.querySelector('#loading-title')
    if (title) title.textContent = 'The summoning failed'
    showBootError(boot.reason, () => {
      panel.remove()
      root.classList.remove('cs-deciding', 'cs-boot-failed')
      void enterDecide(opts)
    })
    return
  }
  mascot?.setMood('idle')
  root.classList.add('cs-ready')
  panel.querySelector('#progress-wrap')?.setAttribute('hidden', '')
  panel.querySelector<HTMLElement>('#decide-main')?.removeAttribute('hidden')

  // The pointer head ships beside the weights (kev_head.safetensors, written
  // by scripts/kev-merge.py) and loads from the same base the weights came
  // from — dev mirror or HF CDN — so a guest that copied the weights decides
  // with the same calibration. Its fitted temperature is shown, not hidden:
  // it is part of the score.
  const { loadKevHead, createKev } = await import('./zero-tvm/kev.js')
  const { resolveModelBase } = await import('./zero-tvm/weight-loader.js')
  const head = await loadKevHead(await resolveModelBase(spec))
  const kev = createKev(boot.engine, boot.tokenizer, head)

  const stateEl = panel.querySelector<HTMLTextAreaElement>('#kev-state')!
  const qEl = panel.querySelector<HTMLTextAreaElement>('#kev-questions')!
  const go = panel.querySelector<HTMLButtonElement>('#kev-go')!
  const stats = panel.querySelector<HTMLElement>('#kev-stats')!
  const out = panel.querySelector<HTMLElement>('#kev-out')!
  stateEl.value = DEFAULT_STATE
  qEl.value = JSON.stringify(DEFAULT_QUESTIONS, null, 1)
  // Ready only now: the button exists from mount (so its place never shifts),
  // but clicking it before the engine and head are resident would silently do
  // nothing — the same "control that is not a control" the gate was fixed for.
  go.disabled = false

  go.addEventListener('click', () => {
    void (async () => {
      let questions: KevQuestion[]
      try {
        questions = JSON.parse(qEl.value) as KevQuestion[]
        if (!Array.isArray(questions) || questions.length === 0) throw new Error('need a non-empty array')
      } catch (e) {
        stats.textContent = `questions JSON: ${(e as Error).message}`
        return
      }
      const state = stateEl.value.trim()
      if (!state) { stats.textContent = 'state is empty'; return }
      go.disabled = true
      mascot?.setMood('thinking')
      stats.textContent = 'deciding…'
      try {
        const stateIds = boot.tokenizer.encode(state).length
        const t0 = performance.now()
        const answers = await kev.decide(state, questions)
        const ms = performance.now() - t0
        stats.textContent =
          `${questions.length} question${questions.length === 1 ? '' : 's'} in ${ms.toFixed(0)} ms`
          + ` · state ${stateIds} tokens · head T=${head.temperature}`
        out.innerHTML = answers.map((a, i) =>
          `<div style="border-top:1px solid #2a2a32;padding:8px 0">`
          + `<div style="font-size:12px;color:#9a958c;margin-bottom:4px">Q${i + 1}</div>${answerHtml(a)}</div>`,
        ).join('')
      } catch (e) {
        stats.textContent = `decide failed: ${(e as Error).message}`
      } finally {
        mascot?.setMood('idle')
        go.disabled = false
      }
    })()
  })
}
