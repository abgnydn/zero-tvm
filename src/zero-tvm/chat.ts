/**
 * ZERO-TVM CHAT — the standalone chat page. No WebLLM. No TVM.
 *
 * Thin page module: gate/branding/switcher wiring for zero-tvm.html. The
 * decode engine lives in engine-core.ts (buildDecodeEngine); the boot
 * configuration and the turn loop live in chat-flow.ts, SHARED with the
 * landing entrance's in-place chat — one decode loop, one turn loop, so the
 * two surfaces cannot drift. This file owns only what is specific to this
 * page: the pre-download gate, the static-markup rebranding, the model
 * switcher, the sampling popover, and the header/welcome mascots.
 */

// cache-probe is NOT imported here. It pulls in the loaders, which read
// GPUBufferUsage at module scope, and a static import evaluates that chain
// before a single line of this file runs — so on a browser without WebGPU the
// module threw a ReferenceError and the page sat on its pre-activated
// "Preparing Zero-TVM / Starting… 0%" overlay forever, with no gate, no error
// and no mention of what would work. That is the site's primary CTA. Every
// other surface here (landing.ts, share.ts) already imports it dynamically for
// exactly this reason; the product page was the one that did not.
import { specFromSearch, modelBranding } from './model-select.js'
import { initModelSwitcher } from './model-switcher.js'
import { mountMascot, mascotPalette } from '../mascot.js'
import { initSamplingControls } from './sampling-ui.js'
import { setBadge } from './loading-ui.js'
import { bootChatEngine, wireChatSurface, showBootError } from './chat-flow.js'
import { setChatIdentity, quantTagFor } from './chat-ui.js'

// ============================================================
// Active model — ?model=qwen3 selects Qwen3-4B, default is Phi-3.
// ============================================================

const SPEC = specFromSearch(location.search)
const BRAND = modelBranding(SPEC)
// Memory mode (?pool=): expert slots held resident per MoE layer, the rest
// streamed from OPFS on demand. Saves RAM, not download — the full checkpoint
// still lands in the cache. 'half'/'quarter' are fractions of the expert
// count; a bare number is slots. Non-MoE specs ignore it.
function poolFromSearch(): number {
  if (!SPEC.moe) return 0
  const v = new URLSearchParams(location.search).get('pool')
  if (!v) return 0
  const E = SPEC.moe.experts
  const slots = v === 'half' ? Math.round(E / 2) : v === 'quarter' ? Math.round(E / 4) : Number(v)
  return Number.isFinite(slots) && slots > 0 ? slots : 0
}
const POOL_SLOTS_UI = poolFromSearch()

/** The gate no longer asks about memory — the landing's character screen
 *  already did, and ?pool= carries the answer. The gate just SAYS what was
 *  chosen, from the same registry row the landing rendered. */
function noteMemoryMode(): void {
  if (!POOL_SLOTS_UI) return
  const mode = (BRAND.poolModes ?? []).find((m2) => m2.slots === POOL_SLOTS_UI)
  const notes = document.querySelector('#start-dialog .dialog-notes ul')
  if (!notes) return
  const li = document.createElement('li')
  li.textContent = `Memory build: ${mode ? mode.label : `${POOL_SLOTS_UI} expert slots`}`
    + (mode?.note ? ` — ${mode.note}` : '')
  li.style.color = 'var(--warn, #d9a441)'
  notes.prepend(li)
}
// Quant label for the message header + gate copy — derived in chat-ui.ts so
// the remote guest page shows the same truth.
const QUANT_TAG = quantTagFor(SPEC)
setChatIdentity(BRAND.name, QUANT_TAG)
// No dispatches-per-token figure in the chrome. The formula that used to live
// here was a hand-derived constant per spec family, and pre-publish review
// reconstructed the real counts from the recorders: it was wrong for six of
// the ten shipped models — it modelled the MoE block as one dispatch where the
// engine runs seven, the affine dense FFN as one where it runs three, and it
// ignored split-K attention, which has been default-on since 2026-07-27 (so
// even Phi-3's celebrated 228 is actually 260 on the path everyone runs).
// A number that needs a per-family derivation to stay true is a number that
// rots; the kernel-roles claim is the one that does not.

// ============================================================
// UI helpers
// ============================================================

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T | null

function log(msg: string) {
  const el = document.getElementById('progress-log') as HTMLPreElement | null
  if (!el) { console.log(msg); return }
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

function hideLoadingOverlay() { $('loading-overlay')?.classList.remove('active') }

// Message surface (bubbles, streaming markdown, actions, scroll) lives in
// chat-ui.ts, shared verbatim with the remote-guest page (share.ts); the boot
// configuration and turn loop live in chat-flow.ts, shared with the landing.

// ============================================================
// Boot helpers — SW registration, cache probe, download-gate UI
// (replaces the legacy <dialog>-based bootstrap; SW gives a single
// shared weight cache across /zero-tvm chat + compiler-chat + webllm-bench)
// ============================================================

async function registerWeightsSW(): Promise<void> {
  if (!('serviceWorker' in navigator)) return
  try {
    await navigator.serviceWorker.register('/weights-cache-sw.js', { scope: '/' })
    // Wait for the active SW so first weight fetch is intercepted (and
    // shared with WebLLM in compiler-chat.html / webllm-bench.html).
    await navigator.serviceWorker.ready
  } catch (e) {
    // Best-effort — chat still works, just no shared cache.
    console.warn('[zero-tvm] weight-cache SW registration failed:', e)
  }
}

function showDownloadGate(): Promise<void> {
  return new Promise((resolve) => {
    // Prefer the page's own styled <dialog id="start-dialog"> (zero-tvm.html).
    // Its #start-btn must leave the DOM entirely once the gate is passed —
    // the e2e contract is "#start-btn present iff the gate is showing".
    const staticDialog = document.getElementById('start-dialog') as HTMLDialogElement | null
    if (staticDialog?.showModal) {
      setBadge('Waiting')
      const btn = staticDialog.querySelector<HTMLButtonElement>('#start-btn')
      btn?.addEventListener('click', () => {
        staticDialog.close()
        staticDialog.remove()
        resolve()
      }, { once: true })
      staticDialog.showModal()
      return
    }
    const chat = document.getElementById('chat')
    if (!chat) { resolve(); return }
    // Same #start-screen / #start-btn IDs as compiler-chat.html so existing
    // e2e harness (which clicks #start-btn) works on both pages uniformly.
    chat.innerHTML = `
      <div id="start-screen" class="start-screen" style="display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:60vh;gap:1.25rem;text-align:center;padding:2rem">
        <div class="title" style="font-size:1.1rem;font-weight:600;color:#ccc"></div>
        <div class="desc" style="font-size:0.85rem;color:#888;max-width:440px;line-height:1.6">
          First load downloads the ${QUANT_TAG} weights and caches them
          locally (OPFS). Every subsequent visit is instant; weights are
          served from the on-device cache.
        </div>
        <button id="start-btn" class="start-btn" style="background:#10b981;color:#000;font-weight:600;font-size:0.95rem;padding:0.7rem 2rem;border:none;border-radius:8px;cursor:pointer">Download &amp; Start</button>
        <div class="start-hint" style="font-size:0.68rem;color:#444"></div>
      </div>`
    const titleEl = chat.querySelector('#start-screen .title') as HTMLElement | null
    if (titleEl) titleEl.textContent = `${BRAND.name}, in your browser`
    const hintEl = chat.querySelector('#start-screen .start-hint') as HTMLElement | null
    if (hintEl) hintEl.textContent = `${SPEC.hfRepo.split('/')[1]} · ${BRAND.sizeLabel} · cached after first load`
    setBadge('Waiting')
    const btn = document.getElementById('start-btn') as HTMLButtonElement | null
    btn?.addEventListener('click', () => {
      btn.disabled = true
      btn.textContent = 'Starting...'
      chat.innerHTML = ''
      resolve()
    }, { once: true })
  })
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  setBadge('Initializing…', 'loading')
  log(`Zero-TVM ${BRAND.name} — No WebLLM, No TVM`)
  log('10 hand-written WGSL kernel roles')
  log('')

  const boot = await bootChatEngine({
    spec: SPEC,
    poolSlots: POOL_SLOTS_UI,
    search: location.search,
    // Chat also surfaces device loss in the loading overlay's error panel.
    onDeviceLost: (info) => {
      showBootError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
  })
  if (!boot.ok) {
    // Retry re-runs the full boot without a reload — the weight loader's
    // OPFS tier skips every shard that already completed.
    showBootError(boot.reason, () => { void main() })
    log(`ERROR: ${boot.reason}`)
    return
  }
  const { tokenizer, engine } = boot
  // The sampling popover is mounted before this (the header is interactive
  // during the download), so hand it the engine once there is one.
  liveEngine = engine
  hideLoadingOverlay()

  wireChatSurface({
    spec: SPEC,
    tokenizer,
    engine,
    // One pulse per real token — while talking the pulse drives the header
    // character's MOUTH, so the flap envelope IS the measured cadence.
    onToken: () => headerMascot?.pulse(),
    onPhase: (p) => headerMascot?.setMood(
      p === 'generating' ? 'talking' : p === 'thinking' ? 'thinking' : 'idle'),
    // /roster returns to the entrance, where the character select lives.
    commands: { roster: () => { location.href = '/' } },
    log,
  })
}

/** Rewrite the static Phi-3 markup (gate dialog, header, welcome) for the
 * active model. The HTML ships with Phi-3 copy, so this is a no-op there. */
function applyModelBranding(): void {
  const set = (sel: string, text: string) => {
    const el = document.querySelector(sel) as HTMLElement | null
    if (el) el.textContent = text
  }
  // The welcome pills are written for EVERY model including Phi-3, unlike the
  // rest of this function. They were static markup, so the page told a
  // Llama-3.2-1B visitor it was running "q4f16_1 · 3.8 B" with "4 K context" —
  // Phi-3's numbers, on a 1B MLX model with 8x the context. Deriving them even
  // when they happen to be right is the only way they cannot drift again.
  set('#welcome-quant', `${quantTagFor(SPEC)} · ${BRAND.params}`)
  set('#welcome-ctx', `${Math.round(SPEC.maxContext / 1024)} K context`)
  // The fourth suggestion asked the model where "a 3.8B model like Phi-3-mini"
  // falls short, on every model. A 1B answering for a 3.8B is a worse answer
  // AND a wrong caption, so the size and the name come off the same record the
  // rest of the page reads. The leading token of `params` is the size in every
  // row ('1B dense', '35B-A3B MoE · 3-bit experts').
  {
    const size = BRAND.params.split(/[\s·]/)[0]
    const btn = document.querySelector('.suggest[data-prompt*="fall short"]') as HTMLElement | null
    if (btn) {
      btn.dataset.prompt = `List four kinds of questions where a ${size} model like ${BRAND.name} `
        + 'is likely to fall short compared to a frontier LLM, with one-sentence reasons.'
      set('.suggest[data-prompt*="fall short"] .suggest-hint', `Where a ${size} model falls short`)
    }
  }
  mountChatMascots()
  // The gate's three stats are derived for EVERY model, Phi-3 included. The
  // early return used to sit above this block on the theory that the static
  // markup is already Phi-3's — but the markup carried '60+ t/s' against the
  // registry's ~70, and shipped the context slot as a bare em-dash. Phi-3 is
  // the default, so that was what most visitors saw, and '60+' is close enough
  // to WebLLM's 59.95 to be the one number on this site that must not be
  // confusable with theirs.
  const statVals = document.querySelectorAll('#start-dialog .dialog-stats .dstat .val')
  if (statVals[0]) statVals[0].textContent = BRAND.sizeLabel   // download size
  if (statVals[1]) {
    // An unmeasured model (qwen36's rateLabel is '') used to render an empty
    // value over an orphaned "M2 Max" caption — honest, but it read as a
    // rendering fault. The landing card hides the row; the gate now agrees.
    if (BRAND.rateLabel) statVals[1].textContent = BRAND.rateLabel  // total tok/s, M2 Max
    else (statVals[1].parentElement as HTMLElement).style.display = 'none'
  }
  // Context is derived from the spec (maxPages x pageSize), like every other
  // figure on this page. The slot used to read a static 'f16 / WebGPU'.
  set('#gate-ctx', `${Math.round(SPEC.maxContext / 1024)}K`)
  // The RAM requirement was data the registry carried and no chat surface
  // showed: a ?model=qwen36 deep link offered a 19.5 GB download without ever
  // saying "needs ~24 GB free RAM". It is the first note now, ahead of the
  // cache line, because it is the one a visitor cannot undo by waiting.
  if (BRAND.ramNote) {
    const notes = document.querySelector('#start-dialog .dialog-notes ul')
    if (notes) {
      const li = document.createElement('li')
      li.textContent = BRAND.ramNote
      li.style.color = 'var(--warn, #d9a441)'
      notes.prepend(li)
    }
  }
  noteMemoryMode()
  if (SPEC.id === 'phi3-mini') return
  document.title = `Zero-TVM Chat — ${BRAND.name}`
  set('#start-dialog .dialog-title', `${BRAND.name} on raw WebGPU`)
  set('.model-info h1', BRAND.name)
  set('#welcome .welcome-title', `Chat with ${BRAND.name}`)
}

/**
 * The model's character on the three surfaces that persist: the header, the
 * empty state, and every assistant reply.
 *
 * The replies get a STILL captured from the welcome canvas rather than a
 * canvas each — a long conversation would otherwise run one animated WebGPU
 * surface per message alongside the engine on the same device. Everything
 * here is additive: no mascot means the lettered tile and the letter avatar
 * stay exactly as they were, which is also what a browser without WebGPU
 * gets (and it cannot run the engine either, so it never reaches this page
 * with a model loaded).
 */
/** The header mascot's handle, kept so generation can drive its heartbeat —
 *  one pulse() per real token, so the bounce IS the measured cadence. */
let headerMascot: import('../mascot.js').MascotHandle | null = null

function mountChatMascots(): void {
  const host = document.querySelector('.model-info')
  if (host && !document.getElementById('header-mascot')) {
    const c = document.createElement('canvas')
    c.id = 'header-mascot'
    c.className = 'header-mascot'
    c.setAttribute('aria-hidden', 'true')
    host.parentElement?.insertBefore(c, host)
    void mountMascot(c, SPEC).then((m) => {
      if (!m) { c.remove(); return }
      headerMascot = m
      // Memory mode leans the figure and pulls the streamed experts' sprites
      // into a far ghost orbit — the pool, drawn from the same number the
      // picker chose.
      if (POOL_SLOTS_UI && SPEC.moe) m.setSpec(SPEC, false, POOL_SLOTS_UI / (SPEC.moe.experts + 1))
    }).catch(() => c.remove())
  }

  const logo = document.getElementById('welcome-logo')
  if (logo && !logo.querySelector('canvas')) {
    const c = document.createElement('canvas')
    c.setAttribute('aria-hidden', 'true')
    logo.appendChild(c)
    logo.classList.add('has-mascot')
    void mountMascot(c, SPEC).then(async (m) => {
      if (!m) { c.remove(); logo.classList.remove('has-mascot'); return }
      const url = await m.snapshot(true)
      document.documentElement.style.setProperty('--mascot-avatar', `url("${url}")`)
      document.documentElement.classList.add('has-mascot-avatar')
    }).catch(() => { c.remove(); logo.classList.remove('has-mascot') })
  }
}

/** Set once the engine exists. The sampling controls read it lazily, so the
 *  popover works during the weight download and simply applies to whatever
 *  engine is there when the next token is produced. */
let liveEngine: import('./engine-core.js').DecodeEngine | null = null

/** Replace the loading overlay with what this browser is missing and why.
 *  Runs before anything GPU-touching is imported. */
function showUnsupported(): void {
  const title = document.getElementById('loading-title')
  const status = document.getElementById('progress-status')
  const detail = document.getElementById('progress-detail')
  if (title) title.textContent = 'This browser cannot run the engine'
  if (status) {
    status.textContent = 'zero-tvm needs WebGPU with shader-f16. Chrome and Edge ship it; '
      + 'Safari from 26. The MoE models additionally need GPU subgroups (Chromium).'
  }
  if (detail) detail.textContent = ''
  document.getElementById('start-dialog')?.remove()
  document.querySelector('.progress-track')?.remove()
}

async function boot(): Promise<void> {
  // BEFORE the gate, not after. The old order was gate → main() → bootEngine()
  // → navigator.gpu check, so a visitor who cannot run this read "~2 GB",
  // pressed "Download & start", and only then learned their browser was never
  // going to work.
  if (!('gpu' in navigator)) { showUnsupported(); return }
  applyModelBranding()
  initModelSwitcher(SPEC)
  // The gate used to show a generic logo and a second model picker. By the
  // time anyone reaches it they have already chosen — on the landing, or via
  // ?model= — so the picker was asking a settled question. It shows THIS
  // model's character instead, and the download it is about to commit to.
  // Failure is silence: a browser without WebGPU still gets the gate, and the
  // canvas simply removes itself.
  // THEME FROM THE CHARACTER. The accent on this page is the model's own lane
  // colour, so the creature on the gate and the buttons around it are the same
  // colour — and that colour says what the architecture is rather than being
  // picked per page. Amber stays the site's colour on the landing, where no
  // single model is in view.
  {
    const { accent, accentHi } = mascotPalette(SPEC)
    const root = document.documentElement.style
    root.setProperty('--accent', accent)
    root.setProperty('--accent-hi', accentHi)
    root.setProperty('--accent-2', accentHi)
    root.setProperty('--accent-dim', `${accent}22`)
    root.setProperty('--accent-tint', `${accent}1f`)
  }
  {
    const canvas = document.getElementById('gate-mascot') as HTMLCanvasElement | null
    if (canvas) {
      void mountMascot(canvas, SPEC).then((m) => {
        if (!m) { canvas.remove(); return }
        // Posed to the build the landing chose (?pool=) — the same lean the
        // character screen showed is the one the gate confirms.
        if (POOL_SLOTS_UI && SPEC.moe) m.setSpec(SPEC, false, POOL_SLOTS_UI / (SPEC.moe.experts + 1))
      }).catch(() => canvas.remove())
    }
  }
  initSamplingControls(() => liveEngine)
  // Register the weight-cache SW first so it intercepts the very first
  // network fetch (from bootEngine's loadWeights call) — and any future
  // fetch from WebLLM in compiler-chat / webllm-bench.
  await registerWeightsSW()
  // Skip the gate when weights are already in OPFS — returning visitors
  // (or visitors who started from compiler-chat) should boot straight in.
  // Remove the static dialog so no hidden #start-btn lingers in the DOM
  // (the gate e2e asserts "#start-btn present iff the gate is showing").
  const { isModelCached } = await import('./cache-probe.js')
  if (await isModelCached(SPEC)) {
    document.getElementById('start-dialog')?.remove()
  } else {
    await showDownloadGate()
  }
  await main()
}
boot().catch((e) => { console.error(e); log(`FATAL: ${e}`) })
