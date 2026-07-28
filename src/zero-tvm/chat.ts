/**
 * ZERO-TVM CHAT — Phi-3 inference. No WebLLM. No TVM.
 *
 * Thin page module: DOM chat state + boot wiring. The decode engine itself
 * lives in engine-core.ts (buildDecodeEngine) — this page boots it through
 * loading-ui.ts's bootEngine with the throughput configuration: fused
 * QKV+RoPE+KV-append, URL-flag shader variants (variants.ts), opt-in int8 KV
 * (?kv8=1), and the pipelined generate with a readback ring.
 *
 * The devtools bench harnesses (window.bench / benchBatched / specSim) are
 * wired by bench-console.ts; the streaming Markdown renderer by markdown.ts.
 */

import { WEIGHTS_OPFS_DIR } from './weight-loader.js'
import { buildChatPrompt } from './tokenizer.js'
import {
  allocKVPages,
  allocKVPagesInt8,
  buildDecodeEngine,
} from './engine-core.js'
import { parseVariantFlags } from './variants.js'
import { bootEngine, setBadge } from './loading-ui.js'
import { renderMarkdown, wireCopyButton, ICON_REFRESH } from './markdown.js'
import { installBenchConsole } from './bench-console.js'

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

function setStats(text: string) {
  const el = $('stats')
  if (el) el.textContent = text
}

function hideLoadingOverlay() { $('loading-overlay')?.classList.remove('active') }
function showLoadingError(msg: string) {
  const el = $('loading-error')
  if (!el) return
  el.textContent = msg
  el.classList.add('visible')
}

function hideWelcome() {
  const w = $('welcome')
  if (w && !w.classList.contains('hidden')) w.classList.add('hidden')
}

function autoGrow(ta: HTMLTextAreaElement) {
  ta.style.height = 'auto'
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
}

function scrollToBottom(force = false) {
  const main = $('chat-main')
  if (!main) return
  const distance = main.scrollHeight - main.scrollTop - main.clientHeight
  if (force || distance < 120) main.scrollTop = main.scrollHeight
}

let scrollFabWired = false
function wireScrollFab() {
  if (scrollFabWired) return
  const main = $('chat-main'); const fab = $('scroll-fab')
  if (!main || !fab) return
  scrollFabWired = true
  const update = () => {
    const distance = main.scrollHeight - main.scrollTop - main.clientHeight
    fab.classList.toggle('visible', distance > 140)
  }
  main.addEventListener('scroll', update, { passive: true })
  fab.addEventListener('click', () => scrollToBottom(true))
  update()
}

// ---------- Message construction ----------

function addUserMsg(text: string): void {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg user'
  const bubble = document.createElement('div')
  bubble.className = 'bubble'
  bubble.textContent = text
  wrap.appendChild(bubble)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)
}

interface AiMsgHandle {
  wrap: HTMLElement
  body: HTMLElement
  cursor: HTMLElement
  /** Replace the role tag with a thinking indicator (before first token). */
  showThinking(): void
  /** Re-render the full text (Markdown) into body on next animation frame. */
  render(text: string): void
  /** Remove cursor, add copy + stats + regenerate actions, stop streaming. */
  finish(opts: { fullText: string; tokens: number; tokPerS: number; onRegenerate?: () => void }): void
}

function addAiMsg(): AiMsgHandle {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg ai'
  wrap.innerHTML = `
    <div class="role">
      <span class="role-dot">Z</span>
      <span class="role-name">Phi-3-mini</span>
      <span class="model-tag">q4f16_1</span>
    </div>
    <div class="body"></div>
    <div class="actions"></div>
  `
  const body = wrap.querySelector('.body') as HTMLElement
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  body.appendChild(cursor)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)

  let rafPending = false
  let latestText = ''
  const doRender = () => {
    rafPending = false
    const frag = renderMarkdown(latestText)
    body.replaceChildren(frag, cursor)
    scrollToBottom(false)
  }

  return {
    wrap, body, cursor,
    showThinking() {
      // Replace the empty body (just cursor) with a thinking indicator.
      body.replaceChildren()
      const t = document.createElement('span')
      t.className = 'thinking'
      t.innerHTML = '<span></span><span></span><span></span>'
      body.appendChild(t)
    },
    render(text: string) {
      latestText = text
      if (!rafPending) {
        rafPending = true
        requestAnimationFrame(doRender)
      }
    },
    finish({ fullText, tokens, tokPerS, onRegenerate }) {
      // Ensure final markdown render without cursor.
      const frag = renderMarkdown(fullText)
      body.replaceChildren(frag)
      const actions = wrap.querySelector('.actions') as HTMLElement | null
      if (!actions) return
      actions.replaceChildren()

      const copy = document.createElement('button')
      copy.type = 'button'
      copy.className = 'action-btn'
      wireCopyButton(copy, () => fullText)
      actions.appendChild(copy)

      if (onRegenerate) {
        const regen = document.createElement('button')
        regen.type = 'button'
        regen.className = 'action-btn'
        regen.innerHTML = `${ICON_REFRESH}<span>Regenerate</span>`
        regen.addEventListener('click', onRegenerate)
        actions.appendChild(regen)
      }

      const stats = document.createElement('span')
      stats.className = 'msg-stats'
      stats.textContent = `${tokens} tok · ${tokPerS.toFixed(1)} tok/s`
      actions.appendChild(stats)
    },
  }
}

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

async function hasWeightsCached(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return false
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle(WEIGHTS_OPFS_DIR)
    // ndarray-cache.json is the smallest sentinel — its presence means at
    // least the manifest has been fetched (and shards followed in-session).
    await dir.getFileHandle('ndarray-cache.json')
    return true
  } catch {
    return false
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
        <div class="title" style="font-size:1.1rem;font-weight:600;color:#ccc">Phi-3-mini, in your browser</div>
        <div class="desc" style="font-size:0.85rem;color:#888;max-width:440px;line-height:1.6">
          First load downloads ~1.8 GB of Phi-3-mini-q4f16_1 weights and caches them
          locally (OPFS). Every subsequent visit — including the WebLLM comparison
          page — is instant; weights are served from the same shared browser cache.
        </div>
        <button id="start-btn" class="start-btn" style="background:#10b981;color:#000;font-weight:600;font-size:0.95rem;padding:0.7rem 2rem;border:none;border-radius:8px;cursor:pointer">Download &amp; Start</button>
        <div class="start-hint" style="font-size:0.68rem;color:#444">Phi-3-mini-4k-instruct-q4f16_1 · ~1.8 GB · cached after first load</div>
      </div>`
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
  log('Zero-TVM Phi-3 — No WebLLM, No TVM')
  log('10 hand-written WGSL kernel roles across 27 kernels (18 files + 9 generated)')
  log('')

  const boot = await bootEngine({
    // Subgroups (when the adapter has them) for the _sg shader variants;
    // timestamp-query for profileStep / `await bench(0, 0, true)`.
    optionalFeatures: ['subgroups' as GPUFeatureName, 'timestamp-query' as GPUFeatureName],
    probeSubgroups: true,
    // Chat also surfaces device loss in the loading overlay's error panel.
    onDeviceLost: (info) => {
      showLoadingError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
    buildEngine: ({ device, weights, sgSizeOk }) => {
      const flags = parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      // KV cache: int8 halves memory at the cost of one extra dispatch per layer.
      log(flags.int8KV ? 'Allocating int8 KV cache (~800 MB)…' : 'Allocating KV cache (~1.6 GB)…')
      const kv = flags.int8KV ? allocKVPagesInt8(device) : allocKVPages(device)
      log('Building decode engine…')
      return buildDecodeEngine(device, weights, kv, {
        variants: flags,
        fused: true,
        int8KV: flags.int8KV,
      })
    },
    // Pipeline warmup: one throwaway single-token generation behind the
    // progress bar so the first chat message streams at steady-state speed.
    // The KV slots it writes are overwritten by the first real turn's
    // prefill (chat always prefills from 0).
    warmup: async (engine, tokenizer) => {
      const warmupIds = buildChatPrompt([{ role: 'user', content: 'Hi.' }], tokenizer)
      await engine.generatePipelined(warmupIds, 1, () => {})
    },
  })
  if (!boot.ok) {
    showLoadingError(boot.reason)
    log(`ERROR: ${boot.reason}`)
    return
  }
  const { tokenizer, engine } = boot
  hideLoadingOverlay()

  // ─────────── Wire chat UI ───────────
  const form     = $('composer') as HTMLFormElement | null
  const inp      = $('inp') as HTMLTextAreaElement | null
  const sendBtn  = $('btn') as HTMLButtonElement | null
  const stopBtn  = $('stop-btn') as HTMLButtonElement | null
  const newBtn   = $('new-chat-btn') as HTMLButtonElement | null
  const suggest  = $('suggest-grid')
  const ctxHint  = $('ctx-hint')
  const welcome  = $('welcome')
  if (!form || !inp || !sendBtn || !stopBtn) {
    console.error('[zero-tvm] chat UI elements missing'); return
  }

  const SYSTEM_PROMPT = 'You are a helpful, concise assistant. Use Markdown (numbered lists, **bold**, and fenced ```code``` blocks with a language tag) when it clarifies the answer.'
  const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: SYSTEM_PROMPT },
  ]
  let generating = false
  let stopRequested = false

  function updateSendEnabled() {
    if (!inp || !sendBtn) return
    sendBtn.disabled = generating || inp.value.trim().length === 0
  }
  function setBusy(busy: boolean) {
    if (!inp || !sendBtn || !stopBtn || !newBtn) return
    generating = busy
    sendBtn.hidden = busy
    stopBtn.hidden = !busy
    stopBtn.disabled = !busy
    inp.disabled = busy
    newBtn.disabled = busy
    inp.placeholder = busy ? 'Generating…' : 'Ask Phi-3 anything…'
  }

  function updateCtxHint(nTokens?: number) {
    if (!ctxHint) return
    // True ceiling is the KV page table (MAX_PAGES × PAGE_SIZE = 4112), not
    // the model's nominal 4096 — derived from the same constant the engine uses.
    if (!nTokens) {
      ctxHint.textContent = 'Zero TVM · 10 WGSL kernels · 228 dispatches/token'
    } else if (nTokens >= engine.maxContext) {
      ctxHint.textContent = `Context full — ${engine.maxContext} / ${engine.maxContext} tokens · start a new chat`
    } else {
      const pct = Math.round((nTokens / engine.maxContext) * 100)
      ctxHint.textContent = `Context ${nTokens} / ${engine.maxContext} tokens (${pct}%) · 228 dispatches/token`
    }
  }

  function resetChat() {
    if (generating) return
    history.length = 0
    history.push({ role: 'system', content: SYSTEM_PROMPT })
    const msgs = $('messages')
    if (msgs) msgs.replaceChildren()
    welcome?.classList.remove('hidden')
    setStats('')
    setBadge('Ready', 'ready')   // clears a lingering "Context full" state
    updateCtxHint()
    inp?.focus()
  }

  // Initial enabled state
  inp.disabled = false
  inp.placeholder = 'Ask Phi-3 anything…'
  newBtn && (newBtn.disabled = false)
  autoGrow(inp)
  updateSendEnabled()
  updateCtxHint()
  wireScrollFab()
  inp.focus()

  inp.addEventListener('input', () => { autoGrow(inp); updateSendEnabled() })
  inp.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!sendBtn.disabled) form.requestSubmit()
    }
  })
  form.addEventListener('submit', (e) => {
    e.preventDefault()
    if (!sendBtn.disabled) void send()
  })
  stopBtn.addEventListener('click', () => {
    stopRequested = true
    stopBtn.disabled = true
  })
  newBtn?.addEventListener('click', resetChat)
  suggest?.addEventListener('click', (e) => {
    const target = (e.target as HTMLElement).closest('.suggest') as HTMLElement | null
    if (!target) return
    const prompt = target.dataset.prompt
    if (!prompt) return
    inp.value = prompt
    autoGrow(inp); updateSendEnabled()
    form.requestSubmit()
  })

  async function runGeneration(ai: AiMsgHandle, promptIds: number[]): Promise<string> {
    log(`Prompt: ${promptIds.length} tokens`)
    updateCtxHint(promptIds.length)
    ai.showThinking()
    const t0 = performance.now()
    let count = 0
    let firstToken = true
    const allIds: number[] = []
    let fullResponse = ''

    try {
      // Cooperative cancellation: the engine polls the flag between pipeline
      // submissions and drains its in-flight readbacks before returning.
      await engine.generatePipelined(promptIds, 500, (id) => {
        if (firstToken) {
          // Swap thinking indicator for the real body on first token.
          ai.body.replaceChildren(ai.cursor)
          firstToken = false
        }
        count++
        allIds.push(id)
        fullResponse = tokenizer.decode(allIds)
        ai.render(fullResponse)
        const elapsed = (performance.now() - t0) / 1000
        setStats(`${count} tok · ${(count / elapsed).toFixed(1)} tok/s`)
      }, () => stopRequested)
    } catch (e) {
      fullResponse += `\n\n_[Error: ${e}]_`
      log(`Error: ${e}`)
    }
    const totalSec = (performance.now() - t0) / 1000
    const tokPerS = count / Math.max(totalSec, 0.001)
    ai.finish({
      fullText: fullResponse,
      tokens: count,
      tokPerS,
      onRegenerate: () => { void regenerate() },
    })
    // Reflect the true KV position count (prompt + generated) in the context hint.
    updateCtxHint(promptIds.length + count)
    if (promptIds.length + count >= engine.maxContext) {
      setBadge('Context full', 'error')
      log(`Context window full (${engine.maxContext} tokens) — start a new chat to continue.`)
    }
    return fullResponse
  }

  /** Shared bracketing around a single decode turn: busy flags, prefill
   * tokenization, run, commit response to history, re-enable UI. */
  async function runTurn(): Promise<void> {
    setBusy(true); stopRequested = false
    const ai = addAiMsg()
    const promptIds = buildChatPrompt(history, tokenizer)
    // Every turn re-prefills the whole history, so once it no longer fits the
    // KV window the only way forward is a fresh conversation. Refuse cleanly
    // instead of letting prefill corrupt the cache.
    if (promptIds.length >= engine.maxContext) {
      const msg = `Context full — the conversation (${promptIds.length} tokens) exceeds the ${engine.maxContext}-token window. Start a new chat.`
      ai.finish({ fullText: `_${msg}_`, tokens: 0, tokPerS: 0 })
      setBadge('Context full', 'error')
      updateCtxHint(promptIds.length)
      log(msg)
      setBusy(false); stopRequested = false
      updateSendEnabled()
      return
    }
    const fullResponse = await runGeneration(ai, promptIds)
    history.push({ role: 'assistant', content: fullResponse })
    setBusy(false); stopRequested = false
    updateSendEnabled()
    inp?.focus()
  }

  async function send(): Promise<void> {
    if (generating || !inp) return
    const text = inp.value.trim()
    if (!text) return
    inp.value = ''
    autoGrow(inp)
    addUserMsg(text)
    history.push({ role: 'user', content: text })
    await runTurn()
  }

  async function regenerate(): Promise<void> {
    if (generating) return
    if (history.length === 0 || history[history.length - 1].role !== 'assistant') return
    history.pop()
    $('messages')?.querySelector('.msg.ai:last-child')?.remove()
    await runTurn()
  }

  // Devtools bench harnesses (window.bench / benchBatched / specSim).
  installBenchConsole({
    engine,
    tokenizer,
    isBusy: () => generating,
    setBusy,
    onIdle: updateSendEnabled,
  })
}

async function boot(): Promise<void> {
  // Register the weight-cache SW first so it intercepts the very first
  // network fetch (from bootEngine's loadWeights call) — and any future
  // fetch from WebLLM in compiler-chat / webllm-bench.
  await registerWeightsSW()
  // Skip the gate when weights are already in OPFS — returning visitors
  // (or visitors who started from compiler-chat) should boot straight in.
  // Remove the static dialog so no hidden #start-btn lingers in the DOM
  // (the gate e2e asserts "#start-btn present iff the gate is showing").
  if (await hasWeightsCached()) {
    document.getElementById('start-dialog')?.remove()
  } else {
    await showDownloadGate()
  }
  await main()
}
boot().catch((e) => { console.error(e); log(`FATAL: ${e}`) })
