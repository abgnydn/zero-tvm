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

import { opfsDirFor } from './weight-loader.js'
import { specFromSearch, modelBranding, buildChatPromptFor } from './model-select.js'
import { initModelSwitcher } from './model-switcher.js'
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
// Active model — ?model=qwen3 selects Qwen3-4B, default is Phi-3.
// ============================================================

const SPEC = specFromSearch(location.search)
const BRAND = modelBranding(SPEC)
// Dispatches per decode token: Phi-3 runs the fused path (32×7+4 = 228);
// qkNorm specs (Qwen3) run unfused with the fused qk_norm+RoPE+append kernel
// (default ?fuseqk on: 36×8+4 = 292; the reference chain is 10/layer); hybrid
// (Qwen3.5) mixes GDN layers (10 with the fused input projection) and
// gated-attention layers (12): 24×10+8×12+4 = 340.
const GDN_LAYERS = SPEC.layerKinds.filter((k) => k === 'gdn').length
const DISPATCHES_PER_TOKEN = GDN_LAYERS > 0
  ? GDN_LAYERS * 10 + (SPEC.layers - GDN_LAYERS) * 12 + 4
  : SPEC.qkNorm ? SPEC.layers * 8 + 4 : SPEC.layers * 7 + 4

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
function showLoadingError(msg: string, onRetry?: () => void) {
  const el = $('loading-error')
  if (!el) return
  el.textContent = msg
  if (onRetry) {
    // Weight downloads resume from OPFS, so a retry is cheap — no reload
    // needed. Re-runs the whole boot; completed shards come straight from cache.
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.id = 'retry-download-btn'
    btn.textContent = 'Retry download'
    btn.style.cssText = 'display:block;margin-top:0.6rem;padding:0.45rem 1.1rem;background:#10b981;color:#000;font-weight:600;font-size:0.8rem;border:none;border-radius:6px;cursor:pointer'
    btn.addEventListener('click', () => {
      el.classList.remove('visible')
      el.replaceChildren()
      onRetry()
    }, { once: true })
    el.appendChild(btn)
  }
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
  /** Remove cursor, add copy + stats + regenerate actions, stop streaming.
   *  `notice` renders a persistent banner under the reply — used when the
   *  reply was cut short, so it never just stops mid-word in silence. */
  finish(opts: {
    fullText: string
    tokens: number
    tokPerS: number
    onRegenerate?: () => void
    notice?: { text: string; onContinue?: () => void }
  }): void
}

function addAiMsg(): AiMsgHandle {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg ai'
  wrap.innerHTML = `
    <div class="role">
      <span class="role-dot">Z</span>
      <span class="role-name"></span>
      <span class="model-tag">q4f16_1</span>
    </div>
    <div class="body"></div>
    <div class="actions"></div>
  `
  ;(wrap.querySelector('.role-name') as HTMLElement).textContent = BRAND.name
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
    finish({ fullText, tokens, tokPerS, onRegenerate, notice }) {
      // Ensure final markdown render without cursor.
      const frag = renderMarkdown(fullText)
      body.replaceChildren(frag)
      // A resumed reply re-runs finish() on the same message — drop the
      // banner that made it resumable before rebuilding.
      wrap.querySelector('.truncation')?.remove()
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

      // .actions is hover-revealed; the cut-short banner must not be, so it
      // gets its own always-visible row between the body and the actions.
      if (notice) {
        const el = document.createElement('div')
        el.className = 'truncation'
        const label = document.createElement('span')
        label.textContent = notice.text
        el.appendChild(label)
        if (notice.onContinue) {
          const btn = document.createElement('button')
          btn.type = 'button'
          btn.className = 'truncation-btn'
          btn.textContent = 'Continue'
          btn.addEventListener('click', notice.onContinue)
          el.appendChild(btn)
        }
        wrap.insertBefore(el, actions)
      }
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
    const dir = await root.getDirectoryHandle(opfsDirFor(SPEC))
    // The sentinel is the SPEC's own manifest, not a hardcoded name. It was
    // 'ndarray-cache.json' — already wrong for Qwen3.5, whose manifest MLC
    // renamed to tensor-cache.json, so that model re-showed the download gate
    // on every visit no matter what was cached. An MLX checkpoint has no
    // manifest of that shape at all; its index is the safetensors index.
    await dir.getFileHandle(SPEC.manifestName ?? 'ndarray-cache.json')
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
        <div class="title" style="font-size:1.1rem;font-weight:600;color:#ccc"></div>
        <div class="desc" style="font-size:0.85rem;color:#888;max-width:440px;line-height:1.6">
          First load downloads the q4f16_1 weights and caches them
          locally (OPFS). Every subsequent visit — including the WebLLM comparison
          page — is instant; weights are served from the same shared browser cache.
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
  log('10 hand-written WGSL kernel roles across 55 kernels (37 files + 18 generated)')
  log('')

  const boot = await bootEngine({
    spec: SPEC,
    // Subgroups (when the adapter has them) for the _sg shader variants;
    // timestamp-query for profileStep / `await bench(0, 0, true)`.
    optionalFeatures: ['subgroups' as GPUFeatureName, 'timestamp-query' as GPUFeatureName],
    probeSubgroups: true,
    // Chat also surfaces device loss in the loading overlay's error panel.
    onDeviceLost: (info) => {
      showLoadingError(`GPU device lost: ${info.message || info.reason}. Reload the page to recover.`)
    },
    buildEngine: ({ device, weights, sgSizeOk, spec }) => {
      const flags = parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      // qkNorm specs (Qwen3) run the unfused composition — qkv_fused folds
      // RoPE+append into the projection with no per-head norm hook, and the
      // int8-KV path rides on qkv_fused_scratch, so ?kv8 is gated off too
      // until an unfused-int8 composition is verified end-to-end.
      const fused = !spec.qkNorm
      const int8KV = flags.int8KV && fused
      if (flags.int8KV && !int8KV) {
        log(`?kv8=1 ignored for ${spec.id} — int8 KV requires the fused QKV path (Phi-3 only for now)`)
      }
      // KV cache: int8 halves memory at the cost of one extra dispatch per layer.
      const kvMiB = Math.round(
        (spec.maxContext * spec.kvBytesPerToken) / (1024 * 1024) / (int8KV ? 2 : 1),
      )
      log(
        `Allocating ${int8KV ? 'int8 ' : ''}KV cache — ${spec.maxPages} pages × ` +
        `${spec.pageSize} = ${spec.maxContext} tokens · ~${kvMiB} MiB`,
      )
      const kv = int8KV ? allocKVPagesInt8(device, spec) : allocKVPages(device, spec)
      log('Building decode engine…')
      // Prefill A/B flags: ?reuse=0 disables cross-turn prefix reuse,
      // ?chunk=0 disables chunked GDN prefill (hybrid specs). Both default on.
      const q = new URLSearchParams(location.search)
      return buildDecodeEngine(device, weights, kv, {
        variants: flags,
        fused,
        int8KV,
        spec,
        prefixReuse: q.get('reuse') !== '0',
        chunkedPrefill: q.get('chunk') !== '0',
      })
    },
    // Pipeline warmup: one throwaway single-token generation behind the
    // progress bar so the first chat message streams at steady-state speed.
    // The KV slots it writes are overwritten by the first real turn's
    // prefill (chat always prefills from 0).
    warmup: async (engine, tokenizer, log) => {
      // One forward pass warms every pipeline — all 40 layers dispatch the same
      // ones. The full-prompt warmup is a POLISH step (first message streams at
      // steady state) and it costs promptLen per-token passes, because chunked
      // prefill is off for MoE. On a big model that polish is minutes of dead
      // screen; the first message can pay its own warm cost instead.
      const full = buildChatPromptFor(SPEC, [{ role: 'user', content: 'Hi.' }], tokenizer)
      const warmupIds = SPEC.moe ? full.slice(0, 1) : full
      const t0 = performance.now()
      log?.(`Warming up pipeline — ${warmupIds.length} token(s)${SPEC.moe ? ' (single-pass: MoE)' : ''}…`)
      await engine.generatePipelined(warmupIds, 1, () => {})
      log?.(`Warmup done in ${((performance.now() - t0) / 1000).toFixed(1)}s`)
    },
  })
  if (!boot.ok) {
    // Retry re-runs the full boot without a reload — the weight loader's
    // OPFS tier skips every shard that already completed.
    showLoadingError(boot.reason, () => { void main() })
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
  /** A history entry. Assistant turns also keep the exact token ids the model
   *  produced — re-encoding the rendered text is not guaranteed to reproduce
   *  them (BPE re-segments at the cut), and the resume path needs the ids to
   *  hand the model back its own unterminated turn. */
  type Turn = { role: 'system' | 'user' | 'assistant'; content: string; ids?: number[] }
  const history: Turn[] = [
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
    inp.placeholder = busy ? 'Generating…' : `Ask ${BRAND.name} anything…`
  }

  function updateCtxHint(nTokens?: number) {
    if (!ctxHint) return
    // True ceiling is the KV page table (spec.maxPages × spec.pageSize), which
    // is what the engine enforces — not the model's nominal maxSeq.
    if (!nTokens) {
      ctxHint.textContent = `Zero TVM · 10 WGSL kernels · ${DISPATCHES_PER_TOKEN} dispatches/token`
    } else if (nTokens >= engine.maxContext) {
      ctxHint.textContent = `Context full — ${engine.maxContext} / ${engine.maxContext} tokens · start a new chat`
    } else {
      const pct = Math.round((nTokens / engine.maxContext) * 100)
      ctxHint.textContent = `Context ${nTokens} / ${engine.maxContext} tokens (${pct}%) · ${DISPATCHES_PER_TOKEN} dispatches/token`
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
  inp.placeholder = `Ask ${BRAND.name} anything…`
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

  interface GenResult {
    text: string
    /** Every token id in the message, including any carried over from a
     *  reply this run resumed. */
    ids: number[]
  }

  /**
   * Decode one reply into `ai`.
   *
   * `priorIds` non-empty means this is a RESUME: the new tokens extend the
   * same message (and the same history entry) rather than starting a fresh
   * one, and the whole id list is decoded together so a multi-byte character
   * split across the join renders correctly.
   */
  async function runGeneration(
    ai: AiMsgHandle,
    promptIds: number[],
    priorIds: number[],
    onContinue: () => void,
  ): Promise<GenResult> {
    // The per-reply cap is the KV room the prompt leaves behind, not a magic
    // constant: the engine refuses to step past maxContext anyway, so this is
    // the true ceiling and the only budget that can cut a reply short.
    const budget = engine.maxContext - promptIds.length
    log(`Prompt: ${promptIds.length} tokens · reply budget ${budget} tokens`)
    updateCtxHint(promptIds.length)
    const resuming = priorIds.length > 0
    if (!resuming) ai.showThinking()
    const t0 = performance.now()
    let count = 0
    let firstToken = true
    const allIds: number[] = priorIds.slice()
    let fullResponse = resuming ? tokenizer.decode(allIds) : ''

    try {
      // Cooperative cancellation: the engine polls the flag between pipeline
      // submissions and drains its in-flight readbacks before returning.
      await engine.generatePipelined(promptIds, budget, (id) => {
        if (firstToken) {
          // Swap thinking indicator for the real body on first token. When
          // resuming, the body already holds the text so far — the next
          // render() re-attaches the cursor to it.
          if (!resuming) ai.body.replaceChildren(ai.cursor)
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
    const used = promptIds.length + count
    // Why the reply ended. The engine emits exactly `budget` tokens only when
    // it ran out of KV window; a Stop lands somewhere below that. Anything
    // else means the model chose to stop, which needs no notice.
    const contextFull = count >= budget
    const stopped = stopRequested && !contextFull
    ai.finish({
      fullText: fullResponse,
      // The badge sits under the whole message, so it counts the whole
      // message; tok/s is a rate, so it measures just this run.
      tokens: allIds.length,
      tokPerS,
      onRegenerate: () => { void regenerate() },
      notice: contextFull
        ? { text: `Cut off — the ${engine.maxContext}-token context window is full. Start a new chat to keep going.` }
        : stopped
          ? { text: 'Stopped — this reply is incomplete.', onContinue }
          : undefined,
    })
    // Reflect the true KV position count (prompt + generated) in the context hint.
    updateCtxHint(used)
    if (used >= engine.maxContext) {
      setBadge('Context full', 'error')
      log(`Context window full (${engine.maxContext} tokens) — start a new chat to continue.`)
    }
    return { text: fullResponse, ids: allIds }
  }

  /** Shared bracketing around a single decode turn: busy flags, prefill
   * tokenization, run, commit response to history, re-enable UI. */
  async function runTurn(): Promise<void> {
    setBusy(true); stopRequested = false
    // Resuming an earlier reply stops being meaningful once a new turn is
    // under way — retire the button but keep the "this was cut short" text.
    $('messages')?.querySelectorAll('.truncation-btn').forEach((b) => b.remove())
    const ai = addAiMsg()
    const promptIds = buildChatPromptFor(SPEC, history, tokenizer)
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
    // Commit the (empty) assistant turn before decoding so the resume path has
    // a stable identity to bind its Continue button to.
    const entry: Turn = { role: 'assistant', content: '' }
    history.push(entry)
    const r = await runGeneration(ai, promptIds, [], () => { void continueTurn(entry, ai) })
    entry.content = r.text
    entry.ids = r.ids
    setBusy(false); stopRequested = false
    updateSendEnabled()
    inp?.focus()
  }

  /**
   * Resume a reply that was cut short — instead of opening a new turn and
   * asking the model to "continue".
   *
   * The prompt is the conversation WITHOUT the partial reply (so the chat
   * template's generation prompt is the last thing in it) followed by that
   * reply's own token ids — the ids the model actually emitted, not a
   * re-encoding of the rendered text. The assistant turn is therefore still
   * open — no <|im_end|> / <|end|> — and the model simply keeps writing the
   * sentence it was in. Nothing is injected into the conversation.
   *
   * Prefix reuse: pure-attention specs (Phi-3, Qwen3) reuse the whole thing,
   * since the resumed prompt is a prefix of what the engine absorbed. Hybrid
   * (Qwen3.5) re-prefills — its GDN state is non-rewindable and sits a step
   * or two PAST the last emitted token (the pipeline's in-flight overrun), so
   * computeReuseStart's all-or-nothing rule declines. Correct either way;
   * only the hybrid pays for it.
   */
  async function continueTurn(entry: Turn, ai: AiMsgHandle): Promise<void> {
    if (generating || !entry.ids) return
    // The conversation moved on (new message / regenerate / reset) — the
    // stale button is no longer about the last thing the model said.
    if (history[history.length - 1] !== entry) return
    setBusy(true); stopRequested = false
    const promptIds = buildChatPromptFor(SPEC, history.slice(0, -1), tokenizer)
      .concat(entry.ids)
    const r = await runGeneration(ai, promptIds, entry.ids, () => { void continueTurn(entry, ai) })
    entry.content = r.text
    entry.ids = r.ids
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

/** Rewrite the static Phi-3 markup (gate dialog, header, welcome) for the
 * active model. The HTML ships with Phi-3 copy, so this is a no-op there. */
function applyModelBranding(): void {
  if (SPEC.id === 'phi3-mini') return
  document.title = `Zero-TVM Chat — ${BRAND.name}`
  const set = (sel: string, text: string) => {
    const el = document.querySelector(sel) as HTMLElement | null
    if (el) el.textContent = text
  }
  set('#start-dialog .dialog-title', `${BRAND.name} on raw WebGPU`)
  const statVals = document.querySelectorAll('#start-dialog .dialog-stats .dstat .val')
  if (statVals[0]) statVals[0].textContent = BRAND.sizeLabel   // download size
  if (statVals[1]) statVals[1].textContent = BRAND.rateLabel   // measured decode rate
  set('.model-info h1', BRAND.name)
  set('#welcome .welcome-title', `Chat with ${BRAND.name}`)
}

async function boot(): Promise<void> {
  applyModelBranding()
  initModelSwitcher(SPEC)
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
