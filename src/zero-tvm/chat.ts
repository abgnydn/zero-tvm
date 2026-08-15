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

// cache-probe is NOT imported here. It pulls in the loaders, which read
// GPUBufferUsage at module scope, and a static import evaluates that chain
// before a single line of this file runs — so on a browser without WebGPU the
// module threw a ReferenceError and the page sat on its pre-activated
// "Preparing Zero-TVM / Starting… 0%" overlay forever, with no gate, no error
// and no mention of what would work. That is the site's primary CTA. Every
// other surface here (landing.ts, share.ts) already imports it dynamically for
// exactly this reason; the product page was the one that did not.
import { specFromSearch, modelBranding, buildChatPromptFor } from './model-select.js'
import { initModelSwitcher } from './model-switcher.js'
import { mountMascot, mascotPalette } from '../mascot.js'
import { initSamplingControls } from './sampling-ui.js'
import {
  allocKVPages,
  allocKVPagesInt8,
  buildDecodeEngine,
} from './engine-core.js'
import { parseVariantFlags } from './variants.js'
import { bootEngine, setBadge } from './loading-ui.js'
import { installBenchConsole } from './bench-console.js'
import {
  setChatIdentity, quantTagFor, autoGrow,
  wireScrollFab, addUserMsg, addAiMsg, type AiMsgHandle,
} from './chat-ui.js'

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
let POOL_SLOTS_UI = poolFromSearch()

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

// Message surface (bubbles, streaming markdown, actions, scroll) lives in
// chat-ui.ts, shared verbatim with the remote-guest page (share.ts).

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

  const boot = await bootEngine({
    spec: SPEC,
    // Memory mode: the loader allocates slot-sized expert stacks instead of
    // full ones — that allocation difference IS the saving.
    ...(POOL_SLOTS_UI ? { expertPool: POOL_SLOTS_UI } : {}),
    // Subgroups (when the adapter has them) for the _sg shader variants;
    // timestamp-query for profileStep / `await bench(0, 0, true)`.
    // subgroup-matrix: requested when the adapter offers it (experimental
    // Chrome). compile() creates the sgmat GEMM only on such devices, and the
    // chunk path prefers it — gated by token-identity on every chunking spec
    // (chunk-prefill-test.mjs), the same empirical bar chunking itself holds.
    optionalFeatures: ['subgroups' as GPUFeatureName, 'timestamp-query' as GPUFeatureName,
      'chromium-experimental-subgroup-matrix' as GPUFeatureName],
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
      // ...and MLX-affine specs likewise: qkv_fused dequantises symmetric
      // group-32 inline, so affine checkpoints take the unfused matmul path
      // whether or not they carry qkNorm.
      const fused = !spec.qkNorm && spec.weightFormat !== 'mlx-safetensors'
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
      // Sampling: ?temp=0.8&topp=0.95&minp=0.02&seed=1234. Absent or ?temp=0
      // is greedy argmax, which is what the whole page has always done and
      // what every reference comparison assumes.
      const numFlag = (k: string, dflt: number): number => {
        const raw = q.get(k)
        if (raw === null) return dflt
        const v = Number(raw)
        return Number.isFinite(v) ? v : dflt
      }
      const temperature = numFlag('temp', 0)
      if (temperature > 0) {
        log(`Sampling: temperature ${temperature}, top-p ${numFlag('topp', 1)}, min-p ${numFlag('minp', 0)}`)
      }
      return buildDecodeEngine(device, weights, kv, {
        variants: flags,
        fused,
        int8KV,
        spec,
        ...(POOL_SLOTS_UI ? { expertPool: POOL_SLOTS_UI } : {}),
        prefixReuse: q.get('reuse') !== '0',
        chunkedPrefill: q.get('chunk') !== '0',
        pooledChunkedPrefill: q.get('chunk') === '1',
        sampling: temperature > 0
          ? {
              temperature,
              topP: numFlag('topp', 1),
              minP: numFlag('minp', 0),
              seed: numFlag('seed', 0),
            }
          : undefined,
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
  // The sampling popover is mounted before this (the header is interactive
  // during the download), so hand it the engine once there is one.
  liveEngine = engine
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
      ctxHint.textContent = 'Zero TVM · 10 WGSL kernel roles'
    } else if (nTokens >= engine.maxContext) {
      ctxHint.textContent = `Context full — ${engine.maxContext} / ${engine.maxContext} tokens · start a new chat`
    } else {
      const pct = Math.round((nTokens / engine.maxContext) * 100)
      ctxHint.textContent = `Context ${nTokens} / ${engine.maxContext} tokens (${pct}%)`
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
        headerMascot?.pulse()
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
      const url = await m.snapshot()
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
