/**
 * CHAT FLOW — the shared brain of every local chat surface.
 *
 * Two pages run the engine and talk to it: zero-tvm.html (chat.ts) and the
 * landing entrance's in-place chat (landing-chat.ts). Both boot through
 * bootChatEngine (the throughput configuration: fused paths, URL-flag shader
 * variants, pooled experts, pipelined generate) and drive the conversation
 * through wireChatSurface (history, resume-on-stop, regenerate, context
 * accounting). One decode loop, one turn loop — the same rule that keeps
 * chat-ui.ts shared with the rooms guest keeps this file shared between the
 * two local surfaces, so they cannot drift.
 *
 * DOM contract (ids, both pages): #composer > #inp #btn #stop-btn,
 * #new-chat-btn, #messages, #stats, #ctx-hint, plus loading-ui.ts's progress
 * markup. Optional: #welcome, #suggest-grid, #scroll-fab.
 *
 * NOTE this module's import chain reads GPUBufferUsage at module scope
 * (engine-core → weight-loader) — pages that must render without WebGPU
 * import it dynamically, after the `'gpu' in navigator` check.
 */

import { modelBranding, buildChatPromptFor } from './model-select.js'
import {
  allocKVPages,
  allocKVPagesInt8,
  buildDecodeEngine,
  type DecodeEngine,
} from './engine-core.js'
import { parseVariantFlags, ENGINE_GPU_FEATURES, PROFILE_GPU_FEATURES } from './variants.js'
import { bootEngine, setBadge, log as bootLog, type BootResult } from './loading-ui.js'
import type { Tokenizer } from './tokenizer.js'
import type { ModelSpec } from '../compiler/model-spec.js'
import { installBenchConsole } from './bench-console.js'
import {
  setChatIdentity, quantTagFor, autoGrow,
  wireScrollFab, addUserMsg, addAiMsg, type AiMsgHandle,
} from './chat-ui.js'
import { openLastCanvas } from './markdown.js'

const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T | null

function setStats(text: string) {
  const el = $('stats')
  if (el) el.textContent = text
}

/** Surface a boot/runtime failure in #loading-error. Weight downloads resume
 *  from OPFS, so a retry is cheap — no reload needed; completed shards come
 *  straight from cache. */
export function showBootError(msg: string, onRetry?: () => void): void {
  const el = $('loading-error')
  if (!el) return
  el.textContent = msg
  if (onRetry) {
    const btn = document.createElement('button')
    btn.type = 'button'
    btn.id = 'retry-download-btn'
    btn.textContent = 'Retry download'
    btn.style.cssText = 'display:block;margin-top:0.6rem;padding:0.45rem 1.1rem;'
      + 'background:var(--accent);color:#000;font-weight:600;font-size:0.8rem;'
      + 'border:none;border-radius:6px;cursor:pointer'
    btn.addEventListener('click', () => {
      el.classList.remove('visible')
      el.replaceChildren()
      onRetry()
    }, { once: true })
    el.appendChild(btn)
  }
  el.classList.add('visible')
}

// ============================================================
// Boot — the chat configuration of bootEngine
// ============================================================

export interface BootChatOptions {
  spec: ModelSpec
  /** Expert slots held resident per MoE layer (the memory build the entrance
   *  chose); 0 or absent = full model. Non-MoE specs ignore it. */
  poolSlots?: number
  /** URL flag string (variants ?sg=0/?matmul=/?kv8=1, prefill ?reuse=0/?chunk=,
   *  sampling ?temp=&topp=&minp=&seed=). Default: the page's own search. */
  search?: string
  onDeviceLost?: (info: GPUDeviceLostInfo) => void
  /** Half-open [start, end) of layers this tab holds. Absent = the whole
   *  model. Set when the entrance opens a room for ONE STAGE of a split, so
   *  the first machine hosts in place instead of navigating to share.html. */
  layerRange?: { start: number; end: number }
}

/** Boot the throughput engine with progress wired into the standard markup. */
export function bootChatEngine(opts: BootChatOptions): Promise<BootResult> {
  const { spec } = opts
  const search = opts.search ?? location.search
  const poolSlots = opts.poolSlots ?? 0
  return bootEngine({
    spec,
    layerRange: opts.layerRange,
    // Memory mode: the loader allocates slot-sized expert stacks instead of
    // full ones — that allocation difference IS the saving.
    ...(poolSlots ? { expertPool: poolSlots } : {}),
    // Subgroups (when the adapter has them) for the _sg shader variants;
    // timestamp-query for profileStep / `await bench(0, 0, true)`.
    // subgroup-matrix: requested when the adapter offers it (experimental
    // Chrome). compile() creates the sgmat GEMM only on such devices, and the
    // chunk path prefers it — gated by token-identity on every chunking spec
    // (chunk-prefill-test.mjs), the same empirical bar chunking itself holds.
    // timestamp-query was requested here UNCONDITIONALLY while lib/index.ts
    // gated it behind a flag for serialising Metal command execution — so every
    // browser number was taken with it and every native number without, and
    // they were compared. ?profile=1 now opts in, on this path only.
    optionalFeatures: [...ENGINE_GPU_FEATURES,
      ...(new URLSearchParams(location.search).get('profile') === '1' ? PROFILE_GPU_FEATURES : [])],
    probeSubgroups: true,
    onDeviceLost: opts.onDeviceLost,
    buildEngine: ({ device, weights, sgSizeOk, spec: s }) => {
      const flags = parseVariantFlags(search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      // int8 KV no longer needs the fused QKV path. It rode qkv_fused_scratch
      // when it was written, which meant Phi-3 alone — a 4k window, the one
      // model where the cache is never the constraint. The quantizer only ever
      // needed a (k_slot, v_slot) pair, which the unfused chain has in
      // kOut/vOut after RoPE, so it now covers unfused, hybrid and chunked
      // prefill. Verified token-identical against f16 on llama32, qwen35 and
      // qwen36q3 (docs/TURBOQUANT_PLAN.md, "Phase 1 BUILT").
      //
      // This gate outlived the rule it encoded and was refusing ?kv8=1 for
      // every model the engine now supports — the same stale duplicate that
      // had to be removed from lib/index.ts. MLA is the one real exclusion:
      // it caches a latent, not per-head K/V, so the kernel does not apply.
      // Which QKV composition this spec takes, independent of int8 now. Both
      // qkNorm specs and MLX-affine checkpoints go unfused: qkv_fused folds
      // RoPE+append into the projection with no per-head norm hook, and
      // dequantises symmetric group-32 inline.
      const fused = !s.qkNorm && s.weightFormat !== 'mlx-safetensors'
      const int8KV = flags.int8KV && !s.mla
      if (flags.int8KV && !int8KV) {
        bootLog(`?kv8=1 ignored for ${s.id} — int8 KV does not cover MLA, which caches a latent rather than per-head K/V`)
      }
      // KV cache: int8 halves memory at the cost of one extra dispatch per layer.
      const kvMiB = Math.round(
        (s.maxContext * s.kvBytesPerToken) / (1024 * 1024) / (int8KV ? 2 : 1),
      )
      bootLog(
        `Allocating ${int8KV ? 'int8 ' : ''}KV cache — ${s.maxPages} pages × ` +
        `${s.pageSize} = ${s.maxContext} tokens · ~${kvMiB} MiB`,
      )
      const kv = int8KV ? allocKVPagesInt8(device, s) : allocKVPages(device, s)
      bootLog('Building decode engine…')
      // Prefill A/B flags: ?reuse=0 disables cross-turn prefix reuse,
      // ?chunk=0 disables chunked GDN prefill (hybrid specs). Both default on.
      const q = new URLSearchParams(search)
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
        bootLog(`Sampling: temperature ${temperature}, top-p ${numFlag('topp', 1)}, min-p ${numFlag('minp', 0)}`)
      }
      return buildDecodeEngine(device, weights, kv, {
        variants: flags,
        fused,
        int8KV,
        spec: s,
        ...(poolSlots ? { expertPool: poolSlots } : {}),
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
    // Pipeline warmup: one throwaway generation behind the progress bar so the
    // first chat message streams at steady-state speed. The KV slots it writes
    // are overwritten by the first real turn's prefill (chat prefills from 0).
    warmup: async (engine, tokenizer, log) => {
      // One forward pass warms every pipeline — all layers dispatch the same
      // ones. The full-prompt warmup is a POLISH step (first message streams at
      // steady state) and it costs promptLen per-token passes, because chunked
      // prefill is off for MoE. On a big model that polish is minutes of dead
      // screen; the first message can pay its own warm cost instead.
      const full = buildChatPromptFor(spec, [{ role: 'user', content: 'Hi.' }], tokenizer)
      const warmupIds = spec.moe ? full.slice(0, 1) : full
      const t0 = performance.now()
      log?.(`Warming up pipeline — ${warmupIds.length} token(s)${spec.moe ? ' (single-pass: MoE)' : ''}…`)
      await engine.generatePipelined(warmupIds, 1, () => {})
      log?.(`Warmup done in ${((performance.now() - t0) / 1000).toFixed(1)}s`)
    },
  })
}

// ============================================================
// The turn loop
// ============================================================

/** What the model is doing right now, for surfaces that animate it: 'thinking'
 *  is prefill (before the first token of a fresh reply), 'generating' is
 *  streaming, 'idle' is between turns. */
export type ChatPhase = 'thinking' | 'generating' | 'idle'

export interface ChatSurfaceOptions {
  spec: ModelSpec
  tokenizer: Tokenizer
  engine: DecodeEngine
  /** One call per REAL emitted token — the mascot's heartbeat, so the bounce
   *  is the measured cadence and nothing else. */
  onToken?: () => void
  onPhase?: (phase: ChatPhase) => void
  log?: (msg: string) => void
  /** Surface-specific slash commands (name without the slash → action),
   *  merged over the built-ins (/new, /canvas). */
  commands?: Record<string, () => void>
  /** Shared single-owner latch for the engine — REQUIRED when anything else
   *  can drive the same engine (the entrance's room host). The turn loop
   *  acquires it around every generation and disables the composer while
   *  another driver holds it; without it, two generatePipelined calls
   *  interleave into one KV cache. */
  lock?: import('./engine-lock.js').EngineLock
}

/** Wire the full conversational surface onto the page's chat markup. */
export function wireChatSurface(opts: ChatSurfaceOptions): void {
  const { spec: SPEC, tokenizer, engine } = opts
  const BRAND = modelBranding(SPEC)
  const log = opts.log ?? bootLog
  const onPhase = opts.onPhase ?? (() => {})
  // Message-header identity (role name + quant tag) for chat-ui's bubbles.
  setChatIdentity(BRAND.name, quantTagFor(SPEC))

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

  // The second paragraph is why "What is zero-tvm?" is a suggested question a
  // 1B model can answer truthfully: none of these models saw zerotvm.com in
  // training, so the facts ride in the system prompt. Claims kept to the ones
  // that do not rot (no counts — the kernel-roles claim is the stable one).
  const SYSTEM_PROMPT = 'You are a helpful, concise assistant. Use Markdown (numbered lists, **bold**, and fenced ```code``` blocks with a language tag) when it clarifies the answer.\n\n'
    + 'You are running inside zero-tvm (zerotvm.com): an LLM inference engine written by hand in WGSL and TypeScript, running entirely in this browser tab on the user\'s own GPU through WebGPU — no server, no WebLLM, no TVM. Model weights download once from Hugging Face and cache on this device; prompts, generated tokens and the KV cache never leave the machine. The whole forward pass is 10 hand-written WGSL kernel roles, readable end-to-end. If asked what zero-tvm is, answer from this paragraph.'
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

  // ── History persistence ── the conversation survives a reload the way the
  // weights do. localStorage, keyed by model — on-device only, the same
  // privacy story as the OPFS weight cache. Assistant ids ride along so
  // REGENERATE works on a restored conversation; a cut-short reply's
  // Continue button does NOT survive a reload (the cut flag is not stored),
  // only its text does.
  const STORE_KEY = `zt-chat-${SPEC.id}`
  function persist(): void {
    try {
      const turns = history.filter((t) => t.role !== 'system').slice(-40)
      localStorage.setItem(STORE_KEY, JSON.stringify(turns))
    } catch { /* quota / private mode — chat still works, just not across reloads */ }
  }
  /** Rebuild the message surface and history from the stored conversation.
   *  The system prompt is NOT stored — it is always this build's, so prompt
   *  changes apply to old conversations too. */
  function restoreHistory(): void {
    let turns: Turn[] = []
    try { turns = JSON.parse(localStorage.getItem(STORE_KEY) ?? '[]') as Turn[] } catch { return }
    if (!Array.isArray(turns) || turns.length === 0) return
    turns.forEach((t, i) => {
      if (t.role === 'user' && typeof t.content === 'string') {
        history.push({ role: 'user', content: t.content })
        addUserMsg(t.content)
      } else if (t.role === 'assistant' && typeof t.content === 'string') {
        const entry: Turn = { role: 'assistant', content: t.content,
          ids: Array.isArray(t.ids) ? t.ids : undefined }
        history.push(entry)
        addAiMsg().finish({
          fullText: t.content,
          tokens: entry.ids?.length ?? 0,
          tokPerS: 0,  // a restored reply has no rate to claim
          // Regenerate only offered where it acts — on the last reply.
          ...(i === turns.length - 1 ? { onRegenerate: () => { void regenerate() } } : {}),
        })
      }
    })
    updateCtxHint(buildChatPromptFor(SPEC, history, tokenizer).length)
  }

  function updateSendEnabled() {
    if (!inp || !sendBtn) return
    sendBtn.disabled = generating || inp.value.trim().length === 0
  }
  function setBusy(busy: boolean) {
    if (!inp || !sendBtn || !stopBtn) return
    generating = busy
    sendBtn.hidden = busy
    stopBtn.hidden = !busy
    stopBtn.disabled = !busy
    inp.disabled = busy
    if (newBtn) newBtn.disabled = busy
    inp.placeholder = busy ? 'Generating…' : `Ask ${BRAND.name} anything…`
  }

  function updateCtxHint(nTokens?: number) {
    if (!ctxHint) return
    // True ceiling is the KV page table (spec.maxPages × spec.pageSize), which
    // is what the engine enforces — not the model's nominal maxSeq.
    if (!nTokens) {
      ctxHint.textContent = 'Zero TVM · 10 WGSL kernel roles · “/” for commands'
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
    try { localStorage.removeItem(STORE_KEY) } catch { /* private mode */ }
    const msgs = $('messages')
    if (msgs) msgs.replaceChildren()
    welcome?.classList.remove('hidden')
    setStats('')
    setBadge('Ready', 'ready')   // clears a lingering "Context full" state
    updateCtxHint()
    if (!matchMedia('(pointer: coarse)').matches) inp?.focus()
  }

  // Initial enabled state
  inp.disabled = false
  inp.placeholder = `Ask ${BRAND.name} anything…`
  if (newBtn) newBtn.disabled = false
  autoGrow(inp)
  updateSendEnabled()
  updateCtxHint()
  wireScrollFab()
  restoreHistory()
  // No focus steal on touch devices — popping the keyboard unprompted buries
  // half the panel behind it.
  const finePointer = !matchMedia('(pointer: coarse)').matches
  if (finePointer) inp.focus()

  // Another driver on the same engine (the room host serving a guest): hold
  // the composer while it generates, honestly labeled.
  opts.lock?.onChange((held) => {
    if (generating) return   // our own hold — setBusy owns the UI
    inp.disabled = held
    sendBtn.disabled = held || inp.value.trim().length === 0
    inp.placeholder = held ? 'Serving a room guest…' : `Ask ${BRAND.name} anything…`
  })

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
    /** Where the trailing generation prompt starts — the point the NEXT turn
     *  diverges from this one, and so where the engine must put its GDN rewind
     *  snapshot. Without it, hybrid cross-turn reuse falls back to a full
     *  re-prefill on every conversation shorter than one prefill chunk. */
    rewindAt?: number,
  ): Promise<GenResult> {
    // The per-reply cap is the KV room the prompt leaves behind, not a magic
    // constant: the engine refuses to step past maxContext anyway, so this is
    // the true ceiling and the only budget that can cut a reply short.
    const budget = engine.maxContext - promptIds.length
    log(`Prompt: ${promptIds.length} tokens · reply budget ${budget} tokens`)
    updateCtxHint(promptIds.length)
    const resuming = priorIds.length > 0
    if (!resuming) ai.showThinking()
    onPhase(resuming ? 'generating' : 'thinking')
    const t0 = performance.now()
    let count = 0
    let firstToken = true
    const allIds: number[] = priorIds.slice()
    let fullResponse = resuming ? tokenizer.decode(allIds) : ''

    try {
      // Cooperative cancellation: the engine polls the flag between pipeline
      // submissions and drains its in-flight readbacks before returning.
      await engine.generatePipelined(promptIds, budget, (id) => {
        opts.onToken?.()
        if (firstToken) {
          // Swap thinking indicator for the real body on first token. When
          // resuming, the body already holds the text so far — the next
          // render() re-attaches the cursor to it.
          if (!resuming) ai.body.replaceChildren(ai.cursor)
          firstToken = false
          onPhase('generating')
        }
        count++
        allIds.push(id)
        fullResponse = tokenizer.decode(allIds)
        ai.render(fullResponse)
        const elapsed = (performance.now() - t0) / 1000
        setStats(`${count} tok · ${(count / elapsed).toFixed(1)} tok/s`)
      }, () => stopRequested, undefined, rewindAt)
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
    onPhase('idle')
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
    // One engine, one owner: wait out a room guest's generation if the host
    // is serving one. Released in the shared epilogue below.
    const tok = await opts.lock?.acquire()
    try {
    // Resuming an earlier reply stops being meaningful once a new turn is
    // under way — retire the button but keep the "this was cut short" text.
    $('messages')?.querySelectorAll('.truncation-btn').forEach((b) => b.remove())
    const ai = addAiMsg()
    const split = { genStart: 0 }
    const promptIds = buildChatPromptFor(SPEC, history, tokenizer, split)
    // Every turn re-prefills the whole history, so once it no longer fits the
    // KV window the only way forward is a fresh conversation. Refuse cleanly
    // instead of letting prefill corrupt the cache.
    if (promptIds.length >= engine.maxContext) {
      const msg = `Context full — the conversation (${promptIds.length} tokens) exceeds the ${engine.maxContext}-token window. Start a new chat.`
      ai.finish({ fullText: `_${msg}_`, tokens: 0, tokPerS: 0 })
      setBadge('Context full', 'error')
      updateCtxHint(promptIds.length)
      log(msg)
      return
    }
    // Commit the (empty) assistant turn before decoding so the resume path has
    // a stable identity to bind its Continue button to.
    const entry: Turn = { role: 'assistant', content: '' }
    history.push(entry)
    const r = await runGeneration(ai, promptIds, [], () => { void continueTurn(entry, ai) },
      split.genStart)
    entry.content = r.text
    entry.ids = r.ids
    persist()
    } finally {
      // ONE epilogue for every exit — the early return, the happy path, and a
      // throw anywhere in between. Previously each path released by hand and a
      // throw released nothing.
      opts.lock?.release(tok)
      setBusy(false); stopRequested = false
      updateSendEnabled()
      if (finePointer) inp?.focus()
    }
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
    const tok = await opts.lock?.acquire()
    try {
      const contSplit = { genStart: 0 }
      const promptIds = buildChatPromptFor(SPEC, history.slice(0, -1), tokenizer, contSplit)
        .concat(entry.ids)
      const r = await runGeneration(ai, promptIds, entry.ids, () => { void continueTurn(entry, ai) },
        contSplit.genStart)
      entry.content = r.text
      entry.ids = r.ids
      persist()
    } finally {
      opts.lock?.release(tok)
      setBusy(false); stopRequested = false
      updateSendEnabled()
      if (finePointer) inp?.focus()
    }
  }

  // ── Slash commands ── handled locally, never sent to the model. The
  // built-ins are shared; a surface adds its own (the entrance adds /roster).
  const COMMANDS: Record<string, () => void> = {
    new: () => resetChat(),
    canvas: () => {
      if (!openLastCanvas()) sysNote('No runnable code block yet — ask for html, svg, or js.')
    },
    ...opts.commands,
  }
  /** A quiet system line in the message column — command feedback, not a
   *  conversation turn (never enters history). */
  function sysNote(text: string): void {
    const el = document.createElement('div')
    el.className = 'sys-note'
    el.textContent = text
    $('messages')?.appendChild(el)
  }
  async function send(): Promise<void> {
    if (generating || !inp) return
    const text = inp.value.trim()
    if (!text) return
    // Only EXACT command names are intercepted — "/etc/hosts?" is a question
    // for the model, not a typo'd command.
    const cmdName = text.startsWith('/') ? text.slice(1).split(/\s/)[0].toLowerCase() : ''
    if (cmdName && COMMANDS[cmdName]) {
      inp.value = ''
      autoGrow(inp)
      updateSendEnabled()
      COMMANDS[cmdName]()
      return
    }
    inp.value = ''
    autoGrow(inp)
    addUserMsg(text)
    history.push({ role: 'user', content: text })
    persist()  // the question survives a reload even if generation dies
    await runTurn()
  }

  async function regenerate(): Promise<void> {
    if (generating) return
    if (history.length === 0 || history[history.length - 1].role !== 'assistant') return
    history.pop()
    persist()
    // The LAST ai message, not '.msg.ai:last-child' — a sys-note (slash
    // command feedback) after the reply makes that selector match nothing,
    // and the popped history would drift from the DOM (lens 2026-08-17).
    const ais = $('messages')?.querySelectorAll('.msg.ai')
    if (ais && ais.length > 0) ais[ais.length - 1].remove()
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
