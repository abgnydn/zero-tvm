/**
 * AGENT-HOST — this tab hosts a model and answers OpenAI-shaped requests
 * relayed by scripts/agent-server.mjs.
 *
 *   npm run dev
 *   node scripts/agent-server.mjs
 *   open http://localhost:5173/agent-host.html?model=qwen35
 *
 * Then any OpenAI-compatible client (pi, Cline, curl) points at
 * http://127.0.0.1:8017/v1 and talks to a model running on WebGPU in a tab.
 *
 * WHY A TAB. The engine only runs in a browser — engine-core imports
 * weight-loader, which reads GPUBufferUsage at module scope, and the weights
 * live in OPFS. The model cannot move to the server, so the server comes to
 * the model. share.ts already does this over WebRTC for another human; this
 * does it over localhost HTTP for a program.
 *
 * TRANSPORT. EventSource down for jobs, fetch POST up for token batches. No
 * WebSocket: `ws` is only in this tree transitively via puppeteer.
 *
 * THE THING THAT WILL BITE. Chrome throttles a BACKGROUNDED tab hard — share.ts
 * measured ~23 tok/s against ~65 in a focused tab. An agent run is long and
 * unattended, which is exactly when the tab is not focused. Keep it visible, or
 * keep the window frontmost on a second display.
 */

import { bootEngine } from './loading-ui.js'
import { buildDecodeEngine, allocKVPages } from './engine-core.js'
import { parseVariantFlags, type VariantFlags } from './variants.js'
import { poolSave, poolTryRestore, type PoolConfig } from './kv-pool.js'
import { specFromSearch, buildChatPromptFor, modelBranding } from './model-select.js'
import {
  withTools, parseToolCalls, renderAssistantCalls,
  type ToolDef, type ToolDialect, type ToolChatMessage,
} from './tool-calls.js'

/** One unit of work as scripts/agent-server.mjs relays it. */
interface Job {
  id: string
  messages: {
    role: 'system' | 'user' | 'assistant'
    content: string
    /** OpenAI `tool_calls` off an assistant turn, relayed verbatim. */
    toolCalls?: { id?: string; type?: string; function?: { name?: string; arguments?: string } }[]
  }[]
  tools: ToolDef[] | null
  maxTokens: number | null
  temperature: number | null
  topP: number | null
  stop: string[] | null
}

const SERVER = new URL(location.href).searchParams.get('server')
  ?? 'http://127.0.0.1:8017'

const el = (id: string) => document.getElementById(id)!
const setStatus = (t: string, cls = '') => { const s = el('status'); s.textContent = t; s.className = cls }
const logLine = (t: string) => {
  const p = el('log')
  p.textContent = `${new Date().toLocaleTimeString()}  ${t}\n${p.textContent}`.split('\n').slice(0, 200).join('\n')
}

const spec = specFromSearch(location.search)
const brand = modelBranding(spec)
el('model').textContent = `${brand.name} — ${brand.params}`

/** Which tool convention this spec's template speaks.
 *
 *  ChatML does NOT determine this — the Qwen line ships both call formats
 *  under it, so every entry here is read off that checkpoint's own
 *  chat_template.jinja and pinned in tool-calls.ts against the vendor text:
 *
 *    chatml-xml   Qwen3.6-35B-A3B, Qwen3.8-27B — the template spells out
 *                 `<tool_call><function=NAME><parameter=K>V</parameter>…`
 *    chatml-json  Qwen3-4B, Qwen3-30B-A3B, Qwen3.5 — a JSON object inside
 *                 <tool_call> tags
 *
 *  A model whose template was never read must NOT be guessed at: the wrong
 *  dialect renders tool blocks the model cannot follow and parses calls it
 *  never made, which reads as "the model is bad at tools" rather than as a
 *  wiring bug. Qwen3.8 landed exactly in that trap — an id-prefix test for
 *  'qwen36' sent it to chatml-json while its template demands XML.
 */
const XML_DIALECT_SPECS = new Set(['qwen36', 'qwen3-8-27b'])
function dialectFor(id: string): ToolDialect {
  if (id === 'llama3') return 'llama3'
  return [...XML_DIALECT_SPECS].some((p) => spec.id.startsWith(p))
    ? 'chatml-xml' : 'chatml-json'
}

// No top-level await: the build targets es2020 (vite.config.ts), where it is a
// hard error. Everything below the boot lives inside main(), started at the
// bottom of the file — the same shape chat.ts uses.
/**
 * Consent gate. This page had NO button of any kind — `querySelectorAll(
 * 'button')` returned an empty list — and called bootEngine on load, so simply
 * opening it started the weight download. Measured during review: 19 MB off the
 * HF CDN in 12 seconds with nothing pressed, on the way to ~1.8 GB.
 *
 * It is worse here than anywhere else, because this page ALSO cannot work
 * without a local agent-server on 127.0.0.1:8017 — so the bandwidth bought a
 * page that then sits on "booting…" forever. And the manifest the aborted
 * download leaves in OPFS used to disarm the chat page's gate (see
 * cache-probe.ts).
 */
async function confirmBoot(): Promise<void> {
  const { isModelCached } = await import('./cache-probe.js')
  if (await isModelCached(spec)) return
  const dlg = document.createElement('dialog')
  dlg.id = 'agent-gate'
  dlg.style.cssText = 'width:calc(100vw - 2rem);max-width:32rem;background:#0f1216;color:#d6dbe1;border:1px solid #21262d;'
    + 'border-radius:10px;padding:1.5rem 1.6rem;font:inherit'
  dlg.innerHTML = `
    <h2 style="font-size:1rem;margin:0 0 .7rem">Run ${spec.id} in this tab?</h2>
    <p style="color:#8b949e;margin:0 0 .7rem">
      This is a developer surface. It downloads the model
      and then talks to an agent-server on <code>127.0.0.1:8017</code> —
      start it with <code>npm run agent</code> from the repo. Without that
      server running, this page has nothing to serve.
    </p>
    <p class="warn" style="margin:0 0 1.1rem">Weights download once and cache locally.</p>
    <div style="display:flex;gap:.8rem;justify-content:flex-end">
      <a href="/" style="color:#7d8590">Not now</a>
      <button type="button" id="agent-gate-go" autofocus
        style="background:#d29922;color:#0b0d10;border:0;border-radius:8px;padding:.55rem 1.1rem;
               font:inherit;font-weight:600;cursor:pointer">Download &amp; boot</button>
    </div>`
  document.body.appendChild(dlg)
  dlg.showModal()
  await new Promise<void>((resolve) => {
    dlg.querySelector<HTMLButtonElement>('#agent-gate-go')?.addEventListener('click', () => {
      dlg.close(); dlg.remove(); resolve()
    }, { once: true })
  })
}

async function main(): Promise<void> {
  let flags: VariantFlags | null = null
  // Same reason as share.ts: say it BEFORE the consent dialog offers a
  // download the engine could never use.
  if (!('gpu' in navigator)) {
    setStatus('cannot boot: this browser has no WebGPU — Chrome and Edge ship it', 'err')
    return
  }
  await confirmBoot()
  const boot = await bootEngine({
    spec,
    optionalFeatures: ['subgroups', 'chromium-experimental-subgroup-matrix' as GPUFeatureName],
    probeSubgroups: true,
    // The default boot is the SCALAR composition — no subgroups, no chunked
    // prefill. This surface exists for agents, where prefill dominates, so it
    // builds the chat-class engine the same way chat.ts does (and remembers
    // the flags: the KV pool fingerprints them).
    buildEngine: ({ device, weights, sgSizeOk, spec: sp }) => {
      flags = parseVariantFlags(location.search, {
        hasSubgroupsFeature: (device.features as ReadonlySet<string>).has('subgroups'),
        sgSizeOk,
      })
      return buildDecodeEngine(device, weights, allocKVPages(device, sp), { spec: sp, variants: flags })
    },
  })
  if (!boot.ok) {
    setStatus(`cannot boot: ${boot.reason}`, 'err')
    throw new Error(boot.reason)
  }
  const { engine, tokenizer } = boot
  const dialect = dialectFor(spec.chatTemplateId)

  // KV POOL (paging Phase 1): a saved prefix survives reload/crash/restart and
  // restores in ~0.4s where re-prefill costs the whole prompt. ?pool=0 opts
  // out. The gate behind this is scripts/kv-pool-test.mjs — cold restore is
  // token-identical to full prefill under the per-token path, and shares the
  // existing cross-turn-reuse tolerance under chunking.
  const poolOn = new URL(location.href).searchParams.get('pool') !== '0'
  const poolCfg = (): PoolConfig => ({
    spec,
    weightRevision: spec.hfRepo,
    variants: flags!,
    fused: false,
    int8KV: false,
    prefillPath: !spec.moe && flags!.subgroups ? 'chunked' : 'per-token',
    adapter: { vendor: 'browser', architecture: 'webgpu' },
  })
  let poolTried = false
  // Save on IDLE, not per-job: exportKV's readback takes ~100-300ms, and doing
  // it between `done` and `busy=false` made every back-to-back request bounce
  // off "busy" (the agent-server test caught it — a pipelined client saw its
  // second request refused). A snapshot 1.5s after the last job is the same
  // durability for a restart, and a new job cancels the pending save. `saving`
  // serializes the one real hazard: a readback overlapping the next
  // generation's writes would tear the snapshot.
  let saveTimer: number | null = null
  let saving: Promise<void> | null = null
  const scheduleSave = () => {
    if (!poolOn) return
    if (saveTimer !== null) clearTimeout(saveTimer)
    saveTimer = self.setTimeout(() => {
      saveTimer = null
      if (busy) return          // a job started; it will reschedule when done
      saving = (async () => {
        try {
          const sv = await poolSave(engine, poolCfg())
          if (sv.tokens) logLine(`pool: saved ${sv.tokens} tokens`)
        } catch (e2) { logLine(`pool: save failed (${(e2 as Error).message})`) }
      })().finally(() => { saving = null })
    }, 1500)
  }
  setStatus(`ready — ${spec.maxContext.toLocaleString()} token context, ${dialect}`, 'ok')
  logLine(`engine up: ${spec.id}, dialect ${dialect}`)

/** Batched so a 60-token/s stream is ~15 POSTs/s rather than 60. */
class Emitter {
  private buf = ''
  private timer: number | null = null
  constructor(private readonly id: string) {}

  push(text: string): void {
    this.buf += text
    if (this.timer === null) {
      this.timer = self.setTimeout(() => { this.timer = null; void this.flush() }, 60)
    }
  }

  async flush(): Promise<void> {
    if (this.timer !== null) { clearTimeout(this.timer); this.timer = null }
    if (!this.buf) return
    const text = this.buf
    this.buf = ''
    await post({ type: 'delta', id: this.id, text })
  }
}

async function post(body: unknown): Promise<void> {
  try {
    await fetch(`${SERVER}/agent/emit`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    })
  } catch (e) {
    logLine(`emit failed: ${(e as Error).message}`)
  }
}

let busy = false

async function run(job: Job): Promise<void> {
  // ONE AT A TIME. There is a single engine with a single KV cache; two
  // concurrent generations would interleave into each other's cache and return
  // fluent nonsense. A client that pipelines requests gets told, not served.
  if (busy) {
    await post({ type: 'error', id: job.id, message: 'busy: this tab serves one request at a time' })
    return
  }
  busy = true
  if (saveTimer !== null) { clearTimeout(saveTimer); saveTimer = null }
  if (saving) await saving          // never generate over a readback
  const t0 = performance.now()
  const emit = new Emitter(job.id)
  try {
    // An assistant turn that called a tool arrives as
    // {content: null, tool_calls: [...]} — flattening it to '' would erase the
    // call from the transcript, and from turn 2 onward the model would not see
    // what it had already done. Render it back in this dialect's own syntax,
    // which is the same text the model emitted in the first place.
    let messages: ToolChatMessage[] = job.messages.map((m) => {
      if (m.role !== 'assistant' || !m.toolCalls?.length) return { role: m.role, content: m.content }
      const calls = m.toolCalls.map((c) => {
        let args: Record<string, unknown> = {}
        try { args = JSON.parse(c.function?.arguments || '{}') } catch { /* keep {} */ }
        return { name: c.function?.name ?? '', arguments: args }
      })
      return { role: m.role, content: renderAssistantCalls(dialect, m.content ?? '', calls) }
    })
    if (job.tools?.length) messages = withTools(dialect, messages, job.tools)

    const promptIds = buildChatPromptFor(spec, messages as Parameters<typeof buildChatPromptFor>[1], tokenizer)
    if (poolOn && !poolTried) {
      // Once, on the first request after boot — a reload is the case the pool
      // exists for. Later requests ride the in-session absorbed record.
      poolTried = true
      const r = await poolTryRestore(engine, poolCfg(), promptIds)
      logLine(r.restored ? `pool: restored ${r.restored} tokens` : `pool: miss (${r.reason})`)
    }
    if (promptIds.length >= spec.maxContext) {
      throw new Error(
        `prompt is ${promptIds.length} tokens, context is ${spec.maxContext}. `
        + `Lower the client's context window, or use a spec with a bigger one `
        + `(qwen35 and llama32 are 32,768).`)
    }

    // Leave room to answer. A client's max_tokens is a ceiling, not a promise —
    // asking for 8k out of a 32k window with a 30k prompt cannot be honoured.
    const room = spec.maxContext - promptIds.length
    const budget = Math.max(1, Math.min(job.maxTokens ?? room, room))

    engine.setSampling(
      job.temperature != null && job.temperature > 0
        ? { temperature: job.temperature, topP: job.topP ?? 1 }
        : null,
    )

    let out = ''
    // A holder rather than a bare `let`: `stopped` is only ever assigned inside
    // the onToken closure, so TS's control-flow analysis narrows it to null
    // after the await and rejects reading .length off it.
    const hit: { stop: string | null } = { stop: null }
    const ids = await engine.generatePipelined(promptIds, budget, (id) => {
      const piece = tokenizer.decode([id])
      out += piece
      emit.push(piece)
      // Client-supplied stop strings. The engine only knows the spec's stop
      // TOKENS, and a client that passes `stop` expects them honoured.
      if (job.stop?.length) {
        for (const s of job.stop) if (s && out.endsWith(s)) { hit.stop = s; break }
      }
    }, () => hit.stop !== null)

    await emit.flush()
    if (hit.stop) out = out.slice(0, out.length - hit.stop.length)

    // Tool calls are parsed from the COMPLETE output. Parsing incrementally
    // would risk announcing a call the model has not finished writing —
    // tool-calls.ts is explicit that a prefix must never claim a finished call.
    const parsed = job.tools?.length ? parseToolCalls(dialect, out) : { text: out, calls: [] }
    const toolCalls = parsed.calls.map((c, i) => ({
      id: `call_${job.id.slice(0, 8)}_${i}`,
      type: 'function' as const,
      function: { name: c.name, arguments: JSON.stringify(c.arguments ?? {}) },
    }))

    const secs = (performance.now() - t0) / 1000
    logLine(`${job.id.slice(0, 8)}  ${promptIds.length} in / ${ids.length} out`
      + `  ${(ids.length / secs).toFixed(1)} tok/s`
      + (toolCalls.length ? `  ${toolCalls.length} tool call(s)` : ''))

    await post({
      type: 'done',
      id: job.id,
      text: parsed.text,
      toolCalls,
      finishReason: toolCalls.length ? 'tool_calls'
        : ids.length >= budget ? 'length' : 'stop',
      usage: {
        prompt_tokens: promptIds.length,
        completion_tokens: ids.length,
        total_tokens: promptIds.length + ids.length,
      },
    })
  } catch (e) {
    await emit.flush()
    logLine(`${job.id.slice(0, 8)}  ERROR ${(e as Error).message}`)
    await post({ type: 'error', id: job.id, message: (e as Error).message })
  } finally {
    busy = false
    scheduleSave()
  }
}

function connect(): void {
  const es = new EventSource(`${SERVER}/agent/jobs?model=${encodeURIComponent(spec.id)}`)
  es.onopen = () => { setStatus(`hosting ${spec.id} → ${SERVER}/v1`, 'ok'); logLine('connected to agent-server') }
  es.onmessage = (ev) => { void run(JSON.parse(ev.data) as Job) }
  es.onerror = () => {
    // EventSource retries on its own; the status line is the only honest
    // signal that the server went away, since jobs simply stop arriving.
    setStatus(`agent-server unreachable at ${SERVER} — retrying`, 'err')
  }
}
  connect()
}

void main()
