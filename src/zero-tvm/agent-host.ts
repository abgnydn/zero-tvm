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
import { specFromSearch, buildChatPromptFor, modelBranding } from './model-select.js'
import {
  withTools, parseToolCalls, type ToolDef, type ToolDialect, type ToolChatMessage,
} from './tool-calls.js'

/** One unit of work as scripts/agent-server.mjs relays it. */
interface Job {
  id: string
  messages: { role: 'system' | 'user' | 'assistant'; content: string }[]
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

/** Which tool convention this spec's template speaks. Read off chatTemplateId
 *  rather than the model name: the same family ships both JSON and XML call
 *  formats across versions, and tool-calls.ts pins each against vendor jinja. */
function dialectFor(id: string): ToolDialect {
  if (id === 'llama3') return 'llama3'
  return spec.id.startsWith('qwen36') ? 'chatml-xml' : 'chatml-json'
}

const boot = await bootEngine({
  spec,
  optionalFeatures: ['subgroups'],
  probeSubgroups: true,
})
if (!boot.ok) {
  setStatus(`cannot boot: ${boot.reason}`, 'err')
  throw new Error(boot.reason)
}
const { engine, tokenizer } = boot
const dialect = dialectFor(spec.chatTemplateId)
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
  const t0 = performance.now()
  const emit = new Emitter(job.id)
  try {
    let messages: ToolChatMessage[] = job.messages
    if (job.tools?.length) messages = withTools(dialect, messages, job.tools)

    const promptIds = buildChatPromptFor(spec, messages as Parameters<typeof buildChatPromptFor>[1], tokenizer)
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
