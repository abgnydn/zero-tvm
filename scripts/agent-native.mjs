#!/usr/bin/env node
// AGENT-NATIVE — the OpenAI server with the engine IN-PROCESS on dawn.node.
// No browser, no tab, no throttling, robustness off.
//
//   npm i --no-save webgpu                # once (the dawn.node prebuilt)
//   npm run dev                           # weights mirror (localhost only)
//   node scripts/agent-native.mjs qwen3mlx [--ctx N] [--port 8017] [--pool 0]
//
// Same /v1 surface as agent-server.mjs, but where that file relays jobs to a
// browser tab over SSE, this one calls the engine directly — the whole
// tab/relay layer does not exist here. Everything a client sees (streaming
// shape, finish_reason, 4xx-not-5xx, model-mismatch 400, tool calls parsed
// from complete output only) is kept identical; the agent-server tests define
// that contract.
//
// Measured basis (scripts/dawn-probe.mjs, BENCH.md): dawn.node runs the sgmat
// kernel at unsafe-Chrome parity (4,833 GF) with 0.14 ms submit latency, all
// features granted, disable_robustness accepted.

import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'
import { installShims } from './native/shims.mjs'
import { panelHtml } from './native/panel.mjs'

const args = process.argv.slice(2)
const flag = (n) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : null }
const param = args.find((a) => !a.startsWith('--') && a !== flag('ctx') && a !== flag('port') && a !== flag('pool')) ?? 'qwen3mlx'
const CTX = Number(flag('ctx')) || 0
const PORT = Number(flag('port')) || 8017
const POOL = flag('pool') !== '0'

await installShims({ unsafe: !args.includes('--safe') })
// dist-lib is real resolved JS — the src tree's `.js`-suffixed TS imports
// cannot load under Node, which is why the host rides the library build.
const lib = await import('../dist-lib/index.js')
const { createEngineRaw, hostSurface } = lib
const S = await hostSurface()

console.log(`[native] booting ${param}${CTX ? ` ctx=${CTX}` : ''} on dawn.node…`)
const t0 = Date.now()
const { engine, tokenizer, spec, variants, info } = await createEngineRaw({
  model: param,
  ...(CTX ? { ctx: CTX } : {}),
  onProgress: (p) => { if (p.stage === 'weights' || p.stage === 'ready') process.stdout.write(`\r[native] ${p.message}          `) },
})
console.log(`\n[native] ${info.name} ready in ${((Date.now() - t0) / 1000).toFixed(1)}s — ctx ${spec.maxContext.toLocaleString()}`)

// Same table as agent-host.ts — read off each checkpoint's own template, not
// guessed from the family name: Qwen ships BOTH call formats under ChatML,
// and the wrong one reads as "the model is bad at tools". Qwen3.8's template
// demands the XML form (<tool_call><function=…><parameter=…>).
const dialect = spec.chatTemplateId === 'llama3' ? 'llama3'
  : ['qwen36', 'qwen3-8-27b'].some((p) => spec.id.startsWith(p)) ? 'chatml-xml' : 'chatml-json'
const poolCfg = () => ({
  spec, weightRevision: spec.hfRepo, variants, fused: false, int8KV: false,
  prefillPath: !spec.moe && variants.subgroups ? 'chunked' : 'per-token',
  adapter: { vendor: 'dawn-node', architecture: 'native' },
})

// ---- the same request semantics as the browser host ------------------------

const flattenContent = (c) => typeof c === 'string' ? c
  : Array.isArray(c) ? c.map((p) => typeof p === 'string' ? p : p?.type === 'text' ? (p.text ?? '') : '').join('') : ''

function normalize(messages) {
  const out = []
  for (const m of messages ?? []) {
    const role = m.role === 'developer' ? 'system' : m.role === 'tool' ? 'user' : m.role
    if (role !== 'system' && role !== 'user' && role !== 'assistant') continue
    let content = flattenContent(m.content)
    if (m.role === 'assistant' && m.tool_calls?.length) {
      const calls = m.tool_calls.map((c) => {
        let a = {}; try { a = JSON.parse(c.function?.arguments || '{}') } catch { /* keep {} */ }
        return { name: c.function?.name ?? '', arguments: a }
      })
      content = S.renderAssistantCalls(dialect, content ?? '', calls)
    }
    out.push({ role, content })
  }
  return out
}

let busy = false
/** Cost of the most recent request — prefill/decode rates, TTFT, context
 *  used. Printed per request and served on /health so any device can watch. */
let lastStats = null
let poolTried = false
let saveTimer = null

async function runJob(body, onDelta) {
  let messages = normalize(body.messages)
  if (body.tools?.length) messages = S.withTools(dialect, messages, body.tools)
  const promptIds = S.buildChatPromptFor(spec, messages, tokenizer)

  if (POOL && !poolTried) {
    poolTried = true
    const r = await S.poolTryRestore(engine, poolCfg(), promptIds)
    console.log(r.restored ? `[native] pool: restored ${r.restored} tokens` : `[native] pool: miss (${r.reason})`)
  }
  if (promptIds.length >= spec.maxContext) {
    throw new Error(`prompt is ${promptIds.length} tokens, context is ${spec.maxContext}`)
  }
  const room = spec.maxContext - promptIds.length
  const budget = Math.max(1, Math.min(body.max_tokens ?? body.max_completion_tokens ?? room, room))
  engine.setSampling(body.temperature != null && body.temperature > 0
    ? { temperature: body.temperature, topP: body.top_p ?? 1 } : null)

  let out = ''
  const stopStrs = typeof body.stop === 'string' ? [body.stop] : (body.stop ?? [])
  const hit = { stop: null }
  // PER-REQUEST TIMING. The first token's arrival splits the request in two:
  // everything before it is prefill (prompt processing), everything after is
  // decode. Both rates are wall clock over real token counts — the same
  // definitions bench/run.mjs uses, so a number seen here is comparable to
  // BENCH.md rather than being a second, private notion of speed.
  const tStart = Date.now()
  let tFirst = 0
  const ids = await engine.generatePipelined(promptIds, budget, (id) => {
    if (!tFirst) tFirst = Date.now()
    const piece = tokenizer.decode([id])
    out += piece
    onDelta?.(piece)
    for (const st of stopStrs) if (st && out.endsWith(st)) { hit.stop = st; break }
  }, () => hit.stop !== null)
  if (hit.stop) out = out.slice(0, out.length - hit.stop.length)
  {
    const ttft = (tFirst || Date.now()) - tStart
    const decodeMs = Date.now() - (tFirst || Date.now())
    lastStats = {
      promptTokens: promptIds.length,
      genTokens: ids.length,
      ttftMs: ttft,
      prefillTokPerSec: ttft > 0 ? +(promptIds.length / (ttft / 1000)).toFixed(1) : 0,
      decodeTokPerSec: decodeMs > 0 && ids.length > 1 ? +((ids.length - 1) / (decodeMs / 1000)).toFixed(1) : 0,
      contextUsed: promptIds.length + ids.length,
      at: new Date().toISOString(),
    }
    // Prefill rate counts the prompt against time-to-first-token; with prefix
    // reuse a follow-up turn re-reads almost nothing, so a huge rate there is
    // the cache working, not the GPU getting faster.
    const pf = ttft > 0 ? (promptIds.length / (ttft / 1000)) : 0
    const dec = decodeMs > 0 && ids.length > 1 ? ((ids.length - 1) / (decodeMs / 1000)) : 0
    console.log(`[native] prompt ${promptIds.length.toLocaleString()} tok · ttft ${(ttft / 1000).toFixed(2)}s`
      + ` (${pf.toFixed(0)} tok/s prefill) · gen ${ids.length} tok`
      + ` (${dec.toFixed(1)} tok/s) · ctx ${(promptIds.length + ids.length).toLocaleString()}/${spec.maxContext.toLocaleString()}`)
  }

  const parsed = body.tools?.length ? S.parseToolCalls(dialect, out) : { text: out, calls: [] }
  const toolCalls = parsed.calls.map((c, i) => ({
    id: `call_${randomUUID().slice(0, 8)}_${i}`, type: 'function',
    function: { name: c.name, arguments: JSON.stringify(c.arguments ?? {}) },
  }))
  return {
    text: parsed.text, toolCalls,
    finishReason: toolCalls.length ? 'tool_calls' : ids.length >= budget ? 'length' : 'stop',
    usage: { prompt_tokens: promptIds.length, completion_tokens: ids.length, total_tokens: promptIds.length + ids.length },
  }
}

const scheduleSave = () => {
  if (!POOL) return
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(async () => {
    saveTimer = null
    if (busy) return
    try {
      const sv = await S.poolSave(engine, poolCfg())
      if (sv.tokens) console.log(`[native] pool: saved ${sv.tokens} tokens`)
    } catch (e) { console.log(`[native] pool save failed: ${e.message}`) }
  }, 1500)
}

// ---- HTTP ------------------------------------------------------------------

const json = (res, code, body) => {
  const s = JSON.stringify(body)
  res.writeHead(code, { 'content-type': 'application/json' })
  res.end(s)
}
const fail = (res, code, message, type = 'invalid_request_error') =>
  json(res, code, { error: { message, type, param: null, code: null } })
const readBody = (req) => new Promise((resolve, reject) => {
  const chunks = []
  req.on('data', (c) => chunks.push(c))
  req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')) } catch (e) { reject(e) } })
  req.on('error', reject)
})

createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`)
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type, authorization')
  if (req.method === 'OPTIONS') { res.writeHead(204).end(); return }

  if (url.pathname === '/v1/models') {
    json(res, 200, { object: 'list', data: [{ id: spec.id, object: 'model', created: 0, owned_by: 'zero-tvm-native' }] })
    return
  }
  // A BROWSER at the root gets the panel; everything else (the launcher's
  // readiness poll, curl, any client) keeps the JSON it has always had —
  // matched on Accept so no existing caller changes behavior.
  if (url.pathname === '/' && (req.headers.accept ?? '').includes('text/html')) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
    res.end(panelHtml())
    return
  }
  if (url.pathname === '/' || url.pathname === '/health') {
    // `last` is the previous request's measured cost — the numbers LM Studio
    // prints in its log, readable from any device on the tailnet.
    json(res, 200, { ok: true, hosting: spec.id, native: true, busy, ctx: spec.maxContext, last: lastStats })
    return
  }
  if (url.pathname !== '/v1/chat/completions' || req.method !== 'POST') {
    return fail(res, 404, `no route ${req.method} ${url.pathname}`)
  }

  let body
  try { body = await readBody(req) } catch { return fail(res, 400, 'bad JSON') }
  if (!normalize(body.messages).length) return fail(res, 400, 'messages must contain at least one entry')
  if (body.model && body.model !== spec.id && body.model !== 'ztvm' && body.model !== 'zero-tvm') {
    return fail(res, 400, `this host runs "${spec.id}", the request asked for "${body.model}"`)
  }
  if (busy) return fail(res, 400, 'busy: this host serves one request at a time')
  busy = true
  if (saveTimer) { clearTimeout(saveTimer); saveTimer = null }

  const id = randomUUID()
  const created = Math.floor(Date.now() / 1000)
  try {
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' })
      const send = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`)
      const chunk = (delta, finish = null) => send({
        id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model: spec.id,
        choices: [{ index: 0, delta, finish_reason: finish }],
      })
      chunk({ role: 'assistant', content: '' })
      const done = await runJob(body, (t) => chunk({ content: t }))
      if (done.toolCalls.length) chunk({ tool_calls: done.toolCalls.map((c, i) => ({ index: i, ...c })) })
      chunk({}, done.finishReason)
      if (body.stream_options?.include_usage) {
        send({ id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model: spec.id, choices: [], usage: done.usage })
      }
      res.write('data: [DONE]\n\n')
      res.end()
    } else {
      const done = await runJob(body, null)
      json(res, 200, {
        id: `chatcmpl-${id}`, object: 'chat.completion', created, model: spec.id,
        choices: [{
          index: 0,
          message: { role: 'assistant', content: done.text, ...(done.toolCalls.length ? { tool_calls: done.toolCalls } : {}) },
          finish_reason: done.finishReason,
        }],
        usage: done.usage,
      })
    }
  } catch (e) {
    if (!res.headersSent) fail(res, 400, e.message)
    else { res.write(`data: ${JSON.stringify({ error: { message: e.message, type: 'server_error' } })}\n\n`); res.end() }
  } finally {
    busy = false
    scheduleSave()
  }
}).listen(PORT, '127.0.0.1', () => {
  console.log(`[native] OpenAI surface: http://127.0.0.1:${PORT}/v1  (model id "${spec.id}" or "ztvm")`)
  console.log(`[native] panel:          http://127.0.0.1:${PORT}/`)
})
