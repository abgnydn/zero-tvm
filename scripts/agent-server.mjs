#!/usr/bin/env node
// AGENT-SERVER — an OpenAI-shaped front door for the model running in a browser tab.
//
//   npm run dev                                    # terminal 1 (vite, :5173)
//   node scripts/agent-server.mjs                  # terminal 2 (this, :8017)
//   open http://localhost:5173/agent-host.html?model=qwen35
//
// Then point any OpenAI-compatible client at http://127.0.0.1:8017/v1 :
//
//   pi     ~/.pi/agent/models.json → provider baseUrl, api "openai-completions"
//   Cline  VS Code → API Provider "OpenAI Compatible" → Base URL + model id
//
// WHY THIS SHAPE. The engine only runs in a browser — engine-core imports
// weight-loader, which reads GPUBufferUsage at module scope, and the weights
// live in OPFS. So the model cannot move to the server; the server has to come
// to the model. share.ts already proves the pattern (a tab that hosts the model
// and answers remote requests over WebRTC); this is the same idea with an HTTP
// front door instead of a room link.
//
// TRANSPORT: SSE down, POST up. No WebSocket library — `ws` is only in this
// tree transitively via puppeteer and would vanish on a dependency bump. SSE is
// also exactly what the OpenAI streaming API speaks, so both halves of this
// file use one mechanism.
//
//   browser  GET  /agent/jobs   → an SSE stream of work
//   browser  POST /agent/emit   → token batches and the final result
//   client   POST /v1/chat/completions
//   client   GET  /v1/models
//
// LOCALHOST ONLY, and deliberately: it binds 127.0.0.1 and has no auth. It
// hands whatever a client sends straight to a model in a tab.

import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'

const PORT = Number(process.env.PORT) || 8017
const VERBOSE = process.argv.includes('--verbose')

/** Jobs handed to the browser but not yet finished, by id. */
const pending = new Map()
/** The browser's SSE connection, or null when no tab is hosting. */
let host = null
let hostModel = null

const log = (...a) => console.log(`[agent-server]`, ...a)
const vlog = (...a) => { if (VERBOSE) console.log(`[agent-server]`, ...a) }

const json = (res, code, body) => {
  const s = JSON.stringify(body)
  res.writeHead(code, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(s) })
  res.end(s)
}

/** OpenAI's error envelope. A bare string here makes clients report a parse
 *  failure instead of the actual problem, which is a miserable thing to debug
 *  from inside an editor extension. */
const fail = (res, code, message, type = 'invalid_request_error') =>
  json(res, code, { error: { message, type, param: null, code: null } })

const readBody = (req) => new Promise((resolve, reject) => {
  const chunks = []
  req.on('data', (c) => chunks.push(c))
  req.on('end', () => {
    try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')) }
    catch (e) { reject(e) }
  })
  req.on('error', reject)
})

/**
 * Content can be a string OR an array of parts — every modern client sends the
 * array form at least sometimes (image blocks, cache_control markers), and a
 * shim that assumes strings silently renders "[object Object]" into the prompt.
 * Non-text parts are dropped with a note rather than crashing: this engine has
 * no vision tower wired, so an image cannot be honoured, and failing the whole
 * request over one is worse than answering without it.
 */
function flattenContent(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  const out = []
  for (const part of content) {
    if (typeof part === 'string') out.push(part)
    else if (part?.type === 'text') out.push(part.text ?? '')
    else if (part?.type === 'image_url' || part?.type === 'image') out.push('[image omitted: no vision tower]')
  }
  return out.join('')
}

/**
 * OpenAI roles → what the chat templates understand.
 * `developer` is the newer spelling of `system` and clients do send it.
 * `tool` turns are folded to `user`, which is what every template in this repo
 * renders a tool response as.
 */
function normalizeMessages(messages) {
  const out = []
  for (const m of messages ?? []) {
    const role = m.role === 'developer' ? 'system' : m.role === 'tool' ? 'user' : m.role
    if (role !== 'system' && role !== 'user' && role !== 'assistant') continue
    out.push({ role, content: flattenContent(m.content), toolCalls: m.tool_calls, toolCallId: m.tool_call_id })
  }
  return out
}

function dispatch(job) {
  if (!host) throw new Error('no browser tab is hosting a model')
  pending.set(job.id, job)
  host.write(`data: ${JSON.stringify(job.request)}\n\n`)
  vlog(`-> tab  job ${job.id.slice(0, 8)} (${job.request.messages.length} messages)`)
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`)
  const path = url.pathname

  // CORS so the vite-served page on :5173 can reach this on :8017.
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type, authorization')
  if (req.method === 'OPTIONS') { res.writeHead(204).end(); return }

  // ---- the browser host ---------------------------------------------------

  if (path === '/agent/jobs' && req.method === 'GET') {
    if (host) { host.end() }   // a reloaded tab replaces the old one
    hostModel = url.searchParams.get('model') || 'zero-tvm'
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    })
    res.write(': connected\n\n')
    host = res
    log(`tab connected, hosting "${hostModel}"`)
    // Anything already queued was waiting for exactly this.
    for (const job of pending.values()) dispatch(job)
    req.on('close', () => {
      if (host === res) { host = null; log('tab disconnected') }
    })
    return
  }

  if (path === '/agent/emit' && req.method === 'POST') {
    let msg
    try { msg = await readBody(req) } catch { return fail(res, 400, 'bad JSON') }
    const job = pending.get(msg.id)
    if (!job) { json(res, 200, { ok: true, unknown: true }); return }
    try {
      if (msg.type === 'delta') job.onDelta(msg.text)
      else if (msg.type === 'done') { pending.delete(msg.id); job.onDone(msg) }
      else if (msg.type === 'error') { pending.delete(msg.id); job.onError(new Error(msg.message)) }
    } catch (e) { log('emit handler threw:', e.message) }
    json(res, 200, { ok: true })
    return
  }

  // ---- the OpenAI surface -------------------------------------------------

  if (path === '/v1/models' && req.method === 'GET') {
    // Neither client NEEDS this: pi takes its model list entirely from
    // ~/.pi/agent/models.json and never asks, and Cline only calls it to fill
    // the settings dropdown, falling back to a free-text box on any error.
    // Kept because it costs nothing and makes the dropdown work — and it
    // answers even with no tab connected, so "is the server up" is separable
    // from "is a model loaded".
    json(res, 200, {
      object: 'list',
      data: [{
        id: hostModel ?? 'zero-tvm',
        object: 'model',
        created: 0,
        owned_by: 'zero-tvm',
      }],
    })
    return
  }

  if (path === '/v1/chat/completions' && req.method === 'POST') {
    let body
    try { body = await readBody(req) } catch { return fail(res, 400, 'bad JSON') }
    if (!host) {
      // 400, NOT 503. pi retries 5xx with backoff — a 503 here makes it hang
      // for ~2 minutes and then fail, instead of telling you to open the tab.
      // Verified against pi 0.84.1's own retry path. The same reasoning applies
      // to every unrecoverable condition below: prefer 4xx.
      return fail(res, 400,
        'no browser tab is hosting a model — open agent-host.html and wait for "hosting"')
    }
    const messages = normalizeMessages(body.messages)
    if (!messages.length) return fail(res, 400, 'messages must contain at least one entry')

    const id = randomUUID()
    const created = Math.floor(Date.now() / 1000)
    const model = body.model || hostModel || 'zero-tvm'
    const stream = body.stream === true

    const request = {
      id,
      messages,
      tools: body.tools ?? null,
      maxTokens: body.max_tokens ?? body.max_completion_tokens ?? null,
      temperature: body.temperature ?? null,
      topP: body.top_p ?? null,
      stop: typeof body.stop === 'string' ? [body.stop] : (body.stop ?? null),
    }

    if (!stream) {
      let text = ''
      try {
        const done = await new Promise((resolve, reject) => {
          dispatch({ id, request, onDelta: (t) => { text += t }, onDone: resolve, onError: reject })
        })
        json(res, 200, {
          id: `chatcmpl-${id}`, object: 'chat.completion', created, model,
          choices: [{
            index: 0,
            message: {
              role: 'assistant',
              content: done.text ?? text,
              ...(done.toolCalls?.length ? { tool_calls: done.toolCalls } : {}),
            },
            finish_reason: done.finishReason ?? 'stop',
          }],
          usage: done.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
        })
      } catch (e) {
        pending.delete(id)
        // 4xx for the same reason: these are context overflow, a busy tab, or a
        // dead tab — none of them get better by retrying.
        fail(res, 400, e.message)
      }
      return
    }

    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
    })
    const send = (obj) => res.write(`data: ${JSON.stringify(obj)}\n\n`)
    const chunk = (delta, finish = null) => send({
      id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })

    // The FIRST chunk carries role and no content. Clients key off it to open
    // the assistant message; several render nothing at all without it.
    chunk({ role: 'assistant', content: '' })

    let closed = false
    req.on('close', () => { closed = true; pending.delete(id) })

    try {
      const done = await new Promise((resolve, reject) => {
        dispatch({
          id, request,
          onDelta: (t) => { if (!closed) chunk({ content: t }) },
          onDone: resolve, onError: reject,
        })
      })
      if (!closed) {
        if (done.toolCalls?.length) {
          chunk({ tool_calls: done.toolCalls.map((c, i) => ({ index: i, ...c })) })
        }
        chunk({}, done.finishReason ?? 'stop')
        if (body.stream_options?.include_usage) {
          send({
            id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model,
            choices: [], usage: done.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
          })
        }
        res.write('data: [DONE]\n\n')
        res.end()
      }
    } catch (e) {
      pending.delete(id)
      if (!closed) {
        // Mid-stream there is no status code left to set, so the error has to
        // ride the stream. Clients differ on what they surface; both a chunk
        // and an error frame are sent so neither shows a silent empty reply.
        send({ error: { message: e.message, type: 'server_error' } })
        chunk({ content: `\n\n[zero-tvm error: ${e.message}]` }, 'stop')
        res.write('data: [DONE]\n\n')
        res.end()
      }
    }
    return
  }

  if (path === '/' || path === '/health') {
    json(res, 200, { ok: true, hosting: hostModel, tabConnected: !!host, pending: pending.size })
    return
  }

  fail(res, 404, `no route ${req.method} ${path}`)
})

server.listen(PORT, '127.0.0.1', () => {
  log(`listening on http://127.0.0.1:${PORT}`)
  log(`  OpenAI base URL:  http://127.0.0.1:${PORT}/v1`)
  log(`  waiting for a tab: http://localhost:5173/agent-host.html?model=qwen35`)
})
