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
const param = args.find((a) => !a.startsWith('--') && a !== flag('ctx') && a !== flag('port')
  && a !== flag('pool') && a !== flag('experts') && a !== flag('kv8')) ?? 'qwen3mlx'
const CTX = Number(flag('ctx')) || 0
const PORT = Number(flag('port')) || 8017
// TWO different pools, and confusing them cost a 43k-token prefill twice.
// --pool   = the KV prefix pool on disk: the prefill survives a restart.
// --experts = expert SLOTS per MoE layer, a memory build. It disables chunked
//   prefill (the pooled path structurally cannot chunk), so it trades minutes
//   of prefill for gigabytes of RAM — never turn it on to serve long prompts.
const POOL = flag('pool') !== '0'
const EXPERTS = Number(flag('experts')) || 0
// int8 KV: DEFAULT ON, `--kv8 0` opts out. Same default as the browser, and for
// the same reason — it was measured free end to end (paired perplexity -0.09%
// at 1k windows / +0.10% at 4k, both within noise; greedy output
// token-identical to f16 on three models) and halves the cache. The one thing
// that used to argue against it here, that the KV disk pool went inert under
// int8, stopped being true when the pool learned to carry the scales.
// Cost: ~5-8% of prefill throughput (llama32 681 -> 625 tok/s, qwen35
// 443 -> 419) against half the KV memory.
const KV8 = flag('kv8') !== '0'

await installShims({ unsafe: !args.includes('--safe') })
// dist-lib is real resolved JS — the src tree's `.js`-suffixed TS imports
// cannot load under Node, which is why the host rides the library build.
const lib = await import('../dist-lib/index.js')
const { createEngineRaw, hostSurface } = lib
const S = await hostSurface()

console.log(`[native] booting ${param}${CTX ? ` ctx=${CTX}` : ''} on dawn.node…`)
const t0 = Date.now()
console.log(KV8
  ? '[native] int8 KV cache (default) — half the cache memory; --kv8 0 for f16'
  : '[native] f16 KV cache — int8 disabled by --kv8 0')
if (EXPERTS) console.log(`[native] expert pool ${EXPERTS} slots/layer — saves RAM, but prefill runs PER TOKEN (no chunking)`)
const { engine, tokenizer, spec, variants, info } = await createEngineRaw({
  model: param,
  ...(CTX ? { ctx: CTX } : {}),
  ...(EXPERTS ? { expertPool: EXPERTS } : {}),
  ...(KV8 ? { int8KV: true } : {}),
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
  spec, weightRevision: spec.hfRepo, variants, fused: false, int8KV: KV8,
  prefillPath: !spec.moe && variants.subgroups ? 'chunked' : 'per-token',
  adapter: { vendor: 'dawn-node', architecture: 'native' },
})

// ---- the same request semantics as the browser host ------------------------

const flattenContent = (c) => typeof c === 'string' ? c
  : Array.isArray(c) ? c.map((p) => typeof p === 'string' ? p : p?.type === 'text' ? (p.text ?? '') : '').join('') : ''

/**
 * VERBATIM ASSISTANT TURNS — what the model wrote, not a reconstruction.
 *
 * A client hands assistant turns back as STRUCTURE (content + tool_calls),
 * because that is what the OpenAI shape carries. Re-rendering that structure
 * produces text that is *equivalent* but rarely byte-identical to what the
 * model emitted — different whitespace inside a <tool_call> block is enough.
 * Different text means different tokens, which means the new prompt stops
 * matching the KV cache at the FIRST assistant turn, and the engine has to
 * re-prefill the entire conversation. Measured on a real Cline session: a
 * 16,454-token prompt re-prefilled in full, 98s to first token, when only
 * 4,733 tokens were new.
 *
 * So the raw output is kept, keyed by the structure the client will send
 * back, and substituted on the next turn. Bounded, and a miss simply falls
 * back to re-rendering — the old behaviour, never an error.
 */
const RAW_CACHE = new Map()
const RAW_CACHE_MAX = 64
const rawKey = (content, calls) => JSON.stringify([content ?? '',
  (calls ?? []).map((c) => [c.name, JSON.stringify(c.arguments ?? {})])])
function rememberRaw(text, calls, raw) {
  const k = rawKey(text, calls)
  RAW_CACHE.delete(k)
  RAW_CACHE.set(k, raw)
  while (RAW_CACHE.size > RAW_CACHE_MAX) RAW_CACHE.delete(RAW_CACHE.keys().next().value)
}

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
      const raw = RAW_CACHE.get(rawKey(content, calls))
      content = raw ?? S.renderAssistantCalls(dialect, content ?? '', calls)
    }
    out.push({ role, content })
  }
  return out
}

let busy = false
/** What the CURRENT request is doing, for anything watching: prompt
 *  processing on a long conversation is minutes of wall clock, and a client
 *  that shows nothing during it looks hung. Written from the engine's own
 *  callbacks, read by /health. */
let live = null
/** Cost of the most recent request — prefill/decode rates, TTFT, context
 *  used. Printed per request and served on /health so any device can watch. */
let lastStats = null
let poolTried = false
let saveTimer = null

async function runJob(body, onDelta, isAborted) {
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
  live = { phase: 'prefill', done: 0, total: promptIds.length, generated: 0, startedAt: tStart }
  const ids = await engine.generatePipelined(promptIds, budget, (id) => {
    if (!tFirst) tFirst = Date.now()
    const piece = tokenizer.decode([id])
    out += piece
    onDelta?.(piece)
    live = {
      phase: 'decode', done: promptIds.length, total: promptIds.length,
      generated: (live?.generated ?? 0) + 1, budget, startedAt: tStart, firstAt: tFirst,
    }
    for (const st of stopStrs) if (st && out.endsWith(st)) { hit.stop = st; break }
  }, () => hit.stop !== null || (isAborted?.() ?? false),
  (done, total) => { live = { phase: 'prefill', done, total, generated: 0, startedAt: tStart } })
  if (hit.stop) out = out.slice(0, out.length - hit.stop.length)
  {
    const ttft = (tFirst || Date.now()) - tStart
    const decodeMs = Date.now() - (tFirst || Date.now())
    const pf = engine.getLastPrefill?.() ?? null
    // A full re-prefill on a conversation the cache already holds is the most
    // expensive thing here, and "matches only 105 of 14316" does not say WHOSE
    // fault it is. Which MESSAGE the agreement dies in does:
    //   message 0 (system)  → the client rewrites its own system prompt each
    //                         turn; nothing downstream can reuse anything.
    //   an assistant turn   → our re-render differs from what the model wrote,
    //                         i.e. a RAW_CACHE miss — ours to fix.
    //   a user/tool turn    → the client edited earlier content.
    // Cheap exactly when it matters: the loop stops as soon as the running
    // count passes lcp, so a divergence at token 105 costs one short encode.
    // Counts include each prefix's generation prompt, so treat the boundary as
    // approximate — it identifies the message, not the token.
    if (pf && pf.reused === 0 && pf.absorbed > 0 && pf.lcp < pf.absorbed) {
      try {
        let where = null
        for (let k = 1; k <= messages.length; k++) {
          const n = S.buildChatPromptFor(spec, messages.slice(0, k), tokenizer).length
          if (n > pf.lcp) { where = { i: k - 1, role: messages[k - 1].role, upto: n }; break }
        }
        if (where) {
          console.log(`[native] the divergence falls in message ${where.i} of ${messages.length}`
            + ` (role '${where.role}', ends near token ${where.upto}) — `
            + (where.i === 0 ? 'the SYSTEM prompt changed between turns, so nothing can be reused'
              : where.role === 'assistant' ? 'OUR re-render of that assistant turn differs from what the model wrote (raw-cache miss)'
              : 'the client edited an earlier message'))
        }
      } catch { /* diagnostics never break a request */ }
    }
    lastStats = {
      reusedTokens: pf ? pf.reused : null,
      promptTokens: promptIds.length,
      genTokens: ids.length,
      ttftMs: ttft,
      prefillTokPerSec: ttft > 0 ? +(promptIds.length / (ttft / 1000)).toFixed(1) : 0,
      newTokens: pf ? promptIds.length - pf.reused : null,
      decodeTokPerSec: decodeMs > 0 && ids.length > 1 ? +((ids.length - 1) / (decodeMs / 1000)).toFixed(1) : 0,
      contextUsed: promptIds.length + ids.length,
      at: new Date().toISOString(),
    }
    // Prefill rate counts the prompt against time-to-first-token; with prefix
    // reuse a follow-up turn re-reads almost nothing, so a huge rate there is
    // the cache working, not the GPU getting faster.
    const pfRate = ttft > 0 ? (promptIds.length / (ttft / 1000)) : 0
    const dec = decodeMs > 0 && ids.length > 1 ? ((ids.length - 1) / (decodeMs / 1000)) : 0
    live = null
    const reused = lastStats.reusedTokens
    console.log(`[native] prompt ${promptIds.length.toLocaleString()} tok`
      + (reused ? ` (${reused.toLocaleString()} reused, ${(promptIds.length - reused).toLocaleString()} new)` : ' (no reuse)')
      + ` · ttft ${(ttft / 1000).toFixed(2)}s (${pfRate.toFixed(0)} tok/s prefill) · gen ${ids.length} tok`
      + ` (${dec.toFixed(1)} tok/s) · ctx ${(promptIds.length + ids.length).toLocaleString()}/${spec.maxContext.toLocaleString()}`)
  }

  const parsed = body.tools?.length ? S.parseToolCalls(dialect, out) : { text: out, calls: [] }
  // A call the parser could NOT read must never be handed over as if it were
  // one. It arrives with no name or empty arguments, the client fails the
  // tool, and — because decoding is deterministic — asking again with the
  // identical prompt reproduces the identical broken call. That is how a
  // truncated tool call becomes "6 errors in a row" and then a client-side
  // loop detector firing on 5 identical calls. The usual cause is the token
  // budget cutting the model off mid-call, which is why finishReason below
  // reports 'length' rather than pretending the turn ended with tool calls.
  const broken = parsed.calls.filter((c) => c.error || !c.name)
  const usable = parsed.calls.filter((c) => !c.error && c.name)
  const hitBudget = ids.length >= budget
  for (const c of broken) {
    console.log(`[native] TOOL CALL UNPARSEABLE — ${c.error ?? 'no function name'}`
      + ` (${(c.raw ?? '').length} chars${hitBudget ? ', generation hit the token budget' : ''})`)
  }
  if (broken.length) {
    console.log('[native]   not sending it: the client would fail the tool and retry the identical'
      + ' prompt for the identical result. Returning the raw text so the next turn can see it.')
  }
  // The XML dialect cannot tell the STRING "42" from the number 42 — the block
  // carries no types, as tool-calls.ts says outright, and it guesses by trying
  // JSON.parse. That guess reaches the client as the wrong JSON type and the
  // tool fails its own validation: a command of `true` or a path of `42` is
  // rare but a `limit`-shaped string is not. We are handed the schema, so
  // stop guessing where it can answer.
  const propsFor = (name) =>
    body.tools?.find((t) => t.function?.name === name)?.function?.parameters?.properties ?? {}
  const retype = (c) => {
    const props = propsFor(c.name)
    const args = {}
    for (const [k, v] of Object.entries(c.arguments ?? {})) {
      args[k] = props[k]?.type === 'string' && typeof v !== 'string' && v !== null
        ? String(v) : v
    }
    return args
  }
  const toolCalls = usable.map((c, i) => ({
    id: `call_${randomUUID().slice(0, 8)}_${i}`, type: 'function',
    function: { name: c.name, arguments: JSON.stringify(retype(c)) },
  }))
  // A call can parse perfectly and still be REFUSED by the client, because the
  // client validates it against the schema it sent us. That failure is
  // invisible here — the logs carry no content by design — so a session shows
  // "tool execution failed" while every line on this side looks healthy. We
  // are not the tool runner and must not block the call; but the schema is in
  // hand, so say when one cannot possibly validate. Parameter NAMES only.
  for (const c of usable) {
    const fn = body.tools?.find((t) => t.function?.name === c.name)?.function
    if (!fn) {
      console.log(`[native] TOOL CALL '${c.name}' — the client offered no tool by that name`)
      continue
    }
    const props = fn.parameters?.properties ?? {}
    const keys = Object.keys(c.arguments ?? {})
    const missing = (fn.parameters?.required ?? []).filter((k) => !keys.includes(k))
    const unknown = keys.filter((k) => !(k in props))
    // TYPE matters as much as presence, and is easier to miss: a required
    // string that arrives as null or a number is present, looks fine in a
    // key-only log, and is refused by the client all the same. The XML dialect
    // carries no types and guesses with JSON.parse, so `null` and `42` are
    // exactly what it produces from ordinary text.
    const typeOf = (v) => v === null ? 'null' : Array.isArray(v) ? 'array' : typeof v
    const wrong = keys.filter((k) => {
      const want = props[k]?.type
      if (!want) return false
      const got = typeOf(c.arguments[k])
      return want === 'integer' ? got !== 'number' : got !== want
    }).map((k) => `${k}: want ${props[k].type}, got ${typeOf(c.arguments[k])}`)
    if (missing.length || unknown.length || wrong.length) {
      console.log(`[native] TOOL CALL '${c.name}' will not validate against the client's schema:`
        + (missing.length ? ` MISSING required [${missing.join(', ')}]` : '')
        + (unknown.length ? ` UNKNOWN [${unknown.join(', ')}]` : '')
        + (wrong.length ? ` WRONG TYPE [${wrong.join('; ')}]` : '')
        + ` · model sent [${keys.join(', ') || '(no parameters)'}]`
        + ` · schema wants [${Object.keys(props).join(', ')}]`)
    }
  }
  // Keep the RAW output against the structure the client will hand back, so
  // the next turn re-sends what the model actually wrote and the KV cache
  // still matches. Only for turns that carry calls — plain text round-trips
  // unchanged already.
  // THE failure worth reporting, and it needs no threshold: whatever the token
  // count, a turn with no tool call and no non-whitespace text reaches the
  // client as an empty reply. It then resends the identical prompt, and with
  // deterministic decoding gets the identical nothing — minutes per attempt on
  // a long conversation. Counting tokens instead would need a constant
  // guessing at "too short", which is wrong at some length.
  const visibleText = broken.length && !usable.length ? (parsed.text || out) : parsed.text
  if (!toolCalls.length && !(visibleText ?? '').trim()) {
    const st = engine.getLastStop?.() ?? null
    const roles = messages.map((m) => m.role[0]).join('')
    const last = messages[messages.length - 1]
    console.log(`[native] EMPTY REPLY — the client sees nothing.`
      + ` generated ${ids.length} token(s)`
      + (st ? `, ended by id ${st.id} (${st.stop ? 'a stop id'
        : st.inVocab ? 'not a stop id' : 'OUT OF RANGE — the readback returned no valid id, an ENGINE fault'})` : '')
      + ` · prompt ${promptIds.length} tok · ${messages.length} messages [${roles}]`
      + ` · last role '${last?.role}' (${(last?.content ?? '').length} chars)`
      + ` · tools ${body.tools?.length ?? 0} · stops ${JSON.stringify(spec.stops)}`)
    console.log('[native]   a client that retries this will loop: same prompt, same empty answer.')
  }
  if (usable.length) rememberRaw(parsed.text, usable, out)
  return {
    // With nothing usable, hand back what the model actually wrote rather than
    // an empty string — an empty reply is the other way a client ends up
    // resending the same prompt forever.
    text: visibleText,
    toolCalls,
    finishReason: hitBudget ? 'length' : toolCalls.length ? 'tool_calls' : 'stop',
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
    // `pool` reports whether the KV prefix cache on disk is armed. Without it
    // there is no way to tell a host that will save its prefill from one that
    // will not — the two behave identically until the NEXT boot, and the flag
    // spent weeks silently off because the station passed the wrong one.
    json(res, 200, { ok: true, hosting: spec.id, native: true, busy, ctx: spec.maxContext, pool: POOL, experts: EXPERTS, kv8: KV8, last: lastStats, live })
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
  // CLIENT WENT AWAY. Closing the connection is how every OpenAI client
  // cancels, and it never reached the engine: a cancelled request kept the
  // GPU for the rest of its prefill — minutes, for an answer nobody would
  // read — and the station kept saying "generating". `close` also fires on a
  // normal finish, so the writableEnded check is what distinguishes them.
  let aborted = false
  const onClose = () => { if (!res.writableEnded) aborted = true }
  res.on('close', onClose)
  const isAborted = () => aborted
  try {
    if (body.stream === true) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' })
      const send = (o) => res.write(`data: ${JSON.stringify(o)}\n\n`)
      const chunk = (delta, finish = null) => send({
        id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model: spec.id,
        choices: [{ index: 0, delta, finish_reason: finish }],
      })
      chunk({ role: 'assistant', content: '' })
      const done = await runJob(body, (t) => chunk({ content: t }), isAborted)
      if (done.toolCalls.length) chunk({ tool_calls: done.toolCalls.map((c, i) => ({ index: i, ...c })) })
      chunk({}, done.finishReason)
      if (body.stream_options?.include_usage) {
        send({ id: `chatcmpl-${id}`, object: 'chat.completion.chunk', created, model: spec.id, choices: [], usage: done.usage })
      }
      res.write('data: [DONE]\n\n')
      res.end()
    } else {
      const done = await runJob(body, null, isAborted)
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
    if (aborted) console.log('[native] client disconnected — generation stopped')
    res.off?.('close', onClose)
    busy = false
    scheduleSave()
  }
}).listen(PORT, '127.0.0.1', () => {
  console.log(`[native] OpenAI surface: http://127.0.0.1:${PORT}/v1  (model id "${spec.id}" or "ztvm")`)
  console.log(`[native] panel:          http://127.0.0.1:${PORT}/`)
})
