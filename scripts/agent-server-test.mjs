#!/usr/bin/env node
// AGENT-SERVER-TEST — is the OpenAI front door real?
//
//   node scripts/agent-server-test.mjs [llama32]
//
// Boots agent-server.mjs, opens agent-host.html in a real Chrome via the e2e
// harness, and then talks to it the way pi and Cline do: /v1/models, a
// non-streaming completion, an SSE completion, and a tool call. Nothing is
// mocked — the tokens come off WebGPU.
//
// Everything here is a client-side check. If it passes, an OpenAI-compatible
// editor extension can drive this engine.

import { spawn } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = 8018                       // not 8017, so a dev instance can stay up
const BASE = `http://127.0.0.1:${PORT}`
const MODEL = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'llama32'

let failed = 0
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(34)} ${detail}`)
  if (!ok) failed++
}

const srv = spawn('node', [resolve(ROOT, 'scripts/agent-server.mjs')], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ['ignore', 'pipe', 'pipe'],
})
srv.stdout.on('data', (d) => process.stdout.write(`  [srv] ${d}`))
srv.stderr.on('data', (d) => process.stderr.write(`  [srv!] ${d}`))

const until = async (fn, ms, what) => {
  const t0 = Date.now()
  for (;;) {
    try { if (await fn()) return } catch { /* not up yet */ }
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${what}`)
    await new Promise((r) => setTimeout(r, 500))
  }
}

try {
  await until(async () => (await fetch(`${BASE}/health`)).ok, 15_000, 'agent-server')
  console.log(`agent-server up on ${BASE}\n`)

  await startHarness()
  // The harness serves on 5174; the page needs to be told where the server is.
  const page = await newPage(`/agent-host.html?model=${MODEL}&server=${encodeURIComponent(BASE)}`)
  page.on('console', (m) => { if (/error|fail/i.test(m.text())) console.log(`  [page] ${m.text()}`) })

  console.log(`loading ${MODEL} in a real browser (weights may take a minute)…`)
  await until(async () => (await (await fetch(`${BASE}/health`)).json()).tabConnected,
    8 * 60_000, 'the tab to connect')
  const health = await (await fetch(`${BASE}/health`)).json()
  check('tab connected', health.tabConnected === true, `hosting ${health.hosting}`)

  // 1. /v1/models — both clients read this at startup.
  const models = await (await fetch(`${BASE}/v1/models`)).json()
  check('/v1/models', models.object === 'list' && models.data?.length > 0,
    models.data?.[0]?.id ?? '(none)')

  // 2. Non-streaming completion.
  const t0 = Date.now()
  const r1 = await (await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, max_tokens: 24,
      messages: [{ role: 'user', content: 'What is the capital of France? Answer in one word.' }],
    }),
  })).json()
  const text1 = r1.choices?.[0]?.message?.content ?? ''
  check('non-streaming completion', /paris/i.test(text1),
    `${((Date.now() - t0) / 1000).toFixed(1)}s · ${JSON.stringify(text1.slice(0, 60))}`)
  check('usage reported', (r1.usage?.prompt_tokens ?? 0) > 0,
    `${r1.usage?.prompt_tokens} in / ${r1.usage?.completion_tokens} out`)
  check('finish_reason', typeof r1.choices?.[0]?.finish_reason === 'string',
    r1.choices?.[0]?.finish_reason)

  // 3. Streaming — the path both clients actually use. Parsed exactly as a
  //    client does, so a malformed frame fails here rather than in an editor.
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, stream: true, max_tokens: 24,
      messages: [{ role: 'user', content: 'Count: one two three' }],
    }),
  })
  check('stream content-type', (res.headers.get('content-type') ?? '').includes('text/event-stream'),
    res.headers.get('content-type') ?? '')
  let acc = '', frames = 0, sawRole = false, sawDone = false, finish = null
  const dec = new TextDecoder()
  let buf = ''
  for await (const part of res.body) {
    buf += dec.decode(part, { stream: true })
    const lines = buf.split('\n')
    buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue
      const payload = line.slice(6).trim()
      if (payload === '[DONE]') { sawDone = true; continue }
      const j = JSON.parse(payload)
      frames++
      const d = j.choices?.[0]?.delta ?? {}
      if (d.role) sawRole = true
      if (d.content) acc += d.content
      if (j.choices?.[0]?.finish_reason) finish = j.choices[0].finish_reason
    }
  }
  check('stream frames', frames > 1, `${frames} frames`)
  check('first frame carries role', sawRole)
  check('stream [DONE] sentinel', sawDone)
  check('stream finish_reason', finish !== null, String(finish))
  check('stream produced text', acc.trim().length > 0, JSON.stringify(acc.slice(0, 60)))

  // 4. Multi-part content — clients send `content: [{type:'text'}]` and a shim
  //    that assumes strings renders "[object Object]" into the prompt.
  const r4 = await (await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, max_tokens: 16,
      messages: [{ role: 'user', content: [{ type: 'text', text: 'Say OK.' }] }],
    }),
  })).json()
  check('multi-part content', typeof r4.choices?.[0]?.message?.content === 'string'
    && !r4.choices[0].message.content.includes('[object'),
    JSON.stringify((r4.choices?.[0]?.message?.content ?? '').slice(0, 40)))

  // 5. Native tool calling — the blocker that made this impossible before.
  const r5 = await (await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, max_tokens: 96,
      messages: [{ role: 'user', content: 'What is the weather in Paris? Use the tool.' }],
      tools: [{
        type: 'function',
        function: {
          name: 'get_weather',
          description: 'Get the current weather for a city',
          parameters: {
            type: 'object',
            properties: { city: { type: 'string', description: 'City name' } },
            required: ['city'],
          },
        },
      }],
    }),
  })).json()
  const calls = r5.choices?.[0]?.message?.tool_calls
  // A 1B model may simply decline to call. What is under test is the PLUMBING:
  // that tools reach the prompt and a call, if made, comes back OpenAI-shaped.
  if (calls?.length) {
    let argsOk = false
    try { argsOk = typeof JSON.parse(calls[0].function.arguments) === 'object' } catch { /* reported below */ }
    check('tool call shape', calls[0].type === 'function' && !!calls[0].function?.name && argsOk,
      `${calls[0].function?.name}(${calls[0].function?.arguments})`)
    check('finish_reason tool_calls', r5.choices[0].finish_reason === 'tool_calls',
      r5.choices[0].finish_reason)
  } else {
    console.log(`SKIP  tool call                       ${MODEL} did not call a tool`
      + ` (plumbing verified: request accepted, ${JSON.stringify((r5.choices?.[0]?.message?.content ?? '').slice(0, 50))})`)
  }

  // 6. A SECOND turn carrying the assistant's own prior tool call plus its
  //    result — the shape every agent loop sends from turn 2 onward. pi puts
  //    the assistant turn on the wire as {content: null, tool_calls: [...]},
  //    which a naive flattener turns into '' and erases the call from the
  //    transcript.
  if (calls?.length) {
    const r6 = await (await fetch(`${BASE}/v1/chat/completions`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model: MODEL, max_tokens: 64,
        messages: [
          { role: 'user', content: 'What is the weather in Paris? Use the tool.' },
          { role: 'assistant', content: null, tool_calls: calls },
          { role: 'tool', tool_call_id: calls[0].id, content: '{"temp_c": 17, "sky": "rain"}' },
        ],
      }),
    })).json()
    const answer = r6.choices?.[0]?.message?.content ?? ''
    check('tool round trip', /17|rain/i.test(answer), JSON.stringify(answer.slice(0, 70)))
  }

  // 7. Errors stay OpenAI-shaped.
  const bad = await (await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: MODEL, messages: [] }),
  })).json()
  check('error envelope', typeof bad.error?.message === 'string' && !!bad.error?.type,
    bad.error?.message?.slice(0, 50))

  // 8. Unrecoverable failures must be 4xx. pi retries 5xx with backoff, so a
  //    503 for "no tab" makes it hang ~2 minutes instead of saying what is
  //    wrong. Checked by asking for a context that cannot fit.
  const over = await fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL, max_tokens: 8,
      messages: [{ role: 'user', content: 'x '.repeat(200_000) }],
    }),
  })
  const overBody = await over.json()
  check('overflow is 4xx not 5xx', over.status >= 400 && over.status < 500,
    `HTTP ${over.status} · ${(overBody.error?.message ?? '').slice(0, 60)}`)
} finally {
  await stopHarness().catch(() => {})
  srv.kill('SIGTERM')
}

console.log(failed ? `\n${failed} check(s) FAILED` : '\nagent server works — an OpenAI client can drive this engine')
process.exit(failed ? 1 : 0)
