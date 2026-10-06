#!/usr/bin/env node
// AGENT-QUEUE-TEST — the server serves one job at a time, in order.
//
//   node scripts/agent-queue-test.mjs
//
// The engine has ONE KV cache: two concurrent generations interleave and
// corrupt each other silently. So this server must hold the second HTTP
// request open until the first finishes, never dispatch two jobs to the tab
// at once, and drop a client that disconnects while waiting. All of that is
// HTTP + SSE mechanics — no browser, no GPU, no weights. The fake tab below
// speaks the same two endpoints the real agent-host.html does (/agent/jobs
// SSE down, /agent/emit POST up) and answers with canned tokens.
//
// Fails LOUDLY on the old behavior: without the queue the second POST is
// dispatched immediately, so "B waits while A runs" goes red.

import { spawn } from 'node:child_process'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import http from 'node:http'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
// 8899, not 8021: launchd holds 8021 on macOS. Not 8017/8018/8019 (station,
// agent-server-test, station engine child) or the vite/relay ports.
const PORT = Number(process.env.PORT) || 8899
const BASE = `http://127.0.0.1:${PORT}`

let failed = 0
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(40)} ${detail}`)
  if (!ok) failed++
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const srv = spawn('node', [resolve(ROOT, 'scripts/agent-server.mjs')], {
  env: { ...process.env, PORT: String(PORT) },
  stdio: ['ignore', 'ignore', 'inherit'],
})

async function until(fn, ms, what) {
  const t0 = Date.now()
  for (;;) {
    try { if (await fn()) return } catch { /* not up yet */ }
    if (Date.now() - t0 > ms) throw new Error(`timed out waiting for ${what}`)
    await sleep(200)
  }
}

function post(path, body, { signal } = {}) {
  return fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

/** Fake browser tab: opens the jobs SSE stream, calls onJob per frame. */
function fakeTab(onJob) {
  let buf = ''
  const req = http.get(`${BASE}/agent/jobs?model=queue-test`, (res) => {
    res.on('data', (c) => {
      buf += c.toString()
      let i
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, i)
        buf = buf.slice(i + 2)
        for (const line of frame.split('\n')) {
          if (!line.startsWith('data: ')) continue
          try { onJob(JSON.parse(line.slice(6))) } catch { /* : connected */ }
        }
      }
    })
  })
  return { close: () => req.destroy() }
}

const emit = (msg) => post('/agent/emit', msg).then((r) => r.json())
const done = (id, text) => emit({
  id, type: 'done', text, finishReason: 'stop',
  usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 },
})
const completion = (text) => post('/v1/chat/completions', {
  messages: [{ role: 'user', content: text }],
})

try {
  await until(async () => (await fetch(`${BASE}/health`)).ok, 15_000, 'agent-server')

  // No tab: the request is refused, not queued forever.
  {
    const r = await completion('no tab yet')
    check('no tab -> 400, nothing queued', r.status === 400, `status ${r.status}`)
  }

  const seen = []
  const tab = fakeTab((job) => { if (job?.id) seen.push(job.id) })
  await sleep(300)

  // Two overlapping requests: A dispatches at once, B waits.
  const pA = completion('first question').then(async (r) => ({ status: r.status, body: await r.json() }))
  await until(() => seen.length >= 1, 3000, 'job A dispatched')
  const pB = completion('second question').then(async (r) => ({ status: r.status, body: await r.json() }))
  await sleep(1500)
  check('B waits while A runs', seen.length === 1, `${seen.length} job(s) at tab`)
  {
    const h = await (await fetch(`${BASE}/health`)).json()
    check('health shows 1 queued', h.queued === 1 && h.pending === 1, JSON.stringify(h))
  }

  // Finishing A serves B next, in order, with B's own answer.
  await done(seen[0], 'answer-a')
  const a = await pA
  check('A resolves 200 with its text',
    a.status === 200 && a.body.choices[0].message.content === 'answer-a',
    `status ${a.status}`)
  await until(() => seen.length >= 2, 3000, 'job B dispatched')
  await done(seen[1], 'answer-b')
  const b = await pB
  check('B resolves 200 with its text, after A',
    b.status === 200 && b.body.choices[0].message.content === 'answer-b',
    `status ${b.status}`)

  // A client that disconnects while queued never reaches the tab.
  const seenBefore = seen.length
  const pC = completion('third question').then(async (r) => ({ status: r.status, body: await r.json() }))
  await until(() => seen.length >= seenBefore + 1, 3000, 'job C dispatched')
  const ctl = new AbortController()
  const pD2 = fetch(`${BASE}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages: [{ role: 'user', content: 'abandoned' }] }),
    signal: ctl.signal,
  }).then((r) => r.json()).catch((e) => ({ aborted: String(e.name) }))
  await sleep(1000)
  ctl.abort()
  const d = await pD2
  // Deterministic gate: the server must have PROCESSED the abort (dropped D
  // from its queue) before C finishes — otherwise done(C) can pump D first
  // and the assertion below races network ordering, not queue behavior.
  await until(async () => (await (await fetch(`${BASE}/health`)).json()).queued === 0,
    3000, 'server drops aborted wait')
  await done(seen[seen.length - 1], 'answer-c')
  await pC
  await sleep(800)
  check('abandoned queued request never dispatched',
    seen.length === seenBefore + 1 && d.aborted === 'AbortError',
    `tab saw ${seen.length - seenBefore} new job(s)`)
  {
    const h = await (await fetch(`${BASE}/health`)).json()
    check('queue drains to zero', h.queued === 0 && h.pending === 0, JSON.stringify(h))
  }
  tab.close()
} finally {
  srv.kill()
}
console.log(failed ? '\nQUEUE BROKEN' : '\none at a time, in order')
process.exit(failed ? 1 : 0)
