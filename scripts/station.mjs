#!/usr/bin/env node
/**
 * STATION — the desktop-style control surface for zero-tvm.
 *
 *   npm run station              # then open http://127.0.0.1:8017/
 *
 * The piece that did not exist before: something that OWNS model lifecycle.
 * `npm run agent` spawns an engine and walks away, so loading a different
 * model meant killing a process by hand and every client's endpoint went
 * down with it. This supervisor holds the port, runs the engine as a CHILD
 * on an internal port, and proxies /v1 through — so a model swap never
 * changes the URL a client is configured with, exactly like LM Studio.
 *
 *   browser / Cline ──▶ :8017 (station) ──proxy──▶ :8019 (agent-native)
 *                        owns UI + lifecycle        the engine, one model
 *
 * Everything the UI shows is measured or registry-derived: sizes and context
 * ceilings come from the spec, rates from BENCH.md's labels (blank when a
 * model has never been measured), and live throughput from the engine's own
 * per-request record. The station estimates nothing.
 */

import { createServer, request as httpRequest } from 'node:http'
import { spawn } from 'node:child_process'
import { readFileSync, writeFileSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { sampleMemory } from './native/sysmem.mjs'
import { SHIPPED_MODELS, modelBranding, specWithCtx } from '../src/zero-tvm/model-registry.ts'
import { stationUi } from './native/station-ui.mjs'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const PORT = Number(process.env.PORT) || 8017
/** The engine child listens here; only the station talks to it. */
const ENGINE_PORT = Number(process.env.ENGINE_PORT) || 8019

// ── the catalogue the UI renders ────────────────────────────────────────────
// One row per shipped chat model, with the numbers that decide whether it
// fits: resident weights, the context the spec ships with, the checkpoint's
// trained ceiling, and what a token of KV costs. `rate` is measured or blank.
const CATALOGUE = SHIPPED_MODELS
  .filter((m) => !m.spec.embeddingOnly && !modelBranding(m.spec).pending)
  .map(({ param, spec }) => {
    const b = modelBranding(spec)
    const gb = /([\d.]+)\s*GB/.exec(b.sizeLabel)
    return {
      param,
      id: spec.id,
      name: b.name,
      params: b.params,
      sizeLabel: b.sizeLabel,
      weightsGb: gb ? parseFloat(gb[1]) : null,
      rateLabel: b.rateLabel,
      ramNote: b.ramNote ?? '',
      defaultCtx: spec.maxContext,
      // Never below what the spec already ships: KV pages round UP, so Phi-3's
      // table holds 4112 tokens against a 4096-token trained window, and a
      // naive clamp to maxSeq would SHRINK the shipped default.
      maxCtx: Math.max(spec.maxSeq, spec.maxContext),
      kvBytesPerToken: spec.kvBytesPerToken,
      moe: !!spec.moe,
      poolModes: (b.poolModes ?? []).map((p) => ({ slots: p.slots, label: p.label })),
    }
  })

// ── engine lifecycle ────────────────────────────────────────────────────────
let child = null
/** 'idle' | 'loading' | 'ready' | 'failed' */
let phase = 'idle'
let loaded = null          // { param, ctx, pool }
let logLines = []          // the child's recent output, for the UI
let failure = ''
/** Completed requests, newest first — the trend is the point: decode drifting
 *  down or TTFT climbing as a conversation fills the window. */
let history = []
let lastSeenAt = null
/** Latest system-memory sample (macOS). null elsewhere; the UI omits the row. */
let memory = null
/** What to reload after a restart, so a reboot does not leave clients on 503. */
const REMEMBER = join(ROOT, '.station.json')

const pushLog = (s) => {
  for (const line of String(s).split('\n')) {
    const t = line.trimEnd()
    if (t) logLines.push(t)
  }
  if (logLines.length > 60) logLines = logLines.slice(-60)
}

function stopEngine() {
  if (!child) return
  const c = child
  child = null
  phase = 'idle'
  loaded = null
  try { c.kill('SIGTERM') } catch { /* already gone */ }
  setTimeout(() => { try { c.kill('SIGKILL') } catch { /* gone */ } }, 2500)
}

/** Wait for the child's own /health to answer — the engine is ready only when
 *  it says so, never when the process merely started. */
async function waitReady(timeoutMs = 12 * 60_000) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (!child) return false                   // cancelled
    if (child.exitCode !== null) return false  // died while loading
    try {
      const r = await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`,
        { signal: AbortSignal.timeout(1500) })
      if ((await r.json()).ok) return true
    } catch { /* still booting */ }
    await new Promise((r) => setTimeout(r, 700))
  }
  return false
}

async function loadModel({ param, ctx, pool }) {
  stopEngine()
  await new Promise((r) => setTimeout(r, 400))   // let the port free
  logLines = []
  failure = ''
  history = []          // a new engine's numbers are not the old one's
  lastSeenAt = null
  phase = 'loading'
  loaded = { param, ctx: ctx || 0, pool: pool || 0 }
  const argv = [join(ROOT, 'scripts/agent-native.mjs'), param, '--port', String(ENGINE_PORT)]
  if (ctx) argv.push('--ctx', String(ctx))
  // `pool` here is EXPERT SLOTS (a memory build), not the KV prefix pool.
  // This used to pass it as --pool, so the default build (slots 0) sent
  // `--pool 0` and silently turned OFF the disk cache that lets a prefill
  // survive a restart — the one thing that makes a reload cheap.
  if (pool) argv.push('--experts', String(pool))
  pushLog(`$ node ${argv.slice(1).join(' ')}`)
  child = spawn('node', argv, { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'] })
  child.stdout.on('data', pushLog)
  child.stderr.on('data', pushLog)
  child.on('exit', (code) => {
    if (phase === 'loading' || phase === 'ready') {
      phase = 'failed'
      failure = `the engine exited (code ${code}) — see the log`
    }
    child = null
  })
  const ok = await waitReady()
  if (!ok) {
    if (phase !== 'failed') { phase = 'failed'; failure = failure || 'the engine did not come up in time' }
    stopEngine()
    return false
  }
  phase = 'ready'
  remember({ param, ctx: ctx || 0, pool: pool || 0 })
  return true
}

function remember(choice) {
  try { writeFileSync(REMEMBER, JSON.stringify(choice)) } catch { /* not fatal */ }
}
function forget() {
  try { rmSync(REMEMBER, { force: true }) } catch { /* not fatal */ }
}

// Poll the engine for its per-request record and accumulate the ones we have
// not seen. The engine keeps only the LAST; the trend lives here.
setInterval(async () => {
  if (phase !== 'ready') return
  try {
    const h = await (await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`,
      { signal: AbortSignal.timeout(1200) })).json()
    const l = h.last
    if (l && l.at !== lastSeenAt) {
      lastSeenAt = l.at
      history.unshift({ ...l, model: loaded?.param ?? '' })
      if (history.length > 20) history.length = 20
    }
  } catch { /* mid-restart */ }
}, 1000)

setInterval(async () => { memory = await sampleMemory() }, 3000)
void sampleMemory().then((m) => { memory = m })

// ── proxy: everything the engine owns ───────────────────────────────────────
function proxy(req, res) {
  const p = httpRequest(
    { host: '127.0.0.1', port: ENGINE_PORT, path: req.url, method: req.method, headers: req.headers },
    (up) => { res.writeHead(up.statusCode ?? 502, up.headers); up.pipe(res) },
  )
  p.on('error', (e) => {
    res.writeHead(502, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ error: { message: `engine unreachable: ${e.message}`, type: 'server_error' } }))
  })
  req.pipe(p)
}

const json = (res, code, body) => {
  res.writeHead(code, { 'content-type': 'application/json' })
  res.end(JSON.stringify(body))
}
const readJson = (req) => new Promise((resolve, reject) => {
  let b = ''
  req.on('data', (c) => { b += c; if (b.length > 1e6) reject(new Error('body too large')) })
  req.on('end', () => { try { resolve(b ? JSON.parse(b) : {}) } catch (e) { reject(e) } })
})

createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`)
  res.setHeader('access-control-allow-origin', '*')
  res.setHeader('access-control-allow-headers', 'content-type, authorization')
  if (req.method === 'OPTIONS') { res.writeHead(204).end(); return }

  // The UI.
  if (url.pathname === '/' && (req.headers.accept ?? '').includes('text/html')) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
    res.end(stationUi())
    return
  }

  // State for the UI: catalogue + what is loaded + the engine's own stats.
  if (url.pathname === '/api/state') {
    let engine = null
    if (phase === 'ready') {
      try {
        engine = await (await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`,
          { signal: AbortSignal.timeout(1500) })).json()
      } catch { /* mid-restart */ }
    }
    json(res, 200, { phase, loaded, failure, log: logLines.slice(-24), engine,
      history, memory, models: CATALOGUE, port: PORT })
    return
  }

  if (url.pathname === '/api/load' && req.method === 'POST') {
    let body
    try { body = await readJson(req) } catch { return json(res, 400, { error: 'bad JSON' }) }
    const hit = CATALOGUE.find((m) => m.param === body.param)
    if (!hit) return json(res, 400, { error: `unknown model "${body.param}"` })
    if (phase === 'loading') return json(res, 409, { error: 'already loading' })
    // Refuse to yank an engine out from under a client mid-generation.
    if (phase === 'ready') {
      try {
        const h = await (await fetch(`http://127.0.0.1:${ENGINE_PORT}/health`,
          { signal: AbortSignal.timeout(1500) })).json()
        if (h.busy) return json(res, 409, { error: 'the loaded model is generating right now — wait for it to finish' })
      } catch { /* unreachable; replacing it is fine */ }
    }
    // Clamp here as well as in the engine, so the UI can show what it will get.
    const ctx = Math.min(Number(body.ctx) || hit.defaultCtx, hit.maxCtx)
    json(res, 202, { accepted: { param: hit.param, ctx, pool: Number(body.pool) || 0 } })
    void loadModel({ param: hit.param, ctx, pool: Number(body.pool) || 0 })
    return
  }

  if (url.pathname === '/api/unload' && req.method === 'POST') {
    stopEngine()
    forget()            // an explicit clear also cancels auto-load
    history = []
    lastSeenAt = null
    json(res, 200, { ok: true })
    return
  }

  if (url.pathname === '/api/history/clear' && req.method === 'POST') {
    history = []
    json(res, 200, { ok: true })
    return
  }

  if (url.pathname === '/health') {
    json(res, 200, { ok: true, station: true, phase, loaded, engineOn: phase === 'ready' })
    return
  }

  // Everything else belongs to the engine (/v1/*).
  if (phase !== 'ready') {
    return json(res, 503, {
      error: {
        message: phase === 'loading'
          ? `loading ${loaded?.param ?? 'a model'} — try again shortly`
          : 'no model is loaded — open the station UI and load one',
        type: 'server_error',
      },
    })
  }
  proxy(req, res)
}).listen(PORT, '127.0.0.1', () => {
  console.log(`[station] UI:       http://127.0.0.1:${PORT}/`)
  console.log(`[station] OpenAI:   http://127.0.0.1:${PORT}/v1   (stable across model swaps)`)
  console.log(`[station] ${CATALOGUE.length} models available — load one from the UI`)
  // Reload whatever was loaded last, so a restart does not leave every client
  // on 503 until someone opens a browser. `--no-autoload` opts out; an
  // explicit Clear in the UI forgets the choice.
  if (!process.argv.includes('--no-autoload')) {
    try {
      const saved = JSON.parse(readFileSync(REMEMBER, 'utf8'))
      if (CATALOGUE.some((m) => m.param === saved.param)) {
        console.log(`[station] auto-loading ${saved.param} (ctx ${saved.ctx || 'default'})`)
        void loadModel(saved)
      }
    } catch { /* nothing remembered — idle is correct */ }
  }
})

for (const sig of ['SIGINT', 'SIGTERM']) {
  process.on(sig, () => { stopEngine(); process.exit(0) })
}
