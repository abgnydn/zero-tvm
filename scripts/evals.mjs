#!/usr/bin/env node
// EVALS — every check in one place: what it proves, what it covers, and a
// button that runs it while you watch.
//
// Why this exists. The checks were spread across nineteen npm scripts, five
// test directories and fifty-six standalone runners, in three output formats
// with three different meanings for a non-zero exit. release-check.mjs could
// see none of it — it shelled out and read $?, which is the least informative
// thing any of these produce. That is how a suite that ran 1 of 8 arms printed
// "correct", how one that ran 0 exited 0, and how a checklist row that could
// not run at all vanished from its own summary while the run still exited 0.
//
// So this streams the child's own output instead of reducing it to a code, and
// it shows each check's COVERAGE — the box it runs in — because the defect
// that cost this repo the most was a check that was true at six tokens against
// a shipped cap of 1024. That check passed. It always passed.
//
//   npm run evals          # then http://127.0.0.1:8021/
//
// One run at a time, deliberately: the mutation gate rewrites src/ and the GPU
// suites contend for one adapter. A queue is honest; parallel runs are not.

import { createServer } from 'node:http'
import { spawn } from 'node:child_process'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import { CHECKS, PARAMS, ROOT, byId, resolve } from './eval-registry.mjs'

const PORT = Number(process.env.EVALS_PORT ?? 8021)
const RUNS_DIR = join(ROOT, '.evals')
const HISTORY = join(RUNS_DIR, 'history.json')
if (!existsSync(RUNS_DIR)) mkdirSync(RUNS_DIR, { recursive: true })

/** Durable across restarts: which checks ran, when, with what, and how they
 *  ended. A verdict computed fresh each time would forget the run you did
 *  yesterday, which is exactly the state this repo was already in. */
const history = existsSync(HISTORY) ? JSON.parse(readFileSync(HISTORY, 'utf8')) : {}
const saveHistory = () => writeFileSync(HISTORY, JSON.stringify(history, null, 2))

let active = null            // { id, child, started, values, lines[] }
const listeners = new Set()  // SSE responses

const emit = (obj) => {
  const s = `data: ${JSON.stringify(obj)}\n\n`
  for (const r of listeners) { try { r.write(s) } catch { /* client gone */ } }
}

function startRun(id, values) {
  if (active) return { error: `already running: ${active.id}` }
  const check = byId(id)
  if (!check) return { error: `unknown check: ${id}` }

  const { cmd, cwd, env, values: used } = resolve(check, values)
  const started = Date.now()
  active = { id, started, values: used, lines: [], cmd, cwd }

  const child = spawn(cmd[0], cmd.slice(1), {
    cwd,
    env: { ...process.env, ...env, FORCE_COLOR: '0' },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  active.child = child

  emit({ type: 'start', id, cmd, cwd, values: used, started })

  const onData = (buf, stream) => {
    for (const line of buf.toString().split('\n')) {
      if (!line) continue
      active.lines.push(line)
      // Keep memory bounded on a suite that prints thousands of lines; the
      // full text is on disk regardless.
      if (active.lines.length > 4000) active.lines.splice(0, 1000)
      emit({ type: 'line', id, stream, line })
    }
  }
  child.stdout.on('data', (b) => onData(b, 'out'))
  child.stderr.on('data', (b) => onData(b, 'err'))

  child.on('close', (code, signal) => {
    const ms = Date.now() - started
    const text = active.lines.join('\n')
    // Exit code alone has lied three separate times in this repo, so record
    // what the output SAYS as well: a suite that skipped everything and one
    // that verified everything both exit 0.
    const skipped = (text.match(/^\s*SKIP\b/gm) ?? []).length
    const passed = (text.match(/^\s*PASS\b/gm) ?? []).length
    const failed = (text.match(/^\s*FAIL\b/gm) ?? []).length
    const hollow = code === 0 && passed === 0 && skipped > 0
    const verdict = signal ? 'killed'
      : code !== 0 ? 'fail'
      : hollow ? 'hollow'
      : 'pass'

    writeFileSync(join(RUNS_DIR, `${id}-${started}.log`), text)
    history[id] = { id, verdict, code, signal, ms, started, values: active.values,
      passed, failed, skipped, cmd, log: `${id}-${started}.log` }
    saveHistory()
    emit({ type: 'end', id, verdict, code, signal, ms, passed, failed, skipped })
    active = null
  })

  return { ok: true }
}

const send = (res, status, body, type = 'application/json') => {
  res.writeHead(status, { 'content-type': type, 'cache-control': 'no-store' })
  res.end(type === 'application/json' ? JSON.stringify(body) : body)
}

createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`)

  if (url.pathname === '/') {
    return send(res, 200, readFileSync(join(ROOT, 'scripts/evals.html'), 'utf8'), 'text/html; charset=utf-8')
  }

  if (url.pathname === '/api/checks') {
    return send(res, 200, {
      params: PARAMS,
      active: active && { id: active.id, started: active.started, values: active.values },
      checks: CHECKS.map((c) => ({
        id: c.id, name: c.name, needs: c.needs, group: c.group,
        proves: c.proves, covers: c.covers ?? {}, why: c.why,
        gap: c.gap, note: c.note, warn: c.warn,
        args: (c.args ?? []).map((a) => ({ ...a })),
        preview: resolve(c, {}).cmd.join(' '),
        last: history[c.id] ?? null,
      })),
    })
  }

  if (url.pathname === '/api/run' && req.method === 'POST') {
    const body = await new Promise((r) => {
      let b = ''; req.on('data', (d) => { b += d }); req.on('end', () => r(b))
    })
    const { id, values } = JSON.parse(body || '{}')
    const out = startRun(id, values ?? {})
    return send(res, out.error ? 409 : 200, out)
  }

  if (url.pathname === '/api/kill' && req.method === 'POST') {
    if (!active) return send(res, 409, { error: 'nothing running' })
    active.child.kill('SIGTERM')
    return send(res, 200, { ok: true })
  }

  if (url.pathname === '/api/log') {
    const id = url.searchParams.get('id')
    const h = history[id]
    if (!h) return send(res, 404, { error: 'no run' })
    return send(res, 200, readFileSync(join(RUNS_DIR, h.log), 'utf8'), 'text/plain; charset=utf-8')
  }

  if (url.pathname === '/api/stream') {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-store',
      connection: 'keep-alive',
    })
    res.write(': connected\n\n')
    listeners.add(res)
    // Replay the in-flight run so a page opened mid-run is not blank.
    if (active) {
      res.write(`data: ${JSON.stringify({ type: 'start', id: active.id, cmd: active.cmd,
        cwd: active.cwd, values: active.values, started: active.started })}\n\n`)
      for (const line of active.lines.slice(-400)) {
        res.write(`data: ${JSON.stringify({ type: 'line', id: active.id, stream: 'out', line })}\n\n`)
      }
    }
    const ping = setInterval(() => { try { res.write(': ping\n\n') } catch { /* gone */ } }, 20_000)
    req.on('close', () => { clearInterval(ping); listeners.delete(res) })
    return
  }

  send(res, 404, { error: 'not found' })
}).listen(PORT, '127.0.0.1', () => {
  console.log(`evals  →  http://127.0.0.1:${PORT}/`)
  console.log(`${CHECKS.length} checks · logs in .evals/`)
})
