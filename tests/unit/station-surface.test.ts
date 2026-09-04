/**
 * UNIT TESTS — station HTTP surface (scripts/station.mjs).
 *
 * Spawns the real station with test ports and --no-autoload, so no model
 * loads and no GPU is needed. Pins the routing and error envelopes — the
 * exact class that shipped wrong before (wrong status, wrong shape):
 *   - /health, /api/state, /api/load validation, /api/unload
 *   - 503 idle vs loading messages on /v1/*
 *   - foreign-Origin 403, localhost passthrough
 *
 * Deliberately NOT covered here: loadModel/swap/kill lifecycle (spawns a
 * real agent-native engine — needs weights + GPU; cover in the e2e harness),
 * and the proxy-success path (requires phase 'ready').
 */

import { spawn, type ChildProcess } from 'node:child_process'
import { readFileSync, writeFileSync, rmSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { describe, test, expect, beforeAll, afterAll } from 'vitest'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const PORT = 18017
const ENGINE_PORT = 18019
const BASE = `http://127.0.0.1:${PORT}`
const REMEMBER = resolve(ROOT, '.station.json')

let srv: ChildProcess | null = null
let rememberBackup: string | null = null
let rememberExisted = false

async function untilUp(): Promise<void> {
  const t0 = Date.now()
  for (;;) {
    try {
      const r = await fetch(`${BASE}/health`)
      if (r.ok) return
    } catch { /* not up yet */ }
    if (Date.now() - t0 > 20_000) throw new Error('station did not come up')
    await new Promise((r) => setTimeout(r, 200))
  }
}

beforeAll(async () => {
  // /api/unload deletes ROOT/.station.json — back it up so this test never
  // eats the developer's remembered model.
  try {
    rememberBackup = readFileSync(REMEMBER, 'utf8')
    rememberExisted = true
  } catch { rememberExisted = false }
  srv = spawn('node', [resolve(ROOT, 'scripts/station.mjs'), '--no-autoload'], {
    env: { ...process.env, PORT: String(PORT), ENGINE_PORT: String(ENGINE_PORT) },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  await untilUp()
}, 30_000)

afterAll(async () => {
  srv?.kill('SIGTERM')
  await new Promise((r) => setTimeout(r, 500))
  try {
    if (rememberExisted) writeFileSync(REMEMBER, rememberBackup!)
    else rmSync(REMEMBER, { force: true })
  } catch { /* best effort */ }
})

describe('station surface (no engine loaded)', () => {
  test('/health reports idle station', async () => {
    const h = await (await fetch(`${BASE}/health`)).json()
    expect(h.ok).toBe(true)
    expect(h.station).toBe(true)
    expect(h.phase).toBe('idle')
    expect(h.engineOn).toBe(false)
  })

  test('/ serves the UI html', async () => {
    const r = await fetch(BASE + '/', { headers: { accept: 'text/html' } })
    expect(r.status).toBe(200)
    expect(r.headers.get('content-type')).toContain('text/html')
    const html = await r.text()
    expect(html.length).toBeGreaterThan(1000)
  })

  test('/api/state carries catalogue + idle phase', async () => {
    const s = await (await fetch(`${BASE}/api/state`)).json()
    expect(s.phase).toBe('idle')
    expect(s.loaded).toBeNull()
    expect(Array.isArray(s.models)).toBe(true)
    expect(s.models.length).toBeGreaterThan(0)
    expect(s.port).toBe(PORT)
  })

  test('/api/load rejects an unknown model with 400', async () => {
    const r = await fetch(`${BASE}/api/load`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ param: 'definitely-not-a-model' }),
    })
    expect(r.status).toBe(400)
    const b = await r.json()
    expect(typeof b.error).toBe('string')
    expect(b.error).toContain('definitely-not-a-model')
  })

  test('/api/load rejects malformed JSON with 400', async () => {
    const r = await fetch(`${BASE}/api/load`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: '{oops',
    })
    expect(r.status).toBe(400)
  })

  test('/v1/* with no engine is 503 with the idle message', async () => {
    const r = await fetch(`${BASE}/v1/models`)
    expect(r.status).toBe(503)
    const b = await r.json()
    expect(b.error?.type).toBe('server_error')
    expect(b.error?.message).toContain('no model is loaded')
  })

  test('foreign Origin is 403, localhost passes', async () => {
    const bad = await fetch(`${BASE}/health`, { headers: { origin: 'https://evil.example' } })
    expect(bad.status).toBe(403)
    const ok = await fetch(`${BASE}/health`, { headers: { origin: 'http://127.0.0.1:5173' } })
    expect(ok.status).toBe(200)
    expect(ok.headers.get('access-control-allow-origin')).toBe('http://127.0.0.1:5173')
  })

  test('/api/history/clear and /api/unload succeed idle', async () => {
    const c = await fetch(`${BASE}/api/history/clear`, { method: 'POST' })
    expect(c.status).toBe(200)
    const u = await fetch(`${BASE}/api/unload`, { method: 'POST' })
    expect(u.status).toBe(200)
    expect((await u.json()).ok).toBe(true)
  })
})
