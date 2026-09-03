#!/usr/bin/env node
// ROOM-ROUTING-TEST — the multi-host room logic, without browsers.
//
// The signaling DO decides which host serves which guest, and re-decides when
// a host disappears. That is plain routing logic, so it is tested with plain
// WebSocket clients against `wrangler dev`: seconds, deterministic, and it
// exercises exactly the code paths a two-machine swarm depends on.
//
// (An earlier version drove two real hosts in one browser. Two full engines on
// one GPU hangs long before it proves anything about routing — the wrong test
// for this logic. The browser path is covered by scripts/share-e2e.mjs.)
//
//   node scripts/room-routing-test.mjs

import { spawn } from 'node:child_process'
import { resolve } from 'node:path'
// Node's own global WebSocket (>=22) — no dependency for a test of a protocol.

const ROOT = resolve(import.meta.dirname, '..')
const PORT = 8790
const ROOM = 'routing-test-room-0001'
const BASE = `ws://localhost:${PORT}/room/${ROOM}`

let failed = false
const check = (name, pass, detail = '') => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(30)} ${detail}`)
  if (!pass) failed = true
}

/** A test peer: collects every message, and can wait for one by predicate. */
function peer(role, label) {
  const ws = new WebSocket(`${BASE}?role=${role}`)
  const seen = []
  const waiters = []
  ws.addEventListener('message', (e) => {
    const msg = JSON.parse(String(e.data))
    seen.push(msg)
    for (let i = waiters.length - 1; i >= 0; i--) {
      if (waiters[i].pred(msg)) { waiters.splice(i, 1)[0].resolve(msg) }
    }
  })
  return {
    label, ws, seen,
    open: () => new Promise((r) => ws.addEventListener('open', r, { once: true })),
    send: (obj) => ws.send(JSON.stringify(obj)),
    close: () => ws.close(),
    /** Resolve with the first message matching `pred` (past or future). */
    expect(pred, what, timeoutMs = 5000) {
      const hit = seen.find(pred)
      if (hit) return Promise.resolve(hit)
      return new Promise((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`${label}: never saw ${what}`)), timeoutMs)
        waiters.push({ pred, resolve: (m) => { clearTimeout(t); resolve(m) } })
      })
    },
  }
}

const procs = []
try {
  console.log('starting wrangler dev …')
  const w = spawn('npx', ['wrangler', 'dev', '--port', String(PORT)],
    { cwd: resolve(ROOT, 'workers/share-signal'), stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, WRANGLER_SEND_METRICS: 'false' } })
  procs.push(w)
  for (let i = 0; ; i++) {
    try { await fetch(`http://localhost:${PORT}/`); break } catch {
      if (i > 120) throw new Error('wrangler dev never came up')
      await new Promise((r) => setTimeout(r, 250))
    }
  }

  // ── a guest with nobody serving ──
  const early = peer('guest', 'early guest')
  await early.open()
  await early.expect((m) => m.type === 'no-host', 'no-host')
  check('empty room says no-host', true)

  // ── first host arrives and adopts the waiting guest ──
  const hostA = peer('host', 'host A')
  await hostA.open()
  await early.expect((m) => m.type === 'host-changed', 'host-changed after a host appeared')
  const adopted = await hostA.expect((m) => m.type === 'peer-joined', 'peer-joined for the waiting guest')
  check('late host adopts waiter', true, `guest ${adopted.from}`)
  const guest1Id = adopted.from

  // ── second host, second guest: least-loaded assignment ──
  const hostB = peer('host', 'host B')
  await hostB.open()
  await hostA.expect((m) => m.type === 'room' && m.hosts === 2, 'room with 2 hosts')
  check('room counts hosts', true, '2 hosts announced')

  const guest2 = peer('guest', 'guest 2')
  await guest2.open()
  const joined2 = await hostB.expect((m) => m.type === 'peer-joined', 'peer-joined on the idle host')
  check('least-loaded assignment', true, `second guest went to host B (${joined2.from})`)
  check('not double-assigned', !hostA.seen.some((m) => m.type === 'peer-joined' && m.from === joined2.from),
    'host A was not offered the second guest')

  // ── relaying both ways, to the ASSIGNED host only ──
  guest2.send({ type: 'answer', sdp: { type: 'answer', sdp: 'from-guest-2' } })
  const relayed = await hostB.expect((m) => m.type === 'answer', 'guest 2 answer')
  check('guest → assigned host', relayed.from === joined2.from, `from ${relayed.from}`)
  check('other host untouched', !hostA.seen.some((m) => m.type === 'answer'), 'host A saw no answer')

  hostB.send({ to: joined2.from, type: 'offer', sdp: { type: 'offer', sdp: 'from-host-b' } })
  const offered = await guest2.expect((m) => m.type === 'offer', 'offer from host B')
  check('host → its guest', offered.sdp.sdp === 'from-host-b', 'offer arrived')

  // ── relay hardening: hosts cannot spoof relay-originated control types ──
  const hostChangedBefore = early.seen.filter((m) => m.type === 'host-changed').length
  hostA.send({ to: guest1Id, type: 'host-changed' })
  await new Promise((r) => setTimeout(r, 400))
  check('host cannot spoof host-changed',
    early.seen.filter((m) => m.type === 'host-changed').length === hostChangedBefore,
    'relay dropped spoofed control message')

  // ── churn: host B disappears mid-session ──
  // NOTE (not a check): assign()'s dead-host fallback cannot be exercised
  // from outside — the relay is single-threaded, so no host vanishes between
  // leastLoadedHost() and assign() at any existing call site. It is
  // defense-in-depth for a future async refactor, verified by reading
  // worker.js, not by this harness. A test asserting "dead host got no
  // peer-joined" would pass on code without the fallback and prove nothing.
  hostB.close()
  await guest2.expect((m) => m.type === 'host-changed', 'host-changed after its host died')
  const takenOver = await hostA.expect((m) => m.type === 'peer-joined' && m.from === joined2.from,
    'peer-joined for the orphaned guest')
  check('takeover on host death', true, `host A picked up guest ${takenOver.from}`)
  await hostA.expect((m) => m.type === 'room' && m.hosts === 1, 'room back to 1 host')

  // The takeover must survive: the orphan can talk to its new host.
  guest2.send({ type: 'ice', candidate: { candidate: 'after-takeover' } })
  const afterIce = await hostA.expect((m) => m.type === 'ice' && m.from === joined2.from, 'ice after takeover')
  check('orphan talks to survivor', afterIce.candidate.candidate === 'after-takeover')

  // ── last host leaves: guests are told, not left hanging ──
  hostA.close()
  await guest2.expect((m) => m.type === 'host-left', 'host-left when the room empties')
  await early.expect((m) => m.type === 'host-left', 'host-left on the other guest too')
  check('empty room tells guests', true, 'both guests notified')

  // ── a guest leaving is reported to its host ──
  const hostC = peer('host', 'host C')
  await hostC.open()
  const g3 = await hostC.expect((m) => m.type === 'peer-joined', 'reassigned guest on the new host')
  guest2.close()
  early.close()
  await hostC.expect((m) => m.type === 'peer-left', 'peer-left when a guest closes')
  check('guest departure reported', true, `after ${g3.from} was adopted`)
  hostC.close()
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  for (const p of procs) { try { p.kill('SIGTERM') } catch { /* gone */ } }
  await new Promise((r) => setTimeout(r, 300))
  for (const p of procs) { try { p.kill('SIGKILL') } catch { /* gone */ } }
}
console.log(failed ? '\nroom routing FAILED' : '\nroom routing correct — assignment, relaying, takeover, departure')
process.exit(failed ? 1 : 0)
