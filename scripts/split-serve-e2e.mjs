#!/usr/bin/env node
// SPLIT-SERVE-E2E — one model, N machines, a real answer.
//
// Browser 0 holds layers [0, k1) and hosts the room; every other browser joins
// holding one later slice. None of them can answer anything alone. A guest tab
// then asks a question and the reply streams back, walking the whole chain
// once per token.
//
// Separate browser PROFILES because that is the only way to make the stages
// genuinely independent: separate GPU contexts, separate OPFS, separate engines
// that each loaded only their own layers.
//
//   node scripts/split-serve-e2e.mjs            # llama32, 2 stages at 8/16
//   STAGES=6 node scripts/split-serve-e2e.mjs   # six of them
//   MODEL=qwen3mlx SPLIT=18 node scripts/split-serve-e2e.mjs
//
// Helpers join in REVERSE order on purpose. Six tabs opened at once connect in
// whatever order they finish loading, so the host must hold a stage that does
// not continue the chain yet instead of refusing it — joining backwards means
// every single one arrives out of order, which is the worst case.
//
// Small model on purpose: the point is the transport, and six slices of a 1B
// fit on one machine's GPU. Splitting the 35B needs real machines.

import { spawn } from 'node:child_process'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const ROOT = resolve(import.meta.dirname, '..')
// VITE_PORT is overridable ONLY so the port guard below can be exercised on a
// scratch port without disturbing a real run on 5194.
const VITE_PORT = Number(process.env.VITE_PORT ?? 5194)
const SIGNAL_PORT = 8791
const MODEL = process.env.MODEL ?? 'llama32'
const spec = specForParam(MODEL)
const STAGES = Number(process.env.STAGES ?? 2)
/** PHONE=1 — run the guest tab as an emulated iPhone (see the guest block). */
const PHONE = process.env.PHONE === '1'
// With STAGES=2 the cut stays wherever SPLIT says, so the two-stage run keeps
// its old shape; beyond that the layers are divided evenly.
const BOUNDS = STAGES === 2
  ? [0, Number(process.env.SPLIT ?? Math.floor(spec.layers / 2)), spec.layers]
  : Array.from({ length: STAGES + 1 }, (_, i) => Math.round(i * spec.layers / STAGES))
const PROFILE = (i) => resolve(ROOT, `.tests-cache/chrome-stage-${i}`)
const BASE = `http://localhost:${VITE_PORT}`

const procs = []
function run(cmd, args, cwd) {
  const p = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, WRANGLER_SEND_METRICS: 'false' } })
  // Keep the tail of BOTH streams. stdout is where vite announces the port it
  // actually bound — the only evidence available that the thing answering is
  // the thing we started — and it was being piped and then never read, which
  // is also how a chatty child eventually blocks on a full pipe.
  p.log = ''
  const keep = (b) => { p.log = (p.log + b).slice(-4000) }
  p.stdout.on('data', keep)
  p.stderr.on('data', (b) => { keep(b); if (/error|EADDRINUSE/i.test(String(b))) process.stderr.write(`[${cmd}] ${b}`) })
  // A child that never started is a child that died. Without the 'error'
  // handler an ENOENT here is an unhandled event that takes the whole harness
  // down with a stack trace instead of a sentence.
  p.on('error', (e) => { p.dead = `failed to start (${e.code ?? e.message})` })
  p.on('exit', (code, sig) => { p.dead = `exited ${code ?? sig}` })
  procs.push(p)
  return p
}

/**
 * A SERVER THIS HARNESS DID NOT START IS NOT THIS HARNESS'S SERVER.
 *
 * `fetch(url)` proves that SOMETHING answers on the port. It does not prove it
 * is ours. On 2026-08-25 an orphaned vite held 5194 serving source two commits
 * behind; this file spawned its own vite, that vite died, and the wait
 * resolved against the STRANGER in 103 ms — after which every browser, every
 * assertion and the final PASS were about code the run never loaded. That is
 * this repo's signature defect: something occupying the position of a check
 * without performing one.
 *
 * `--strictPort` was already on and did not save it, for two measured reasons
 * (both reproduced on this machine, vite 6.4.1, macOS):
 *
 *   1. When the squatter binds the SAME loopback family, vite does print
 *      `Error: Port N is already in use` and exit 1 — but nothing here read
 *      the child's exit code, and it loses the race anyway: the squatter is
 *      already listening, so it answers the very first poll. Measured: the old
 *      `waitHttp` resolved against a decoy in 28 ms, well inside vite's
 *      startup.
 *   2. When the squatter binds the OTHER loopback family there is no
 *      EADDRINUSE AT ALL. vite binds `localhost`, which resolves to `[::1]`
 *      here; a squatter on `127.0.0.1` coexists with it happily, both report
 *      themselves up, and which one a client reaches is decided by whatever
 *      that client's resolver does with "localhost". Verified: `curl
 *      localhost:P` reached vite while `curl 127.0.0.1:P` reached the
 *      squatter, at the same instant.
 *
 * So EADDRINUSE cannot be the check. Two things replace it, and both are
 * needed: refuse a port that ALREADY answers (either family, since either can
 * be the one Chrome picks), and then wait for OUR child to announce itself
 * rather than for the port to respond.
 */
async function requirePortFree(port, what) {
  for (const host of ['127.0.0.1', '[::1]']) {
    const answered = await fetch(`http://${host}:${port}/`, { signal: AbortSignal.timeout(2000) })
      .then(() => true, () => false)
    if (!answered) continue
    throw new Error(
      `http://${host}:${port}/ already answers — something this harness did not start is `
      + `serving ${what}. Refusing to run: a previous version of this file would have tested `
      + `that server instead and printed PASS. Stop it (lsof -nP -iTCP:${port} -sTCP:LISTEN) `
      + `and rerun.`)
  }
}

/** vite's own ready line, pinned to the port we asked for. Matching this is
 *  what makes the server ours rather than merely present. */
const VITE_READY = new RegExp(`Local:\\s+https?://localhost:${VITE_PORT}/`)

/**
 * Wait for the server WE STARTED, not for the port to answer.
 *
 * `ready` is a regex over the child's own output; when it is given, no HTTP
 * response counts until the child has said the words itself. A child that
 * dies is fatal here rather than something to keep polling past.
 */
async function waitServer(proc, url, timeoutMs, what, ready) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (proc.dead) {
      throw new Error(`${what} ${proc.dead} before serving ${url}\n--- its output ---\n${proc.log.trim()}`)
    }
    if (!ready || ready.test(proc.log)) {
      try { await fetch(url); return } catch { /* bound but not serving yet */ }
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`timed out waiting for ${what} at ${url}\n--- its output ---\n${proc.log.trim()}`)
}
const launch = (userDataDir) => puppeteer.launch({
  headless: false,
  userDataDir,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--enable-dawn-features=allow_unsafe_apis'],
  defaultViewport: { width: 900, height: 700 },
  protocolTimeout: 15 * 60 * 1000,
})

/**
 * Every serving role asks before it downloads anything — #share-gate-go, the
 * same button share-e2e.mjs clicks. This script never did, and since it wipes
 * its profiles on the way out (line ~195) the weights are always cold, so the
 * gate always waits: stage 0 sat on "Checking this device…" until the 10
 * minute timeout. The run has been failing here rather than in anything it
 * was written to test.
 *
 * The button starts disabled while the device is probed, so wait for it to be
 * enabled rather than merely present.
 */
async function passGate(page, who) {
  await page.waitForSelector('#share-gate-go', { visible: true, timeout: 60_000 })
  await page.waitForFunction(() => {
    const b = document.getElementById('share-gate-go')
    return !!b && !b.disabled
  }, { timeout: 60_000, polling: 200 })
  await page.click('#share-gate-go')
  console.log(`${who}: download confirmed`)
}

let failed = false
/** browsers[0] hosts; the rest are helper stages. */
const browsers = []
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(22)} ${detail}`)
  if (!pass) failed = true
}

try {
  if (BOUNDS.some((b, i) => i && b <= BOUNDS[i - 1])) {
    throw new Error(`${spec.layers} layers cannot be cut into ${STAGES} non-empty stages`)
  }
  console.log(`${spec.id}: ${spec.layers} layers across ${STAGES} stages — `
    + BOUNDS.slice(0, -1).map((b, i) => `${b}-${BOUNDS[i + 1]}`).join(' → '))
  // Before anything is spawned: neither port may already be serving.
  await requirePortFree(SIGNAL_PORT, 'the signaling relay')
  await requirePortFree(VITE_PORT, 'the dev server')
  const signal = run('npx', ['wrangler', 'dev', '--port', String(SIGNAL_PORT)], resolve(ROOT, 'workers/share-signal'))
  const vite = run(resolve(ROOT, 'node_modules/.bin/vite'), ['--port', String(VITE_PORT), '--strictPort', '--clearScreen', 'false'], ROOT)
  await waitServer(signal, `http://localhost:${SIGNAL_PORT}/`, 30_000, 'wrangler dev', null)
  await waitServer(vite, `${BASE}/share.html`, 30_000, 'vite', VITE_READY)
  for (let i = 0; i < STAGES; i++) mkdirSync(PROFILE(i), { recursive: true })

  // ── stage 0: layers [0, BOUNDS[1]), hosts the room ──
  const b0 = await launch(PROFILE(0))
  browsers.push(b0)
  const stage0 = await b0.newPage()
  stage0.on('pageerror', (e) => console.error(`[stage 0] ${e.message}`))
  await stage0.goto(`${BASE}/share.html?model=${MODEL}&layers=0-${BOUNDS[1]}&sig=${SIGNAL_PORT}`, { waitUntil: 'domcontentloaded' })
  await passGate(stage0, 'stage 0')
  await stage0.waitForFunction(() => window.__shareReady === true, { timeout: 10 * 60_000, polling: 1000 })
  await stage0.click('#awake')
  const link = await stage0.evaluate(() => window.__shareLink)
  const roomId = link.split('#')[1]
  console.log(`stage 0 up (layers 0-${BOUNDS[1]}) — room ${roomId}`)

  // ── the rest, LAST FIRST, so every one of them arrives before its
  //    predecessor and has to be held rather than refused ──
  for (let i = STAGES - 1; i >= 1; i--) {
    const br = await launch(PROFILE(i))
    browsers.push(br)
    const pg = await br.newPage()
    pg.on('pageerror', (e) => console.error(`[stage ${i}] ${e.message}`))
    await pg.goto(`${BASE}/share.html?model=${MODEL}&layers=${BOUNDS[i]}-${BOUNDS[i + 1]}&sig=${SIGNAL_PORT}#${roomId}`,
      { waitUntil: 'domcontentloaded' })
    await passGate(pg, `stage ${i}`)
    await pg.waitForFunction(() => window.__helperReady === true, { timeout: 10 * 60_000, polling: 1000 })
    await pg.click('#awake')
    console.log(`stage ${i} up (layers ${BOUNDS[i]}-${BOUNDS[i + 1]})`)
  }

  await stage0.waitForFunction(() => window.__stagePaired === true, { timeout: 120_000, polling: 200 })
  check('chain assembled', true, await stage0.$eval('#room-stats', (el) => el.textContent ?? ''))

  // ── a guest asks the N-machine model a question ──
  //
  // PHONE=1 emulates the guest as an iPhone: viewport, touch, and the mobile
  // user agent, on the real Chrome engine. That is the honest scope of it —
  // it exercises the guest's LAYOUT and the WebRTC path from a phone-shaped
  // client. It does NOT reproduce a phone's GPU, and it cannot: the guest
  // role is the one that needs no WebGPU and downloads nothing, which is
  // exactly why it is the role worth emulating. Whether a given phone can
  // HOLD a stage is a question about maxBufferSize and free RAM on the
  // device, and only the device can answer it.
  const guest = await b0.newPage()
  if (PHONE) {
    await guest.emulate({
      name: 'iPhone 15 Pro',
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15'
        + ' (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1',
      viewport: { width: 393, height: 852, deviceScaleFactor: 3,
                  isMobile: true, hasTouch: true, isLandscape: false },
    })
  }
  guest.on('pageerror', (e) => console.error(`[guest] ${e.message}`))
  await guest.goto(link, { waitUntil: 'domcontentloaded' })
  await guest.waitForFunction(() => !document.getElementById('inp').disabled, { timeout: 90_000, polling: 200 })
  await guest.type('#inp', 'What is the capital of France? Answer in one short sentence.')
  await guest.click('#btn')
  await guest.waitForFunction(() => {
    const ais = document.querySelectorAll('.msg.ai')
    const last = ais[ais.length - 1]
    const btn = document.getElementById('btn')
    return !!last && (last.textContent ?? '').length > 5 && !btn.hidden && !btn.disabled
  }, { timeout: 5 * 60_000, polling: 300 })

  if (PHONE) {
    // The thing that actually breaks a phone guest: a page wider than the
    // screen, or a composer the keyboard would sit on top of. Both are
    // measurable here even though the GPU is not.
    const m = await guest.evaluate(() => ({
      overflowX: document.documentElement.scrollWidth - window.innerWidth,
      composerVisible: (() => {
        const c = document.getElementById('inp')
        if (!c) return false
        const r = c.getBoundingClientRect()
        return r.width > 0 && r.top < window.innerHeight
      })(),
    }))
    check('phone guest: no sideways scroll', m.overflowX <= 0, `${m.overflowX}px overflow`)
    check('phone guest: composer on screen', m.composerVisible, 'the input is reachable')
  }

  const reply = await guest.$eval('.msg.ai:last-of-type', (el) => el.textContent ?? '')
  console.log(`reply: ${JSON.stringify(reply.slice(-90))}`)
  check('split model answered', /paris/i.test(reply), 'the reply mentions Paris')
  const log = await stage0.$eval('#req-log', (el) => el.textContent ?? '')
  const hop = /([\d.]+) ms\/hop/.exec(log)
  check('hop measured', !!hop, hop ? `${hop[1]} ms per token across ${STAGES} stages` : log.slice(0, 80))
  console.log(`stage 0 log: ${JSON.stringify(log.slice(0, 200))}`)

  // ── a stage leaves. The model must become UNANSWERABLE and say which layers
  //    it lost — a chain that silently keeps generating past a missing slice
  //    would produce fluent nonsense, the one outcome worth more than an error.
  //    browsers[1] is the last stage (helpers were launched back to front).
  await browsers[1].close().catch(() => {})
  browsers.splice(1, 1)
  const lost = BOUNDS[STAGES - 1]
  let stats = ''
  for (let t = 0; t < 120 && !stats.includes(`waiting for layers ${lost}`); t++) {
    await new Promise((r) => setTimeout(r, 500))
    stats = await stage0.$eval('#room-stats', (el) => el.textContent ?? '')
  }
  check('chain reports the gap', stats.includes(`waiting for layers ${lost}`), stats)
  const stillPaired = await stage0.evaluate(() => window.__stagePaired === true)
  check('model marked incomplete', !stillPaired, 'the host no longer claims a complete model')
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  for (const b of browsers) await b.close().catch(() => {})
  for (let i = 0; i < STAGES; i++) rmSync(PROFILE(i), { recursive: true, force: true })
  for (const p of procs) { try { p.kill('SIGTERM') } catch { /* gone */ } }
  await new Promise((r) => setTimeout(r, 400))
  for (const p of procs) { try { p.kill('SIGKILL') } catch { /* gone */ } }
}
console.log(failed ? '\nsplit serve FAILED'
  : `\nsplit serve PASS — ${STAGES} browsers, a slice of the model each, one answer`)
process.exit(failed ? 1 : 0)
