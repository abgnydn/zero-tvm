#!/usr/bin/env node
// SHARE-E2E — the whole sharing path, end to end, on one machine:
//
//   wrangler dev (signaling DO, :8789) + vite (:5191) + one real Chrome with
//   two tabs: the HOST boots the actual engine on share.html?model=qwen3 and
//   the GUEST joins through the generated #room link. The tabs negotiate a
//   real RTCPeerConnection (loopback ICE — no STUN needed on one machine) and
//   the guest's question streams back over the DataChannel.
//
//   PASS = the guest renders a streamed reply containing "Paris" and the
//   composer re-arms (done-message contract).
//
// Deliberately NOT in tests/e2e/: it needs wrangler dev and ~2.3 GB of
// primed qwen3 mirror, and it runs on its own ports/profile so it can run
// WHILE the normal e2e suite (port 5189) is busy.
//
//   node scripts/share-e2e.mjs
//   SIGNAL_PORT=8795 VITE_PORT=5291 node scripts/share-e2e.mjs   # both movable

import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'

const ROOT = resolve(import.meta.dirname, '..')
// BOTH ports are overridable, and for the same reason the guard below exists:
// a harness that cannot move off an occupied port is a harness that cannot
// run. VITE_PORT had this hatch and SIGNAL_PORT did not — and SIGNAL_PORT was
// 8787, which is WRANGLER'S OWN DEFAULT, so any `wrangler dev` on the machine
// took this file out entirely. Including this repo's own share-signal worker:
// measured 2026-09-01, `node scripts/share-e2e.mjs` refused in 0.58 s while a
// copy differing only in the port reached puppeteer.launch in 1.3 s. The guard
// was right both times; there was simply no way to answer it.
// 8789, not 8787, for the same reason: the default must not be the port the
// tool most likely to be running already claims.
const VITE_PORT = Number(process.env.VITE_PORT ?? 5191)
const SIGNAL_PORT = Number(process.env.SIGNAL_PORT ?? 8789)
const PROFILE = resolve(ROOT, '.tests-cache/chrome-share-profile')

const procs = []
function run(cmd, args, cwd) {
  const p = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, WRANGLER_SEND_METRICS: 'false' } })
  p.log = ''
  const keep = (b) => { p.log = (p.log + b).slice(-4000) }
  p.stdout.on('data', keep)
  p.stderr.on('data', (b) => { keep(b); if (/error|EADDRINUSE/i.test(String(b))) process.stderr.write(`[${cmd}] ${b}`) })
  p.on('error', (e) => { p.dead = `failed to start (${e.code ?? e.message})` })
  p.on('exit', (code, sig) => { p.dead = `exited ${code ?? sig}` })
  procs.push(p)
  return p
}

// A server this harness did not start is not this harness's server. This file
// had the same silent-adoption hole split-serve-e2e.mjs was found with on
// 2026-08-25 — waitHttp resolving against an orphaned vite and the whole run
// reporting PASS for code it never loaded. The reasoning, and the measurements
// showing why --strictPort does NOT close it, are written out once in
// scripts/split-serve-e2e.mjs; this is the same two-part guard.
async function requirePortFree(port, what) {
  for (const host of ['127.0.0.1', '[::1]']) {
    const answered = await fetch(`http://${host}:${port}/`, { signal: AbortSignal.timeout(2000) })
      .then(() => true, () => false)
    if (!answered) continue
    throw new Error(
      `http://${host}:${port}/ already answers — something this harness did not start is `
      + `serving ${what}. Refusing to run. Stop it (lsof -nP -iTCP:${port} -sTCP:LISTEN) and rerun.`)
  }
}

/** A child's output with ANSI escapes stripped, for MATCHING only — the dumps
 *  below keep the colour. Under FORCE_COLOR=1 vite's ready line is
 *  `<esc>[1mLocal<esc>[22m:   <esc>[36mhttp://localhost:<esc>[1m5294<esc>[22m/`:
 *  escapes sit between `Local` and `:` AND inside the port digits, so a regex
 *  written for plain text never fires. Measured before this: a vite that
 *  printed "ready in 125 ms" was declared timed-out 31.9 s later. Piped stdout
 *  turns colour off by default, so the happy path never hit it — only a CI or
 *  a shell that forces colour did, and there it looked like a hung server. */
const plain = (s) => s.replace(/\u001B\[[0-9;]*[A-Za-z]/g, '')

/** vite's own ready line, pinned to the port we asked for. */
const VITE_READY = new RegExp(`Local:\\s+https?://localhost:${VITE_PORT}/`)
/** wrangler's, held to the same standard: `[wrangler:info] Ready on
 *  http://localhost:<port>`. Without it the relay's wait fell back to "the
 *  port answers", which is exactly the check requirePortFree distrusts. */
const SIGNAL_READY = new RegExp(`Ready on https?://(localhost|127\\.0\\.0\\.1|\\[::1\\]):${SIGNAL_PORT}\\b`)

/** Wait for the server WE STARTED, not for the port to answer. */
async function waitServer(proc, url, timeoutMs, what, ready) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (proc.dead) {
      throw new Error(`${what} ${proc.dead} before serving ${url}\n--- its output ---\n${proc.log.trim()}`)
    }
    if (!ready || ready.test(plain(proc.log))) {
      try { await fetch(url); return } catch { /* bound but not serving yet */ }
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`timed out waiting for ${what} at ${url}\n--- its output ---\n${proc.log.trim()}`)
}

let failed = false
let browser = null
try {
  console.log('starting wrangler dev (signal) + vite …')
  await requirePortFree(SIGNAL_PORT, 'the signaling relay')
  await requirePortFree(VITE_PORT, 'the dev server')
  const signal = run('npx', ['wrangler', 'dev', '--port', String(SIGNAL_PORT)], resolve(ROOT, 'workers/share-signal'))
  const vite = run(resolve(ROOT, 'node_modules/.bin/vite'), ['--port', String(VITE_PORT), '--strictPort', '--clearScreen', 'false'], ROOT)
  await waitServer(signal, `http://localhost:${SIGNAL_PORT}/`, 30_000, 'wrangler dev', SIGNAL_READY)
  await waitServer(vite, `http://localhost:${VITE_PORT}/share.html`, 30_000, 'vite', VITE_READY)

  mkdirSync(PROFILE, { recursive: true })
  browser = await puppeteer.launch({
    headless: false,
    userDataDir: PROFILE,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--enable-dawn-features=allow_unsafe_apis'],
    defaultViewport: { width: 1100, height: 820 },
    protocolTimeout: 10 * 60 * 1000,
  })

  // ── HOST ──
  const host = await browser.newPage()
  host.on('pageerror', (e) => console.error(`[host pageerror] ${e.message}`))
  // MODEL env picks the hosted model (default qwen3) — same knob as
  // peer-weights-e2e.mjs, for machines whose primed mirrors differ.
  // &sig= is not optional now that SIGNAL_PORT is not 8787: the page's dev
  // default IS ws://localhost:8787 (room-host.ts's signalEnv), so without it
  // the tabs would dial a relay this harness did not start — the same
  // silent-adoption shape requirePortFree exists to refuse, one layer up.
  // roomLink carries ?sig= into the guest link, so the guest follows.
  await host.goto(`http://localhost:${VITE_PORT}/share.html?model=${process.env.MODEL ?? 'qwen3'}&sig=${SIGNAL_PORT}`, { waitUntil: 'domcontentloaded' })
  // The hosting consent gate ALWAYS shows since f29b451 (the download
  // question disappears when the weights are cached; the hosting question
  // never does) — the harness, like a human, must click through it.
  await host.waitForSelector('#share-gate-go', { visible: true, timeout: 30_000 })
  await host.click('#share-gate-go')
  console.log('host: booting engine (first run downloads from the local mirror) …')
  await host.waitForFunction(() => window.__shareReady === true, { timeout: 8 * 60_000, polling: 1000 })
  const link = await host.evaluate(() => window.__shareLink)
  console.log(`host ready — room link: ${link}`)
  // Keep-awake toggle: wake lock may be denied under automation (caught in
  // page code); what this asserts is that the AudioContext path doesn't throw.
  await host.click('#awake')

  // ── GUEST ──
  const guest = await browser.newPage()
  guest.on('pageerror', (e) => console.error(`[guest pageerror] ${e.message}`))
  await guest.goto(link, { waitUntil: 'domcontentloaded' })
  await guest.waitForFunction(() => !document.getElementById('inp').disabled, { timeout: 60_000, polling: 200 })
  console.log('guest: DataChannel open (info received)')

  await guest.type('#inp', 'What is the capital of France? Answer in one short sentence.')
  await guest.click('#btn')
  await guest.waitForFunction(() => {
    const ais = document.querySelectorAll('.msg.ai')
    const last = ais[ais.length - 1]
    const btn = document.getElementById('btn')
    return !!last && (last.textContent ?? '').length > 5 && !btn.hidden && !btn.disabled
  }, { timeout: 4 * 60_000, polling: 300 })

  const reply = await guest.$eval('.msg.ai:last-of-type', (el) => el.textContent ?? '')
  const meta = await guest.$eval('.msg.ai:last-of-type .msg-stats', (el) => el.textContent ?? '')
  console.log(`guest reply: ${JSON.stringify(reply.slice(0, 140))}`)
  console.log(`stats: ${meta}`)

  const hostLog = await host.$eval('#req-log', (el) => el.textContent ?? '')
  console.log(`host request log: ${JSON.stringify(hostLog.slice(0, 120))}`)

  if (!/paris/i.test(reply)) { console.error('FAIL: reply does not mention Paris'); failed = true }
  if (!/tok\/s/.test(meta)) { console.error('FAIL: no done-stats on the reply'); failed = true }
  if (!/tok\/s|generating/.test(hostLog)) { console.error('FAIL: host request log empty — visibility contract broken'); failed = true }
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  await browser?.close().catch(() => {})
  for (const p of procs) { try { p.kill('SIGTERM') } catch { /* gone */ } }
  await new Promise((r) => setTimeout(r, 400))
  for (const p of procs) { try { p.kill('SIGKILL') } catch { /* gone */ } }
}
console.log(failed ? '\nshare e2e FAILED' : '\nshare e2e PASS — host tab served a guest over a real DataChannel')
process.exit(failed ? 1 : 0)
