#!/usr/bin/env node
// PEER-WEIGHTS-E2E — replicate a model's weight cache between two browsers.
//
// TWO Chrome instances with SEPARATE profiles, which is the whole point: OPFS
// is per-origin-per-profile, so one profile's cache is genuinely invisible to
// the other. (Two tabs of one profile would share the cache and the pull would
// skip every file — a green test proving nothing.)
//
//   A: .tests-cache/chrome-share-profile  — already holds qwen3 from the share
//      e2e; hosts share.html?model=qwen3
//   B: .tests-cache/chrome-peer-profile   — wiped at start; joins the room and
//      clicks "Copy N GB to this device"
//
// PASS = every file arrives, every piece passes its SHA-256, and B's OPFS ends
// up with the same file count and byte total as A's inventory.
//
//   node scripts/peer-weights-e2e.mjs
//
// Needs the qwen3 mirror primed (node scripts/download-weights.mjs --model qwen3)
// and a run of scripts/share-e2e.mjs first, to populate profile A's OPFS.

import { spawn } from 'node:child_process'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'

const ROOT = resolve(import.meta.dirname, '..')
// Overridable ONLY so the port guard below can be exercised on a scratch port.
const VITE_PORT = Number(process.env.VITE_PORT ?? 5192)
const SIGNAL_PORT = 8788
const HOST_PROFILE = resolve(ROOT, '.tests-cache/chrome-share-profile')
const PEER_PROFILE = resolve(ROOT, '.tests-cache/chrome-peer-profile')
const MODEL = process.env.MODEL ?? 'qwen3'

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

/** vite's own ready line, pinned to the port we asked for. */
const VITE_READY = new RegExp(`Local:\\s+https?://localhost:${VITE_PORT}/`)

/** Wait for the server WE STARTED, not for the port to answer. */
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
  defaultViewport: { width: 1000, height: 760 },
  protocolTimeout: 15 * 60 * 1000,
})

/** File count + byte total of a model dir in THIS browser's OPFS. */
const opfsStat = async (page, dirName) => page.evaluate(async (name) => {
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle(name)
    let files = 0, bytes = 0
    for await (const [, h] of dir) {
      if (h.kind !== 'file') continue
      files++
      bytes += (await h.getFile()).size
    }
    return { files, bytes }
  } catch { return { files: 0, bytes: 0 } }
}, dirName)

let failed = false
let hostBrowser = null
let peerBrowser = null
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(18)} ${detail}`)
  if (!pass) failed = true
}

try {
  console.log('starting wrangler dev (signal) + vite …')
  await requirePortFree(SIGNAL_PORT, 'the signaling relay')
  await requirePortFree(VITE_PORT, 'the dev server')
  const signal = run('npx', ['wrangler', 'dev', '--port', String(SIGNAL_PORT)], resolve(ROOT, 'workers/share-signal'))
  const vite = run(resolve(ROOT, 'node_modules/.bin/vite'), ['--port', String(VITE_PORT), '--strictPort', '--clearScreen', 'false'], ROOT)
  await waitServer(signal, `http://localhost:${SIGNAL_PORT}/`, 30_000, 'wrangler dev', null)
  await waitServer(vite, `http://localhost:${VITE_PORT}/share.html`, 30_000, 'vite', VITE_READY)

  // Fresh receiving profile every run — a leftover cache would make the pull
  // skip files and the test would pass without moving bytes.
  rmSync(PEER_PROFILE, { recursive: true, force: true })
  mkdirSync(PEER_PROFILE, { recursive: true })

  hostBrowser = await launch(HOST_PROFILE)
  const host = await hostBrowser.newPage()
  host.on('pageerror', (e) => console.error(`[host pageerror] ${e.message}`))
  await host.goto(`http://localhost:${VITE_PORT}/share.html?model=${MODEL}&sig=${SIGNAL_PORT}`, { waitUntil: 'domcontentloaded' })
  console.log(`host: booting ${MODEL} …`)
  await host.waitForFunction(() => window.__shareReady === true, { timeout: 8 * 60_000, polling: 1000 })
  const link = await host.evaluate(() => window.__shareLink)
  const dirName = await host.evaluate(async () => {
    const { specFromSearch } = await import('/src/zero-tvm/model-select.ts')
    const { opfsDirFor } = await import('/src/zero-tvm/model-registry.ts')
    return opfsDirFor(specFromSearch(location.search))
  })
  // Keep the host tab exempt from background-tab throttling — the peer browser
  // takes focus a moment later, and a throttled host serves at ~1 MB/s.
  await host.click('#awake')
  const hostStat = await opfsStat(host, dirName)
  console.log(`host ready — ${dirName}: ${hostStat.files} files, ${(hostStat.bytes / 1e9).toFixed(2)} GB`)
  if (!hostStat.files) throw new Error(`host profile has no cached weights for ${MODEL} — run scripts/share-e2e.mjs first`)

  peerBrowser = await launch(PEER_PROFILE)
  const peer = await peerBrowser.newPage()
  peer.on('pageerror', (e) => console.error(`[peer pageerror] ${e.message}`))
  await peer.goto(link, { waitUntil: 'domcontentloaded' })
  const before = await opfsStat(peer, dirName)
  check('peer starts empty', before.files === 0, `${before.files} files`)

  await peer.waitForFunction(() => !document.getElementById('local-copy')?.classList.contains('hidden'),
    { timeout: 60_000, polling: 200 })
  const offer = await peer.$eval('#lc-btn', (el) => el.textContent ?? '')
  console.log(`peer sees offer: ${JSON.stringify(offer)}`)

  const t0 = Date.now()
  await peer.click('#lc-btn')
  // Progress lines while it runs, so a stall is visible rather than a silent wait.
  const ticker = setInterval(() => {
    peer.$eval('#lc-status', (el) => el.textContent ?? '').then((s) => console.log(`  ${s}`)).catch(() => {})
  }, 15_000)
  await peer.waitForFunction(() => window.__pullDone !== undefined, { timeout: 12 * 60_000, polling: 500 })
  clearInterval(ticker)
  const res = await peer.evaluate(() => window.__pullDone)
  const secs = (Date.now() - t0) / 1000
  console.log(`pull finished: ${(res.bytes / 1e9).toFixed(2)} GB in ${secs.toFixed(0)}s `
    + `(${(res.bytes / 1e6 / secs).toFixed(0)} MB/s over the DataChannel)`)

  const after = await opfsStat(peer, dirName)
  check('files replicated', after.files === hostStat.files, `${after.files} vs host ${hostStat.files}`)
  check('bytes replicated', after.bytes === hostStat.bytes, `${after.bytes} vs host ${hostStat.bytes}`)
  const status = await peer.$eval('#lc-status', (el) => el.textContent ?? '')
  check('offers local chat', /Open the chat on this device/.test(status), JSON.stringify(status.slice(0, 60)))
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  await hostBrowser?.close().catch(() => {})
  await peerBrowser?.close().catch(() => {})
  // The replicated copy is real weights — don't leave gigabytes behind.
  rmSync(PEER_PROFILE, { recursive: true, force: true })
  for (const p of procs) { try { p.kill('SIGTERM') } catch { /* gone */ } }
  await new Promise((r) => setTimeout(r, 400))
  for (const p of procs) { try { p.kill('SIGKILL') } catch { /* gone */ } }
}
console.log(failed ? '\npeer-weights e2e FAILED' : '\npeer-weights e2e PASS — cache replicated browser-to-browser')
process.exit(failed ? 1 : 0)
