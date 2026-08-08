#!/usr/bin/env node
// SPLIT-COST-BENCH — what does cutting a model in half actually cost?
//
// split-serve-e2e proves two browsers can each hold half a model and produce
// one answer. It does NOT say what the split costs, because it has nothing to
// compare against. This runs both shapes back to back on the same machine, the
// same model, the same prompt:
//
//   WHOLE  one browser holds every layer and serves a guest
//   SPLIT  two browsers hold [0,k) and [k,L) and serve the same guest
//
// interleaved W,S,W,S so a machine that drifts mid-run shows up as a gap
// between the two WHOLE rounds rather than as a split cost.
//
//   node scripts/split-cost-bench.mjs
//   MODEL=qwen3mlx ROUNDS=3 node scripts/split-cost-bench.mjs
//
// READ THE RESULT AS AN UPPER BOUND. Both stages share one GPU here, so the
// SPLIT rounds pay the per-token round trip AND contention for the same
// silicon. On two actual machines only the round trip remains. This harness
// cannot separate the two — it can only bound the total.
//
// The prompt is deliberately long-running: a 7-token reply measures startup,
// not decode (BENCH.md's 8-token window made a 4.7 tok/s model look like 23).

import { spawn } from 'node:child_process'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const ROOT = resolve(import.meta.dirname, '..')
const VITE_PORT = 5195
const SIGNAL_PORT = 8792
const MODEL = process.env.MODEL ?? 'llama32'
const spec = specForParam(MODEL)
const SPLIT = Number(process.env.SPLIT ?? Math.floor(spec.layers / 2))
const ROUNDS = Number(process.env.ROUNDS ?? 2)
const PROMPT = process.env.PROMPT
  ?? 'List the eight planets of the Solar System, with one sentence about each.'
const A_PROFILE = resolve(ROOT, '.tests-cache/chrome-cost-a')
const B_PROFILE = resolve(ROOT, '.tests-cache/chrome-cost-b')
const BASE = `http://localhost:${VITE_PORT}`

const procs = []
function run(cmd, args, cwd) {
  const p = spawn(cmd, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, WRANGLER_SEND_METRICS: 'false' } })
  p.stderr.on('data', (b) => { if (/error|EADDRINUSE/i.test(String(b))) process.stderr.write(`[${cmd}] ${b}`) })
  procs.push(p)
  return p
}
async function waitHttp(url, timeoutMs) {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    try { await fetch(url); return } catch { await new Promise((r) => setTimeout(r, 250)) }
  }
  throw new Error(`timed out waiting for ${url}`)
}
const launch = (userDataDir) => puppeteer.launch({
  headless: false,
  userDataDir,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--enable-dawn-features=allow_unsafe_apis'],
  defaultViewport: { width: 900, height: 700 },
  protocolTimeout: 20 * 60 * 1000,
})

let A = null
let B = null
let stageA = null
let stageB = null

/** One measurement: point stage A (and, when splitting, stage B) at the right
 *  layer range, ask the question through a guest tab, read the host's own
 *  request log. Returns { tokens, rate, hop }. */
async function round(mode) {
  const layersA = mode === 'split' ? `&layers=0-${SPLIT}` : ''
  await stageA.goto(`${BASE}/share.html?model=${MODEL}${layersA}&sig=${SIGNAL_PORT}`,
    { waitUntil: 'domcontentloaded' })
  await stageA.waitForFunction(() => window.__shareReady === true, { timeout: 10 * 60_000, polling: 1000 })
  await stageA.click('#awake')
  const link = await stageA.evaluate(() => window.__shareLink)

  if (mode === 'split') {
    const roomId = link.split('#')[1]
    await stageB.goto(`${BASE}/share.html?model=${MODEL}&layers=${SPLIT}-${spec.layers}&sig=${SIGNAL_PORT}#${roomId}`,
      { waitUntil: 'domcontentloaded' })
    await stageB.waitForFunction(() => window.__helperReady === true, { timeout: 10 * 60_000, polling: 1000 })
    await stageB.click('#awake')
    await stageA.waitForFunction(() => window.__stagePaired === true, { timeout: 60_000, polling: 200 })
  }

  const guest = await A.newPage()
  await guest.goto(link, { waitUntil: 'domcontentloaded' })
  await guest.waitForFunction(() => !document.getElementById('inp').disabled, { timeout: 90_000, polling: 200 })
  await guest.type('#inp', PROMPT)
  await guest.click('#btn')
  await guest.waitForFunction(() => {
    const ais = document.querySelectorAll('.msg.ai')
    const last = ais[ais.length - 1]
    const btn = document.getElementById('btn')
    return !!last && (last.textContent ?? '').length > 5 && !btn.hidden && !btn.disabled
  }, { timeout: 15 * 60_000, polling: 300 })

  const log = await stageA.$eval('#req-log', (el) => el.textContent ?? '')
  await guest.close()
  // Park the helper so a WHOLE round that follows does not share the GPU with
  // a still-resident second engine.
  if (mode === 'split') await stageB.goto('about:blank', { waitUntil: 'domcontentloaded' })

  const m = /(\d+) tok · ([\d.]+) tok\/s/.exec(log)
  if (!m) throw new Error(`no rate in host log: ${log.slice(0, 120)}`)
  const hop = /([\d.]+) ms\/hop/.exec(log)
  return { tokens: Number(m[1]), rate: Number(m[2]), hop: hop ? Number(hop[1]) : null }
}

const results = { whole: [], split: [] }
let failed = false
try {
  console.log(`${spec.id}: WHOLE (${spec.layers} layers, one browser) vs SPLIT (0-${SPLIT} + ${SPLIT}-${spec.layers}, two browsers)`)
  console.log(`prompt: ${JSON.stringify(PROMPT)}\n`)
  run('npx', ['wrangler', 'dev', '--port', String(SIGNAL_PORT)], resolve(ROOT, 'workers/share-signal'))
  run(resolve(ROOT, 'node_modules/.bin/vite'), ['--port', String(VITE_PORT), '--strictPort', '--clearScreen', 'false'], ROOT)
  await waitHttp(`http://localhost:${SIGNAL_PORT}/`, 30_000)
  await waitHttp(`${BASE}/share.html`, 30_000)
  for (const p of [A_PROFILE, B_PROFILE]) mkdirSync(p, { recursive: true })

  A = await launch(A_PROFILE)
  B = await launch(B_PROFILE)
  stageA = await A.newPage()
  stageB = await B.newPage()
  stageA.on('pageerror', (e) => console.error(`[stage A] ${e.message}`))
  stageB.on('pageerror', (e) => console.error(`[stage B] ${e.message}`))

  for (let i = 0; i < ROUNDS; i++) {
    for (const mode of ['whole', 'split']) {
      const r = await round(mode)
      results[mode].push(r)
      console.log(`  ${mode.padEnd(5)} run ${i + 1}: ${String(r.tokens).padStart(4)} tok · ${r.rate.toFixed(2)} tok/s`
        + (r.hop === null ? '' : ` · ${r.hop.toFixed(1)} ms/hop`))
    }
  }
} catch (e) {
  console.error(`ERROR: ${e.message}`)
  failed = true
} finally {
  await A?.close().catch(() => {})
  await B?.close().catch(() => {})
  for (const p of [A_PROFILE, B_PROFILE]) rmSync(p, { recursive: true, force: true })
  for (const p of procs) { try { p.kill('SIGTERM') } catch { /* gone */ } }
  await new Promise((r) => setTimeout(r, 400))
  for (const p of procs) { try { p.kill('SIGKILL') } catch { /* gone */ } }
}

const rates = (rs) => rs.map((r) => r.rate)
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length
if (results.whole.length && results.split.length) {
  const w = rates(results.whole)
  const s = rates(results.split)
  // Spread across the WHOLE rounds is the drift control: if the machine moved
  // more than the whole-vs-split gap, the gap is not a measurement.
  const drift = w.length > 1 ? (Math.max(...w) - Math.min(...w)) / mean(w) : null
  console.log(`\nwhole ${w.map((x) => x.toFixed(2)).join(' / ')} tok/s`)
  console.log(`split ${s.map((x) => x.toFixed(2)).join(' / ')} tok/s`)
  if (drift !== null) console.log(`drift across WHOLE rounds: ${(drift * 100).toFixed(1)}%`)
  console.log(`\nsplit keeps ${(100 * mean(s) / mean(w)).toFixed(1)}% of whole-model throughput`
    + ` — UPPER BOUND on the cost (one GPU serves both stages here)`)
  const hops = results.split.map((r) => r.hop).filter((h) => h !== null)
  if (hops.length) console.log(`hop: ${hops.map((h) => h.toFixed(1)).join(' / ')} ms per token (loopback)`)
}
process.exit(failed ? 1 : 0)
