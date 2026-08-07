#!/usr/bin/env node
// SPLIT-SERVE-E2E — one model, two machines, a real answer.
//
// Browser A holds layers [0, k) and hosts the room; browser B joins it holding
// [k, layers). Neither can answer anything alone. A guest tab then asks a
// question and the reply streams back, one WebRTC round trip per token.
//
// Separate browser PROFILES because that is the only way to make the two
// stages genuinely independent: separate GPU contexts, separate OPFS, separate
// engines that each loaded only their own layers.
//
//   node scripts/split-serve-e2e.mjs            # llama32, split at 8/16
//   MODEL=qwen3mlx SPLIT=18 node scripts/split-serve-e2e.mjs
//
// Small model on purpose: the point is the transport, and two halves of a 1B
// fit on one machine's GPU. Splitting the 35B needs two actual machines.

import { spawn } from 'node:child_process'
import { mkdirSync, rmSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer from 'puppeteer'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const ROOT = resolve(import.meta.dirname, '..')
const VITE_PORT = 5194
const SIGNAL_PORT = 8791
const MODEL = process.env.MODEL ?? 'llama32'
const spec = specForParam(MODEL)
const SPLIT = Number(process.env.SPLIT ?? Math.floor(spec.layers / 2))
const A_PROFILE = resolve(ROOT, '.tests-cache/chrome-stage-a')
const B_PROFILE = resolve(ROOT, '.tests-cache/chrome-stage-b')
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
  protocolTimeout: 15 * 60 * 1000,
})

let failed = false
let A = null
let B = null
const check = (name, pass, detail) => {
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name.padEnd(22)} ${detail}`)
  if (!pass) failed = true
}

try {
  console.log(`splitting ${spec.id} at layer ${SPLIT} of ${spec.layers}`)
  run('npx', ['wrangler', 'dev', '--port', String(SIGNAL_PORT)], resolve(ROOT, 'workers/share-signal'))
  run(resolve(ROOT, 'node_modules/.bin/vite'), ['--port', String(VITE_PORT), '--strictPort', '--clearScreen', 'false'], ROOT)
  await waitHttp(`http://localhost:${SIGNAL_PORT}/`, 30_000)
  await waitHttp(`${BASE}/share.html`, 30_000)
  for (const p of [A_PROFILE, B_PROFILE]) mkdirSync(p, { recursive: true })

  // ── stage 0: layers [0, SPLIT), hosts the room ──
  A = await launch(A_PROFILE)
  const stageA = await A.newPage()
  stageA.on('pageerror', (e) => console.error(`[stage A] ${e.message}`))
  await stageA.goto(`${BASE}/share.html?model=${MODEL}&layers=0-${SPLIT}&sig=${SIGNAL_PORT}`, { waitUntil: 'domcontentloaded' })
  await stageA.waitForFunction(() => window.__shareReady === true, { timeout: 8 * 60_000, polling: 1000 })
  await stageA.click('#awake')
  const link = await stageA.evaluate(() => window.__shareLink)
  const roomId = link.split('#')[1]
  console.log(`stage A up (layers 0-${SPLIT}) — room ${roomId}`)

  // ── stage 1: layers [SPLIT, layers), joins as a helper ──
  B = await launch(B_PROFILE)
  const stageB = await B.newPage()
  stageB.on('pageerror', (e) => console.error(`[stage B] ${e.message}`))
  await stageB.goto(`${BASE}/share.html?model=${MODEL}&layers=${SPLIT}-${spec.layers}&sig=${SIGNAL_PORT}#${roomId}`,
    { waitUntil: 'domcontentloaded' })
  await stageB.waitForFunction(() => window.__helperReady === true, { timeout: 8 * 60_000, polling: 1000 })
  await stageB.click('#awake')
  console.log(`stage B up (layers ${SPLIT}-${spec.layers})`)

  await stageA.waitForFunction(() => window.__stagePaired === true, { timeout: 60_000, polling: 200 })
  await stageB.waitForFunction(() => window.__helperPaired === true, { timeout: 60_000, polling: 200 })
  check('stages paired', true, await stageA.$eval('#room-stats', (el) => el.textContent ?? ''))

  // ── a guest asks the two-machine model a question ──
  const guest = await A.newPage()
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

  const reply = await guest.$eval('.msg.ai:last-of-type', (el) => el.textContent ?? '')
  console.log(`reply: ${JSON.stringify(reply.slice(-90))}`)
  check('split model answered', /paris/i.test(reply), 'the reply mentions Paris')
  const log = await stageA.$eval('#req-log', (el) => el.textContent ?? '')
  const hop = /([\d.]+) ms\/hop/.exec(log)
  check('hop measured', !!hop, hop ? `${hop[1]} ms per token across the two stages` : log.slice(0, 80))
  console.log(`stage A log: ${JSON.stringify(log.slice(0, 160))}`)
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
console.log(failed ? '\nsplit serve FAILED' : '\nsplit serve PASS — two browsers, half a model each, one answer')
process.exit(failed ? 1 : 0)
