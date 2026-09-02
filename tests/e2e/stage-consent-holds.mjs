#!/usr/bin/env node
/**
 * WHAT A `?layers=` / `?split=` LINK IS ALLOWED TO SAY ABOUT ITSELF — the
 * browser half of the proof.
 *
 * The layer bound (stage-range.ts) stopped `0-N` and made the two gates agree
 * about a RANGE. It did not make them agree about what a range MEANS, and
 * three things were measured wrong in a real Chrome:
 *
 *   1  /share.html?model=qwen38&layers=0-63  (of 64)
 *      RAM warning HIDDEN, weights line "a slice of the full ~14.1 GB, not all
 *      of it" — for ~98% of the checkpoint. `0-64` is caught by the bound;
 *      `0-63` is a LEGITIMATE host stage in a two-machine split, so the answer
 *      is not a tighter end bound. The SUPPRESSION RULE was wrong: both gates
 *      dropped `ramNote` for any non-null stage, so 13% and 98% read the same.
 *
 *   2  /share.html?model=&layers=0-31       (Phi-3, 32 layers, MLC)
 *      A whole consent screen for a split that cannot exist: title
 *      "Phi-3-mini · layers 0–31 of 32", the whole model sold as "a slice …
 *      not all of it", RAM note removed, and a hosting-a-split paragraph.
 *      loadWeights refuses a layerRange on MLC before a byte is fetched, so
 *      the screen is followed by a crash. landing.ts's `splitFor` has had
 *      `if (!canSplitAcrossDevices(spec)) return null` since it was written;
 *      share.html's routing point did not.
 *
 *   2b /share.html?model=qwen38&layers=8-40
 *      Same shape, one rule further: a HOST stage that does not start at layer
 *      0 has no embedding. share.ts threw for it — AFTER painting the scene
 *      with the stage it was about to refuse, and with raw engine prose.
 *
 *   3  /?model=&chat=1, then location.hash = '#swarm'
 *      Every other input path asks whether the gate is up (the click delegate,
 *      `engage`, `keyIntent`). `openSwarm` did not, and it WALKS THE ROSTER
 *      when the character on stage cannot be split — so the scene showed one
 *      character while the open dialog named another. Consent was not
 *      bypassed (accepting boots the captured plan), but landing.ts states
 *      that a change of character while the gate is up "cannot happen".
 *
 * Why a standalone script and not a vitest file: the same reasons
 * gate-holds.mjs gives — tests/unit stays GPU-free, and tests/e2e/harness.ts
 * owns a port and a shared Chrome profile that other agents are using. This
 * brings its own vite (5296, movable with VITE_PORT) and its own throwaway
 * profile.
 *
 * NOTHING IS DOWNLOADED: every non-localhost request is aborted at the network
 * layer. Every fact below is read from the consent screen, which is painted
 * before a byte is asked for — that is the whole point.
 *
 *   node tests/e2e/stage-consent-holds.mjs          # run every case
 *   node tests/e2e/stage-consent-holds.mjs --show   # watch it happen
 */

import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import puppeteer from 'puppeteer'
// The guard scripts/{split-serve,share,peer-weights}-e2e.mjs got on 2026-08-25,
// and tests/e2e/{harness.ts,gate-holds.mjs} got on 2026-09-02. This file was
// the third certification harness and did not get it, which matters here for
// the same reason it mattered there: this script IS the evidence that
// a `?layers=` link cannot lie about itself, and it was reading that evidence
// off whatever server answered 5296. Measured 2026-09-02 on this machine, with
// the port as the only edit to the pre-fix file and a foreign HTTP server
// holding it:
//
//   - Squatter on `[::1]:5304` (the address vite wants, and the one
//     `localhost` resolves to first here): vite died with `[vite] error when
//     starting dev server: Error: Port 5304 is already in use`, the old
//     `waitForUrl` was already satisfied by the squatter, and the run drove
//     Chrome at the STRANGER — `shareReady` timed out after 25 s on a page
//     with no `#share-gate-go`, and the process died before printing a single
//     result line. It never noticed the server was not its own.
//   - Squatter on `[::]:5304` (dual-stack): `--strictPort` raised NOTHING.
//     `lsof` read `*:5304 (LISTEN)` for the squatter and `[::1]:5304 (LISTEN)`
//     for vite AT THE SAME TIME — vite binds `[::1]` only here, so the two
//     coexist, `curl -4` reached the stranger and `curl -6` reached vite. The
//     run reported `5/5 passed`, and that green says nothing about which
//     server was graded: it happened to be reached over IPv6. A client whose
//     resolver picked IPv4 — another Chrome, a CI — got the stranger.
//
// That second one is why `--strictPort` is not the check, and why the guard
// probes BOTH loopback families. The rest of the reasoning is in port-guard.ts.
// (tests/e2e/probe-bound-and-head.mjs is still on the old `waitForUrl` shape.)
import { requirePortFree, viteReady, waitForChild, watch } from './port-guard.ts'

// Movable, because the guard turns "occupied" into a refusal and a harness that
// cannot step aside is a harness that cannot run. Same variable name as the
// sibling harnesses — one knob, one grammar.
const PORT = Number(process.env.VITE_PORT ?? 5296)
const BASE = `http://localhost:${PORT}`
const ROOT = resolve(import.meta.dirname, '../..')
const SHOW = process.argv.includes('--show')

/** Walked up, not joined: this repo is worked on from git worktrees, which
 *  have no node_modules of their own. */
function findBin(name) {
  for (let d = ROOT; d !== dirname(d); d = dirname(d)) {
    const p = join(d, 'node_modules/.bin', name)
    if (existsSync(p)) return p
  }
  throw new Error(`${name} not found in any node_modules/.bin above ${ROOT}`)
}
const VITE = findBin('vite')

const profile = mkdtempSync(join(tmpdir(), 'ztvm-stage-'))
let vite = null
let browser = null

async function start() {
  // Before the spawn: a port that already answers is refused outright, on BOTH
  // loopback families — either can be the one a resolver picks for `localhost`,
  // and the dual-stack squatter is the case `--strictPort` misses entirely.
  // After it: wait for OUR child's own ready line on OUR port, never for "the
  // port responds" — a squatter answers that poll from t=0, long before vite's
  // EADDRINUSE reaches stderr, which is how the run above drove a whole Chrome
  // at a tree that was not this one.
  await requirePortFree(PORT, 'the dev server')
  const proc = spawn(VITE, ['--port', String(PORT), '--strictPort', '--clearScreen', 'false'], {
    stdio: ['ignore', 'pipe', 'pipe'], cwd: ROOT, env: process.env,
  })
  vite = proc
  const watched = watch(proc, (s) => {
    if (/error|ERROR|EADDRINUSE|already in use/.test(s)) process.stderr.write(`[vite] ${s}`)
  })
  await waitForChild(watched, `${BASE}/index.html`, 30_000, 'vite', viteReady(PORT))
  browser = await puppeteer.launch({
    headless: !SHOW,
    userDataDir: profile,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
      '--enable-dawn-features=allow_unsafe_apis', '--mute-audio'],
    defaultViewport: { width: 1100, height: 900 },
  })
}

async function stop() {
  await browser?.close().catch(() => {})
  if (vite && !vite.killed) {
    vite.kill('SIGTERM')
    await new Promise((r) => setTimeout(r, 300))
    if (!vite.killed) vite.kill('SIGKILL')
  }
  try { rmSync(profile, { recursive: true, force: true }) } catch { /* best effort */ }
}

/** A page with the outside world unplugged, its page errors recorded. */
async function open(path) {
  const page = await browser.newPage()
  page.errors = []
  page.on('pageerror', (e) => { page.errors.push(e.message) })
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
  await page.setRequestInterception(true)
  page.on('request', (r) => {
    const u = r.url()
    if (u.startsWith(BASE) || u.startsWith('data:') || u.startsWith('blob:')) r.continue()
    else r.abort('failed')
  })
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  return page
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ── reading the two consent screens ───────────────────────────────────────

/** share.html's gate: the two paragraphs and the button, plus the nameplate
 *  and title the scene was painted with before any of it. */
const shareSnap = (page) => page.evaluate(() => {
  const t = (id) => document.getElementById(id)
  const ram = t('gate-ram')
  return {
    title: document.title,
    name: document.querySelector('.mb-name')?.textContent?.trim() ?? null,
    params: document.querySelector('.mb-params-text')?.textContent?.trim() ?? null,
    consent: document.querySelector('.cs-room-consent p')?.textContent?.trim() ?? null,
    weights: t('gate-weights')?.textContent?.trim() ?? null,
    ram: ram && !ram.hidden ? ram.textContent.trim() : null,
    ramHidden: ram ? ram.hidden : null,
    go: t('share-gate-go')?.textContent?.trim() ?? null,
    layersRow: [...document.querySelectorAll('.mb-stats div')]
      .find((d) => d.querySelector('dt')?.textContent?.includes('Layers'))
      ?.querySelector('dd')?.textContent?.trim() ?? null,
  }
})

/** The entrance's gate. */
const landingSnap = (page) => page.evaluate(() => {
  const q = (s) => document.querySelector(s)?.textContent?.trim() ?? null
  const gate = document.querySelector('.cs-url-gate')
  return {
    stage: q('.mb-name'),
    title: q('#cs-gate-title'),
    what: q('#cs-gate-what'),
    cost: q('#cs-gate-cost'),
    go: q('#cs-gate-go'),
    gateOpen: gate ? gate.open : null,
    swarm: !!document.getElementById('model-browser')?.classList.contains('cs-swarm'),
    url: location.search + location.hash,
  }
})

/** share.html's gate is up and its cache probe has answered. */
const shareReady = (page) => page.waitForFunction(
  () => {
    const g = document.getElementById('share-gate-go')
    return !!g && !g.disabled
  }, { timeout: 25_000, polling: 100 })

/** The entrance's gate is up and settled. */
const landingReady = (page) => page.waitForFunction(
  () => {
    const g = document.getElementById('cs-gate-go')
    return !!g && !g.disabled
  }, { timeout: 25_000, polling: 100 })

// ── the report ────────────────────────────────────────────────────────────
const results = []
function check(name, pass, detail) {
  results.push({ name, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}`)
  for (const line of detail) console.log(`        ${line}`)
}

// ── 1 · the RAM warning survives a stage that is nearly the whole model ────
//
// Three URLs, one checkpoint (qwen38, 64 layers, ~14.1 GB, needs ~18 GB free
// RAM). The bound already refuses 0-64. 0-63 is a real host stage and must not
// buy silence on the RAM limit, and it must not READ like a 12% stage either.
async function ramNoteSurvivesAStage() {
  const whole = await open('/share.html?model=qwen38')
  await shareReady(whole)
  const a = await shareSnap(whole)

  const bounded = await open('/share.html?model=qwen38&layers=0-64')
  await shareReady(bounded)
  const b = await shareSnap(bounded)

  const big = await open('/share.html?model=qwen38&layers=0-63')
  await shareReady(big)
  const c = await shareSnap(big)

  const small = await open('/share.html?model=qwen38&layers=0-8')
  await shareReady(small)
  const d = await shareSnap(small)

  const saysRam = (s) => (s.ram ?? '').includes('18 GB')
  check('1  share.html — a 98% stage still states the RAM limit, and does not read like a 13% one',
    saysRam(a) && saysRam(b) && saysRam(c) && saysRam(d)
    && (c.weights ?? '') !== (d.weights ?? ''), [
      `?model=qwen38                 ram: ${JSON.stringify(a.ram)}`,
      `?model=qwen38&layers=0-64     ram: ${JSON.stringify(b.ram)}   (the bound catches this one)`,
      `?model=qwen38&layers=0-63     ram: ${JSON.stringify(c.ram)}`,
      `                              weights: ${JSON.stringify(c.weights)}`,
      `?model=qwen38&layers=0-8      ram: ${JSON.stringify(d.ram)}`,
      `                              weights: ${JSON.stringify(d.weights)}`,
    ])
  for (const p of [whole, bounded, big, small]) await p.close()
}

// ── 1b · …and the entrance says the same thing about the same stage ───────
//
// `?split=0,63,64&stage=0` is the entrance's spelling of the same 98% host
// stage, and gateCopy suppressed ramNote by the same rule. The two surfaces
// must not drift.
async function ramNoteOnTheEntrance() {
  const whole = await open('/?model=qwen38&chat=1')
  await landingReady(whole)
  const a = await landingSnap(whole)

  const big = await open('/?model=qwen38&split=0,63,64&stage=0&chat=1')
  await landingReady(big)
  const b = await landingSnap(big)

  const small = await open('/?model=qwen38&split=0,8,64&stage=0&chat=1')
  await landingReady(small)
  const c = await landingSnap(small)

  check('1b the entrance says the same about the same stage',
    (a.cost ?? '').includes('18 GB') && (b.cost ?? '').includes('18 GB')
    && (c.cost ?? '').includes('18 GB') && (b.what ?? '') !== (c.what ?? ''), [
      `?model=qwen38&chat=1                        cost: ${JSON.stringify(a.cost)}`,
      `&split=0,63,64&stage=0  (98% of its layers)  cost: ${JSON.stringify(b.cost)}`,
      `                                            what: ${JSON.stringify(b.what)}`,
      `&split=0,8,64&stage=0   (13% of its layers)  cost: ${JSON.stringify(c.cost)}`,
      `                                            what: ${JSON.stringify(c.what)}`,
    ])
  for (const p of [whole, big, small]) await p.close()
}

// ── 2 · a checkpoint the loader cannot cut has no stage ───────────────────
async function noStageOnAnUncuttableCheckpoint() {
  const page = await open('/share.html?model=&layers=0-31')
  await shareReady(page)
  const s = await shareSnap(page)
  check('2  share.html — Phi-3 (MLC, uncuttable) is never described as a stage of itself',
    !/layers/i.test(s.title) && !/layers/i.test(s.params ?? '')
    && !/Layers 0/i.test(s.weights ?? '') && !/holds layers/i.test(s.consent ?? '')
    && (s.ram === null || !/layers/i.test(s.ram)) && s.layersRow === null, [
      '?model=&layers=0-31 — Phi-3-mini has 32 layers and ships MLC shards;',
      'loadWeights refuses a layerRange on it before a byte is fetched.',
      `title:    ${JSON.stringify(s.title)}`,
      `plate:    ${JSON.stringify(s.name)} · ${JSON.stringify(s.params)}`,
      `sheet:    Layers row = ${JSON.stringify(s.layersRow)}`,
      `consent:  ${JSON.stringify((s.consent ?? '').slice(0, 150))}`,
      `weights:  ${JSON.stringify(s.weights)}`,
      `ram:      ${JSON.stringify(s.ram)}`,
      `button:   ${JSON.stringify(s.go)}`,
    ])
  await page.close()
}

// ── 2b · a HOST stage that does not start at layer 0 ──────────────────────
async function hostStageStartsAtZero() {
  const page = await open('/share.html?model=qwen38&layers=8-40')
  await shareReady(page).catch(() => {})
  const s = await shareSnap(page)
  check('2b share.html — a host stage that skips the embedding is no stage, not a crash',
    !/layers 8/i.test(s.title) && s.layersRow === null
    && !/8[–-]40/.test(s.consent ?? '') && page.errors.length === 0, [
      '?model=qwen38&layers=8-40 — a hosting stage has to start at layer 0.',
      `title:      ${JSON.stringify(s.title)}`,
      `sheet:      Layers row = ${JSON.stringify(s.layersRow)}`,
      `consent:    ${JSON.stringify((s.consent ?? '').slice(0, 130))}`,
      `weights:    ${JSON.stringify(s.weights)}`,
      `pageerror:  ${JSON.stringify(page.errors)}`,
    ])
  await page.close()
}

// ── 3 · the hash cannot walk the roster behind an open gate ───────────────
async function hashCannotWalkTheRoster() {
  const page = await open('/?model=&chat=1')
  await landingReady(page)
  const before = await landingSnap(page)
  await page.evaluate(() => { location.hash = '#swarm' })
  await sleep(900)
  const after = await landingSnap(page)
  // …and the refusal is not permanent: the question is answered, the same link
  // works. A gate check that quietly disabled the swarm mode for the rest of
  // the session would be a worse bug than the one it closes.
  await page.click('#cs-gate-no')
  await sleep(300)
  await page.evaluate(() => document.querySelector('a[href="#swarm"]')?.click())
  await sleep(900)
  const declined = await landingSnap(page)
  check('3  #swarm arriving while the gate is up does not move the character behind it',
    after.stage === before.stage && after.swarm === false
    && after.what === before.what && declined.swarm === true, [
      '/?model=&chat=1 — the gate names Phi-3, which cannot be split, so openSwarm',
      'walks the roster to the first checkpoint that can be.',
      `before:  stage ${JSON.stringify(before.stage)}, gate says ${JSON.stringify((before.what ?? '').slice(0, 60))}, swarm mode: ${before.swarm}`,
      '[location.hash = "#swarm"]',
      `after:   stage ${JSON.stringify(after.stage)}, gate says ${JSON.stringify((after.what ?? '').slice(0, 60))}, swarm mode: ${after.swarm}`,
      `         gate still open: ${after.gateOpen}, url: ${JSON.stringify(after.url)}`,
      '[decline, then click the nav\'s #swarm link]',
      `then:    stage ${JSON.stringify(declined.stage)}, swarm mode: ${declined.swarm}`,
    ])
  await page.close()
}

// ── run ───────────────────────────────────────────────────────────────────
try {
  await start()
  await ramNoteSurvivesAStage()
  await ramNoteOnTheEntrance()
  await noStageOnAnUncuttableCheckpoint()
  await hostStageStartsAtZero()
  await hashCannotWalkTheRoster()
} finally {
  await stop()
}
const failed = results.filter((r) => !r.pass)
console.log(`\n${results.length - failed.length}/${results.length} passed`)
process.exit(failed.length ? 1 : 0)
