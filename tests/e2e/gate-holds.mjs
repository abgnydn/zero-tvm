#!/usr/bin/env node
/**
 * DOES THE CONSENT GATE ACTUALLY HOLD? — the browser half of the proof.
 *
 * `?chat=1` may not spend a stranger's bandwidth without an agreement, so
 * landing.ts grew `openUrlGate`. A review found the gate was a PICTURE of a
 * gate: it hid the verbs with `display:none` and left every other way in
 * wired. This script reproduces each way in, in a real Chrome, and is the
 * red-before / green-after evidence for the fix.
 *
 * Why a standalone script and not a vitest file:
 *   - tests/unit must stay GPU-free and fast; these need a browser.
 *   - tests/e2e/harness.ts owns port 5189 and a SHARED Chrome profile. Other
 *     agents are running that suite right now, so this brings its own vite
 *     (5285) and its own throwaway profile and touches neither.
 *
 * NOTHING IS DOWNLOADED. Every request that is not localhost is aborted at
 * the network layer, so "what booted" is read from the boot card's own title
 * (`#loading-title` — "Summoning <name>") and the weights never start. That is
 * the whole point of the bug: the title names the model whose 16.4 GB the page
 * was about to fetch.
 *
 *   node tests/e2e/gate-holds.mjs          # run every case
 *   node tests/e2e/gate-holds.mjs --show   # watch it happen (headful, slow)
 */

import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import puppeteer from 'puppeteer'

const PORT = 5285
const BASE = `http://localhost:${PORT}`
const ROOT = resolve(import.meta.dirname, '../..')
const SHOW = process.argv.includes('--show')

/** Walked up, not joined: this repo is worked on from git worktrees, which
 *  have no node_modules of their own — node resolves imports by walking up to
 *  the checkout's, and the vite we spawn has to be found the same way. */
function findBin(name) {
  for (let d = ROOT; d !== dirname(d); d = dirname(d)) {
    const p = join(d, 'node_modules/.bin', name)
    if (existsSync(p)) return p
  }
  throw new Error(`${name} not found in any node_modules/.bin above ${ROOT}`)
}
const VITE = findBin('vite')

// ── the harness ───────────────────────────────────────────────────────────
const profile = mkdtempSync(join(tmpdir(), 'ztvm-gate-'))
let vite = null
let browser = null

async function waitForUrl(url, ms) {
  const t0 = Date.now()
  let last = null
  while (Date.now() - t0 < ms) {
    try { if ((await fetch(url)).ok) return } catch (e) { last = e }
    await new Promise((r) => setTimeout(r, 150))
  }
  throw new Error(`timed out waiting for ${url}: ${last}`)
}

async function start() {
  vite = spawn(VITE, ['--port', String(PORT), '--strictPort', '--clearScreen', 'false'], {
    stdio: ['ignore', 'pipe', 'pipe'], cwd: ROOT, env: process.env,
  })
  vite.stderr?.on('data', (b) => process.stderr.write(`[vite] ${b}`))
  await waitForUrl(`${BASE}/index.html`, 30_000)
  browser = await puppeteer.launch({
    headless: !SHOW,
    userDataDir: profile,
    args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
      '--enable-dawn-features=allow_unsafe_apis', '--mute-audio'],
    defaultViewport: { width: 1100, height: 860 },
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

/** A page with the outside world unplugged and the intro splash out of the way. */
async function open(path) {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error(`  [pageerror] ${e.message}`))
  // reduce-motion removes the splash synchronously and stops the stat count-up,
  // so every read below is of a settled scene rather than of an animation.
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
  await page.setRequestInterception(true)
  page.on('request', (r) => {
    const u = r.url()
    if (u.startsWith(BASE) || u.startsWith('data:') || u.startsWith('blob:')) r.continue()
    else r.abort('failed')          // no weights leave the building
  })
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  return page
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

/** Everything the assertions below care about, read in one pass. */
const snap = (page) => page.evaluate(() => {
  const root = document.getElementById('model-browser')
  const gate = document.querySelector('.cs-url-gate')
  const verbs = document.querySelector('.cs-verbs')
  const title = document.querySelector('#loading-title')
  const q = (s) => document.querySelector(s)?.textContent?.trim() ?? null
  return {
    stage: q('.mb-name'),
    gateWhat: q('#cs-gate-what'),
    gateGo: q('#cs-gate-go'),
    gateTag: gate?.tagName ?? null,
    gateDisplay: gate ? getComputedStyle(gate).display : null,
    gateAttrs: gate ? [...gate.attributes].map((a) => a.name).sort() : null,
    gateOpenProp: gate && 'open' in gate ? gate.open : null,
    verbsDisplay: verbs ? getComputedStyle(verbs).display : null,
    chatting: !!root?.classList.contains('cs-chatting'),
    bootTitle: title?.textContent?.trim() ?? null,
    ctxRow: [...document.querySelectorAll('.mb-stats div')]
      .find((d) => d.querySelector('dt')?.textContent?.includes('Context'))
      ?.querySelector('dd')?.textContent?.trim() ?? null,
    ctaHref: document.querySelector('.mb-cta')?.getAttribute('href') ?? null,
    url: location.search + location.hash,
  }
})

/**
 * Record the FIRST thing the boot card says.
 *
 * `#loading-title` opens as "Summoning <name>" — the name of the model whose
 * gigabytes are about to be fetched, which is exactly the fact under test —
 * and is rewritten to "The summoning failed" a moment later, because this
 * script cuts the network. Watch for the first value rather than reading a
 * settled DOM, or the evidence is gone by the time it is read.
 */
const armBootWatch = (page) => page.evaluate(() => {
  window.__firstBoot = null
  const grab = () => {
    const t = document.querySelector('#loading-title')
    if (t && window.__firstBoot === null) window.__firstBoot = t.textContent.trim()
  }
  new MutationObserver(grab).observe(document.body, { childList: true, subtree: true })
  grab()
})
const bootedName = (page) => page.evaluate(() =>
  (window.__firstBoot ?? '').replace(/^Summoning\s+/, '') || null)

/** The gate is up and settled (the cache probe has answered). */
async function gateReady(page) {
  await page.waitForFunction(
    () => {
      const g = document.querySelector('#cs-gate-go')
      return !!g && !g.disabled
    },
    { timeout: 20_000, polling: 100 },
  )
}

// ── the report ────────────────────────────────────────────────────────────
const results = []
function check(name, pass, detail) {
  results.push({ name, pass, detail })
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}`)
  for (const line of detail) console.log(`        ${line}`)
}

// ── HIGH-1 · one keypress walks straight past the gate ────────────────────
async function high1() {
  const page = await open('/?model=qwen38&chat=1')
  await gateReady(page)
  const before = await snap(page)
  await armBootWatch(page)
  await page.focus('#model-browser')
  await page.keyboard.press('Enter')
  await sleep(1200)
  const after = await snap(page)
  const booted = await bootedName(page)
  check('HIGH-1  Enter on the scene does not walk past the gate', !after.chatting && booted === null, [
    `before:      gate ${before.gateDisplay !== 'none' ? 'up' : 'down'}, .cs-verbs display: ${before.verbsDisplay}, chatting: ${before.chatting}`,
    'focus #model-browser, press Enter',
    `AFTER Enter: chatting: ${after.chatting}, booted: ${JSON.stringify(booted)}, gate display: ${after.gateDisplay}, url: ${JSON.stringify(after.url)}`,
  ])
  await page.close()
}

// ── HIGH-2 · the gate names a model it will not boot (keyboard route) ─────
async function high2Keyboard() {
  const page = await open('/?model=llama32&chat=1')
  await gateReady(page)
  const before = await snap(page)
  await page.focus('#model-browser')
  await page.keyboard.press('ArrowDown')
  await page.keyboard.press('ArrowDown')
  await sleep(300)
  const moved = await snap(page)
  await armBootWatch(page)
  await page.click('#cs-gate-go')
  await sleep(1500)
  const agreed = (before.gateWhat ?? '').match(/asks to run (\S+)/)?.[1] ?? '?'
  const booted = await bootedName(page)
  check('HIGH-2  the booted model is the model the gate named (arrow keys)',
    booted !== null && booted === agreed && moved.stage === before.stage, [
      `gate says:     ${JSON.stringify((before.gateWhat ?? '').slice(0, 96))}`,
      '[ArrowDown x2]',
      `gate NOW says: ${JSON.stringify((moved.gateWhat ?? '').slice(0, 96))}`,
      `stage is now:  ${moved.stage}   (was ${before.stage})`,
      "click the gate's own button",
      `BOOTED:        ${JSON.stringify(booted)}`,
      `agreed: ${agreed}   booted: ${booted}`,
    ])
  await page.close()
}

// ── HIGH-2 · same bypass by POINTER — a roster card, not a key ────────────
async function high2Pointer() {
  const page = await open('/?model=llama32&chat=1')
  await gateReady(page)
  const before = await snap(page)
  // The roster rail sits behind the gate. Click a different character's card.
  await page.evaluate(() => {
    document.querySelector('.mb-dot[data-i="1"]')?.dispatchEvent(
      new MouseEvent('click', { bubbles: true, cancelable: true }))
  })
  await sleep(300)
  const moved = await snap(page)
  await armBootWatch(page)
  await page.click('#cs-gate-go')
  await sleep(1500)
  const agreed = (before.gateWhat ?? '').match(/asks to run (\S+)/)?.[1] ?? '?'
  const booted = await bootedName(page)
  check('HIGH-2  the booted model is the model the gate named (roster click)',
    booted !== null && booted === agreed && moved.stage === before.stage, [
      `stage before: ${before.stage}   after clicking roster card 1: ${moved.stage}`,
      `agreed: ${agreed}   booted: ${JSON.stringify(booted)}`,
    ])
  await page.close()
}

// ── F2 · the gate must give the entrance back ────────────────────────────
async function f2Decline() {
  const page = await open('/?model=llama32&chat=1')
  await gateReady(page)
  const up = await snap(page)
  const hasDecline = await page.$('#cs-gate-no')
  if (hasDecline) await page.click('#cs-gate-no')
  await sleep(400)
  const after = await snap(page)
  check('F2      declining gives the verbs back and clears the act-without-a-click keys',
    !!hasDecline && after.verbsDisplay === 'flex' && !/chat=1/.test(after.url) && after.gateDisplay === 'none', [
      `while gated: .cs-verbs display: ${up.verbsDisplay}, decline button: ${hasDecline ? 'present' : 'ABSENT'}`,
      `after decline: .cs-verbs display: ${after.verbsDisplay}, gate display: ${after.gateDisplay}, url: ${JSON.stringify(after.url)}`,
    ])
  await page.close()
}

// ── F1/F3 · it has to BE a dialog, and say what it is ────────────────────
async function f3Semantics() {
  const page = await open('/?model=llama32&chat=1')
  await gateReady(page)
  const s = await snap(page)
  const need = ['role', 'aria-modal', 'aria-labelledby', 'aria-describedby']
  const missing = need.filter((a) => !(s.gateAttrs ?? []).includes(a))
  check('F3      the gate is a modal dialog with an accessible name and description',
    s.gateTag === 'DIALOG' && s.gateOpenProp === true && missing.length === 0, [
      `element: <${(s.gateTag ?? '?').toLowerCase()}>  open=${s.gateOpenProp}`,
      `attributes: ${JSON.stringify(s.gateAttrs)}`,
      `missing: ${JSON.stringify(missing)}`,
    ])
  await page.close()
}

// ── MEDIUM · ?ctx= has no lower bound ────────────────────────────────────
async function mediumCtx() {
  const share = await open('/share.html?model=qwen38&ctx=0.5')
  await sleep(1500)
  const shareText = await share.evaluate(() => document.body.innerText)
  const shareCtx = /Context\s*\n?\s*([^\n]+)/i.exec(shareText)?.[1]?.trim() ?? '(not found)'
  await share.close()

  const std = await open('/zero-tvm.html?model=qwen38&ctx=0.5')
  await sleep(1500)
  const stdText = await std.evaluate(() => document.body.innerText)
  const stdCtx = /([\d.]+K?)\s*CONTEXT/i.exec(stdText)?.[1] ?? '(not found)'
  await std.close()

  // qwen38 compiled default is 16384 tokens = 16K. A sub-page ?ctx= must be
  // refused outright, not honoured as a 16-token window.
  check('MEDIUM  ?ctx=0.5 is refused on every surface, not honoured',
    !/^16 tokens?$/i.test(shareCtx) && !/^0K?$/i.test(stdCtx), [
      `share.html?model=qwen38&ctx=0.5   Context: ${JSON.stringify(shareCtx)}`,
      `zero-tvm.html?model=qwen38&ctx=0.5   ${JSON.stringify(stdCtx)} CONTEXT`,
    ])
}

// ── LOW · the entrance drops a ?ctx= the room plan writes ─────────────────
async function lowCtx() {
  // 5000 is inside qwen38's [256, 262144] range but is not one of the three
  // enumerated builds. The engine rounds it UP to a page boundary (16-token
  // pages → 5008), and that effective number is what every surface must quote.
  const page = await open('/?model=qwen38&ctx=5000')
  await sleep(600)
  const s = await snap(page)
  check('LOW     an in-range ?ctx= the roster does not enumerate is still honoured',
    /ctx=5008/.test(s.ctaHref ?? '') && /5k/.test(s.ctxRow ?? ''), [
      `stage: ${s.stage}`,
      `Context row: ${JSON.stringify(s.ctxRow)}`,
      `ENTER href:  ${JSON.stringify(s.ctaHref)}`,
    ])
  await page.close()
}

// ── run ───────────────────────────────────────────────────────────────────
const only = process.argv.find((a) => a.startsWith('--only='))?.slice(7)
const cases = [
  ['high1', high1], ['high2-key', high2Keyboard], ['high2-pointer', high2Pointer],
  ['f2', f2Decline], ['f3', f3Semantics], ['medium', mediumCtx], ['low', lowCtx],
]
try {
  await start()
  for (const [name, fn] of cases) {
    if (only && name !== only) continue
    try { await fn() } catch (e) { check(name, false, [`threw: ${e.message}`]) }
  }
} finally {
  await stop()
}
const failed = results.filter((r) => !r.pass)
console.log(`\n${results.length - failed.length}/${results.length} passed`)
process.exit(failed.length ? 1 : 0)
