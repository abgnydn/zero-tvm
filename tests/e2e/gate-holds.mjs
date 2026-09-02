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
// The guard scripts/{split-serve,share,peer-weights}-e2e.mjs got on 2026-08-25
// and this file did not — which matters most here, because this is the script
// that certifies the gate. Measured 2026-09-02, with a foreign server holding
// this file's port on both loopback families and the port as the only edit to
// the old file: it printed `[vite] error when starting dev server: Error: Port
// 5301 is already in use` from its own dying child, ran the case against that
// server anyway, and reported `0/1 passed`. The symmetric case is the
// dangerous one — a foreign tree that happens to pass hands out the green this
// script's fix is published on.
import { requirePortFree, viteReady, waitForChild, watch } from './port-guard.ts'

// Movable, because the guard turns "occupied" into a refusal and a harness
// that cannot step aside is a harness that cannot run. Same variable name as
// the four sibling harnesses.
const PORT = Number(process.env.VITE_PORT ?? 5285)
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

async function start() {
  // Before the spawn: a port that already answers is refused outright. After
  // it: wait for OUR child's own ready line on OUR port, never for "the port
  // responds" — a squatter answers that poll long before vite's EADDRINUSE
  // reaches stderr, and on the other loopback family vite raises no EADDRINUSE
  // at all. Both measurements are in port-guard.ts.
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

  // (a) The bypass as reported, aimed straight at the listener that carried
  //     it. A keydown whose target is #model-browser is exactly what the
  //     splash-dismissing click produced — the scene is a DIV with
  //     tabIndex = 0 — and it reaches the handler whatever the focus ring is
  //     doing, so this does not depend on where showModal() put focus.
  await page.evaluate(() => {
    document.getElementById('model-browser')
      ?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
  })
  await sleep(800)
  const dispatched = await snap(page)
  const bootedA = await bootedName(page)

  // (b) And through the real keyboard. The scene is inert under showModal(),
  //     so focus cannot even land on it — the press goes to whatever the
  //     dialog holds. Either way nothing may boot.
  await page.focus('#model-browser')
  const focusAfter = await page.evaluate(() => {
    const a = document.activeElement
    return a ? `${a.tagName}${a.id ? `#${a.id}` : ''}` : 'none'
  })
  await page.keyboard.press('Enter')
  await sleep(1200)
  const after = await snap(page)
  const bootedB = await bootedName(page)

  check('HIGH-1  Enter on the scene does not walk past the gate',
    !dispatched.chatting && bootedA === null && dispatched.gateDisplay !== 'none'
    && !after.chatting && bootedB === null, [
      `before:        gate ${before.gateDisplay !== 'none' ? 'up' : 'down'}, .cs-verbs display: ${before.verbsDisplay}, chatting: ${before.chatting}`,
      'dispatch keydown Enter AT #model-browser (the reported path):',
      `               chatting: ${dispatched.chatting}, booted: ${JSON.stringify(bootedA)}, gate display: ${dispatched.gateDisplay}, url: ${JSON.stringify(dispatched.url)}`,
      `focus #model-browser → activeElement is ${focusAfter} (the scene is inert), press Enter:`,
      `               chatting: ${after.chatting}, booted: ${JSON.stringify(bootedB)}, gate display: ${after.gateDisplay}, url: ${JSON.stringify(after.url)}`,
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

// ── F2 · Escape is a decline, and it works while the probe is still out ──
async function f2Escape() {
  const page = await open('/?model=llama32&chat=1')
  // Deliberately NOT gateReady: the dialog is on screen before the cache
  // probe resolves, and the first version wired "Not now" and the close
  // handler only after that await — so for the length of the probe, Escape
  // closed the dialog with nothing listening and left the scene gated, the
  // verbs hidden, and no dialog to act on.
  await page.waitForSelector('.cs-url-gate[open]', { timeout: 15_000 })
  await page.keyboard.press('Escape')
  await sleep(400)
  const after = await snap(page)
  check('F2      Escape declines, even mid-probe',
    after.gateDisplay === 'none' && after.verbsDisplay === 'flex' && !/chat=1/.test(after.url), [
      `after Escape: gate display: ${after.gateDisplay}, .cs-verbs display: ${after.verbsDisplay}, url: ${JSON.stringify(after.url)}`,
    ])
  await page.close()
}

// ── layout · the two widths the review measured as fine ──────────────────
async function mobile() {
  const bad = []
  for (const [w, h] of [[390, 844], [360, 740]]) {
    const page = await open('/?model=qwen36q3&chat=1')
    await page.setViewport({ width: w, height: h })
    await gateReady(page)
    const m = await page.evaluate(() => {
      const d = document.querySelector('.cs-url-gate')
      const r = d.getBoundingClientRect()
      const btns = [...d.querySelectorAll('button')].map((b) => {
        const q = b.getBoundingClientRect()
        return { id: b.id, w: Math.round(q.width), inView: q.top >= 0 && q.bottom <= innerHeight }
      })
      return {
        page: document.documentElement.scrollWidth <= innerWidth + 1,
        fits: r.left >= -1 && r.right <= innerWidth + 1 && r.height <= innerHeight,
        box: `${Math.round(r.width)}x${Math.round(r.height)} at ${Math.round(r.left)},${Math.round(r.top)}`,
        btns,
      }
    })
    const ok = m.page && m.fits && m.btns.length === 2 && m.btns.every((b) => b.w > 0 && b.inView)
    if (!ok) bad.push(`${w}x${h}`)
    check(`layout  the gate fits ${w}x${h} and both buttons are reachable`, ok, [
      `dialog ${m.box}; no horizontal page scroll: ${m.page}; inside the viewport: ${m.fits}`,
      `buttons: ${JSON.stringify(m.btns)}`,
    ])
    await page.close()
  }
  return bad
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
  const shareCtx = /Context\s*\n?\s*([^\n]+)/i.exec(shareText)?.[1]?.trim() ?? null
  await share.close()

  const std = await open('/zero-tvm.html?model=qwen38&ctx=0.5')
  await sleep(1500)
  const stdText = await std.evaluate(() => document.body.innerText)
  const stdCtx = /([\d.]+K?)\s*CONTEXT/i.exec(stdText)?.[1] ?? null

  await std.close()

  // FOUND FIRST, then judged. Both assertions below are NEGATIVE — "the page
  // does not say 16 tokens", "the page does not say 0K" — and a `?? '(not
  // found)'` fallback satisfies both, so a surface that stopped rendering its
  // context figure at all (a rename, a layout change, a page that failed to
  // boot) read as a PASS for a check that had nothing to read. Same shape as
  // stage-honesty.test.ts's ANCHOR guard: an anchor that went missing must be
  // a failure, not a silent green.
  const found = shareCtx !== null && stdCtx !== null
  // qwen38 compiled default is 16384 tokens = 16K. A sub-page ?ctx= must be
  // refused outright, not honoured as a 16-token window.
  check('MEDIUM  ?ctx=0.5 is refused on every surface, not honoured',
    found && !/^16 tokens?$/i.test(shareCtx) && !/^0K?$/i.test(stdCtx), [
      `share.html?model=qwen38&ctx=0.5   Context: ${JSON.stringify(shareCtx ?? '(NO CONTEXT FIGURE ON THE PAGE)')}`,
      `zero-tvm.html?model=qwen38&ctx=0.5   ${JSON.stringify(stdCtx ?? '(NO CONTEXT FIGURE ON THE PAGE)')} CONTEXT`,
      ...(found ? [] : ['a surface rendered no context figure at all — this check read nothing']),
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
  ['f2', f2Decline], ['f2-esc', f2Escape], ['f3', f3Semantics],
  ['mobile', mobile], ['medium', mediumCtx], ['low', lowCtx],
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
