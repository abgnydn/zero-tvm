#!/usr/bin/env node
/**
 * TWO DOM FACTS THE ENTRANCE'S CHAT SURFACE HAS TO KEEP.
 *
 * Both are wiring/layout facts — what the cascade resolved to, and what a
 * pending network request does to a button — so neither is decidable without a
 * browser. tests/unit/entrance-gate.test.ts already says where facts of this
 * shape live: "those are DOM facts and they are checked in a real browser".
 *
 * ── 1 · THE CONSENT GATE'S BOUND HAS TO COVER THE WHOLE PROBE ────────────
 *
 * openUrlGate opens the dialog with `#cs-gate-go` disabled and enables it when
 * the OPFS cache probe answers. The probe was bounded at 2500 ms so that a
 * probe which never settles degrades to ASKING WITHOUT THE CACHED WORDING
 * rather than to never asking — the commit's own words. But the bound wrapped
 * `isModelCached(...)` only, and the line above it,
 *
 *     const { isModelCached } = await import('./zero-tvm/cache-probe.js')
 *
 * was outside it. One held module request and the button never enables.
 * Measured on the unfixed tree by holding that single request open:
 *
 *     held requests: 1
 *     after 20002 ms: #cs-gate-go disabled=true label="Download & enter →"
 *     go enabled within 20s? false
 *
 * Escape and "Not now" still work — they are wired before the await — so the
 * visitor is not stranded with no way BACK. They are stranded with no way
 * FORWARD, which is the exact state the bound was added to prevent.
 *
 * The module is held, not failed, on purpose: a failed import rejects and the
 * try/catch already covers that. Only a request that never settles reaches
 * the missing bound.
 *
 * ── 2 · EVERY CONTROL IN THE CHAT HEAD HAS TO BE REACHABLE ON A PHONE ────
 *
 * mountRoomTool adds `⟁ Room` and `⟁ Awake` to `.cs-chat-head`, which already
 * carries the sigil, the identity, the live badge and `⟨ Roster`. The row is a
 * nowrap flex line, and the two new tools pushed `⟨ Roster` — the ONLY in-chat
 * way back to the character select — off the end of it. Measured on the
 * unfixed tree, mountRoomTool driven the way landing-chat.ts drives it:
 *
 *     360x740   head scroll/client 378/330   roster right edge +48.4 past it
 *     390x844   head scroll/client 378/360   roster right edge +18.4 past it
 *     320x568   head scroll/client 378/290   roster right edge +88.4 past it
 *
 * At 320 the link's left edge is at 317.8 in a 320px viewport: 2.2px of a
 * 75.6px control on screen.
 *
 * THE ASSERTION IS ABOUT REACHABILITY, NOT ABOUT PAGE SCROLL. The companion
 * claim "no sideways scroll, documentElement.scrollWidth == innerWidth" is
 * TRUE on the unfixed tree, and true PRECISELY BECAUSE `#model-browser` is
 * `overflow: hidden` and clips the control that does not fit. A number can be
 * right and support nothing. So what is measured here is each control's own
 * box against the head's CLIENT box, after scrolling THE HEAD as far as a
 * person could — and only when the head is user-scrollable (overflow-x auto or
 * scroll). Not scrollIntoView: an overflow:hidden box is scrollable
 * programmatically, so scrollIntoView pans the very container that does the
 * clipping and reports every control reachable. A thumb cannot do that.
 *
 * Why a standalone script and not a vitest file: tests/unit must stay GPU-free
 * and fast, and tests/e2e/harness.ts owns port 5189 and a SHARED Chrome
 * profile that other agents are using right now. This brings its own vite
 * (5303) and its own throwaway profile and touches neither.
 *
 *   node tests/e2e/probe-bound-and-head.mjs          # run both cases
 *   node tests/e2e/probe-bound-and-head.mjs --show   # watch it happen
 *   node tests/e2e/probe-bound-and-head.mjs --only=head
 */

import { spawn } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import puppeteer from 'puppeteer'

const PORT = Number(process.env.VITE_PORT ?? 5303)
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
const profile = mkdtempSync(join(tmpdir(), 'ztvm-head-'))
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

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// ── the report ────────────────────────────────────────────────────────────
const results = []
function check(name, pass, detail) {
  results.push({ name, pass })
  console.log(`${pass ? 'PASS' : 'FAIL'}  ${name}`)
  for (const line of detail) console.log(`        ${line}`)
}

// ── 1 · the gate's bound covers the module fetch ──────────────────────────
/**
 * The gate, with the cache-probe MODULE request held open forever.
 *
 * Everything else is served normally: this is not an offline page, it is a
 * page with one slow import. `--enable-unsafe-webgpu` is passed above because
 * the probe runs only under `'gpu' in navigator`; if the browser has no
 * navigator.gpu the branch is skipped, nothing is held, and the case would
 * pass while proving nothing — so that is asserted as a premise, not assumed.
 */
async function probeBound() {
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error(`  [pageerror] ${e.message}`))
  await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])

  const held = []
  await page.setRequestInterception(true)
  page.on('request', (r) => {
    // Held, NOT aborted and NOT failed: a rejected import is already covered
    // by the try/catch around the probe. The unbounded await is only reachable
    // through a request that never settles.
    if (/cache-probe/.test(r.url())) { held.push(r.url()); return }
    void r.continue().catch(() => {})
  })
  await page.goto(`${BASE}/?model=qwen38&chat=1`, {
    waitUntil: 'domcontentloaded', timeout: 30_000,
  })
  await page.waitForSelector('.cs-url-gate[open]', { timeout: 15_000 })

  const hasGpu = await page.evaluate(() => 'gpu' in navigator)

  // The bound is 2500 ms. 20 s is eight times it: long enough that "slow" and
  // "never" are not confused, short enough to run in a loop.
  const t0 = Date.now()
  let enabled = false
  while (Date.now() - t0 < 20_000) {
    enabled = await page.evaluate(() => {
      const g = document.querySelector('#cs-gate-go')
      return !!g && !g.disabled
    })
    if (enabled) break
    await sleep(100)
  }
  const waited = Date.now() - t0
  const label = await page.evaluate(() =>
    document.querySelector('#cs-gate-go')?.textContent?.trim() ?? null)

  // The way BACK still has to work while the probe is out — that half was
  // already right and must not be traded away for the way forward.
  await page.keyboard.press('Escape')
  await sleep(400)
  const after = await page.evaluate(() => ({
    gateOpen: document.querySelector('.cs-url-gate')?.open ?? null,
    verbs: document.querySelector('.cs-verbs')
      ? getComputedStyle(document.querySelector('.cs-verbs')).display : null,
    url: location.search,
  }))

  check('gate    a held cache-probe module still lets the visitor say yes',
    hasGpu && held.length > 0 && enabled && after.gateOpen === false && after.verbs === 'flex', [
      `premise: navigator.gpu present: ${hasGpu}; held requests: ${held.length}`
      + `${held.length ? ` (${held[0].replace(BASE, '')})` : ''}`,
      `after ${waited} ms: #cs-gate-go enabled=${enabled} label=${JSON.stringify(label)}`,
      `go enabled within 20s? ${enabled}`,
      `Escape after that: gateOpen=${after.gateOpen} verbs=${after.verbs} url=${JSON.stringify(after.url)}`,
    ])
  await page.close()
}

// ── 2 · every chat-head control is reachable on a phone ───────────────────
/**
 * The entrance's chat panel with the room tools mounted, measured at three
 * phone widths.
 *
 * mountRoomTool is called the way landing-chat.ts calls it once the engine is
 * live — the strip's geometry needs the spec and the split and nothing else,
 * so no weight is fetched and no engine exists. Same driving as
 * tests/e2e/entrance-motion-and-mobile.test.ts's tall-strip case.
 */
const HEAD_PROBE = `(async () => {
  const load = new Function('p', 'return import(p)')
  const { panelMarkup } = await load('/src/landing-chat.ts')
  const { specForParam, modelBranding } = await load('/src/zero-tvm/model-registry.ts')
  const { splitBounds } = await load('/src/zero-tvm/room-url.ts')
  const { mountRoomTool } = await load('/src/landing-room.ts')

  const root = document.getElementById('model-browser')
  if (!root) return null
  const spec = specForParam('qwen38')
  const brand = modelBranding(spec)
  const bounds = splitBounds(spec.layers, 4)
  const range = { start: bounds[0], end: bounds[1] }

  const panel = document.createElement('section')
  panel.className = 'cs-chat'
  panel.innerHTML = panelMarkup(spec, brand, brand.params, range)
  root.appendChild(panel)
  root.classList.add('cs-chatting')
  // enterChat marks the boot card done once the engine is ready — the head is
  // only this crowded AFTER that, because mountRoomTool runs then too.
  panel.querySelector('.cs-boot')?.classList.add('cs-done')

  mountRoomTool({
    root, panel, spec, param: 'qwen38',
    engine: null, tokenizer: null, mascot: null, lock: null,
    poolSlots: 0, openStrip: false,
    stageRange: range,
    split: { bounds: [...bounds], index: 0, ctx: 4096 },
  })

  const head = panel.querySelector('.cs-chat-head')
  if (!head) return null
  const cs = getComputedStyle(head)

  // The head's CONTENT box in viewport coordinates — what "inside the head"
  // means. Not the border box: a control sitting on the padding is out.
  const contentBox = () => {
    const r = head.getBoundingClientRect()
    return {
      left: r.left + head.clientLeft,
      right: r.left + head.clientLeft + head.clientWidth,
      top: r.top + head.clientTop,
      bottom: r.top + head.clientTop + head.clientHeight,
    }
  }

  // NOT scrollIntoView. An overflow:hidden box is scrollable
  // PROGRAMMATICALLY — scrollIntoView will happily pan #model-browser, which
  // is exactly the box that clips this row — and a thumb cannot. Only the head
  // is scrolled, and only when the head is scrollable BY A PERSON: overflow-x
  // auto or scroll. Anything else and "reachable" would be a fact about the
  // test rather than about the phone.
  const userScrollable = cs.overflowX === 'auto' || cs.overflowX === 'scroll'
  const show = (el) => {
    if (!userScrollable) return
    head.scrollLeft = 0
    const from = el.getBoundingClientRect().left - contentBox().left
    head.scrollLeft = Math.max(0, Math.min(head.scrollWidth - head.clientWidth,
      from + el.getBoundingClientRect().width - head.clientWidth))
  }

  // Every control a person can act on. The sigil is decorative
  // (aria-hidden) and the badge is a readout, so neither is a control — but
  // both are measured, because a readout pushed off the row is still a
  // regression and the numbers should say which.
  const parts = [...head.children].map((el) => {
    const id = el.id ? '#' + el.id : '.' + (el.className || el.tagName.toLowerCase()).split(' ')[0]
    head.scrollLeft = 0
    const r0 = el.getBoundingClientRect()
    const box0 = contentBox()
    show(el)
    const r = el.getBoundingClientRect()
    const box = contentBox()
    return {
      id,
      w: +r.width.toFixed(1),
      // Where it sits before anything is scrolled: the refuter's column.
      pastRightAtRest: +(r0.right - box0.right).toFixed(1),
      leftAtRest: +r0.left.toFixed(1),
      // Reachable = fully inside the head's content box AND inside the
      // viewport, once the head has been scrolled as far as a person could.
      inHead: r.left >= box.left - 1 && r.right <= box.right + 1
        && r.top >= box.top - 1 && r.bottom <= box.bottom + 1,
      inViewport: r.left >= -1 && r.right <= innerWidth + 1
        && r.top >= -1 && r.bottom <= innerHeight + 1,
      actionable: el.tagName === 'BUTTON' || el.tagName === 'A',
    }
  })
  head.scrollLeft = 0

  const idW = panel.querySelector('.cs-chat-id')?.getBoundingClientRect().width ?? null
  // What the row costs the conversation, and whether the composer survives it.
  // A head that wraps takes its height from .chat-main, so this is the price
  // and it has to be reported next to the reachability it buys.
  const pr = panel.getBoundingClientRect()
  const mainH = panel.querySelector('.chat-main').getBoundingClientRect().height
  const composerPast = panel.querySelector('.composer-wrap').getBoundingClientRect().bottom - pr.bottom

  return {
    scrollW: head.scrollWidth,
    clientW: head.clientWidth,
    headH: +head.getBoundingClientRect().height.toFixed(2),
    mainH: +mainH.toFixed(2),
    composerPast: +composerPast.toFixed(2),
    flexWrap: cs.flexWrap,
    overflowX: cs.overflowX,
    // TRUE on the unfixed tree, and it proves nothing: #model-browser is
    // overflow:hidden, so it clips rather than scrolls. Recorded to keep the
    // number and the conclusion visibly separate.
    docFits: document.documentElement.scrollWidth <= innerWidth + 1,
    chatIdW: idW === null ? null : +idW.toFixed(1),
    parts,
  }
})()`

async function headReach() {
  const rows = []
  let allOk = true
  for (const [w, h] of [[320, 568], [360, 740], [390, 844]]) {
    const page = await browser.newPage()
    page.on('pageerror', (e) => console.error(`  [pageerror] ${e.message}`))
    await page.emulateMediaFeatures([{ name: 'prefers-reduced-motion', value: 'reduce' }])
    await page.goto(`${BASE}/`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
    await page.setViewport({ width: w, height: h, deviceScaleFactor: 1 })
    await page.reload({ waitUntil: 'networkidle0', timeout: 30_000 })
    await page.waitForSelector('#model-browser', { timeout: 20_000 })
    const m = await page.evaluate(HEAD_PROBE)

    if (!m) {
      check(`head    ${w}x${h}: the panel mounted`, false, ['mountRoomTool did not produce a head'])
      allOk = false
      await page.close()
      continue
    }

    // PREMISE. Without both room tools this is not the crowded head and the
    // case covers nothing.
    const ids = m.parts.map((p) => p.id)
    const premise = ids.includes('#room-btn') && ids.includes('#awake')
      && ids.includes('#cs-roster-link')
    const controls = m.parts.filter((p) => p.actionable && p.w > 0)
    const unreachable = controls.filter((p) => !(p.inHead && p.inViewport))
    // The row must not buy its reachability by pushing the composer out of the
    // panel — that is the failure boot-card-fits-the-panel.test.ts exists for,
    // and a taller head is exactly the shape that could cause it.
    const ok = premise && unreachable.length === 0 && m.composerPast <= 1 && m.mainH > 0
    if (!ok) allOk = false

    const roster = m.parts.find((p) => p.id === '#cs-roster-link')
    rows.push({ size: `${w}x${h}`, scrollW: m.scrollW, clientW: m.clientW,
      headH: m.headH, chatIdW: m.chatIdW, mainH: m.mainH,
      rosterPast: roster?.pastRightAtRest, rosterLeft: roster?.leftAtRest })

    check(`head    ${w}x${h}: every chat-head control is reachable`, ok, [
      `premise: head carries ${JSON.stringify(ids)}`,
      `head scrollWidth/clientWidth: ${m.scrollW}/${m.clientW}`
      + `   height: ${m.headH}   flex-wrap: ${m.flexWrap}   overflow-x: ${m.overflowX}`,
      `.cs-chat-id width: ${m.chatIdW}`,
      `the row's price: .chat-main ${m.mainH}px, composer ${m.composerPast}px past the panel`,
      `⟨ Roster at rest: ${roster?.pastRightAtRest}px past the head's right edge, `
      + `left edge at ${roster?.leftAtRest} in a ${w}px viewport, width ${roster?.w}`,
      `unreachable controls: ${JSON.stringify(unreachable.map((p) =>
        `${p.id} inHead=${p.inHead} inViewport=${p.inViewport}`))}`,
      `(page does not scroll sideways: ${m.docFits} — TRUE either way, `
      + '#model-browser clips. Not the assertion.)',
    ])
    await page.close()
  }

  console.log('')
  console.log('        head        scroll/client  head h   .cs-chat-id  .chat-main  ⟨ Roster past right  ⟨ Roster left')
  for (const r of rows) {
    console.log(`        ${r.size.padEnd(11)} ${String(`${r.scrollW}/${r.clientW}`).padEnd(14)} `
      + `${String(r.headH).padEnd(8)} ${String(r.chatIdW).padEnd(12)} ${String(r.mainH).padEnd(11)} `
      + `${String(r.rosterPast).padEnd(20)} ${r.rosterLeft}`)
  }
  console.log('')
  return allOk
}

// ── run ───────────────────────────────────────────────────────────────────
const only = process.argv.find((a) => a.startsWith('--only='))?.slice(7)
const cases = [['probe', probeBound], ['head', headReach]]

await start()
try {
  for (const [name, fn] of cases) {
    if (only && only !== name) continue
    await fn()
  }
} finally {
  await stop()
}

const failed = results.filter((r) => !r.pass)
console.log(`\n${results.length - failed.length}/${results.length} passed`)
for (const f of failed) console.log(`  FAILED: ${f.name}`)
process.exit(failed.length ? 1 : 0)
