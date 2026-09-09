/**
 * E2E TEST HARNESS
 *
 * Spawns a Vite dev server on a non-default port (so it doesn't fight with
 * the developer's own `npm run dev`), launches Puppeteer Chrome with the
 * WebGPU + shader-f16 flags Zero-TVM needs, and exposes a few helpers the
 * actual test files use.
 *
 * The browser profile lives at `.tests-cache/chrome-profile` so that the
 * IndexedDB-cached Phi-3 weights survive between test runs — first run is
 * slow (2 GB download), every run after is fast.
 */

import { spawn, ChildProcess } from 'node:child_process'
import { existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import puppeteer, { Browser, Page } from 'puppeteer'
import { requirePortFree, viteReady, waitForChild, watch } from './port-guard.ts'

// 5173 is the developer's own vite, 5174 is the user's neural-pulse project,
// so we pick a port well away from the common vite range.
//
// MOVABLE, for the same reason the guard below exists: a harness that cannot
// step off an occupied port is a harness that cannot run, and the guard turns
// "occupied" into a refusal. Same variable name as the three scripts/ harnesses
// — one knob, one grammar.
const PORT = Number(process.env.VITE_PORT ?? 5189)
export const BASE = `http://localhost:${PORT}`
const REPO_ROOT = process.cwd()
export const USER_DATA_DIR = resolve(REPO_ROOT, '.tests-cache/chrome-profile')

/** Walked up, not joined — gate-holds.mjs's `findBin`, for the same reason:
 *  this repo is worked on from git worktrees, which have no node_modules of
 *  their own, so `<cwd>/node_modules/.bin/vite` is ENOENT there.
 *
 *  This was INVISIBLE until the guard below landed. `spawn` reported ENOENT on
 *  the `error` event, but the old wait raced that against "does the port
 *  answer" and the port answered first whenever anything else was listening —
 *  so from a worktree this harness never started a vite AT ALL, and either
 *  timed out or graded a stranger's server. It surfaced the moment the wait
 *  started asking for OUR child's ready line. */
function findBin(name: string): string {
  for (let d = REPO_ROOT; d !== dirname(d); d = dirname(d)) {
    const p = join(d, 'node_modules/.bin', name)
    if (existsSync(p)) return p
  }
  throw new Error(`${name} not found in any node_modules/.bin above ${REPO_ROOT}`)
}
const VITE_BIN = findBin('vite')

let viteProc: ChildProcess | null = null
let browser: Browser | null = null

export async function startHarness(): Promise<void> {
  mkdirSync(USER_DATA_DIR, { recursive: true })

  // BEFORE THE SPAWN. `--strictPort` is kept below because it stops vite from
  // wandering to another port, but it was never the check this needed: its
  // EADDRINUSE lands on stderr ~0.5-1 s in, while the old wait polled the URL
  // from t≈0 and a squatter answers instantly. Measured with a foreign server
  // holding 5301 on both loopback families and the port as the only edit to
  // the old file: `Tests 2 passed | 7 skipped (9)`, FILE GREEN, for the two
  // reduce-motion assertions — satisfied by a page that is not this checkout.
  // Unfiltered the same run read `7 failed | 2 passed`, every failure a
  // selector this tree has and that server does not. See port-guard.ts.
  await requirePortFree(PORT, 'the dev server')

  // Spawn Vite dev server. We use dev mode (not preview) so the test runs
  // against the same code path the developer sees, with no separate build.
  viteProc = spawn(
    VITE_BIN,
    ['--port', String(PORT), '--strictPort', '--clearScreen', 'false'],
    { stdio: ['ignore', 'pipe', 'pipe'], env: process.env, cwd: REPO_ROOT }
  )
  const vite = watch(viteProc, (s) => {
    if (/error|ERROR|EADDRINUSE|already in use/.test(s)) process.stderr.write(`[vite] ${s}`)
  })

  // …and then wait for OUR CHILD to announce itself on OUR port, not for the
  // port to answer. A child that dies fails here immediately, with its own
  // output attached, instead of after the full timeout.
  await waitForChild(vite, `${BASE}/zero-tvm.html`, 30_000, 'vite', viteReady(PORT))

  // Launch real Chrome (not headless) — shader-f16 is reliably available with
  // the full browser on macOS, less so in headless mode. The flags below are
  // the standard set for opting into experimental WebGPU features.
  browser = await puppeteer.launch({
    headless: false,
    userDataDir: USER_DATA_DIR,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      // Ubuntu 23.10+ runners disable unprivileged user namespaces, so
      // Chrome's sandbox cannot start there at all ("No usable sandbox!").
      // Opt OUT per-run via ZTVM_CHROME_NO_SANDBOX=1 (CI sets it) rather than
      // by default: a headed browser without a sandbox runs page JS
      // unsandboxed, which is fine for localhost test pages but should stay
      // a deliberate choice, not the default.
      ...(process.env.ZTVM_CHROME_NO_SANDBOX === '1' ? ['--no-sandbox'] : []),
      // ZTVM_UNSAFE=1 adds disable_robustness: Chrome's mandatory
      // bounds-checking on every buffer access, measured at 14-23% of prefill
      // by LlamaWeb (arXiv 2605.20706). NOT web-shippable — but ztvm launches
      // its own Chrome, so the LOCAL agent surface may claim it. E2 in
      // docs/PREFILL_RESEARCH.md is the measurement.
      `--enable-dawn-features=allow_unsafe_apis${process.env.ZTVM_UNSAFE === '1' ? ',disable_robustness' : ''}`,
    ],
    defaultViewport: { width: 1100, height: 820 },
    // Default puppeteer protocolTimeout is 180s. The first cold run downloads
    // ~2 GB of weights and a single waitForFunction poll can straddle the
    // download → bump to 10 min so the very first run isn't fragile.
    //
    // Ten minutes is NOT enough for a fidelity-at-depth run: one
    // page.evaluate does the whole prefill, so 20k tokens at qwen38's
    // quarantined cap of 256 is 80 chunks inside a single protocol call and
    // it times out with the engine working correctly. ZTVM_PROTOCOL_MIN
    // raises it for those runs; the default is unchanged.
    protocolTimeout: Number(process.env.ZTVM_PROTOCOL_MIN ?? 10) * 60 * 1000,
  })
}

export async function stopHarness(): Promise<void> {
  await browser?.close().catch(() => {})
  browser = null
  if (viteProc && !viteProc.killed) {
    viteProc.kill('SIGTERM')
    // Give it a moment, then SIGKILL if still alive.
    await new Promise((r) => setTimeout(r, 300))
    if (!viteProc.killed) viteProc.kill('SIGKILL')
  }
  viteProc = null
}

export async function newPage(path: string): Promise<Page> {
  if (!browser) throw new Error('harness not started')
  const page = await browser.newPage()
  // Surface page-side errors so a crash inside the engine isn't silently
  // swallowed by puppeteer.
  page.on('pageerror', (e) => {
    const err = e instanceof Error ? e : new Error(String(e))
    console.error(`[pageerror] ${err.message}\n${err.stack}`)
  })
  page.on('console', (m) => {
    if (m.type() === 'error') { console.error(`[console.error] ${m.text()}`); return }
    // model-smoke.html's say() reports every phase through console.log('[T]',
    // ...). Dropping non-error messages made a long run indistinguishable from
    // a hung one: a 20k-token fidelity-at-depth prefill sat for 40 minutes with
    // the page narrating each step and nothing reaching the terminal. Forward
    // the page's own progress markers; ZTVM_PAGE_LOG=1 forwards everything.
    const t = m.text()
    if (process.env.ZTVM_PAGE_LOG === '1' || t.startsWith('[T]')) console.log(`[page] ${t}`)
  })
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  // The download gate's #start-btn appears either statically (compiler-chat,
  // validate) or injected by chat.ts after the SW + OPFS-sentinel probe.
  // When OPFS already has the weight manifest (returning visitor / cache-shared
  // path) chat.ts skips the gate entirely — wait briefly and don't fail if
  // it never shows up.
  try {
    // visible: true — on zero-tvm.html the button sits inside a closed
    // <dialog> until the async cache check calls showModal(); an
    // existence-only wait resolves too early and the later click is lost.
    await page.waitForSelector('#start-btn', { visible: true, timeout: 8_000 })
  } catch {
    // No gate — auto-boot (cached) path. bootAndWaitReady handles both.
  }
  return page
}

/** Click the "Download & Start" / "Run Validation" button if the gate is
 * present, then wait for the shared loading-ui badge to flip to Ready. The
 * first call may take minutes (weight download); subsequent calls finish
 * quickly. Tolerates the cache-shared / cached-skip path where the gate is
 * never injected. */
export async function bootAndWaitReady(page: Page, timeoutMs: number): Promise<void> {
  const hasGate = await page.$('#start-btn')
  if (hasGate) await page.click('#start-btn')
  await page.waitForFunction(
    () => document.getElementById('badge')?.textContent === 'Ready',
    { timeout: timeoutMs, polling: 500 }
  )
}
