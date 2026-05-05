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
import { mkdirSync } from 'node:fs'
import { resolve } from 'node:path'
import puppeteer, { Browser, Page } from 'puppeteer'

// 5173 is the developer's own vite, 5174 is the user's neural-pulse project,
// so we pick a port well away from the common vite range.
const PORT = 5189
export const BASE = `http://localhost:${PORT}`
const REPO_ROOT = process.cwd()
export const USER_DATA_DIR = resolve(REPO_ROOT, '.tests-cache/chrome-profile')
const VITE_BIN = resolve(REPO_ROOT, 'node_modules/.bin/vite')

let viteProc: ChildProcess | null = null
let browser: Browser | null = null

async function waitForUrl(url: string, timeoutMs: number): Promise<void> {
  const start = Date.now()
  let lastErr: unknown = null
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url)
      if (r.ok) return
      lastErr = new Error(`HTTP ${r.status}`)
    } catch (e) {
      lastErr = e
    }
    await new Promise((r) => setTimeout(r, 200))
  }
  throw new Error(`timed out waiting for ${url}: ${lastErr}`)
}

export async function startHarness(): Promise<void> {
  mkdirSync(USER_DATA_DIR, { recursive: true })

  // Spawn Vite dev server. We use dev mode (not preview) so the test runs
  // against the same code path the developer sees, with no separate build.
  // --strictPort ensures we fail loudly if 5174 is already taken (rather than
  // silently colliding with a leftover server from a previous failed run).
  viteProc = spawn(
    VITE_BIN,
    ['--port', String(PORT), '--strictPort', '--clearScreen', 'false'],
    { stdio: ['ignore', 'pipe', 'pipe'], env: process.env, cwd: REPO_ROOT }
  )
  let viteFatal: Error | null = null
  viteProc.on('error', (e) => {
    viteFatal = e
  })
  viteProc.stdout?.on('data', (b) => {
    const s = b.toString()
    if (/error|ERROR|EADDRINUSE/.test(s)) process.stderr.write(`[vite] ${s}`)
  })
  viteProc.stderr?.on('data', (b) => {
    const s = b.toString()
    process.stderr.write(`[vite] ${s}`)
    if (/EADDRINUSE|already in use/.test(s)) {
      viteFatal = new Error(`port ${PORT} already in use — kill the leftover process`)
    }
  })
  viteProc.on('exit', (code) => {
    if (code !== 0 && code !== null && !viteFatal) {
      viteFatal = new Error(`vite exited with code ${code}`)
    }
  })

  // Race waitForUrl against vite's own startup failure so we don't hang for
  // 30s waiting on a server that already died.
  await Promise.race([
    waitForUrl(`${BASE}/zero-tvm.html`, 30_000),
    new Promise<never>((_, reject) => {
      const interval = setInterval(() => {
        if (viteFatal) {
          clearInterval(interval)
          reject(viteFatal)
        }
      }, 100)
    }),
  ])

  // Launch real Chrome (not headless) — shader-f16 is reliably available with
  // the full browser on macOS, less so in headless mode. The flags below are
  // the standard set for opting into experimental WebGPU features.
  browser = await puppeteer.launch({
    headless: false,
    userDataDir: USER_DATA_DIR,
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--enable-dawn-features=allow_unsafe_apis',
    ],
    defaultViewport: { width: 1100, height: 820 },
    // Default puppeteer protocolTimeout is 180s. The first cold run downloads
    // ~2 GB of weights and a single waitForFunction poll can straddle the
    // download → bump to 10 min so the very first run isn't fragile.
    protocolTimeout: 10 * 60 * 1000,
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
  page.on('pageerror', (e) => console.error(`[pageerror] ${e.message}\n${e.stack}`))
  page.on('console', (m) => {
    if (m.type() === 'error') console.error(`[console.error] ${m.text()}`)
  })
  await page.goto(`${BASE}${path}`, { waitUntil: 'domcontentloaded', timeout: 30_000 })
  // The download gate's #start-btn appears either statically (compiler-chat,
  // validate) or injected by chat.ts after the SW + OPFS-sentinel probe.
  // When OPFS already has the weight manifest (returning visitor / cache-shared
  // path) chat.ts skips the gate entirely — wait briefly and don't fail if
  // it never shows up.
  try {
    await page.waitForSelector('#start-btn', { timeout: 8_000 })
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
