/**
 * A SERVER THIS HARNESS DID NOT START IS NOT THIS HARNESS'S SERVER.
 *
 * scripts/{split-serve,share,peer-weights}-e2e.mjs each grew this guard on
 * 2026-08-25, after an orphaned vite was found answering a harness's polls and
 * carrying a whole run to PASS for code it never loaded. The two harnesses
 * under tests/e2e — the vitest harness and gate-holds.mjs — did not, and they
 * are the two that back this repo's browser evidence. Measured here on
 * 2026-09-02, each against a foreign HTTP server holding 5301 on BOTH loopback
 * families, with the harness's port as the only edit to the pre-fix file:
 *
 *   - tests/e2e/harness.ts reported `Tests 2 passed | 7 skipped (9)`, FILE
 *     GREEN, for `-t "runs no animation under reduce"` — two assertions about
 *     `document.getAnimations()` satisfied by a page that is not this checkout
 *     and has nothing to animate. Unfiltered, the same run was `7 failed |
 *     2 passed`, every failure a selector that exists only in this tree
 *     (`#model-browser`, `.cs-chat .cs-room`). It graded the other server.
 *   - tests/e2e/gate-holds.mjs printed `[vite] error when starting dev server:
 *     Error: Port 5301 is already in use` from its own dying child, then ran
 *     the case against that server anyway and reported `0/1 passed`. The
 *     symmetric case is the dangerous one: a foreign tree that happens to pass
 *     hands out the green this script's fix is published on.
 *
 * WHY `--strictPort` IS NOT THE CHECK, in two parts:
 *
 *   1. It fires far too late to be the check. Its EADDRINUSE reaches the
 *      child's stderr ~0.5-1 s after spawn while the old wait polled the URL
 *      from t≈0, and a squatter is already listening. Both runs above printed
 *      that error and continued regardless — the harness had already decided
 *      the server was up.
 *   2. When the squatter holds only the OTHER loopback family there is no
 *      EADDRINUSE AT ALL. Measured on this machine: a vite here binds
 *      `[::1]:<port>` and nothing else (`lsof` reads `TCP [::1]:5302
 *      (LISTEN)`), and `vite --port 5301 --strictPort` started NORMALLY,
 *      printing its ready line on 127.0.0.1, while a dual-stack `[::]:5301`
 *      socket was open. Which server a client then reaches is decided by that
 *      client's resolver; the earlier run of this same demo was adopted by
 *      Node's fetch and by one Chrome while another Chrome reached vite.
 *
 * So two things replace it, and both are needed: refuse a port that ALREADY
 * answers on EITHER family, and then wait for OUR CHILD to announce itself
 * rather than for the port to respond.
 *
 * ONE COPY, TWO CONSUMERS. harness.ts imports this as TypeScript; gate-holds.mjs
 * imports it as `./port-guard.ts` and Node strips the types (on by default
 * since 22.18/23.6, and this repo's CI pins `node-version: 22`). The three
 * scripts/ harnesses still carry their own copies — tests/unit/e2e-harness-ports.test.ts
 * lifts the matchers out of each of those files by source regex, so migrating
 * them belongs with that test, not here.
 */

import type { ChildProcess } from 'node:child_process'

/**
 * Refuse a port that already answers. Both loopback families, because either
 * can be the one the browser's resolver picks for `localhost`.
 */
export async function requirePortFree(port: number, what: string): Promise<void> {
  for (const host of ['127.0.0.1', '[::1]']) {
    const answered = await fetch(`http://${host}:${port}/`, { signal: AbortSignal.timeout(2000) })
      .then(() => true, () => false)
    if (!answered) continue
    throw new Error(
      `http://${host}:${port}/ already answers — something this harness did not start is `
      + `serving ${what}. Refusing to run: a previous version of this file would have tested `
      + `that server instead and reported PASS. Stop it (lsof -nP -iTCP:${port} -sTCP:LISTEN) `
      + 'or move this harness with VITE_PORT=<free port>, and rerun.')
  }
}

/**
 * A child's output with ANSI escapes stripped, for MATCHING only — error dumps
 * keep the colour. Under FORCE_COLOR=1 vite's ready line is
 * `<esc>[1mLocal<esc>[22m:   <esc>[36mhttp://localhost:<esc>[1m5294<esc>[22m/`:
 * escapes sit between `Local` and `:` AND inside the port digits, so a regex
 * written for plain text never fires and a server that was ready in 125 ms
 * reads as a 30 s hang. Piped stdout turns colour off by default, which is why
 * the happy path never sees it — a CI, or a shell exporting FORCE_COLOR, does.
 */
export const plain = (s: string): string => s.replace(/\u001B\[[0-9;]*[A-Za-z]/g, '')

/**
 * vite's own ready line, PINNED TO THE PORT WE ASKED FOR. The port is the whole
 * point: matching a ready line from some server is precisely the bug the guard
 * above exists to kill.
 */
export const viteReady = (port: number): RegExp =>
  new RegExp(`Local:\\s+https?://localhost:${port}/`)

/** A spawned server plus the tail of everything it has said. */
export interface Watched {
  proc: ChildProcess
  /** The last few KB of the child's combined output, colour intact. */
  log: string
  /** Set once the child dies, with why. */
  dead: string | null
}

/**
 * Keep a child's output and its fate, so the wait below can tell "not ready
 * yet" from "died 400 ms ago". `onLine` sees every chunk as it arrives, for
 * harnesses that want to echo it.
 */
export function watch(proc: ChildProcess, onLine?: (chunk: string) => void): Watched {
  const w: Watched = { proc, log: '', dead: null }
  const keep = (b: unknown): void => {
    const s = String(b)
    w.log = (w.log + s).slice(-8000)
    onLine?.(s)
  }
  proc.stdout?.on('data', keep)
  proc.stderr?.on('data', keep)
  proc.on('error', (e: NodeJS.ErrnoException) => { w.dead = `failed to start (${e.code ?? e.message})` })
  proc.on('exit', (code, sig) => { w.dead = `exited ${code ?? sig}` })
  return w
}

/**
 * Wait for the server WE STARTED. The URL is polled only once the child has
 * printed its own ready line for our port, so an unrelated server answering on
 * that port can never satisfy this — and a child that dies fails immediately
 * with its own output attached, rather than after the full timeout.
 */
export async function waitForChild(
  w: Watched,
  url: string,
  timeoutMs: number,
  what: string,
  ready: RegExp,
): Promise<void> {
  const t0 = Date.now()
  while (Date.now() - t0 < timeoutMs) {
    if (w.dead) {
      throw new Error(`${what} ${w.dead} before serving ${url}\n--- its output ---\n${w.log.trim()}`)
    }
    if (ready.test(plain(w.log))) {
      try { await fetch(url); return } catch { /* bound but not serving that path yet */ }
    }
    await new Promise((r) => setTimeout(r, 150))
  }
  throw new Error(`timed out waiting for ${what} at ${url}\n--- its output ---\n${w.log.trim()}`)
}
