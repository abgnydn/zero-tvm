/**
 * THE THREE MULTI-BROWSER HARNESSES MUST BE ABLE TO RUN.
 *
 * scripts/{split-serve,share,peer-weights}-e2e.mjs share a port guard added on
 * 2026-08-25 after an orphaned vite was found answering a harness's polls and
 * carrying a whole run to PASS for code it never loaded. The guard is right and
 * is not the subject here. Its two FALSE refusals are.
 *
 *   1. share-e2e.mjs could not run on a developer machine at all. SIGNAL_PORT
 *      was a hardcoded 8787 — wrangler's own default — with no environment
 *      escape hatch, so any `wrangler dev` anywhere on the box tripped the
 *      guard, including this repo's own share-signal worker. Measured
 *      2026-09-01: `node scripts/share-e2e.mjs` refused in 0.58 s, every time,
 *      while a copy differing only in the port reached puppeteer.launch in
 *      1.3 s. VITE_PORT already had the hatch; SIGNAL_PORT did not.
 *
 *   2. FORCE_COLOR=1 broke the ready match in all three. VITE_READY assumes
 *      uncoloured output, and with colour forced there are ANSI escapes
 *      between `Local` and `:` AND INSIDE THE PORT DIGITS, so it never fires:
 *      `VITE_PORT=5294 FORCE_COLOR=1 node scripts/peer-weights-e2e.mjs` ran
 *      31.9 s and exited 1 against a vite that had printed "ready in 125 ms".
 *      Piped stdout turns colour off by default, so the happy path never saw
 *      it — a CI, or a shell exporting FORCE_COLOR, turned a working harness
 *      into a mystery timeout.
 *
 * A SOURCE-TEXT TEST, deliberately, like gpu-features-single-source.test.ts.
 * These files are top-level scripts: importing one spawns wrangler, vite and a
 * browser. So the matchers are LIFTED OUT of each file's own source and
 * evaluated here — what is asserted is the regex the harness will actually
 * use, not a copy of it that could drift. The lines it is tested against are
 * byte-exact captures of real vite 6.4.1 and wrangler 4.105.0 output taken on
 * this machine, coloured and plain.
 */

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '../..')

/** Every harness that starts a relay and a dev server and then waits for them.
 *  A fourth one belongs in this list; missing from it, it is unguarded. */
const HARNESSES = [
  'scripts/split-serve-e2e.mjs',
  'scripts/share-e2e.mjs',
  'scripts/peer-weights-e2e.mjs',
]

/** Real output, captured with `spawn` + JSON.stringify so that not one byte is
 *  transcribed by hand. The port is part of each fixture because the port is
 *  the whole point of these regexes: matching a ready line from SOME server is
 *  precisely the bug the guard beside them was written to kill. */
const VITE_COLOUR = '  \u001B[32m➜\u001B[39m  \u001B[1mLocal\u001B[22m:   '
  + '\u001B[36mhttp://localhost:\u001B[1m5297\u001B[22m/\u001B[39m'
const VITE_COLOUR_PORT = 5297
const VITE_PLAIN = '  ➜  Local:   http://localhost:5298/'
const VITE_PLAIN_PORT = 5298
const WRANGLER_COLOUR = '\u001B[32m[wrangler:info]\u001B[39m Ready on '
  + '\u001B[32mhttp://localhost:8799\u001B[39m'
const WRANGLER_COLOUR_PORT = 8799
const WRANGLER_PLAIN = '[wrangler:info] Ready on http://localhost:8796'
const WRANGLER_PLAIN_PORT = 8796

interface Matchers {
  plain: (s: string) => string
  VITE_READY: RegExp
  SIGNAL_READY: RegExp
}

/** The file's own `const plain` / `const VITE_READY` / `const SIGNAL_READY`,
 *  evaluated with the ports bound. A definition that stops being a one-liner
 *  throws here rather than being silently skipped. */
function matchersOf(rel: string, vitePort: number, signalPort: number): Matchers {
  const src = readFileSync(join(ROOT, rel), 'utf8')
  const rhs = (name: string): string => {
    const m = new RegExp(`^const ${name} = (.+)$`, 'm').exec(src)
    if (!m) throw new Error(`${rel}: no single-line \`const ${name} = …\` to lift`)
    return m[1]
  }
  const build = new Function('VITE_PORT', 'SIGNAL_PORT', `
    const plain = ${rhs('plain')}
    const VITE_READY = ${rhs('VITE_READY')}
    const SIGNAL_READY = ${rhs('SIGNAL_READY')}
    return { plain, VITE_READY, SIGNAL_READY }
  `) as (v: number, s: number) => Matchers
  return build(vitePort, signalPort)
}

describe.each(HARNESSES)('%s', (rel) => {
  const src = readFileSync(join(ROOT, rel), 'utf8')

  it('lets the signaling port be moved, like the dev server port', () => {
    // The guard refuses an occupied port. Without a hatch, "occupied" means
    // "unrunnable" rather than "run it somewhere else".
    expect(src, 'SIGNAL_PORT has no environment escape hatch')
      .toMatch(/const SIGNAL_PORT = Number\(process\.env\.SIGNAL_PORT \?\? \d+\)/)
    expect(src, 'VITE_PORT lost its escape hatch')
      .toMatch(/const VITE_PORT = Number\(process\.env\.VITE_PORT \?\? \d+\)/)
  })

  it("does not default to 8787, which is wrangler's own default port", () => {
    const m = /const SIGNAL_PORT = Number\(process\.env\.SIGNAL_PORT \?\? (\d+)\)/.exec(src)
    expect(m, 'no SIGNAL_PORT default to read').not.toBeNull()
    // Any `wrangler dev` started without --port claims 8787. A harness whose
    // default IS 8787 is a harness this repo's own worker can take out.
    expect(Number(m![1]), 'the default is the one port most likely to be taken already')
      .not.toBe(8787)
  })

  it("waits on an ANSI-stripped view of the child's log", () => {
    expect(src, 'waitServer still matches the raw log, so forced colour breaks it')
      .toMatch(/ready\.test\(plain\(proc\.log\)\)/)
  })

  it("sees vite's ready line whether or not colour is forced", () => {
    const colour = matchersOf(rel, VITE_COLOUR_PORT, 8791)
    expect(
      colour.VITE_READY.test(colour.plain(VITE_COLOUR)),
      "FORCE_COLOR=1 hides vite's ready line from the harness — a 30 s timeout "
      + 'on a server that was serving in 125 ms',
    ).toBe(true)
    const bare = matchersOf(rel, VITE_PLAIN_PORT, 8791)
    expect(bare.VITE_READY.test(bare.plain(VITE_PLAIN)), 'the plain line stopped matching').toBe(true)
  })

  it("sees wrangler's ready line, so the relay is waited for too", () => {
    // Before this, the relay was waited for with `ready = null`: "the port
    // answers", which is the exact check requirePortFree exists to distrust.
    expect(src, 'the relay is still waited for by nothing but an HTTP response')
      .toMatch(/waitServer\(signal, [^\n]*SIGNAL_READY\)/)
    const colour = matchersOf(rel, 5194, WRANGLER_COLOUR_PORT)
    expect(colour.SIGNAL_READY.test(colour.plain(WRANGLER_COLOUR)), 'coloured wrangler line missed').toBe(true)
    const bare = matchersOf(rel, 5194, WRANGLER_PLAIN_PORT)
    expect(bare.SIGNAL_READY.test(bare.plain(WRANGLER_PLAIN)), 'plain wrangler line missed').toBe(true)
  })

  it('still refuses a server announcing a DIFFERENT port', () => {
    // The reason these regexes carry the port at all. A relaxed match would
    // adopt the neighbouring server the guard had just refused.
    const other = matchersOf(rel, VITE_COLOUR_PORT + 1, WRANGLER_PLAIN_PORT + 1)
    expect(other.VITE_READY.test(other.plain(VITE_COLOUR)), "matched another vite's port").toBe(false)
    expect(other.SIGNAL_READY.test(other.plain(WRANGLER_PLAIN)), "matched another relay's port").toBe(false)
  })
})
