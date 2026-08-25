// Every status the release checklist can produce must be reported.
//
// `release-check.mjs` builds a `results` array, tagging each check PASS, FAIL,
// UNRUN or MISSING. MISSING — set when a check's `cwd` does not exist — was
// pushed into that array and then referenced nowhere: not printed in the loop,
// not in the summary's tallies, not in the failure list. The row vanished, the
// run still exited 0, and the list under-reported its own length. That is how a
// release note came to claim twelve rows for a thirteen-row checklist: the
// number was read off a run where `sites-shared` was absent and `facts registry`
// had been swallowed.
//
// A checklist that quietly drops a row is the failure this whole gate exists to
// prevent, so this asserts the property directly: every status literal the
// script can assign must also appear in the reporting half of the file.
//
// Source text, not an import: release-check.mjs runs at module scope, spawns
// the whole suite, and ends in process.exit.

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '../..')
const REL = 'scripts/release-check.mjs'
const SRC = readFileSync(join(ROOT, REL), 'utf8')

/** The file splits at the summary: everything before is assignment, after is
 *  reporting. `console.log('')` right after the loop is the seam. */
const SEAM = SRC.indexOf("const failed = results.filter")

describe('release-check accounting', () => {
  it('has a reporting half to check', () => {
    expect(SEAM, `${REL}: could not locate the summary section`).toBeGreaterThan(0)
  })

  // Not `status: 'X'` — PASS and FAIL are assigned through a ternary
  // (`status: ok ? 'PASS' : 'FAIL'`), which that shape misses. A regex that
  // silently saw only two of the four statuses would be the same class of bug
  // as the one under test, so this collects every all-caps string literal in
  // the assignment half instead.
  const assigned = [...new Set(
    [...SRC.slice(0, SEAM).matchAll(/'([A-Z]{3,})'/g)].map((m) => m[1]),
  )]

  it('assigns more than one status', () => {
    // Guards the guard: a regex that matched nothing would make the cases below
    // vacuous, which is the exact shape of the bug this file is about.
    expect(assigned.length, `parsed statuses: ${assigned.join(', ')}`).toBeGreaterThanOrEqual(4)
  })

  it.each(assigned)('%s is reported, not just recorded', (status) => {
    const reporting = SRC.slice(SEAM)
    expect(reporting.includes(status), `${REL} assigns status '${status}' but never mentions it `
      + 'again after the loop. A check with that status is counted in no tally and printed in '
      + 'no list — it disappears, and the run still exits 0. Print it and count it.')
      .toBe(true)
  })

  it('states the total so a dropped row is visible', () => {
    // The tally lines are the only place a reader can notice the list is short.
    expect(SRC.slice(SEAM), `${REL}'s summary should name CHECKS.length, so a swallowed row `
      + 'shows up as a total that does not add up').toContain('CHECKS.length')
  })
})
