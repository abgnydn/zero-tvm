// The mutation gate's own config, checked. It is the one gate nothing gated.
//
// `expect` names the test file that must be among the failures — not merely
// THAT the suite went red. Without it the comparison is
// `!run.failed.includes(undefined)`, which is always true, so the entry reports
// WRONG CHECK and the gate exits 1 with a message about a test named
// "undefined".
//
// That is not hypothetical. It shipped, as a semantic merge conflict git could
// not see: one branch added the chunk-quarantine mutation, another added the
// `expect` field and filled it in for the nine entries that existed on its own
// side. Neither could know about the other's line. Both were green alone, the
// textual merge was clean, and main went red on the merge — on the very job the
// second branch had added.
//
// So this asserts the property that merge violated: every mutation names a test
// file, and that file exists. It fails on the merge commit and passes on the
// fix.
//
// Read as TEXT, never imported. scripts/mutation-gate.mjs runs at module scope
// and ends in `process.exit` — importing it from a test would rewrite src/ in
// place mid-suite and kill the runner.

import { describe, expect, it } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '../..')
const GATE = 'scripts/mutation-gate.mjs'
const SRC = readFileSync(join(ROOT, GATE), 'utf8')

/** The MUTATIONS array as source text, split into one string per entry. */
function entries(): { id: string; body: string }[] {
  const open = SRC.indexOf('const MUTATIONS = [')
  if (open < 0) throw new Error(`${GATE}: no MUTATIONS array — this test cannot see its subject`)
  const close = SRC.indexOf('\n]', open)
  if (close < 0) throw new Error(`${GATE}: MUTATIONS array is not closed at column 0`)
  const block = SRC.slice(open, close)
  return block
    .split(/\n {2}\{\n/)
    .slice(1)
    .map((body) => ({ id: (body.match(/id: '([^']+)'/) ?? [])[1] ?? '(unnamed)', body }))
}

describe('the mutation gate config', () => {
  const all = entries()

  it('parses a non-empty MUTATIONS array', () => {
    // Guards the guard: a parse that silently yields nothing would make every
    // it.each below vacuous, which is the failure this whole file is about.
    expect(all.length, `parsed ${all.length} mutations out of ${GATE}`).toBeGreaterThanOrEqual(8)
  })

  it('gives every mutation a unique id', () => {
    const ids = all.map((m) => m.id)
    expect(new Set(ids).size, `duplicate mutation ids: ${ids.join(', ')}`).toBe(ids.length)
  })

  it.each(all.map((m) => [m.id, m.body] as const))('%s names the test that must fail', (id, body) => {
    const named = (body.match(/expect: '([^']+)'/) ?? [])[1]
    expect(named, `mutation '${id}' has no \`expect\`. The gate compares with `
      + '`!run.failed.includes(m.expect)`, and includes(undefined) is always false, so this '
      + 'entry reports WRONG CHECK and main goes red. Name the test file whose own name '
      + 'claims this defect — derive it by running the gate and reading which file failed.')
      .toBeTruthy()
  })

  it.each(all.map((m) => [m.id, m.body] as const))('%s names a test file that exists', (id, body) => {
    const named = (body.match(/expect: '([^']+)'/) ?? [])[1]
    if (!named) return       // the case above owns that failure; do not report it twice
    const hits = ['tests/unit', 'tests/tokenizer', 'tests/kernels']
      .map((d) => join(ROOT, d, named))
      .filter((p) => existsSync(p))
    expect(hits.length, `mutation '${id}' expects '${named}', which exists in none of `
      + 'tests/unit, tests/tokenizer or tests/kernels. A gate cannot be caught by a file '
      + 'that is not there.').toBeGreaterThan(0)
  })
})
