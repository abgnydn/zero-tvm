// Exported, tested, and called by nothing.
//
// renderToolResults wrapped tool results in <tool_response> markers correctly,
// had unit tests, and had ZERO CALLERS for weeks while both hosts hand-rolled
// the wrong thing. Its tests were green the entire time. A green test on an
// uncalled function is worse than no test: it reads as coverage.
//
// So this walks the engine's own source and asserts that every exported symbol
// in the paths where that happened is referenced from somewhere that is not a
// test. It cannot prove a function is CORRECTLY called — host-normalize.test.ts
// and chatml-generations.test.ts do that — only that something calls it.
//
// Deliberately scoped to the modules where a silent no-op is dangerous. A
// broader sweep would drown in legitimately-unused library surface and get
// muted, which is how a check stops working.

import { describe, expect, it } from 'vitest'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { join } from 'node:path'

const ROOT = join(import.meta.dirname, '../..')

/** Where an uncalled export means the engine is quietly doing the wrong thing.
 *  Not the whole tree: src/lib/index.ts is a published API surface whose job is
 *  to export things this repo does not itself call. */
const WATCHED = [
  'src/zero-tvm/tool-calls.ts',
  'src/zero-tvm/tokenizer-bpe.ts',
  'src/zero-tvm/prefix-reuse.ts',
  'src/zero-tvm/model-registry.ts',
  'src/compiler/constraints.ts',
]

/** Callers: anything that is not a test and not the file itself. HTML entries
 *  count — several modules are only reached from a page. */
/** Dead exports kept ON PURPOSE, each with the reason. An allowlist entry is a
 *  recorded decision; a check that gets muted wholesale is not. Anything here
 *  should be re-read when the file it lives in is next touched. */
const KEPT: Record<string, string> = {
  QWEN3_DEFAULT_SYSTEM_MESSAGE:
    "MLC's conv_template 'qwen3' carries this default persona and the HF template does not, "
    + 'so our prompts deliberately omit it — render-diff.py confirms we match the HF template '
    + 'byte-for-byte. The constant records the difference, which is worth keeping even though '
    + 'nothing injects it. Delete it only together with that note.',
}

const SEARCH_DIRS = ['src', 'scripts', 'workers']
const SEARCH_FILES = ['index.html', 'zero-tvm.html', 'docs.html', 'validate.html', 'share.html', 'agent-host.html']

function walk(dir: string, out: string[] = []): string[] {
  let entries: string[]
  try { entries = readdirSync(join(ROOT, dir)) } catch { return out }
  for (const e of entries) {
    const rel = `${dir}/${e}`
    const st = statSync(join(ROOT, rel))
    if (st.isDirectory()) { if (e !== 'node_modules') walk(rel, out) }
    else if (/\.(ts|mjs|js|html)$/.test(e)) out.push(rel)
  }
  return out
}

const corpus = [
  ...SEARCH_DIRS.flatMap((d) => walk(d)),
  ...SEARCH_FILES,
].map((rel) => {
  try { return { rel, text: readFileSync(join(ROOT, rel), 'utf8') } } catch { return null }
}).filter(Boolean) as Array<{ rel: string; text: string }>

/** `export function name(` and `export const name =` — the two forms this repo
 *  uses. Types and interfaces are excluded: an unused type is harmless. */
function exportsOf(text: string): string[] {
  const names: string[] = []
  for (const m of text.matchAll(/^export\s+(?:async\s+)?function\s+([A-Za-z_$][\w$]*)/gm)) names.push(m[1])
  for (const m of text.matchAll(/^export\s+const\s+([A-Za-z_$][\w$]*)\s*[:=]/gm)) names.push(m[1])
  return names
}

describe('no orphan exports in the paths where silence is dangerous', () => {
  for (const file of WATCHED) {
    const text = readFileSync(join(ROOT, file), 'utf8')
    const names = exportsOf(text)

    it(`${file} — ${names.length} exports, none dead`, () => {
      // TWO categories, and only one is a bug.
      //
      //   dead          referenced nowhere at all, not even in its own file. A
      //                 function that runs for nobody. renderToolResults was
      //                 this, with passing tests, while both hosts hand-rolled
      //                 the wrong behaviour.
      //   internal-only used inside its own module but exported anyway. Not a
      //                 defect — the export is just wider than it needs to be —
      //                 so it is reported and does not fail. Making it fail
      //                 would train people to mute this file.
      const dead: string[] = []
      const internalOnly: string[] = []
      for (const name of names) {
        // Word-boundary match, deliberately loose. A false NEGATIVE (we believe
        // something is called when it barely is) is far cheaper here than a
        // false positive, which gets the whole check switched off.
        const re = new RegExp(`\\b${name}\\b`)
        const elsewhere = corpus.some((f) => f.rel !== file && re.test(f.text))
        if (elsewhere) continue
        // Its own file, excluding the line that defines it and any comment.
        const ownUses = text.split('\n')
          .filter((l) => !l.trimStart().startsWith('//') && !l.trimStart().startsWith('*'))
          .filter((l) => !/^export\s/.test(l.trimStart()))
          .filter((l) => re.test(l)).length
        if (KEPT[name]) continue
        ;(ownUses > 0 ? internalOnly : dead).push(name)
      }
      if (internalOnly.length) {
        console.log(`      ${file}: exported but used only internally — ${internalOnly.join(', ')}`)
      }
      expect(dead, `DEAD exports in ${file}: ${dead.join(', ')}. `
        + 'Referenced nowhere, including its own file. An uncalled export with a '
        + 'green test reads as coverage and is not — renderToolResults was exactly '
        + 'that for weeks while both hosts hand-rolled the wrong behaviour. '
        + 'Wire it up or delete it.')
        .toEqual([])
    })
  }

  it('every allowlisted export is still actually dead', () => {
    // An allowlist that outlives its reason is how a check rots. If one of these
    // gained a caller, the entry is stale and should go.
    for (const [name, why] of Object.entries(KEPT)) {
      const re = new RegExp(`\\b${name}\\b`)
      const users = corpus.filter((f) => re.test(f.text) && !/^src\/zero-tvm\/tokenizer-bpe\.ts$/.test(f.rel))
      expect(users.map((u) => u.rel), `${name} is allowlisted as intentionally-unused (${why.slice(0, 60)}…) `
        + 'but now has callers — drop the allowlist entry').toEqual([])
    }
  })

  it('the check can actually see an orphan', () => {
    // Guards the guard: if the export regex stops matching, every file above
    // trivially passes with zero exports found and nothing says so.
    const sample = 'export function definitelyNotCalledAnywhere() {}\nexport const alsoNot = 1\n'
    expect(exportsOf(sample)).toEqual(['definitelyNotCalledAnywhere', 'alsoNot'])
    const re = new RegExp('\\bdefinitelyNotCalledAnywhere\\b')
    expect(corpus.some((f) => re.test(f.text))).toBe(false)
  })

  it('finds a real export in every watched file', () => {
    // If a file's exports parse to zero, its orphan test is vacuous.
    for (const file of WATCHED) {
      const n = exportsOf(readFileSync(join(ROOT, file), 'utf8')).length
      expect(n, `${file} parsed to ZERO exports — the regex missed this file's style`).toBeGreaterThan(0)
    }
  })
})
