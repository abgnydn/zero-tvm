#!/usr/bin/env node
/**
 * PRE-PUSH GATE — a fix must arrive with the test that would have caught it.
 *
 * Every bug this repo has shipped was silent. The Phi-3 ropeFreqs P0 ran broken
 * for ten days through two deploys. The int8 pack phase wrote half of every
 * head-dim-256 row and produced fluent WRONG text rather than noise, and the
 * kernel suite that covered it tested head-dim 128 — so re-introducing it would
 * still have passed. A GDN rewind restored a dead conversation's state in 2,929
 * of 20,000 simulated conversations while six live turns looked perfect.
 *
 * None of those were caught by reasoning about the diff. They were caught by
 * something that ran. So: a commit that says it fixes something must also touch
 * a test.
 *
 * The check is deliberately crude — it cannot know whether the test is the
 * RIGHT one. It exists to make "I'll add the test later" a decision you have to
 * make out loud rather than one you drift into.
 *
 * Escape, when the fix genuinely cannot be tested here (a WGSL numeric fix
 * needs a GPU this sandbox does not have; a deploy-config change has no unit):
 *
 *     SKIP_TEST_GATE=1 git push
 *
 * Say why in the commit message when you use it.
 */
import { execSync } from 'node:child_process'

if (process.env.SKIP_TEST_GATE === '1') {
  console.log('[gate] SKIP_TEST_GATE=1 — regression-test gate bypassed on purpose')
  process.exit(0)
}

const sh = (cmd) => execSync(cmd, { encoding: 'utf8' }).trim()

// What is actually about to be pushed. On a branch with no upstream yet,
// fall back to the last commit rather than scanning all of history.
// An explicit range may be passed to exercise the gate against known history —
// a gate nobody has seen fire is not known to work.
let range = process.argv[2]
if (!range) {
  try {
    const upstream = sh('git rev-parse --abbrev-ref --symbolic-full-name @{u}')
    range = `${upstream}..HEAD`
  } catch {
    range = 'HEAD~1..HEAD'
  }
}

let commits
try {
  commits = sh(`git log --format=%H ${range}`).split('\n').filter(Boolean)
} catch {
  process.exit(0)   // nothing to compare against; never block on a git edge case
}
if (commits.length === 0) process.exit(0)

// Words that claim a defect was repaired. "fixes #12" and "prefix" would both
// trip a naive substring match, so this looks for them as words.
const CLAIMS_FIX = /\b(fix|fixes|fixed|bug|regression|broke|broken|wrong|corrupt\w*|silent\w*)\b/i
const IS_TEST = /(^|\/)(tests?|__tests__)\//

const offenders = []
for (const sha of commits) {
  // SUBJECT only. Bodies here explain bugs at length — the commit that ADDS a
  // diagnostic describes the defect it diagnoses, the commit that records a
  // measurement quotes what was wrong before. Matching those fired on 9 of 18
  // real commits and would have been ignored within a week. The subject is
  // where a commit says what it IS, and it fires on 2 of the same 18: both of
  // them genuine fixes that shipped without a test.
  const subject = sh(`git log -1 --format=%s ${sha}`)
  if (!CLAIMS_FIX.test(subject)) continue
  const files = sh(`git show --name-only --format= ${sha}`).split('\n').filter(Boolean)
  // Docs-only commits describe fixes without being one.
  const touchesCode = files.some((f) => /^(src|scripts|workers)\//.test(f))
  if (!touchesCode) continue
  if (files.some((f) => IS_TEST.test(f))) continue
  offenders.push({ sha: sha.slice(0, 8), subject })
}

if (offenders.length === 0) process.exit(0)

console.error('\n[gate] These commits claim to fix something and touch code, but add no test:\n')
for (const o of offenders) console.error(`  ${o.sha}  ${o.subject}`)
console.error(`
A fix without a test is a fix that comes back. The test that matters is the one
that FAILS on the old code — write it, run it against the bug, then fix.

If it genuinely cannot be tested here (needs a GPU, is a config or deploy
change), push with:

    SKIP_TEST_GATE=1 git push

and say why in the commit message.
`)
process.exit(1)
