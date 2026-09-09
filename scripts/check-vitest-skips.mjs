#!/usr/bin/env node
// CHECK-VITEST-SKIPS — every skipped test must carry a documented reason.
//
// vitest exits 0 when tests skip (describe.skipIf, test.skip, .todo), so a
// suite that runs nothing reports green — the same failure mode the mutation
// gate exists for, one level up. This reads a --reporter=json report and
// fails on any skipped/pending/todo test NOT named in
// scripts/ci-skip-allowlist.json. Each allowlist entry names the capability
// the test needs and where to get it; an entry without a reason is rejected
// by this script's own shape check.
//
//   npx vitest run tests/e2e/gate.test.ts --reporter=json --outputFile=/tmp/r.json
//   node scripts/check-vitest-skips.mjs /tmp/r.json
//
// The allowlist starts empty: on software WebGPU (lavapipe) every collected
// e2e test is expected to RUN. Mirror-gated spec suites (qwen*, llama32)
// are not collected by the CI job at all — absence from the report is not
// a skip, and this script only judges what ran or skipped.

import { readFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const reportPath = process.argv[2]
if (!reportPath) {
  console.error('usage: check-vitest-skips.mjs <vitest-json-report>')
  process.exit(2)
}

let report
try {
  report = JSON.parse(readFileSync(reportPath, 'utf8'))
} catch (e) {
  console.error(`cannot read report ${reportPath}: ${e.message}`)
  process.exit(2)
}

const allowlist = JSON.parse(
  readFileSync(join(ROOT, 'scripts/ci-skip-allowlist.json'), 'utf8'))
if (!Array.isArray(allowlist)) {
  console.error('ci-skip-allowlist.json must be a JSON array')
  process.exit(2)
}
for (const [i, e] of allowlist.entries()) {
  if (typeof e?.pattern !== 'string' || typeof e?.reason !== 'string'
    || !e.pattern || !e.reason) {
    console.error(`ci-skip-allowlist.json[${i}]: each entry needs a non-empty {pattern, reason}`)
    process.exit(2)
  }
}

const SKIPPED = new Set(['skipped', 'pending', 'todo'])
let unexpected = 0
let files = 0
for (const fileResult of report.testResults ?? []) {
  files++
  const file = basename(fileResult.name ?? '?')
  for (const a of fileResult.assertionResults ?? []) {
    if (!SKIPPED.has(a.status)) continue
    const id = `${file} :: ${[...(a.ancestorTitles ?? []), a.title].join(' > ')}`
    const allowed = allowlist.find((e) => id.includes(e.pattern))
    if (allowed) {
      console.log(`  SKIP-ALLOWED  ${id}\n                ${allowed.reason}`)
    } else {
      console.log(`  SKIP-UNEXPECTED  ${id}`)
      unexpected++
    }
  }
}

console.log(`\nchecked ${files} files: `
  + (unexpected ? `${unexpected} UNEXPECTED skip(s) — allowlist them with a reason or fix the run` : 'no unexpected skips'))
process.exit(unexpected ? 1 : 0)
