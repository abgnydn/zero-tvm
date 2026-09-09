#!/usr/bin/env node
// RELEASE CHECK — everything that must be green before a deploy, in one place.
//
// The gates in this repo were real and scattered: unit tests in CI, kernel
// numerics in a human's shell, quality A/Bs run on demand, the agentic and
// needle evals invented mid-investigation and never wired to anything. Nothing
// forced any of them before a publish, so "did we run the GPU suites?" was
// answered from memory.
//
// This runs what CAN run here and NAMES what cannot, with the command. It does
// not pretend a suite passed because it was skipped: a gate that needs hardware
// this process does not have is reported UNRUN, and UNRUN is not a pass.
//
//   node scripts/release-check.mjs            # everything runnable
//   node scripts/release-check.mjs --list     # just show the checklist
import { spawnSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const LIST_ONLY = process.argv.includes('--list')

/** `needs`: what the check requires that this process may not have.
 *  'none'  runnable anywhere — run it and gate on it.
 *  'gpu'   needs a real adapter. dawn.node returns a null adapter in a sandbox
 *          and mlx has no Metal device, so these are the human's to run.
 *  'model' needs a checkpoint on disk as well as a GPU. */
const CHECKS = [
  { name: 'typecheck', needs: 'none', cmd: ['npm', 'run', 'typecheck'] },
  { name: 'unit + tokenizer', needs: 'none', cmd: ['npm', 'run', 'test:unit'] },
  { name: 'build (site)', needs: 'none', cmd: ['npm', 'run', 'build'] },
  { name: 'build (lib)', needs: 'none', cmd: ['npm', 'run', 'build:lib'] },
  { name: 'facts registry', needs: 'none', cmd: ['node', 'bin/check-facts.mjs'],
    cwd: join(ROOT, '..', 'sites-shared'),
    why: 'no published number may be unregistered or drifted' },

  { name: 'kernel numerics (qwen35)', needs: 'gpu', cmd: ['npm', 'run', 'test:kernels:qwen35'],
    why: 'the hybrid GDN + int8 pack at head-dim 256' },
  { name: 'kernel numerics (mlx loader)', needs: 'gpu', cmd: ['npm', 'run', 'test:kernels:mlx'] },
  { name: 'chunked prefill identity', needs: 'model', cmd: ['node', 'scripts/chunk-prefill-test.mjs', 'qwen35'],
    why: 'chunking changes arithmetic ORDER; token identity is the only honest check' },
  { name: 'e2e (browser)', needs: 'gpu', cmd: ['npm', 'run', 'test:e2e'] },

  // The three below are what this batch learned to run. They are evals, not
  // unit tests: they catch "fluent but wrong", which is the failure mode every
  // defect in this repo has had.
  { name: 'render vs vendor template', needs: 'model',
    cmd: ['uv', 'run', 'python', join(ROOT, 'scripts/render-diff.py'), '--help'],
    why: 'our prompt must equal the checkpoint\'s own jinja — run --shapes and --depth per model' },
  { name: 'agentic loop', needs: 'model', cmd: ['node', 'scripts/agentic-eval.mjs', '<param>'],
    why: 'short AND PAD=24000 — a model correct at 600 tokens can be broken at 16k' },
  { name: 'long-context retrieval', needs: 'model', cmd: ['node', 'scripts/needle-test.mjs', '<param>'],
    why: 'can it still see the start of its context from the end' },
  { name: 'fidelity at depth', needs: 'model',
    cmd: ['uv', 'run', 'python', 'scripts/mlx-ref.py', '--depth', '16000', '...'],
    why: 'THE gap that let qwen38 ship broken — validate-model alone checks ~20 tokens' },
]

const pad = (s, n) => s.padEnd(n)
const W = Math.max(...CHECKS.map((c) => c.name.length))

if (LIST_ONLY) {
  for (const c of CHECKS) console.log(`  ${pad(c.name, W)}  ${c.needs === 'none' ? 'here' : c.needs.toUpperCase()}  ${c.why ?? ''}`)
  process.exit(0)
}

const results = []
for (const c of CHECKS) {
  if (c.needs !== 'none') { results.push({ ...c, status: 'UNRUN' }); continue }
  if (c.cwd && !existsSync(c.cwd)) {
    // Printed AND counted. It used to be pushed and then referenced nowhere:
    // not in the loop's output, not in the summary's three tallies, not in the
    // failure list. The row simply vanished and the run still exited 0, so the
    // checklist under-reported its own length — which is how a release note came
    // to claim twelve rows for a thirteen-row list.
    console.log(`  ${pad(c.name, W)}  MISSING — ${c.cwd} is not there`)
    results.push({ ...c, status: 'MISSING' }); continue
  }
  process.stdout.write(`  ${pad(c.name, W)}  … `)
  const r = spawnSync(c.cmd[0], c.cmd.slice(1), { cwd: c.cwd ?? ROOT, encoding: 'utf8' })
  const ok = r.status === 0
  results.push({ ...c, status: ok ? 'PASS' : 'FAIL', out: (r.stdout ?? '') + (r.stderr ?? '') })
  console.log(ok ? 'PASS' : 'FAIL')
}

console.log('')
const failed = results.filter((r) => r.status === 'FAIL')
for (const f of failed) {
  console.log(`── ${f.name} ──`)
  console.log(f.out.split('\n').slice(-14).join('\n'))
}

const unrun = results.filter((r) => r.status === 'UNRUN')
const missing = results.filter((r) => r.status === 'MISSING')
const tally = `${results.filter((r) => r.status === 'PASS').length} passed · ${failed.length} failed`
  + ` · ${unrun.length} NOT RUN HERE${missing.length ? ` · ${missing.length} MISSING` : ''}`
console.log(`\n${tally}  (of ${CHECKS.length})\n`)
if (missing.length) {
  console.log('  These could not run at all — a path they need is absent. Not passes:')
  for (const m of missing) console.log(`    ${pad(m.name, W)}  needs ${m.cwd}`)
  console.log('')
}
if (unrun.length) {
  console.log('  These need a GPU and a checkpoint. They are NOT passes — run them:')
  for (const u of unrun) console.log(`    ${pad(u.name, W)}  ${u.cmd.join(' ')}`)
  console.log('')
}
// Exit non-zero on a real failure only. Unrun checks are reported loudly but do
// not fail the run, because the whole point is that a human finishes the list —
// exiting 1 here would train people to ignore the exit code.
process.exit(failed.length ? 1 : 0)
