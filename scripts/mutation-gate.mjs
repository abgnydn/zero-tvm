#!/usr/bin/env node
// MUTATION GATE — does the suite actually catch the bugs it was written for?
//
// CLAUDE.md's rule is "the test that matters is the one that FAILS on the old
// code — write it, run it against the bug, watch it fail, then fix." That was
// done by hand for every fix in this repo, which means it was done at the
// moment the test was written and never again. A later refactor can quietly
// remove the teeth from a test that still passes, and nothing notices: a green
// suite looks identical whether it is checking something or not.
//
// So each entry below re-introduces a bug that actually shipped here, and
// asserts the suite goes RED. A mutation that passes is a hole — the test for
// it is decorative.
//
// This is the cheap half of the idea. It cannot catch what no test covers; it
// only proves the tests that exist still bite.
//
//   node scripts/mutation-gate.mjs            # all
//   node scripts/mutation-gate.mjs --list
//   node scripts/mutation-gate.mjs think-block
import { readFileSync, writeFileSync, rmSync, statSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { basename, dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const arg = process.argv[2]

// Mutations rewrite src/ IN PLACE, so two concurrent runs corrupt each other
// and each sees a false red baseline. Serialize on a lockfile via flock(1):
// re-exec this same script under `flock LOCKFILE`, which is held for the whole
// run. `--locked` only ever comes from the re-exec below (it starts with --,
// so the `chosen` filter ignores it). No flock or no .git dir: warn and run
// unlocked rather than refuse — the gate must stay runnable by hand.
if (!process.argv.includes('--locked')) {
  const underGit = (() => {
    try { return statSync(join(ROOT, '.git')).isDirectory() } catch { return false }
  })()
  const flockProbe = spawnSync('flock', ['--version'], { encoding: 'utf8' })
  if (flockProbe.error || !underGit) {
    console.log('  WARNING: no flock lock (flock(1) missing or no .git dir) — do not run two gates at once')
  } else {
    const inner = spawnSync('flock',
      [join(ROOT, '.git', 'mutation-gate.lock'), process.execPath, ...process.argv.slice(1), '--locked'],
      { cwd: ROOT, stdio: 'inherit' })
    process.exit(inner.status ?? 1)
  }
}

/** Each mutation names the DEFECT it reinstates, not the edit it makes. `find`
 *  must match exactly once — a mutation that silently matches nothing is a test
 *  that silently passes, which is the failure this file exists to prevent. */
// `expect` is the test file that must be among the failures — not merely THAT the
// suite went red. A mutation caught only by a downstream test is a hole that
// reports green: the check whose name says it covers that defect never fired,
// and deleting that check would go unnoticed. Each value here was DERIVED by
// running this gate and reading which files failed, never guessed.
const MUTATIONS = [
  {
    id: 'chunk-quarantine',
    expect: 'chunk-cap-quarantine.test.ts',
    defect: "the per-spec chunk-cap ceiling ignored, so qwen38 defaults to 1024 again",
    shipped: 'invented tool names at ~16k, correct at cap 256 and per-token',
    file: 'src/zero-tvm/engine-core.ts',
    find: '  return Math.min(sgmatAvail ? 1024 : 64, spec.maxChunkCap ?? Infinity)',
    replace: '  return sgmatAvail ? 1024 : 64',
  },
  {
    id: 'think-block',
    expect: 'chatml-generations.test.ts',
    defect: 'the empty <think> block back on EVERY past assistant turn',
    shipped: 'rendered a prompt no Qwen template produces; 286 spurious blocks at 24k',
    file: 'src/zero-tvm/tokenizer-bpe.ts',
    find: "    const think = gen === 'qwen38'",
    replace: "    const think = msg.role === 'assistant' || gen === 'qwen38'",
  },
  {
    id: 'tool-response',
    expect: 'chatml-generations.test.ts',
    defect: 'tool results sent as bare user turns, unwrapped',
    shipped: 'the model read every tool result as something the user typed',
    file: 'src/zero-tvm/tool-calls.ts',
    // Mutates the WRAPPING, which is the defect the id names. The previous
    // version disabled the chatml-xml `.trim()` on the line above and left
    // `<tool_response>` intact — so it reinstated a different, smaller bug than
    // its own `defect` string described, and the entry read as coverage for
    // something it never tested.
    find: "    content: body.map((r) => `<tool_response>\\n${r}\\n</tool_response>`).join('\\n'),",
    replace: "    content: body.join('\\n'),",
  },
  {
    id: 'tool-dialect',
    expect: 'tool-calls.test.ts',
    defect: 'the Qwen3.5+ dialect resolved to the Qwen3-era JSON form',
    shipped: 'both Qwen3.5 builds were served a tools block they cannot follow',
    file: 'src/zero-tvm/tool-calls.ts',
    find: "  if (chatTemplateId === 'chatml-q35' || chatTemplateId === 'chatml-q38') return 'chatml-xml'",
    replace: "  if (chatTemplateId === 'chatml-q38') return 'chatml-xml'",
  },
  {
    id: 'template-detect',
    expect: 'chat-template-detect.test.ts',
    defect: 'detectChatTemplate collapsing the three ChatML generations to one',
    shipped: 'add-model generated Qwen3.8 with the wrong template id',
    file: 'src/compiler/constraints.ts',
    find: "  if (t.includes('preserve_thinking is undefined')) return 'chatml-q38'",
    replace: "  if (false) return 'chatml-q38'",
  },
  {
    id: 'trim',
    expect: 'chatml-generations.test.ts',
    defect: 'no |trim on message content for the generations that require it',
    shipped: 'whitespace at a turn boundary differed from the vendor renderer',
    file: 'src/zero-tvm/tokenizer-bpe.ts',
    find: "  const trims = gen !== 'qwen3'",
    replace: '  const trims = false',
  },
  {
    id: 'kv-scales',
    expect: 'kv-figure.test.ts',
    defect: 'the int8 KV figure dropping the per-attention-layer scales',
    shipped: 'the entrance understated what it allocates, in the reassuring direction',
    file: 'src/zero-tvm/model-registry.ts',
    find: '      + 4 * spec.kvHeads * spec.layerKinds.filter((k) => k === \'attn\').length',
    replace: '      + 4 * spec.kvHeads',
  },
  {
    id: 'quant-label',
    expect: 'quant-label.test.ts',
    defect: 'the 3-bit build labelled the same as the 4-bit one',
    shipped: 'the sheet showed no quantisation at all for most of the roster',
    file: 'src/zero-tvm/model-registry.ts',
    find: "  return spec.moe?.bits === 3 ? `3-bit experts, 4-bit elsewhere · ${fmt}` : `4-bit · ${fmt}`",
    replace: '  return `4-bit · ${fmt}`',
  },
  {
    id: 'kv-flag-pairing',
    expect: 'kv-alloc-matches-flags.test.ts',
    defect: 'the engine ignoring the ?kv8 variant flag and defaulting to f16',
    shipped: 'the agent host, both room paths and validate ran f16 while their flags said int8',
    file: 'src/zero-tvm/engine-core.ts',
    find: '  return variants.int8KV ?? false',
    replace: '  return false',
  },
  {
    id: 'station-flags',
    expect: 'engine-args.test.ts',
    defect: 'the station passing expert slots as the KV disk-pool flag',
    shipped: 'the default build silently turned off restart-survival',
    file: 'scripts/native/engine-args.ts',
    find: "  if (pool) argv.push('--experts', String(pool))",
    replace: "  argv.push('--pool', String(pool || 0))",
  },
]

if (process.argv.includes('--list')) {
  for (const m of MUTATIONS) console.log(`  ${m.id.padEnd(16)} ${m.defect}`)
  process.exit(0)
}

const chosen = arg && !arg.startsWith('--') ? MUTATIONS.filter((m) => m.id === arg) : MUTATIONS
if (!chosen.length) { console.error(`no mutation "${arg}" — try --list`); process.exit(2) }

/** Run the unit suite and report BOTH whether it went red and WHICH test files
 *  did. "The suite went red somewhere" is a weaker claim than it looks: a
 *  mutation caught only by a downstream test is a hole that reports green,
 *  because the check whose name says it covers that defect never fired. */
const REPORT = join(ROOT, '.mutation-gate-report.json')
function suite() {
  rmSync(REPORT, { force: true })
  const r = spawnSync('npm', ['run', 'test:unit', '--', '--reporter=json', `--outputFile=${REPORT}`],
    { cwd: ROOT, encoding: 'utf8' })
  let failed = null   // null = the report could not be read, NOT "nothing failed"
  try {
    const j = JSON.parse(readFileSync(REPORT, 'utf8'))
    failed = [...new Set((j.testResults ?? [])
      .filter((t) => t.status === 'failed')
      .map((t) => basename(t.name)))].sort()
  } catch { /* leave null — the caller must not read it as an empty set */ }
  rmSync(REPORT, { force: true })
  return { green: r.status === 0, failed }
}
const statusOf = (files) =>
  spawnSync('git', ['status', '--porcelain', '--', ...files], { cwd: ROOT, encoding: 'utf8' }).stdout.trim()

const touched = [...new Set(chosen.map((m) => m.file))]
const dirtBefore = statusOf(touched)

process.stdout.write('  baseline (unmutated) … ')
if (!suite().green) {
  console.log('FAILING')
  console.error('\n  The suite is red BEFORE any mutation. Fix that first — nothing below means anything.')
  process.exit(2)
}
console.log('green')

let holes = 0
for (const m of chosen) {
  const path = join(ROOT, m.file)
  const original = readFileSync(path, 'utf8')
  const n = original.split(m.find).length - 1
  if (n !== 1) {
    console.log(`  ${m.id.padEnd(16)} STALE — its anchor matches ${n} times, not once`)
    console.log(`      ${m.file}: the code moved and this mutation no longer reinstates ${m.defect}`)
    holes++
    continue
  }
  writeFileSync(path, original.replace(m.find, m.replace))
  let run
  try {
    run = suite()
  } finally {
    writeFileSync(path, original)   // ALWAYS restore, including on a throw
  }
  const caught = !run.green
  if (run.failed === null) {
    console.log(`  ${m.id.padEnd(16)} HARNESS FAILURE — could not read the vitest JSON report`)
    console.error('\n  Without it this gate can only say the suite went red somewhere, which is'
      + '\n  the weaker claim it exists to stop accepting. Fix the reporter, then re-run.')
    process.exit(2)
  }
  const by = run.failed.length ? run.failed.join(', ') : '(nothing failed)'
  console.log(`  ${m.id.padEnd(16)} ${caught ? 'caught' : 'NOT CAUGHT'}   ${m.defect}`)
  if (!caught) {
    console.log(`      shipped once as: ${m.shipped}`)
    holes++
  } else if (!run.failed.includes(m.expect)) {
    // Red, but the wrong check fired. Distinct from NOT CAUGHT and from a pass:
    // the defect is covered only incidentally, so the named test could be
    // deleted tomorrow and this gate would still say caught.
    console.log(`      WRONG CHECK — expected ${m.expect}, got ${by}`)
    console.log(`      ${m.expect} is the test whose name claims this defect. It did not fire,`)
    console.log('      so the coverage is incidental and would not survive that file changing.')
    holes++
  } else {
    console.log(`      by: ${by}`)
  }
}

// Did this run leave anything behind? Compared against the state BEFORE it
// started, not against a clean tree: a file that was already modified stays
// modified, and warning about that is a false alarm people learn to skip. What
// matters is whether the dirt CHANGED.
const dirtAfter = statusOf(touched)
if (dirtAfter !== dirtBefore) {
  console.log('\n  WARNING: a mutated file was not restored to how this run found it:')
  console.log(`    before: ${dirtBefore || '(clean)'}`)
  console.log(`    after:  ${dirtAfter || '(clean)'}`)
}
console.log(`\n${chosen.length - holes}/${chosen.length} mutations without holes`)
if (holes) {
  console.log('A hole is either NOT CAUGHT — the defect is not covered at all — or')
  console.log('WRONG CHECK: the suite went red, but the test whose name claims that')
  console.log('defect never fired, so the coverage is incidental and would not survive')
  console.log('that file changing. The per-mutation line above says which.')
}
process.exit(holes ? 1 : 0)
