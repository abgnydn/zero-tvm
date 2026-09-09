#!/usr/bin/env node
// MOE-POOL-CONTROL — is the pool test's premise true?
//
//   node --experimental-strip-types scripts/moe-pool-control.mjs qwen30b
//
// moe-pool-test.mjs compares a pooled arm against an unpooled one ACROSS
// PROCESSES, on this reasoning (its own comment): "Greedy decoding is
// deterministic, so cross-process token identity is exactly as strong a check
// as in-process."
//
// That premise has never been tested, and every FAIL it reports depends on it.
// If two identical unpooled runs already diverge, then "pool diverges at token
// N" measures run-to-run nondeterminism and says nothing about the pool.
//
// So this runs the SAME arm twice — unpooled both times, same prompt, same
// token count — and reports the first index where they differ. It is the null
// experiment: it should find nothing, and if it finds something, the pool
// results are void until it is explained.
//
// Correctness only. No timings, so it runs on battery.

import { execFileSync } from 'node:child_process'

const model = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'qwen30b'
const TOKENS = Number(process.env.TOKENS) || 512
const WARMUP = 8
const PROMPT = process.env.PROMPT
  ?? 'Explain how a heat pump moves energy against a gradient.'
// SLOTS=0 (default) runs both arms unpooled — the original null experiment.
// SLOTS=N runs both arms POOLED, which asks a different and sharper question:
// is the pooled path deterministic AT ALL? A logic bug in the slot mapping is
// reproducible, so two pooled runs would agree with each other and differ from
// unpooled. A race would make them differ from EACH OTHER.
const SLOTS = Number(process.env.SLOTS) || 0

/** Run one arm in its own process, exactly as moe-pool-test.mjs does. */
function runArm(label) {
  const t = Date.now()
  const out = execFileSync(process.execPath, [
    '--experimental-strip-types', 'scripts/moe-pool-test.mjs', model,
    '--arm', JSON.stringify({ prompt: PROMPT, tokens: TOKENS, warmup: WARMUP, slots: SLOTS }),
  ], { encoding: 'utf8', maxBuffer: 1 << 28 })
  const line = out.trim().split('\n').filter((l) => l.startsWith('{')).at(-1)
  if (!line) throw new Error(`${label}: no JSON from arm\n${out.slice(-2000)}`)
  const r = JSON.parse(line)
  if (r.error) throw new Error(`${label}: ${r.error}`)
  console.log(`[${label}] ${r.tokens.length} tokens in ${((Date.now() - t) / 1000).toFixed(1)}s`)
  return r.tokens
}

const ARM_LABEL = SLOTS ? `pooled(${SLOTS}) vs pooled(${SLOTS})` : 'unpooled vs unpooled'
console.log(`\n${model} — ${ARM_LABEL}, ${TOKENS} tokens, same prompt`)
console.log(`prompt: ${PROMPT}\n`)

const a = runArm('run A')
const b = runArm('run B')

let at = -1
for (let i = 0; i < Math.min(a.length, b.length); i++) {
  if (a[i] !== b[i]) { at = i; break }
}
const sameLen = a.length === b.length

if (at < 0 && sameLen) {
  console.log(`\nPASS  two ${SLOTS ? 'pooled' : 'unpooled'} runs are token-identical  ${a.length} tokens`)
  console.log(SLOTS
    ? '      the pooled path is DETERMINISTIC — divergence from unpooled is a logic bug.'
    : '      cross-process comparison is sound; a pool FAIL means the pool.')
  process.exit(0)
}

console.log(`\nFAIL  two ${SLOTS ? 'pooled' : 'unpooled'} runs differ`)
if (!sameLen) console.log(`      lengths ${a.length} vs ${b.length}`)
if (at >= 0) {
  console.log(`      first difference at token ${at}: ${a[at]} vs ${b[at]}`)
  const from = Math.max(0, at - 4)
  console.log(`      A[${from}..${at}] ${a.slice(from, at + 1).join(',')}`)
  console.log(`      B[${from}..${at}] ${b.slice(from, at + 1).join(',')}`)
}
console.log('      moe-pool-test.mjs compares across processes on the assumption')
console.log('      that this cannot happen. Its pool verdicts are VOID until it is.')
process.exit(1)
