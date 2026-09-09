#!/usr/bin/env node
// DECODE-PROFILE-NATIVE — where does a decode token's GPU time actually go?
//
//   npm run dev                                        # weights mirror
//   node --experimental-strip-types scripts/decode-profile-native.mjs qwen3mlx
//
// Decode is the one axis still behind LM Studio (0.85-0.90x, BENCH.md). Prefill
// got there by measuring kernels before touching them; this is the same first
// step for decode, and it exists because guessing which dispatch dominates a
// 30-dispatch-per-layer chain has no reason to work.
//
// READ THE CAVEAT BEFORE QUOTING ANY ABSOLUTE FROM THIS. profileStep() needs
// the `timestamp-query` device feature, and requesting that feature costs ~3x
// decode on dawn.node/Metal all by itself (BENCH.md 2026-08-13) — Dawn's Metal
// timestamp path serializes command execution device-wide. So:
//   - the per-kernel SHARES are what this measures
//   - the totals are inflated and are NOT a decode rate
//   - a kernel that overlaps others in normal running is over-counted here,
//     because under serialization nothing overlaps
// Use decode-bench-native.mjs for rates. This answers "what to attack", not
// "how fast are we".

import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'qwen3mlx'
const TOP = Number(process.env.TOP) || 18

process.env.ZTVM_PROFILE = '1'   // must be set BEFORE the lib picks features
await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const { engine, tokenizer, spec, buildChatPromptFor } = await createEngineRaw({ model })
const ids = buildChatPromptFor(spec, [{ role: 'user', content: 'Write one sentence about the sea.' }], tokenizer)

const prof = await engine.profileStep(ids)
if (!prof) {
  console.error('profileStep returned null — the device has no timestamp-query, or ZTVM_PROFILE was not set before boot')
  process.exit(1)
}

console.log(`\n${spec.id} — one decode step, ${prof.kernels.length} distinct kernels, ${prof.totalMs.toFixed(2)} ms total`)
console.log('(serialized by timestamp-query: shares are the signal, totals are inflated)\n')
console.log(`  ${'kernel'.padEnd(28)} ${'ms'.padStart(8)} ${'calls'.padStart(6)} ${'share'.padStart(7)}`)
let cum = 0
for (const k of prof.kernels.slice(0, TOP)) {
  cum += k.pctOfTotal
  console.log(`  ${k.label.padEnd(28)} ${k.totalMs.toFixed(3).padStart(8)} ${String(k.calls).padStart(6)} `
    + `${k.pctOfTotal.toFixed(1).padStart(6)}%  cum ${cum.toFixed(1)}%`)
}
const rest = prof.kernels.slice(TOP)
if (rest.length) {
  console.log(`  ${`… ${rest.length} more`.padEnd(28)} ${rest.reduce((s, k) => s + k.totalMs, 0).toFixed(3).padStart(8)} `
    + `${String(rest.reduce((s, k) => s + k.calls, 0)).padStart(6)} ${(100 - cum).toFixed(1).padStart(6)}%`)
}

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
