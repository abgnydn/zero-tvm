#!/usr/bin/env node
// PREFILL-GEMM-AB — is the faster GEMM a faster PREFILL?
//
//   npm run dev                                          # weights mirror
//   node --experimental-strip-types scripts/prefill-gemm-ab.mjs llama32
//
// scripts/gemm-sweep-native.mjs measures kernels in isolation, back to back
// over one buffer set. A prefill dispatches that kernel among many others,
// with bind-group churn and attention between, so a TF win there is a
// hypothesis about tok/s here, not a result. This settles it.
//
// Two engines over the SAME weight buffers, arms alternated A,B,A,B within one
// process — the discipline decode-bench-native.mjs was written to enforce after
// a cross-run comparison produced a wrong conclusion twice in one day. Prefill
// is timed as TTFT on a fresh prefix each round (reuse off, pool off), so no
// round can be served from another's cache.
//
// Numbers are VOID unless the machine is on healthy power: `pmset -g batt`
// must say charged/charging on a real adapter first.

import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'llama32'
const ROUNDS = Number(process.env.ROUNDS) || 4
const PROMPT = Number(process.env.PROMPT) || 1024

await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const boot = async (chunkGemm) => {
  const t = Date.now()
  const built = await createEngineRaw({ model, chunkGemm })
  console.log(`[boot] ${chunkGemm} in ${((Date.now() - t) / 1000).toFixed(1)}s`)
  return built
}
const arms = [
  { name: 'sgmat (E1)', ...(await boot('sgmat')) },
  { name: 'e5', ...(await boot('e5')) },
]

// Distinct ids per round so no round can reuse a previous prefix, and distinct
// per arm-pair so the two arms see the SAME tokens in the same round.
const idsFor = (r) => Array.from({ length: PROMPT }, (_, i) => 1000 + ((i * 37 + r * 101) % 900))

// generatePipelined, NOT forwardLogits: forwardLogits prefills token by token
// (a decodeToken loop) and never enters the chunk path at all, so an A/B built
// on it reads the same number for every GEMM — it did, 205 tok/s twice, before
// this comment existed. One token is generated because prefill only runs as
// part of a generation; it costs ~5 ms against a multi-second prefill.
const rate = async (arm, r) => {
  const ids = idsFor(r)
  const t0 = performance.now()
  await arm.engine.generatePipelined(ids, 1, () => {})
  const ms = performance.now() - t0
  const p = arm.engine.getLastPrefill()
  if (!p?.chunks) throw new Error(`${arm.name}: prefill did not chunk (${JSON.stringify(p)}) — nothing was measured`)
  if (p.reused) throw new Error(`${arm.name}: reused ${p.reused} tokens of prefix — rounds must not share a prefix`)
  return PROMPT / (ms / 1000)
}

const got = { 'sgmat (E1)': [], e5: [] }
for (let r = 0; r < ROUNDS; r++) {
  for (const arm of r % 2 === 0 ? arms : [...arms].reverse()) {
    const tokS = await rate(arm, r)
    got[arm.name].push(tokS)
    console.log(`round ${r}  ${arm.name.padEnd(11)} ${tokS.toFixed(0).padStart(5)} tok/s prefill`)
  }
}

const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
const a = med(got['sgmat (E1)']), b = med(got.e5)
console.log(`\n${model} prefill, ${PROMPT}-token prompt, ${ROUNDS} rounds, medians:`)
console.log(`  sgmat (E1) ${a.toFixed(0)} tok/s`)
console.log(`  e5         ${b.toFixed(0)} tok/s   ${((b / a - 1) * 100).toFixed(1)}%`)
console.log(`  spread     E1 ${Math.min(...got['sgmat (E1)']).toFixed(0)}-${Math.max(...got['sgmat (E1)']).toFixed(0)}`
  + `  e5 ${Math.min(...got.e5).toFixed(0)}-${Math.max(...got.e5).toFixed(0)}`)

// dawn.node holds the loop open after the last submit; leave explicitly.
process.exit(0)
