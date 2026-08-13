#!/usr/bin/env node
// DECODE-KERNEL-AB — do the affine wide-load matmuls make decode faster?
//
//   npm run dev                                        # weights mirror
//   node --experimental-strip-types scripts/decode-kernel-ab.mjs qwen3mlx
//
// Decode is memory-bound and the FFN is 54% of a decode token (BENCH.md,
// "Where a decode token's time goes"). Until 2026-08-13 the MLX-affine family
// had no vec4/vec4h sibling, so every affine model read weights one u32 at a
// time while the symmetric family read four. This measures whether closing
// that gap moves tok/s, on the models that actually run it.
//
// Two engines, arms alternated A,B,B,A within one process, rates over tokens
// 8.. so prefill and warmup are excluded — the discipline that exists because
// a cross-run comparison produced a wrong conclusion twice in one day
// (BENCH.md 2026-08-13, the retracted idle-loop finding).
//
// It also asserts the two arms GENERATE THE SAME TOKENS. A faster kernel that
// changes the output is not a faster kernel, and a wide-load reindexing bug
// produces plausible logits rather than an error.
//
// Numbers are VOID unless the machine is on healthy power: `pmset -g batt`
// must say charged/charging on a real adapter first. A discharging battery
// hard-throttles the SoC and this bench cannot tell that from a slow kernel.

import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'qwen3mlx'
const ROUNDS = Number(process.env.ROUNDS) || 4
const TOKENS = Number(process.env.TOKENS) || 64
const WARMUP = 8

await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const boot = async (name, variantQuery) => {
  const t = Date.now()
  const built = await createEngineRaw({ model, variantQuery })
  console.log(`[boot] ${name} in ${((Date.now() - t) / 1000).toFixed(1)}s`)
  return { name, ...built }
}
// '?vec4aff=0' is the pre-2026-08-13 affine path: narrow u32 weight loads and
// one f16 activation at a time.
const arms = [
  { ...(await boot('wide (vec4h)', '')) },
  { ...(await boot('narrow (u32)', '?vec4aff=0')) },
]

const prompts = [
  'Write a long essay about the history of navigation at sea.',
  'Explain how tides work in detail.',
  'Describe the water cycle thoroughly.',
  'Write a detailed history of mapmaking.',
]

// Distinct prompt per round so no round is served from another's prefix cache;
// the SAME prompt across arms within a round so the comparison is like for like.
const idsFor = (arm, r) => arm.buildChatPromptFor(
  arm.spec, [{ role: 'user', content: prompts[r % prompts.length] }], arm.tokenizer)

const run = async (arm, r) => {
  const ids = idsFor(arm, r)
  // Every round prefills in full. Cross-turn reuse would otherwise serve most
  // of round 2+ from the KV cache — the chat template alone is 30 of these 47
  // tokens — and the arms would no longer be entering decode the same way.
  // This is the call `bench()` was missing when the WebLLM comparison
  // accidentally measured our decode against their decode+prefill.
  arm.engine.resetKVTracking()
  const out = []
  let tWarm = 0
  const t0 = performance.now()
  await arm.engine.generatePipelined(ids, TOKENS, (t) => {
    out.push(t)
    if (out.length === WARMUP) tWarm = performance.now()
  }, () => false)
  const t1 = performance.now()
  if (out.length <= WARMUP) {
    throw new Error(`${arm.name}: generated ${out.length} tokens, need > ${WARMUP} to exclude warmup`)
  }
  const p = arm.engine.getLastPrefill()
  if (p?.reused) throw new Error(`${arm.name}: reused ${p.reused} prompt tokens — rounds must not share a prefix`)
  return { tokS: (out.length - WARMUP) / ((t1 - tWarm) / 1000), out }
}

const got = Object.fromEntries(arms.map((a) => [a.name, []]))
const seen = {}
let mismatch = null
for (let r = 0; r < ROUNDS; r++) {
  for (const arm of r % 2 === 0 ? arms : [...arms].reverse()) {
    const { tokS, out } = await run(arm, r)
    got[arm.name].push(tokS)
    const key = `r${r}`
    if (!seen[key]) seen[key] = { name: arm.name, out, ids: idsFor(arm, r) }
    else if (!mismatch) {
      const at = out.findIndex((t, i) => t !== seen[key].out[i])
      if (out.length !== seen[key].out.length || at >= 0) {
        mismatch = {
          where: `round ${r}: ${arm.name} vs ${seen[key].name} at generated token ${at} `
            + `(${out[at]} vs ${seen[key].out[at]})`,
          ids: seen[key].ids,
          prefix: seen[key].out.slice(0, at),   // the tokens both arms agreed on
        }
      }
    }
    console.log(`round ${r}  ${arm.name.padEnd(18)} ${tokS.toFixed(1).padStart(6)} tok/s decode`)
  }
}

const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
console.log(`\n${model} decode, ${TOKENS} tokens (first ${WARMUP} excluded), ${ROUNDS} rounds:`)
for (const a of arms) {
  console.log(`  ${a.name.padEnd(18)} median ${med(got[a.name]).toFixed(1)} tok/s   `
    + `spread ${Math.min(...got[a.name]).toFixed(1)}-${Math.max(...got[a.name]).toFixed(1)}`)
}

// PAIRED, and that is the whole point. Absolute rates decay ~20% across a run
// as the SoC heats, so a median of absolutes mostly reports where in the decay
// each arm's samples happened to land — two runs of this comparison read +7.9%
// and +17.4% that way while every round agreed within itself. Differencing the
// two arms INSIDE a round cancels the drift, exactly as quality-ab.py's paired
// windows cancel between-window difficulty (unpaired z=0.8, paired z=14.7 on
// the same data). Arm order flips every round, so neither arm sits second —
// the hotter slot — more often than the other.
const ratios = got[arms[0].name].map((w, i) => w / got[arms[1].name][i])
const won = ratios.filter((r) => r > 1).length
console.log(`\n  paired per-round ratio (${arms[0].name} / ${arms[1].name}):`)
console.log(`    ${ratios.map((r) => r.toFixed(3)).join('  ')}`)
console.log(`    median ${((med(ratios) - 1) * 100).toFixed(1)}%   `
  + `range ${((Math.min(...ratios) - 1) * 100).toFixed(1)}% to ${((Math.max(...ratios) - 1) * 100).toFixed(1)}%   `
  + `won ${won}/${ratios.length} rounds`)

if (mismatch) {
  // A divergence is only a BUG if the model was sure. Greedy decoding takes a
  // hard argmax over a continuous quantity, so where the top two logits are a
  // rounding apart, any change in summation order lands on the other side —
  // and a wide load sums 16 values per iteration where the narrow one sums 8.
  // This repo spent an afternoon "fixing" exactly that once (BENCH.md, the
  // reverted split-K change), so the gap gets measured before anything is
  // called broken.
  const TIE = 0.05
  const lg = await arms[0].engine.forwardLogits([...mismatch.ids, ...mismatch.prefix])
  const rank = [...lg.keys()].sort((a, b) => lg[b] - lg[a])
  const gap = lg[rank[0]] - lg[rank[1]]
  const std = Math.sqrt(lg.reduce((s, v) => s + v * v, 0) / lg.length
    - (lg.reduce((s, v) => s + v, 0) / lg.length) ** 2)
  console.log(`\nTOKENS DIVERGE — ${mismatch.where}`)
  console.log(`  top-2 logit gap at that position ${gap.toFixed(4)} against a logit std of ${std.toFixed(2)}`)
  if (gap < TIE) {
    console.log('  TIE: the model had no preference there, so either token is its answer.')
    console.log('  Not a kernel bug — the arms agree everywhere the model was decided.')
  } else {
    console.log('  NOT a tie: the model was decided and the arms disagree. This is a bug.')
    process.exitCode = 1
  }
} else {
  console.log(`\nboth arms generated identical tokens in all ${ROUNDS} rounds`)
}

// dawn.node holds the loop open after the last submit; leave explicitly, but
// drain stdout first — piped stdout is written asynchronously and exit()
// discards what has not flushed, which here is the whole comparison.
await new Promise((r) => process.stdout.write('', r))
process.exit(process.exitCode ?? 0)
