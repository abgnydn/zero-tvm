#!/usr/bin/env node
// MOE-POOL-TEST — is expert pooling CORRECT, and what does it cost?
//
//   npm run dev                                          # weights mirror
//   npm run build:lib                                    # dist-lib
//   node --experimental-strip-types scripts/moe-pool-test.mjs qwen30b
//
// The pool holds N of E experts per layer and fetches the rest on demand
// (src/zero-tvm/expert-pool.ts). Everything it evicts it may have to fetch
// again, so the whole feature is a bet that routing is time-local — measured at
// 50.0 / 77.8 / 94.5% LRU hit rate for 13/32/64 of 128 slots over a 1827-step
// qwen30b trace (docs/MOE_CHUNK_PLAN.md, "Skew is weak; locality is what
// works"). This is the gate on that bet, in four parts:
//
//   1. TOKEN IDENTITY against the unpooled engine. Pooling is a residency
//      change, not a numerics change: the same expert rows reach the same
//      matmul at a different row base, and int4_matmul already reads its base
//      from ids[]. If the tokens move, the slot mapping is wrong — and a wrong
//      slot does not error, it multiplies by another expert's weights and
//      returns a plausible sentence. Every cost number below is void until
//      this holds.
//   2. A pool too small to hold ONE token's routing must THROW. Evicting an
//      expert the current token is still about to use is the same silent
//      wrongness, reached by arithmetic rather than by a bug.
//   3. COST: decode tok/s per pool size, paired against the unpooled arm
//      inside each round with the arm order flipped, medians of the per-round
//      ratios. Absolute rates decay ~20% as the SoC heats, so an unpaired
//      median mostly reports where in that decay each arm's samples landed
//      (BENCH.md 2026-08-13, the retracted idle-loop finding).
//   4. The measured hit rate against what the replay predicted. The replay is
//      the entire justification for building this; if the engine does not
//      reproduce it, the pool is not running the policy that was scored.
//
// TWO ENGINES ARE RESIDENT AT ONCE and each loads its own weights, so qwen30b
// costs ~17 GB (unpooled) plus the pooled arm's share. `POOLS=0.5` runs one
// size per process on a box that cannot hold the whole sweep.
//
// Timings are REFUSED off AC rather than warned about: this repo has published
// numbers off a discharging battery before, and a throttled SoC looks exactly
// like a pool that is thrashing.

import { execFileSync } from 'node:child_process'
import { installShims } from './native/shims.mjs'

const model = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'qwen30b'
const ROUNDS = Number(process.env.ROUNDS) || 4
const TOKENS = Number(process.env.TOKENS) || 64
const WARMUP = 8
const FRACTIONS = (process.env.POOLS ?? '0.10,0.25,0.50').split(',').filter(Boolean).map(Number)

let failed = false
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(30)} ${detail}`)
  if (!ok) failed = true
}
const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
// dawn.node holds the loop open after the last submit and piped stdout is
// written asynchronously, so every exit drains first — exit() otherwise
// discards the report that explains why it exited.
const leave = async (code) => {
  await new Promise((r) => process.stdout.write('', r))
  process.exit(code)
}

// The knobs can switch this gate off without switching off its verdict: below
// one round nothing is compared, and at TOKENS <= WARMUP every round discards
// everything it generated. Either way the identity check ends up agreeing an
// empty list with an empty list and printing PASS.
if (!(ROUNDS >= 1) || !(TOKENS > WARMUP)) {
  console.log(`FAIL  ROUNDS=${ROUNDS} TOKENS=${TOKENS} — needs >= 1 round and > ${WARMUP} tokens, or nothing is compared`)
  await leave(1)
}

// ---- power ---------------------------------------------------------------
// The correctness half is power-independent and still runs on battery; only
// the tok/s half is withheld, because that is the half that can be wrong
// without looking wrong.
const batt = execFileSync('pmset', ['-g', 'batt'], { encoding: 'utf8' })
const onAC = /AC Power/.test(batt) && !/discharging/.test(batt)
if (!onAC) {
  console.log(`TIMINGS REFUSED — pmset -g batt says: ${batt.trim().split('\n').join(' | ')}`)
  console.log('  correctness and hit rates still run. Plug in for tok/s.\n')
}

await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const boot = async (name, opts) => {
  const t = Date.now()
  const built = await createEngineRaw({ model, ...opts })
  console.log(`[boot] ${name} in ${((Date.now() - t) / 1000).toFixed(1)}s`)
  return { name, ...built }
}

const base = await boot('unpooled', {})
const spec = base.spec
if (!spec.moe) throw new Error(`${model} is not a MoE spec — there are no experts to pool`)

const E = spec.moe.experts
const K = spec.moe.topK
// One slot MORE than a single token needs. Below top-K + 1 a token whose
// top-K all miss has every slot claimed by itself and nothing left to evict,
// so the pool would have to evict an expert it is about to read — wrong
// output, not slow output. This is the bound `buildDecodeEngine` states, so it
// is the bound tested, not one re-derived here: a harness that invents its own
// floor tests its own arithmetic.
const FLOOR = K + 1
console.log(`[spec] ${spec.id} · ${spec.layers} layers · ${E} experts · top-${K}`
  + `${spec.sharedExpertIndex >= 0 ? ' + shared (pinned)' : ''} · min pool ${FLOOR}`)

const sizes = [...new Set(FRACTIONS.map((f) => {
  const want = Math.ceil(f * E)
  // Clamping is announced. A fraction quietly raised to the floor would be
  // reported under its own percentage label and compared against a replay
  // prediction for a pool size that never ran.
  if (want < FLOOR) console.log(`[pools] ${(f * 100).toFixed(0)}% of ${E} is ${want} slots — raised to the ${FLOOR}-slot floor`)
  return Math.max(FLOOR, want)
}))]
if (!sizes.length) {
  console.log('FAIL  POOLS is empty — this run would boot an engine and compare it against nothing')
  await leave(1)
}

// docs/MOE_CHUNK_PLAN.md, LRU column of the residency table. Keyed by slot
// count because that is what the replay swept; a different spec gets no
// prediction rather than a borrowed one.
const PREDICTED = { 'qwen3-30b-a3b-4bit': { 13: 50.0, 32: 77.8, 64: 94.5 } }
// The table above is a 1827-step trace. The plan records 243 steps agreeing
// within a point EXCEPT at the half pool, which reads 90.2% there — a short
// run has not had time to fill 64 slots with anything worth keeping. Comparing
// a 256-step run against the long-trace number would manufacture a 4-point
// "gap" out of trace length.
const SHORT = { 'qwen3-30b-a3b-4bit': { 64: 90.2 } }
const GAP = 10   // points; wider than the 4-point trace-length effect

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

const snap = (arm) => {
  const s = arm.engine.getPoolStats?.()
  return s ? { hits: s.hits, requests: s.requests } : null
}

const run = async (arm, r) => {
  const ids = idsFor(arm, r)
  // Every round prefills in full. Cross-turn reuse would otherwise serve most
  // of round 2+ from the KV cache, and the arms would no longer be entering
  // decode the same way — the call `bench()` was missing when the WebLLM
  // comparison accidentally measured our decode against their decode+prefill.
  arm.engine.resetKVTracking()
  const out = []
  let tWarm = 0
  // Snapshot at the FIRST token, not at token 0: prefill touches essentially
  // every expert in every layer and is nearly all misses, so folding it into
  // the hit rate would roughly halve a number the replay computed over decode
  // steps only. The pipeline runs a step or two ahead of the callback, so this
  // boundary is exact to within those steps and not to the dispatch.
  let sPre = null
  const t0 = performance.now()
  await arm.engine.generatePipelined(ids, TOKENS, (t) => {
    out.push(t)
    if (out.length === 1) sPre = snap(arm)
    if (out.length === WARMUP) tWarm = performance.now()
  }, () => false)
  const t1 = performance.now()
  // A run that generated nothing cannot pass anything. Two empty token lists
  // are elementwise identical and a rate over zero tokens is NaN, so both the
  // identity gate and the cost gate would report success on an engine that did
  // not run.
  if (out.length <= WARMUP) {
    throw new Error(`${arm.name}: generated ${out.length} tokens, need > ${WARMUP} to exclude warmup`)
  }
  const p = arm.engine.getLastPrefill()
  if (p?.reused) throw new Error(`${arm.name}: reused ${p.reused} prompt tokens — rounds must not share a prefix`)
  const sEnd = snap(arm)
  return {
    tokS: (out.length - WARMUP) / ((t1 - tWarm) / 1000),
    out,
    decode: sPre && sEnd
      ? { hits: sEnd.hits - sPre.hits, requests: sEnd.requests - sPre.requests, steps: out.length - 1 }
      : null,
    lifetime: sEnd,
  }
}

// The unpooled arm must report NO pool. If it does, the option leaked into the
// baseline and both arms are the same engine — the shape of vacuous pass this
// harness exists to prevent.
check('unpooled arm has no pool', (base.engine.getPoolStats?.() ?? null) === null,
  String(base.engine.getPoolStats?.() ?? 'null'))

// A divergence is only a BUG if the model was sure. Greedy decoding takes a
// hard argmax over a continuous quantity, and a pooled expert matmul reads the
// same rows from a different base address — which can reorder nothing at all,
// or can change which buffer the compiler vectorises. Where the top two logits
// are a rounding apart, either token is the model's answer. This repo spent an
// afternoon "fixing" exactly that once (BENCH.md, the reverted split-K change),
// so the gap gets measured before anything is called broken.
//
// Run per pool size rather than once at the end: a tie at 13 slots says
// nothing about a divergence at 64, and one shared verdict would let the
// second one print PASS on the strength of the first one's retraction.
const TIE = 0.05
const diagnose = async (m) => {
  const lg = await base.engine.forwardLogits([...m.ids, ...m.prefix])
  const rank = [...lg.keys()].sort((a, b) => lg[b] - lg[a])
  const gap = lg[rank[0]] - lg[rank[1]]
  const std = Math.sqrt(lg.reduce((s, v) => s + v * v, 0) / lg.length
    - (lg.reduce((s, v) => s + v, 0) / lg.length) ** 2)
  console.log(`      TOKENS DIVERGE — ${m.where}`)
  console.log(`      top-2 logit gap at that position ${gap.toFixed(4)} against a logit std of ${std.toFixed(2)}`)
  return gap >= TIE
}

const cost = []

for (const slots of sizes) {
  const pct = ((slots / E) * 100).toFixed(0)
  console.log(`\n=== pool ${slots}/${E} (${pct}%) ===`)
  const pooled = await boot(`pool ${slots}`, { expertPool: slots })

  // Written against an option that may not exist yet. An unknown property on
  // an options object is silently dropped at runtime, so without this the
  // pooled arm is a second unpooled engine that agrees with the baseline on
  // every token and reports a plausible tok/s ratio of ~1.0. Say so and stop.
  if (typeof pooled.engine.getPoolStats !== 'function') {
    console.log('\nengine.getPoolStats is not a function — this build has no expert pool.')
    console.log('  The pooled arm would be a second copy of the unpooled one and every')
    console.log('  gate below would pass without testing anything. Not working around it.')
    await leave(1)
  }
  if (pooled.engine.getPoolStats() === null) {
    console.log(`\ngetPoolStats() returned null on an engine built with expertPool: ${slots}`)
    console.log('  — the option did not take effect. Not working around it.')
    await leave(1)
  }

  const got = { [base.name]: [], [pooled.name]: [] }
  const acc = { hits: 0, requests: 0, steps: 0 }
  let mismatch = null
  for (let r = 0; r < ROUNDS; r++) {
    const outs = {}
    for (const arm of r % 2 === 0 ? [base, pooled] : [pooled, base]) {
      const res = await run(arm, r)
      got[arm.name].push(res.tokS)
      outs[arm.name] = res.out
      if (arm === pooled && res.decode) {
        acc.hits += res.decode.hits
        acc.requests += res.decode.requests
        acc.steps += res.decode.steps
      }
      if (onAC) console.log(`round ${r}  ${arm.name.padEnd(12)} ${res.tokS.toFixed(1).padStart(6)} tok/s decode`)
    }
    const a = outs[base.name]
    const b = outs[pooled.name]
    const at = b.findIndex((t, i) => t !== a[i])
    if (!mismatch && (a.length !== b.length || at >= 0)) {
      // at < 0 with unequal lengths means one arm stopped early — still a
      // divergence, and the position to interrogate is where the short arm ran
      // out rather than -1 (which would slice the agreed prefix one short and
      // diagnose the wrong position).
      const pos = at >= 0 ? at : Math.min(a.length, b.length)
      mismatch = {
        where: `pool ${slots}, round ${r}: pooled vs unpooled at generated token ${pos} `
          + (at >= 0 ? `(${b[pos]} vs ${a[pos]})` : `(lengths ${b.length} vs ${a.length})`),
        ids: idsFor(base, r),
        prefix: a.slice(0, pos),   // the tokens both arms agreed on
      }
    }
  }

  const n = got[base.name].length
  const bug = mismatch ? await diagnose(mismatch) : false
  check(`pool ${slots} tokens identical`, !bug,
    !mismatch ? `${n} rounds x ${TOKENS} tokens match the unpooled arm`
      : bug ? 'the model was DECIDED and the pooled arm disagrees — the slot mapping is wrong, '
        + 'and every cost number for this pool is measuring another model'
        : 'the one divergence is a logit tie: the model had no preference there, so either '
          + 'token is its answer. The arms agree everywhere the model was decided.')
  // The pool must be on the hot path. Zero requests means the engine built a
  // pool and then routed around it.
  check(`pool ${slots} was actually used`, acc.requests > 0,
    `${acc.requests} expert requests over ${acc.steps} decode steps`)

  const measured = acc.requests ? (acc.hits / acc.requests) * 100 : 0
  const table = PREDICTED[spec.id] ?? {}
  const short = acc.steps < 500 ? (SHORT[spec.id] ?? {})[slots] : undefined
  const want = short ?? table[slots]
  const life = pooled.engine.getPoolStats()
  console.log(`      hit rate ${measured.toFixed(1)}% over ${acc.steps} decode steps`
    + `   (lifetime incl. prefill ${(life.hitRate * 100).toFixed(1)}%)`)
  if (want !== undefined) {
    const d = measured - want
    console.log(`      replay predicted ${want.toFixed(1)}%${short !== undefined ? ' (short-trace figure)' : ''}`
      + `   gap ${d >= 0 ? '+' : ''}${d.toFixed(1)} points`)
    if (Math.abs(d) > GAP) {
      console.log(`      WARNING: the engine is not reproducing the replay that justified this pool.`)
      console.log(`      LRU over the real router ids scored ${want.toFixed(1)}% here. A gap this wide means`)
      console.log(`      the eviction the engine runs is not the policy that was scored — or the`)
      console.log(`      requests being counted are not the ones the replay counted.`)
    }
  } else {
    console.log(`      no replay prediction for ${spec.id} at ${slots} slots — nothing to compare against`)
  }

  if (onAC) {
    // PAIRED, and that is the whole point: differencing the two arms INSIDE a
    // round cancels the thermal drift that made two runs of the decode-kernel
    // A/B read +7.9% and +17.4% on the same change.
    const ratios = got[pooled.name].map((v, i) => v / got[base.name][i])
    cost.push({ slots, pct, pooled: med(got[pooled.name]), base: med(got[base.name]), ratios, hit: measured })
    console.log(`      median ${med(got[pooled.name]).toFixed(1)} tok/s pooled vs `
      + `${med(got[base.name]).toFixed(1)} unpooled`)
  }

  pooled.engine.destroy()
}

if (onAC && cost.length) {
  console.log(`\n${model} decode, ${TOKENS} tokens (first ${WARMUP} excluded), ${ROUNDS} rounds each:`)
  console.log(`  ${'pool'.padEnd(12)} ${'tok/s'.padStart(7)} ${'vs unpooled'.padStart(12)}   ${'hit'.padStart(6)}   per-round ratios`)
  for (const c of cost) {
    console.log(`  ${`${c.slots}/${E} (${c.pct}%)`.padEnd(12)} ${c.pooled.toFixed(1).padStart(7)} `
      + `${`${((med(c.ratios) - 1) * 100).toFixed(1)}%`.padStart(12)}   ${`${c.hit.toFixed(1)}%`.padStart(6)}   `
      + c.ratios.map((r) => r.toFixed(3)).join('  '))
  }
  console.log(`  unpooled arm median ${med(cost.map((c) => c.base)).toFixed(1)} tok/s`)
}

// ---- the pool that must not run ------------------------------------------
// LAST, and deliberately: this arm loads a third set of weights, and a boot
// that throws part-way leaves whatever it had already allocated unreachable
// (createEngineRaw does not hand back the weight buffers, so nothing here can
// free them). Running it after the sweep keeps that out of the timed arms.
base.engine.destroy()
const tooSmall = FLOOR - 1
try {
  const bad = await createEngineRaw({ model, expertPool: tooSmall })
  // It booted. The only thing that matters now is whether it GENERATES: a pool
  // that cannot hold one token's routing must refuse, because the alternative
  // is evicting an expert mid-token and returning a fluent wrong answer.
  const out = []
  const ids = bad.buildChatPromptFor(bad.spec, [{ role: 'user', content: prompts[0] }], bad.tokenizer)
  await bad.engine.generatePipelined(ids, 16, (t) => out.push(t), () => false)
  check('too-small pool refused', false,
    `${tooSmall} slots (top-${K} needs ${FLOOR}) `
    + `booted AND generated ${out.length} tokens — those tokens cannot be right`)
  bad.engine.destroy()
} catch (e) {
  check('too-small pool refused', true, `${tooSmall} slots threw: ${String(e?.message ?? e).slice(0, 110)}`)
}

console.log(failed
  ? '\nEXPERT POOLING IS NOT CORRECT ON THIS BUILD — cost numbers above are void'
  : `\nexpert pooling reproduces the unpooled tokens at ${sizes.join(', ')} slots of ${E}`
    + (onAC ? '' : ' (timings refused — not on AC)'))

await leave(failed ? 1 : 0)
