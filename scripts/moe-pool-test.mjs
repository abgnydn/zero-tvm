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
const GEN = process.env.GEN ?? 'pipelined'
const SPEC_W = Number(process.env.SPEC) || 0   // 1 = on at top-K; >K = coverage width
const SPECULATE = SPEC_W >= 1

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

// ONE ENGINE PER PROCESS. The first version booted the pooled and unpooled arms
// side by side, and createEngineRaw loads its own weights per call — ~34 GB for
// a 16 GB model, which does not fit on the machine this feature exists for.
// Arms run as separate processes and the driver pairs them, the same
// compromise scripts/lmstudio-ab.mjs and scripts/kernel-ab.mjs already make
// because they compare across runtimes. Greedy decoding is deterministic, so
// cross-process token identity is exactly as strong a check as in-process.
const ARM = process.argv.includes('--arm')
  ? JSON.parse(process.argv[process.argv.indexOf('--arm') + 1])
  : null

if (ARM) {
  // BOTH arms prefill per-token. Pooling cannot chunk, and chunked prefill is
  // empirically-not-bit-equal to per-token — leaving the unpooled arm chunked
  // made the identity gate compare prefill modes, not pooling. The "pooled
  // pipelined diverges at token 116" finding was exactly this: two UNPOOLED
  // arms (pipelined chunked vs blocking per-token) diverge at the same 116.
  const built = await createEngineRaw({ model, chunkedPrefill: false, ...(ARM.slots ? { expertPool: ARM.slots } : {}), ...(ARM.spec ? { expertSpeculate: true, ...(ARM.spec > 1 ? { specWidth: ARM.spec } : {}) } : {}) })
  const out = []
  const ids = built.buildChatPromptFor(
    built.spec, [{ role: 'user', content: ARM.prompt }], built.tokenizer)
  built.engine.resetKVTracking()
  const t0 = performance.now()
  let tWarm = 0
  const onTok = (t) => {
    out.push(t)
    if (out.length === ARM.warmup) tWarm = performance.now()
  }
  // GEN=blocking runs the readback-per-token generate() instead of the
  // pipelined one. generatePipelined keeps PIPELINE_DEPTH tokens in flight with
  // argmax chained on-GPU, and the POOLED path has to read B.moeIds back
  // mid-forward and then overwrite it with slot indices — one buffer, several
  // tokens in flight. This flag is how that is tested rather than argued about.
  if (ARM.gen === 'blocking') {
    await built.engine.generate(ids, 0, ARM.tokens, onTok)
  } else {
    await built.engine.generatePipelined(ids, ARM.tokens, onTok, () => false)
  }
  const t1 = performance.now()
  if (out.length <= ARM.warmup) {
    console.log(JSON.stringify({ error: `generated ${out.length} tokens, need > ${ARM.warmup}` }))
    process.exit(0)
  }
  const p = built.engine.getLastPrefill()
  console.log(JSON.stringify({
    tokens: out,
    tokS: (out.length - ARM.warmup) / ((t1 - tWarm) / 1000),
    reused: p?.reused ?? 0,
    pool: built.engine.getPoolStats?.() ?? null,
  }))
  await new Promise((r) => process.stdout.write('', r))
  process.exit(0)
}

const boot = async (name, opts) => {
  const t = Date.now()
  const built = await createEngineRaw({ model, ...opts })
  console.log(`[boot] ${name} in ${((Date.now() - t) / 1000).toFixed(1)}s`)
  return { name, ...built }
}

const HERE = new URL('..', import.meta.url).pathname
const runArm = (arm) => {
  const out = execFileSync('node', [
    '--experimental-strip-types', `${HERE}scripts/moe-pool-test.mjs`, model,
    '--arm', JSON.stringify(arm),
  ], { encoding: 'utf8', maxBuffer: 1 << 26, cwd: HERE })
  const line = out.trim().split('\n').at(-1)
  const r = JSON.parse(line)
  if (r.error) throw new Error(`arm ${JSON.stringify(arm)}: ${r.error}`)
  return r
}

// Spec comes from the registry, not from a booted engine — no GPU needed and
// no 16 GB load just to read topK.
const { specForParam } = await import('../src/zero-tvm/model-registry.ts')
const spec = specForParam(model)
if (!spec.moe) { console.error(`${model} is not a MoE spec`); process.exit(1) }
const E = spec.moe.experts
const K = spec.moe.topK
const { ExpertPool } = await import('../src/zero-tvm/expert-pool.ts')
const FLOOR = ExpertPool.minSlots(spec.moeSlots, spec.sharedExpertIndex >= 0 ? 1 : 0)

const sizes = [...new Set(FRACTIONS.map((f) => Math.max(FLOOR, Math.round(E * f))))].sort((a, b) => a - b)
console.log(`\n${spec.id} — ${E} experts, top-${K}, floor ${FLOOR} slots`)
console.log(`pool sizes: ${sizes.join(', ')}\n`)

const PREDICTED = { 'qwen3-30b-a3b-4bit': { 13: 50.0, 32: 77.8, 64: 94.5 } }[spec.id] ?? {}
const prompts = [
  'Explain how a heat pump moves energy against a gradient.',
  'Write a function that reverses a linked list in Rust.',
  'Summarise why the Antikythera mechanism surprised historians.',
  'Given 5x - 9 = 26, solve for x and show each step.',
]

// ---- correctness: token identity, per pool size -------------------------
const ref = []
for (let r = 0; r < Math.min(ROUNDS, prompts.length); r++) {
  ref.push(runArm({ prompt: prompts[r], tokens: TOKENS, warmup: WARMUP, slots: 0, gen: GEN }))
}
check('unpooled arm produced tokens', ref.every((x) => x.tokens.length > WARMUP),
  `${ref[0].tokens.length} tokens/round`)
check('unpooled arm reports no pool', ref.every((x) => x.pool === null), 'getPoolStats null')

for (const slots of sizes) {
  let same = true
  const diverged = []
  const rates = [], hits = []
  for (let r = 0; r < ref.length; r++) {
    const got = runArm({ prompt: prompts[r], tokens: TOKENS, warmup: WARMUP, slots, gen: GEN, spec: SPEC_W })
    rates.push(got.tokS)
    if (got.pool) hits.push(got.pool.hitRate)
    if (SPECULATE && got.pool?.speculation) {
      const sp = got.pool.speculation
      console.log(`      speculation r${r}: slot ${(sp.accuracy * 100).toFixed(1)}% · `
        + `top1 ${(sp.top1Rate * 100).toFixed(1)}% · COVERAGE ${(sp.setRate * 100).toFixed(1)}% `
        + `(${sp.steps} steps)`)
    }
    const i = got.tokens.findIndex((t, k) => t !== ref[r].tokens[k])
    if (got.tokens.length !== ref[r].tokens.length || i >= 0) {
      same = false
      // EVERY round, not just the last. This used to overwrite `at` per round,
      // so a multi-round run reported one index and hid the rest — and the
      // pattern ACROSS prompts is the diagnostic: an index that moves with
      // prompt length is positional (a KV page boundary), one that does not is
      // a property of the pool itself.
      diverged.push({ round: r, at: i, prompt: prompts[r].slice(0, 38) })
    }
  }
  check(`pool ${slots}/${E} token-identical`, same,
    same ? `${ref[0].tokens.length} tokens x ${ref.length} rounds`
      : `${diverged.length}/${ref.length} rounds diverge`)
  for (const d of diverged) {
    console.log(`      round ${d.round} diverges at token ${d.at}  "${d.prompt}…"`)
  }
  if (hits.length) {
    const h = med(hits) * 100
    const want = PREDICTED[slots]
    const note = want === undefined ? '' : ` vs ${want.toFixed(1)}% predicted`
    console.log(`      hit rate ${h.toFixed(1)}%${note}`)
    if (want !== undefined && Math.abs(h - want) > 10) {
      console.log(`      WARNING: more than 10 points from the replay that justified building this`)
    }
  }
  const base = med(ref.map((x) => x.tokS))
  console.log(`      ${med(rates).toFixed(1)} tok/s vs ${base.toFixed(1)} unpooled `
    + `(${((med(rates) / base - 1) * 100).toFixed(0)}%)${onAC ? '' : '  [VOID: on battery]'}`)
}

// ---- a pool below the floor must refuse, not corrupt --------------------
let refused = false
try { runArm({ prompt: prompts[0], tokens: 8, warmup: 0, slots: Math.max(1, FLOOR - 1) }) }
catch { refused = true }
check(`pool below floor (${FLOOR - 1}) refuses`, refused, refused ? 'threw at boot' : 'GENERATED ANYWAY')

await leave(failed ? 1 : 0)
