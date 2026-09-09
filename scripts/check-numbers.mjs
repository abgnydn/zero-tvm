#!/usr/bin/env node
// CHECK NUMBERS — is a published throughput physically possible?
//
// /health reported "3803 tok/s prefill" for a 27.8B model on an M2 Max. That
// implies ~211 TFLOP/s on a machine that peaks near 13. Nothing bounded it, and
// it was one step from BENCH.md. It was wrong in the FLATTERING direction, which
// is the direction nobody double-checks.
//
// So: every throughput this project publishes gets checked against the two
// rooflines that physics imposes. This cannot tell you a number is RIGHT — a
// figure well under the ceiling can still be a lie. It tells you when a number
// is IMPOSSIBLE, which is the cheap half and the half that was missing.
//
//   node scripts/check-numbers.mjs
//   node scripts/check-numbers.mjs --hardware m4max
import { readFileSync, existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { SHIPPED_MODELS } from '../src/zero-tvm/model-registry.ts'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')

// Vendor-published figures, not measurements of ours. They are used only as a
// CEILING, so being slightly generous is the safe direction: a bound that is too
// high produces false negatives (we miss a bad number), never false positives
// (we reject a real one). Every entry is deliberately rounded UP.
const HARDWARE = {
  m2max: { name: 'Apple M2 Max', bandwidthGBs: 400, peakTFLOPs: 27 },
  m4max: { name: 'Apple M4 Max', bandwidthGBs: 546, peakTFLOPs: 34 },
}
const hwKey = (process.argv.includes('--hardware') ? process.argv[process.argv.indexOf('--hardware') + 1] : 'm2max')
const HW = HARDWARE[hwKey]
if (!HW) { console.error(`unknown --hardware ${hwKey}; have ${Object.keys(HARDWARE).join(', ')}`); process.exit(2) }

/**
 * ACTIVE weight bytes per decode token, DERIVED from the spec.
 *
 * The first version parsed the roster's authored "~17.2 GB" label, and got two
 * things wrong at once — which is why this is derived now:
 *
 *   MoE. Qwen3-30B-A3B is 30B total and ~3B ACTIVE: decode reads topK experts
 *   of 128, not all of them. Bounding it by total weights called a real, correct
 *   74.96 tok/s "impossible".
 *
 *   Rounding. "~2 GB" for Phi-3 against a true 2.09 GB moved the ceiling by 3%,
 *   which was enough to flag the roofline fact itself.
 *
 * Every estimate below rounds params DOWN, so the ceiling comes out HIGH. For a
 * bound that is the safe direction: a ceiling that is too high misses a bad
 * number, a ceiling that is too low rejects a good one. GDN layers are counted
 * as attention-shaped, which understates them slightly — same direction.
 */
function activeBytesPerToken(spec) {
  const attnPerLayer =
    spec.d * spec.heads * spec.headDim        // q
    + 2 * spec.d * spec.kvHeads * spec.headDim // k, v
    + spec.heads * spec.headDim * spec.d       // o
  const moe = spec.moe
  const ffnPerLayer = moe
    // Only the routed experts a token actually wakes, plus the shared one.
    ? (moe.topK + (moe.sharedExpert ? 1 : 0)) * 3 * spec.d * spec.ffn
    : 3 * spec.d * spec.ffn
  const embed = spec.vocab * spec.d * (spec.tiedEmbeddings ? 1 : 2)
  const params = embed + spec.layers * (attnPerLayer + ffnPerLayer)
  // 4-bit base; the 3-bit build requantises only the expert stacks, and this is
  // a bound, so the coarser 4-bit figure is used for both (rounds bytes UP,
  // ceiling DOWN — the one place we are deliberately conservative, because a
  // 3-bit claim should have to clear the 4-bit bar).
  return params * 0.5
}

/** Decode streams the active weights once per token. The KV it also reads grows
 *  with context and is omitted, which raises the ceiling — the safe direction. */
const decodeCeiling = (bytes) => (HW.bandwidthGBs * 1e9) / bytes

/** Prefill is compute-bound at ~2 FLOP per active parameter per token. */
const prefillCeiling = (bytes) => (HW.peakTFLOPs * 1e12) / (2 * (bytes * 2))

console.log(`bounds on ${HW.name}: ${HW.bandwidthGBs} GB/s, ~${HW.peakTFLOPs} TFLOP/s\n`)
console.log(`  ${'model'.padEnd(26)} ${'active wt'.padStart(9)} ${'decode ≤'.padStart(10)} ${'prefill ≤'.padStart(11)}`)

const ceilings = new Map()
for (const { param, spec } of SHIPPED_MODELS) {
  if (spec.embeddingOnly) continue
  const bytes = activeBytesPerToken(spec)
  const d = decodeCeiling(bytes), p = prefillCeiling(bytes)
  ceilings.set(spec.id, { param, d, p, bytes })
  console.log(`  ${(param || 'phi3').padEnd(26)} ${(bytes / 2 ** 30).toFixed(1).padStart(7)} GB ${d.toFixed(0).padStart(7)} t/s ${p.toFixed(0).padStart(8)} t/s${spec.moe ? '   (active, MoE)' : ''}`)
}

// ── published numbers ───────────────────────────────────────────────────────
let bad = 0, checked = 0
const FACTS = join(ROOT, '..', 'sites-shared', 'facts.json')
if (existsSync(FACTS)) {
  const facts = JSON.parse(readFileSync(FACTS, 'utf8')).facts ?? []
  console.log('\nregistered facts:')
  for (const f of facts) {
    if (!f.id.startsWith('zerotvm.') || f.status === 'withdrawn') continue
    if (f.unit !== 'tok/s' || typeof f.value !== 'number') continue
    // Match a fact to a model by the segment after "zerotvm." — ids read
    // zerotvm.<param>.<what>, and anything unmatched is REPORTED rather than
    // skipped: a number nothing can be bound to is its own problem.
    const seg = f.id.split('.')[1]
    const hit = [...ceilings.values()].find((c) => (c.param || 'phi3').replace(/[^a-z0-9]/g, '') === seg.replace(/[^a-z0-9]/g, ''))
    if (!hit) { console.log(`  ${f.id.padEnd(40)} ${String(f.value).padStart(8)}  (no model matched — unbounded)`); continue }
    // A fact that IS a roofline cannot be judged against a cruder estimate of
    // the same roofline.
    if (/roofline|ceiling/.test(f.id)) {
      console.log(`  ${f.id.padEnd(40)} ${String(f.value).padStart(8)} t/s  (is itself a ceiling — skipped)`)
      continue
    }
    checked++
    // MARGIN. The parameter estimate is approximate by construction, so a
    // number is only called impossible when it clears the ceiling by more than
    // the estimate could plausibly be wrong. The failure this exists to catch
    // was off by 15x; 1.25 is nowhere near it and keeps false positives at zero.
    const over = f.value > hit.d * 1.25
    if (over) bad++
    const pct = (f.value / hit.d) * 100
    // A FLOOR as well as a ceiling. A bound only catches the flattering
    // direction, and the defect that motivated this file was a TRANSPOSITION:
    // the prefill figure was 686% of its ceiling and the decode figure was 4%
    // of its own. Only one half was impossible; both were wrong. Every honest
    // measurement on this engine sits between 15% and 45% of roofline, so
    // something far below that is a question, not a failure.
    const low = !over && pct < 8
    console.log(`  ${f.id.padEnd(40)} ${String(f.value).padStart(8)} t/s  `
      + (over ? `IMPOSSIBLE — ${pct.toFixed(0)}% of a ${hit.d.toFixed(0)} t/s ceiling`
        : low ? `SUSPICIOUS — only ${pct.toFixed(1)}% of ceiling; explain it or check for a transposed pair`
          : `ok (${pct.toFixed(0)}% of ceiling)`))
  }
} else {
  console.log('\n(no facts.json — skipping registered facts)')
}

// ── the live engine, if one is running ──────────────────────────────────────
// This is where the transposed rates came from, so it is worth checking the
// source rather than only what has already been written down.
try {
  const res = await fetch('http://127.0.0.1:8019/health', { signal: AbortSignal.timeout(1500) })
  const h = await res.json()
  const last = h.last
  if (last && last.prefillTokPerSec) {
    // Match on the spec ID, which /health reports verbatim. The first version
    // matched on the PARAM's first five characters — and Phi-3's param is the
    // empty string, so `hosting.includes('')` was true for everything and every
    // model was bounded by Phi-3's ceiling. It read qwen38's decode as 1.1% of
    // ceiling when the truth was 6.7%.
    const hit = ceilings.get(h.hosting)
    if (!hit) {
      console.log(`\nlive engine: ${h.hosting} is not in the roster — cannot bound it`)
      throw new Error('unmatched')
    }
    console.log('\nlive engine:')
    const overP = last.prefillTokPerSec > hit.p * 1.25
    const overD = last.decodeTokPerSec > hit.d * 1.25
    checked += 2
    if (overP) bad++
    if (overD) bad++
    const lowD = !overD && last.decodeTokPerSec < hit.d * 0.08
    console.log(`  prefill ${String(last.prefillTokPerSec).padStart(8)} t/s  ${overP ? `IMPOSSIBLE — ${(last.prefillTokPerSec / hit.p * 100).toFixed(0)}% of a ${hit.p.toFixed(0)} ceiling` : `ok (${(last.prefillTokPerSec / hit.p * 100).toFixed(0)}% of ceiling)`}`)
    console.log(`  decode  ${String(last.decodeTokPerSec).padStart(8)} t/s  ${overD ? `IMPOSSIBLE — ${(last.decodeTokPerSec / hit.d * 100).toFixed(0)}% of a ${hit.d.toFixed(0)} ceiling` : lowD ? `SUSPICIOUS — only ${(last.decodeTokPerSec / hit.d * 100).toFixed(1)}% of ceiling` : `ok (${(last.decodeTokPerSec / hit.d * 100).toFixed(0)}% of ceiling)`}`)
    if (overP && lowD) {
      console.log('  ONE HIGH AND ONE LOW is the signature of a transposed pair —')
      console.log('  check that the total is right before believing either half.')
    }
    if (overP || overD) {
      console.log('  NOTE: prefill and decode were observed TRANSPOSED on 2026-08-19 —')
      console.log('  the total was right and the split was not. Check both before quoting either.')
    }
  }
} catch { /* no engine running; the facts check above is the offline half */ }

console.log(`\n${checked} checked · ${bad} impossible`)
if (bad) {
  console.log('\nAn impossible number is not a rounding error. Find where it is computed')
  console.log('before it reaches BENCH.md, a model card, or a README.')
}
process.exit(bad ? 1 : 0)
