/**
 * PREFIX REUSE — the two pure functions the KV cross-turn reuse rests on.
 *
 * These lived as closures inside `buildDecodeEngine`, closed over `absorbed`,
 * `absorbedValid`, `gdnStatePos`, `prefixReuse` and `hybrid`. Eleven lines
 * deciding whether a turn re-prefills 20,000 tokens or 40, and their only
 * assertion was `checkReuse()` in `bench-console.ts` — which needs a browser,
 * a loaded model, 48 decoded tokens and `?chunk=0`.
 *
 * Extracted so they can be pinned headlessly (`tests/unit/prefix-reuse.test.ts`)
 * before the prefix pool starts changing the rules around them
 * (docs/PAGING_PLAN.md §0.1). The engine calls these; the behaviour is
 * unchanged, deliberately — this is the regression net, not a fix.
 */

/** What the KV cache and (for hybrid) the GDN state currently encode. */
export interface ReuseState {
  /**
   * `absorbed[p]` is the token id whose forward pass was the LAST submitted at
   * position p — what KV slot p encodes, and what the GDN recurrence absorbed
   * at step p.
   */
  absorbed: number[]
  /** Cleared by a readback failure; nothing but a reset brings it back. */
  absorbedValid: boolean
  /** `opts.prefixReuse` — `?reuse=0` turns this off. */
  prefixReuse: boolean
  /** Spec has GDN layers, so the recurrence is not rewindable. */
  hybrid: boolean
  /** How many tokens the GDN recurrent state has absorbed. Ignored if !hybrid. */
  gdnStatePos: number
}

/**
 * Longest reusable prefix of `promptIds` given the absorbed record.
 *
 * Pure attention: `min(LCP, len-1)`. KV slots are idempotent, so stale slots
 * past the LCP are simply overwritten in order by the delta prefill, and at
 * least the final prompt token always runs — it is what produces the first
 * argmax.
 *
 * Hybrid: the GDN recurrence cannot be rewound, so reuse demands that the new
 * prompt extend EVERY absorbed token (`LCP === absorbed.length`) with the state
 * boundary exactly there. In normal chat the ≤1-token stop overrun is the
 * `<|im_end|>` stop id, which IS the next prompt's token, so this holds.
 * Otherwise 0 — a full re-prefill, which re-zeroes the GDN state at position 0.
 */
export function reuseStart(s: ReuseState, promptIds: number[]): number {
  if (!s.prefixReuse || !s.absorbedValid || promptIds.length === 0) return 0
  const max = Math.min(s.absorbed.length, promptIds.length)
  let lcp = 0
  while (lcp < max && s.absorbed[lcp] === promptIds[lcp]) lcp++
  if (!s.hybrid) return Math.min(lcp, promptIds.length - 1)
  if (s.gdnStatePos === s.absorbed.length && lcp === s.absorbed.length && lcp <= promptIds.length - 1) {
    return lcp
  }
  return 0
}

/**
 * Record that position `position` now holds `id`.
 *
 * Two behaviours, and the second is the one with teeth:
 *  - a REWIND (position < length) truncates — later slots are about to be
 *    overwritten anyway;
 *  - a GAP (position > length) means a position was never submitted, so the
 *    record no longer describes the cache. `absorbedValid` goes false and
 *    there is no path back except `resetKVTracking()` — which `chat.ts` never
 *    calls. Reuse is off for the life of that engine.
 */
export function noteAbsorbed(s: ReuseState, position: number, id: number): void {
  if (position > s.absorbed.length) { s.absorbedValid = false; return }
  s.absorbed.length = position
  s.absorbed.push(id)
}
