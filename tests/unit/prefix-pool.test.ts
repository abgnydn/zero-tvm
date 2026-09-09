// PREFIX POOL — the fingerprint and the chain-hash index
// (docs/PAGING_PLAN.md §1.1, §1.2).
//
// Two failures are being tested for, and neither of them raises an error in
// production:
//
//   1. Serving one BUILD's cache to another. The fingerprint has to cover
//      every input that changes the bits written into the KV cache. The test
//      that matters here enumerates VariantFlags REFLECTIVELY, so adding a
//      shader variant without adding it to the fingerprint goes red in CI
//      instead of quietly corrupting a pool.
//   2. Serving one CONVERSATION's cache to another. A block of 16 common
//      punctuation ids is identical across unrelated transcripts; only the
//      parent chaining makes its key depend on the history that produced its
//      V values. That argument is tested directly below, not asserted.
//
// No GPU, no storage, no engine — these are pure functions by construction.

import { describe, expect, it } from 'vitest'
import { SCALAR_VARIANTS, type VariantFlags } from '../../src/zero-tvm/variants.ts'
import {
  EXCLUDED_VARIANT_FLAGS,
  chainBlockHash,
  computeFingerprint,
  createPoolIndex,
  fingerprintPreimage,
  insertPrefix,
  lookupPrefix,
  type ComputeConfig,
} from '../../src/zero-tvm/prefix-pool.ts'

const PAGE = 16   // spec.pageSize on every shipped spec

const config = (over: Partial<ComputeConfig> = {}): ComputeConfig => ({
  specId: 'qwen35-4b',
  weightRevision: 'mlc-ai/Qwen3.5-4B-q4f16_1-MLC@0c1f2ab',
  variants: SCALAR_VARIANTS,
  fused: false,
  int8KV: false,
  layerRange: null,
  prefillPath: 'per-token',
  adapter: { vendor: 'apple', architecture: 'metal-3' },
  ...over,
})

const fp = (over: Partial<ComputeConfig> = {}) => computeFingerprint(config(over))

// Small deterministic PRNG (mulberry32) — the same one tests/kernels/run.mjs
// uses, so a failing seed is reproducible.
function rng(seed: number): () => number {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const ids = (n: number, r: () => number) => Array.from({ length: n }, () => Math.floor(r() * 50000))

describe('the fingerprint — every variant flag, by reflection', () => {
  // THE test. Object.keys over the resolved flags, not a hand-written list:
  // a hand-written list is exactly the thing that goes stale when someone adds
  // a sixth qkv_fused sibling.
  it('flipping any non-excluded VariantFlags field changes the fingerprint', () => {
    const base = fp()
    const flags = SCALAR_VARIANTS as unknown as Record<string, boolean | number | string>
    const keys = Object.keys(flags)

    // Guard the enumeration itself: a loop over zero keys passes every
    // assertion inside it. VariantFlags has 15 fields today.
    expect(keys.length).toBeGreaterThanOrEqual(15)

    const changed: string[] = []
    const unchanged: string[] = []
    for (const key of keys) {
      const v = flags[key]
      const flippedValue = typeof v === 'boolean' ? !v
                         : typeof v === 'number' ? v + 1
                         : v === 'scalar' ? 'tiled' : 'scalar'
      const variants = { ...SCALAR_VARIANTS, [key]: flippedValue } as VariantFlags
      ;(fp({ variants }) === base ? unchanged : changed).push(key)
    }
    expect(unchanged.sort()).toEqual([...EXCLUDED_VARIANT_FLAGS].sort())
    expect(changed.length).toBe(keys.length - EXCLUDED_VARIANT_FLAGS.size)
  })

  it('the exclusion set is exactly {splitK} — growing it needs a deliberate edit', () => {
    // splitK reorders the attention OUTPUT reduction and combines partials.
    // It changes how the cache is read; it writes nothing into it.
    expect([...EXCLUDED_VARIANT_FLAGS]).toEqual(['splitK'])
    expect(fp({ variants: { ...SCALAR_VARIANTS, splitK: 8 } })).toBe(fp())
    expect(fp({ variants: { ...SCALAR_VARIANTS, splitK: 2 } })).toBe(fp({ variants: { ...SCALAR_VARIANTS, splitK: 16 } }))
  })

  it('the five qkv_fused siblings land on five different fingerprints', () => {
    // sgQkv / qkvTile / qkvTile2 / vec4Qkv select among the siblings at
    // variants.ts:296-300, and they differ in f16 reduction order — so they do
    // not write bit-identical K into the cache.
    const seen = new Set([
      fp(),
      fp({ variants: { ...SCALAR_VARIANTS, subgroups: true, sgQkv: true } }),
      fp({ variants: { ...SCALAR_VARIANTS, subgroups: true, sgQkv: true, qkvTile: true } }),
      fp({ variants: { ...SCALAR_VARIANTS, subgroups: true, sgQkv: true, qkvTile2: true } }),
      fp({ variants: { ...SCALAR_VARIANTS, subgroups: true, sgQkv: true, vec4Qkv: true } }),
    ])
    expect(seen.size).toBe(5)
  })
})

describe('the fingerprint — everything outside VariantFlags', () => {
  it('separates specs, weight revisions and adapters', () => {
    const base = fp()
    expect(fp({ specId: 'qwen3-4b' })).not.toBe(base)
    expect(fp({ weightRevision: 'mlc-ai/Qwen3.5-4B-q4f16_1-MLC@deadbee' })).not.toBe(base)
    expect(fp({ adapter: { vendor: 'intel', architecture: 'metal-3' } })).not.toBe(base)
    expect(fp({ adapter: { vendor: 'apple', architecture: 'metal-4' } })).not.toBe(base)
  })

  it('separates the fused composition and the int8 KV LAYOUT', () => {
    // int8KV is not a precision knob — it is pages plus a separate scales
    // buffer, so an f16 payload restored into an int8 cache is not merely
    // lossy, it is misaligned.
    expect(fp({ fused: true })).not.toBe(fp())
    expect(fp({ int8KV: true })).not.toBe(fp())
  })

  it('separates pipeline stages — a stage caches only its own layers', () => {
    const whole = fp()
    expect(fp({ layerRange: { start: 0, end: 16 } })).not.toBe(whole)
    expect(fp({ layerRange: { start: 16, end: 32 } })).not.toBe(whole)
    expect(fp({ layerRange: { start: 0, end: 16 } })).not.toBe(fp({ layerRange: { start: 16, end: 32 } }))
  })

  it('separates the prefill path that wrote the pages', () => {
    // Chunked builds the prefix with the batched-matmul kernels, whose
    // reduction order differs from the per-token GEMV — the documented reason
    // checkReuse() needs ?chunk=0 for its exactly-0 assertion
    // (bench-console.ts:388-395).
    expect(fp({ prefillPath: 'chunked' })).not.toBe(fp({ prefillPath: 'per-token' }))
  })
})

describe('the fingerprint — what it deliberately ignores', () => {
  it('is identical for two configs differing only in sampling', () => {
    // K/V at position p is a function of tokens[0..p] and the weights alone.
    // Keying on temperature would give every position of the slider its own
    // namespace and reduce the pool to nothing in any chat that has one.
    const hot = { ...config(), temperature: 0.9, topK: 40, topP: 0.95, seed: 7, repetitionPenalty: 1.1 }
    const cold = { ...config(), temperature: 0.0, topK: 1, topP: 1.0, seed: 999, repetitionPenalty: 1.0 }
    expect(computeFingerprint(hot as ComputeConfig)).toBe(computeFingerprint(cold as ComputeConfig))
    expect(computeFingerprint(hot as ComputeConfig)).toBe(fp())
    expect(fingerprintPreimage(hot as ComputeConfig)).not.toContain('temperature')
  })

  it('does not depend on the key order the flags object was built in', () => {
    const shuffled = Object.fromEntries(
      Object.entries(SCALAR_VARIANTS).reverse(),
    ) as unknown as VariantFlags
    expect(fp({ variants: shuffled })).toBe(fp())
  })
})

describe('the fingerprint — shape', () => {
  it('is 128 bits of lowercase hex and is stable across calls', () => {
    expect(fp()).toMatch(/^[0-9a-f]{32}$/)
    expect(fp()).toBe(fp())
  })

  it('the preimage names the field that moved', () => {
    // A hex digest tells a user with a 0% hit rate nothing. This is the line
    // a boot log prints beside it.
    expect(fingerprintPreimage(config())).toContain('spec=qwen35-4b')
    expect(fingerprintPreimage(config())).toContain('prefill=per-token')
    expect(fingerprintPreimage(config())).toContain('  matmul=scalar')
    expect(fingerprintPreimage(config())).not.toContain('splitK')
  })
})

describe('the chain walk — page-aligned LCP', () => {
  const r = rng(1)
  const stored = ids(20005, r)

  it('matches a sequence against itself, to the last WHOLE block', () => {
    // 20005 tokens is 1250 whole blocks plus 5. The 5 live in a page that is
    // still being filled — sharing that page is the tail copy-on-write, which
    // is Phase 3, not this.
    const ix = createPoolIndex(PAGE)
    expect(insertPrefix(ix, fp(), stored)).toEqual({ blocks: 1250, inserted: 1250, collisions: 0 })
    const m = lookupPrefix(ix, fp(), stored)
    expect(m.tokens).toBe(20000)
    expect(m.blocks).toBe(1250)
    expect(m.hashes.length).toBe(1250)
    expect(m.reason).toBe('prompt-exhausted')
  })

  it('returns 0 for a prompt that shares nothing, and for one shorter than a block', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    expect(lookupPrefix(ix, fp(), ids(64, rng(2))).tokens).toBe(0)
    expect(lookupPrefix(ix, fp(), stored.slice(0, 15)).reason).toBe('prompt-exhausted')
    expect(lookupPrefix(ix, fp(), stored.slice(0, 15)).tokens).toBe(0)
    expect(lookupPrefix(ix, fp(), []).tokens).toBe(0)
  })

  it('rounds a divergence INSIDE a block down to the previous boundary', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    // Share 20 tokens: block 0 is whole and shared, block 1 diverges at its
    // 5th id. One block, 16 tokens — the other 4 are re-prefilled.
    const prompt = [...stored.slice(0, 20), ...ids(40, rng(3))]
    const m = lookupPrefix(ix, fp(), prompt)
    expect(m.tokens).toBe(16)
    expect(m.reason).toBe('unknown-block')
  })

  it('gives back every whole block when the divergence sits exactly on a boundary', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    const prompt = [...stored.slice(0, 32), ...ids(40, rng(4))]
    expect(lookupPrefix(ix, fp(), prompt).tokens).toBe(32)
  })

  it('extends: a longer prompt keeps the whole stored prefix', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored.slice(0, 20000))
    const m = lookupPrefix(ix, fp(), [...stored.slice(0, 20000), ...ids(37, rng(5))])
    expect(m.tokens).toBe(20000)
    expect(m.reason).toBe('unknown-block')
  })

  it('does NOT apply the leave-one-token-to-run rule', () => {
    // reuseStart() clamps to len-1 because the final prompt token produces the
    // first argmax. That is the attach step's job; here `tokens` has to keep
    // meaning "tokens this index can prove", or the property below stops
    // holding.
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored.slice(0, 32))
    expect(lookupPrefix(ix, fp(), stored.slice(0, 32)).tokens).toBe(32)
  })
})

describe('the chain — the reason it is not optional', () => {
  // The argument from PAGING_PLAN §1.2, constructed rather than asserted: two
  // unrelated conversations that share one block at the same offset. In a
  // tool-call transcript that block is something like `", "path": "src/"` and
  // it recurs constantly.
  const PUNCT = [11, 497, 788, 705, 1820, 788, 330, 3548, 14, 11, 497, 788, 705, 1820, 788, 330]
  const convA = [...ids(16, rng(10)), ...PUNCT, ...ids(16, rng(11))]
  const convB = [...ids(16, rng(12)), ...PUNCT, ...ids(16, rng(13))]

  it('the shared block really is shared, and really is at the same offset', () => {
    expect(convA.slice(16, 32)).toEqual(convB.slice(16, 32))
    expect(convA.slice(0, 16)).not.toEqual(convB.slice(0, 16))
  })

  it('WITHOUT the chain the two share a key — the bug this design avoids', () => {
    // Hashing each block against the bare fingerprint is the unchained scheme.
    // Both conversations produce the identical key for block 1, the id compare
    // passes (the ids really are equal), and B is served A's V values, which
    // encode a different history. Fluent, wrong, silent.
    const unchainedA = chainBlockHash(fp(), convA.slice(16, 32))
    const unchainedB = chainBlockHash(fp(), convB.slice(16, 32))
    expect(unchainedA).toBe(unchainedB)
  })

  it('WITH the chain the same block hashes differently and the lookup misses', () => {
    const h1A = chainBlockHash(fp(), convA.slice(0, 16))
    const h1B = chainBlockHash(fp(), convB.slice(0, 16))
    expect(chainBlockHash(h1A, convA.slice(16, 32))).not.toBe(chainBlockHash(h1B, convB.slice(16, 32)))

    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), convA)
    const m = lookupPrefix(ix, fp(), convB)
    expect(m.tokens).toBe(0)
    expect(m.reason).toBe('unknown-block')
  })
})

describe('the index refuses what it cannot prove', () => {
  const r = rng(20)
  const stored = ids(64, r)

  it('a forced collision degrades to a MISS, not to a stranger KV', () => {
    // Plant a record under the exact key the walk will compute, holding
    // somebody else's ids — a 128-bit collision, by hand. The id compare is
    // what makes the hash strength not load-bearing.
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    const h1 = chainBlockHash(fp(), stored.slice(0, 16))
    const key = chainBlockHash(h1, stored.slice(16, 32))
    ix.blocks.set(key, { parent: h1, ids: ids(16, rng(21)) })

    const m = lookupPrefix(ix, fp(), stored)
    expect(m.blocks).toBe(1)
    expect(m.reason).toBe('ids-differ')
  })

  it('a tampered parent link is refused', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    const h1 = chainBlockHash(fp(), stored.slice(0, 16))
    const key = chainBlockHash(h1, stored.slice(16, 32))
    ix.blocks.set(key, { parent: 'f'.repeat(32), ids: stored.slice(16, 32) })

    const m = lookupPrefix(ix, fp(), stored)
    expect(m.blocks).toBe(1)
    expect(m.reason).toBe('parent-differs')
  })

  it('never serves a block across fingerprints, however identical the ids', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    expect(lookupPrefix(ix, fp({ int8KV: true }), stored).tokens).toBe(0)
    expect(lookupPrefix(ix, fp({ specId: 'qwen3-4b' }), stored).tokens).toBe(0)
    expect(lookupPrefix(ix, fp({ variants: { ...SCALAR_VARIANTS, matmul: 'tiled' } }), stored).tokens).toBe(0)
    // ...but splitK is excluded, so an A/B of it keeps the pool.
    expect(lookupPrefix(ix, fp({ variants: { ...SCALAR_VARIANTS, splitK: 8 } }), stored).tokens).toBe(64)
  })

  it('re-inserting a shared prefix stores nothing twice', () => {
    // Two conversations forking at token 32: the shared blocks are one record,
    // which is the aliasing the whole design is for.
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    const branch = [...stored.slice(0, 32), ...ids(32, rng(22))]
    expect(insertPrefix(ix, fp(), branch)).toEqual({ blocks: 4, inserted: 2, collisions: 0 })
    expect(ix.blocks.size).toBe(6)
    // Both branches still resolve to their own full length.
    expect(lookupPrefix(ix, fp(), stored).tokens).toBe(64)
    expect(lookupPrefix(ix, fp(), branch).tokens).toBe(64)
  })

  it('an insert never overwrites a colliding record — the other conversation owns it', () => {
    const ix = createPoolIndex(PAGE)
    insertPrefix(ix, fp(), stored)
    const h1 = chainBlockHash(fp(), stored.slice(0, 16))
    const planted = ids(16, rng(23))
    ix.blocks.set(h1, { parent: fp(), ids: planted })
    expect(insertPrefix(ix, fp(), stored).collisions).toBe(1)
    expect(ix.blocks.get(h1)!.ids).toEqual(planted)
  })
})

describe('property: the walk never claims more than the true LCP', () => {
  it('holds over 400 random fork points', () => {
    const r = rng(7)
    for (let iter = 0; iter < 400; iter++) {
      const storedLen = 1 + Math.floor(r() * 200)
      const shared = Math.floor(r() * (storedLen + 1))
      const stored = ids(storedLen, r)
      // A prompt sharing exactly `shared` tokens: same prefix, then a token
      // guaranteed different, then noise.
      const tail = ids(1 + Math.floor(r() * 60), r).map((v) => v + 1)
      const prompt = shared < storedLen
        ? [...stored.slice(0, shared), (stored[shared] + 1 + Math.floor(r() * 100)) % 50000, ...tail]
        : [...stored, ...tail]
      // Recompute the LCP rather than trusting `shared` — the "guaranteed
      // different" token above is only guaranteed if the arithmetic is right.
      let lcp = 0
      while (lcp < Math.min(stored.length, prompt.length) && stored[lcp] === prompt[lcp]) lcp++

      const ix = createPoolIndex(PAGE)
      insertPrefix(ix, fp(), stored)
      const m = lookupPrefix(ix, fp(), prompt)

      expect(m.tokens).toBeLessThanOrEqual(lcp)
      // Tight form: with one stored sequence the walk finds every whole shared
      // block and stops at the first that is not.
      expect(m.tokens).toBe(Math.min(Math.floor(lcp / PAGE), Math.floor(stored.length / PAGE)) * PAGE)
      expect(m.tokens % PAGE).toBe(0)
      expect(m.hashes.length).toBe(m.blocks)
    }
  })
})
