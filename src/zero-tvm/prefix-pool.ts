/**
 * PREFIX POOL — the fingerprint and the block index (docs/PAGING_PLAN.md §1.1, §1.2).
 *
 * The two pure halves of the pooled-KV design: what makes a cached page VALID
 * (the fingerprint) and what makes it THIS conversation's (the chain hash plus
 * an exact id compare). No GPU, no engine, no IndexedDB, no OPFS — persistence
 * and attachment are later steps. These are written first because they are the
 * part where a mistake produces fluent wrong output rather than an error, and
 * they are the part that can be pinned headlessly.
 *
 * The invariant everything here serves, stated once:
 *
 *     KV[0..n) = f(tokens[0..n), spec, resolved variant set, KV layout)
 *
 * Anything keyed on less than that tuple hands one build's cache — or one
 * conversation's — to another. `share.ts` already leans on the same rule: one
 * host engine serves many guests through one `absorbed` record, and that is
 * safe only because the reuse test is literal token equality.
 */

import type { VariantFlags } from './variants.ts'

// ─────────────────────────────────────────────────────────────────────────
// 128-bit chain hash
// ─────────────────────────────────────────────────────────────────────────

/** 32 lowercase hex characters. */
export type Hash128 = string

const M64 = 0xffffffffffffffffn

// Two independent 64-bit lanes over the same word stream. splitmix64's
// finalizer is a bijection with good avalanche, which is all this needs: the
// hash is the INDEX, the stored token ids are the PROOF. A collision has to
// survive an exact id compare in `lookupPrefix` before it can do damage, so it
// degrades to a miss (a re-prefill), not to a stranger's KV. Nothing here is a
// defence against an adversary who can choose token ids — that is the id
// compare's job, and it is unconditional.
const LANE_A = 0x243f6a8885a308d3n
const LANE_B = 0x13198a2e03707344n

function mix64(x: bigint): bigint {
  x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & M64
  x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & M64
  return (x ^ (x >> 31n)) & M64
}

function absorb(seed: bigint, words: ArrayLike<number>): bigint {
  let h = seed
  for (let i = 0; i < words.length; i++) {
    h = mix64(h ^ ((BigInt(words[i] >>> 0) + 0x9e3779b97f4a7c15n) & M64))
  }
  return h
}

function hashWords(words: ArrayLike<number>): Hash128 {
  const a = absorb(LANE_A, words)
  const b = absorb(LANE_B, words)
  return a.toString(16).padStart(16, '0') + b.toString(16).padStart(16, '0')
}

/** A hash back into the four u32 words that `chainBlockHash` prepends. */
function hashWordsOf(h: Hash128): number[] {
  return [
    Number.parseInt(h.slice(0, 8), 16),
    Number.parseInt(h.slice(8, 16), 16),
    Number.parseInt(h.slice(16, 24), 16),
    Number.parseInt(h.slice(24, 32), 16),
  ]
}

/**
 * h_i = H(h_{i-1} ‖ ids[16(i-1) .. 16i))
 *
 * The parent chaining is not optional. Without it — hashing each block against
 * the bare fingerprint — a block whose 16 ids are `", "path": "src/"`, which is
 * every third line of a tool-call transcript, matches across two unrelated
 * conversations at the same offset. The id compare passes (the ids really are
 * equal) and the graft is silently wrong, because V at those positions encodes
 * a different attention history. Chaining makes the key depend on every token
 * before the block, which is exactly what the KV values depend on.
 */
export function chainBlockHash(parent: Hash128, blockIds: readonly number[]): Hash128 {
  return hashWords([...hashWordsOf(parent), ...blockIds])
}

// ─────────────────────────────────────────────────────────────────────────
// §1.1 — the fingerprint
// ─────────────────────────────────────────────────────────────────────────

/**
 * Everything that has to be bit-identical for a pooled page to be valid.
 *
 * Deliberately NOT here, and the omissions are the interesting part:
 *
 *  - **All sampling parameters** (temperature, top-k/p, seed, repetition
 *    penalty). K/V at position p is a function of tokens[0..p] and the weights
 *    alone — the sampler runs after the cache write and its counter is already
 *    the position (`engine-core.ts:1560-1567`). Adding temperature "to be safe"
 *    would give every position of a slider its own cache namespace and kill
 *    reuse in any chat that has one. Extra sampling fields on the object handed
 *    in are ignored, not hashed; the top-level field list below is a whitelist
 *    for exactly that reason.
 *  - **`variants.splitK`** (see EXCLUDED_VARIANT_FLAGS).
 */
export interface ComputeConfig {
  /** `spec.id` — pins dims, layer kinds, RoPE, page size, tokenizer. */
  specId: string
  /**
   * What identifies the exact weight BYTES: repo + revision, or the local
   * mirror's build id. `spec.id` does not — `?model=qwen36q3` is the same spec
   * id against a checkpoint that gets reconverted.
   */
  weightRevision: string
  /**
   * The RESOLVED variant set. Hashed by reflection over its own keys (minus
   * EXCLUDED_VARIANT_FLAGS), so a new flag enters the fingerprint by default
   * instead of being forgotten — the fail-safe direction. These select among
   * five `qkv_fused` siblings (`variants.ts:296-300`) and differ in f16
   * reduction order, so two variant sets do not write bit-identical K/V.
   */
  variants: VariantFlags
  /**
   * `opts.fused` — the fused composition writes K/V from `qkv_fused*`; the
   * unfused one writes them from `rope` → `kv_append`. Different kernels,
   * different rounding. Not a VariantFlags field, so it is named here.
   */
  fused: boolean
  /**
   * `opts.int8KV` — a different cache LAYOUT, not just a precision: pages plus
   * a separate scales buffer (`allocKVPagesInt8`). Note this is also a
   * VariantFlags field; the page parses the flag but the ENGINE takes the
   * option, and the two can disagree, so the effective one is hashed here too.
   */
  int8KV: boolean
  /** `opts.layerRange` — a pipeline stage's cache holds only its own layers. */
  layerRange: { start: number; end: number } | null
  /**
   * The prefill path that actually WROTE these pages. Chunked and per-token
   * are not bit-equal: the chunked path builds the prefix with the batched
   * matmul kernels, whose reduction order differs from the per-token GEMV
   * (`bench-console.ts:388-395` documents this as the reason `checkReuse()`
   * needs `?chunk=0` for its exactly-0 assertion). This is the path that ran,
   * not the option that was requested — the gate at `engine-core.ts:2173-2174`
   * decides.
   */
  prefillPath: 'chunked' | 'per-token'
  /**
   * `adapter.info`. A different GPU compiles the same WGSL to different
   * arithmetic. (Whether one adapter is bit-stable across driver updates is
   * open — PAGING_PLAN PART 4 item 6; if it is not, this needs an epoch
   * counter beside it.)
   */
  adapter: { vendor: string; architecture: string }
}

/**
 * The one VariantFlags field deliberately outside the fingerprint.
 *
 * `splitK` partitions the attention OUTPUT reduction across the KV range and
 * combines the partials (`attention_splitk.wgsl` + `attention_combine`). It
 * changes how the cache is READ and reduced; it writes nothing. Keying on it
 * would split the pool in two for a flag whose whole point is that it is an
 * A/B toggle on the same cache.
 *
 * Adding to this set is a correctness decision — the reflective test in
 * tests/unit/prefix-pool.test.ts asserts the set's exact contents so it cannot
 * grow quietly.
 */
export const EXCLUDED_VARIANT_FLAGS: ReadonlySet<string> = new Set<string>(['splitK'])

// Bumped only when the LAYOUT of the preimage below changes. Not a driver
// epoch — see ComputeConfig.adapter.
const PREIMAGE_VERSION = 'zero-tvm/kv-prefix/1'

/**
 * The canonical string the fingerprint hashes. Exported because a hex digest
 * is not diagnosable: when a boot logs a fingerprint and the hit rate is 0,
 * this is the line that says which field moved.
 */
export function fingerprintPreimage(cfg: ComputeConfig): string {
  const v = cfg.variants as unknown as Record<string, unknown>
  const variantLines = Object.keys(v)
    .filter((k) => !EXCLUDED_VARIANT_FLAGS.has(k))
    .sort()   // key order is an accident of construction; the fingerprint is not
    .map((k) => `  ${k}=${String(v[k])}`)
  return [
    PREIMAGE_VERSION,
    `spec=${cfg.specId}`,
    `weights=${cfg.weightRevision}`,
    `fused=${cfg.fused}`,
    `int8kv=${cfg.int8KV}`,
    `layers=${cfg.layerRange ? `${cfg.layerRange.start}-${cfg.layerRange.end}` : 'all'}`,
    `prefill=${cfg.prefillPath}`,
    `adapter=${cfg.adapter.vendor}|${cfg.adapter.architecture}`,
    'variants:',
    ...variantLines,
  ].join('\n')
}

/**
 * h_0 — the root of the chain. Two configurations that are not bit-identical
 * get disjoint chains, so a pooled block from one can never be walked into by
 * the other, whatever its token ids are.
 */
export function computeFingerprint(cfg: ComputeConfig): Hash128 {
  const s = fingerprintPreimage(cfg)
  const words = new Array<number>(s.length)
  for (let i = 0; i < s.length; i++) words[i] = s.charCodeAt(i)
  return hashWords(words)
}

// ─────────────────────────────────────────────────────────────────────────
// §1.2 — the block index and the walk
// ─────────────────────────────────────────────────────────────────────────

/**
 * One pageSize-token block, keyed in the index by its chain hash.
 *
 * `ids` is the proof, and it is why the hash does not have to be
 * cryptographic. `parent` is belt-and-braces: it can only disagree with the
 * walk position under a full 128-bit collision or a tampered index, and both
 * of those are exactly the cases worth turning into a miss.
 */
export interface PoolBlock {
  parent: Hash128
  ids: readonly number[]
}

/**
 * The in-memory index. Tier 1 in the plan's terms — a manifest, not payloads.
 * Blocks from different fingerprints share this map safely: h_0 is the
 * fingerprint, so their chains never intersect.
 */
export interface PoolIndex {
  /** `spec.pageSize` — 16 on every shipped spec. Fixed at creation so two
   *  calls cannot disagree about where the block boundaries are. */
  pageSize: number
  blocks: Map<Hash128, PoolBlock>
}

export function createPoolIndex(pageSize: number): PoolIndex {
  return { pageSize, blocks: new Map() }
}

export interface InsertResult {
  /** Full blocks the sequence covers. */
  blocks: number
  /** Blocks new to the index. The rest were already there — the shared prefix. */
  inserted: number
  /**
   * Keys that already existed holding DIFFERENT ids: a real 128-bit collision.
   * The existing record wins; overwriting it would break whichever conversation
   * legitimately owns those bytes. Zero in every run that is not pathological,
   * which is why it is counted rather than assumed.
   */
  collisions: number
}

/**
 * Record the full blocks of `ids` under `fingerprint`'s chain.
 *
 * Only whole blocks are chained. A trailing partial block belongs to a page
 * that is still being filled; sharing it is the tail-page copy-on-write
 * problem, which is Phase 3.
 */
export function insertPrefix(ix: PoolIndex, fingerprint: Hash128, ids: readonly number[]): InsertResult {
  const blocks = Math.floor(ids.length / ix.pageSize)
  let h = fingerprint
  let inserted = 0
  let collisions = 0
  for (let i = 0; i < blocks; i++) {
    const block = ids.slice(i * ix.pageSize, (i + 1) * ix.pageSize)
    const next = chainBlockHash(h, block)
    const existing = ix.blocks.get(next)
    if (!existing) {
      ix.blocks.set(next, { parent: h, ids: block })
      inserted++
    } else if (existing.parent !== h || !sameIds(existing.ids, block)) {
      collisions++
    }
    h = next
  }
  return { blocks, inserted, collisions }
}

export type PoolMissReason =
  /** No entry ever stored this chain position — the ordinary miss. */
  | 'unknown-block'
  /** A hash hit whose ids differ: a collision, refused. */
  | 'ids-differ'
  /** A hash hit whose parent link disagrees with the walk: refused. */
  | 'parent-differs'
  /** Fewer than pageSize prompt tokens left — nothing more is poolable. */
  | 'prompt-exhausted'

export interface PoolMatch {
  /** Full blocks matched. */
  blocks: number
  /** `blocks * pageSize` — the page-aligned LCP. */
  tokens: number
  /** h_1 … h_blocks, in order. These are the payload keys a restore needs. */
  hashes: Hash128[]
  reason: PoolMissReason
}

/**
 * Walk the chain until a miss.
 *
 * The result is the page-aligned longest common prefix and nothing more. In
 * particular it does NOT apply `reuseStart`'s "always leave one token to run"
 * rule (`prefix-reuse.ts`) — that belongs to whoever attaches, because the
 * final prompt token must run to produce the first argmax, and clamping here
 * would make `tokens` stop meaning "tokens this index can prove".
 */
export function lookupPrefix(ix: PoolIndex, fingerprint: Hash128, promptIds: readonly number[]): PoolMatch {
  const maxBlocks = Math.floor(promptIds.length / ix.pageSize)
  const hashes: Hash128[] = []
  let h = fingerprint
  for (let i = 0; i < maxBlocks; i++) {
    const block = promptIds.slice(i * ix.pageSize, (i + 1) * ix.pageSize)
    const next = chainBlockHash(h, block)
    const rec = ix.blocks.get(next)
    if (!rec) return { blocks: i, tokens: i * ix.pageSize, hashes, reason: 'unknown-block' }
    if (rec.parent !== h) return { blocks: i, tokens: i * ix.pageSize, hashes, reason: 'parent-differs' }
    if (!sameIds(rec.ids, block)) return { blocks: i, tokens: i * ix.pageSize, hashes, reason: 'ids-differ' }
    hashes.push(next)
    h = next
  }
  return { blocks: maxBlocks, tokens: maxBlocks * ix.pageSize, hashes, reason: 'prompt-exhausted' }
}

function sameIds(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}
