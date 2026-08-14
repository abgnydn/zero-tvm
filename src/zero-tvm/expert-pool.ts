/**
 * EXPERT POOL — hold N of E experts per layer, resolve the rest on demand.
 *
 * 93% of a sparse-MoE checkpoint is expert weights and only top-K of them run
 * per token, so the rest are resident for nothing. This keeps a fixed number of
 * slots per layer and maps expert ids onto them.
 *
 * THE KERNEL DOES NOT CHANGE. int4_matmul's moe variant already reads its row
 * base from `ids[]` and multiplies by the row stride, so a buffer holding SLOTS
 * experts with `ids[]` holding SLOT indices is the same code path it already
 * runs. The whole feature is buffer size plus this bookkeeping.
 *
 * LRU, and that is measured rather than assumed: on a 1827-step trace of
 * qwen30b, LRU reaches 94.5% at a half pool against LFU's 57.3% and Belady's
 * 95.3% (docs/MOE_CHUNK_PLAN.md). Expert use here is time-local, not skewed,
 * which is the opposite of what one published analysis reports — and it means
 * a cleverer policy has under a point left to win.
 *
 * Pure and synchronous on purpose: the I/O and the GPU work belong to the
 * caller, so this is unit-testable without either.
 */

export interface PoolResolution {
  /** Slot index per requested expert, in request order. */
  slots: Int32Array
  /** Experts that were not resident, paired with the slot each landed in. */
  misses: Array<{ expert: number; slot: number }>
  /** Experts evicted to make room, in eviction order. Callers that keep any
   *  per-slot state (an upload fence, a dirty flag) need to drop it. */
  evicted: number[]
}

export class ExpertPool {
  readonly slots: number
  /** Expert currently in each slot, -1 when the slot has never been filled. */
  private readonly expertOf: Int32Array
  private readonly slotOf = new Map<number, number>()
  /** Monotonic use stamp per slot. Cheaper than a list and exact for LRU. */
  private readonly usedAt: Float64Array
  private clock = 0
  private free: number[]
  /** Experts pinned into a slot for the lifetime of the pool. */
  private readonly pinned = new Set<number>()

  hits = 0
  requests = 0

  constructor(slots: number, opts: { pin?: readonly number[] } = {}) {
    if (slots < 1) throw new Error(`expert pool needs at least one slot, got ${slots}`)
    this.slots = slots
    this.expertOf = new Int32Array(slots).fill(-1)
    this.usedAt = new Float64Array(slots)
    this.free = Array.from({ length: slots }, (_, i) => slots - 1 - i)
    // A shared expert runs for EVERY token, so evicting it is always wrong and
    // counting it as a hit would flatter the numbers. It takes a slot up front.
    for (const e of opts.pin ?? []) {
      const slot = this.free.pop()
      if (slot === undefined) throw new Error(`cannot pin ${e}: pool of ${slots} is full`)
      this.expertOf[slot] = e
      this.slotOf.set(e, slot)
      this.pinned.add(e)
    }
  }

  /** Which slot holds `expert`, or -1. Does not count as a use. */
  slotFor(expert: number): number {
    return this.slotOf.get(expert) ?? -1
  }

  /**
   * Map a token's expert choices onto slots, evicting as needed.
   *
   * Every requested expert is resident on return, so the caller uploads the
   * `misses` and then dispatches. Duplicates inside one request resolve to the
   * same slot and are charged once — a token can route two of its top-K to the
   * same expert, and loading it twice would be a wasted transfer.
   */
  resolve(experts: ArrayLike<number>): PoolResolution {
    const slots = new Int32Array(experts.length)
    const misses: Array<{ expert: number; slot: number }> = []
    const evicted: number[] = []
    // Everything touched by THIS request is un-evictable for the rest of it,
    // or a pool smaller than top-K would evict an expert it is about to use.
    const held = new Set<number>()

    for (let i = 0; i < experts.length; i++) {
      const e = experts[i]
      let slot = this.slotOf.get(e)
      this.requests++
      if (slot !== undefined) {
        this.hits++
      } else {
        slot = this.take(held, evicted)
        this.expertOf[slot] = e
        this.slotOf.set(e, slot)
        misses.push({ expert: e, slot })
      }
      this.usedAt[slot] = ++this.clock
      held.add(e)
      slots[i] = slot
    }
    return { slots, misses, evicted }
  }

  /** A free slot, or the least recently used one that this request has not
   *  already claimed and that is not pinned. */
  private take(held: Set<number>, evicted: number[]): number {
    const f = this.free.pop()
    if (f !== undefined) return f

    let victim = -1
    let oldest = Infinity
    for (let s = 0; s < this.slots; s++) {
      const e = this.expertOf[s]
      if (e < 0) return s
      if (held.has(e) || this.pinned.has(e)) continue
      if (this.usedAt[s] < oldest) { oldest = this.usedAt[s]; victim = s }
    }
    if (victim < 0) {
      // Every slot is either pinned or in use by this very request.
      throw new Error(
        `expert pool of ${this.slots} cannot satisfy a request needing more slots than it has`)
    }
    evicted.push(this.expertOf[victim])
    this.slotOf.delete(this.expertOf[victim])
    return victim
  }

  get hitRate(): number { return this.requests ? this.hits / this.requests : 0 }
}
