import { describe, it, expect } from 'vitest'
import { ExpertPool } from '../../src/zero-tvm/expert-pool.ts'

describe('ExpertPool', () => {
  it('places each expert in a slot and reports it as a miss the first time', () => {
    const p = new ExpertPool(4)
    const r = p.resolve([7, 3])
    expect(r.misses.map((m) => m.expert)).toEqual([7, 3])
    expect(new Set(r.slots)).toHaveProperty('size', 2)
    expect(p.slotFor(7)).toBe(r.slots[0])
    expect(p.hitRate).toBe(0)
  })

  it('hits on a second request for a resident expert', () => {
    const p = new ExpertPool(4)
    p.resolve([1, 2])
    const r = p.resolve([1, 2])
    expect(r.misses).toEqual([])
    expect(p.hitRate).toBe(0.5)      // 2 misses then 2 hits
  })

  it('resolves duplicates within one request to one slot and charges one load', () => {
    // A token can route two of its top-K to the same expert. Loading it twice
    // would be a wasted transfer, which is the whole cost this pool manages.
    const p = new ExpertPool(4)
    const r = p.resolve([5, 5, 5])
    expect(r.misses).toHaveLength(1)
    expect(r.slots[0]).toBe(r.slots[1])
    expect(r.slots[2]).toBe(r.slots[1])
  })

  it('evicts the least recently used expert, not the oldest loaded', () => {
    const p = new ExpertPool(2)
    p.resolve([1])
    p.resolve([2])
    p.resolve([1])            // 1 is now MORE recent than 2
    const r = p.resolve([3])
    expect(r.evicted).toEqual([2])
    expect(p.slotFor(1)).toBeGreaterThanOrEqual(0)
    expect(p.slotFor(2)).toBe(-1)
  })

  it('never evicts an expert the same request is still using', () => {
    // With a pool exactly the size of top-K, a naive LRU evicts the expert it
    // loaded one step earlier in the same token and thrashes forever.
    const p = new ExpertPool(3)
    const r = p.resolve([10, 11, 12])
    expect(new Set(r.slots).size).toBe(3)
    expect(r.evicted).toEqual([])
  })

  it('states a minimum that accounts for pinned slots', () => {
    // The naive floor is the request size. With anything pinned that is wrong,
    // and it only shows up on a token whose experts exclude the pinned one.
    expect(ExpertPool.minSlots(9, 0)).toBe(9)
    expect(ExpertPool.minSlots(9, 1)).toBe(10)
    const tooSmall = new ExpertPool(9, { pin: [99] })
    expect(() => tooSmall.resolve([1, 2, 3, 4, 5, 6, 7, 8, 9])).toThrow(/pinned/)
    const ok = new ExpertPool(ExpertPool.minSlots(9, 1), { pin: [99] })
    expect(() => ok.resolve([1, 2, 3, 4, 5, 6, 7, 8, 9])).not.toThrow()
  })

  it('refuses a request larger than the pool instead of silently corrupting it', () => {
    const p = new ExpertPool(2)
    expect(() => p.resolve([1, 2, 3])).toThrow(/cannot serve/)
  })

  it('keeps a pinned expert resident under pressure', () => {
    // The shared expert runs for every token; evicting it is always wrong.
    const p = new ExpertPool(3, { pin: [99] })
    for (let i = 0; i < 20; i++) p.resolve([99, i])
    expect(p.slotFor(99)).toBeGreaterThanOrEqual(0)
  })

  it('reproduces the trace-measured behaviour: hit rate rises with pool size', () => {
    // Same shape as the real measurement — a fixed working set, requests drawn
    // with locality. A pool at least the working-set size must reach 100%.
    const stream: number[] = []
    for (let i = 0; i < 500; i++) stream.push(i % 8)
    const small = new ExpertPool(4)
    const big = new ExpertPool(8)
    for (const e of stream) { small.resolve([e]); big.resolve([e]) }
    expect(big.hitRate).toBeGreaterThan(small.hitRate)
    expect(big.hitRate).toBeGreaterThan(0.98)
  })
})
