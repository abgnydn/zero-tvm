/**
 * UNIT TESTS — degraded-concurrency tracker + backoff (src/zero-tvm/map-limited.ts).
 *
 * The weight loader drops shard concurrency 8 → 3 on the first mid-stream
 * failure and recovers after ten consecutive clean fetches. That policy used
 * to live in weight-loader.ts module state — next to GPUBufferUsage reads
 * that make the module unimportable in Node — so no test could touch it.
 * createDegradedTracker carries the identical policy in the import-safe
 * module; these tests pin it, including the streak-reset edge the old code
 * got wrong twice (first as a never-resetting flag, then as a streak the
 * failure path never zeroed).
 */

import { describe, test, expect } from 'vitest'
import { backoffMs, createDegradedTracker } from '../../src/zero-tvm/map-limited.js'

describe('backoffMs', () => {
  test('stays inside [0.5, 1.5) × base·2^attempt', () => {
    for (let attempt = 0; attempt < 6; attempt++) {
      for (let i = 0; i < 200; i++) {
        const ms = backoffMs(attempt)
        expect(ms).toBeGreaterThanOrEqual(0.5 * 500 * 2 ** attempt)
        expect(ms).toBeLessThan(1.5 * 500 * 2 ** attempt)
      }
    }
  })

  test('honors a custom base', () => {
    for (let i = 0; i < 100; i++) {
      const ms = backoffMs(2, 100)
      expect(ms).toBeGreaterThanOrEqual(200)
      expect(ms).toBeLessThan(600)
    }
  })

  test('means double per attempt: E[jitter] is 1.0', () => {
    // Bands overlap ([0.5,1.5) spans 3x against 2x growth), so per-sample
    // ordering is NOT guaranteed — but the mean of attempt a+1 is 2x the
    // mean of attempt a. 2000 samples make ±10% watertight (SE ≈ 0.6%).
    for (let attempt = 0; attempt < 4; attempt++) {
      let sum = 0
      const N = 2000
      for (let i = 0; i < N; i++) sum += backoffMs(attempt)
      const mean = sum / N
      const expected = 500 * 2 ** attempt
      expect(mean).toBeGreaterThan(expected * 0.9)
      expect(mean).toBeLessThan(expected * 1.1)
    }
  })
})

describe('createDegradedTracker', () => {
  test('starts healthy at full concurrency, successes are no-ops', () => {
    const t = createDegradedTracker({ full: 8, degraded: 3 })
    expect(t.degraded).toBe(false)
    expect(t.limitOf()).toBe(8)
    const notes: string[] = []
    t.markSuccess((m) => notes.push(m))
    expect(t.degraded).toBe(false)
    expect(notes).toEqual([])
  })

  test('first transient degrades once and announces', () => {
    const t = createDegradedTracker({ full: 8, degraded: 3 })
    const notes: string[] = []
    t.markTransient((m) => notes.push(m))
    expect(t.degraded).toBe(true)
    expect(t.limitOf()).toBe(3)
    expect(notes).toHaveLength(1)
    expect(notes[0]).toContain('8 → 3')
    // A second failure while degraded announces nothing new.
    t.markTransient((m) => notes.push(m))
    expect(notes).toHaveLength(1)
  })

  test('nine clean fetches do not recover; the tenth does', () => {
    const t = createDegradedTracker({ full: 8, degraded: 3 })
    const notes: string[] = []
    t.markTransient()
    for (let i = 0; i < 9; i++) t.markSuccess((m) => notes.push(m))
    expect(t.degraded).toBe(true)
    expect(t.limitOf()).toBe(3)
    expect(notes).toEqual([])
    t.markSuccess((m) => notes.push(m))
    expect(t.degraded).toBe(false)
    expect(t.limitOf()).toBe(8)
    expect(notes).toHaveLength(1)
    expect(notes[0]).toContain('3 → 8')
  })

  test('a failure zeroes the streak: nine + failure + nine stays degraded', () => {
    const t = createDegradedTracker({ full: 8, degraded: 3 })
    t.markTransient()
    for (let i = 0; i < 9; i++) t.markSuccess()
    t.markTransient()
    for (let i = 0; i < 9; i++) t.markSuccess()
    expect(t.degraded).toBe(true)
    t.markSuccess()
    expect(t.degraded).toBe(false)
  })

  test('recovery resets the streak: the next degradation needs ten again', () => {
    // Strictly, the reset is hygiene, not a behavior change: every degraded
    // episode begins with markTransient (the only path that sets degraded),
    // which zeroes the streak first — so the counter is 0 at each episode
    // start with or without the reset. It is kept because the invariant
    // ("streak counts clean fetches in the CURRENT episode") should hold
    // structurally, not incidentally via call order. This test pins the
    // ten-again behavior against a future refactor that breaks that order.
    const t = createDegradedTracker({ full: 8, degraded: 3 })
    t.markTransient()
    for (let i = 0; i < 10; i++) t.markSuccess()
    expect(t.degraded).toBe(false)
    t.markTransient()
    t.markSuccess()
    expect(t.degraded).toBe(true)
    for (let i = 0; i < 9; i++) t.markSuccess()
    expect(t.degraded).toBe(false)
  })

  test('recoverAfter is configurable', () => {
    const t = createDegradedTracker({ full: 4, degraded: 1, recoverAfter: 2 })
    t.markTransient()
    t.markSuccess()
    expect(t.degraded).toBe(true)
    t.markSuccess()
    expect(t.degraded).toBe(false)
    expect(t.limitOf()).toBe(4)
  })
})
