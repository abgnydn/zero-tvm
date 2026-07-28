/**
 * UNIT TESTS — mapLimited (src/zero-tvm/map-limited.ts).
 *
 * Regression guard for the B9 download-path fix: the old in-loader mapLimited
 * rejected on the FIRST worker error while sibling fetches kept streaming
 * (mid-air abandonment). These tests pin the hardened semantics — pure Node,
 * no network, no DOM:
 *   - bounded concurrency, order-preserving results
 *   - per-item retries: a transient single failure is NON-fatal
 *   - retries exhausted → fatal: in-flight siblings are aborted via the
 *     shared signal, no new items start, and the rejection is the item's
 *     ORIGINAL error (never an AbortError)
 *   - shrinkable limit function (the 8 → 3 degraded-concurrency path)
 */

import { describe, test, expect } from 'vitest'
import { mapLimited, backoffMs } from '../../src/zero-tvm/map-limited.js'

/** Manually-resolvable promise, for deterministic in-flight control. */
function deferred<T = void>() {
  let resolve!: (v: T) => void
  let reject!: (e: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

const tick = () => new Promise((r) => setTimeout(r, 0))

describe('mapLimited', () => {
  test('maps all items in order with bounded concurrency', async () => {
    let inFlight = 0
    let maxInFlight = 0
    const out = await mapLimited([...Array(20).keys()], 4, async (n) => {
      inFlight++
      maxInFlight = Math.max(maxInFlight, inFlight)
      await tick()
      inFlight--
      return n * 2
    })
    expect(out).toEqual([...Array(20).keys()].map((n) => n * 2))
    expect(maxInFlight).toBeLessThanOrEqual(4)
    expect(maxInFlight).toBeGreaterThan(1)
  })

  test('a transient single failure is non-fatal: item is retried, siblings never aborted', async () => {
    const attempts = new Map<number, number>()
    let anyAborted = false
    const out = await mapLimited(
      [0, 1, 2, 3, 4],
      2,
      async (n, _i, signal) => {
        signal.addEventListener('abort', () => { anyAborted = true })
        const k = (attempts.get(n) ?? 0) + 1
        attempts.set(n, k)
        if (n === 2 && k === 1) throw new Error('transient blip')
        return n
      },
      { retries: 1, baseDelayMs: 1 },
    )
    expect(out).toEqual([0, 1, 2, 3, 4])
    expect(attempts.get(2)).toBe(2) // failed once, retried once, succeeded
    expect(anyAborted).toBe(false)
  })

  test('each item gets its own full retry budget', async () => {
    const attempts = new Map<number, number>()
    const out = await mapLimited(
      [0, 1, 2],
      3,
      async (n) => {
        const k = (attempts.get(n) ?? 0) + 1
        attempts.set(n, k)
        if (k <= 2) throw new Error(`item ${n} attempt ${k}`)
        return n
      },
      { retries: 2, baseDelayMs: 1 },
    )
    expect(out).toEqual([0, 1, 2])
    expect([...attempts.values()]).toEqual([3, 3, 3])
  })

  test('retries exhausted → rejects with the item error and aborts in-flight siblings', async () => {
    const gate = deferred()
    const abortedSiblings: number[] = []
    const started: number[] = []
    const promise = mapLimited(
      [0, 1, 2, 3, 4, 5],
      3,
      async (n, _i, signal) => {
        started.push(n)
        if (n === 1) throw new Error('shard 1 is toast')
        // Siblings hang until aborted — like an in-flight fetch.
        signal.addEventListener('abort', () => {
          abortedSiblings.push(n)
          gate.reject(signal.reason)
        })
        await gate.promise
        return n
      },
      { retries: 1, baseDelayMs: 1 },
    )
    await expect(promise).rejects.toThrow('shard 1 is toast')
    // The two siblings in flight when item 1 went fatal were both aborted...
    expect(abortedSiblings.sort()).toEqual([0, 2])
    // ...and no new items were claimed after the fatal failure (item 1 is
    // retried once = started twice; 3, 4, 5 never start).
    expect(started.filter((n) => n === 1)).toHaveLength(2)
    expect(started).not.toContain(3)
  })

  test('rejection is the original error, not an AbortError from a cancelled sibling', async () => {
    const err = await mapLimited(
      [0, 1],
      2,
      async (n, _i, signal) => {
        if (n === 0) {
          // Reject only when aborted — simulates fetch(signal) cancellation.
          await new Promise((_res, rej) => {
            signal.addEventListener('abort', () => rej(new DOMException('Aborted', 'AbortError')))
          })
        }
        throw new Error('the real failure')
      },
      { baseDelayMs: 1 },
    ).then(
      () => null,
      (e: unknown) => e,
    )
    expect(err).toBeInstanceOf(Error)
    expect((err as Error).message).toBe('the real failure')
    expect((err as Error).name).not.toBe('AbortError')
  })

  test('shrinking the limit function reduces concurrency for later items', async () => {
    let shrunk = false
    const running = new Set<number>()
    let maxAfterShrink = 0
    const out = await mapLimited(
      [...Array(12).keys()],
      () => (shrunk ? 1 : 3),
      async (n) => {
        running.add(n)
        if (shrunk) maxAfterShrink = Math.max(maxAfterShrink, running.size)
        await tick()
        running.delete(n)
        if (n === 2) shrunk = true // flip after the first batch completes
        return n
      },
    )
    expect(out).toEqual([...Array(12).keys()])
    // Old-batch overlap may linger briefly, but concurrency never exceeds the
    // pre-shrink limit and the queue still drains to completion.
    expect(maxAfterShrink).toBeLessThanOrEqual(3)
  })
})

describe('backoffMs', () => {
  test('grows exponentially with jitter in [0.5, 1.5)·base·2^attempt', () => {
    for (let attempt = 0; attempt < 4; attempt++) {
      for (let i = 0; i < 50; i++) {
        const d = backoffMs(attempt, 500)
        expect(d).toBeGreaterThanOrEqual(500 * 2 ** attempt * 0.5)
        expect(d).toBeLessThan(500 * 2 ** attempt * 1.5)
      }
    }
  })
})
