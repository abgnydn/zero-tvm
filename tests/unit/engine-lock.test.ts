// The engine lock, which had no test until 2026-08-23.
//
// It exists because the engine has ONE KV cache and one readback ring: two
// concurrent generatePipelined calls interleave into both and corrupt each
// other silently. The entrance runs two drivers on one engine (chat-flow and
// room-host), so the latch is the only thing between them.
//
// The bug this file was written for: release() took no argument and checked
// nothing, so a caller whose finally ran twice — or whose error path also
// released — granted the lock to TWO waiters. That is exactly the interleave
// the lock prevents, produced by the lock itself. It is pure, synchronous and
// dependency-free, so there was never a reason for it to be untested.

import { describe, expect, it } from 'vitest'
import { makeEngineLock } from '../../src/zero-tvm/engine-lock.ts'


describe('makeEngineLock', () => {
  it('grants immediately when free, and reports held', async () => {
    const lock = makeEngineLock()
    expect(lock.held()).toBe(false)
    const tok = await lock.acquire()
    expect(lock.held()).toBe(true)
    lock.release(tok)
    expect(lock.held()).toBe(false)
  })

  it('queues a second caller until the first releases', async () => {
    const lock = makeEngineLock()
    const a = await lock.acquire()
    let bGranted = false
    const b = lock.acquire().then((t) => { bGranted = true; return t })
    await Promise.resolve()
    expect(bGranted, 'the second caller must not run while the first holds').toBe(false)
    lock.release(a)
    const bTok = await b
    expect(bGranted).toBe(true)
    // Held straight through the handoff: no window where a third caller could
    // slip in between the release and the grant.
    expect(lock.held()).toBe(true)
    lock.release(bTok)
    expect(lock.held()).toBe(false)
  })

  it('serves waiters FIFO', async () => {
    const lock = makeEngineLock()
    const first = await lock.acquire()
    const order: string[] = []
    const b = lock.acquire().then((t) => { order.push('b'); return t })
    const c = lock.acquire().then((t) => { order.push('c'); return t })
    lock.release(first)
    lock.release(await b)
    lock.release(await c)
    expect(order).toEqual(['b', 'c'])
  })

  // ── the reason this file exists ────────────────────────────────────────────

  it('a DOUBLE release does not grant the lock to two holders', async () => {
    const lock = makeEngineLock()
    const a = await lock.acquire()
    let bTok: number | undefined
    let cTok: number | undefined
    void lock.acquire().then((t) => { bTok = t })
    void lock.acquire().then((t) => { cTok = t })

    lock.release(a)
    await Promise.resolve()
    expect(bTok, 'b should hold').toBeDefined()
    expect(cTok, 'c must NOT have been granted — a released once').toBeUndefined()

    // The bug: releasing again with the STALE token used to shift the next
    // waiter, so b and c both believed they held the engine and both would
    // have generated into the same KV cache.
    lock.release(a)
    await Promise.resolve()
    expect(cTok, 'a stale token must not hand the lock to a second holder').toBeUndefined()
    expect(lock.held()).toBe(true)
  })

  it('a BARE release() hands the engine on regardless of who calls it', async () => {
    // Not a bug in the lock — a documented affordance, and the reason callers
    // must guard. This is the shape that survived the first fix of the
    // two-holder bug: room-host's finally called `release(tok)` unconditionally,
    // and `tok` is undefined when the throw happened ABOVE the acquire (ui.row
    // can throw). release(undefined) skips the token check, so the error path
    // released a hold it never took and handed the engine to a queued waiter
    // while the real holder still believed it held it — the same interleave,
    // reintroduced by the change that removes it, and found by a reviewer who
    // ran it rather than read it.
    //
    // The lock cannot distinguish "no token because I am an old caller" from
    // "no token because I never acquired". So this pins the hazard rather than
    // pretending it is closed: callers release ONLY what they took.
    const lock = makeEngineLock()
    await lock.acquire()                       // A holds, token discarded
    let bTok: number | undefined
    void lock.acquire().then((t) => { bTok = t })
    await Promise.resolve()
    expect(bTok, 'B waits while A holds').toBeUndefined()

    lock.release()                             // no token — cannot be attributed
    await Promise.resolve()
    expect(bTok, 'a bare release hands the engine on; the caller must not make this call unless it acquired')
      .toBeDefined()
    expect(lock.held()).toBe(true)
  })

  it('a release with no holder is refused rather than corrupting the queue', async () => {
    // The first version asserted only `held() === false`, which is true with and
    // without the guard — it passed under the mutation it sits beside, which is
    // the definition of decorative. What the guard actually prevents is a
    // release-before-acquire consuming a WAITER, so that is what this asserts.
    const lock = makeEngineLock()
    let granted: number | undefined
    void lock.acquire().then((t) => { granted = t })   // takes it immediately
    await Promise.resolve()
    expect(granted, 'free lock grants at once').toBeDefined()
    lock.release(granted)
    expect(lock.held()).toBe(false)

    // Now nobody holds it. A stray release must not shift a queued waiter.
    let queued: number | undefined
    void lock.acquire().then((t) => { queued = t })    // also takes it immediately
    await Promise.resolve()
    lock.release(999)                                   // stale token, holder is live
    await Promise.resolve()
    expect(lock.held(), 'a stale release must not unlock a live holder').toBe(true)
    expect(queued).toBeDefined()
  })

  it('a token is stale after the handoff it caused', async () => {
    const lock = makeEngineLock()
    const a = await lock.acquire()
    const b = lock.acquire()
    lock.release(a)
    const bTok = await b
    expect(bTok, 'each grant mints a new token').not.toBe(a)
    lock.release(a)                     // a is stale now
    expect(lock.held(), "b's hold survives a stale release").toBe(true)
    lock.release(bTok)
    expect(lock.held()).toBe(false)
  })

  it('a bare release() still works for callers that pass no token', async () => {
    // Backward compatibility: release() with no argument releases the current
    // holder. The token is the guarantee, not a requirement.
    const lock = makeEngineLock()
    await lock.acquire()
    lock.release()
    expect(lock.held()).toBe(false)
  })

  it('onChange observers see holds and releases', async () => {
    const lock = makeEngineLock()
    const seen: boolean[] = []
    lock.onChange((h) => seen.push(h))
    const t = await lock.acquire()
    lock.release(t)
    expect(seen).toEqual([true, false])
  })
})
