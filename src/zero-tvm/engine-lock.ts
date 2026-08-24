/**
 * ENGINE LOCK — one owner at a time for a single-stream engine.
 *
 * The engine has ONE KV cache and one readback ring; two concurrent
 * generatePipelined calls interleave into both and corrupt each other
 * silently (the agent server refuses concurrency for the same reason). The
 * entrance pairs TWO drivers on one engine — the local chat (chat-flow) and
 * the room host (room-host) — so they share this latch: whoever holds it
 * generates, the other waits its turn. Found by the 2026-08-17 lens round.
 *
 * Plain FIFO promise queue; no timeouts (a generation ends or the page dies).
 *
 * acquire() hands back a TOKEN and release() checks it. Without that, a double
 * release with two waiters queued granted the lock to two holders at once —
 * the precise interleave this file exists to prevent, reachable from any
 * caller whose finally runs twice or whose error path also releases. A stale
 * or absent token is refused and logged rather than silently handing the
 * engine to a second driver.
 */

export interface EngineLock {
  /** Resolves when the lock is yours, with the token to release it.
   *  Pair with release(token) in a finally. */
  acquire: () => Promise<number>
  /** Release. Pass the token from acquire(); a stale one is refused. */
  release: (token?: number) => void
  held: () => boolean
  /** Observe holds (e.g. disable a composer while the other driver runs). */
  onChange: (cb: (held: boolean) => void) => void
}

export function makeEngineLock(): EngineLock {
  let locked = false
  /** Monotonic; identifies the CURRENT holder. Every grant mints a new one, so
   *  a release carrying an older token is a bug in the caller, not a handoff. */
  let token = 0
  const waiters: ((t: number) => void)[] = []
  const observers: ((held: boolean) => void)[] = []
  const notify = (): void => { for (const cb of observers) cb(locked) }
  return {
    acquire() {
      if (!locked) {
        locked = true
        token += 1
        notify()
        return Promise.resolve(token)
      }
      return new Promise((resolve) => waiters.push((t) => { resolve(t) }))
    },
    release(t) {
      if (!locked) {
        console.warn('engine-lock: release() with no holder — ignored')
        return
      }
      if (t !== undefined && t !== token) {
        console.warn(`engine-lock: release(${t}) but the holder is ${token} — ignored`)
        return
      }
      const next = waiters.shift()
      if (next) {
        token += 1          // new holder, new token: the old one is now stale
        next(token)
        return              // hand straight to the next in line, still locked
      }
      locked = false
      notify()
    },
    held: () => locked,
    onChange(cb) { observers.push(cb) },
  }
}
