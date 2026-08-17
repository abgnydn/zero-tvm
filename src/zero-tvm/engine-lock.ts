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
 */

export interface EngineLock {
  /** Resolves when the lock is yours. Pair with release() in a finally. */
  acquire: () => Promise<void>
  release: () => void
  held: () => boolean
  /** Observe holds (e.g. disable a composer while the other driver runs). */
  onChange: (cb: (held: boolean) => void) => void
}

export function makeEngineLock(): EngineLock {
  let locked = false
  const waiters: (() => void)[] = []
  const observers: ((held: boolean) => void)[] = []
  const notify = (): void => { for (const cb of observers) cb(locked) }
  return {
    acquire() {
      if (!locked) {
        locked = true
        notify()
        return Promise.resolve()
      }
      return new Promise((resolve) => waiters.push(() => { resolve() }))
    },
    release() {
      const next = waiters.shift()
      if (next) { next(); return }   // hand straight to the next in line
      locked = false
      notify()
    },
    held: () => locked,
    onChange(cb) { observers.push(cb) },
  }
}
