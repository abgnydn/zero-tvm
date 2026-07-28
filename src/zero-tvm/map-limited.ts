/**
 * MAP-LIMITED — bounded-concurrency map with abort + per-item retries.
 *
 * Extracted from weight-loader.ts so the semantics are unit-testable in Node
 * (weight-loader.ts touches GPUBufferUsage at module scope and can't be
 * imported outside a browser).
 *
 * Failure semantics (the B9 fix):
 *   - A worker rejection is NON-FATAL until that item has exhausted its own
 *     retry budget — the item is re-run with exponential backoff + jitter.
 *   - Once an item is out of retries, the failure is fatal: the shared
 *     AbortSignal fires so in-flight siblings can cancel cleanly, no new
 *     items are started, and mapLimited waits for every runner to settle
 *     before rejecting with the ORIGINAL item error (never an AbortError).
 *   - `limit` may be a function, re-read between items — the caller can
 *     shrink concurrency mid-flight (e.g. after a network failure). Runner 0
 *     is exempt so progress is always guaranteed.
 */

/** Exponential backoff with jitter: base·2^attempt scaled by [0.5, 1.5). */
export function backoffMs(attempt: number, base = 500): number {
  return base * Math.pow(2, attempt) * (0.5 + Math.random())
}

export interface MapLimitedOptions {
  /** Extra worker runs per item after its first failure (default 0). */
  retries?: number
  /** Backoff base for per-item retries (default 500ms). */
  baseDelayMs?: number
}

export async function mapLimited<T, R>(
  items: T[],
  limit: number | (() => number),
  worker: (item: T, idx: number, signal: AbortSignal) => Promise<R>,
  opts: MapLimitedOptions = {},
): Promise<R[]> {
  const retries = opts.retries ?? 0
  const baseDelayMs = opts.baseDelayMs ?? 500
  const limitOf = typeof limit === 'function' ? limit : () => limit
  const out: R[] = new Array(items.length)
  const ac = new AbortController()
  let firstError: unknown
  let failed = false
  let next = 0

  const runOne = async (runnerId: number) => {
    // Runner 0 ignores the (possibly shrunken) limit so at least one runner
    // always drains the queue.
    while (!failed && next < items.length && (runnerId === 0 || runnerId < limitOf())) {
      const i = next++
      for (let attempt = 0; ; attempt++) {
        try {
          out[i] = await worker(items[i], i, ac.signal)
          break
        } catch (e) {
          // A sibling already failed and aborted us — swallow, sibling's
          // error is the one that propagates.
          if (ac.signal.aborted) return
          if (attempt < retries) {
            await new Promise((r) => setTimeout(r, backoffMs(attempt, baseDelayMs)))
            if (ac.signal.aborted) return
            continue
          }
          // Out of retries: fatal. Cancel in-flight siblings, stop claiming
          // new items, and record the error for the final rejection.
          failed = true
          firstError = e
          ac.abort(e instanceof Error ? e : new Error(String(e)))
          return
        }
      }
    }
  }

  const runners = Math.max(1, Math.min(limitOf(), items.length))
  // Runners never reject (all errors are captured above), so this waits for
  // every in-flight worker to settle — no abandoned siblings.
  await Promise.all(Array.from({ length: runners }, (_, id) => runOne(id)))
  if (failed) throw firstError instanceof Error ? firstError : new Error(String(firstError))
  return out
}
