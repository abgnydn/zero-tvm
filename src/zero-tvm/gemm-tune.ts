/**
 * GEMM AUTOTUNE — measure, don't assume, which chunk GEMM this device runs
 * fastest; remember the answer per device.
 *
 * The static ladder (pickChunkGemm: e5 > sgmat > matvec) encodes what won on
 * ONE machine (M2 Max, where E5 cleared the in-engine A/B bar). On other
 * GPUs the order can differ — and nothing measures it. This module times the
 * runnable candidates on synthetic gateUp-shaped work at boot and persists
 * the winner, keyed by device + spec.
 *
 * SCOPE, deliberately narrow:
 * - Only the identity-gated ladder candidates (e5/sgmat/matvec). `tiled`
 *   stays explicit-only until a quiet-machine A/B promotes it — a timing
 *   probe cannot take over that decision.
 * - An EXPLICIT chunkGemm bypasses tuning entirely (and keeps its
 *   throw-instead-of-fallback semantics). Bench arms pin their kernels, so
 *   measured numbers stay comparable across runs.
 * - Any probe failure (or a single runnable candidate) falls back to the
 *   static ladder, silently. A tuner that can break boot is worse than no
 *   tuner; a tuner that can only make boot ~1s slower is bounded.
 * - The probe times CONTENT-INDEPENDENT work: zeroed activations through the
 *   real weight buffers. Math content does not affect dispatch timing, and
 *   real weights keep the memory-access pattern faithful (cache/TLB behavior
 *   included) in a way a toy buffer would not.
 */

import type { ModelSpec } from '../compiler/model-spec.ts'
import type { ChunkGemmName } from './engine-core.ts'

/** Bump when a GEMM shader changes shape or speed: stored picks go stale. */
export const TUNE_VERSION = 1
/** Driver updates move performance too — re-tune monthly regardless. */
export const TUNE_TTL_MS = 30 * 24 * 3600 * 1000

export interface TuneStore {
  load(key: string): string | null
  save(key: string, value: string): void
}

export interface TunedPick {
  version: number
  pick: ChunkGemmName
  /** Median ms per candidate at probe time, for the log line. */
  ms: Partial<Record<ChunkGemmName, number>>
  /** Candidates timed; a new runnable one (driver update enabling the matrix
   *  unit) invalidates the pick so it gets measured too. */
  candidates: ChunkGemmName[]
  at: number
}

export function tuneKey(adapterId: string, specId: string): string {
  return `ztvm.gemmTune.v${TUNE_VERSION}.${adapterId}.${specId}`
}

/** Best-effort device identity for the tune key. adapterInfo ships in
 *  current Chromium; anywhere older, stable limits stand in (same GPU,
 *  same limits — collisions across different GPUs are possible but only
 *  cost a re-tune, never correctness). Null when nothing identifies the
 *  device: the caller tunes without persisting. */
export function deviceFingerprint(device: GPUDevice): string | null {
  try {
    const info = (device as unknown as {
      adapterInfo?: { vendor?: string; architecture?: string; device?: string }
    }).adapterInfo
    if (info && (info.vendor || info.architecture || info.device)) {
      return [info.vendor ?? '?', info.architecture ?? '?', info.device ?? '?']
        .join('/').replace(/[^\w./-]/g, '_')
    }
    const l = device.limits
    if (l) {
      return `limits/${l.maxComputeWorkgroupsPerDimension}`
        + `/${l.maxStorageBufferBindingSize}/${l.maxBufferSize}`
    }
  } catch { /* a fingerprint must never break boot */ }
  return null
}

/** Median of a timing sample. */
export function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b)
  const m = s.length >> 1
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2
}

/** Geometric mean across probe sizes of each candidate's median. Lowest
 *  wins; exact ties break toward the static ladder order (the argument order),
 *  so a no-information probe reproduces the ladder instead of coin-flipping. */
export function pickFromTimings(
  medians: Record<string, number[]>,
  ladder: ChunkGemmName[],
): ChunkGemmName {
  let best: ChunkGemmName = ladder[ladder.length - 1]
  let bestScore = Infinity
  for (const name of ladder) {
    const ms = medians[name]
    if (!ms || !ms.length) continue
    const geo = Math.exp(ms.reduce((a, m) => a + Math.log(Math.max(m, 1e-9)), 0) / ms.length)
    // Strictly-less-than keeps the earlier (higher-ladder) candidate on ties.
    if (geo < bestScore) { bestScore = geo; best = name }
  }
  return best
}

/** A stored pick is usable only if it is current-version, fresh, for a pick
 *  that still runs, and measured against a candidate set covering today's
 *  runnable ones. Anything else means re-tune. Pure — unit-tested. */
export function loadTunedPick(
  store: TuneStore | null,
  key: string,
  runnable: ChunkGemmName[],
  now: number = Date.now(),
): ChunkGemmName | null {
  if (!store || !runnable.length) return null
  let raw: string | null = null
  try { raw = store.load(key) } catch { return null }
  if (!raw) return null
  let saved: TunedPick
  try { saved = JSON.parse(raw) as TunedPick } catch { return null }
  if (saved.version !== TUNE_VERSION) return null
  if (typeof saved.at !== 'number' || now - saved.at > TUNE_TTL_MS) return null
  if (!runnable.includes(saved.pick)) return null
  if (!Array.isArray(saved.candidates) || !runnable.every((c) => saved.candidates.includes(c))) return null
  return saved.pick
}

export function saveTunedPick(
  store: TuneStore | null,
  key: string,
  pick: ChunkGemmName,
  ms: Partial<Record<ChunkGemmName, number>>,
  candidates: ChunkGemmName[],
): void {
  if (!store) return
  try {
    store.save(key, JSON.stringify({
      version: TUNE_VERSION, pick, ms, candidates, at: Date.now(),
    } satisfies TunedPick))
  } catch { /* persistence must never break boot */ }
}

/** Probe sizes (chunk rows). Ragged head, mid, full cap — the shapes whose
 *  relative order decides real prefill, not one point. Filtered to what fits
 *  the build's chunk capacity by the caller. */
export const TUNE_MS = [64, 256, 1024]
export const TUNE_WARMUP = 2
export const TUNE_ROUNDS = 4

/**
 * Time each candidate's `run(m)` closure — one recorded GEMM dispatch of M
 * rows — interleaved A,B,C per round so clock drift cancels instead of
 * compounding (the prefill-gemm-ab discipline). Returns per-round ms keyed
 * by candidate then M; aggregate with pickFromTimings (median per size,
 * geomean across sizes). Throws nothing: a candidate that fails to time is
 * dropped, and an empty result means "fall back to the ladder".
 */
export async function timeClosures(
  device: GPUDevice,
  runs: Partial<Record<ChunkGemmName, (m: number) => void>>,
  Ms: number[],
): Promise<Partial<Record<ChunkGemmName, Record<number, number[]>>>> {
  const names = Object.keys(runs) as ChunkGemmName[]
  const out: Partial<Record<ChunkGemmName, Record<number, number[]>>> = {}
  try {
    for (const m of Ms) {
      for (let w = 0; w < TUNE_WARMUP; w++) {
        for (const name of names) runs[name]!(m)
      }
      await device.queue.onSubmittedWorkDone()
      for (let r = 0; r < TUNE_ROUNDS; r++) {
        for (const name of names) {
          const t0 = performance.now()
          runs[name]!(m)
          await device.queue.onSubmittedWorkDone()
          const dt = performance.now() - t0
          ;((out[name] ??= {})[m] ??= []).push(dt)
        }
      }
    }
  } catch { /* timing is advisory; the ladder is the fallback */ }
  return out
}

/** localStorage-backed store, or null outside a browser (Node tunes per boot
 *  without persisting — bench determinism comes from explicit chunkGemm). */
export function browserTuneStore(): TuneStore | null {
  try {
    if (typeof localStorage === 'undefined') return null
    return {
      load: (k) => localStorage.getItem(k),
      save: (k, v) => localStorage.setItem(k, v),
    }
  } catch { return null }
}

/** The full boot-time decision. Returns a runnable pick, or null when the
 *  static ladder should decide (explicit request, single candidate, stored
 *  miss + probe failure — all collapse to the same "you pick"). `runProbe`
 *  is the engine's closure over its real pipelines and weight buffers; this
 *  function owns policy only, which is what the unit tests hold. */
export async function resolveTunedGemm(opts: {
  spec: ModelSpec
  runnable: ChunkGemmName[]
  explicit?: ChunkGemmName
  store?: TuneStore | null
  adapterId?: string | null
  runProbe: () => Promise<Partial<Record<ChunkGemmName, Record<number, number[]>>>>
}): Promise<{ pick: ChunkGemmName; source: 'explicit' | 'stored' | 'probed' | 'ladder' } | null> {
  if (opts.explicit) return { pick: opts.explicit, source: 'explicit' }
  if (opts.runnable.length <= 1) return null
  const store = opts.store ?? null
  const key = opts.adapterId ? tuneKey(opts.adapterId, opts.spec.id) : null
  if (key) {
    const stored = loadTunedPick(store, key, opts.runnable)
    if (stored) return { pick: stored, source: 'stored' }
  }
  let timings: Partial<Record<ChunkGemmName, Record<number, number[]>>> = {}
  try { timings = await opts.runProbe() } catch { timings = {} }
  const medians: Record<string, number[]> = {}
  for (const name of opts.runnable) {
    const perSize = timings[name]
    if (!perSize) continue
    const arr = Object.values(perSize).map(median).filter((m) => Number.isFinite(m))
    if (arr.length) medians[name] = arr
  }
  if (!Object.keys(medians).length) return null
  const order: ChunkGemmName[] = ['e5', 'sgmat', 'tiled', 'matvec']
  const ladder = order.filter((n) => opts.runnable.includes(n))
  const pick = pickFromTimings(medians, ladder.length ? ladder : opts.runnable)
  const ms: Partial<Record<ChunkGemmName, number>> = {}
  for (const [name, arr] of Object.entries(medians)) ms[name as ChunkGemmName] = median(arr)
  if (key) saveTunedPick(store, key, pick, ms, opts.runnable)
  return { pick, source: 'probed' }
}
