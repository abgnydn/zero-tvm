/**
 * WEIGHT LOADER — Load MLC-layout weights from OPFS / browser Cache API / HuggingFace.
 *
 * No TVM, no WebLLM runtime.
 * Reads MLC's ndarray-cache.json, fetches shard binaries with bounded
 * concurrency + retry/backoff, uploads to GPU as each shard arrives.
 * The model repo + parameter naming come from a ModelSpec (default PHI3).
 *
 * Tiered cache: OPFS (shared with the weight-cache service worker, persisted
 * when granted) → browser Cache API (populated by prior WebLLM sessions) →
 * HuggingFace fetch.
 */

import { PHI3, type ModelSpec } from '../compiler/model-spec.js'
import { openMlxCheckpoint, assembleMlx, planModel, type MlxSource } from './weight-loader-mlx.js'
import { mapLimited, backoffMs } from './map-limited.js'

// ============================================================
// Model URL + cache dir
// ============================================================

/** HuggingFace snapshot base URL for a spec's MLC repo. */
export function modelBaseUrl(spec: ModelSpec): string {
  return `https://huggingface.co/${spec.hfRepo}/resolve/main/`
}

export const PHI3_MODEL_BASE = modelBaseUrl(PHI3)

/**
 * Canonical shared OPFS dir for the Phi-3 weights. The weight-cache service
 * worker pre-seeds this dir and compiler-chat / webllm-bench read it, so all
 * pages download the ~1.8 GB once.
 *
 * KEEP IN SYNC with SHARED_DIR in public/weights-cache-sw.js — the SW is plain
 * JS and cannot import this constant, so the two definitions are paired by hand.
 *
 * Both names are DEFINED in model-registry.ts (the WebGPU-free module) and
 * re-exported here, where every existing caller imports them from: peer weight
 * replication runs on machines with no GPU, and this module reads
 * GPUBufferUsage at import time.
 */
import { WEIGHTS_OPFS_DIR, opfsDirFor } from './model-registry.js'
export { WEIGHTS_OPFS_DIR, opfsDirFor }

/** Max concurrent shard fetches. HTTP/2 multiplexes fine but we don't want to
 *  hold ~2 GB of in-flight ArrayBuffers on low-RAM devices. */
const FETCH_CONCURRENCY = 8

/** Concurrency after the first mid-stream network failure of the session.
 *  HTTP/2 stream resets on HF's Xet CDN correlate with high parallelism, so
 *  once a transfer dies we finish the download at a gentler pace. */
const DEGRADED_FETCH_CONCURRENCY = 3

/** Per-shard network retry budget (attempts = FETCH_RETRIES + 1).
 *  Exponential backoff with jitter: ~0.5s, 1s, 2s, 4s (see backoffMs). */
const FETCH_RETRIES = 4

/** Set on the first mid-stream network failure; sticky for the page session
 *  so a retried loadWeights also runs at the reduced concurrency. */
let networkDegraded = false

// ============================================================
// ndarray-cache.json types
// ============================================================

interface FlatRecord {
  name: string
  shape: number[]
  dtype: string
  format: string
  dataPath: string
  byteOffset: number
  nbytes: number
}

interface ShardGroup {
  dataPath: string
  format: string
  byteOffset: number
  nbytes: number
  records: FlatRecord[]
}

interface NDArrayCache {
  records: (FlatRecord | ShardGroup)[]
}

// ============================================================
// Structured progress
// ============================================================

export interface WeightLoadStats {
  shardsLoaded: number
  totalShards: number
  bytesLoaded: number
  totalBytes: number
  mbPerSec: number
  etaSec: number
  persisted: boolean
}

// ============================================================
// OPFS helpers — persistent per-origin storage, faster than IDB for large blobs.
// Graceful no-op when OPFS is unavailable (Safari, older Firefox).
// ============================================================

type OPFSDir = FileSystemDirectoryHandle | null

interface OpenedOPFS {
  dir: OPFSDir
  persisted: boolean
}

async function openOPFS(dirName: string): Promise<OpenedOPFS> {
  try {
    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) {
      return { dir: null, persisted: false }
    }
    // Best-effort persistence so ~2 GB of weights aren't evicted under disk
    // pressure. Browsers may grant without a prompt for PWA-installed / highly
    // engaged sites; a denial is not fatal.
    let persisted = false
    try {
      persisted = await navigator.storage.persisted?.() ?? false
      if (!persisted) persisted = await navigator.storage.persist?.() ?? false
    } catch { /* no-op */ }
    const root = await navigator.storage.getDirectory()
    // Opportunistic cleanup of old versioned dirs (`zero-tvm-weights-<rev>`)
    // left behind by earlier loader revisions, so stale weights don't hog
    // quota. The `-` in the prefix is load-bearing: it can never match the
    // canonical shared dir the SW writes and other pages read.
    try {
      const stale: string[] = []
      for await (const [name, handle] of (root as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
        if (handle.kind === 'directory' && name.startsWith(`${WEIGHTS_OPFS_DIR}-`)) {
          stale.push(name)
        }
      }
      const remove = (root as unknown as { removeEntry(n: string, o?: { recursive: boolean }): Promise<void> })
      await Promise.all(stale.map((n) => remove.removeEntry(n, { recursive: true })))
    } catch { /* iteration unsupported or in-use — skip */ }
    const dir = await root.getDirectoryHandle(dirName, { create: true })
    return { dir, persisted }
  } catch {
    return { dir: null, persisted: false }
  }
}

function opfsKey(dataPath: string): string {
  // OPFS filenames can't contain '/'. Flatten to a safe ASCII key.
  return dataPath.replace(/[^A-Za-z0-9._-]/g, '_')
}

async function opfsRead(dir: OPFSDir, dataPath: string): Promise<ArrayBuffer | null> {
  if (!dir) return null
  try {
    const fh = await dir.getFileHandle(opfsKey(dataPath))
    const file = await fh.getFile()
    return await file.arrayBuffer()
  } catch {
    return null
  }
}

async function opfsWrite(dir: OPFSDir, dataPath: string, data: ArrayBuffer): Promise<void> {
  if (!dir) return
  try {
    const fh = await dir.getFileHandle(opfsKey(dataPath), { create: true })
    // createWritable is the broadly-supported async writer.
    const writable = await (fh as unknown as { createWritable(): Promise<FileSystemWritableFileStream> }).createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    // OPFS writes are best-effort — failures just mean next visit pays network again.
  }
}

// ============================================================
// Fetch with retry + backoff + mid-stream resume
// ============================================================

/**
 * Fetch a URL's full body with retry, exponential backoff + jitter, and
 * mid-stream resume.
 *
 * The body is streamed inside the retry loop, so a connection that dies
 * partway through the transfer (e.g. the HTTP/2 stream resets HF's Xet CDN
 * produces when ad-blockers block its AWS-WAF telemetry) is retried instead
 * of fatal. Every retry re-requests the ORIGINAL `url` — HF's signed CDN
 * redirect URLs expire, so a resume must re-resolve the redirect — with a
 * Range header to continue from the bytes already received when the server
 * honors it (206). A 200 answer to a Range request restarts cleanly.
 *
 * The first mid-stream failure flips the session-wide `networkDegraded` flag
 * (concurrency drops from FETCH_CONCURRENCY to DEGRADED_FETCH_CONCURRENCY).
 */
async function fetchBufWithRetry(
  url: string,
  signal?: AbortSignal,
  onRetry?: (msg: string) => void,
  retries = FETCH_RETRIES,
): Promise<ArrayBuffer> {
  const leaf = url.split('/').at(-1)
  let chunks: Uint8Array<ArrayBuffer>[] = []
  let received = 0
  let lastErr: unknown
  // A mid-stream / connection failure — degrade session concurrency once.
  const transient = (e: unknown) => {
    lastErr = e
    if (!networkDegraded) {
      networkDegraded = true
      onRetry?.(`network unstable — dropping shard concurrency ${FETCH_CONCURRENCY} → ${DEGRADED_FETCH_CONCURRENCY}`)
    }
  }
  for (let attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      onRetry?.(`[retry ${attempt}/${retries}] ${leaf}` +
        (received > 0 ? ` — resuming at ${(received / 1e6).toFixed(1)} MB` : '') +
        ` (${lastErr})`)
      await new Promise((r) => setTimeout(r, backoffMs(attempt - 1)))
    }
    if (signal?.aborted) throw signal.reason ?? new DOMException('Aborted', 'AbortError')

    let resp: Response
    try {
      const headers: Record<string, string> = {}
      if (received > 0) headers.Range = `bytes=${received}-`
      resp = await fetch(url, { credentials: 'omit', signal, headers })
    } catch (e) {
      if (signal?.aborted) throw e
      // Connection failed before headers. If this was a Range attempt it may
      // be a preflight/proxy that dislikes Range — restart plain next time.
      if (received > 0) { chunks = []; received = 0 }
      transient(e)
      continue
    }
    if (resp.status === 416) {
      // Our resume offset confused the server — restart from scratch.
      chunks = []; received = 0
      lastErr = new Error(`HTTP 416 ${resp.statusText}`)
      continue
    }
    if (!resp.ok) {
      const err = new Error(`HTTP ${resp.status} ${resp.statusText}`)
      // 4xx other than 429/408 is not worth retrying.
      if (resp.status < 500 && resp.status !== 429 && resp.status !== 408) throw err
      lastErr = err
      continue
    }
    // Server ignored our Range (200, or an OPFS-hit from the weight-cache SW)
    // — the body is the whole file, drop the partial bytes.
    if (resp.status !== 206 && received > 0) { chunks = []; received = 0 }
    const lenHeader = resp.headers.get('content-length')
    const expectTotal = lenHeader ? received + Number(lenHeader) : null
    try {
      if (resp.body) {
        const reader = resp.body.getReader()
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value as Uint8Array<ArrayBuffer>)
          received += value.byteLength
        }
      } else {
        const buf = await resp.arrayBuffer()
        chunks.push(new Uint8Array(buf))
        received += buf.byteLength
      }
    } catch (e) {
      if (signal?.aborted) throw e
      transient(e) // mid-stream death — next attempt resumes via Range
      continue
    }
    if (expectTotal !== null && received !== expectTotal) {
      transient(new Error(`truncated body: got ${received} of ${expectTotal} bytes`))
      continue
    }
    const only = chunks.length === 1 ? chunks[0] : null
    if (only && only.byteOffset === 0 && only.byteLength === only.buffer.byteLength) {
      return only.buffer
    }
    const out = new Uint8Array(received)
    let off = 0
    for (const c of chunks) { out.set(c, off); off += c.byteLength }
    return out.buffer
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
}

// ============================================================
// Tiered shard fetch: local mirror (dev) → OPFS → Cache API → HuggingFace
// ============================================================

// In dev, the Vite server mirrors HF hub snapshots under
// /local-weights/<repo-name>/* (see vite.config.ts; the bare /local-weights/*
// form still serves the Phi-3 snapshot for WebLLM's URL shape). This makes
// cold-start e2e testing instant without re-downloading 2 GB over the network.
const DEV = !!(import.meta as unknown as { env?: { DEV?: boolean } }).env?.DEV

/** Dev-mirror base for a spec: /local-weights/<repo-name>/ (null in prod). */
function localMirrorBase(spec: ModelSpec): string | null {
  return DEV ? `/local-weights/${spec.hfRepo.split('/')[1]}/` : null
}

/**
 * Base URL for a spec's SMALL files (tokenizer, templates): the dev mirror when
 * it is primed, the HF repo otherwise. Exists because the tokenizer went
 * straight to HF, which works only when the repo is PUBLISHED — a
 * locally-converted checkpoint (the Qwen3.6 q3-expert build) has no HF repo,
 * and its boot died on a 401 for tokenizer.json. Probes with HEAD so an 11 MB
 * tokenizer is not downloaded twice.
 */
export async function resolveModelBase(spec: ModelSpec, probeFile = 'tokenizer.json'): Promise<string> {
  const mirror = localMirrorBase(spec)
  if (mirror) {
    try {
      if ((await fetch(mirror + probeFile, { method: 'HEAD' })).ok) return mirror
    } catch { /* mirror down — fall through to HF */ }
  }
  return modelBaseUrl(spec)
}

async function fetchShard(
  url: string,
  dataPath: string,
  mirrorBase: string | null,
  opfs: OPFSDir,
  cacheStores: Cache[],
  pendingWrites: Promise<void>[],
  onProgress?: (msg: string) => void,
  signal?: AbortSignal,
): Promise<ArrayBuffer> {
  const leaf = url.split('/').at(-1)
  const cache = (buf: ArrayBuffer) => { pendingWrites.push(opfsWrite(opfs, dataPath, buf)) }

  // Tier 0: dev-only local mirror served by Vite from .weights-local / HF cache
  if (mirrorBase) {
    try {
      const resp = await fetch(mirrorBase + dataPath, { signal })
      if (resp.ok) {
        onProgress?.(`[local] ${leaf}`)
        const buf = await resp.arrayBuffer()
        cache(buf)
        return buf
      }
    } catch { /* mirror not primed — fall through */ }
  }

  // Tier 1: OPFS (already on disk — no re-cache needed)
  const fromOPFS = await opfsRead(opfs, dataPath)
  if (fromOPFS) {
    onProgress?.(`[opfs] ${leaf}`)
    return fromOPFS
  }

  // Tier 2: browser Cache API (WebLLM's tvmjs populates this). Stores are
  // opened once up-front so we don't re-scan caches.keys() per shard.
  for (const store of cacheStores) {
    try {
      const resp = await store.match(url)
      if (resp) {
        onProgress?.(`[cache] ${leaf}`)
        const buf = await resp.arrayBuffer()
        cache(buf)
        return buf
      }
    } catch { /* store closed / unreachable — continue */ }
  }

  // Tier 3: network (retry/backoff + mid-stream Range resume)
  onProgress?.(`[fetch] ${leaf}`)
  const buf = await fetchBufWithRetry(url, signal, onProgress)
  cache(buf)
  return buf
}

/**
 * Ranged reads over the checkpoint's shards, with the same retry/backoff the
 * shard fetcher uses.
 *
 * The Cache API tier is skipped deliberately: it is populated by WebLLM's
 * tvmjs for MLC repos and is a guaranteed miss for a safetensors checkpoint,
 * so probing it 2841 times would only cost time. OPFS caching happens a level
 * up, keyed by BUILT buffer rather than by shard.
 */
function mlxSource(
  baseUrl: string,
  mirrorBase: string | null,
  onProgress?: (msg: string) => void,
  signal?: AbortSignal,
): MlxSource {
  const url = (file: string) => (mirrorBase ?? baseUrl) + file
  return {
    whole: async (file) => new Uint8Array(await fetchBufWithRetry(url(file), signal, onProgress)),
    range: async (file, begin, end) => {
      let lastErr: unknown
      for (let attempt = 0; attempt <= FETCH_RETRIES; attempt++) {
        if (attempt > 0) await new Promise((r) => setTimeout(r, backoffMs(attempt - 1)))
        try {
          const resp = await fetch(url(file), {
            credentials: 'omit',
            signal,
            headers: { Range: `bytes=${begin}-${end - 1}` },
          })
          // A server that ignores Range answers 200 with the WHOLE file — which
          // for a 5.3 GB shard is exactly the allocation this design exists to
          // avoid. Refuse rather than OOM.
          if (resp.status === 200) {
            throw new Error(`${file}: server ignored Range (200, not 206) — cannot stream a multi-GB shard`)
          }
          if (!resp.ok) throw new Error(`${file}: HTTP ${resp.status}`)
          const buf = new Uint8Array(await resp.arrayBuffer())
          if (buf.byteLength !== end - begin) {
            throw new Error(`${file}: asked for ${end - begin} bytes, got ${buf.byteLength}`)
          }
          return buf
        } catch (e) {
          if (signal?.aborted) throw e
          lastErr = e
          onProgress?.(`[retry ${attempt + 1}/${FETCH_RETRIES}] ${file} ${begin}-${end} (${e})`)
        }
      }
      throw new Error(`range fetch failed after ${FETCH_RETRIES} retries: ${lastErr}`)
    },
  }
}

async function openAllCacheStores(): Promise<Cache[]> {
  try {
    if (typeof caches === 'undefined') return []
    const names = await caches.keys()
    return await Promise.all(names.map((n) => caches.open(n)))
  } catch {
    return []
  }
}

// ============================================================
// Flatten all records from ndarray-cache.json
// ============================================================

function flattenRecords(cache: NDArrayCache): FlatRecord[] {
  const out: FlatRecord[] = []
  for (const rec of cache.records) {
    if ('records' in rec && Array.isArray(rec.records)) {
      // Nested shard format
      for (const r of rec.records) {
        out.push({ ...r, dataPath: r.dataPath ?? rec.dataPath })
      }
    } else {
      out.push(rec as FlatRecord)
    }
  }
  return out
}

// ============================================================
// Main loader — bounded parallel shard fetch, streaming GPU upload
// ============================================================

const USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function opfsStatusLabel(opfs: OPFSDir, persisted: boolean): string {
  if (!opfs) return 'OPFS unavailable'
  return persisted ? 'OPFS persistent' : 'OPFS best-effort'
}

export function formatEta(sec: number): string {
  if (!isFinite(sec) || sec <= 0) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec - m * 60)
  return `${m}m ${s}s`
}

function uploadRecord(device: GPUDevice, shard: ArrayBuffer, rec: FlatRecord): GPUBuffer {
  // MLC's "f32-to-bf16" shard format stores float32 tensors as bf16 (the
  // upper 16 bits of each f32) — nbytes is the STORED size (numel × 2).
  // Expand to real f32 on upload: the GDN gate kernel binds A_log/dt_bias as
  // array<f32> (they must stay f32 — -exp(A_log) can overflow in f16).
  // float16/uint32 records in the same shards are stored raw.
  const bf16AsF32 = rec.dtype === 'float32' && rec.format === 'f32-to-bf16'
  const uploadBytes = bf16AsF32 ? rec.nbytes * 2 : rec.nbytes
  // A record larger than the device's storage-binding ceiling uploads fine but
  // fails bind-group creation later, which invalidates every submit touching
  // it — the engine then runs "successfully" with all-zero outputs. Fail loudly
  // at load time instead (fix: raise requiredLimits at requestDevice(), see
  // bootEngine in loading-ui.ts / tests/kernels/gpu.mjs).
  if (uploadBytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error(
      `Weight record '${rec.name}' (${uploadBytes} bytes) exceeds ` +
      `device.limits.maxStorageBufferBindingSize (${device.limits.maxStorageBufferBindingSize}). ` +
      `Request a higher requiredLimits.maxStorageBufferBindingSize at device creation.`
    )
  }
  const gpuBuf = device.createBuffer({
    size: Math.max(uploadBytes, 4),
    usage: USAGE,
    label: rec.name,
  })
  if (bf16AsF32) {
    const src = new Uint16Array(shard, rec.byteOffset, rec.nbytes / 2)
    const expanded = new Uint32Array(src.length)
    for (let i = 0; i < src.length; i++) expanded[i] = src[i] << 16  // bf16 → f32
    device.queue.writeBuffer(gpuBuf, 0, expanded)
    return gpuBuf
  }
  // Uint8Array view avoids the ArrayBuffer.slice() copy before writeBuffer.
  device.queue.writeBuffer(gpuBuf, 0, new Uint8Array(shard, rec.byteOffset, rec.nbytes))
  return gpuBuf
}

export async function loadWeights(
  device: GPUDevice,
  onProgress?: (msg: string) => void,
  onStats?: (stats: WeightLoadStats) => void,
  spec: ModelSpec = PHI3,
  /**
   * Load ONE PIPELINE STAGE — layers [start, end) plus the embedding only if
   * it starts the model and the lm_head / final norm only if it ends it. This
   * is what makes a split worth doing: the stage's VRAM is its share, not the
   * whole checkpoint. MLX checkpoints only — the MLC path fetches whole shards,
   * so skipping layers would save no download.
   */
  layerRange?: { start: number; end: number },
): Promise<LoadedWeights> {
  if (layerRange && spec.weightFormat !== 'mlx-safetensors') {
    throw new Error(`loadWeights: layerRange needs an MLX checkpoint; ${spec.id} ships MLC shards`)
  }
  const baseUrl = modelBaseUrl(spec)
  const opfsDir = opfsDirFor(spec)
  const mirrorBase = localMirrorBase(spec)

  // Manifest — newer MLC repos renamed ndarray-cache.json → tensor-cache.json
  // (the spec says which name its repo ships).
  const manifestName = spec.manifestName ?? 'ndarray-cache.json'
  onProgress?.(`Loading ${manifestName}…`)
  const { dir: opfs, persisted } = await openOPFS(opfsDir)
  onProgress?.(opfs
    ? `OPFS ready (${opfsDir}, ${persisted ? 'persistent' : 'best-effort'})`
    : 'OPFS unavailable')
  const cacheStores = await openAllCacheStores()
  if (cacheStores.length > 0) onProgress?.(`Cache API: ${cacheStores.length} store(s) open`)

  // Tracks fire-and-forget OPFS writes so loadWeights doesn't resolve while
  // shards are still being persisted (would leave a torn cache on tab close).
  const pendingWrites: Promise<void>[] = []

  // ── MLX safetensors path ─────────────────────────────────────────────────
  // Branches HERE, after the OPFS/Cache setup above (which is format-agnostic
  // and stays single-sourced) and before the manifest read below, which is
  // MLC-shaped. The unit of work is a BufferPlan, never a shard — see
  // weight-loader-mlx.ts for why a 5.3 GB shard cannot be one ArrayBuffer.
  if (spec.weightFormat === 'mlx-safetensors') {
    const src = mlxSource(baseUrl, mirrorBase, onProgress)
    onProgress?.('Reading safetensors headers…')
    const { locate, shards } = await openMlxCheckpoint(src)
    const total = planModel(spec, layerRange).length
    onProgress?.(`${shards.length} shards, ${total} buffers to build`)
    const t0 = performance.now()
    const lw = await assembleMlx(device, spec, src, locate, USAGE, {
      onProgress,
      cacheRead: (key) => opfsRead(opfs, key),
      // Not when serving from the dev mirror: the bytes are already on local
      // disk and OPFS-caching them would write a second 19.5 GB copy.
      cacheWrite: mirrorBase ? undefined
        : (key, data) => { pendingWrites.push(opfsWrite(opfs, key, data)) },
      onBuffer: (done, n, bytes) => {
        // Every 25th, so a 947-buffer load does not spend its time on strings.
        if (done % 25 !== 0 && done !== n) return
        const el = (performance.now() - t0) / 1000
        // Buffers stand in for shards here — same progress semantics, different
        // unit, because a plan is what this loader actually completes.
        const est = bytes / (done / n)
        onProgress?.(`${done}/${n} buffers — ${(bytes / 1e9).toFixed(2)} GB in ${el.toFixed(0)}s`)
        onStats?.({
          shardsLoaded: done, totalShards: n,
          bytesLoaded: bytes, totalBytes: Math.round(est),
          mbPerSec: el > 0 ? bytes / 1e6 / el : 0,
          etaSec: done > 0 ? (el / done) * (n - done) : 0,
          persisted,
        })
      },
    }, layerRange)
    await Promise.all(pendingWrites)
    onProgress?.(`Weights ready (${((performance.now() - t0) / 1000).toFixed(0)}s)`)
    return lw as unknown as LoadedWeights
  }

  const manifestBytes = await fetchShard(
    baseUrl + manifestName,
    manifestName,
    mirrorBase,
    opfs,
    cacheStores,
    pendingWrites,
    onProgress,
  )
  const manifest: NDArrayCache = JSON.parse(new TextDecoder().decode(manifestBytes))

  const allRecords = flattenRecords(manifest)
  onProgress?.(`Manifest: ${allRecords.length} parameters`)

  // Group records by dataPath so each shard is fetched exactly once.
  const byShard = new Map<string, FlatRecord[]>()
  for (const r of allRecords) {
    const existing = byShard.get(r.dataPath)
    if (existing) existing.push(r)
    else byShard.set(r.dataPath, [r])
  }

  // Bounded parallel shard fetch + per-shard GPU upload as they arrive.
  const gpuBuffers = new Map<string, GPUBuffer>()
  const totalShards = byShard.size
  const totalBytes = allRecords.reduce((s, r) => s + r.nbytes, 0)
  const startMs = performance.now()
  let shardsLoaded = 0
  let bytesLoaded = 0

  const shardEntries = [...byShard.entries()]

  try {
    // Concurrency is re-read between shards so the first mid-stream network
    // failure (networkDegraded) drops parallelism for the rest of the session.
    // Per-shard failures are non-fatal until that shard exhausts its own
    // retries (FETCH_RETRIES+1 network attempts inside fetchBufWithRetry, ×2
    // full tier walks via mapLimited's retries) — only then does the shared
    // AbortSignal cancel the sibling fetches still in flight.
    await mapLimited(
      shardEntries,
      () => (networkDegraded ? DEGRADED_FETCH_CONCURRENCY : FETCH_CONCURRENCY),
      async ([dataPath, records], _idx, signal) => {
        let shard: ArrayBuffer
        try {
          shard = await fetchShard(baseUrl + dataPath, dataPath, mirrorBase, opfs, cacheStores, pendingWrites, onProgress, signal)
        } catch (e) {
          if (signal.aborted) throw e
          throw new Error(`Shard '${dataPath}' failed: ${e instanceof Error ? e.message : e}`)
        }
        for (const rec of records) {
          gpuBuffers.set(rec.name, uploadRecord(device, shard, rec))
        }
        shardsLoaded++
        bytesLoaded += shard.byteLength
        const elapsedSec = (performance.now() - startMs) / 1000
        const mbPerSec = (bytesLoaded / 1e6) / Math.max(elapsedSec, 0.1)
        const etaSec = mbPerSec > 0 ? (totalBytes - bytesLoaded) / (mbPerSec * 1e6) : 0
        const mb = (bytesLoaded / 1e6).toFixed(0)
        const total = (totalBytes / 1e6).toFixed(0)
        onProgress?.(`[${shardsLoaded}/${totalShards}] ${dataPath} · ${mb}/${total} MB · ${mbPerSec.toFixed(1)} MB/s · ETA ${formatEta(etaSec)}`)
        onStats?.({ shardsLoaded, totalShards, bytesLoaded, totalBytes, mbPerSec, etaSec, persisted })
      },
      { retries: 1 },
    )
  } catch (e) {
    // Make "partial progress is cached" true before reporting it: completed
    // shards' OPFS writes are already in flight — let them land.
    await Promise.allSettled(pendingWrites)
    const msg = e instanceof Error ? e.message : String(e)
    throw new Error(
      `${msg} — ${shardsLoaded}/${totalShards} shards (${(bytesLoaded / 1e6).toFixed(0)} MB) are safely cached (OPFS); ` +
      `reloading resumes from cache and only re-downloads what's missing. ` +
      `If you use an ad/privacy blocker, allow huggingface.co and hf.co — ` +
      `HF's bot protection can kill downloads when blocked.`,
    )
  }

  function find(...candidates: string[]): GPUBuffer {
    for (const c of candidates) {
      const b = gpuBuffers.get(c)
      if (b) return b
    }
    throw new Error(
      `Weight not found. Tried: ${candidates.join(', ')}\n` +
        `Available: ${[...gpuBuffers.keys()].slice(0, 20).join(', ')}`,
    )
  }

  // Resolve the weight layout this engine expects — candidate names come from
  // the spec's paramNaming table (first match wins, same contract as before).
  const N = spec.paramNaming
  const embdWeights = find(...N.embedWeight)
  const embdScales = find(...N.embedScale)
  const initNormGamma = find(...N.layer(0).norm1)
  const lmHeadWeights = find(...N.lmHeadWeight)
  const lmHeadScales = find(...N.lmHeadScale)
  const finalNormGamma = find(...N.finalNorm)

  // GDN input-projection packing (hybrid specs): in_proj_qkv ‖ z ‖ a ‖ b are
  // four separate K=d int4 records; concatenating their rows (q4f16_1 is
  // row-major — [rows, K/8] u32 weights, [rows, K/32] f16 scales, so row
  // concatenation IS byte concatenation) yields one gdnProjRows-row matmul
  // the engine runs as a single dispatch. Packed on the GPU with
  // copyBufferToBuffer (queue-ordered after the writeBuffer uploads); the
  // four source buffers are destroyed after the copies are submitted.
  const packEnc = device.createCommandEncoder({ label: 'gdnProjPack' })
  const packDoomed: GPUBuffer[] = []
  function packGdnProj(rec: Array<{ buf: GPUBuffer; rows: number }>, bytesPerRow: number, label: string): GPUBuffer {
    const totalRows = rec.reduce((s, r) => s + r.rows, 0)
    const packed = device.createBuffer({ size: totalRows * bytesPerRow, usage: USAGE, label })
    let off = 0
    for (const r of rec) {
      packEnc.copyBufferToBuffer(r.buf, 0, packed, off, r.rows * bytesPerRow)
      off += r.rows * bytesPerRow
      packDoomed.push(r.buf)
    }
    return packed
  }

  const layers: LoadedWeights['layers'] = []
  for (let L = 0; L < spec.layers; L++) {
    const n = N.layer(L)
    // Hybrid specs (Qwen3.5): self_attn records exist only on attention
    // layers, linear_attn records only on GDN layers — resolve per
    // layerKinds[L]. Pure-attention specs have layerKinds all 'attn'.
    const kind = spec.layerKinds[L]
    layers.push({
      normGamma1: find(...n.norm1),
      normGamma2: find(...n.norm2),
      ffnWeights: find(...n.ffnWeight),
      ffnScales: find(...n.ffnScale),
      ffnDownWeights: find(...n.ffnDownWeight),
      ffnDownScales: find(...n.ffnDownScale),
      ...(kind === 'attn' ? {
        qkvWeights: find(...n.qkvWeight),
        qkvScales: find(...n.qkvScale),
        oProjWeights: find(...n.oProjWeight),
        oProjScales: find(...n.oProjScale),
        // Per-head q/k RMSNorm gammas (Qwen3) — only when the naming declares them.
        qNormGamma: n.qNorm ? find(...n.qNorm) : undefined,
        kNormGamma: n.kNorm ? find(...n.kNorm) : undefined,
      } : {
        gdn: {
          // Fused input projection: rows [qkv | z | a | b] (see packGdnProj).
          projWeights: packGdnProj([
            { buf: find(...n.gdnQkvWeight!), rows: spec.gdnQkvDim },
            { buf: find(...n.gdnZWeight!), rows: spec.gdnVDim },
            { buf: find(...n.gdnAWeight!), rows: spec.gdnVHeads },
            { buf: find(...n.gdnBWeight!), rows: spec.gdnVHeads },
          ], spec.dPacked * 4, `gdnProjWeights_${L}`),
          projScales: packGdnProj([
            { buf: find(...n.gdnQkvScale!), rows: spec.gdnQkvDim },
            { buf: find(...n.gdnZScale!), rows: spec.gdnVDim },
            { buf: find(...n.gdnAScale!), rows: spec.gdnVHeads },
            { buf: find(...n.gdnBScale!), rows: spec.gdnVHeads },
          ], spec.dScales * 2, `gdnProjScales_${L}`),
          aLog: find(...n.gdnALog!),        // f32 (bf16-expanded by uploadRecord)
          dtBias: find(...n.gdnDtBias!),    // f32 (bf16-expanded by uploadRecord)
          convWeight: find(...n.gdnConvWeight!),  // f16 [C,1,K] — channel-major flat, gdn_conv's layout
          normGamma: find(...n.gdnNormWeight!),
          outWeights: find(...n.gdnOutWeight!),
          outScales: find(...n.gdnOutScale!),
        },
      }),
    })
  }

  // Submit the GDN packing copies (a no-op encoder for pure-attention specs)
  // and release the unpacked source buffers — WebGPU defers the actual free
  // until the submitted copies complete.
  device.queue.submit([packEnc.finish()])
  for (const b of packDoomed) b.destroy()

  // Drain any in-flight OPFS writes so cache is fully persisted before we hand
  // control back. Errors here are non-fatal — opfsWrite already swallows them.
  if (pendingWrites.length > 0) await Promise.allSettled(pendingWrites)

  const totalSec = (performance.now() - startMs) / 1000
  const avgMbPerSec = (bytesLoaded / 1e6) / Math.max(totalSec, 0.1)
  onProgress?.(`All weights loaded · ${(totalBytes / 1e6).toFixed(0)} MB in ${totalSec.toFixed(1)}s · avg ${avgMbPerSec.toFixed(1)} MB/s · ${opfsStatusLabel(opfs, persisted)}`)

  return {
    device,
    embdWeights, embdScales,
    lmHeadWeights, lmHeadScales,
    initNormGamma,
    finalNormGamma,
    layers,
  }
}

/**
 * AFFINE CHECKPOINTS carry a per-group bias beside every scale (`w = s·q + b`),
 * which the symmetric MLC layout has no analogue for. Every `*Biases` field
 * below is present exactly when `spec.weightFormat === 'mlx-safetensors'` and
 * absent otherwise, and every affine bind group binds it at @binding(5) — so a
 * missing one is a bind-group validation error, not a wrong number.
 */
export interface LoadedWeights {
  device: GPUDevice
  embdWeights: GPUBuffer
  embdScales: GPUBuffer
  embdBiases?: GPUBuffer
  lmHeadWeights: GPUBuffer
  lmHeadScales: GPUBuffer
  lmHeadBiases?: GPUBuffer
  initNormGamma: GPUBuffer     // layer 0 input_layernorm
  finalNormGamma: GPUBuffer    // model.norm (after all layers)
  layers: Array<{
    // Attention-layer records — present iff spec.layerKinds[L] === 'attn'
    // (always, for pure-attention specs like Phi-3/Qwen3).
    qkvWeights?: GPUBuffer
    qkvScales?: GPUBuffer
    qkvBiases?: GPUBuffer
    oProjWeights?: GPUBuffer
    oProjScales?: GPUBuffer
    oProjBiases?: GPUBuffer
    normGamma1: GPUBuffer    // input_layernorm (pre-attention)
    normGamma2: GPUBuffer    // post_attention_layernorm (pre-FFN)
    // Dense FFN — absent on a MoE spec, which has `moe` below instead.
    ffnWeights?: GPUBuffer
    ffnScales?: GPUBuffer
    ffnBiases?: GPUBuffer
    ffnDownWeights?: GPUBuffer
    ffnDownScales?: GPUBuffer
    ffnDownBiases?: GPUBuffer
    qNormGamma?: GPUBuffer   // per-head q RMSNorm over head_dim (Qwen3 only)
    kNormGamma?: GPUBuffer   // per-head k RMSNorm over head_dim (Qwen3 only)
    // GatedDeltaNet records — present iff spec.layerKinds[L] === 'gdn' (Qwen3.5).
    gdn?: {
      /** Fused input projection int4 [gdnProjRows, d] — rows are
       *  in_proj_qkv ‖ in_proj_z ‖ in_proj_a ‖ in_proj_b, packed by the
       *  loader so the engine runs ONE matmul dispatch per GDN layer. */
      projWeights: GPUBuffer
      projScales: GPUBuffer
      projBiases?: GPUBuffer
      aLog: GPUBuffer        // A_log f32 [gdnVHeads]
      dtBias: GPUBuffer      // dt_bias f32 [gdnVHeads]
      convWeight: GPUBuffer  // conv1d_weight f16 [gdnQkvDim * gdnConvK]
      normGamma: GPUBuffer   // gated-RMSNorm gamma f16 [gdnHeadV]
      outWeights: GPUBuffer  // out_proj int4 [d, gdnVDim]
      outScales: GPUBuffer
      outBiases?: GPUBuffer
    }
    /** Multi-head latent attention — present iff spec.mla (DeepSeek-V2). The
     *  pe rows of both projections are ALREADY de-interleaved by the loader, so
     *  the runtime rotates them with the ordinary half-split RoPE. */
    mla?: {
      qWeights: GPUBuffer       // int4 affine [heads*(nope+rope), d], pe rows permuted
      qScales: GPUBuffer
      qBiases: GPUBuffer
      kvaWeights: GPUBuffer     // int4 affine [kvLora+rope, d], pe rows permuted
      kvaScales: GPUBuffer
      kvaBiases: GPUBuffer
      kvaNormGamma: GPUBuffer   // kv_a_layernorm gamma f16 [kvLora]
      /** kv_b_proj DEQUANTIZED to f16: heads blocks of [kvLora][nope] (K
       *  transposed) followed by heads blocks of [vHeadDim][kvLora]. One buffer,
       *  bound as two regions — the V region starts at heads*kvLora*nope*2 B,
       *  which must be computed from spec.mla and never hardcoded. */
      kvbF16: GPUBuffer
    }
    /** Sparse-MoE FFN — present iff spec.moe. Every stacked tensor carries the
     *  SHARED expert at index spec.sharedExpertIndex, and the router carries
     *  its gate as row E, so the block has no special case for it. */
    moe?: {
      routerWeights: GPUBuffer   // int8 affine [experts+1, d]
      routerScales: GPUBuffer
      routerBiases: GPUBuffer
      gateWeights: GPUBuffer     // int4 affine [experts+1, moeIntermediate, d]
      gateScales: GPUBuffer
      gateBiases: GPUBuffer
      upWeights: GPUBuffer
      upScales: GPUBuffer
      upBiases: GPUBuffer
      downWeights: GPUBuffer     // int4 affine [experts+1, d, moeIntermediate]
      downScales: GPUBuffer
      downBiases: GPUBuffer
    }
  }>
}
