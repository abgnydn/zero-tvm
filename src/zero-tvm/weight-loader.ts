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
 */
export const WEIGHTS_OPFS_DIR = 'zero-tvm-weights'

/**
 * Per-model OPFS dir. Phi-3 keeps the historical unsuffixed name for cache
 * compatibility; other models get a `.`-joined suffix. The separator MUST NOT
 * be `-`: openOPFS() deletes stale `zero-tvm-weights-<rev>` dirs left by old
 * loader revisions, and a dash-suffixed model dir would match that sweep.
 */
export function opfsDirFor(spec: ModelSpec): string {
  return spec.id === PHI3.id ? WEIGHTS_OPFS_DIR : `${WEIGHTS_OPFS_DIR}.${spec.id}`
}

/** Max concurrent shard fetches. HTTP/2 multiplexes fine but we don't want to
 *  hold ~2 GB of in-flight ArrayBuffers on low-RAM devices. */
const FETCH_CONCURRENCY = 8

/** Per-shard retry budget. Exponential backoff: 500ms, 1.5s, 4.5s. */
const FETCH_RETRIES = 3

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
// Fetch with retry + backoff
// ============================================================

async function fetchWithRetry(url: string, retries = FETCH_RETRIES): Promise<Response> {
  let lastErr: unknown
  for (let i = 0; i <= retries; i++) {
    try {
      const resp = await fetch(url, { credentials: 'omit' })
      if (resp.ok) return resp
      // 4xx other than 429 is not worth retrying.
      if (resp.status < 500 && resp.status !== 429) {
        throw new Error(`HTTP ${resp.status} ${resp.statusText}`)
      }
      lastErr = new Error(`HTTP ${resp.status} ${resp.statusText}`)
    } catch (e) {
      lastErr = e
    }
    if (i < retries) {
      const delayMs = 500 * Math.pow(3, i) // 500, 1500, 4500
      await new Promise((r) => setTimeout(r, delayMs))
    }
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

async function fetchShard(
  url: string,
  dataPath: string,
  mirrorBase: string | null,
  opfs: OPFSDir,
  cacheStores: Cache[],
  pendingWrites: Promise<void>[],
  onProgress?: (msg: string) => void,
): Promise<ArrayBuffer> {
  const leaf = url.split('/').at(-1)
  const cache = (buf: ArrayBuffer) => { pendingWrites.push(opfsWrite(opfs, dataPath, buf)) }

  // Tier 0: dev-only local mirror served by Vite from .weights-local / HF cache
  if (mirrorBase) {
    try {
      const resp = await fetch(mirrorBase + dataPath)
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

  // Tier 3: network (with retry/backoff)
  onProgress?.(`[fetch] ${leaf}`)
  const resp = await fetchWithRetry(url)
  const buf = await resp.arrayBuffer()
  cache(buf)
  return buf
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
// Bounded concurrency helper
// ============================================================

async function mapLimited<T, R>(
  items: T[],
  limit: number,
  worker: (item: T, idx: number) => Promise<R>,
): Promise<R[]> {
  const out: R[] = new Array(items.length)
  let next = 0
  const runOne = async () => {
    while (next < items.length) {
      const i = next++
      out[i] = await worker(items[i], i)
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, runOne))
  return out
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

function formatEta(sec: number): string {
  if (!isFinite(sec) || sec <= 0) return '—'
  if (sec < 60) return `${Math.round(sec)}s`
  const m = Math.floor(sec / 60)
  const s = Math.round(sec - m * 60)
  return `${m}m ${s}s`
}

function uploadRecord(device: GPUDevice, shard: ArrayBuffer, rec: FlatRecord): GPUBuffer {
  // A record larger than the device's storage-binding ceiling uploads fine but
  // fails bind-group creation later, which invalidates every submit touching
  // it — the engine then runs "successfully" with all-zero outputs. Fail loudly
  // at load time instead (fix: raise requiredLimits at requestDevice(), see
  // bootEngine in loading-ui.ts / tests/kernels/gpu.mjs).
  if (rec.nbytes > device.limits.maxStorageBufferBindingSize) {
    throw new Error(
      `Weight record '${rec.name}' (${rec.nbytes} bytes) exceeds ` +
      `device.limits.maxStorageBufferBindingSize (${device.limits.maxStorageBufferBindingSize}). ` +
      `Request a higher requiredLimits.maxStorageBufferBindingSize at device creation.`
    )
  }
  const gpuBuf = device.createBuffer({
    size: Math.max(rec.nbytes, 4),
    usage: USAGE,
    label: rec.name,
  })
  // Uint8Array view avoids the ArrayBuffer.slice() copy before writeBuffer.
  device.queue.writeBuffer(gpuBuf, 0, new Uint8Array(shard, rec.byteOffset, rec.nbytes))
  return gpuBuf
}

export async function loadWeights(
  device: GPUDevice,
  onProgress?: (msg: string) => void,
  onStats?: (stats: WeightLoadStats) => void,
  spec: ModelSpec = PHI3,
): Promise<LoadedWeights> {
  const baseUrl = modelBaseUrl(spec)
  const opfsDir = opfsDirFor(spec)
  const mirrorBase = localMirrorBase(spec)

  // Manifest
  onProgress?.('Loading ndarray-cache.json…')
  const { dir: opfs, persisted } = await openOPFS(opfsDir)
  onProgress?.(opfs
    ? `OPFS ready (${opfsDir}, ${persisted ? 'persistent' : 'best-effort'})`
    : 'OPFS unavailable')
  const cacheStores = await openAllCacheStores()
  if (cacheStores.length > 0) onProgress?.(`Cache API: ${cacheStores.length} store(s) open`)

  // Tracks fire-and-forget OPFS writes so loadWeights doesn't resolve while
  // shards are still being persisted (would leave a torn cache on tab close).
  const pendingWrites: Promise<void>[] = []

  const manifestBytes = await fetchShard(
    baseUrl + 'ndarray-cache.json',
    'ndarray-cache.json',
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

  await mapLimited(shardEntries, FETCH_CONCURRENCY, async ([dataPath, records]) => {
    const shard = await fetchShard(baseUrl + dataPath, dataPath, mirrorBase, opfs, cacheStores, pendingWrites, onProgress)
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
  })

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

  const layers: LoadedWeights['layers'] = []
  for (let L = 0; L < spec.layers; L++) {
    const n = N.layer(L)
    layers.push({
      qkvWeights: find(...n.qkvWeight),
      qkvScales: find(...n.qkvScale),
      oProjWeights: find(...n.oProjWeight),
      oProjScales: find(...n.oProjScale),
      normGamma1: find(...n.norm1),
      normGamma2: find(...n.norm2),
      ffnWeights: find(...n.ffnWeight),
      ffnScales: find(...n.ffnScale),
      ffnDownWeights: find(...n.ffnDownWeight),
      ffnDownScales: find(...n.ffnDownScale),
      // Per-head q/k RMSNorm gammas (Qwen3) — only when the naming declares them.
      qNormGamma: n.qNorm ? find(...n.qNorm) : undefined,
      kNormGamma: n.kNorm ? find(...n.kNorm) : undefined,
    })
  }

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

export interface LoadedWeights {
  device: GPUDevice
  embdWeights: GPUBuffer
  embdScales: GPUBuffer
  lmHeadWeights: GPUBuffer
  lmHeadScales: GPUBuffer
  initNormGamma: GPUBuffer     // layer 0 input_layernorm
  finalNormGamma: GPUBuffer    // model.norm (after all layers)
  layers: Array<{
    qkvWeights: GPUBuffer
    qkvScales: GPUBuffer
    oProjWeights: GPUBuffer
    oProjScales: GPUBuffer
    normGamma1: GPUBuffer    // input_layernorm (pre-attention)
    normGamma2: GPUBuffer    // post_attention_layernorm (pre-FFN)
    ffnWeights: GPUBuffer
    ffnScales: GPUBuffer
    ffnDownWeights: GPUBuffer
    ffnDownScales: GPUBuffer
    qNormGamma?: GPUBuffer   // per-head q RMSNorm over head_dim (Qwen3 only)
    kNormGamma?: GPUBuffer   // per-head k RMSNorm over head_dim (Qwen3 only)
  }>
}
