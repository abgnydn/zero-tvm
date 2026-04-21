/**
 * WEIGHT LOADER — Load Phi-3 MLC weights from OPFS / browser Cache API / HuggingFace.
 *
 * No TVM, no WebLLM runtime.
 * Reads MLC's ndarray-cache.json, fetches shard binaries in parallel,
 * uploads to GPU as each shard arrives.
 *
 * Tiered cache: OPFS (fastest, survives cross-origin) → browser Cache API
 * (populated by prior WebLLM sessions) → HuggingFace fetch.
 */

// ============================================================
// Model URL
// ============================================================

export const PHI3_MODEL_BASE =
  'https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC/resolve/main/'

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
// OPFS helpers — persistent per-origin storage, faster than IDB for large blobs.
// Graceful no-op when OPFS is unavailable (Safari, older Firefox).
// ============================================================

type OPFSDir = FileSystemDirectoryHandle | null

async function openOPFS(): Promise<OPFSDir> {
  try {
    if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return null
    const root = await navigator.storage.getDirectory()
    return await root.getDirectoryHandle('zero-tvm-weights', { create: true })
  } catch {
    return null
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
    const writable = await (fh as any).createWritable()
    await writable.write(data)
    await writable.close()
  } catch {
    // OPFS writes are best-effort — failures just mean next visit pays network again.
  }
}

// ============================================================
// Tiered shard fetch: OPFS → Cache API → HuggingFace
// ============================================================

// In dev, the Vite server mirrors the HF hub snapshot under /local-weights/*
// (see vite.config.ts). This makes cold-start e2e testing instant without
// re-downloading 2 GB over the network.
const LOCAL_MIRROR_BASE = (import.meta as any).env?.DEV ? '/local-weights/' : null

async function fetchShard(
  url: string,
  dataPath: string,
  opfs: OPFSDir,
  onProgress?: (msg: string) => void,
): Promise<ArrayBuffer> {
  const leaf = url.split('/').at(-1)

  // Tier 0: dev-only local mirror served by Vite from ~/.cache/huggingface/hub
  if (LOCAL_MIRROR_BASE) {
    try {
      const resp = await fetch(LOCAL_MIRROR_BASE + dataPath)
      if (resp.ok) {
        onProgress?.(`[local] ${leaf}`)
        const buf = await resp.arrayBuffer()
        void opfsWrite(opfs, dataPath, buf)
        return buf
      }
    } catch { /* mirror not primed — fall through */ }
  }

  // Tier 1: OPFS
  const fromOPFS = await opfsRead(opfs, dataPath)
  if (fromOPFS) {
    onProgress?.(`[opfs] ${leaf}`)
    return fromOPFS
  }

  // Tier 2: browser Cache API (WebLLM's tvmjs populates this)
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) {
        onProgress?.(`[cache] ${leaf}`)
        const buf = await resp.arrayBuffer()
        // Promote into OPFS for faster next visit (fire-and-forget).
        void opfsWrite(opfs, dataPath, buf)
        return buf
      }
    }
  } catch { /* no Cache API */ }

  // Tier 3: network
  onProgress?.(`[fetch] ${leaf}`)
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`)
  const buf = await resp.arrayBuffer()
  void opfsWrite(opfs, dataPath, buf)
  return buf
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
// Main loader — parallel shard fetch, streaming GPU upload
// ============================================================

const USAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function uploadRecord(device: GPUDevice, shard: ArrayBuffer, rec: FlatRecord): GPUBuffer {
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
): Promise<LoadedWeights> {
  const baseUrl = PHI3_MODEL_BASE

  // Manifest
  onProgress?.('Loading ndarray-cache.json...')
  const opfs = await openOPFS()
  const manifestBytes = await fetchShard(baseUrl + 'ndarray-cache.json', 'ndarray-cache.json', opfs, onProgress)
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

  // Parallel shard fetch + per-shard GPU upload as they arrive.
  const gpuBuffers = new Map<string, GPUBuffer>()
  let completed = 0
  const totalShards = byShard.size
  const totalBytes = allRecords.reduce((s, r) => s + r.nbytes, 0)
  let loadedBytes = 0

  await Promise.all(
    [...byShard.entries()].map(async ([dataPath, records]) => {
      const shard = await fetchShard(baseUrl + dataPath, dataPath, opfs, onProgress)
      for (const rec of records) {
        gpuBuffers.set(rec.name, uploadRecord(device, shard, rec))
      }
      completed++
      loadedBytes += shard.byteLength
      const mb = (loadedBytes / 1e6).toFixed(0)
      const total = (totalBytes / 1e6).toFixed(0)
      onProgress?.(`[${completed}/${totalShards}] ${dataPath} · ${mb}/${total} MB`)
    }),
  )

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

  // Resolve the weight layout this engine expects.
  const embdWeights = find('transformer.embd.q_weight', 'embed_tokens.q_weight', 'model.embed_tokens.q_weight')
  const embdScales = find('transformer.embd.q_scale', 'embed_tokens.q_scale', 'model.embed_tokens.q_scale')
  const initNormGamma = find('transformer.h.0.ln.weight', 'model.layers.0.input_layernorm.weight')
  const lmHeadWeights = find('lm_head.q_weight', 'model.lm_head.q_weight')
  const lmHeadScales = find('lm_head.q_scale', 'model.lm_head.q_scale')
  const finalNormGamma = find('transformer.norm.weight', 'model.norm.weight', 'norm.weight')

  const LAYERS = 32
  const layers: LoadedWeights['layers'] = []
  for (let L = 0; L < LAYERS; L++) {
    const h = `transformer.h.${L}`   // MLC prefix
    const p = `model.layers.${L}`    // HF prefix fallback
    layers.push({
      qkvWeights: find(`${h}.mixer.qkv_proj.q_weight`, `${p}.self_attn.qkv_proj.q_weight`),
      qkvScales: find(`${h}.mixer.qkv_proj.q_scale`, `${p}.self_attn.qkv_proj.q_scale`),
      oProjWeights: find(`${h}.mixer.out_proj.q_weight`, `${p}.self_attn.o_proj.q_weight`),
      oProjScales: find(`${h}.mixer.out_proj.q_scale`, `${p}.self_attn.o_proj.q_scale`),
      normGamma1: find(`${h}.ln.weight`, `${p}.input_layernorm.weight`),
      normGamma2: find(`${h}.post_attention_layernorm.weight`, `${p}.post_attention_layernorm.weight`),
      ffnWeights: find(`${h}.mlp.gate_up_proj.q_weight`, `${p}.mlp.gate_up_proj.q_weight`),
      ffnScales: find(`${h}.mlp.gate_up_proj.q_scale`, `${p}.mlp.gate_up_proj.q_scale`),
      ffnDownWeights: find(`${h}.mlp.down_proj.q_weight`, `${p}.mlp.down_proj.q_weight`),
      ffnDownScales: find(`${h}.mlp.down_proj.q_scale`, `${p}.mlp.down_proj.q_scale`),
    })
  }

  onProgress?.(`All weights loaded · ${(totalBytes / 1e6).toFixed(0)} MB · ${opfs ? 'OPFS active' : 'OPFS unavailable'}`)

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
  }>
}
