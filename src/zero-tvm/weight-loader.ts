/**
 * WEIGHT LOADER — Load Phi-3 MLC weights from browser Cache API or HuggingFace.
 *
 * No TVM, no WebLLM runtime.
 * Reads MLC's ndarray-cache.json, fetches shard binaries, uploads to GPU.
 *
 * The model (Phi-3-mini-4k-instruct-q4f16_1-MLC) is cached in the browser
 * by WebLLM via tvmjs's Cache API. We reuse that cache if available,
 * falling back to direct HuggingFace fetch.
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
// Fetch with browser cache fallback
// ============================================================

export type ProgressCallback = (msg: string) => void
/** downloaded: cumulative bytes so far, total: total model bytes, file: current shard name */
export type ByteProgressCallback = (downloaded: number, total: number, file: string) => void

async function fetchCached(
  url: string,
  onProgress?: ProgressCallback,
  onByteProgress?: ByteProgressCallback
): Promise<ArrayBuffer> {
  const fileName = url.split('/').at(-1) || url

  // Try every Cache API store (tvmjs may use a variety of names)
  try {
    const cacheNames = await caches.keys()
    for (const name of cacheNames) {
      const store = await caches.open(name)
      const resp = await store.match(url)
      if (resp) {
        onProgress?.(`cache  ${fileName}`)
        return resp.arrayBuffer()
      }
    }
  } catch { /* no Cache API */ }

  onProgress?.(`fetch  ${fileName}`)
  const resp = await fetch(url)
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`)

  // Stream with byte-level progress if possible
  let arrayBuf: ArrayBuffer
  if (resp.body && onByteProgress) {
    const total = parseInt(resp.headers.get('content-length') || '0', 10)
    const reader = resp.body.getReader()
    const chunks: Uint8Array[] = []
    let downloaded = 0

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      chunks.push(value)
      downloaded += value.byteLength
      onByteProgress(downloaded, total, fileName)
    }

    const result = new Uint8Array(downloaded)
    let offset = 0
    for (const chunk of chunks) {
      result.set(chunk, offset)
      offset += chunk.byteLength
    }
    arrayBuf = result.buffer
  } else {
    arrayBuf = await resp.arrayBuffer()
  }

  // Save to Cache API so subsequent loads resume from where we left off
  try {
    const store = await caches.open('zero-tvm-weights')
    await store.put(url, new Response(arrayBuf.slice(0)))
  } catch { /* Cache API unavailable — still works, just won't persist */ }

  return arrayBuf
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
// Shard binary fetch (cached per dataPath)
// ============================================================

type ShardCache = Map<string, ArrayBuffer>

async function getShard(
  baseUrl: string,
  dataPath: string,
  shardCache: ShardCache,
  onProgress?: ProgressCallback,
  onByteProgress?: ByteProgressCallback
): Promise<ArrayBuffer> {
  if (shardCache.has(dataPath)) return shardCache.get(dataPath)!
  const buf = await fetchCached(baseUrl + dataPath, onProgress, onByteProgress)
  shardCache.set(dataPath, buf)
  return buf
}

// ============================================================
// Build GPUBuffer from a flat record
// ============================================================


// ============================================================
// Parameter name patterns for Phi-3 MLC
//
// MLC uses GroupQuantize which produces two arrays per weight:
//   {base_name}.q_weight  — uint32 packed int4
//   {base_name}.q_scale   — float16 scales
//
// Norms are unquantized float16.
// ============================================================

function find(index: Map<string, FlatRecord>, ...candidates: string[]): FlatRecord {
  for (const c of candidates) {
    const r = index.get(c)
    if (r) return r
  }
  throw new Error(`Weight not found. Tried: ${candidates.join(', ')}\nAvailable: ${[...index.keys()].slice(0, 20).join(', ')}`)
}

// ============================================================
// Main loader
// ============================================================

export async function loadWeights(
  device: GPUDevice,
  onProgress?: ProgressCallback,
  onByteProgress?: ByteProgressCallback
): Promise<LoadedWeights> {
  const baseUrl = PHI3_MODEL_BASE

  // 1. Fetch ndarray-cache.json
  onProgress?.('Loading ndarray-cache.json...')
  const cacheJsonBuf = await fetchCached(baseUrl + 'ndarray-cache.json', onProgress)
  const cacheJson: NDArrayCache = JSON.parse(new TextDecoder().decode(cacheJsonBuf))

  const allRecords = flattenRecords(cacheJson)
  const index = new Map<string, FlatRecord>()
  for (const r of allRecords) index.set(r.name, r)

  console.log('[weight-loader] Parameters found:', [...index.keys()])

  // Compute total model size from unique shard files
  const shardSizes = new Map<string, number>()
  for (const r of allRecords) {
    const end = r.byteOffset + r.nbytes
    const cur = shardSizes.get(r.dataPath) || 0
    if (end > cur) shardSizes.set(r.dataPath, end)
  }
  const totalModelBytes = [...shardSizes.values()].reduce((a, b) => a + b, 0)
  const completedShards = new Set<string>()

  onProgress?.(`Found ${index.size} parameters (${(totalModelBytes / 1e9).toFixed(1)} GB)`)

  const shardCache: ShardCache = new Map()

  // Wrap byte progress to report cumulative across all shards
  const trackByteProgress: ByteProgressCallback | undefined = onByteProgress
    ? (downloaded, _total, file) => {
        // Each shard reports its own 0→total. We track cumulative by adding
        // completed shards + current shard's partial progress.
        let completed = 0
        for (const s of completedShards) completed += shardSizes.get(s) || 0
        onByteProgress!(completed + downloaded, totalModelBytes, file)
      }
    : undefined

  // Mark shard as fully downloaded after fetch completes
  const origGetShard = getShard
  async function getShardTracked(
    base: string, dataPath: string, cache: ShardCache,
    prog?: ProgressCallback, byteProg?: ByteProgressCallback
  ): Promise<ArrayBuffer> {
    const buf = await origGetShard(base, dataPath, cache, prog, byteProg)
    completedShards.add(dataPath)
    return buf
  }

  // Helper that creates a GPUBuffer from named param
  async function load(name: string, ...alts: string[]): Promise<GPUBuffer> {
    const rec = find(index, name, ...alts)
    const shard = await getShardTracked(baseUrl, rec.dataPath, shardCache, onProgress, trackByteProgress)
    const slice = shard.slice(rec.byteOffset, rec.byteOffset + rec.nbytes)
    const buf = device.createBuffer({
      size: Math.max(rec.nbytes, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: rec.name,
    })
    device.queue.writeBuffer(buf, 0, slice)
    return buf
  }

  // 2. Global weights
  // Actual MLC cache names discovered from ndarray-cache.json:
  //   transformer.embd.q_weight / q_scale  — embedding
  //   transformer.norm.weight               — final RMSNorm before LM head
  //   lm_head.q_weight / q_scale            — LM head
  //   transformer.h.N.ln.weight             — layer N input_layernorm (normGamma1)
  //   transformer.h.N.post_attention_layernorm.weight — normGamma2
  //   transformer.h.N.mixer.qkv_proj.*      — QKV
  //   transformer.h.N.mixer.out_proj.*      — O proj
  //   transformer.h.N.mlp.gate_up_proj.*    — FFN gate+up
  //   transformer.h.N.mlp.down_proj.*       — FFN down

  onProgress?.('Loading embedding weights...')
  const embdWeights = await load(
    'transformer.embd.q_weight',
    'embed_tokens.q_weight',
    'model.embed_tokens.q_weight'
  )
  const embdScales = await load(
    'transformer.embd.q_scale',
    'embed_tokens.q_scale',
    'model.embed_tokens.q_scale'
  )

  // Layer 0's input_layernorm — used as the initial norm before the first QKV
  // (loaded again per-layer below, but needed here for the initial rmsNorm in chat.ts)
  const initNormGamma = await load(
    'transformer.h.0.ln.weight',
    'model.layers.0.input_layernorm.weight'
  )

  // LM head
  const lmHeadWeights = await load(
    'lm_head.q_weight',
    'model.lm_head.q_weight'
  )
  const lmHeadScales = await load(
    'lm_head.q_scale',
    'model.lm_head.q_scale'
  )

  // 3. Per-layer weights (32 layers)
  const LAYERS = 32
  const layers: LoadedWeights['layers'] = []

  for (let L = 0; L < LAYERS; L++) {
    onProgress?.(`Loading layer ${L}/${LAYERS}...`)
    const h = `transformer.h.${L}`   // actual MLC prefix
    const p = `model.layers.${L}`    // standard HF prefix (fallback)

    const qkvWeights = await load(
      `${h}.mixer.qkv_proj.q_weight`,
      `${p}.self_attn.qkv_proj.q_weight`
    )
    const qkvScales = await load(
      `${h}.mixer.qkv_proj.q_scale`,
      `${p}.self_attn.qkv_proj.q_scale`
    )
    const oProjWeights = await load(
      `${h}.mixer.out_proj.q_weight`,
      `${p}.self_attn.o_proj.q_weight`
    )
    const oProjScales = await load(
      `${h}.mixer.out_proj.q_scale`,
      `${p}.self_attn.o_proj.q_scale`
    )
    const normGamma1 = await load(
      `${h}.ln.weight`,
      `${p}.input_layernorm.weight`
    )
    const normGamma2 = await load(
      `${h}.post_attention_layernorm.weight`,
      `${p}.post_attention_layernorm.weight`
    )
    const ffnWeights = await load(
      `${h}.mlp.gate_up_proj.q_weight`,
      `${p}.mlp.gate_up_proj.q_weight`
    )
    const ffnScales = await load(
      `${h}.mlp.gate_up_proj.q_scale`,
      `${p}.mlp.gate_up_proj.q_scale`
    )
    const ffnDownWeights = await load(
      `${h}.mlp.down_proj.q_weight`,
      `${p}.mlp.down_proj.q_weight`
    )
    const ffnDownScales = await load(
      `${h}.mlp.down_proj.q_scale`,
      `${p}.mlp.down_proj.q_scale`
    )

    layers.push({
      qkvWeights, qkvScales,
      oProjWeights, oProjScales,
      normGamma1, normGamma2,
      ffnWeights, ffnScales,
      ffnDownWeights, ffnDownScales,
    })
  }

  // Final norm (applied after all 32 layers, before LM head)
  // model.norm.weight is separate from layer input_layernorm weights
  const finalNormGamma = await load(
    'transformer.norm.weight',
    'model.norm.weight',
    'norm.weight'
  )

  onProgress?.('All weights loaded!')

  return {
    device,
    embdWeights, embdScales,
    lmHeadWeights, lmHeadScales,
    initNormGamma,      // = layer 0's input_layernorm (used for initial embed norm)
    finalNormGamma,     // = model.norm (final norm before LM head)
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
