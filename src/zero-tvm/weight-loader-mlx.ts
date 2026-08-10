/**
 * MLX SAFETENSORS LOADER — the I/O half of the Qwen3.6 port.
 *
 * `mlx-weights.ts` decides WHAT each GPU buffer is made of (which records, in
 * which order, with which dtype conversion) and is byte-verified against the
 * checkpoint. This module is the part that has to go and get the bytes, and it
 * differs from the MLC path in one structural way:
 *
 *   THE UNIT OF WORK IS A BufferPlan, NEVER A SHARD.
 *
 * The MLC loader fetches a shard, then slices GPU buffers out of it. That
 * cannot work here: a Qwen3.6 shard is 5.3 GB, and both `fetchBufWithRetry`
 * (which accumulates into one Uint8Array) and `File.arrayBuffer()` allocate a
 * single ArrayBuffer, which V8 caps near 4.29 GB. So every read is a byte
 * RANGE, sized by the record it serves, and what gets cached in OPFS is the
 * BUILT buffer — after concatenation and bf16 conversion — keyed by plan name.
 * A warm reload then skips the ranges and the conversion together.
 *
 * The safetensors index and the four shard headers (~90 KB each) are fetched
 * whole; they are the only things small enough to be.
 *
 * Imports carry the .ts extension on purpose — like model-spec.ts and
 * mlx-weights.ts, this module is imported directly by tests/kernels/*.mjs under
 * Node type stripping, which does no extension searching.
 */

import type { ModelSpec } from '../compiler/model-spec.ts'
import {
  parseSafetensorsHeader, planLayer, planGlobal, buildBuffer, planBytes,
  type BufferPlan, type TensorInfo,
} from './mlx-weights.ts'

/** Where a record lives: which shard, and its byte range within the file. */
interface RecordLoc {
  file: string
  /** Absolute byte offset in the shard file (header already added). */
  begin: number
  end: number
  info: TensorInfo
}

export interface MlxSource {
  /** Fetch [begin, end) of a shard file. */
  range(file: string, begin: number, end: number): Promise<Uint8Array<ArrayBuffer>>
  /** Fetch a whole small file (index / header probe). */
  whole(file: string): Promise<Uint8Array<ArrayBuffer>>
}

/**
 * Resolve every record's shard and absolute byte range.
 *
 * Reads the index, then each shard's 8-byte header length followed by the
 * header itself. Four ranged reads of ~90 KB against 20 GB of shards — the
 * whole point of not fetching shards.
 */
export async function openMlxCheckpoint(src: MlxSource): Promise<{
  locate: (record: string) => RecordLoc
  records: string[]
  shards: string[]
}> {
  const index = JSON.parse(new TextDecoder().decode(await src.whole('model.safetensors.index.json')))
  const map: Record<string, string> = index.weight_map
  const shards = [...new Set(Object.values(map))].sort()

  const headers: Record<string, { tensors: Record<string, TensorInfo>; dataStart: number }> = {}
  for (const file of shards) {
    const lenBytes = await src.range(file, 0, 8)
    const view = new DataView(lenBytes.buffer, lenBytes.byteOffset, 8)
    if (view.getUint32(4, true) !== 0) throw new Error(`${file}: safetensors header longer than 4 GB`)
    const headerLen = view.getUint32(0, true)
    const head = await src.range(file, 0, 8 + headerLen)
    headers[file] = parseSafetensorsHeader(head.buffer.slice(head.byteOffset, head.byteOffset + head.byteLength))
  }

  const locate = (record: string): RecordLoc => {
    const file = map[record]
    if (!file) throw new Error(`record not in checkpoint index: ${record}`)
    const h = headers[file]
    const info = h.tensors[record]
    if (!info) throw new Error(`record in index but not in ${file}'s header: ${record}`)
    return { file, begin: h.dataStart + info.begin, end: h.dataStart + info.end, info }
  }
  return { locate, records: Object.keys(map), shards }
}

/**
 * Every buffer the model needs, in the order they should be built.
 *
 * With `range` this is ONE PIPELINE STAGE's buffers: its layers, plus the
 * embedding only if it starts the model and the lm_head / final norm only if
 * it ends it. That is what lets two machines hold a model neither of them
 * could — nothing else in the loader has to know about the split.
 */
export function planModel(
  spec: ModelSpec,
  range?: { start: number; end: number },
): { plan: BufferPlan; layer: number | null }[] {
  const L0 = range?.start ?? 0
  const L1 = range?.end ?? spec.layers
  const out: { plan: BufferPlan; layer: number | null }[] = []
  for (const p of planGlobal(spec, undefined, range)) out.push({ plan: p, layer: null })
  for (let L = L0; L < L1; L++) {
    for (const p of planLayer(spec, L)) out.push({ plan: p, layer: L })
  }
  return out
}

/**
 * Stable OPFS / GPU-label key for a plan. Layer plans repeat their names.
 *
 * An OP-BEARING plan additionally carries a short hash of its op — today
 * DeepSeek's kv_b prep keys as `l0.mla_kvb.fd87`.
 * Every other plan is a pure function of the checkpoint, so its bytes cannot
 * change without the checkpoint changing; an op's bytes depend on code that is
 * still being written. A stale OPFS entry survives the fix silently, and
 * peer-weights.ts then replicates it to other machines by filename with a
 * SHA-256 that proves transport, not correctness.
 *
 * Only op-bearing plans get the suffix, so every existing key stays
 * byte-identical and the gigabytes of Qwen3.6 caches already on disk are not
 * invalidated by this feature.
 */
export function planKey(plan: BufferPlan, layer: number | null): string {
  const base = layer === null ? `g.${plan.name}` : `l${layer}.${plan.name}`
  return plan.op ? `${base}.${opHash(plan)}` : base
}

/** FNV-1a over the op's parameters AND its part record names, folded to 16
 *  bits — 4 hex characters, which `opfsKey` passes through unescaped. The
 *  permutation MAP is hashed, not just its length: shipping a permutation and
 *  then correcting it is exactly the case this exists for. */
function opHash(plan: BufferPlan): string {
  const op = plan.op!
  const params = op.kind === 'permuteRows'
    ? `rows=${op.rows};src=${op.src.join(',')}`
    : `bits=${op.bits};group=${op.group};k=${op.k};heads=${op.heads};nope=${op.nope};v=${op.v}`
  const text = `${op.kind};${params};${plan.parts.map((p) => `${p.record}:${p.convert}`).join(',')}`
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (((h >>> 16) ^ h) & 0xffff).toString(16).padStart(4, '0')
}

export interface BuiltBuffer {
  key: string
  layer: number | null
  name: string
  data: Uint8Array
}

/**
 * Build one plan's bytes: ranged-read each record, convert, concatenate.
 *
 * `readRecord` is synchronous inside `buildBuffer`, so the ranges are fetched
 * first and handed over as a lookup — which also lets the caller run the
 * fetches concurrently.
 */
export async function buildPlan(
  plan: BufferPlan,
  locate: (record: string) => RecordLoc,
  src: MlxSource,
): Promise<Uint8Array<ArrayBuffer>> {
  const parts = new Map<string, Uint8Array>()
  for (const part of plan.parts) {
    const loc = locate(part.record)
    parts.set(part.record, await src.range(loc.file, loc.begin, loc.end))
  }
  const { data } = buildBuffer(plan, (n) => {
    const d = parts.get(n)
    if (!d) throw new Error(`internal: range for ${n} was not fetched`)
    return d
  }, (n) => locate(n).info.dtype)
  return data
}

/** Total GPU bytes the model will occupy, before fetching anything. */
export function modelBytes(spec: ModelSpec, locate: (record: string) => RecordLoc): number {
  const src = (r: string) => { const l = locate(r); return l.end - l.begin }
  return planModel(spec).reduce((a, { plan }) => a + planBytes(plan, src), 0)
}

// ============================================================
// GPU assembly
// ============================================================

/** Where a plan's buffer lands in LoadedWeights. */
type Slot = [group: 'root' | 'layer' | 'gdn' | 'moe' | 'mla', field: string]

/**
 * plan name -> LoadedWeights field. Exhaustive on purpose: `assembleMlx`
 * THROWS on a plan it cannot place, so adding a buffer to planLayer without
 * saying where it goes fails at load rather than leaving a field undefined for
 * a bind group to trip over later.
 */
const SLOTS: Record<string, Slot> = {
  embed_w: ['root', 'embdWeights'], embed_s: ['root', 'embdScales'], embed_b: ['root', 'embdBiases'],
  lm_head_w: ['root', 'lmHeadWeights'], lm_head_s: ['root', 'lmHeadScales'], lm_head_b: ['root', 'lmHeadBiases'],
  final_norm: ['root', 'finalNormGamma'],
  norm1: ['layer', 'normGamma1'], norm2: ['layer', 'normGamma2'],
  c_attn_w: ['layer', 'qkvWeights'], c_attn_s: ['layer', 'qkvScales'], c_attn_b: ['layer', 'qkvBiases'],
  o_proj_w: ['layer', 'oProjWeights'], o_proj_s: ['layer', 'oProjScales'], o_proj_b: ['layer', 'oProjBiases'],
  q_norm: ['layer', 'qNormGamma'], k_norm: ['layer', 'kNormGamma'],
  // MLA (DeepSeek-V2). NOT under the c_attn_* names: those are the fused QKV a
  // GQA layer binds, and MLA's q_proj is not a q++k++v concatenation.
  mla_q_w: ['mla', 'qWeights'], mla_q_s: ['mla', 'qScales'], mla_q_b: ['mla', 'qBiases'],
  mla_kva_w: ['mla', 'kvaWeights'], mla_kva_s: ['mla', 'kvaScales'], mla_kva_b: ['mla', 'kvaBiases'],
  mla_kva_norm: ['mla', 'kvaNormGamma'], mla_kvb: ['mla', 'kvbF16'],
  gdn_proj_w: ['gdn', 'projWeights'], gdn_proj_s: ['gdn', 'projScales'], gdn_proj_b: ['gdn', 'projBiases'],
  gdn_out_w: ['gdn', 'outWeights'], gdn_out_s: ['gdn', 'outScales'], gdn_out_b: ['gdn', 'outBiases'],
  gdn_conv: ['gdn', 'convWeight'], gdn_norm: ['gdn', 'normGamma'],
  gdn_a_log: ['gdn', 'aLog'], gdn_dt_bias: ['gdn', 'dtBias'],
  moe_gate_proj_w: ['moe', 'gateWeights'], moe_gate_proj_s: ['moe', 'gateScales'], moe_gate_proj_b: ['moe', 'gateBiases'],
  moe_up_proj_w: ['moe', 'upWeights'], moe_up_proj_s: ['moe', 'upScales'], moe_up_proj_b: ['moe', 'upBiases'],
  moe_down_proj_w: ['moe', 'downWeights'], moe_down_proj_s: ['moe', 'downScales'], moe_down_proj_b: ['moe', 'downBiases'],
  router_w: ['moe', 'routerWeights'], router_s: ['moe', 'routerScales'], router_b: ['moe', 'routerBiases'],
  ffn_w: ['layer', 'ffnWeights'], ffn_s: ['layer', 'ffnScales'], ffn_b: ['layer', 'ffnBiases'],
  ffn_down_w: ['layer', 'ffnDownWeights'], ffn_down_s: ['layer', 'ffnDownScales'], ffn_down_b: ['layer', 'ffnDownBiases'],
}

export interface MlxLoadHooks {
  /** Cached BUILT buffer for a plan key, or null. Skips the ranges AND the
   *  bf16 conversion, which is why the cache is keyed by plan and not shard. */
  cacheRead?: (key: string) => Promise<ArrayBuffer | null>
  cacheWrite?: (key: string, data: ArrayBuffer) => void
  onProgress?: (msg: string) => void
  onBuffer?: (done: number, total: number, bytes: number) => void
}

/**
 * Fetch, convert and upload every buffer, then assemble LoadedWeights.
 *
 * Buffers are uploaded one at a time and the host copy dropped immediately:
 * peak host memory is ONE plan (242 MB for the embedding table), not the model.
 * There is no concurrency here on purpose — 19.5 GB of GPU allocation is the
 * scarce resource, not request parallelism, and overlapping fetches would hold
 * several host copies alive at once.
 */
export async function assembleMlx(
  device: GPUDevice,
  spec: ModelSpec,
  src: MlxSource,
  locate: (record: string) => RecordLoc,
  usage: number,
  hooks: MlxLoadHooks = {},
  /** Load ONE pipeline stage's buffers (see planModel). */
  range?: { start: number; end: number },
): Promise<Record<string, unknown>> {
  const plans = planModel(spec, range)
  const root: Record<string, unknown> = { device }
  const layers: Record<string, unknown>[] = Array.from({ length: spec.layers }, () => ({}))
  let done = 0
  let bytes = 0

  for (const { plan, layer } of plans) {
    const slot = SLOTS[plan.name]
    if (!slot) throw new Error(`assembleMlx: no LoadedWeights slot for plan '${plan.name}'`)
    const key = planKey(plan, layer)

    let data: Uint8Array<ArrayBuffer>
    const cached = await hooks.cacheRead?.(key)
    if (cached) {
      data = new Uint8Array(cached) as Uint8Array<ArrayBuffer>
    } else {
      data = await buildPlan(plan, locate, src)
      hooks.cacheWrite?.(key, data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength))
    }

    const buf = device.createBuffer({ size: data.byteLength, usage, label: key })
    device.queue.writeBuffer(buf, 0, data)
    bytes += data.byteLength
    done++
    hooks.onBuffer?.(done, plans.length, bytes)

    const [group, field] = slot
    if (group === 'root') root[field] = buf
    else if (layer === null) throw new Error(`assembleMlx: '${plan.name}' is a layer plan with no layer`)
    else if (group === 'layer') layers[layer][field] = buf
    else {
      const sub = (layers[layer][group] ??= {}) as Record<string, unknown>
      sub[field] = buf
    }
  }

  root.layers = layers
  // Tied embeddings: the checkpoint ships no lm_head records (planGlobal skips
  // the trio), and mlx_lm's own forward reuses the quantized embedding matrix.
  // The buffer layouts agree — both are [vocab] rows of K=d — so the alias is
  // the whole implementation, exactly as the MLC path's paramNaming points
  // lmHeadWeight at the embed records.
  if (spec.tiedEmbeddings) {
    root.lmHeadWeights = root.embdWeights
    root.lmHeadScales = root.embdScales
    root.lmHeadBiases = root.embdBiases
  }
  // The MLC path exposes the first layer's input_layernorm separately; the
  // engine reads it before the layer loop. On a pipeline stage the first
  // LOADED layer is the one that matters, not layer 0 (which may be a
  // different machine's).
  root.initNormGamma = layers[range?.start ?? 0].normGamma1
  return root
}
