/**
 * KV-POOL — persistent prefix cache. Paging Phase 1 (docs/PAGING_PLAN.md).
 *
 * What it buys, in this machine's own measurements: a 20k-token prefix costs
 * ~99 s to re-prefill and 0.36 s to restore (BENCH.md, KV paging feasibility,
 * 56× under the ship threshold at the SLOWEST measured rates). Today that
 * cost is paid on every reload, crash, tab close, or model switch — the
 * agentic cases. LM Studio's prompt cache is RAM-resident and dies with the
 * process; this one lives in OPFS and survives all of the above.
 *
 * SHAPE (deliberately smaller than the plan's full tiers, and why):
 *   - Bytes in OPFS under `zero-tvm-kvpool/<fingerprint>/`, ONE entry per
 *     fingerprint, latest-wins. The plan's tier-1 IndexedDB index and
 *     multi-entry pool with refcounts exist for cross-conversation sharing
 *     and eviction — Phase 3 concerns. One entry already covers the headline
 *     case (an agent's growing transcript), and `prefix-pool.ts`'s chain-hash
 *     index is ready for the upgrade.
 *   - The entry stores its FULL token ids, and attach compares them exactly.
 *     The fingerprint (`prefix-pool.ts`) gates compatibility: spec, weight
 *     revision, resolved variant set, fused/int8/layerRange, and the prefill
 *     path that wrote the bytes. Bytes under the wrong fingerprint are a
 *     different model's memories; nothing downstream could detect the swap,
 *     so nothing upstream is allowed to make it.
 *
 * SAFETY RULE, verbatim from the plan: KV[0..n) = f(tokens[0..n), spec,
 * variant set, KV dtype). Everything the pool keys on is inside that tuple;
 * sampling parameters are deliberately OUTSIDE it (K/V at position p does not
 * depend on temperature), so a temperature slider does not kill reuse.
 */

import type { DecodeEngine } from './engine-core.js'
import type { ModelSpec } from '../compiler/model-spec.js'
import type { VariantFlags } from './variants.js'
import { computeFingerprint, type ComputeConfig, type Hash128 } from './prefix-pool.js'

const POOL_DIR = 'zero-tvm-kvpool'
/** Bump on any change to the entry layout — a stale entry must read as a
 *  miss, never as bytes. */
const FORMAT = 1

export interface PoolConfig {
  spec: ModelSpec
  weightRevision: string
  variants: VariantFlags
  fused: boolean
  int8KV: boolean
  /** The prefill path that will WRITE future bytes and wrote saved ones.
   *  Chunked and per-token are not bit-equal. */
  prefillPath: 'chunked' | 'per-token'
  /** adapter.info vendor/architecture — WGSL codegen is not promised
   *  bit-stable across GPUs (PAGING_PLAN §4.6). */
  adapter: { vendor: string; architecture: string }
}

export function poolFingerprint(cfg: PoolConfig): Hash128 {
  const cc: ComputeConfig = {
    specId: cfg.spec.id,
    weightRevision: cfg.weightRevision,
    variants: cfg.variants,
    fused: cfg.fused,
    int8KV: cfg.int8KV,
    layerRange: null,   // pooled engines are whole-model; stages refuse in exportKV
    prefillPath: cfg.prefillPath,
    adapter: cfg.adapter,
  }
  return computeFingerprint(cc)
}

async function poolDir(fingerprint: Hash128, create: boolean): Promise<FileSystemDirectoryHandle | null> {
  try {
    const root = await navigator.storage.getDirectory()
    const pool = await root.getDirectoryHandle(POOL_DIR, { create })
    return await pool.getDirectoryHandle(fingerprint, { create })
  } catch {
    return null
  }
}

interface EntryMeta {
  format: number
  tokens: number
  ids: number[]
  layerBytes: number
  layerCount: number
  gdnBytes: number[]
  savedAt: number
}

/**
 * Persist the engine's absorbed prefix. Returns the token count saved, or 0
 * with a reason when the state is not poolable — callers log it, because a
 * pool that silently never saves looks identical to one that works until the
 * first restart.
 */
export async function poolSave(engine: DecodeEngine, cfg: PoolConfig): Promise<{ tokens: number; reason?: string }> {
  const snap = await engine.exportKV()
  if (!snap) return { tokens: 0, reason: 'engine state not poolable (MLA/int8/stage/invalid/mid-chunk)' }
  const dir = await poolDir(poolFingerprint(cfg), true)
  if (!dir) return { tokens: 0, reason: 'OPFS unavailable' }

  const meta: EntryMeta = {
    format: FORMAT,
    tokens: snap.tokens,
    ids: snap.ids,
    layerBytes: snap.layers[0]?.byteLength ?? 0,
    layerCount: snap.layers.length,
    gdnBytes: snap.gdn.map((g) => g.byteLength),
    savedAt: Date.now(),
  }
  // Bytes first, meta LAST — meta.json is the commit record. A crash between
  // the two leaves bytes without meta, which reads as a miss, not corruption.
  const bin = await (await dir.getFileHandle('entry.bin', { create: true })).createWritable()
  for (const l of snap.layers) await bin.write(l)
  for (const g of snap.gdn) await bin.write(g)
  await bin.close()
  const mw = await (await dir.getFileHandle('meta.json', { create: true })).createWritable()
  await mw.write(JSON.stringify(meta))
  await mw.close()
  return { tokens: snap.tokens }
}

/**
 * Restore the saved prefix into `engine` IF the incoming prompt extends it.
 *
 * The prompt must carry the entry's ids as an EXACT PREFIX — compared id by
 * id, not by hash — and must be strictly longer (at least one token must run:
 * it produces the first logits; hybrid additionally cannot rewind, and the
 * reuseStart rules the engine already applies enforce the rest). On any
 * doubt: full prefill. A wrong restore is fluent nonsense; a refused one
 * costs seconds.
 */
export async function poolTryRestore(
  engine: DecodeEngine,
  cfg: PoolConfig,
  promptIds: readonly number[],
): Promise<{ restored: number; reason?: string }> {
  const dir = await poolDir(poolFingerprint(cfg), false)
  if (!dir) return { restored: 0, reason: 'no entry for this fingerprint' }
  let meta: EntryMeta
  let bin: ArrayBuffer
  try {
    meta = JSON.parse(await (await (await dir.getFileHandle('meta.json')).getFile()).text()) as EntryMeta
    bin = await (await (await dir.getFileHandle('entry.bin')).getFile()).arrayBuffer()
  } catch {
    return { restored: 0, reason: 'no entry for this fingerprint' }
  }
  if (meta.format !== FORMAT) return { restored: 0, reason: `entry format ${meta.format} != ${FORMAT}` }
  if (meta.tokens >= promptIds.length) return { restored: 0, reason: 'prompt does not extend the entry' }
  for (let i = 0; i < meta.tokens; i++) {
    if (promptIds[i] !== meta.ids[i]) return { restored: 0, reason: `id mismatch at ${i}` }
  }
  const expect = meta.layerBytes * meta.layerCount + meta.gdnBytes.reduce((a, b) => a + b, 0)
  if (bin.byteLength !== expect) return { restored: 0, reason: 'entry truncated' }

  const layers: ArrayBuffer[] = []
  let off = 0
  for (let i = 0; i < meta.layerCount; i++) {
    layers.push(bin.slice(off, off + meta.layerBytes))
    off += meta.layerBytes
  }
  const gdn: ArrayBuffer[] = []
  for (const g of meta.gdnBytes) {
    gdn.push(bin.slice(off, off + g))
    off += g
  }
  const ok = engine.importKV({ tokens: meta.tokens, ids: meta.ids, layers, gdn })
  return ok ? { restored: meta.tokens } : { restored: 0, reason: 'engine refused the snapshot' }
}

/** Wipe one fingerprint's entry, or the whole pool. */
export async function poolClear(fingerprint?: Hash128): Promise<void> {
  try {
    const root = await navigator.storage.getDirectory()
    if (!fingerprint) { await root.removeEntry(POOL_DIR, { recursive: true }); return }
    const pool = await root.getDirectoryHandle(POOL_DIR)
    await pool.removeEntry(fingerprint, { recursive: true })
  } catch { /* absent is the goal */ }
}
