/**
 * SLAB SOURCE — where ONE expert's bytes come from.
 *
 * `expert-pool.ts` decides which expert goes in which slot; this decides where
 * the bytes for a miss are read. A slab is one expert's share of one stacked
 * MoE tensor — `[experts, N, K]` sliced at the expert axis — and it is the unit
 * an expert miss transfers: 1.3-2.5 MiB, read and then `writeBuffer`'d into a
 * pool slot. docs/MOE_CHUNK_PLAN.md ("Repriced 2026-08-14") measures that pair
 * at 1.007 ms and shows the two do not pipeline, which is why the pool has to
 * be big and why nothing here may cost a second read.
 *
 * TWO SOURCES, ONE GEOMETRY.
 *
 *   - `opfsSlabSource` reads the BUILT buffer the MLX loader already caches in
 *     OPFS (`weight-loader.ts` -> `assembleMlx`'s cacheWrite, keyed by
 *     `planKey`). That file is the byte concatenation the loader wrote:
 *     the routed stack followed by the shared expert. Nothing new is written
 *     to disk for streaming.
 *   - `fetchSlabSource` reads the checkpoint's shards over HTTP byte ranges,
 *     the same `MlxSource.range` the loader uses (dev mirror or HF). This is
 *     the path when there is no OPFS cache at all — which is the normal dev
 *     case, since `loadWeights` deliberately skips `cacheWrite` when serving
 *     from a local mirror rather than writing a second copy of the checkpoint.
 *
 * EVERY OFFSET IS DERIVED, NEVER ASSUMED. The expert stride comes from the
 * safetensors header — `shape[1] * shape[2] * sizeof(dtype)` — and is
 * cross-checked against the byte range the same header declares. A wrong
 * offset does not fault: it uploads a real expert's real bytes to the wrong
 * slot, or a half-row shear of two experts, and the model keeps generating
 * fluent text. So the checks are here rather than downstream, where there is
 * nothing left to check against. For the same reason a short read throws: a
 * slab of the wrong length must never reach the GPU.
 *
 * BOTH SOURCES RETURN THE SAME BYTES — the ones the loader uploaded. The OPFS
 * buffer is post-conversion (the plan's `bf16->f16` already ran when it was
 * built), so a fetched slab runs the same conversion on its way past. That
 * conversion is element-wise and 2 bytes wide either way, so it moves no
 * offset; but a raw bf16 slab written into a slot the kernel reads as f16 puts
 * every scale off by ~1e21, with no error anywhere.
 *
 * THE SHARED EXPERT is index `experts` of the same logical stack, exactly as
 * `planLayer` builds it — appended to the routed stack in the OPFS buffer, and
 * a separate `...shared_expert.<proj>` record (possibly in another shard) over
 * HTTP. Both resolve through the same `expert` argument, so no caller carries
 * the special case.
 *
 * NO GPU AT MODULE SCOPE. This module and everything it imports stay free of
 * `GPUBufferUsage` and friends, so it is importable in Node — see the
 * import-time rule at the top of src/lib/index.ts. `weight-loader.ts` is NOT
 * importable here for that reason; the OPFS directory handle and the
 * `MlxSource` are passed in by the caller that already has them.
 *
 * Imports carry the .ts extension like the two modules below, so this file
 * survives Node type stripping (no extension searching) as well as Vite.
 */

import type { ModelSpec } from '../compiler/model-spec.ts'
import { bf16ToF16, planLayer, type Convert, type TensorInfo } from './mlx-weights.ts'
import { planKey, type MlxSource } from './weight-loader-mlx.ts'

/** Which of the three expert projections. */
export type SlabProj = 'gate' | 'up' | 'down'
/** Which member of the MLX-affine trio: weight, scales, biases. */
export type SlabKind = 'w' | 's' | 'b'

export interface SlabSource {
  /** Byte length of one expert's slab for this tensor. */
  slabBytes(layer: number, proj: SlabProj, kind: SlabKind): number
  /** One expert's bytes, ready to writeBuffer into a pool slot. */
  read(layer: number, proj: SlabProj, kind: SlabKind, expert: number): Promise<Uint8Array>
}

/**
 * Where a record lives. Structurally what `openMlxCheckpoint`'s `locate`
 * returns (that module keeps its `RecordLoc` private), so the loader's own
 * resolver is passed straight in.
 */
export interface RecordLocation {
  file: string
  /** Absolute byte offset in the shard file — the header is already added. */
  begin: number
  end: number
  info: TensorInfo
}

export type Locate = (record: string) => RecordLocation

// ============================================================
// geometry — the whole correctness surface
// ============================================================

/** safetensors dtype -> bytes per element. An unlisted dtype throws rather
 *  than defaulting: a guessed width is a wrong stride, and a wrong stride is
 *  the silent-garbage failure this module exists to prevent. */
const DTYPE_BYTES: Record<string, number> = {
  BOOL: 1, U8: 1, I8: 1,
  U16: 2, I16: 2, F16: 2, BF16: 2,
  U32: 4, I32: 4, F32: 4,
  U64: 8, I64: 8, F64: 8,
}

interface SlabGeometry {
  /** ROUTED experts, from the stacked tensor's own shape[0]. */
  experts: number
  /** Bytes of one expert's slab. */
  stride: number
  stacked: RecordLocation
  /** The shared expert's record, or null when the checkpoint has none. */
  shared: RecordLocation | null
  /** OPFS key of the built buffer holding stacked ++ shared. */
  key: string
  /** Byte length that built buffer must have. */
  builtBytes: number
  /** What the plan does to these bytes on their way to the GPU. Already
   *  applied in the OPFS buffer; applied per slab on the fetch path. */
  convert: Convert
  /** Source dtype, which decides whether `convert` is a no-op (see
   *  buildBuffer: an MLX checkpoint may ship scales F16 already). */
  dtype: string
}

/**
 * Derive one tensor's slab geometry from the checkpoint header.
 *
 * The record NAMES come from `planLayer` rather than being spelled again here:
 * the OPFS file this reads is the buffer that same plan produced, so a naming
 * drift that would silently point the two at different tensors cannot happen.
 */
function slabGeometry(
  spec: ModelSpec,
  locate: Locate,
  layer: number,
  proj: SlabProj,
  kind: SlabKind,
): SlabGeometry {
  const name = `moe_${proj}_proj_${kind}`
  const plan = planLayer(spec, layer).find((p) => p.name === name)
  if (!plan) {
    throw new Error(`slab-source: ${spec.id} layer ${layer} has no '${name}' buffer — not a MoE layer`)
  }
  if (plan.op) throw new Error(`slab-source: '${name}' carries a ${plan.op.kind} op; its bytes are not the stacked tensor`)
  if (plan.parts.length > 2) {
    throw new Error(`slab-source: '${name}' has ${plan.parts.length} parts, expected [stacked] or [stacked, shared]`)
  }

  const stacked = locate(plan.parts[0].record)
  const shape = stacked.info.shape
  if (shape.length !== 3) {
    throw new Error(`slab-source: ${plan.parts[0].record} has shape [${shape}], expected [experts, N, K]`)
  }
  const elem = DTYPE_BYTES[stacked.info.dtype]
  if (!elem) throw new Error(`slab-source: ${plan.parts[0].record} has unsupported dtype ${stacked.info.dtype}`)
  const experts = shape[0]
  const stride = shape[1] * shape[2] * elem
  // The header states the byte range AND the shape; they are independent
  // fields and must agree. When they do not, one of the two is being read
  // wrong, and every offset below would be plausible and wrong.
  const declared = stacked.end - stacked.begin
  if (declared !== experts * stride) {
    throw new Error(
      `slab-source: ${plan.parts[0].record} declares ${declared} B but `
      + `[${shape}] at ${stacked.info.dtype} is ${experts * stride} B`)
  }
  // The router has one row per expert; a stack of a different depth means the
  // spec and the checkpoint disagree about which model this is.
  if (spec.moe && spec.moe.experts !== experts) {
    throw new Error(
      `slab-source: ${spec.id} declares ${spec.moe.experts} experts but `
      + `${plan.parts[0].record} is stacked ${experts} deep`)
  }

  let shared: RecordLocation | null = null
  if (plan.parts.length === 2) {
    shared = locate(plan.parts[1].record)
    const s = shared.info.shape
    // Stacking as index E only works at identical row width — MoeDims says so
    // in prose, and this is where it is enforced against the actual bytes.
    if (s.length !== 2 || s[0] !== shape[1] || s[1] !== shape[2] || shared.info.dtype !== stacked.info.dtype) {
      throw new Error(
        `slab-source: ${plan.parts[1].record} is [${s}] ${shared.info.dtype}, `
        + `cannot stack as expert ${experts} of [${shape}] ${stacked.info.dtype}`)
    }
    if (shared.end - shared.begin !== stride) {
      throw new Error(
        `slab-source: ${plan.parts[1].record} declares ${shared.end - shared.begin} B, `
        + `expected one slab of ${stride} B`)
    }
    if (plan.parts[1].convert !== plan.parts[0].convert) {
      throw new Error(`slab-source: '${name}' converts its two parts differently `
        + `(${plan.parts[0].convert} vs ${plan.parts[1].convert})`)
    }
  }

  // Only the width-preserving conversions can be applied one slab at a time
  // and leave `stride` meaning what it says. MoE plans use exactly these two;
  // a `bf16->f32` part would double the built bytes and needs its own audit.
  const convert = plan.parts[0].convert
  if (convert !== 'raw' && convert !== 'bf16->f16') {
    throw new Error(`slab-source: '${name}' converts ${convert}, which changes the slab's byte length`)
  }

  return {
    experts, stride, stacked, shared,
    key: planKey(plan, layer),
    builtBytes: (experts + (shared ? 1 : 0)) * stride,
    convert,
    dtype: stacked.info.dtype,
  }
}

/**
 * The plan's conversion, applied to ONE slab.
 *
 * Same three cases `buildBuffer` runs, and the same rule on overflow: a
 * clamped scale is a wrong weight, not a rounding error, and it surfaces as
 * NaN logits thousands of dispatches later.
 */
function convertSlab(g: SlabGeometry, raw: Uint8Array): Uint8Array {
  if (g.convert === 'raw') return raw
  if (g.dtype === 'F16') return raw                 // already what the kernels read
  if (g.dtype !== 'BF16') {
    throw new Error(`slab-source: ${g.stacked.file} slab is ${g.dtype}, expected BF16 for ${g.convert}`)
  }
  const { out, overflow } = bf16ToF16(new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength / 2))
  if (overflow > 0) {
    throw new Error(`slab-source: ${overflow} bf16 value(s) in this slab exceed f16 range — `
      + 'these weights cannot be represented at f16 and the model would run wrong')
  }
  return new Uint8Array(out.buffer)
}

/** Cache the derivation: `planLayer` rebuilds a whole layer's plans per call,
 *  and this is on the miss path of every token. */
function geometryCache(spec: ModelSpec, locate: Locate) {
  const cache = new Map<string, SlabGeometry>()
  return (layer: number, proj: SlabProj, kind: SlabKind): SlabGeometry => {
    const k = `${layer}.${proj}.${kind}`
    let g = cache.get(k)
    if (!g) cache.set(k, (g = slabGeometry(spec, locate, layer, proj, kind)))
    return g
  }
}

/** Slab index -> byte offset within the logical stack, shared expert included. */
function slabOffset(g: SlabGeometry, expert: number): number {
  const slabs = g.experts + (g.shared ? 1 : 0)
  if (!Number.isInteger(expert) || expert < 0 || expert >= slabs) {
    throw new Error(`slab-source: expert ${expert} is outside 0..${slabs - 1}`
      + (g.shared ? ` (${g.experts} routed + shared)` : ` (${g.experts} routed, no shared expert)`))
  }
  return expert * g.stride
}

// ============================================================
// OPFS — the browser path
// ============================================================

/**
 * The slice of OPFS this needs, declared structurally so a real
 * `FileSystemDirectoryHandle` satisfies it and a test can stand in for one
 * without a browser. Only reads appear here — the loader owns the writes.
 */
export interface SlabSyncHandle {
  getSize(): number
  read(buffer: Uint8Array, options: { at: number }): number
  close(): void
}

export interface SlabFileHandle {
  /** Present in OPFS. The fast path: no File snapshot, no per-read copy. */
  createSyncAccessHandle?(): Promise<SlabSyncHandle>
  getFile(): Promise<{
    size: number
    slice(begin: number, end: number): { arrayBuffer(): Promise<ArrayBuffer> }
  }>
}

export interface SlabDirectory {
  getFileHandle(name: string): Promise<SlabFileHandle>
}

/** Same flattening `weight-loader.ts`'s private `opfsKey` applies before it
 *  writes. Plan keys are already in this alphabet, so it is the identity
 *  today; it is applied anyway so the two cannot drift apart. */
function opfsName(key: string): string {
  return key.replace(/[^A-Za-z0-9._-]/g, '_')
}

/**
 * Read slabs out of the OPFS weight cache the MLX loader already wrote.
 *
 * A sync access handle is used where the runtime has one — it reads straight
 * into a caller-owned buffer, where `getFile()` snapshots the file and slices
 * a fresh Blob per read. The handle holds an EXCLUSIVE lock on the file, so
 * build this after the load has finished writing (a lock conflict is not
 * fatal: `createSyncAccessHandle` rejects and this falls back to `getFile`
 * for the rest of the session). `close()` releases every handle it holds.
 *
 * There is no fetch fallback here. A missing cache entry throws, because the
 * alternative is a per-miss network round trip inside a decode loop that has
 * budgeted 1 ms for a local read — the caller picks the source, not this.
 */
export function opfsSlabSource(
  spec: ModelSpec,
  locate: Locate,
  dir: SlabDirectory,
): SlabSource & { close(): void } {
  const geo = geometryCache(spec, locate)
  const files = new Map<string, Promise<SlabFileHandle>>()
  const syncs = new Map<string, SlabSyncHandle>()
  let syncUsable = true

  const fileFor = (key: string): Promise<SlabFileHandle> => {
    let f = files.get(key)
    if (!f) {
      f = dir.getFileHandle(opfsName(key)).catch((e: unknown) => {
        files.delete(key)
        throw new Error(`slab-source: OPFS has no cached buffer '${opfsName(key)}' (${e})`)
      })
      files.set(key, f)
    }
    return f
  }

  const syncFor = async (key: string): Promise<SlabSyncHandle | null> => {
    if (!syncUsable) return null
    const open = syncs.get(key)
    if (open) return open
    const fh = await fileFor(key)
    if (!fh.createSyncAccessHandle) { syncUsable = false; return null }
    try {
      const h = await fh.createSyncAccessHandle()
      syncs.set(key, h)
      return h
    } catch {
      // Locked by a writer, or a runtime that only allows this in a worker.
      syncUsable = false
      return null
    }
  }

  return {
    slabBytes: (layer, proj, kind) => geo(layer, proj, kind).stride,

    read: async (layer, proj, kind, expert) => {
      const g = geo(layer, proj, kind)
      const at = slabOffset(g, expert)
      const sync = await syncFor(g.key)
      if (sync) {
        // A cache file of the wrong size is a stale or torn entry, and every
        // offset in it would be believable. Check before reading, not after.
        const size = sync.getSize()
        if (size !== g.builtBytes) {
          throw new Error(`slab-source: ${g.key} is ${size} B, expected ${g.builtBytes} B `
            + `(${g.experts}${g.shared ? '+1' : ''} slabs of ${g.stride} B)`)
        }
        const out = new Uint8Array(new ArrayBuffer(g.stride))
        const n = sync.read(out, { at })
        if (n !== g.stride) throw new Error(`slab-source: ${g.key} read ${n} of ${g.stride} B at ${at}`)
        return out
      }
      const file = await (await fileFor(g.key)).getFile()
      if (file.size !== g.builtBytes) {
        throw new Error(`slab-source: ${g.key} is ${file.size} B, expected ${g.builtBytes} B `
          + `(${g.experts}${g.shared ? '+1' : ''} slabs of ${g.stride} B)`)
      }
      const buf = await file.slice(at, at + g.stride).arrayBuffer()
      if (buf.byteLength !== g.stride) {
        throw new Error(`slab-source: ${g.key} returned ${buf.byteLength} of ${g.stride} B at ${at}`)
      }
      return new Uint8Array(buf)
    },

    close: () => {
      for (const h of syncs.values()) { try { h.close() } catch { /* already gone */ } }
      syncs.clear()
      files.clear()
    },
  }
}

// ============================================================
// HTTP byte range — dev mirror / HuggingFace
// ============================================================

/**
 * Read slabs straight out of the checkpoint's shards.
 *
 * `src` is the loader's own ranged reader (`MlxSource.range`, which carries
 * the retry/backoff and refuses a 200 answer to a Range request), so this adds
 * only the offset arithmetic. The shared expert is a SEPARATE record here — it
 * is a separate tensor in the checkpoint and may live in another shard — where
 * in the OPFS buffer it is the appended tail; `slabGeometry` has already
 * proved the two are the same shape, so the caller sees one index space.
 */
export function fetchSlabSource(
  spec: ModelSpec,
  locate: Locate,
  src: Pick<MlxSource, 'range'>,
): SlabSource {
  const geo = geometryCache(spec, locate)
  return {
    slabBytes: (layer, proj, kind) => geo(layer, proj, kind).stride,

    read: async (layer, proj, kind, expert) => {
      const g = geo(layer, proj, kind)
      const at = slabOffset(g, expert)
      const routed = expert < g.experts
      const rec = routed ? g.stacked : g.shared!
      // The stacked tensor is sliced at the expert axis; the shared record IS
      // one slab, so its own begin is the offset.
      const begin = routed ? g.stacked.begin + at : rec.begin
      const bytes = await src.range(rec.file, begin, begin + g.stride)
      // MlxSource's own implementation checks this too. Repeated because it is
      // this module's contract, and because a caller may pass a plainer reader.
      if (bytes.byteLength !== g.stride) {
        throw new Error(`slab-source: ${rec.file} returned ${bytes.byteLength} of ${g.stride} B `
          + `at ${begin} (layer ${layer} ${proj} ${kind} expert ${expert})`)
      }
      // The OPFS buffer is already converted; this is where the fetch path
      // catches up, so the two sources are interchangeable byte for byte.
      return convertSlab(g, bytes)
    },
  }
}
