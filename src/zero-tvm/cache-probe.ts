/**
 * IS THIS MODEL ALREADY ON THIS DEVICE?
 *
 * Two surfaces ask: the landing picker (to badge a card) and the chat page (to
 * skip the download gate). They asked it in two places with two copies of the
 * answer, and both copies were wrong the same way — they looked for the MLC
 * manifest, which the MLX loader never writes. It returns before that line and
 * caches BUILT BUFFERS keyed by plan name instead. So every MLX model, seven
 * of the ten shipped and all the large ones, reported "not cached" however
 * much of it was on disk, and re-showed a 16 GB download gate on every visit.
 *
 * One copy now, because the failure is silent in both directions: a wrong
 * answer here shows a download that is not needed, or hides one that is.
 *
 * This module used to be unsafe to import statically on a browser without
 * WebGPU, because the loaders read `GPUBufferUsage` at module scope. Those
 * reads are lazy now (weight-loader.ts, engine-core.ts, compiler.ts), so a
 * static import is fine — nothing here touches a WebGPU global until a device
 * exists.
 */

import type { ModelSpec } from '../compiler/model-spec.js'
import { opfsDirFor, opfsKey, localMirrorBase } from './weight-loader.js'
import { planModel, planKey } from './weight-loader-mlx.js'

/**
 * DEV ONLY: is the vite mirror primed for this model? When the mirror serves
 * the weights, the loader deliberately writes NO OPFS copy (it would be a
 * second local copy of up to 19.5 GB) — so in dev, OPFS silence is not
 * absence, and every mirror-primed MLX model showed "not on this device"
 * forever. A primed mirror IS on this device: boot reads it at disk speed,
 * which is exactly what the badge promises.
 *
 * MLX only, probed by the safetensors index name: the middleware falls
 * unknown repos through to the default Phi-3 snapshot, an MLC dir that can
 * never answer 200 for an MLX-only filename — so no false positives. MLC
 * models keep pure-OPFS semantics. localMirrorBase is null in prod builds,
 * so this whole path compiles away to a no-op there.
 */
async function mirrorHasModel(spec: ModelSpec): Promise<boolean> {
  if (spec.weightFormat !== 'mlx-safetensors') return false
  const mirror = localMirrorBase(spec)
  if (!mirror) return false
  try {
    return (await fetch(mirror + 'model.safetensors.index.json', { method: 'HEAD' })).ok
  } catch {
    return false
  }
}

export async function isModelCached(spec: ModelSpec): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return false
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle(opfsDirFor(spec))
    if (spec.weightFormat === 'mlx-safetensors') {
      // The LAST plan, not the first: it is written last, so its presence means
      // the download FINISHED rather than started. A half-cached model that
      // reported cached would skip the gate and then stall on a fetch.
      const plans = planModel(spec)
      const last = plans[plans.length - 1]
      await dir.getFileHandle(planKey(last.plan, last.layer))
    } else {
      // The manifest is written FIRST, so its presence means the download
      // STARTED. Checking only for it is the exact hazard the MLX branch above
      // was written to avoid, and the MLC path — which serves Phi-3, the
      // default model — had it: one interrupted download, or one visit to a
      // page that boots the engine on load, leaves ndarray-cache.json behind
      // and the chat gate never appears again. The visitor goes straight to
      // "Downloading…" with no consent, on every later visit.
      //
      // The spec's own manifest name, not a hardcoded one — Qwen3.5's repo
      // ships tensor-cache.json.
      const fh = await dir.getFileHandle(spec.manifestName ?? 'ndarray-cache.json')
      const manifest = JSON.parse(await (await fh.getFile()).text()) as {
        records?: { dataPath?: string }[]
      }
      // Every shard the manifest names must be on disk, not just the last:
      // shards are fetched with bounded PARALLELISM, so completion order is
      // not manifest order and "the last one exists" proves nothing here.
      const shards = [...new Set((manifest.records ?? [])
        .map((r) => r.dataPath).filter((p): p is string => !!p))]
      if (shards.length === 0) return false
      for (const p of shards) await dir.getFileHandle(opfsKey(p))
    }
    return true
  } catch {
    return await mirrorHasModel(spec)
  }
}
