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
import { opfsDirFor, opfsKey } from './weight-loader.js'
import { planModel, planKey } from './weight-loader-mlx.js'

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
    return false
  }
}
