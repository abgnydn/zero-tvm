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
 * Imports the loaders, which read `GPUBufferUsage` at module scope — import
 * this module dynamically from any page that must still render without WebGPU.
 */

import type { ModelSpec } from '../compiler/model-spec.js'
import { opfsDirFor } from './weight-loader.js'
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
      // The spec's own manifest, not a hardcoded name — Qwen3.5's is
      // tensor-cache.json, and hardcoding ndarray-cache.json re-showed its
      // gate on every visit no matter what was cached.
      await dir.getFileHandle(spec.manifestName ?? 'ndarray-cache.json')
    }
    return true
  } catch {
    return false
  }
}
