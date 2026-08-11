/**
 * MODEL REGISTRY — the single table every "which models exist" surface renders
 * from: the landing page's model cards, the chat header's switcher, the gate
 * dialog's branding, and the `?model=` URL mapping itself.
 *
 * This exists so the landing page CANNOT go stale: adding a spec here is the
 * whole act of shipping it to every surface. (The old landing hand-wrote its
 * model list and stats into the HTML; after Qwen3.6 shipped, twelve separate
 * claims on that page were wrong.)
 *
 * scripts/add-model.mjs appends rows at the ADD-MODEL markers below after its
 * constraint check and compile gate pass — a generated model reaches every
 * surface through exactly this file. Numerical trust comes separately:
 * scripts/validate-model.mjs diffs a registered model against mlx_lm before
 * its card is worth believing.
 *
 * DELIBERATELY LIGHT: imports only model-spec.ts, which is dependency-free —
 * with the explicit .ts extension, which is load-bearing: scripts/
 * validate-model.mjs imports this file under Node type stripping (no
 * extension searching) to resolve a `?model=` param to its spec.
 * The landing page runs this in browsers WITHOUT WebGPU (where it must still
 * render cards and say "needs WebGPU") — and weight-loader.ts, the previous
 * home of modelBranding's neighbours, reads `GPUBufferUsage` at module scope,
 * which throws at import time on such browsers.
 */

import {
  makeModelSpec,
  PHI3, QWEN3_4B, QWEN35_4B, QWEN36_35B_A3B, QWEN36_35B_A3B_Q3,
  type ModelSpec,
} from '../compiler/model-spec.ts'
import { QWEN3_4B_MLX } from '../compiler/model-spec.ts'
import { LLAMA_3_2_1B_INSTRUCT_4BIT } from '../compiler/model-spec.ts'
import { QWEN3_30B_A3B_4BIT } from '../compiler/model-spec.ts'
import { QWEN3_EMBEDDING_06B } from '../compiler/model-spec.ts'
// ADD-MODEL:IMPORTS

/** URL `?model=` value for each spec; Phi-3 is the no-flag default. */
export const SHIPPED_MODELS: ReadonlyArray<{ param: string; spec: ModelSpec }> = [
  { param: '', spec: PHI3 },
  { param: 'qwen3', spec: QWEN3_4B },
  { param: 'qwen35', spec: QWEN35_4B },
  // The 35B MoE ships twice: 3-bit experts first (the build most machines can
  // actually run — ~20 GB free RAM), full 4-bit for the boxes that can.
  { param: 'qwen36q3', spec: QWEN36_35B_A3B_Q3 },
  { param: 'qwen36', spec: QWEN36_35B_A3B },
  { param: 'qwen3mlx', spec: QWEN3_4B_MLX },
  { param: 'llama32', spec: LLAMA_3_2_1B_INSTRUCT_4BIT },
  { param: 'qwen30b', spec: QWEN3_30B_A3B_4BIT },
  // Not a chat model: ?model=embed serves forwardEmbedding, not generation.
  { param: 'embed', spec: QWEN3_EMBEDDING_06B },
  // ADD-MODEL:MODELS
]

/** `?model=<param>` → spec; unknown values (and no flag) boot Phi-3, so every
 *  pre-registry URL keeps its exact behavior. */
export function specForParam(model: string | null): ModelSpec {
  const hit = model && SHIPPED_MODELS.find((m) => m.param === model)
  return hit ? hit.spec : PHI3
}

/**
 * `?ctx=N` — override the KV budget, in tokens. The compiled maxPages is a
 * BUDGET CHOICE (a flat ~1 GiB constant from 2026-07-30), not a hardware or
 * model limit: our KV cost/token is ~3x LM Studio's per-token working set on
 * the same checkpoint, and maxSeq is far above the default on every spec but
 * Phi-3. Rebuilding through makeModelSpec keeps every derived field
 * (maxContext, cache byte sizes, page tables) coherent — a raw maxContext
 * edit would desynchronise them silently.
 *
 * Clamped to maxSeq: past the trained window the model degrades without
 * erroring, which is a worse failure than refusing. Floor is one page. KV is
 * allocated EAGERLY, one buffer per attention layer — a ctx the machine
 * cannot hold fails at boot (or, on unified memory, can kill the GPU process
 * mid-run once physical RAM is actually exhausted: Metal commits lazily, so
 * allocation success is not a promise). An expert knob, not a default.
 *
 * Lives HERE rather than in model-select.ts so it is testable headlessly —
 * model-select imports weight-loader, which reads GPUBufferUsage at module
 * scope and cannot load under Node.
 */
export function specWithCtx(base: ModelSpec, ctx: number): ModelSpec {
  if (!Number.isFinite(ctx) || ctx <= 0) return base
  const pages = Math.max(1, Math.min(
    Math.ceil(ctx / base.pageSize),
    Math.floor(base.maxSeq / base.pageSize)))
  if (pages === base.maxPages) return base
  return makeModelSpec({ ...base, maxPages: pages })
}

/** Root OPFS directory the weight cache lives under. */
export const WEIGHTS_OPFS_DIR = 'zero-tvm-weights'

/**
 * Per-model OPFS dir. Phi-3 keeps the historical unsuffixed name for cache
 * compatibility; other models get a `.`-joined suffix. The separator MUST NOT
 * be `-`: openOPFS() deletes stale `zero-tvm-weights-<rev>` dirs left by old
 * loader revisions, and a dash-suffixed model dir would match that sweep.
 *
 * Lives HERE rather than in weight-loader.ts (which re-exports it) because
 * peer-weights.ts replicates this directory on machines that may have no
 * WebGPU at all — and weight-loader reads GPUBufferUsage at module scope.
 */
export function opfsDirFor(spec: ModelSpec): string {
  return spec.id === PHI3.id ? WEIGHTS_OPFS_DIR : `${WEIGHTS_OPFS_DIR}.${spec.id}`
}

export interface ModelBrand {
  name: string
  /** Parameter-count line for cards ("4B hybrid", "35B-A3B MoE"). */
  params: string
  sizeLabel: string
  /** Measured decode/total rate, or '' when no honest number exists. */
  rateLabel: string
  /** Present when the model needs more memory than a typical machine has. */
  ramNote?: string
}

/** Page-facing copy per spec id (gate dialog, header, landing cards).
 *  rateLabels are measured numbers only — see BENCH.md for the protocol each
 *  one came from; '' means no protocol-quality number exists yet. */
const BRANDINGS: Record<string, ModelBrand> = {
  [QWEN36_35B_A3B_Q3.id]: {
    // ~55 t/s measured by the machine owner on a quiet 32 GB M2 Max
    // (2026-08-05, single session — not yet a protocol round).
    //
    // NO ACCURACY EVIDENCE. Every other entry below names a validation record;
    // this one cannot. The 3-bit choice rests on a block-output cosine (0.936),
    // which is fidelity and cannot see quality, and there is no e2e test file
    // for either MoE build — so nothing anywhere has ever checked this model's
    // output. scripts/quality-ab.py against the 4-bit build is the missing run.
    // See docs/QUALITY.md.
    name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · 3-bit experts',
    sizeLabel: '~16.4 GB', rateLabel: '~55 t/s',
    ramNote: 'needs ~20 GB free RAM',
  },
  [QWEN36_35B_A3B.id]: {
    name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · full 4-bit',
    sizeLabel: '~19.5 GB', rateLabel: '',
    ramNote: 'needs ~24 GB free RAM (64 GB Mac recommended)',
  },
  [QWEN35_4B.id]: { name: 'Qwen3.5-4B', params: '4B hybrid (DeltaNet)', sizeLabel: '~2.6 GB', rateLabel: '~65 t/s' },  // 65.28 total, M2 Max (BENCH.md 2026-07-30)
  [QWEN3_4B.id]: { name: 'Qwen3-4B', params: '4B dense · q4f16_1', sizeLabel: '~2.3 GB', rateLabel: '~60 t/s' },  // 59.85 total, M2 Max (BENCH.md 2026-07-30)
  [PHI3.id]: { name: 'Phi-3-mini', params: '3.8B dense', sizeLabel: '~2 GB', rateLabel: '~70 t/s' },  // 69.55 total, M2 Max (BENCH.md 2026-07-30)
  // The first pipeline-added model (scripts/add-model.mjs, 2026-08-06) — the
  // same Qwen3-4B as the MLC build above, from the MLX-affine checkpoint;
  // validated against mlx_lm (cosine 0.999879, greedy token-exact).
  [QWEN3_4B_MLX.id]: { name: 'Qwen3-4B', params: '4B dense · MLX 4-bit', sizeLabel: '~2.1 GB', rateLabel: '' },
  [LLAMA_3_2_1B_INSTRUCT_4BIT.id]: { name: 'Llama-3.2-1B-Instruct', params: '1B dense', sizeLabel: '~0.6 GB', rateLabel: '' },
  // First MoE WITHOUT a shared expert, and the first 4-bit router — both are
  // spec flags now rather than engine assumptions. Validated against mlx_lm in
  // f32: cosine 0.999985, greedy token-exact through the stop id. (Against
  // mlx's own bf16 forward it scores 0.9978 — that is bf16's error, not the
  // engine's; see the dtype note in scripts/mlx-ref.py.)
  [QWEN3_30B_A3B_4BIT.id]: { name: 'Qwen3-30B-A3B', params: '30B-A3B MoE', sizeLabel: '~16.0 GB', rateLabel: '', ramNote: 'needs ~20 GB free RAM' },
  // Not a chat model. Its output is the pooled hidden state, so rateLabel stays
  // '' — tok/s is not the unit here.
  [QWEN3_EMBEDDING_06B.id]: { name: 'Qwen3-Embedding-0.6B', params: '0.6B embedding · last-token pooled', sizeLabel: '~0.35 GB', rateLabel: '' },
  // ADD-MODEL:BRANDINGS
}

/** Page-facing copy for a model; unknown ids fall back to Phi-3 (the default
 *  spec), preserving the old switch's default arm. */
export function modelBranding(spec: ModelSpec): ModelBrand {
  return BRANDINGS[spec.id] ?? BRANDINGS[PHI3.id]
}
