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
import { QWEN3_5_9B_MLX_4BIT } from '../compiler/model-spec.ts'
import { QWEN3_8_27B_4BIT } from '../compiler/model-spec.ts'
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
  { param: 'qwen35mlx', spec: QWEN3_5_9B_MLX_4BIT },
  { param: 'qwen38', spec: QWEN3_8_27B_4BIT },
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
/**
 * KV bytes per token AS ALLOCATED — the number the entrance turns into "will
 * this fit".
 *
 * Lives here rather than in landing.ts because landing.ts touches the DOM and
 * cannot be imported by a unit test. It was there, a test re-implemented the
 * formula to check it, and the mutation gate caught that immediately: mutating
 * the real one left the copy passing. A figure this load-bearing has to be
 * tested at the definition, not next to it.
 *
 * int8 is the default (`?kv8=0` opts out): half the f16 width plus one f16
 * scale per (token, kv head, side) PER ATTENTION LAYER — allocKVPagesInt8 makes
 * one scales buffer per attention layer, and leaving the layer count out
 * understated Phi-3 by 2% in the reassuring direction. MLA caches a latent and
 * the int8 path does not cover it, so it stays f16.
 */
export function kvBytesPerTokenShown(spec: ModelSpec, int8: boolean): number {
  return int8 && !spec.mla
    ? spec.kvBytesPerToken / 2
      + 4 * spec.kvHeads * spec.layerKinds.filter((k) => k === 'attn').length
    : spec.kvBytesPerToken
}

/**
 * How this build is quantised, in the checkpoint's own terms.
 *
 * The sheet used to say nothing here unless a model shipped TWO builds, because
 * the only place quantisation appeared was the variant picker — and a picker
 * needs a choice. So five of seven groups displayed no quantisation at all,
 * which reads as "not quantised". Every model on this roster is 4-bit; the
 * engine has no unquantised path (constraints.ts refuses f16/bf16 weights).
 *
 * DERIVED, never authored: a hand-typed label drifts from the checkpoint the
 * moment a variant is added, and this is a claim about what the user is about
 * to download.
 */
export function quantLabel(spec: ModelSpec): string {
  const mlx = spec.weightFormat === 'mlx-safetensors'
  const fmt = mlx ? 'MLX affine · group 64' : 'MLC q4f16_1 · group 32'
  // The 3-bit build requantises the EXPERT STACKS only; everything else —
  // attention, router, the shared expert — stays at 4-bit, and a flat "3-bit"
  // would overstate what was given up.
  return spec.moe?.bits === 3 ? `3-bit experts, 4-bit elsewhere · ${fmt}` : `4-bit · ${fmt}`
}

/** OPFS weight directory for a spec.
 *
 *  Phi-3 keeps the bare directory name it shipped with, so no existing cache is
 *  orphaned. `weightsRevision`, when a spec sets it, is part of the path: a
 *  reconverted checkpoint under an unchanged repo name lands in a NEW directory
 *  rather than being read as the old one. Without that, re-uploading a build
 *  silently serves every warm client stale tensors — the loader's size check
 *  cannot see it, because a clean reconvert has the same byte count. */
export function opfsDirFor(spec: ModelSpec): string {
  const rev = spec.weightsRevision ? `.${spec.weightsRevision}` : ''
  // Phi-3 keeps the bare historical name, but it must still honour a revision if
  // one is ever set — dropping it there would be a silent no-op on the one spec
  // whose directory nothing else distinguishes.
  return spec.id === PHI3.id ? `${WEIGHTS_OPFS_DIR}${rev}` : `${WEIGHTS_OPFS_DIR}.${spec.id}${rev}`
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
  /** A MEASURED quality cost this build pays for its size, shown on the sheet.
   *  The 3-bit 35B is offered first in its group and looks strictly better than
   *  the 4-bit on every figure the card renders — size, speed, RAM. It is not:
   *  it is +10.4% perplexity. A decision surface that shows only the flattering
   *  half of a trade is not neutral, and this is where the roster records what
   *  a variant costs. */
  qualityNote?: string
  /** Spec generated and compile-gated but NOT yet numerics-validated against
   *  its reference (mlx_lm logits + greedy). The entrance hides pending
   *  characters — a live roster card is a claim the model runs, and this
   *  registry does not claim what it has not checked. ?model= still boots it
   *  for exactly that validation. Remove the flag when validate-model passes. */
  pending?: true
  /** Memory builds for the landing character screen — expert-pool presets,
   *  measured where a measurement exists (BENCH.md 2026-08-15 AC price
   *  curve; est = derived from the expert fraction, not measured). slots=0
   *  is the full model. ONE source for every page: the landing renders these
   *  and the chat boots them via ?pool=, so the two cannot disagree. */
  poolModes?: readonly { slots: number; label: string; note?: string }[]
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
    // MEASURED 2026-08-14, and it costs something: +10.4% perplexity against
    // the 4-bit build (26.179 -> 28.908, paired z = 18.5, worse on 24 of 24
    // windows — scripts/quality-ab.py, BENCH.md). The harness calls that
    // MARGINAL: real, but perplexity is the wrong instrument to decide it and
    // no task benchmark has been run on either build. It buys 15 GB against
    // 19 GB, which is the difference between running and not running on a
    // 32 GB machine. See docs/QUALITY.md.
    name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · 3-bit experts',
    sizeLabel: '~16.4 GB download', rateLabel: '~66 t/s',
    qualityNote: '+10.4% perplexity against the 4-bit build (paired, worse on 24 of 24 windows) — what the 3 GB saves costs',
    poolModes: [
      { slots: 0, label: 'Full · 15.7 GB resident · ~66 t/s' },
      { slots: 128, label: 'Half · ~8.4 GB resident · ~15 t/s', note: 'pooled rates are the 2026-08-15 AC round, whose own unpooled control read 58.6 t/s — compare against that, not the 66 above' },
      { slots: 64, label: 'Quarter · ~4.8 GB resident · ~12 t/s', note: 'long prompts run token-by-token' },
    ],   // 65.56 total, M2 Max (BENCH.md 2026-08-13)
    ramNote: 'needs ~20 GB free RAM',
  },
  [QWEN36_35B_A3B.id]: {
    name: 'Qwen3.6-35B-A3B', params: '35B-A3B MoE · full 4-bit',
    sizeLabel: '~19.5 GB download',
    poolModes: [
      { slots: 0, label: 'Full · 19.7 GB resident' },
      // No rate: the 4-bit build has never been measured pooled (BENCH.md
      // records it as unmeasured) — the ~15 that used to sit here was the q3
      // build's number wearing the wrong card. Lens round 2026-08-17.
      { slots: 128, label: 'Half · ~11 GB est' },
      { slots: 64, label: 'Quarter · ~6 GB est', note: 'long prompts run token-by-token' },
    ], rateLabel: '',
    ramNote: 'needs ~24 GB free RAM (64 GB Mac recommended)',
  },
  // THESE THREE ARE FROM AN OLDER BUILD (2026-07-30) than the labels below it,
  // which were re-measured 2026-08-13. Every kernel round since — E5 prefill,
  // the affine wide loads, MoE chunking — lands in the newer numbers and not
  // in these, so reading the column as a cross-model ranking overstates the
  // MLX builds. Refreshing them means re-downloading Phi-3 and Qwen3-4B from
  // HF (neither is in .weights-local); qwen35 IS local and can be redone alone.
  [QWEN35_4B.id]: { name: 'Qwen3.5-4B', params: '4B hybrid (DeltaNet)', sizeLabel: '~2.6 GB', rateLabel: '~65 t/s' },  // 65.28 total, M2 Max (BENCH.md 2026-07-30)
  [QWEN3_4B.id]: { name: 'Qwen3-4B', params: '4B dense · q4f16_1', sizeLabel: '~2.3 GB', rateLabel: '~60 t/s' },  // 59.85 total, M2 Max (BENCH.md 2026-07-30)
  [PHI3.id]: { name: 'Phi-3-mini', params: '3.8B dense', sizeLabel: '~2 GB', rateLabel: '~70 t/s' },  // 69.55 total, M2 Max (BENCH.md 2026-07-30)
  // The first pipeline-added model (scripts/add-model.mjs, 2026-08-06) — the
  // same Qwen3-4B as the MLC build above, from the MLX-affine checkpoint;
  // validated against mlx_lm (cosine 0.999879, greedy token-exact).
  [QWEN3_4B_MLX.id]: { name: 'Qwen3-4B', params: '4B dense · MLX 4-bit', sizeLabel: '~2.1 GB', rateLabel: '~81 t/s' },  // 80.78 total, M2 Max (BENCH.md 2026-08-13)
  [LLAMA_3_2_1B_INSTRUCT_4BIT.id]: { name: 'Llama-3.2-1B-Instruct', params: '1B dense', sizeLabel: '~0.6 GB', rateLabel: '~256 t/s' },  // 255.90 total, M2 Max (BENCH.md 2026-08-13)
  // First MoE WITHOUT a shared expert, and the first 4-bit router — both are
  // spec flags now rather than engine assumptions. Validated against mlx_lm in
  // f32: cosine 0.999985, greedy token-exact through the stop id. (Against
  // mlx's own bf16 forward it scores 0.9978 — that is bf16's error, not the
  // engine's; see the dtype note in scripts/mlx-ref.py.)
  [QWEN3_30B_A3B_4BIT.id]: { name: 'Qwen3-30B-A3B', params: '30B-A3B MoE', sizeLabel: '~17.2 GB', rateLabel: '~75 t/s', ramNote: 'needs ~20 GB free RAM',
    poolModes: [
      { slots: 0, label: 'Full · ~17 GB · ~75 t/s' },
      { slots: 96, label: '¾ · ~13 GB est' },
      { slots: 64, label: 'Half · ~9.5 GB est', note: 'long prompts run token-by-token' },
    ] },  // 74.96 total, M2 Max (BENCH.md 2026-08-13); size = index.json total_size 17.17 GB — the old ~16.0 was short by 1.2 GB
  // Not a chat model. Its output is the pooled hidden state, so rateLabel stays
  // '' — tok/s is not the unit here.
  [QWEN3_EMBEDDING_06B.id]: { name: 'Qwen3-Embedding-0.6B', params: '0.6B embedding · last-token pooled', sizeLabel: '~0.35 GB', rateLabel: '' },
  [QWEN3_5_9B_MLX_4BIT.id]: { name: 'Qwen3.5-9B', params: '9B hybrid (DeltaNet)', sizeLabel: '~4.7 GB', rateLabel: '~43 t/s' },
  // Validated against mlx_lm 2026-08-17: logits cosine 0.999996, top-5 5/5,
  // all 24 greedy tokens exact ("The capital of France is **Paris**…"), 0 GPU
  // errors, 85 shaders compile under its dims. rateLabel stays '' until a
  // protocol round measures it.
  [QWEN3_8_27B_4BIT.id]: { name: 'Qwen3.8-27B', params: '27B hybrid (DeltaNet)', sizeLabel: '~14.1 GB', rateLabel: '', ramNote: 'needs ~18 GB free RAM' },
  // ADD-MODEL:BRANDINGS
}

/** Page-facing copy for a model; unknown ids fall back to Phi-3 (the default
 *  spec), preserving the old switch's default arm. */
export function modelBranding(spec: ModelSpec): ModelBrand {
  return BRANDINGS[spec.id] ?? BRANDINGS[PHI3.id]
}
