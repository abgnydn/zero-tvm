/**
 * SHADER VARIANT SELECTION
 *
 * One place for the URL-flag A/B harness and the variant→pipeline resolution
 * that chat.ts historically carried inline. Both engine consumers use it:
 * the chat page parses location.search via parseVariantFlags(); validate (and
 * any other page that wants the plain reference path) just takes
 * SCALAR_VARIANTS, which resolves every role to the scalar pipeline.
 *
 * URL toggles (chat page):
 *   ?sg=0       disable all subgroup shaders
 *   ?sgattn=0   disable subgroup attention only (keep _sg argmax)
 *   ?sgargmax=0 disable subgroup argmax only (keep _sg attention)
 *   ?sgqkv=0    disable subgroup fused-QKV only
 *   ?qkvtile=1  opt into tiled qkv (4 pairs/WG + input cache)
 *   ?qkvtile2=1 opt into tiled2sg qkv (2 pairs/WG, 64 threads, input cache)
 *   ?sgffn=0    disable tiled+subgroup FFN only
 *   ?matmul=    tiled8 | tiled | sg | scalar (default: tiled when sg on)
 *   ?kv8=1      opt into the int8 KV cache path
 *   ?vec4=0     opt OUT of vec4<u32> weight + activation loads in the
 *               generated int4 matmuls (default ON for matmul=tiled|sg with
 *               sg32; measured +4.5% alone, +7.1% with ?vec4qkv — BENCH.md
 *               2026-07-25 M2 Max A/B)
 *   ?vec4qkv=0  opt OUT of the vec4-load qkv_fused sibling (32-thread;
 *               default ON with sg32; measured +4.2% alone)
 *
 * Measured experiments (2026-07-25/27 on M2 Max — see BENCH.md):
 *   ?splitk=N      split-K flash-decode attention, N partitions per head +
 *                  a combine pass (f16 KV only; ignored w/ ?kv8=1).
 *                  +3.1% at 128-token / +4.0% at 1024-token decode —
 *                  default N=8 with sg32; ?splitk=0 disables.
 *   ?fuseprologue=1  fold the FFN-entry add_norm into the FFN kernel's
 *                  shared-memory prologue; layer tail becomes add3_norm.
 *                  Measured −13.7% on M2 Max (falsified there; kept for
 *                  A/B on other GPUs).
 */

import type { Pipelines } from '../compiler/compiler.js'
// Value imports come from model-spec (dependency-free) so this module stays
// importable under plain Node type stripping (tests) — compiler.ts pulls in
// Vite `?raw` shader imports. The explicit .ts extension is load-bearing
// (see shader-prelude.ts).
import { PHI3, type ModelSpec } from '../compiler/model-spec.ts'

export type MatmulVariant = 'tiled8' | 'tiled' | 'sg' | 'scalar'

export interface VariantFlags {
  /** Compile the _sg pipeline family (subgroups feature granted + size-32 probe passed + not ?sg=0). */
  subgroups: boolean
  sgAttn: boolean
  sgArgmax: boolean
  sgQkv: boolean
  qkvTile: boolean
  qkvTile2: boolean
  sgFfn: boolean
  matmul: MatmulVariant
  /** int8 KV cache opt-in (?kv8=1). Consumed by the page's KV allocation, not by pipeline resolution. */
  int8KV: boolean
  /** vec4-load int4 matmuls — default ON (measured +4.5% on M2 Max); ?vec4=0 to disable. */
  vec4: boolean
  /** K%512 half-unroll vec4 siblings (_vec4h) for the instances the K%1024
   *  gate excludes (Qwen3 d=2560 / ffn=9728). Default follows `vec4`;
   *  ?vec4h=0 disables just the half variants for A/B. */
  vec4Half: boolean
  /** vec4-load qkv_fused sibling — default ON (measured +4.2% on M2 Max); ?vec4qkv=0 to disable. */
  vec4Qkv: boolean
  /** Fused qk_norm+RoPE+KV-append on the unfused qkNorm decode path (Qwen3):
   *  3 post-matmul dispatches become 1 (10 → 8 per layer). Default ON;
   *  ?fuseqk=0 restores the reference 3-dispatch chain for A/B. */
  fuseQkNorm: boolean
  /** ?splitk=N — split-K attention partitions per head; 0 = off (opt-in: ~+3% at short context). */
  splitK: number
  /** ?fuseprologue=1 — add_norm folded into the FFN prologue (opt-in; −13.7% on M2 Max — falsified there). */
  fusePrologue: boolean
}

/** Everything scalar / off — the reference path validate.ts runs on. */
export const SCALAR_VARIANTS: VariantFlags = {
  subgroups: false,
  sgAttn: false,
  sgArgmax: false,
  sgQkv: false,
  qkvTile: false,
  qkvTile2: false,
  sgFfn: false,
  matmul: 'scalar',
  int8KV: false,
  vec4: false,
  vec4Half: false,
  vec4Qkv: false,
  fuseQkNorm: false,
  splitK: 0,
  fusePrologue: false,
}

/**
 * Parse the URL toggles for bisecting the subgroup path. `hasSubgroupsFeature`
 * is whether the *device* was granted 'subgroups'; `sgSizeOk` is the result of
 * the one-shot subgroup-size probe (our _sg shaders assume subgroup size ==
 * 32 — a full 32-thread workgroup in one subgroup).
 */
export function parseVariantFlags(
  search: string,
  opts: { hasSubgroupsFeature: boolean; sgSizeOk: boolean },
): VariantFlags {
  const q = new URLSearchParams(search)
  const sgAll = q.get('sg') !== '0'
  return {
    subgroups: sgAll && opts.hasSubgroupsFeature && opts.sgSizeOk,
    sgAttn: sgAll && q.get('sgattn') !== '0',
    sgArgmax: sgAll && q.get('sgargmax') !== '0',
    sgQkv: sgAll && q.get('sgqkv') !== '0',
    // Opt-in: tiled qkv (4 pairs/WG + input cache). Default off until A/B verified.
    qkvTile: sgAll && q.get('qkvtile') === '1',
    // Opt-in: tiled2sg qkv (2 pairs/WG, 64 threads, input cache). Default off until A/B verified.
    qkvTile2: sgAll && q.get('qkvtile2') === '1',
    sgFfn: sgAll && q.get('sgffn') !== '0',
    // Matmul variants:
    //   tiled  — 4 output rows per workgroup, shares input reads across rows (default on Apple sg32)
    //   tiled8 — 8 rows per WG; measured ~5% slower than tiled4 on M-series (register pressure /
    //            reduced occupancy wins out over the halved input-vector DRAM traffic). Kept compiled
    //            so `?matmul=tiled8` can re-test on other GPUs.
    //   sg     — naive 32-thread + subgroupAdd (experimental, slower than scalar on Apple)
    //   scalar — original 64-thread tree reduction
    matmul: (q.get('matmul') ?? (sgAll ? 'tiled' : 'scalar')) as MatmulVariant,
    int8KV: q.get('kv8') === '1',
    // vec4 loads: measured +7% decode on M2 Max (BENCH.md 2026-07-25 A/B),
    // default ON where the sg32 builds exist; ?vec4=0 / ?vec4qkv=0 to A/B off.
    vec4: sgAll && q.get('vec4') !== '0',
    // K%512 half-unroll siblings ride the vec4 default; ?vec4h=0 A/Bs just
    // them (the K%1024 full-vec4 instances stay on).
    vec4Half: sgAll && q.get('vec4') !== '0' && q.get('vec4h') !== '0',
    vec4Qkv: sgAll && q.get('vec4qkv') !== '0',
    // Fused qk_norm+RoPE+append on the qkNorm decode path (plain 32-thread
    // kernel — no subgroups needed). ?fuseqk=0 restores the reference chain.
    fuseQkNorm: q.get('fuseqk') !== '0',
    // split-K attention: measured +3.1% at 128-token decode and +4.0% at
    // 1024-token decode on M2 Max (the win grows with KV depth, as the
    // occupancy hypothesis predicts) — default N=8 where the sg32 path
    // exists; ?splitk=0 to disable, ?splitk=N to re-tune. Prologue fusion
    // measured -14% on M2 Max (falsified there; flag kept for other GPUs).
    // BENCH.md "Measured experiments".
    splitK: parseSplitK(q.get('splitk') ?? (sgAll ? '8' : '0')),
    fusePrologue: q.get('fuseprologue') === '1',
  }
}

// ?splitk=N: partitions per head for the split-K attention experiment.
// Sensible values are 2/4/8; anything in 2..16 is accepted (the engine sizes
// its partials scratch from the parsed value). 0/absent/garbage = off.
function parseSplitK(raw: string | null): number {
  const n = Number.parseInt(raw ?? '0', 10)
  return Number.isFinite(n) && n >= 2 && n <= 16 ? n : 0
}

// Map a matmul variant name to the int4 + int4-f32 pipelines and the
// rows-per-workgroup count the caller needs for dispatch. Falls back to
// scalar when a requested _sg/_tiled pipeline isn't compiled (subgroups
// feature disabled or variant name unknown).
//
// `k` is the reduction dim of the matmul INSTANCE this resolution serves:
// the vec4 layout consumes 32 K-elements per thread per iteration, so it
// requires K % 1024 == 0 (Phi-3: every K qualifies; Qwen3: K=2560/9728
// fail, K=4096 passes). Instances that fail the 1024 gate but satisfy
// K % 512 == 0 resolve to the `_vec4h` half-unroll siblings (vec2<u32>
// weight loads — both Qwen3 shapes qualify) when `vec4Half` is on; anything
// else falls back to the non-vec4 pipeline of the same variant, keeping
// rowsPerWG identical.
export function resolveMatmul(
  variant: MatmulVariant,
  P: Pipelines,
  vec4 = false,
  k?: number,
  vec4Half = vec4,
): { pipeline: GPUComputePipeline; pipelineF32: GPUComputePipeline; rowsPerWG: number; label: string } {
  // ?vec4=1 experiment: vec4-load builds exist for the tiled and sg shapes.
  // tiled8 has no vec4 build (known-regressed tile size) and scalar can't
  // have one (64-thread WG breaks K divisibility) — those fall through
  // to their non-vec4 pipelines.
  let useHalf = false
  if (k !== undefined && k % 1024 !== 0) {
    useHalf = vec4Half && k % 512 === 0
    vec4 = false
  }
  if (useHalf && variant === 'tiled' && P.int4MatmulTiledVec4h && P.int4MatmulF32TiledVec4h) {
    return { pipeline: P.int4MatmulTiledVec4h, pipelineF32: P.int4MatmulF32TiledVec4h, rowsPerWG: 4, label: 'tiled_vec4h' }
  }
  if (useHalf && variant === 'sg' && P.int4MatmulSgVec4h && P.int4MatmulF32SgVec4h) {
    return { pipeline: P.int4MatmulSgVec4h, pipelineF32: P.int4MatmulF32SgVec4h, rowsPerWG: 1, label: 'sg_vec4h' }
  }
  if (vec4 && variant === 'tiled' && P.int4MatmulTiledVec4 && P.int4MatmulF32TiledVec4) {
    return { pipeline: P.int4MatmulTiledVec4, pipelineF32: P.int4MatmulF32TiledVec4, rowsPerWG: 4, label: 'tiled_vec4' }
  }
  if (vec4 && variant === 'sg' && P.int4MatmulSgVec4 && P.int4MatmulF32SgVec4) {
    return { pipeline: P.int4MatmulSgVec4, pipelineF32: P.int4MatmulF32SgVec4, rowsPerWG: 1, label: 'sg_vec4' }
  }
  if (variant === 'tiled8' && P.int4MatmulTiled8 && P.int4MatmulF32Tiled8) {
    return { pipeline: P.int4MatmulTiled8, pipelineF32: P.int4MatmulF32Tiled8, rowsPerWG: 8, label: 'tiled8' }
  }
  if (variant === 'tiled' && P.int4MatmulTiled && P.int4MatmulF32Tiled) {
    return { pipeline: P.int4MatmulTiled, pipelineF32: P.int4MatmulF32Tiled, rowsPerWG: 4, label: 'tiled' }
  }
  if (variant === 'sg' && P.int4MatmulSg && P.int4MatmulF32Sg) {
    return { pipeline: P.int4MatmulSg, pipelineF32: P.int4MatmulF32Sg, rowsPerWG: 1, label: 'sg' }
  }
  return { pipeline: P.int4Matmul, pipelineF32: P.lmHead, rowsPerWG: 1, label: 'scalar' }
}

/** The concrete per-role pipeline picks the engine dispatches with. */
export interface ResolvedPipelines {
  /** K = d instances (unfused QKV projection; also the LM head via matmulF32). */
  matmul: GPUComputePipeline
  matmulF32: GPUComputePipeline
  /** K = qDim instance (o_proj) — differs from `matmul` only when the vec4
   *  K-divisibility gate splits the instances (Qwen3: d=2560 fails, qDim=4096
   *  passes). Same rowsPerWG as `matmul`. */
  matmulOProj: GPUComputePipeline
  /** K = ffn instance (down_proj); same rowsPerWG as `matmul`. */
  matmulFfnDown: GPUComputePipeline
  matmulRowsPerWG: number
  matmulLabel: string
  attention: GPUComputePipeline
  attentionLabel: string
  /** Split-K attention pipelines — non-null only when ?splitk=N resolved. */
  attentionSplitK: GPUComputePipeline | null
  attentionCombine: GPUComputePipeline | null
  /** Partitions per head; 0 when split-K is off. */
  splitK: number
  argmax: GPUComputePipeline
  argmaxLabel: string
  qkvFused: GPUComputePipeline
  /** 1 pair/WG for scalar and _sg variants; 2 pairs/WG for tiled variants. */
  qkvPairsPerWG: number
  qkvLabel: string
  ffn: GPUComputePipeline
  ffnRowsPerWG: number
  ffnLabel: string
  /** true when ffn is a prologue-fused kernel (?fuseprologue=1) — the engine
   *  must bind (residual, delta, gamma) inputs and use add3_norm at the tail. */
  ffnPrologue: boolean
}

/**
 * Select subgroup variants when available. Bindings are identical to the
 * scalar versions, so callers can swap the pipeline reference and keep the
 * same bind group. Every pick falls back to scalar when the requested _sg
 * pipeline wasn't compiled (subgroups feature off).
 */
export function resolveVariantPipelines(flags: VariantFlags, P: Pipelines, spec: ModelSpec = PHI3): ResolvedPipelines {
  // Per-instance vec4 gating: each matmul role passes its own K, so vec4
  // survives exactly on the instances whose K % 1024 == 0 (all of them for
  // Phi-3 — resolution unchanged there).
  const { pipeline: matmul, pipelineF32: matmulF32, rowsPerWG: matmulRowsPerWG, label: matmulLabel } =
    resolveMatmul(flags.matmul, P, flags.vec4, spec.d, flags.vec4Half)
  const { pipeline: matmulOProj } = resolveMatmul(flags.matmul, P, flags.vec4, spec.qDim, flags.vec4Half)
  const { pipeline: matmulFfnDown } = resolveMatmul(flags.matmul, P, flags.vec4, spec.ffn, flags.vec4Half)
  const attention = (flags.sgAttn && P.attentionSg) ? P.attentionSg : P.attention
  // ?splitk=N experiment: partial pass follows the sgAttn toggle (subgroup
  // reduce when available), combine is feature-free.
  const splitK = flags.splitK >= 2 ? flags.splitK : 0
  const attentionSplitK = splitK
    ? ((flags.sgAttn && P.attentionSplitKSg) ? P.attentionSplitKSg : P.attentionSplitK)
    : null
  const attentionCombine = splitK ? P.attentionCombine : null
  const attentionLabel = attentionSplitK
    ? `splitk${splitK}${attentionSplitK === P.attentionSplitKSg ? '_sg' : ''}`
    : attention === P.attentionSg ? '_sg' : 'scalar'
  const argmax = (flags.sgArgmax && P.argmaxSg) ? P.argmaxSg : P.argmax
  // The vec4 qkv sibling loads 32 K-elements (K = d) per thread per
  // iteration — same K % 1024 gate as the vec4 matmuls (Qwen3's d=2560
  // fails → falls back to qkv_fused_sg).
  const vec4QkvOk = flags.vec4Qkv && spec.d % 1024 === 0
  const qkvFused = (vec4QkvOk && P.qkvFusedSgVec4) ? P.qkvFusedSgVec4
                 : (flags.qkvTile2 && P.qkvFusedTiled2Sg) ? P.qkvFusedTiled2Sg
                 : (flags.qkvTile && P.qkvFusedTiledSg) ? P.qkvFusedTiledSg
                 : (flags.sgQkv && P.qkvFusedSg) ? P.qkvFusedSg
                 : P.qkvFused
  // 1 pair/WG for scalar and _sg variants; 2 pairs/WG for tiled variants.
  const qkvPairsPerWG = (qkvFused === P.qkvFusedTiledSg || qkvFused === P.qkvFusedTiled2Sg) ? 2 : 1
  const qkvLabel = qkvFused === P.qkvFusedSgVec4 ? '_sg_vec4'
                 : qkvFused === P.qkvFusedTiled2Sg ? 'tiled2_sg'
                 : qkvFused === P.qkvFusedTiledSg ? 'tiled_sg'
                 : qkvFused === P.qkvFusedSg ? '_sg'
                 : 'scalar'
  // ?fuseprologue=1 experiment: swap the FFN kernel for its prologue-fused
  // sibling (same tile shape as the non-prologue pick would have had).
  const ffnPrologue = flags.fusePrologue
  const ffn = ffnPrologue
    ? ((flags.sgFfn && P.fusedFfnTiledSgPrologue) ? P.fusedFfnTiledSgPrologue : P.fusedFfnPrologue)
    : ((flags.sgFfn && P.fusedFfnTiledSg) ? P.fusedFfnTiledSg : P.fusedFfn)
  const ffnRowsPerWG = (ffn === P.fusedFfnTiledSg || ffn === P.fusedFfnTiledSgPrologue) ? 4 : 1
  const ffnLabel = ffn === P.fusedFfnTiledSgPrologue ? 'tiled_sg_prologue'
                 : ffn === P.fusedFfnPrologue ? 'scalar_prologue'
                 : ffn === P.fusedFfnTiledSg ? 'tiled_sg'
                 : 'scalar'
  return {
    matmul, matmulF32, matmulOProj, matmulFfnDown, matmulRowsPerWG, matmulLabel,
    attention, attentionLabel,
    attentionSplitK, attentionCombine, splitK: attentionSplitK ? splitK : 0,
    argmax, argmaxLabel: argmax === P.argmaxSg ? '_sg' : 'scalar',
    qkvFused, qkvPairsPerWG, qkvLabel,
    ffn, ffnRowsPerWG, ffnLabel, ffnPrologue,
  }
}
