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
 */

import type { Pipelines } from '../compiler/compiler.js'

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
  }
}

// Map a matmul variant name to the int4 + int4-f32 pipelines and the
// rows-per-workgroup count the caller needs for dispatch. Falls back to
// scalar when a requested _sg/_tiled pipeline isn't compiled (subgroups
// feature disabled or variant name unknown).
export function resolveMatmul(
  variant: MatmulVariant,
  P: Pipelines,
): { pipeline: GPUComputePipeline; pipelineF32: GPUComputePipeline; rowsPerWG: number; label: string } {
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
  matmul: GPUComputePipeline
  matmulF32: GPUComputePipeline
  matmulRowsPerWG: number
  matmulLabel: string
  attention: GPUComputePipeline
  attentionLabel: string
  argmax: GPUComputePipeline
  argmaxLabel: string
  qkvFused: GPUComputePipeline
  /** 1 pair/WG for scalar and _sg variants; 2 pairs/WG for tiled variants. */
  qkvPairsPerWG: number
  qkvLabel: string
  ffn: GPUComputePipeline
  ffnRowsPerWG: number
  ffnLabel: string
}

/**
 * Select subgroup variants when available. Bindings are identical to the
 * scalar versions, so callers can swap the pipeline reference and keep the
 * same bind group. Every pick falls back to scalar when the requested _sg
 * pipeline wasn't compiled (subgroups feature off).
 */
export function resolveVariantPipelines(flags: VariantFlags, P: Pipelines): ResolvedPipelines {
  const { pipeline: matmul, pipelineF32: matmulF32, rowsPerWG: matmulRowsPerWG, label: matmulLabel } =
    resolveMatmul(flags.matmul, P)
  const attention = (flags.sgAttn && P.attentionSg) ? P.attentionSg : P.attention
  const argmax = (flags.sgArgmax && P.argmaxSg) ? P.argmaxSg : P.argmax
  const qkvFused = (flags.qkvTile2 && P.qkvFusedTiled2Sg) ? P.qkvFusedTiled2Sg
                 : (flags.qkvTile && P.qkvFusedTiledSg) ? P.qkvFusedTiledSg
                 : (flags.sgQkv && P.qkvFusedSg) ? P.qkvFusedSg
                 : P.qkvFused
  // 1 pair/WG for scalar and _sg variants; 2 pairs/WG for tiled variants.
  const qkvPairsPerWG = (qkvFused === P.qkvFusedTiledSg || qkvFused === P.qkvFusedTiled2Sg) ? 2 : 1
  const qkvLabel = qkvFused === P.qkvFusedTiled2Sg ? 'tiled2_sg'
                 : qkvFused === P.qkvFusedTiledSg ? 'tiled_sg'
                 : qkvFused === P.qkvFusedSg ? '_sg'
                 : 'scalar'
  const ffn = (flags.sgFfn && P.fusedFfnTiledSg) ? P.fusedFfnTiledSg : P.fusedFfn
  const ffnRowsPerWG = ffn === P.fusedFfnTiledSg ? 4 : 1
  return {
    matmul, matmulF32, matmulRowsPerWG, matmulLabel,
    attention, attentionLabel: attention === P.attentionSg ? '_sg' : 'scalar',
    argmax, argmaxLabel: argmax === P.argmaxSg ? '_sg' : 'scalar',
    qkvFused, qkvPairsPerWG, qkvLabel,
    ffn, ffnRowsPerWG, ffnLabel: ffn === P.fusedFfnTiledSg ? 'tiled_sg' : 'scalar',
  }
}
