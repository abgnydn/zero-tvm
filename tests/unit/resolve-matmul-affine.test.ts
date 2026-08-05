/**
 * resolveMatmul must pick the AFFINE family for an MLX checkpoint.
 *
 * This is the one resolution mistake that costs nothing at runtime: a symmetric
 * kernel reads MLX buffers happily, it just groups the row by 32 instead of 64
 * and never applies the bias. Two of Qwen3.6's three K instances (d=2048,
 * qDim=4096) even satisfy the vec4 K%1024 gate, so the pre-affine resolution
 * landed on int4_matmul_tiled_vec4 — which is exactly wrong and exactly silent.
 *
 * No GPU: Pipelines is stubbed with string sentinels so the assertion is about
 * WHICH pipeline is chosen, not what it computes.
 */
import { describe, it, expect } from 'vitest'
import { resolveVariantPipelines, SCALAR_VARIANTS } from '../../src/zero-tvm/variants.ts'
import { PHI3, QWEN35_4B, QWEN36_35B_A3B } from '../../src/compiler/model-spec.ts'

// Every Pipelines field as its own name, so a resolution is readable as a string.
const NAMES = [
  'int4Matmul', 'int4MatmulSg', 'int4MatmulTiled', 'int4MatmulTiled8',
  'lmHead', 'int4MatmulF32Sg', 'int4MatmulF32Tiled', 'int4MatmulF32Tiled8',
  'int4MatmulSgVec4', 'int4MatmulTiledVec4', 'int4MatmulF32SgVec4', 'int4MatmulF32TiledVec4',
  'int4MatmulSgVec4h', 'int4MatmulTiledVec4h', 'int4MatmulF32SgVec4h', 'int4MatmulF32TiledVec4h',
  'int4MatmulAffine', 'int4MatmulF32Affine', 'int4MatmulSgAffine', 'int4MatmulTiledAffine',
  'int4MatmulF32SgAffine', 'int4MatmulF32TiledAffine',
]
// The shipped presets are scalar; the chat page turns subgroups on per URL flag.
// These resolutions are what a subgroup-capable device gets.
const TILED = { ...SCALAR_VARIANTS, subgroups: true, matmul: 'tiled' as const, vec4: true, vec4Half: true }

const stub = () => {
  const P: Record<string, unknown> = {}
  for (const n of NAMES) P[n] = n
  // Everything else resolveVariantPipelines touches; identity is irrelevant.
  for (const n of ['attention', 'attentionSg', 'attentionSplitK', 'attentionSplitKSg',
    'attentionCombine', 'attentionPrefill', 'fusedFfn', 'fusedFfnTiledSg', 'fusedFfnPrologue',
    'fusedFfnTiledSgPrologue', 'qkvFused', 'qkvFusedSg', 'qkvFusedSgVec4', 'qkvFusedTiledSg',
    'qkvFusedTiled2Sg', 'qkvFusedScratch', 'argmax', 'argmaxSg']) P[n] = n
  return P as never
}

describe('resolveMatmul affine gating', () => {
  it('Qwen3.6 resolves every matmul instance to the affine family', () => {
    const R = resolveVariantPipelines(TILED, stub(), QWEN36_35B_A3B)
    expect(R.matmul).toBe('int4MatmulTiledAffine')
    expect(R.matmulF32).toBe('int4MatmulF32TiledAffine')
    expect(R.matmulOProj).toBe('int4MatmulTiledAffine')
    expect(R.matmulFfnDown).toBe('int4MatmulTiledAffine')
    expect(R.matmulLabel).toContain('affine')
  })

  it('Qwen3.6 never resolves to a vec4 pipeline (the generator forbids affine+vec4)', () => {
    const R = resolveVariantPipelines(TILED, stub(), QWEN36_35B_A3B)
    for (const p of [R.matmul, R.matmulF32, R.matmulOProj, R.matmulFfnDown]) {
      expect(String(p)).not.toMatch(/Vec4/)
    }
  })

  it('the three symmetric specs resolve exactly as before', () => {
    const q35 = resolveVariantPipelines(TILED, stub(), QWEN35_4B)
    expect(q35.matmul).toBe('int4MatmulTiledVec4h')   // d=2560: %1024 fails, %512 passes
    expect(q35.matmulOProj).toBe('int4MatmulTiledVec4') // qDim=4096
    expect(q35.matmulFfnDown).toBe('int4MatmulTiledVec4') // ffn=9216 = 9*1024
    const phi = resolveVariantPipelines(TILED, stub(), PHI3)
    expect(phi.matmul).toBe('int4MatmulTiledVec4')
    expect(phi.matmulLabel).not.toContain('affine')
  })
})
