// MLA SPEC DERIVATIONS — the arithmetic that decides whether a DeepSeek spec
// describes the checkpoint or merely looks like it does.
//
// Every number below is checkable against the real layer-0 bundle
// (.weights-local/kernel-refs/dsv2layer0/meta.json), which is why they are
// here rather than in a comment. Two of them have already been wrong once:
//
//   - headDim. Setting it to the QUERY width (nope + rope = 192) reads
//     plausibly and makes the engine derive o_proj's packed row count from
//     qDim = 3072, against a row that is 256 words. Finite numbers, wrong
//     model. makeModelSpec throws now; this pins the value it throws for.
//   - the softmax scale. yarn multiplies attention LOGITS by mscale^2, which
//     is not part of RoPE, and the scale's denominator is the QUERY head width
//     (nope + rope), not headDim. Getting either wrong is a uniform factor on
//     every logit — far too small to look broken, far too large to be noise.

import { describe, expect, it } from 'vitest'
import { makeModelSpec, mlxParamNaming, ropeAttnScale } from '../../src/compiler/model-spec.ts'

/** DeepSeek-V2-Lite's real dims, from its config.json and the bundle. */
const DSV2 = {
  id: 'deepseek-v2-lite-test', d: 2048, layers: 27, heads: 16, kvHeads: 16,
  headDim: 128, ffn: 1408, vocab: 102400, pageSize: 16, maxPages: 2112,
  maxSeq: 163840, ropeTheta: 10000, rmsEps: 1e-6, tiedEmbeddings: false,
  qkNorm: false, stops: [100001], chatTemplateId: 'deepseek' as const,
  tokenizerKind: 'byteLevel' as const, hfRepo: 'mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx',
  weightFormat: 'mlx-safetensors' as const, paramNaming: mlxParamNaming(''),
  mla: { kvLoraRank: 512, qLoraRank: null, qkNopeHeadDim: 128, qkRopeHeadDim: 64, vHeadDim: 128 },
  ropeScaling: {
    ropeType: 'yarn' as const, factor: 40, betaFast: 32, betaSlow: 1,
    originalMaxPositionEmbeddings: 4096, mscale: 0.707, mscaleAllDim: 0.707,
  },
}
const spec = makeModelSpec(DSV2)

describe('MLA spec derivations', () => {
  it('derives qDim from headDim, which o_proj contracts over', () => {
    // o_proj.weight is [2048, 256] in the bundle: 256 u32 words of 4-bit values
    // = 2048 inputs. The engine computes its packed row count as qDim/8, so
    // qDim MUST be 2048.
    expect(spec.qDim).toBe(2048)
    expect(spec.qDim / 8).toBe(256)
  })

  it('rotates only the decoupled pe slice', () => {
    // Not headDim, and not headDim * partialRotaryFactor — DeepSeek's config
    // has no partial_rotary_factor at all, so the default would give 128.
    expect(spec.rotaryDim).toBe(64)
    expect(spec.halfRotary).toBe(32)
  })

  it('caches a latent and one shared key, not K and V per head', () => {
    expect(spec.mlaCachePerToken).toBe(576)
    // The claim the whole feature rests on, stated as a ratio rather than a
    // sentence: what an equivalent per-head cache would cost.
    const mha = 2 * spec.kvHeads * spec.headDim * 2 * spec.layers
    expect(spec.kvBytesPerToken).toBe(576 * 2 * spec.layers)
    expect(mha / spec.kvBytesPerToken).toBeCloseTo(7.11, 2)
  })

  it('derives the projection row counts the loader plans against', () => {
    expect(spec.mlaQProjRows).toBe(3072)   // heads * (128 + 64)
    expect(spec.mlaKvaRows).toBe(576)      // 512 + 64
    expect(spec.mlaKvbRows).toBe(4096)     // heads * (128 + 128)
  })

  it('keeps the shared-key region 256-byte alignable', () => {
    expect(spec.mlaLatentBytes % 256).toBe(0)
    expect(spec.mlaCacheBytes).toBe(spec.maxContext * 576 * 2)
    expect(spec.mlaLatentBytes).toBe(spec.maxContext * 512 * 2)
  })

  it('reproduces the reference bundle\'s softmax scale exactly', () => {
    // THE assertion. .weights-local/kernel-refs/dsv2layer0/meta.json carries
    // softmax_scale 0.1147213867929261, produced independently in numpy as
    // (nope + rope)^-0.5 * mscale^2. Denominator sqrt(headDim) instead of
    // sqrt(nope+rope) gives 0.1405; dropping mscale^2 gives 0.0722. Both run.
    const scale = ropeAttnScale(spec) / Math.sqrt(DSV2.mla.qkNopeHeadDim + DSV2.mla.qkRopeHeadDim)
    expect(scale).toBeCloseTo(0.1147213867929261, 9)
    expect(ropeAttnScale(spec)).toBeCloseTo(1.5896, 4)
  })

  it('refuses a spec whose headDim is the query width', () => {
    expect(() => makeModelSpec({ ...DSV2, headDim: 192 })).toThrow(/vHeadDim/)
  })

  it('refuses a q_a_proj/q_b_proj checkpoint rather than mis-planning it', () => {
    // V2-full and V3 split the query projection. The loader plans a single
    // q_proj; silently accepting these would look for records that do not exist.
    expect(() => makeModelSpec({ ...DSV2, mla: { ...DSV2.mla, qLoraRank: 1536 } }))
      .toThrow(/qLoraRank/)
  })

  it('leaves every non-MLA spec derivation at zero', () => {
    const dense = makeModelSpec({ ...DSV2, mla: undefined, ropeScaling: undefined })
    expect(dense.mlaCachePerToken).toBe(0)
    expect(dense.mlaCacheBytes).toBe(0)
    expect(dense.rotaryDim).toBe(128)                       // back to headDim
    expect(dense.kvBytesPerToken).toBe(2 * 16 * 128 * 2 * 27)
  })
})
