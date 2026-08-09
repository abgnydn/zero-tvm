// A STACK THAT MIXES DENSE AND MoE LAYERS — spec derivation and weight plans.
//
// DeepSeek-V2 makes layer 0 an ordinary FFN at intermediate_size 10944 while
// every other layer is MoE at moe_intermediate_size 1408. Two widths and two
// block kinds in one stack, which `spec.ffn` alone cannot say.
//
// The failure this guards is silent: run the expert block over a dense layer's
// weights and you get fluent nonsense, no error. Qwen ships the same idea under
// `mlp_only_layers`, so this is not one vendor's quirk.

import { describe, expect, it } from 'vitest'
import { makeModelSpec, mlxParamNaming } from '../../src/compiler/model-spec.ts'
import { planLayer } from '../../src/zero-tvm/mlx-weights.ts'

const base = {
  id: 'mixed-stack', d: 2048, layers: 4, heads: 16, kvHeads: 16, headDim: 128,
  vocab: 128, pageSize: 16, maxPages: 4, maxSeq: 4096, ropeTheta: 10000,
  rmsEps: 1e-6, tiedEmbeddings: false, qkNorm: false, stops: [0],
  chatTemplateId: 'deepseek' as const, tokenizerKind: 'byteLevel' as const,
  hfRepo: 'none', weightFormat: 'mlx-safetensors' as const,
  paramNaming: mlxParamNaming(''),
}
const MOE = { experts: 64, topK: 6, normTopkProb: false, sharedExpert: true, routerBits: 16 as const }

const mixed = makeModelSpec({
  ...base, ffn: 1408, moe: { ...MOE, denseLayers: [0], denseFfn: 10944 },
})
const uniform = makeModelSpec({ ...base, ffn: 1408, moe: MOE })
const names = (spec: typeof mixed, L: number) => planLayer(spec, L, `model.layers.${L}`).map((p) => p.name)

describe('mixed dense/MoE stack', () => {
  it('marks only the named layers dense', () => {
    expect(mixed.ffnKinds).toEqual(['dense', 'moe', 'moe', 'moe'])
    expect(uniform.ffnKinds).toEqual(['moe', 'moe', 'moe', 'moe'])
  })

  it('gives each layer its own FFN width', () => {
    // The whole reason a per-layer kind is not enough on its own: a dense
    // layer of a MoE model is nearly 8x wider here.
    expect(mixed.ffnWidthAt(0)).toBe(10944)
    expect(mixed.ffnWidthAt(1)).toBe(1408)
    expect(uniform.ffnWidthAt(0)).toBe(1408)
  })

  it('plans ordinary mlp records for a dense layer and expert records for the rest', () => {
    expect(names(mixed, 0)).toContain('ffn_w')
    expect(names(mixed, 0)).not.toContain('router_w')
    expect(names(mixed, 1)).toContain('router_w')
    expect(names(mixed, 1)).not.toContain('ffn_w')
  })

  it('plans a bare weight for an unquantized router, not a trio', () => {
    // routerBits 16 means the quantizer skipped mlp.gate; asking for
    // `mlp.gate.scales` would look for a record the checkpoint does not have.
    const moeLayer = names(mixed, 1)
    expect(moeLayer).toContain('router_w')
    expect(moeLayer).not.toContain('router_s')
    const quantRouter = makeModelSpec({ ...base, ffn: 1408, moe: { ...MOE, routerBits: 8 } })
    expect(names(quantRouter, 0)).toContain('router_s')
  })

  it('refuses dense layers without a width rather than defaulting one', () => {
    // Silently reusing moe_intermediate_size would build the dense layer's
    // buffers 8x too small and read past every row.
    expect(() => makeModelSpec({
      ...base, ffn: 1408, moe: { ...MOE, denseLayers: [0] },
    })).toThrow(/denseFfn/)
  })
})
