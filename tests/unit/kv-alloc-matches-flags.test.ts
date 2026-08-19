// The KV cache a surface allocates must be the one its flags asked for.
//
// int8KV was expressed TWICE — as VariantFlags.int8KV (parsed from ?kv8,
// default ON) and as DecodeEngineOptions.int8KV — and the engine read only the
// option. Pairing them was a hand-written step at every construction site, and
// four of five got it wrong: the agent host, both room paths and the validate
// boot allocated an f16 cache while their own flags reported int8.
//
// Silent, and consequential three ways. ?kv8=0 was a NO-OP on those surfaces,
// so a bisection could "clear" int8 KV on a surface that never ran it. They ran
// different attention kernels from the chat page for the same model, because
// `splitK = int8Mode ? 0 : R.splitK`. And they allocated twice the KV per token
// that the entrance publishes.
//
// allocKVFor is the single place that pairing happens now. These tests assert
// the pairing, not the call sites — a call-site test would have to enumerate
// them, and the defect was a call site nobody enumerated.

import { describe, expect, it } from 'vitest'
import { SHIPPED_MODELS, specForParam } from '../../src/zero-tvm/model-registry.ts'
import { SCALAR_VARIANTS, parseVariantFlags } from '../../src/zero-tvm/variants.ts'
import { resolveInt8Mode } from '../../src/zero-tvm/engine-core.ts'

// IMPORTED, not re-implemented. The first version of this file mirrored the
// branch locally and the mutation gate proved it worthless: reverting the engine
// to the buggy version left every assertion below green. Both the engine and
// allocKVFor call this same function, so allocation and execution cannot
// disagree about which cache is in use.
const wantsScales = (spec: { mla?: unknown }, flags: { int8KV?: boolean }) =>
  resolveInt8Mode({}, flags, spec)

describe('the KV cache matches the flags that asked for it', () => {
  it('?kv8 defaults ON, so a surface that ignores it runs the wrong cache', () => {
    // The root of the defect: the default is int8, so FORGETTING to thread the
    // flag does not fall back to the same behaviour — it inverts it.
    const flags = parseVariantFlags('', { hasSubgroupsFeature: true, sgSizeOk: true })
    expect(flags.int8KV).toBe(true)
    expect(wantsScales(specForParam('qwen38'), flags)).toBe(true)
  })

  it('?kv8=0 actually reaches the allocator', () => {
    const off = parseVariantFlags('?kv8=0', { hasSubgroupsFeature: true, sgSizeOk: true })
    expect(off.int8KV).toBe(false)
    expect(wantsScales(specForParam('qwen38'), off)).toBe(false)
    // If this ever fails, the bisection arm documented in docs/VERIFICATION.md
    // is inert and every "int8 is cleared" conclusion drawn with it is void.
  })

  it('MLA opts out however the flag is set — it caches a latent', () => {
    const mla = SHIPPED_MODELS.map((m) => m.spec).find((s) => s.mla)
    if (!mla) return
    const on = parseVariantFlags('', { hasSubgroupsFeature: true, sgSizeOk: true })
    expect(wantsScales(mla, on)).toBe(false)
  })

  it('every shipped non-MLA model gets scales under the default flags', () => {
    const flags = parseVariantFlags('', { hasSubgroupsFeature: true, sgSizeOk: true })
    for (const { param, spec } of SHIPPED_MODELS) {
      if (spec.embeddingOnly) continue
      expect(wantsScales(spec, flags), `${param} would get an f16 cache under default flags`)
        .toBe(!spec.mla)
    }
  })

  it('the scalar preset is int8-off, and says so consistently', () => {
    // SCALAR_VARIANTS is the reference path. It must not silently disagree with
    // itself the way the hosts did.
    expect(SCALAR_VARIANTS.int8KV).toBe(false)
    expect(wantsScales(specForParam('qwen38'), SCALAR_VARIANTS)).toBe(false)
  })
})
