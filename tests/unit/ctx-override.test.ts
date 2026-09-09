// `?ctx=` — the KV-budget override, and the rules that keep it from lying.
//
// The compiled maxPages is a budget constant (~1 GiB, one commit in July),
// not a limit. This knob rebuilds the spec through makeModelSpec so every
// derived field moves together; these tests pin the clamps, because both
// failure directions are silent: past maxSeq the model degrades without
// erroring, and a stale derived field (kvBytesPerToken × maxContext vs the
// actual buffer sizes) is fluent-wrong-output territory.
//
// Expectations are hand-derived from the specs' own published numbers, not
// from running specFromSearch on anything but the input under test.

import { describe, expect, it } from 'vitest'
// THE SHIPPED FUNCTION, not a copy of it. This file used to re-declare
// specFromSearch locally — "through the registry, not model-select, because
// the latter imports weight-loader, which reads GPUBufferUsage at module
// scope and cannot load under Node". That stopped being true when
// weight-loader made the read LAZY (see the `usage()` comment there), and the
// copy outlived the reason for it: model-select grew a ctxFor round-trip that
// shrank the default spec, and every assertion below still passed, because
// they were being run against the local copy of the OLD formula. Right
// assertions, wrong subject — docs/VERIFICATION.md's second family. Import it.
import { specFromSearch } from '../../src/zero-tvm/model-select.ts'
import { SHIPPED_MODELS } from '../../src/zero-tvm/model-registry.ts'

/** `?model=` for a registry entry — '' is a real param (Phi-3), so a link to
 *  the default carries no query at all. */
const search = (param: string): string => (param ? `?model=${param}` : '')

describe('?ctx= override', () => {
  it('is absent → the compiled spec, byte-for-byte the same object', () => {
    const a = specFromSearch('?model=qwen35')
    const b = specFromSearch('?model=qwen35&ctx=0')
    const c = specFromSearch('?model=qwen35&ctx=junk')
    expect(a.maxContext).toBe(32768)
    expect(b).toBe(a)          // no rebuild on a nonsense value
    expect(c).toBe(a)
  })

  it('is absent → EVERY shipped spec is the object the registry holds', () => {
    // The sweep, not one sample: the only spec this can bite is one whose
    // maxContext exceeds its maxSeq, and there is exactly one of those today.
    // A sample that happened to miss it is how the shrink below shipped.
    for (const { param, spec } of SHIPPED_MODELS) {
      expect(specFromSearch(search(param)), param || '(default)').toBe(spec)
    }
  })

  it('the DEFAULT model keeps its 257th KV page — pages round UP, past maxSeq', () => {
    // Phi-3 is the one spec where maxContext (4112) is ABOVE maxSeq (4096):
    // 4096 tokens need ceil(4096/16) = 256 pages... but the compiled budget is
    // 257, and 257 × 16 = 4112. scripts/station.mjs states the rule this pins
    // — "KV pages round UP, so Phi-3's table holds 4112 tokens against a
    // 4096-token trained window, and a naive clamp to maxSeq would SHRINK the
    // shipped default". Routing the no-flag default through the maxSeq clamp
    // is that naive clamp, and it cost the default model on zero-tvm.html and
    // validate.html one page without a word.
    const s = specFromSearch('')
    expect(s.id).toBe('phi3-mini')
    expect(s.maxPages).toBe(257)
    expect(s.maxContext).toBe(4112)
  })

  it('an EXPLICIT ctx is clamped even when it names the spec default', () => {
    // The two paths are genuinely different, and this is the pair that proves
    // it: not asking is not the same as asking for the number you already
    // have. An explicit budget is a request to re-size, and re-sizing clamps
    // to the trained window.
    expect(specFromSearch('?ctx=4112').maxPages).toBe(256)
    expect(specFromSearch('').maxPages).toBe(257)
  })

  it('raises qwen35 to its native window: 262,144 tokens, 8 GiB of KV', () => {
    // maxSeq 262144 / pageSize 16 = 16384 pages. kvBytesPerToken is 32 KiB
    // (KV on 8 of 32 layers), so full KV is 8 GiB — 1 GiB per attention
    // layer, inside the measured 4 GiB per-buffer limit on this machine.
    const s = specFromSearch('?model=qwen35&ctx=262144')
    expect(s.maxPages).toBe(16384)
    expect(s.maxContext).toBe(262144)
    expect(s.kvBytesPerToken * s.maxContext).toBe(8 * 2 ** 30)
    expect((s.kvBytesPerToken * s.maxContext) / 8).toBe(2 ** 30)  // per attn layer
  })

  it('CLAMPS to maxSeq — past the trained window is silent degradation', () => {
    const s = specFromSearch('?model=qwen35&ctx=999999999')
    expect(s.maxContext).toBe(262144)
  })

  it('rounds a non-multiple of pageSize UP, never down', () => {
    // 1000 tokens needs 63 pages of 16 (1008 slots); rounding down to 62
    // would refuse the last 8 tokens of a prompt the caller sized to fit.
    const s = specFromSearch('?model=qwen35&ctx=1000')
    expect(s.maxPages).toBe(63)
    expect(s.maxContext).toBe(1008)
  })

  it('keeps the derived fields coherent after a rebuild', () => {
    const s = specFromSearch('?model=llama32&ctx=131072')
    // llama32: maxSeq 131072, 32 KiB/token over 16 layers → 4 GiB total KV.
    expect(s.maxContext).toBe(131072)
    expect(s.kvBytesPerToken).toBe(32 * 1024)
    expect(s.id).toBe(specFromSearch('?model=llama32').id)  // same identity → same branding
  })

  it('Phi-3 cannot be pushed past its 4k window at all', () => {
    // Its maxSeq IS its default window (the one spec where the budget rule
    // was never the binder), so ctx can only shrink it.
    const s = specFromSearch('?ctx=1000000')
    expect(s.maxContext).toBe(4096)
  })
})
