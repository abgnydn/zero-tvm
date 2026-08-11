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
// Through the registry, not model-select — the latter imports weight-loader,
// which reads GPUBufferUsage at module scope and cannot load under Node.
import { specForParam, specWithCtx } from '../../src/zero-tvm/model-registry.ts'

const specFromSearch = (search: string) => {
  const q = new URLSearchParams(search)
  return specWithCtx(specForParam(q.get('model')), Number(q.get('ctx')))
}

describe('?ctx= override', () => {
  it('is absent → the compiled spec, byte-for-byte the same object', () => {
    const a = specFromSearch('?model=qwen35')
    const b = specFromSearch('?model=qwen35&ctx=0')
    const c = specFromSearch('?model=qwen35&ctx=junk')
    expect(a.maxContext).toBe(32768)
    expect(b).toBe(a)          // no rebuild on a nonsense value
    expect(c).toBe(a)
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
