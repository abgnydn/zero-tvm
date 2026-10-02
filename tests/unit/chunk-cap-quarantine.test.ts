// The prefill chunk cap, and the per-spec quarantine on it.
//
// Chunked prefill has never been bit-equal to the per-token path. What it is
// held to is empirical TOKEN identity, measured at specific caps. On qwen38
// that failed until 2026-09-28 — at ~16k of context the model answered
// correctly per-token and at cap 256 and invented tool names at the shipped
// cap of 1024. The mechanism is found now (an unfolded cSiluMul grid past
// the 65535 dispatch limit — see chunk-grid-limits.test.ts), the dispatch
// folds, and the quarantine below is lifted. The mechanism tests stay: the
// quarantine FEATURE is how the next such bisection bounds the damage.
//
// This asserts the REAL resolveChunkCap, never a restatement of it. The first
// test written for resolveInt8Mode copied its branch instead, stayed green when
// the engine was reverted, and the mutation gate is what caught it — twice in
// one day, on two different functions. Importing the function is the fix.

import { describe, expect, it } from 'vitest'
import { resolveChunkCap } from '../../src/zero-tvm/engine-core.ts'
import { SHIPPED_MODELS } from '../../src/zero-tvm/model-registry.ts'
import { QWEN3_8_27B_4BIT } from '../../src/compiler/model-spec.ts'

describe('resolveChunkCap', () => {
  it('a spec with no ceiling gets the device default', () => {
    expect(resolveChunkCap({}, {}, true)).toBe(1024)
    expect(resolveChunkCap({}, {}, false)).toBe(64)
  })

  it('a ceiling clamps the default, on both device paths', () => {
    expect(resolveChunkCap({}, { maxChunkCap: 256 }, true)).toBe(256)
    // Below the ceiling the ceiling must not RAISE anything — a device without
    // the matrix unit runs 64 and a 256 quarantine is not permission to chunk
    // wider than that path was ever measured at.
    expect(resolveChunkCap({}, { maxChunkCap: 256 }, false)).toBe(64)
  })

  it('an explicit cap is honoured above the ceiling — the sweep must be able to cross it', () => {
    // Silently clamping a diagnostic is how an A/B measures the same code twice:
    // ask for 1024, get 256, read two identical numbers, conclude the cap does
    // not matter. The engine warns instead. If this ever starts clamping, the
    // CAP sweep that localised this defect stops working.
    expect(resolveChunkCap({ chunkCap: 1024 }, { maxChunkCap: 256 }, true)).toBe(1024)
    expect(resolveChunkCap({ chunkCap: 4096 }, { maxChunkCap: 256 }, false)).toBe(4096)
  })
})

describe('the quarantine list is currently empty', () => {
  it('qwen38 is unquarantined: the fold fixed the mechanism it was bounding', () => {
    // Lifted 2026-09-28: CAP=1024 holds token identity on both arms at 2k
    // (twice) and 4k with 0 GPU errors, and gdn_chunk_chain_scale is
    // bit-exact at 1024 on qwen38 dims. If a cap regression ever reappears,
    // THIS is where it gets bounded again — not as a tuning default.
    expect(QWEN3_8_27B_4BIT.maxChunkCap).toBeUndefined()
    expect(resolveChunkCap({}, QWEN3_8_27B_4BIT, true)).toBe(1024)
  })

  it('no shipped spec is quarantined without being listed here', () => {
    // A quarantine is a known-broken marker. If one appears on another spec,
    // this test should be what makes someone say so out loud rather than it
    // spreading quietly as if it were a tuning default.
    const QUARANTINED = new Set<string>([])
    const found = SHIPPED_MODELS
      .filter((m) => m.spec.maxChunkCap != null)
      .map((m) => m.param)
    expect(new Set(found)).toEqual(QUARANTINED)
  })
})
