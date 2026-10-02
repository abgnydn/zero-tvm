// CHUNK-GRID LIMITS — every chunked-prefill elementwise grid must fit in one
// dispatch dimension at the spec's effective max chunk size, or be folded.
//
// On 2026-09-28 qwen38 at cap 1024 recorded `Dispatch workgroup count X
// (69632) exceeds max compute workgroups per dimension (65535)` on cSiluMul
// (n * FFN_WGS = 1024 * 17408/256) and produced fluent wrong tokens while
// every kernel's numerics stayed exact: an invalid command buffer poisons
// the passes after it, and the suite covering silu_mul ran SEQ=3. The
// per-token path and cap 256 never crossed the limit, which is why the
// defect read as a depth-dependent numerics bug for a month.
//
// cSiluMul now folds across z (foldGridX, same as the LM head), so qwen38 is
// exempt from the fit below — by NAME, not by silence: any spec added to the
// over-limit set without a fold must fail here before it can corrupt.

import { describe, expect, it } from 'vitest'
import { SHIPPED_MODELS } from '../../src/zero-tvm/model-registry.ts'
import { foldGridX } from '../../src/zero-tvm/engine-core.ts'

const LIMIT = 65535
// The shipped default cap. maxChunkCap quarantines only clamp DOWN; an
// explicit override warns through and still records the dispatch, so the
// override-reachable size is what this pins, not the quarantined one.
const N = 1024

/** Specs whose unfolded chunk grid exceeds the limit AND whose dispatch
 *  folds across z. Membership here is the claim; the proof is CAP=1024
 *  token identity on both arms at PROMPT=2000 (twice) and 4000 with 0 GPU
 *  errors, plus gdn_chunk_chain_scale bit-exact at 1024 on these dims. */
const FOLDED = new Set(['qwen3-8-27b-4bit'])

const gridOf = (spec: (typeof SHIPPED_MODELS)[number]['spec']): number =>
  spec.moe
    ? Math.ceil((N * spec.moeSlots * spec.ffn) / 256)
    : N * (spec.ffn / 256)

describe('foldGridX', () => {
  it('is the identity below the limit', () => {
    expect(foldGridX(68)).toEqual({ x: 68, z: 1 })
    expect(foldGridX(16384)).toEqual({ x: 16384, z: 1 })
  })

  it('caps x and carries the remainder in z, with full coverage', () => {
    // qwen38's exact failing grid: 1024 chunks x 17408/256 rows.
    expect(foldGridX(69632)).toEqual({ x: 16384, z: 5 })
    for (const total of [65535, 65536, 69632, 1000000]) {
      const f = foldGridX(total)
      expect(f.x).toBeLessThanOrEqual(16384)
      expect(f.x * f.z).toBeGreaterThanOrEqual(total)
      // Minimal z: one fewer would not cover the grid.
      expect((f.z - 1) * f.x).toBeLessThan(total)
    }
  })
})

describe('chunk grids fit maxComputeWorkgroupsPerDimension or fold', () => {
  it.each(SHIPPED_MODELS.map((m) => [m.param, m.spec] as const))(
    '%s chunk silu grid fits at n=1024, or folds by name',
    (_param, spec) => {
      // MLA cannot chunk at all — the only spec shape without a chunk path.
      if (spec.mla) return
      const grid = gridOf(spec)
      if (grid <= LIMIT) return
      expect(
        FOLDED.has(spec.id),
        `${spec.id}: chunk silu grid ${grid} exceeds ${LIMIT} at n=${N} with no fold — `
        + 'fold the dispatch across z (cSiluMul precedent) or quarantine the cap',
      ).toBe(true)
    },
  )

  it('the folded set is exactly the specs that need it', () => {
    // Guards the guard: an exemption nobody re-checks becomes a hole new
    // specs fall through. If this fails because a spec left the set, delete
    // the name; if it fails because one joined, fold first.
    const over = SHIPPED_MODELS.filter((m) => !m.spec.mla && gridOf(m.spec) > LIMIT)
      .map((m) => m.spec.id)
      .sort()
    expect(over).toEqual([...FOLDED].sort())
  })
})
