/**
 * UNIT TESTS — chunk-GEMM selection (pickChunkGemm in engine-core.ts).
 *
 * The chunk-prefill ladder (E5 matrix-unit > sgmat > tiled > matvec) decides
 * which kernel every prefill runs, and an explicit ?chunkgemm= that cannot
 * run must throw rather than silently substitute — a silent fallback measures
 * the same code twice and reads two identical numbers as "equivalent".
 * Both properties live in this pure function so they pin headless; the
 * numerics they select are still gated by chunk-prefill-test.mjs on a GPU.
 */

import { describe, test, expect } from 'vitest'
import { pickChunkGemm, type ChunkGemmName } from '../../src/zero-tvm/engine-core.js'

const FULL = {
  chunkTiled: false,
  dimsOK: true,
  capTiles64: true,
  e5Ready: true,
  sgmatReady: true,
  cap: 1024,
} as const

const used = (caps: Parameters<typeof pickChunkGemm>[0]): ChunkGemmName =>
  pickChunkGemm(caps).used

describe('automatic selection (no explicit ?chunkgemm=)', () => {
  test('full house runs E5', () => {
    expect(used({ ...FULL })).toBe('e5')
  })

  test('missing E5 degrades to sgmat by itself', () => {
    expect(used({ ...FULL, e5Ready: false })).toBe('sgmat')
  })

  test('cap not tiling by 64 degrades E5 to sgmat', () => {
    expect(used({ ...FULL, capTiles64: false, cap: 96 })).toBe('sgmat')
  })

  test('no matrix unit runs matvec, and does NOT throw', () => {
    const pick = pickChunkGemm({ ...FULL, e5Ready: false, sgmatReady: false })
    expect(pick.used).toBe('matvec')
    expect(pick.rejected).toBeNull()
  })

  test('bad dims run matvec, and do NOT throw when nothing was asked', () => {
    const pick = pickChunkGemm({ ...FULL, dimsOK: false })
    expect(pick.used).toBe('matvec')
    expect(pick.rejected).toBeNull()
  })

  test('chunkTiled opts into tiled when the matrix unit is absent', () => {
    expect(used({ ...FULL, e5Ready: false, sgmatReady: false, chunkTiled: true })).toBe('tiled')
  })

  test('matrix unit beats chunkTiled', () => {
    expect(used({ ...FULL, chunkTiled: true })).toBe('e5')
  })
})

describe('explicit ?chunkgemm= that can run', () => {
  test.each(['e5', 'sgmat', 'tiled', 'matvec'] as const)('%s is honored', (want) => {
    const pick = pickChunkGemm({ ...FULL, want })
    expect(pick.used).toBe(want)
    expect(pick.rejected).toBeNull()
  })

  test('sgmat is honored when E5 is missing', () => {
    const pick = pickChunkGemm({ ...FULL, want: 'sgmat', e5Ready: false })
    expect(pick.used).toBe('sgmat')
    expect(pick.rejected).toBeNull()
  })
})

describe('explicit ?chunkgemm= that cannot run throws', () => {
  test('e5 at a non-64 cap names the cap', () => {
    const pick = pickChunkGemm({ ...FULL, want: 'e5', capTiles64: false, cap: 96 })
    expect(pick.used).toBe('sgmat')
    expect(pick.rejected).toContain("asked for 'e5', can only run 'sgmat'")
    expect(pick.rejected).toContain('cap must be %64 (got 96)')
  })

  test('e5 without pipelines names the missing unit', () => {
    const pick = pickChunkGemm({ ...FULL, want: 'e5', e5Ready: false, sgmatReady: false })
    expect(pick.used).toBe('matvec')
    expect(pick.rejected).toContain('usually a device without chromium-experimental-subgroup-matrix')
  })

  test('sgmat without pipelines falls to matvec loudly', () => {
    const pick = pickChunkGemm({ ...FULL, want: 'sgmat', sgmatReady: false, e5Ready: false })
    expect(pick.used).toBe('matvec')
    expect(pick.rejected).toContain("asked for 'sgmat'")
  })

  test('bad dims report the dims, whatever was asked', () => {
    for (const want of ['e5', 'sgmat', 'tiled'] as const) {
      const pick = pickChunkGemm({ ...FULL, want, dimsOK: false })
      expect(pick.used).toBe('matvec')
      expect(pick.rejected).toContain('dims do not tile')
    }
  })

  test('matvec is always honor-able: never rejects', () => {
    const pick = pickChunkGemm({
      want: 'matvec', chunkTiled: false, dimsOK: false,
      capTiles64: false, e5Ready: false, sgmatReady: false, cap: 96,
    })
    expect(pick.used).toBe('matvec')
    expect(pick.rejected).toBeNull()
  })

  test('chunkTiled hijacking an explicit matvec rejects', () => {
    // want matvec + chunkTiled + good dims selects tiled, which is not what
    // was asked: the throw names the unbuilt-pipeline tail, not the dims.
    const pick = pickChunkGemm({ ...FULL, want: 'matvec', chunkTiled: true })
    expect(pick.used).toBe('tiled')
    expect(pick.rejected).toContain("asked for 'matvec', can only run 'tiled'")
    expect(pick.rejected).toContain('that pipeline was not built')
  })
})
