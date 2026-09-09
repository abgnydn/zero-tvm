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
 *
 * ── WHY THIS FILE WAS REWRITTEN 2026-08-23 ───────────────────────────────────
 *
 * It asserted a resolution the engine does not take, and stayed green for it.
 * Three compounding faults, all of the same kind — the fixture could not
 * represent the shipped configuration:
 *
 *  1. The stub was a hand-typed NAME LIST that had fallen four entries behind
 *     the real `Pipelines`: the four `*Vec4hAffine` fields, compiled since
 *     2026-08-13, were simply absent. A resolution that wanted one got
 *     `undefined`, so the ladder fell through to the rung below and the test
 *     asserted THAT.
 *  2. The flag set spread SCALAR_VARIANTS and never set `vec4Affine`, which the
 *     shipped page defaults ON (variants.ts: `sgAll && ?vec4 !== 0`). So the
 *     browser took the vec4h rung and the test took the narrow one.
 *  3. Its second case asserted Qwen3.6 "never resolves to a vec4 pipeline",
 *     which is the NEGATION of shipped behaviour — held green only by (1) and
 *     (2), and by a comment claiming the generator forbids affine+vec4, which
 *     it does not.
 *
 * The stub is now a Proxy that answers ANY field with its own name, so there is
 * no list to fall behind. That is the actual fix — not "remember to update the
 * list", which is what failed. A resolution reaching for a pipeline this file
 * has never heard of gets that pipeline's name, not `undefined`, so the ladder
 * cannot silently fall through to a rung the engine would not take.
 */
import { describe, it, expect } from 'vitest'
import { resolveVariantPipelines, SCALAR_VARIANTS, parseVariantFlags } from '../../src/zero-tvm/variants.ts'
import type { Pipelines } from '../../src/compiler/compiler.ts'
import { PHI3, QWEN35_4B, QWEN36_35B_A3B } from '../../src/compiler/model-spec.ts'

/** Every Pipelines field answers with its own name, so a resolution reads as a
 *  string. There is deliberately no enumeration: the previous hand-kept NAMES
 *  array fell four entries behind the real interface and turned every
 *  resolution that wanted one of them into a silent fall-through. A Proxy
 *  cannot fall behind. */
const NAMES: Record<keyof Pipelines, string> = new Proxy({} as Record<keyof Pipelines, string>, {
  get: (_, k: string) => k,
  has: () => true,
  ownKeys: () => [],
})

const stub = (): Pipelines => NAMES as unknown as Pipelines

/** What a subgroup-capable browser actually produces — vec4Affine ON, which is
 *  the flag the old fixture could not express. Derived from the real parser so
 *  it cannot drift from the page. */
const ON_DEVICE = { hasSubgroupsFeature: true, sgSizeOk: true }
const BROWSER = parseVariantFlags('?matmul=tiled', ON_DEVICE)
/** The same device, with the bisection lever thrown. */
const SCALAR_ON_DEVICE = parseVariantFlags('?matmul=scalar', ON_DEVICE)

describe('the fixture can represent the shipped configuration', () => {
  it('a subgroup device defaults vec4Affine ON', () => {
    // If this ever goes false, every affine assertion below is testing the
    // narrow rung while the page takes the wide one — the exact fault that
    // made this file green and wrong.
    expect(BROWSER.vec4Affine, 'vec4Affine is what selects the vec4h affine rungs').toBe(true)
    expect(BROWSER.subgroups).toBe(true)
    expect(SCALAR_ON_DEVICE.matmul).toBe('scalar')
    expect(SCALAR_ON_DEVICE.subgroups, '?matmul=scalar must not also turn subgroups off').toBe(true)
  })
})

describe('resolveMatmul affine gating', () => {
  it('Qwen3.6 resolves every matmul instance to the affine family', () => {
    const R = resolveVariantPipelines(BROWSER, stub(), QWEN36_35B_A3B)
    for (const p of [R.matmul, R.matmulF32, R.matmulOProj, R.matmulFfnDown]) {
      expect(String(p)).toMatch(/Affine/)
    }
    expect(R.matmulLabel).toContain('affine')
  })

  it('Qwen3.6 takes the vec4h affine rung where K % 512 allows — the shipped path', () => {
    // d=2048 and qDim=4096 are both % 512, so the browser resolves the WIDE
    // affine kernel. The previous version of this file asserted the narrow one
    // and called the wide one impossible.
    const R = resolveVariantPipelines(BROWSER, stub(), QWEN36_35B_A3B)
    expect(R.matmul).toBe('int4MatmulTiledVec4hAffine')
    expect(R.matmulF32).toBe('int4MatmulF32TiledVec4hAffine')
  })

  it('affine + vec4 coexist — the generator does not forbid it', () => {
    // Pinning the corrected fact, because the old comment asserting the
    // opposite is what made the wrong assertion look reasonable.
    expect(String(resolveVariantPipelines(BROWSER, stub(), QWEN36_35B_A3B).matmul))
      .toMatch(/Vec4h.*Affine|Affine/)
  })

  it('?vec4=0 falls back to the narrow affine rung', () => {
    const narrow = parseVariantFlags('?matmul=tiled&vec4=0', ON_DEVICE)
    expect(narrow.vec4Affine).toBe(false)
    const R = resolveVariantPipelines(narrow, stub(), QWEN36_35B_A3B)
    expect(R.matmul).toBe('int4MatmulTiledAffine')
  })

  it('the three symmetric specs resolve exactly as before', () => {
    const q35 = resolveVariantPipelines(BROWSER, stub(), QWEN35_4B)
    expect(q35.matmul).toBe('int4MatmulTiledVec4h')   // d=2560: %1024 fails, %512 passes
    expect(q35.matmulOProj).toBe('int4MatmulTiledVec4') // qDim=4096
    expect(q35.matmulFfnDown).toBe('int4MatmulTiledVec4') // ffn=9216 = 9*1024
    const phi = resolveVariantPipelines(BROWSER, stub(), PHI3)
    expect(phi.matmul).toBe('int4MatmulTiledVec4')
    expect(phi.matmulLabel).not.toContain('affine')
  })
})

describe('?matmul=scalar reaches scalar on an MLX checkpoint', () => {
  // The bisection lever. Before 2026-08-23 the affine ladder's two `sg` rungs
  // did not consult `variant` — unlike the tiled rungs beside them and unlike
  // the entire symmetric ladder — so on a subgroup-capable device (the chat
  // page's default) ?matmul=scalar answered sg_vec4h_affine or sg_affine.
  // The one instrument aimed at silent kernel wrongness was dead on exactly
  // the models most likely to have it, and ?sg=0 only appeared to substitute
  // because it stops COMPILING the pipeline rather than declining to resolve it.
  for (const spec of [QWEN36_35B_A3B]) {
    it(`${spec.id} resolves scalar_affine, not an sg rung`, () => {
      const R = resolveVariantPipelines(SCALAR_ON_DEVICE, stub(), spec)
      expect(R.matmulLabel).toBe('scalar_affine')
      for (const p of [R.matmul, R.matmulF32, R.matmulOProj, R.matmulFfnDown]) {
        expect(String(p), 'no subgroup or vec4 rung may survive ?matmul=scalar')
          .not.toMatch(/Sg|Vec4|Tiled/)
      }
    })
  }

  it('and still picks the affine family, not the symmetric one', () => {
    const R = resolveVariantPipelines(SCALAR_ON_DEVICE, stub(), QWEN36_35B_A3B)
    expect(R.matmul).toBe('int4MatmulAffine')
    expect(R.matmulF32).toBe('int4MatmulF32Affine')
  })

  it('a symmetric spec reaches plain scalar the same way', () => {
    const R = resolveVariantPipelines(SCALAR_ON_DEVICE, stub(), PHI3)
    expect(R.matmulLabel).toBe('scalar')
  })
})

describe('model selection reachability', () => {
  it('?model=qwen36 selects the MoE spec, and the three existing mappings are unchanged', async () => {
    // model-select imports weight-loader, which reads GPUBufferUsage at MODULE
    // scope — so importing it under Node throws before any code runs. The real
    // values do not matter here; only that the constant exists.
    ;(globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage ??= {
      STORAGE: 128, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, MAP_READ: 1,
    }
    const { specFromSearch } = await import('../../src/zero-tvm/model-select.ts')
    expect(specFromSearch('?model=qwen36').id).toBe('qwen36-35b-a3b')
    expect(specFromSearch('?model=qwen35').id).toBe(QWEN35_4B.id)
    expect(specFromSearch('?model=qwen3').id).toBe('qwen3-4b')
    expect(specFromSearch('').id).toBe(PHI3.id)
  })

  it('SCALAR_VARIANTS is still the all-off preset it is used as', () => {
    expect(SCALAR_VARIANTS.subgroups).toBe(false)
    expect(SCALAR_VARIANTS.vec4Affine).toBe(false)
    expect(SCALAR_VARIANTS.matmul).toBe('scalar')
  })
})
