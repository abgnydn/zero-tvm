/**
 * UNIT TEST — the library entry point (src/lib/index.ts).
 *
 * The regression that matters is the IMPORT itself. weight-loader.ts reads
 * GPUBufferUsage at module scope, so an eager import of it — directly, or
 * through model-select.ts / engine-core.ts / loading-ui.ts — makes
 * `import 'zero-tvm'` throw `GPUBufferUsage is not defined` in Node, in a
 * bundler's SSR pass, and in any browser without WebGPU. This file runs in
 * plain Node with no GPU globals: if one of index.ts's dynamic imports is
 * ever hoisted to the top level, it fails HERE rather than in a consumer's
 * build. Nothing below touches a GPU.
 */

import { describe, test, expect } from 'vitest'
import { SHIPPED_MODELS, modelBranding } from '../../src/zero-tvm/model-registry.js'

describe('lib entry point', () => {
  test('imports without throwing when there is no WebGPU', async () => {
    // Pin the premise: this process really has no WebGPU globals, so a
    // passing import below means something.
    expect((globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage).toBeUndefined()
    expect((globalThis.navigator as { gpu?: unknown } | undefined)?.gpu).toBeUndefined()

    const mod = await import('../../src/lib/index.js')
    expect(typeof mod.listModels).toBe('function')
    expect(typeof mod.createEngine).toBe('function')
  })

  test('listModels() returns the shipped models', async () => {
    const { listModels } = await import('../../src/lib/index.js')
    const models = listModels()

    // One row per VALIDATED registry entry, in registry order — adding a
    // model with scripts/add-model.mjs must reach this surface with no extra
    // step, but only once its numerics are checked: `pending` marks a spec
    // that is generated and compile-gated and nothing more, and this list is
    // the library's claim of what runs (ModelInfo carries no flag a consumer
    // could filter on, so the filter has to live here).
    const shipped = SHIPPED_MODELS.filter((m) => !modelBranding(m.spec).pending)
    expect(models.map((m) => m.param)).toEqual(shipped.map((m) => m.param))
    expect(models.map((m) => m.id)).toEqual(shipped.map((m) => m.spec.id))
    // ...and a pending model is genuinely absent, not merely reordered.
    for (const m of SHIPPED_MODELS.filter((x) => modelBranding(x.spec).pending)) {
      expect(models.some((r) => r.param === m.param)).toBe(false)
    }

    // Branding is present, not blank — these strings are the whole point of
    // the call. (rateLabel is deliberately '' when no measured number exists.)
    for (const m of models) {
      expect(m.name.length).toBeGreaterThan(0)
      expect(m.params.length).toBeGreaterThan(0)
      expect(m.sizeLabel.length).toBeGreaterThan(0)
    }

    // The default model keeps the empty param — every pre-registry URL and
    // every `createEngine()` with no argument boots Phi-3.
    const dflt = models.find((m) => m.param === '')
    expect(dflt?.name).toBe('Phi-3-mini')
  })

  test('createEngine rejects an unknown model instead of silently booting Phi-3', async () => {
    const { createEngine } = await import('../../src/lib/index.js')
    await expect(createEngine({ model: 'not-a-model' })).rejects.toThrow(/unknown model/)
  })
})
