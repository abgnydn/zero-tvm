/**
 * E2E TESTS — Qwen3-Embedding-0.6B (?model=embed) via the smoke probe.
 *
 * Not a chat model: ?model=embed serves forwardEmbedding, and the chat pages
 * resolve it to Phi-3, so the chat/validate pattern does not apply. This
 * drives model-smoke.html's __embed hook instead and asserts the embedding
 * contract structurally: d-length f32 vector, all finite, L2-normalised,
 * and input-dependent.
 *
 * What this does NOT assert: pooling correctness (last-token vs CLS vs
 * mean). The spec comment documents that only retrieval against gold
 * similarities catches a wrong pooling — magnitude checks cannot. That
 * belongs in a reference comparison (mlx_lm), not here.
 *
 * GATED on the local weight mirror
 * (.weights-local/Qwen3-Embedding-0.6B-4bit-DWQ). If the mirror isn't
 * primed the whole file skips LOUDLY instead of failing.
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3-Embedding-0.6B-4bit-DWQ')
const HAVE_EMBED = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_EMBED) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[embed e2e] SKIPPING — checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it by downloading mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ there.\n',
  )
}

const BOOT_TIMEOUT_MS = 10 * 60 * 1000

// model-smoke.html's probe surface (untyped on DOM Window by design — it is
// a dev-only page, not app code). Cast inline in each browser callback:
// helpers defined here cannot cross the puppeteer boundary.
type SmokeWindow = Window & typeof globalThis & {
  __phase?: string
  __error?: string
  __embed?: (ids: number[]) => Promise<number[]>
}

// d for QWEN3_EMBEDDING_06B (model-spec.ts). Hardcoded so a spec typo that
// changes d fails here instead of asserting against itself.
const D = 1024

describe.skipIf(!HAVE_EMBED)('qwen3-embedding (?model=embed, smoke probe)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('forwardEmbedding returns a unit-norm d-vector per prompt', async () => {
    // model-smoke.html has no download gate or badge — boot is __phase.
    const page = await newPage('/model-smoke.html?model=embed')
    await page.waitForFunction(
      () => {
        const p = window as unknown as SmokeWindow
        return p.__phase === 'loaded' || p.__phase === 'error'
      },
      { timeout: BOOT_TIMEOUT_MS, polling: 1000 },
    )
    if ((await page.evaluate(() => (window as unknown as SmokeWindow).__phase)) === 'error') {
      throw new Error(await page.evaluate(() => (window as unknown as SmokeWindow).__error))
    }

    const idsA = [151644, 198, 3838, 374, 279, 14990, 3361, 248046]
    const idsB = [151644, 198, 16748, 279, 10146, 248046]
    const [vecA, vecB] = await page.evaluate(
      async ([a, b]: [number[], number[]]) => {
        const w = window as unknown as SmokeWindow
        return [await w.__embed!(a), await w.__embed!(b)]
      }, [idsA, idsB] as [number[], number[]])

    for (const [name, vec] of [['A', vecA], ['B', vecB]] as const) {
      expect(vec.length, `${name} has d entries`).toBe(D)
      expect(vec.every((x) => Number.isFinite(x)), `${name} all finite`).toBe(true)
      const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0))
      expect(norm, `${name} L2-normalised`).toBeGreaterThan(0.999)
      expect(norm, `${name} L2-normalised`).toBeLessThan(1.001)
    }

    // Input-dependent: two prompts must not share a direction.
    let dot = 0
    for (let i = 0; i < D; i++) dot += vecA[i] * vecB[i]
    console.log(`[embed] cosine(A, B) = ${dot.toFixed(4)}`)
    expect(Math.abs(dot), 'distinct prompts, distinct vectors').toBeLessThan(0.99)

    await page.close()
  })
})
