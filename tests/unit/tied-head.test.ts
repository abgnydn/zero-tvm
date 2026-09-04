/**
 * UNIT TESTS — tied-embedding head alias (assertTiedHead in weight-loader-mlx.ts).
 *
 * On a tied spec the lm_head IS the embedding table — same buffers, not
 * copies — on every stage including pipeline slices. A loader edit that
 * loads them separately (different offsets, or only on some stages)
 * produces fluent wrong logits, never an error. These pin the alias on
 * stub weight objects; the assembly itself calls the assert, so a future
 * edit that breaks the alias fails at load, not in a user's session.
 */

import { describe, test, expect } from 'vitest'
import { assertTiedHead } from '../../src/zero-tvm/weight-loader-mlx.ts'

const tied = { id: 'qwen3-4b', tiedEmbeddings: true }
const untied = { id: 'phi-3', tiedEmbeddings: false }

describe('assertTiedHead', () => {
  test('untied specs are exempt', () => {
    const buf = {}
    expect(() => assertTiedHead({
      embdWeights: buf, lmHeadWeights: {}, embdScales: buf, lmHeadScales: {},
      embdBiases: buf, lmHeadBiases: {},
    }, untied)).not.toThrow()
  })

  test('aliased buffers pass', () => {
    const w = {}, s = {}, b = {}
    expect(() => assertTiedHead({
      embdWeights: w, lmHeadWeights: w, embdScales: s, lmHeadScales: s,
      embdBiases: b, lmHeadBiases: b,
    }, tied)).not.toThrow()
  })

  test.each([
    ['weights', { embdWeights: {}, lmHeadWeights: {} }],
    ['scales', { embdScales: {}, lmHeadScales: {} }],
    ['biases', { embdBiases: {}, lmHeadBiases: {} }],
  ])('separately-loaded %s throws naming both sides', (_label, partial) => {
    const shared = {}
    expect(() => assertTiedHead({
      embdWeights: shared, lmHeadWeights: shared,
      embdScales: shared, lmHeadScales: shared,
      embdBiases: shared, lmHeadBiases: shared,
      ...partial,
    }, tied)).toThrow(/tied embeddings but lmHead\w+ is not embd\w+/)
  })

  test('error names the spec', () => {
    expect(() => assertTiedHead({ embdWeights: {}, lmHeadWeights: {} }, tied))
      .toThrow('qwen3-4b')
  })
})
