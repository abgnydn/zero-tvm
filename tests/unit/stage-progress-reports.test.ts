// A PIPELINE STAGE'S DOWNLOAD HAS TO REPORT MORE THAN ONCE.
//
// loadWeights sampled its buffer-progress callback at a fixed every-25th, a
// step sized for the whole model (the 27B plans 1127 buffers). A stage plans a
// fraction of that, and the callback also fires on the last buffer — so for
// any stage under 25 buffers the ONLY report was the final one, emitted after
// the download had already finished.
//
// That is not a cosmetic gap. Measured in the browser on 2026-08-29 against a
// one-layer stage on a 5 MB/s link: with the fixed step, #progress-detail
// changed ONCE in 77 seconds and #progress-status went 72.8 s without a word;
// with progressEvery, 22 and 32. The user-visible failure was an iPhone
// serving a split 27B that showed a bar frozen at 10% for minutes with no way
// to tell working from dead.
//
// The counts below come from planModel, not from a hand-written table: the
// number of buffers a stage plans is exactly what the step has to scale with,
// and a copy of it here would drift the first time a spec's layer shape moves.

import { describe, expect, it } from 'vitest'
import { planModel, progressEvery } from '../../src/zero-tvm/weight-loader-mlx.ts'
import {
  QWEN3_8_27B_4BIT, QWEN3_30B_A3B_4BIT, QWEN36_35B_A3B_Q3,
  LLAMA_3_2_1B_INSTRUCT_4BIT, type ModelSpec,
} from '../../src/compiler/model-spec.ts'

/** How many reports the loader's gate lets through for a load of `n` buffers —
 *  the same condition as loadWeights' onBuffer, which is the subject. */
function reports(n: number): number {
  const every = progressEvery(n)
  let out = 0
  for (let done = 1; done <= n; done++) if (done % every === 0 || done === n) out++
  return out
}

// Every MLX spec a split can run today. A stage of ONE layer is the smallest
// real load and the case that was silent.
const SPECS: [string, ModelSpec][] = [
  ['qwen38', QWEN3_8_27B_4BIT],
  ['qwen30b', QWEN3_30B_A3B_4BIT],
  ['qwen36q3', QWEN36_35B_A3B_Q3],
  ['llama32', LLAMA_3_2_1B_INSTRUCT_4BIT],
]

describe('stage download progress', () => {
  for (const [name, spec] of SPECS) {
    it(`${name}: a middle one-layer stage reports as it goes, not once at the end`, () => {
      const mid = Math.floor(spec.layers / 2)
      const n = planModel(spec, { start: mid, end: mid + 1 }).length
      // The premise: a one-layer stage really is smaller than the old
      // every-25th step, which is why that step could never fire.
      expect(n).toBeLessThan(25)
      expect(reports(n)).toBe(n)
    })
  }

  it('a whole-model load stays cheap to report', () => {
    const n = planModel(QWEN3_8_27B_4BIT).length
    expect(n).toBeGreaterThan(1000)
    expect(reports(n)).toBeLessThanOrEqual(40)
  })

  it('never divides by zero or reports every buffer twice', () => {
    expect(progressEvery(0)).toBe(1)
    expect(progressEvery(1)).toBe(1)
    expect(reports(1)).toBe(1)
  })
})
