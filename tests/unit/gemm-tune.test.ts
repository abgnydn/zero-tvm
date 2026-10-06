// GEMM AUTOTUNE policy — the boot-time pick must be deterministic,
// versioned, and boredom-proof: ties reproduce the ladder, stale entries
// re-tune, and a store that throws never breaks boot. Hand-derived small
// numbers throughout; the GPU timing itself is covered by prefill-gemm-ab
// (manual) and by the boot log line on every real boot.

import { describe, expect, it } from 'vitest'
import {
  TUNE_TTL_MS, TUNE_VERSION,
  deviceFingerprint, loadTunedPick, median, pickFromTimings,
  resolveTunedGemm, saveTunedPick, tuneKey,
  type TuneStore,
} from '../../src/zero-tvm/gemm-tune.ts'

const memStore = (init: Record<string, string> = {}): TuneStore => {
  const m = new Map(Object.entries(init))
  return {
    load: (k) => m.get(k) ?? null,
    save: (k, v) => { m.set(k, v) },
  }
}

const saved = (over: object = {}) => JSON.stringify({
  version: TUNE_VERSION, pick: 'e5', ms: { e5: 1 }, candidates: ['e5', 'sgmat', 'matvec'],
  at: Date.now(), ...over,
})

describe('median', () => {
  it('takes the middle, averaging the middle two on even counts', () => {
    expect(median([3, 1, 2])).toBe(2)
    expect(median([4, 1, 3, 2])).toBe(2.5)
  })
})

describe('pickFromTimings', () => {
  it('takes the lowest geometric mean across sizes', () => {
    // e5 wins small, loses big; sgmat steady — geomean decides, not one size.
    const medians = { e5: [1, 1, 16], sgmat: [2, 2, 2] }
    // geomean e5 = 2.52, sgmat = 2 → sgmat.
    expect(pickFromTimings(medians, ['e5', 'sgmat', 'matvec'])).toBe('sgmat')
  })

  it('breaks exact ties toward the static ladder order, never a coin flip', () => {
    const medians = { e5: [2, 2], sgmat: [2, 2], matvec: [2, 2] }
    expect(pickFromTimings(medians, ['e5', 'sgmat', 'matvec'])).toBe('e5')
    expect(pickFromTimings(medians, ['sgmat', 'matvec'])).toBe('sgmat')
  })

  it('ignores candidates with no timings', () => {
    expect(pickFromTimings({ matvec: [5] }, ['e5', 'sgmat', 'matvec'])).toBe('matvec')
  })
})

describe('tuneKey', () => {
  it('names version, device and spec — a shader change re-tunes by version alone', () => {
    expect(tuneKey('apple/m4', 'llama32')).toBe(`ztvm.gemmTune.v${TUNE_VERSION}.apple/m4.llama32`)
  })
})

describe('deviceFingerprint', () => {
  it('prefers adapterInfo, falls back to limits, null on nothing', () => {
    expect(deviceFingerprint({ adapterInfo: { vendor: 'apple' } } as unknown as GPUDevice))
      .toBe('apple/_/_')
    expect(deviceFingerprint({ limits: { maxComputeWorkgroupsPerDimension: 65535, maxStorageBufferBindingSize: 1, maxBufferSize: 2 } } as unknown as GPUDevice))
      .toBe('limits/65535/1/2')
    expect(deviceFingerprint({} as unknown as GPUDevice)).toBe(null)
  })
})

describe('loadTunedPick', () => {
  const key = 'k'
  const runnable = ['e5', 'sgmat', 'matvec'] as const

  it('returns a fresh, versioned, covering pick', () => {
    expect(loadTunedPick(memStore({ [key]: saved() }), key, [...runnable])).toBe('e5')
  })

  it('re-tunes on version bump, TTL expiry, unknown pick, and candidate drift', () => {
    const s = (o: object) => memStore({ [key]: saved(o) })
    expect(loadTunedPick(s({ version: TUNE_VERSION + 1 }), key, [...runnable])).toBe(null)
    expect(loadTunedPick(s({ at: Date.now() - TUNE_TTL_MS - 1 }), key, [...runnable])).toBe(null)
    expect(loadTunedPick(s({ pick: 'tiled' }), key, [...runnable])).toBe(null)
    // A driver update enabled the matrix unit: the stored set lacks sgmat.
    expect(loadTunedPick(s({ candidates: ['matvec'] }), key, [...runnable])).toBe(null)
    expect(loadTunedPick(memStore(), key, [...runnable])).toBe(null)
    expect(loadTunedPick(null, key, [...runnable])).toBe(null)
  })

  it('survives corrupt JSON and a throwing store', () => {
    expect(loadTunedPick(memStore({ [key]: 'not json' }), key, [...runnable])).toBe(null)
    const bad: TuneStore = { load: () => { throw new Error('x') }, save: () => { throw new Error('x') } }
    expect(loadTunedPick(bad, key, [...runnable])).toBe(null)
    expect(() => saveTunedPick(bad, key, 'e5', {}, ['e5'])).not.toThrow()
  })
})

describe('resolveTunedGemm', () => {
  const spec = { id: 'llama32' } as unknown as import('../../src/compiler/model-spec.ts').ModelSpec

  it('explicit always wins and never touches the store or probe', async () => {
    let probed = false
    const r = await resolveTunedGemm({
      spec, runnable: ['e5', 'sgmat', 'matvec'], explicit: 'matvec',
      store: memStore(), adapterId: 'd',
      runProbe: async () => { probed = true; return {} },
    })
    expect(r).toEqual({ pick: 'matvec', source: 'explicit' })
    expect(probed).toBe(false)
  })

  it('a single runnable candidate defers to the ladder without probing', async () => {
    let probed = false
    const r = await resolveTunedGemm({
      spec, runnable: ['matvec'],
      runProbe: async () => { probed = true; return {} },
    })
    expect(r).toBe(null)
    expect(probed).toBe(false)
  })

  it('replays a stored pick, else probes, persists, and reports the source', async () => {
    const store = memStore()
    const probe = async () => ({ e5: { 64: [1, 1.1], 256: [2, 2.2] }, sgmat: { 64: [3, 3], 256: [4, 4] } })
    const first = await resolveTunedGemm({
      spec, runnable: ['e5', 'sgmat'], store, adapterId: 'd', runProbe: probe,
    })
    expect(first?.pick).toBe('e5')
    expect(first?.source).toBe('probed')
    // Second call with a probe that would pick otherwise still replays stored.
    const second = await resolveTunedGemm({
      spec, runnable: ['e5', 'sgmat'], store, adapterId: 'd',
      runProbe: async () => ({ sgmat: { 64: [0.1] } }),
    })
    expect(second).toEqual({ pick: 'e5', source: 'stored' })
  })

  it('a throwing probe falls back to the ladder (null), not to an error', async () => {
    const r = await resolveTunedGemm({
      spec, runnable: ['e5', 'sgmat'], store: memStore(), adapterId: 'd',
      runProbe: async () => { throw new Error('gpu busy') },
    })
    expect(r).toBe(null)
  })
})
