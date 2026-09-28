// GdnRewindRing.lastSavedSlot — the slot a save-then-restore caller must use.
//
// A caller that snapshots and then repeatedly restores the SAME snapshot (the
// kev hybrid rewind fallback) cannot use findBest to locate it: an older slot
// can hold the same position, and rewindSlot breaks that tie toward the
// lowest index — the stale state, silently. lastSavedSlot tracks the
// most-recently-written slot instead. Expectations below are hand-derived
// from that contract, not from running the class.

import { describe, expect, it } from 'vitest'
import { GdnRewindRing } from '../../src/zero-tvm/gdn-rewind.ts'
import type { ModelSpec } from '../../src/compiler/model-spec.ts'

const spec = { layers: 2, gdnConvK: 4, gdnQkvDim: 8, gdnVHeads: 2, gdnStatePerHead: 16 } as unknown as ModelSpec
const buf = (size: number) => ({ size }) as unknown as GPUBuffer
const device = {
  createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish() { return {} } }),
  queue: { submit() {} },
} as unknown as GPUDevice

const ring = (prefixReuse = true) => new GdnRewindRing({
  device,
  spec,
  gdnConvState: [buf(48), null],
  gdnRecurState: [buf(128), null],
  prefixReuse,
  hybrid: true,
  makeBuf: (_d, size) => buf(size),
})

describe('GdnRewindRing.lastSavedSlot', () => {
  it('is -1 on a fresh ring, and save is a no-op when the ring is disabled', () => {
    const r = ring()
    expect(r.lastSavedSlot()).toBe(-1)
    expect(r.position(0)).toBe(-1)
    const off = ring(false)
    expect(off.slotCount).toBe(0)
    off.save(10)
    expect(off.lastSavedSlot()).toBe(-1)
  })

  it('tracks the most recently written slot across saves and wraparound', () => {
    const r = ring()
    r.save(10)
    expect(r.lastSavedSlot()).toBe(0)
    expect(r.position(0)).toBe(10)
    r.save(20)
    expect(r.lastSavedSlot()).toBe(1)
    expect(r.position(0)).toBe(10)
    r.save(30)
    r.save(40)
    expect(r.lastSavedSlot()).toBe(3)
    r.save(50) // wraps: overwrites slot 0, the oldest snapshot
    expect(r.lastSavedSlot()).toBe(0)
    expect(r.position(0)).toBe(50)
  })

  it('resets on invalidate, and restore mirrors the save state', () => {
    const r = ring()
    expect(r.restore(0)).toBe(false) // nothing allocated yet
    r.save(10)
    expect(r.restore(r.lastSavedSlot())).toBe(true)
    r.invalidate()
    expect(r.lastSavedSlot()).toBe(-1)
    expect(r.position(0)).toBe(-1)
  })
})
