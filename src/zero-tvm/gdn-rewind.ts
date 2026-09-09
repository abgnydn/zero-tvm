import { dropSnapshotsAbove, rewindSlot } from './prefix-reuse.ts'
import type { ModelSpec } from '../compiler/model-spec.ts'

/**
 * GDN rewind ring. Hybrid prefix reuse is all-or-nothing without these: the
 * recurrent state cannot be rewound a token at a time, so a divergence would
 * force a full re-prefill. A ring of snapshots taken at chunk boundaries turns
 * that into "replay from the nearest boundary at or before the divergence".
 *
 * Allocated lazily on first use — devices that can never fill the ring should
 * not pay for it at build time.
 */
export class GdnRewindRing {
  readonly slotCount: number
  private readonly device: GPUDevice
  private readonly S: ModelSpec
  private readonly gdnConvState: readonly (GPUBuffer | null)[]
  private readonly gdnRecurState: readonly (GPUBuffer | null)[]
  private readonly makeBuf: (device: GPUDevice, size: number, label: string) => GPUBuffer
  private conv: (GPUBuffer | null)[][] = []
  private recur: (GPUBuffer | null)[][] = []
  private readonly pos: number[]
  private next = 0
  private allocated = false

  constructor(opts: {
    device: GPUDevice
    spec: ModelSpec
    gdnConvState: readonly (GPUBuffer | null)[]
    gdnRecurState: readonly (GPUBuffer | null)[]
    prefixReuse: boolean
    hybrid: boolean
    makeBuf: (device: GPUDevice, size: number, label: string) => GPUBuffer
  }) {
    this.device = opts.device
    this.S = opts.spec
    this.gdnConvState = opts.gdnConvState
    this.gdnRecurState = opts.gdnRecurState
    this.makeBuf = opts.makeBuf
    this.slotCount = opts.hybrid && opts.prefixReuse ? 4 : 0
    this.pos = new Array(this.slotCount).fill(-1)
  }

  private ensureBuffers(): void {
    if (this.slotCount === 0 || this.allocated) return
    this.allocated = true
    for (let slot = 0; slot < this.slotCount; slot++) {
      const conv: (GPUBuffer | null)[] = []
      const recur: (GPUBuffer | null)[] = []
      for (let L = 0; L < this.S.layers; L++) {
        if (this.gdnConvState[L]) {
          conv.push(this.makeBuf(this.device, (this.S.gdnConvK - 1) * this.S.gdnQkvDim * 2, `gdnCkptConv_${slot}_${L}`))
          recur.push(this.makeBuf(this.device, this.S.gdnVHeads * this.S.gdnStatePerHead * 4, `gdnCkptRecur_${slot}_${L}`))
        } else { conv.push(null); recur.push(null) }
      }
      this.conv.push(conv)
      this.recur.push(recur)
    }
  }

  /** Snapshot the live GDN state as the rewind point for `pos`. */
  save(pos: number): void {
    if (this.slotCount === 0) return
    this.ensureBuffers()
    const slot = this.next
    const enc = this.device.createCommandEncoder()
    for (let L = 0; L < this.S.layers; L++) {
      const c = this.gdnConvState[L], r = this.gdnRecurState[L]
      if (!c || !r) continue
      enc.copyBufferToBuffer(c, 0, this.conv[slot][L]!, 0, c.size)
      enc.copyBufferToBuffer(r, 0, this.recur[slot][L]!, 0, r.size)
    }
    this.device.queue.submit([enc.finish()])
    this.pos[slot] = pos
    this.next = (this.next + 1) % this.slotCount
  }

  /** @returns false if the ring holds nothing to restore. */
  restore(slot: number): boolean {
    if (!this.allocated) return false
    const enc = this.device.createCommandEncoder()
    for (let L = 0; L < this.S.layers; L++) {
      const c = this.gdnConvState[L], r = this.gdnRecurState[L]
      if (!c || !r) continue
      enc.copyBufferToBuffer(this.conv[slot][L]!, 0, c, 0, c.size)
      enc.copyBufferToBuffer(this.recur[slot][L]!, 0, r, 0, r.size)
    }
    this.device.queue.submit([enc.finish()])
    return true
  }

  /** Anything that moves the state without replaying tokens voids every slot. */
  invalidate(): void {
    this.pos.fill(-1)
    this.next = 0
  }

  /** Which snapshot to replay from, or -1 if none is usable. */
  findBest(lcp: number, promptLen: number): number {
    return rewindSlot(this.pos, lcp, promptLen)
  }

  /** Void every snapshot above `pos`. */
  dropAbove(pos: number): void {
    dropSnapshotsAbove(this.pos, pos)
  }

  /** Absorbed-token position stored in `slot`; -1 if empty. */
  position(slot: number): number {
    return this.pos[slot]
  }

  /** All non-null GPU buffers owned by the ring, for destruction. */
  buffers(): GPUBuffer[] {
    return [...this.conv.flat(), ...this.recur.flat()].filter((b): b is GPUBuffer => b !== null)
  }
}
