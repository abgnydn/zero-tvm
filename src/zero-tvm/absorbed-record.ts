import { noteAbsorbed as pureNoteAbsorbed, reuseStart, type ReuseState } from './prefix-reuse.ts'

/**
 * The absorbed-token record — what the KV cache and (for hybrid) the GDN
 * recurrence currently encode. Mutations live here; the rules live in
 * prefix-reuse.ts and are reused headlessly by tests.
 */
export class AbsorbedRecord {
  private absorbed: number[] = []
  private valid = true
  private readonly prefixReuse: boolean
  private readonly hybrid: boolean
  private readonly getGdnStatePos: () => number

  constructor(prefixReuse: boolean, hybrid: boolean, getGdnStatePos: () => number) {
    this.prefixReuse = prefixReuse
    this.hybrid = hybrid
    this.getGdnStatePos = getGdnStatePos
  }

  snapshot(): ReuseState {
    return {
      absorbed: this.absorbed,
      absorbedValid: this.valid,
      prefixReuse: this.prefixReuse,
      hybrid: this.hybrid,
      gdnStatePos: this.getGdnStatePos(),
    }
  }

  note(position: number, id: number): void {
    const s = this.snapshot()
    pureNoteAbsorbed(s, position, id)
    this.absorbed = s.absorbed
    this.valid = s.absorbedValid
  }

  truncate(position: number): void {
    this.absorbed.length = position
  }

  clear(): void {
    this.absorbed = []
    this.valid = true
  }

  invalidate(): void {
    this.valid = false
  }

  /** Replace the whole record and mark valid — used by importKV. */
  setFrom(ids: readonly number[]): void {
    this.absorbed = [...ids]
    this.valid = true
  }

  get length(): number { return this.absorbed.length }
  get isValid(): boolean { return this.valid }
  get ids(): readonly number[] { return this.absorbed }

  lcp(promptIds: readonly number[]): number {
    const max = Math.min(this.absorbed.length, promptIds.length)
    let lcp = 0
    while (lcp < max && this.absorbed[lcp] === promptIds[lcp]) lcp++
    return lcp
  }

  computeReuseStart(promptIds: readonly number[]): number {
    return reuseStart(this.snapshot(), [...promptIds])
  }
}
