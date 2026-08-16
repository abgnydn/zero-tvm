/**
 * DEEDS — things that actually happened on this device, kept as a set of ids
 * in localStorage. Presence only, never a count: a deed is lit or it is not,
 * which keeps this inside the house numbers policy (no tallies on
 * decorative surfaces). Recording is idempotent and failure-silent (private
 * mode just never remembers).
 *
 * No DOM, no GPU imports — safe for any surface, including the rooms guest.
 * `recordFeat` dispatches `zt-feat` on document so a mounted deeds rail can
 * repaint without polling.
 */

export type FeatId =
  | 'first-summon'   // booted any model in the entrance
  | 'lane-dense' | 'lane-hybrid' | 'lane-moe'   // spoke with each lane
  | 'heavy'          // boot footprint ≥ 10 GB (weights + KV)
  | 'pooled'         // booted a lean expert-pool build
  | 'long-ctx'       // booted a long-context build
  | 'canvas'         // ran model-written code in the canvas

export interface FeatDef { id: FeatId; name: string; desc: string }

/** Display order + copy. Descriptions state the real trigger — a deed the
 *  reader cannot tell how to earn is decoration. */
export const FEATS: FeatDef[] = [
  { id: 'first-summon', name: 'First Summoning', desc: 'Boot a model in the entrance' },
  { id: 'lane-dense', name: 'Dense Communion', desc: 'Speak with a dense-attention model' },
  { id: 'lane-hybrid', name: 'Delta Communion', desc: 'Speak with a recurrent hybrid' },
  { id: 'lane-moe', name: 'Woke the Swarm', desc: 'Speak with a sparse MoE' },
  { id: 'heavy', name: 'Heavyweight', desc: 'Boot a ≥10 GB footprint' },
  { id: 'pooled', name: 'Lean Build', desc: 'Boot an expert-pool memory build' },
  { id: 'long-ctx', name: 'Long Sight', desc: 'Boot a long-context build' },
  { id: 'canvas', name: 'Made It Real', desc: 'Run model-written code in the canvas' },
]

const KEY = 'zt-feats'

export function feats(): Set<string> {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? '[]') as unknown
    return new Set(Array.isArray(raw) ? raw.filter((x): x is string => typeof x === 'string') : [])
  } catch {
    return new Set()
  }
}

export function recordFeat(id: FeatId): void {
  try {
    const have = feats()
    if (have.has(id)) return
    have.add(id)
    localStorage.setItem(KEY, JSON.stringify([...have]))
    document.dispatchEvent(new CustomEvent('zt-feat', { detail: id }))
  } catch { /* private mode — the deed happened, it just is not remembered */ }
}
