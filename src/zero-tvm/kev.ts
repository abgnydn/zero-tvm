/**
 * KEV — typed decisions over the engine (jaredpalmer/kev, TypeSafe's
 * /v1/systemone shape). One STATE plus typed QUESTIONS in; a probability per
 * option out, from the pointer head over the backbone's hidden rows. Nothing
 * is generated.
 *
 * The sequence is kev's own (kev/model.py `encode`, `option_isolation=False`):
 *
 *   [<state> s…]  then per question  [<q> instr <opt> o </opt> … <decide>]
 *
 * with the branch positions CONTINUING from the state — so state + one branch
 * is an ordinary causal row, and running each branch as its own row is exactly
 * the packed block-causal form (kev's `rows_of` docstring). That is what makes
 * the engine's cross-turn prefix reuse the whole serving design: every branch
 * call is `state + branch_k`, the reuse start lands at the state length, only
 * the branch prefills, and the next branch overwrites its KV slots.
 *
 * The head is q/k Linear(d → 256) + bias, `logit = (K h_opt) · (Q h_decide) /
 * 16`, softmax within the question — kev/model.py PointerHead. On d=1024 that
 * is a few hundred multiply-adds per option; it stays on the CPU.
 *
 * Caller text is escaped `<|name|>` → `<¦name¦>` before tokenizing (kev's
 * `user_tokens`), so no state or option can forge a delimiter.
 */

import type { DecodeEngine } from './engine-core.js'
import type { Tokenizer } from './tokenizer.js'
import { parseSafetensorsHeader } from './mlx-weights.js'

/** Qwen special tokens kev reuses as delimiters (kev/model.py SPECIAL), BY
 *  NAME: Qwen3 numbers them 151659/151660/151648/151649/151661, Qwen3.5
 *  renumbered every special for its 248k vocab, so the ids come from the
 *  checkpoint's own tokenizer at load time (resolveKevDelimiters). */
export const KEV_DELIMITER_NAMES = {
  state: '<|fim_prefix|>',
  question: '<|fim_middle|>',
  optionStart: '<|box_start|>',
  optionEnd: '<|box_end|>',
  decide: '<|fim_suffix|>',
} as const

export type KevDelimiters = Record<keyof typeof KEV_DELIMITER_NAMES, number>

/** The five delimiter ids of THIS tokenizer. Each name must encode to exactly
 *  one id, which is what makes a delimiter a delimiter. */
export function resolveKevDelimiters(tokenizer: Tokenizer): KevDelimiters {
  const out = {} as KevDelimiters
  for (const [key, name] of Object.entries(KEV_DELIMITER_NAMES) as Array<[keyof KevDelimiters, string]>) {
    const ids = tokenizer.encode(name)
    if (ids.length !== 1) throw new Error(`kev: ${name} is not a single token of this tokenizer (${ids.length} ids)`)
    out[key] = ids[0]
  }
  return out
}

/** Training context (kev/model.py MAX_STATE / MAX_BRANCH). A state is cut to
 *  MAX_STATE-1 tokens; a branch must fit beside the state within MAX_BRANCH. */
export const KEV_MAX_STATE = 384
export const KEV_MAX_BRANCH = 1024

/** kev's internal record: what `encode` packs. `options` are already rendered
 *  strings ("name" or "name: description"). */
export interface KevRecord {
  state: string
  questions: Array<{ instr: string; options: string[] }>
}

export interface KevHead {
  /** [256, d] row-major f32 */
  qWeight: Float32Array
  qBias: Float32Array
  kWeight: Float32Array
  kBias: Float32Array
  dp: number
  d: number
  /** Calibration temperature fitted into head.pt (1 = raw). */
  temperature: number
}

// TypeSafe request shape (kev/api.py). Descriptions are strings here; kev
// also accepts nested JSON and renders it as "k: v" lines — not needed yet.
export type KevQuestion =
  | { type: 'choice'; instructions: string; criteria: Record<string, string | null> }
  | { type: 'score'; instructions: string; criteria: string[] }
  | { type: 'noul'; instructions: string; criteria?: { true?: string | null; false?: string | null } }

export type KevAnswer =
  | { type: 'choice'; choice: string; probabilities: Record<string, number>; confidence: number }
  | { type: 'score'; score: number; legend: Record<string, string>; probabilities: Record<string, number>; confidence: number }
  | { type: 'noul'; noul: number }

const SPECIAL_RE = /<\|([A-Za-z0-9_]+)\|>/g
export const escapeUserText = (text: string): string => text.replace(SPECIAL_RE, '<¦$1¦>')

/** kev/api.py option_text: "name" or "name: description". */
export const optionText = (name: string, desc: string | null | undefined): string =>
  desc == null || desc === '' ? name : `${name}: ${desc}`

/** kev/api.py to_record — the TypeSafe request as the record `encode` packs,
 *  plus the keys each question's probabilities come back under. */
export function toRecord(state: string, questions: KevQuestion[]): {
  record: KevRecord
  keys: string[][]
} {
  const qs: KevRecord['questions'] = []
  const keys: string[][] = []
  for (const q of questions) {
    if (q.type === 'noul') {
      const c = q.criteria ?? {}
      qs.push({ instr: q.instructions, options: [optionText('no', c.false), optionText('yes', c.true)] })
      keys.push(['false', 'true'])
    } else if (q.type === 'choice') {
      const names = Object.keys(q.criteria)
      if (names.length < 1) throw new Error('choice: criteria must have at least one option')
      qs.push({ instr: q.instructions, options: names.map((k) => optionText(k, q.criteria[k])) })
      keys.push(names)
    } else {
      if (q.criteria.length < 2) throw new Error('score: criteria must have at least two levels')
      qs.push({ instr: q.instructions, options: [...q.criteria] })
      keys.push(q.criteria.map((_, i) => String(i)))
    }
  }
  return { record: { state, questions: qs }, keys }
}

/** One branch row of a packed record, with the readout offsets WITHIN the
 *  row (state included). */
export interface KevRow {
  ids: number[]
  /** index of <decide> in `ids` */
  decide: number
  /** index of each option's </opt> in `ids`, in option order */
  opts: number[]
}

/**
 * kev/model.py encode + rows_of in one step: the state prefix and, per
 * question, the full causal row `state + branch` with its readout positions.
 */
export function encodeRecord(
  tokenizer: Tokenizer,
  rec: KevRecord,
  limits: { maxState?: number; maxBranch?: number; delimiters?: KevDelimiters } = {},
): { state: number[]; rows: KevRow[]; stateTruncated: boolean } {
  const maxState = limits.maxState ?? KEV_MAX_STATE
  const maxBranch = limits.maxBranch ?? KEV_MAX_BRANCH
  const D = limits.delimiters ?? resolveKevDelimiters(tokenizer)
  const enc = (t: string): number[] => tokenizer.encode(escapeUserText(t))

  const stateTokens = enc(rec.state)
  const state = [D.state, ...stateTokens.slice(0, maxState - 1)]
  const rows: KevRow[] = []
  for (const q of rec.questions) {
    const branch: number[] = [D.question, ...enc(q.instr)]
    const ends: number[] = []
    for (const o of q.options) {
      branch.push(D.optionStart, ...enc(o), D.optionEnd)
      ends.push(branch.length - 1)
    }
    branch.push(D.decide)
    if (branch.length > maxBranch - state.length) {
      throw new Error(`kev: branch too long: ${branch.length} tokens beside a ${state.length}-token state (limit ${maxBranch})`)
    }
    const base = state.length
    rows.push({ ids: [...state, ...branch], decide: base + branch.length - 1, opts: ends.map((e) => base + e) })
  }
  return { state, rows, stateTruncated: stateTokens.length + 1 > maxState }
}

/** PointerHead.forward: logits over the options of one question. */
export function pointerLogits(head: KevHead, hDecide: Float32Array, hOpts: Float32Array[]): number[] {
  const { d, dp } = head
  const proj = (W: Float32Array, b: Float32Array, h: Float32Array): Float32Array => {
    const out = new Float32Array(dp)
    for (let r = 0; r < dp; r++) {
      let acc = b[r]
      const row = r * d
      for (let c = 0; c < d; c++) acc += W[row + c] * h[c]
      out[r] = acc
    }
    return out
  }
  const q = proj(head.qWeight, head.qBias, hDecide)
  const scale = 1 / Math.sqrt(dp)
  return hOpts.map((h) => {
    const k = proj(head.kWeight, head.kBias, h)
    let z = 0
    for (let i = 0; i < dp; i++) z += k[i] * q[i]
    z *= scale
    return head.temperature === 1 ? z : z / head.temperature
  })
}

export function softmax(z: number[]): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** Load kev_head.safetensors (written by scripts/kev-merge.py) from the
 *  checkpoint directory. */
export async function loadKevHead(baseUrl: string): Promise<KevHead> {
  const res = await fetch(baseUrl + 'kev_head.safetensors')
  if (!res.ok) throw new Error(`kev_head.safetensors: HTTP ${res.status} at ${baseUrl}`)
  const buf = await res.arrayBuffer()
  const h = parseSafetensorsHeader(buf)
  const f32 = (name: string): Float32Array => {
    const t = h.tensors[name]
    if (!t) throw new Error(`kev_head.safetensors: missing ${name}`)
    if (t.dtype !== 'F32') throw new Error(`kev_head.safetensors: ${name} is ${t.dtype}, expected F32`)
    return new Float32Array(buf.slice(h.dataStart + t.begin, h.dataStart + t.end))
  }
  const [dp, d] = h.tensors['q.weight'].shape
  // parseSafetensorsHeader drops __metadata__; the temperature lives there.
  const headerLen = new DataView(buf).getUint32(0, true)
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, headerLen))).__metadata__ ?? {}
  const temperature = Number(meta.temperature ?? '1')
  return {
    qWeight: f32('q.weight'), qBias: f32('q.bias'),
    kWeight: f32('k.weight'), kBias: f32('k.bias'),
    dp, d, temperature: Number.isFinite(temperature) && temperature > 0 ? temperature : 1,
  }
}

export interface Kev {
  /** Probabilities per question for a record already in kev's internal form —
   *  the parity surface scripts/kev-parity.mjs compares against the torch
   *  reference. Also returns the rows so token ids can be diffed first. */
  decideRecord(rec: KevRecord): Promise<{ probs: number[][]; rows: KevRow[]; state: number[] }>
  /** The TypeSafe-shaped call. */
  decide(state: string, questions: KevQuestion[]): Promise<KevAnswer[]>
}

/** kev/api.py confidence formulas (typesafe-ai/system-one-adapter-python). */
function choiceConfidence(p: number[]): number {
  if (p.length === 1) return 1
  const u = 1 / p.length
  return Math.max(0, (Math.max(...p) - u) / (1 - u))
}
function scoreConfidence(p: number[]): number {
  if (p.length === 1) return 1
  let mode = 0
  for (let i = 1; i < p.length; i++) if (p[i] > p[mode]) mode = i
  const dist = p.reduce((s, pi, i) => s + pi * Math.abs(i - mode), 0)
  const c = (p.length - 1) / 2
  let umad = 0
  for (let i = 0; i < p.length; i++) umad += Math.abs(i - c)
  umad /= p.length
  return Math.max(0, 1 - dist / umad)
}

export function createKev(engine: DecodeEngine, tokenizer: Tokenizer, head: KevHead): Kev {
  if (head.d !== engine.spec.d) throw new Error(`kev head is d=${head.d}, engine is d=${engine.spec.d}`)
  const delimiters = resolveKevDelimiters(tokenizer)

  async function decideRecord(rec: KevRecord) {
    const { state, rows } = encodeRecord(tokenizer, rec, { delimiters })
    // Every branch packed into one pass over the resident state (the engine
    // falls back to branch-by-branch where it cannot pack). Rows carry
    // `state + branch`; the engine takes the branch and branch-local readout
    // offsets. <decide> last per row, so the head reads options [0, K) and
    // decide at K.
    const S = state.length
    const hidden = await engine.forwardHiddenPacked(state, rows.map((row) => ({
      ids: row.ids.slice(S), positions: [...row.opts, row.decide].map((p) => p - S),
    })))
    const probs = hidden.map((h) => softmax(pointerLogits(head, h[h.length - 1], h.slice(0, -1))))
    return { probs, rows, state }
  }

  async function decide(state: string, questions: KevQuestion[]): Promise<KevAnswer[]> {
    const { record, keys } = toRecord(state, questions)
    const { probs } = await decideRecord(record)
    return questions.map((q, i): KevAnswer => {
      const p = probs[i]
      const k = keys[i]
      if (q.type === 'noul') return { type: 'noul', noul: p[1] }
      const probabilities: Record<string, number> = {}
      k.forEach((key, j) => { probabilities[key] = p[j] })
      if (q.type === 'choice') {
        let best = 0
        for (let j = 1; j < p.length; j++) if (p[j] > p[best]) best = j
        return { type: 'choice', choice: k[best], probabilities, confidence: choiceConfidence(p) }
      }
      const legend: Record<string, string> = {}
      q.criteria.forEach((level, j) => { legend[String(j)] = level })
      const score = p.reduce((s, pi, j) => s + pi * j, 0)
      return { type: 'score', score, legend, probabilities, confidence: scoreConfidence(p) }
    })
  }

  return { decideRecord, decide }
}
