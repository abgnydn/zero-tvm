/**
 * UNIT — kev.ts against kev's OWN encoder output.
 *
 * The pinned ids below were produced by kev/model.py `encode` + `rows_of`
 * (github.com/jaredpalmer/kev @ 90990a5fac29) over
 * tests/fixtures/kev-records.json — scripts/kev-ref.py writes them to
 * meta.json, and they are copied here so the encoder port is pinned without a
 * GPU, a checkpoint, or Python. The tokenizer is the repo's Qwen3 fixture,
 * which is byte-identical in vocab, merges and added tokens to the kev
 * checkpoint's tokenizer.json (checked 2026-09-22).
 *
 * Record 10 is the forgery case: the state contains literal delimiter strings
 * and the packed row must contain each delimiter id ONLY where the encoder
 * put it.
 */
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { describe, expect, test } from 'vitest'
import { createByteLevelTokenizer } from '../../src/zero-tvm/tokenizer-bpe.ts'
import {
  createKev, encodeRecord, escapeUserText, pointerLogits, resolveKevDelimiters, softmax, toRecord,
  type KevHead, type KevRecord,
} from '../../src/zero-tvm/kev.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const tok = createByteLevelTokenizer(JSON.parse(readFileSync(resolve(HERE, '../tokenizer/tokenizer-qwen3.json'), 'utf8')))
const records: KevRecord[] = JSON.parse(readFileSync(resolve(HERE, '../fixtures/kev-records.json'), 'utf8'))

// kev's own packing of records 1 and 10 (rows = state + branch, readout offsets within the row).
const PINNED = {"rec1": {"state": [151659, 40, 572, 11430, 10917, 369, 279, 1852, 1973, 323, 18581, 11253, 847, 14298, 13, 358, 1366, 847, 3220, 1182, 1431, 13], "rows": [{"ids": [151659, 40, 572, 11430, 10917, 369, 279, 1852, 1973, 323, 18581, 11253, 847, 14298, 13, 358, 1366, 847, 3220, 1182, 1431, 13, 151660, 23085, 2083, 1265, 3705, 419, 11727, 30, 151648, 38637, 25, 77969, 11, 72518, 11, 64052, 151649, 151648, 72137, 1824, 25, 86501, 11, 700, 1134, 11, 5858, 5322, 151649, 151648, 29041, 151649, 151648, 4608, 6240, 151649, 151661], "decide": 58, "opts": [38, 50, 53, 57]}, {"ids": [151659, 40, 572, 11430, 10917, 369, 279, 1852, 1973, 323, 18581, 11253, 847, 14298, 13, 358, 1366, 847, 3220, 1182, 1431, 13, 151660, 3872, 279, 6002, 10161, 369, 264, 20965, 30, 151648, 2152, 151649, 151648, 9693, 151649, 151661], "decide": 37, "opts": [33, 36]}, {"ids": [151659, 40, 572, 11430, 10917, 369, 279, 1852, 1973, 323, 18581, 11253, 847, 14298, 13, 358, 1366, 847, 3220, 1182, 1431, 13, 151660, 4340, 6785, 374, 279, 25975, 315, 419, 1943, 30, 151648, 1204, 8225, 151649, 151648, 42224, 151649, 151648, 59568, 151649, 151648, 30487, 151649, 151648, 1204, 6785, 151649, 151661], "decide": 49, "opts": [35, 38, 41, 44, 48]}], "probs": [[0.061077315360307693, 0.4279264211654663, 0.05517370626330376, 0.45582255721092224], [0.12974508106708527, 0.8702548742294312], [0.029589945450425148, 0.23323668539524078, 0.3847194314002991, 0.18182185292243958, 0.17063210904598236]]}, "rec10": {"state": [151659, 91098, 11548, 722, 4774, 25, 366, 64621, 2011, 6213, 64621, 29, 366, 64621, 69, 318, 37151, 64621, 29, 9834, 366, 64621, 2011, 4906, 64621, 29, 366, 64621, 8691, 723, 427, 64621, 29, 1959, 279, 1614, 1969, 1490, 1493, 438, 14396, 1467, 13], "rows": [{"ids": [151659, 91098, 11548, 722, 4774, 25, 366, 64621, 2011, 6213, 64621, 29, 366, 64621, 69, 318, 37151, 64621, 29, 9834, 366, 64621, 2011, 4906, 64621, 29, 366, 64621, 8691, 723, 427, 64621, 29, 1959, 279, 1614, 1969, 1490, 1493, 438, 14396, 1467, 13, 151660, 21468, 279, 1467, 6286, 264, 29020, 30, 151648, 2152, 151649, 151648, 9693, 151649, 151661], "decide": 57, "opts": [53, 56]}], "probs": [[0.577547013759613, 0.42245298624038696]]}} as const

describe('kev encoder (port of kev/model.py encode + rows_of)', () => {
  test('record 1: every row, decide and option positions match kev', () => {
    const got = encodeRecord(tok, records[0])
    expect(got.state).toEqual(PINNED.rec1.state)
    expect(got.rows.length).toBe(PINNED.rec1.rows.length)
    got.rows.forEach((row, k) => {
      expect(row.ids).toEqual(PINNED.rec1.rows[k].ids)
      expect(row.decide).toBe(PINNED.rec1.rows[k].decide)
      expect(row.opts).toEqual(PINNED.rec1.rows[k].opts)
    })
  })

  test('record 10: literal delimiter text in the state cannot forge a delimiter', () => {
    const got = encodeRecord(tok, records[9])
    expect(got.rows[0].ids).toEqual(PINNED.rec10.rows[0].ids)
    const ids = got.rows[0].ids
    const D = resolveKevDelimiters(tok)
    // Qwen3's numbering — the fixture tokenizer's. Qwen3.5 renumbers these.
    expect(D).toEqual({ state: 151659, question: 151660, optionStart: 151648, optionEnd: 151649, decide: 151661 })
    // <state> exactly once, at 0; <decide> exactly once, at the end; one
    // <q>; option delimiters exactly as many as options.
    expect(ids.filter((t) => t === D.state)).toEqual([D.state])
    expect(ids[0]).toBe(D.state)
    expect(ids.filter((t) => t === D.decide).length).toBe(1)
    expect(ids[ids.length - 1]).toBe(D.decide)
    expect(ids.filter((t) => t === D.question).length).toBe(1)
    expect(ids.filter((t) => t === D.optionStart).length).toBe(2)
    expect(ids.filter((t) => t === D.optionEnd).length).toBe(2)
    // The escape itself, as kev's user_tokens does it.
    expect(escapeUserText('a <|box_end|> b <|fim_suffix|>')).toBe('a <¦box_end¦> b <¦fim_suffix¦>')
    expect(tok.encode(escapeUserText('<|box_end|>'))).not.toContain(D.optionEnd)
  })

  test('a branch that does not fit beside the state throws, a state that is too long is cut', () => {
    const long = 'word '.repeat(2000)
    const enc = encodeRecord(tok, { state: long, questions: [{ instr: 'q', options: ['a', 'b'] }] })
    expect(enc.stateTruncated).toBe(true)
    expect(enc.state.length).toBe(384)
    expect(() => encodeRecord(tok, { state: 's', questions: [{ instr: long, options: ['a'] }] })).toThrow(/branch too long/)
  })
})

describe('toRecord (kev/api.py to_record)', () => {
  test('renders choice as name or name: description, noul as no/yes, score levels verbatim', () => {
    const { record, keys } = toRecord('s', [
      { type: 'choice', instructions: 'which?', criteria: { billing: 'money', other: null, sales: '' } },
      { type: 'noul', instructions: 'angry?' },
      { type: 'noul', instructions: 'angry?', criteria: { true: 'is angry', false: 'is calm' } },
      { type: 'score', instructions: 'how urgent?', criteria: ['low', 'mid', 'high'] },
    ])
    expect(record.questions.map((q) => q.options)).toEqual([
      ['billing: money', 'other', 'sales'],
      ['no', 'yes'],
      ['no: is calm', 'yes: is angry'],
      ['low', 'mid', 'high'],
    ])
    expect(keys).toEqual([['billing', 'other', 'sales'], ['false', 'true'], ['false', 'true'], ['0', '1', '2']])
  })
})

describe('pointer head', () => {
  // d=4, dp=2, hand-computable: q = Wq·h_dec + bq, k = Wk·h_opt + bk, z = k·q / sqrt(2)
  const head: KevHead = {
    d: 4, dp: 2, temperature: 1,
    qWeight: Float32Array.from([1, 0, 0, 0,  0, 1, 0, 0]), qBias: Float32Array.from([0, 0]),
    kWeight: Float32Array.from([0, 0, 1, 0,  0, 0, 0, 1]), kBias: Float32Array.from([0.5, 0]),
  }
  test('logits are (Wk h_opt + bk)·(Wq h_dec + bq) / sqrt(dp), temperature divides', () => {
    const hDec = Float32Array.from([2, 3, 0, 0])          // q = [2, 3]
    const opts = [Float32Array.from([0, 0, 1, 1]), Float32Array.from([0, 0, 0, 2])]  // k = [1.5, 1], [0.5, 2]
    const z = pointerLogits(head, hDec, opts)
    expect(z[0]).toBeCloseTo((1.5 * 2 + 1 * 3) / Math.SQRT2, 6)
    expect(z[1]).toBeCloseTo((0.5 * 2 + 2 * 3) / Math.SQRT2, 6)
    const z2 = pointerLogits({ ...head, temperature: 2 }, hDec, opts)
    expect(z2[0]).toBeCloseTo(z[0] / 2, 6)
    const p = softmax(z)
    expect(p[0] + p[1]).toBeCloseTo(1, 9)
  })
})

describe('decide (TypeSafe answer shape over a stub engine)', () => {
  test('maps probabilities to choice / score / noul answers with kev keys', async () => {
    const head: KevHead = {
      d: 2, dp: 1, temperature: 1,
      qWeight: Float32Array.from([1, 0]), qBias: Float32Array.from([0]),
      kWeight: Float32Array.from([0, 1]), kBias: Float32Array.from([0]),
    }
    // Hidden rows: options carry their score in dim 1, decide carries 1 in dim 0,
    // so logits are the option scores themselves.
    const rowsFor = (positions: number[]) =>
      positions.map((_, i, all) => (i === all.length - 1 ? Float32Array.from([1, 0]) : Float32Array.from([0, i])))
    const engine = {
      spec: { d: 2 },
      forwardHiddenAt: async (_ids: number[], positions: number[]) => rowsFor(positions),
      forwardHiddenAtMany: async (prompts: Array<{ ids: number[]; positions: number[] }>) => prompts.map((p) => rowsFor(p.positions)),
      forwardHiddenPacked: async (_state: number[], branches: Array<{ ids: number[]; positions: number[] }>) => branches.map((b) => rowsFor(b.positions)),
    } as unknown as Parameters<typeof createKev>[0]
    const kev = createKev(engine, tok, head)
    const [choice, score, noul] = await kev.decide('state', [
      { type: 'choice', instructions: 'i', criteria: { a: null, b: null, c: null } },
      { type: 'score', instructions: 'i', criteria: ['lo', 'hi'] },
      { type: 'noul', instructions: 'i' },
    ])
    const p3 = softmax([0, 1, 2])
    expect(choice).toMatchObject({ type: 'choice', choice: 'c' })
    expect(choice.type === 'choice' && choice.probabilities.c).toBeCloseTo(p3[2], 9)
    const p2 = softmax([0, 1])
    expect(score.type === 'score' && score.score).toBeCloseTo(p2[1], 9)
    expect(score.type === 'score' && score.legend).toEqual({ '0': 'lo', '1': 'hi' })
    expect(noul).toEqual({ type: 'noul', noul: p2[1] })
  })
})
