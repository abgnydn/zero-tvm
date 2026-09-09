// Chain assembly for a split model — the four ways a live 3-device session
// broke it this week.
//
// This logic ran for weeks reachable only through a WebSocket and three
// RTCDataChannels, so the first thing to exercise it was three real machines
// in one room. Every failure below is one of those, reproduced without a
// browser, a GPU or a network: room-chain.ts takes the wires as callbacks, so
// the rules can be driven directly.
//
// The subject is the assembler itself, not a second copy of its arithmetic:
// each test drives `offer()`/`generate()` and reads what the peer was TOLD
// (stage-accept / stage-reject / stage-wait) plus the assembled order.

import { describe, expect, it } from 'vitest'
import { makeRoomChain, type ChainMsg } from '../../src/zero-tvm/room-chain.ts'
import type { StageReply } from '../../src/zero-tvm/pipeline-peer.ts'

const SPEC_ID = 'llama32-1b'
const LAYERS = 64

/** A residual frame. Nothing reads its bytes here — only its TYPE matters,
 *  which is what tells the loop whether the model ended (see pipeline-peer). */
const residual = (): StageReply => ({ residual: new ArrayBuffer(8) })

type StepFn = (pos: number, r: ArrayBuffer) => Promise<StageReply>

/** The chain with its wires faked: `join` is a peer whose pipeline channel is
 *  open, which is all offerStage/extendChain ever ask about a connection. */
function harness(held: { start: number; end: number } | null, layers = LAYERS) {
  const sent: Array<[string, ChainMsg]> = []
  const rows: Array<[string, string]> = []
  const paired: boolean[] = []
  const pipes = new Set<string>()
  const peers = new Set<string>()
  const steps = new Map<string, StepFn>()
  let engineCalls = 0

  const chain = makeRoomChain({
    spec: { id: SPEC_ID, layers, stops: [999] },
    stageRange: held,
    engine: { pipelineStep: async () => { engineCalls++; return residual() } },
    hasPipe: (g) => pipes.has(g),
    connect: (g) => (pipes.has(g)
      ? { step: (pos, r) => (steps.get(g) ?? (async () => residual()))(pos, r), meanHopMs: () => 0 }
      : null),
    connected: (g) => peers.has(g),
    send: (g, m) => { sent.push([g, m]) },
    row: (w, t) => { rows.push([w, t]) },
    announce: () => {},
    onPaired: (c) => { paired.push(c) },
  })

  return {
    chain, sent, rows, paired,
    join(guest: string, step?: StepFn) {
      pipes.add(guest)
      peers.add(guest)
      if (step) steps.set(guest, step)
    },
    /** Everything this host said back to one peer. */
    to: (guest: string): ChainMsg[] => sent.filter(([g]) => g === guest).map(([, m]) => m),
    /** The assembled chain as plain layer ranges, in order. */
    ranges: (): Array<[number, number]> => chain.stages.map((s) => [s.start, s.end]),
    engineCalls: () => engineCalls,
  }
}

const offer = (start: number, end: number, specId = SPEC_ID) =>
  ({ type: 'stage-offer', start, end, specId }) as const

describe('offers that arrive out of order', () => {
  // The failure: the third machine's offer reached the host BEFORE the second
  // machine's, so the chain had a hole at the head. extendChain loops for
  // exactly this — one arrival can unblock several that came in behind it.
  it('holds an offer that is not adjacent yet, then attaches both when the gap fills', () => {
    const h = harness({ start: 0, end: 32 })
    h.join('late')
    h.join('early')

    // Arrives FIRST but continues layer 33, and the chain only reaches 32.
    h.chain.offer('late', offer(33, 64))
    expect(h.to('late')).toEqual([
      { type: 'stage-wait', message: 'layers 33-64 held — the chain needs 32 next' },
    ])
    expect(h.ranges()).toEqual([])
    expect(h.chain.complete()).toBe(false)

    // The one that closes the gap. BOTH must attach, in layer order.
    h.chain.offer('early', offer(32, 33))
    expect(h.ranges()).toEqual([[32, 33], [33, 64]])
    expect(h.to('early')).toEqual([{ type: 'stage-accept', start: 32, end: 33 }])
    expect(h.to('late')).toEqual([
      { type: 'stage-wait', message: 'layers 33-64 held — the chain needs 32 next' },
      { type: 'stage-accept', start: 33, end: 64 },
    ])
    expect(h.chain.complete()).toBe(true)
    expect(h.paired).toEqual([true])
  })
})

describe('generate on a chain that does not reach the last layer', () => {
  // The failure: with one helper still missing, the host answered from the
  // layers it had. A partial network produces fluent text, not an error, so
  // nothing downstream can notice.
  it('refuses, naming the layers nobody is holding, and never touches the engine', async () => {
    const h = harness({ start: 0, end: 32 })
    h.join('a')
    h.chain.offer('a', offer(32, 48))
    expect(h.ranges()).toEqual([[32, 48]])
    expect(h.chain.complete()).toBe(false)

    await expect(h.chain.generate([1, 2, 3], 4, () => {}, () => false))
      .rejects.toThrow('layers 48-64 of this model are not connected')
    expect(h.engineCalls()).toBe(0)

    // …and it is the INCOMPLETENESS that refuses, not generate() itself:
    // close the chain with the same harness and the loop runs.
    let next = 500
    h.join('b', async () => ({ tokenId: next++ }))
    h.chain.offer('b', offer(48, 64))
    expect(h.chain.complete()).toBe(true)
    const seen: number[] = []
    const out = await h.chain.generate([1, 2, 3], 4, (t) => seen.push(t), () => false)
    expect(out).toEqual([502, 503, 504, 505])
    expect(seen).toEqual(out)
    expect(h.engineCalls()).toBe(7)   // 3 prompt tokens + 4 generated
  })
})

describe('overlaps and gaps', () => {
  // The failure: a helper opened with the host's own layer range (a copied
  // URL), and the host accepted it — the same layers ran twice and the ones
  // after them never ran at all.
  it('refuses an offer that overlaps the range this host holds', () => {
    const h = harness({ start: 0, end: 32 })
    h.join('dup')
    h.chain.offer('dup', offer(16, 40))
    expect(h.to('dup')).toEqual([
      { type: 'stage-reject', message: 'layers 16-40 overlaps the 0-32 this host holds' },
    ])
    expect(h.ranges()).toEqual([])
  })

  it('refuses an offer that overlaps a stage already in the chain', () => {
    const h = harness({ start: 0, end: 32 })
    h.join('a')
    h.join('b')
    h.chain.offer('a', offer(32, 48))
    h.chain.offer('b', offer(40, 56))
    expect(h.to('b')).toEqual([
      { type: 'stage-reject', message: 'layers 40-56 overlap a stage already in the chain' },
    ])
    expect(h.ranges()).toEqual([[32, 48]])
  })

  // A gap is not an error — the missing peer may still be booting. It must
  // simply never read as complete, and the host must say what it is waiting on.
  it('never completes across a gap, and says which layers are missing', () => {
    const h = harness({ start: 0, end: 32 })
    h.join('far')
    h.chain.offer('far', offer(33, 64))
    expect(h.chain.complete()).toBe(false)
    expect(h.chain.end()).toBe(32)
    expect(h.paired).toEqual([])
    expect(h.chain.describe()).toBe('split model — layers 0-32 here · waiting for layers 32-64')
  })
})

describe('an offer past the model', () => {
  // The failure: a helper launched from a link with the wrong layer count
  // offered layers this model does not have.
  it('refuses a range running past the last layer', () => {
    const h = harness({ start: 0, end: 32 })
    h.join('over')
    h.chain.offer('over', offer(32, 72))
    expect(h.to('over')).toEqual([
      { type: 'stage-reject', message: "layers 32-72 runs past this model's 64" },
    ])
    expect(h.ranges()).toEqual([])
    expect(h.chain.complete()).toBe(false)
  })
})
