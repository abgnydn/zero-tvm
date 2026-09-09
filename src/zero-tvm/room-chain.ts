/**
 * ROOM CHAIN — assembling a split model out of peers, and the token loop that
 * walks the assembled stages.
 *
 * Lifted out of hostRoom's closure unchanged. It lived there, reachable only
 * through a WebSocket and three RTCDataChannels, which is why a live 3-device
 * session was the first thing to exercise it: the house rule is that a subject
 * has to be IMPORTABLE before it can be checked, and this was not.
 *
 * Nothing here touches WebRTC, the DOM or the GPU. The pipeline channels
 * arrive as `hasPipe`/`connect` and the peer table as `connected`, so the
 * assembly rules run in a plain unit test (tests/unit/room-chain.test.ts).
 *
 * The rules themselves are unchanged: refuse what can never fit; hold what
 * fits but is not adjacent YET; the accepted stages must tile [k, layers)
 * contiguously, built from peers arriving in any order. Hub-and-spoke on
 * purpose — see share.ts history.
 */

import type { ModelSpec } from '../compiler/model-spec.js'
import type { StageReply } from './pipeline-peer.js'

/** A peer that holds the REST of a split model, offering to run it. */
export interface StageOffer { type: 'stage-offer'; start: number; end: number; specId: string }

/** What the host says back to a peer that offered a stage. */
export type ChainMsg =
  | { type: 'stage-accept'; start: number; end: number }
  | { type: 'stage-reject'; message: string }
  | { type: 'stage-wait'; message: string }

/** One accepted stage: the layers it holds and the wire that reaches it. */
export interface DownStage {
  guest: string
  start: number
  end: number
  step: (pos: number, residual: ArrayBuffer) => Promise<StageReply>
  meanHopMs: () => number
}

export interface ChainDeps {
  spec: Pick<ModelSpec, 'id' | 'layers' | 'stops'>
  /** This host's engine holds only layers [start,end) of a split model. */
  stageRange: { start: number; end: number } | null
  engine: {
    pipelineStep: (
      input: { tokenId: number } | { residual: ArrayBuffer },
      position: number,
    ) => Promise<StageReply>
  }
  /** Is this peer's pipeline channel open? Checked before an offer is held. */
  hasPipe: (guest: string) => boolean
  /** Open a stage client over this peer's pipeline channel — null if it went. */
  connect: (guest: string) => Pick<DownStage, 'step' | 'meanHopMs'> | null
  /** Is this peer still in the host's peer table? */
  connected: (guest: string) => boolean
  send: (guest: string, msg: ChainMsg) => void
  row: (guest: string, text: string) => void
  announce: () => void
  onPaired?: (complete: boolean) => void
}

export interface RoomChain {
  /** The accepted stages, in layer order. */
  stages: DownStage[]
  /** The first layer still missing — where the next stage must start. */
  end: () => number
  complete: () => boolean
  /** What the room looks like right now, for the host's own eyes. */
  describe: () => string
  /** A peer says which layers it holds. */
  offer: (guest: string, offer: StageOffer) => void
  /** A stage left. */
  drop: (guest: string) => void
  /** Is this peer part of the chain — accepted or held? */
  holds: (guest: string) => boolean
  /** The token loop for a SPLIT model. */
  generate: (
    promptIds: number[], budget: number,
    onToken: (id: number) => void, shouldStop: () => boolean,
  ) => Promise<number[]>
}

export function makeRoomChain(deps: ChainDeps): RoomChain {
  const { spec, stageRange } = deps
  const chain: DownStage[] = []
  // Offers that are valid but not yet adjacent to the assembled chain.
  const pendingStages = new Map<string, StageOffer>()

  const chainEnd = (): number => (chain.length ? chain[chain.length - 1].end : stageRange!.end)
  const chainComplete = (): boolean => !!stageRange && chainEnd() === spec.layers

  function describeChain(): string {
    const held = `${stageRange!.start}-${stageRange!.end} here`
    const rest = chain.map((s) => `${s.start}-${s.end}`).join(' → ')
    const missing = chainComplete() ? '' : ` · waiting for layers ${chainEnd()}-${spec.layers}`
    return `split model — layers ${held}${rest ? ` → ${rest}` : ''}${missing}`
  }

  /**
   * A peer says which layers it holds. Refuse what can never fit; hold what
   * fits but is not adjacent YET.
   */
  function offerStage(guest: string, offer: StageOffer): void {
    const bad = !stageRange ? 'this host runs the whole model'
      : offer.specId !== spec.id ? `different model (${offer.specId} vs ${spec.id})`
      : !deps.hasPipe(guest) ? 'the pipeline channel is not open'
      : !(offer.start < offer.end) ? `layers ${offer.start}-${offer.end} is not a range`
      : offer.end > spec.layers ? `layers ${offer.start}-${offer.end} runs past this model's ${spec.layers}`
      : offer.start < stageRange.end ? `layers ${offer.start}-${offer.end} overlaps the ${stageRange.start}-${stageRange.end} this host holds`
      : chain.some((c) => offer.start < c.end && c.start < offer.end)
        ? `layers ${offer.start}-${offer.end} overlap a stage already in the chain`
        : null
    if (bad) {
      deps.send(guest, { type: 'stage-reject', message: bad })
      deps.row(guest, `stage offer refused — ${bad}`)
      return
    }
    pendingStages.set(guest, offer)
    extendChain()
    if (pendingStages.has(guest)) {
      const why = `layers ${offer.start}-${offer.end} held — the chain needs ${chainEnd()} next`
      deps.send(guest, { type: 'stage-wait', message: why })
      deps.row(guest, why)
    }
  }

  /** Attach every pending offer that now continues the chain, repeatedly: one
   *  arrival can unblock several that came in out of order. */
  function extendChain(): void {
    for (;;) {
      const want = chainEnd()
      const hit = [...pendingStages].find(([, o]) => o.start === want)
      if (!hit) break
      const [guest, offer] = hit
      const client = deps.connect(guest)
      if (!client) { pendingStages.delete(guest); continue }
      pendingStages.delete(guest)
      chain.push({ guest, start: offer.start, end: offer.end, step: client.step, meanHopMs: client.meanHopMs })
      deps.send(guest, { type: 'stage-accept', start: offer.start, end: offer.end })
      deps.row(guest, `serving layers ${offer.start}-${offer.end}`)
    }
    deps.announce()
    if (chainComplete()) {
      deps.row('room', `the model is complete across ${chain.length + 1} stages`)
      deps.onPaired?.(true)
    }
  }

  /** A stage left. Everything downstream of it is orphaned: those peers stay
   *  connected, go back to pending, and re-attach if a replacement arrives. */
  function dropStage(guest: string): void {
    const i = chain.findIndex((c) => c.guest === guest)
    if (i < 0) { pendingStages.delete(guest); return }
    const orphaned = chain.splice(i)
    for (const o of orphaned.slice(1)) {
      if (deps.connected(o.guest)) pendingStages.set(o.guest, { type: 'stage-offer', start: o.start, end: o.end, specId: spec.id })
    }
    extendChain()
    deps.onPaired?.(chainComplete())
  }

  /**
   * The token loop for a SPLIT model — the whole-model generatePipelined
   * cannot run here, because this engine holds only the first layers.
   */
  async function generateSplit(
    promptIds: number[], budget: number,
    onToken: (id: number) => void, shouldStop: () => boolean,
  ): Promise<number[]> {
    if (!chainComplete()) throw new Error(`layers ${chainEnd()}-${spec.layers} of this model are not connected`)
    const stages = [...chain]
    const stepAll = async (tokenId: number, pos: number): Promise<number> => {
      let out: StageReply = await deps.engine.pipelineStep({ tokenId }, pos)
      for (const s of stages) {
        if ('tokenId' in out) throw new Error('a stage ended the model before the last one')
        out = await s.step(pos, out.residual)
      }
      if (!('tokenId' in out)) throw new Error('the last stage returned a residual, not a token')
      return out.tokenId
    }
    let tok = 0
    for (let i = 0; i < promptIds.length; i++) tok = await stepAll(promptIds[i], i)
    const out: number[] = []
    for (let n = 0; n < budget; n++) {
      // Stop ids are consumed, never shown — same contract as generate().
      if (spec.stops.includes(tok) || shouldStop()) break
      out.push(tok)
      onToken(tok)
      tok = await stepAll(tok, promptIds.length + n)
    }
    return out
  }

  return {
    stages: chain,
    end: chainEnd,
    complete: chainComplete,
    describe: describeChain,
    offer: offerStage,
    drop: dropStage,
    holds: (guest) => chain.some((c) => c.guest === guest) || pendingStages.has(guest),
    generate: generateSplit,
  }
}
