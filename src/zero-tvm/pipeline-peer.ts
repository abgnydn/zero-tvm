/**
 * PIPELINE PEER — the wire between consecutive stages of one model.
 *
 * Stage 0 holds layers [0, k) and drives the token loop; every later stage
 * answers with whatever its own layers produced. A stage that does NOT end the
 * model hands back a residual for the next one; the stage holding the final
 * layers hands back the token:
 *
 *   stage 0 → [u32 reqId][u32 position][residual bytes]   (d f16: 4-5 KB)
 *   middle  → [u32 reqId][u32 position][residual bytes]   (same frame back)
 *   last    → {"type":"stage-token","req":n,"tokenId":t}  (~40 B)
 *
 * so the reply's TYPE says where in the chain the sender sits — binary means
 * "not done yet", text means "here is the token". Nothing has to be told how
 * long the chain is, which is what lets stages join in any order.
 *
 * The first hop's asymmetry is the whole latency story. Bandwidth is trivial — 5 KB per
 * token is nothing for any link — but the round trip is serial with decode: a
 * token cannot start until the previous one came back. On a LAN (2-5 ms) that
 * is a modest tax on a ~15 ms token; over the open internet it dominates, and
 * this is the honest reason the feature is framed for machines you own on one
 * network rather than volunteers across the world.
 *
 * Requests carry an id and replies echo it, so a late or duplicated frame
 * resolves the right waiter instead of the current one — the failure that
 * would otherwise show up as a single wrong token deep into a reply, with
 * nothing pointing back here.
 *
 * No WebGPU import at module scope: the helper page pulls the engine in
 * dynamically, exactly like share.ts's host path.
 */

import type { DecodeEngine } from './engine-core.js'

const HEADER = 8   // u32 reqId, u32 position

export function frameStep(reqId: number, position: number, residual: ArrayBuffer): ArrayBuffer {
  const out = new Uint8Array(HEADER + residual.byteLength)
  const dv = new DataView(out.buffer)
  dv.setUint32(0, reqId, true)
  dv.setUint32(4, position, true)
  out.set(new Uint8Array(residual), HEADER)
  return out.buffer
}

export function unframeStep(buf: ArrayBuffer): { reqId: number; position: number; residual: ArrayBuffer } {
  const dv = new DataView(buf)
  return {
    reqId: dv.getUint32(0, true),
    position: dv.getUint32(4, true),
    residual: buf.slice(HEADER),
  }
}

/**
 * HELPER side: answer residual frames, forever.
 *
 * Frames are handled strictly in order (the engine has one KV cache and one
 * GDN state; two overlapping steps would interleave writes to both), so the
 * handler chains onto a queue rather than running per message.
 */
export function serveStage(
  dc: RTCDataChannel,
  engine: DecodeEngine,
  onEvent?: (msg: string) => void,
): void {
  dc.binaryType = 'arraybuffer'
  let queue: Promise<void> = Promise.resolve()
  let served = 0
  dc.addEventListener('message', (e) => {
    if (typeof e.data === 'string') return
    queue = queue.then(async () => {
      const { reqId, position, residual } = unframeStep(e.data as ArrayBuffer)
      try {
        const out = await engine.pipelineStep({ residual }, position)
        // A middle stage returns its residual in the same frame it received;
        // only the stage holding the last layer has a token to return.
        if ('tokenId' in out) dc.send(JSON.stringify({ type: 'stage-token', req: reqId, tokenId: out.tokenId }))
        else dc.send(frameStep(reqId, position, out.residual))
        if (++served % 32 === 1) onEvent?.(`served ${served} token${served === 1 ? '' : 's'}`)
      } catch (err) {
        dc.send(JSON.stringify({ type: 'stage-error', req: reqId, message: err instanceof Error ? err.message : String(err) }))
      }
    })
  })
}

/** What a stage hands back: the next stage's input, or the answer. */
export type StageReply = { tokenId: number } | { residual: ArrayBuffer }

/**
 * UPSTREAM side: hand a residual over and wait for what it becomes — the same
 * shape engine.pipelineStep returns, so a caller can walk a chain of remote
 * stages and local ones with one loop.
 *
 * Returns a function, not a class: the caller's token loop is the only thing
 * that needs to exist above this.
 */
export function makeStageClient(dc: RTCDataChannel): {
  step: (position: number, residual: ArrayBuffer) => Promise<StageReply>
  /** Mean round-trip over the steps so far, in ms — the split's real cost. */
  meanHopMs: () => number
} {
  dc.binaryType = 'arraybuffer'
  const waiting = new Map<number, { resolve: (r: StageReply) => void; reject: (e: Error) => void; t0: number }>()
  let nextId = 1
  let hops = 0
  let hopMs = 0

  const settle = (reqId: number, fn: (w: { resolve: (r: StageReply) => void; reject: (e: Error) => void; t0: number }) => void): void => {
    const w = waiting.get(reqId)
    if (!w) return                      // late duplicate of an already-settled step
    waiting.delete(reqId)
    fn(w)
  }
  dc.addEventListener('message', (e) => {
    // Binary reply = a middle stage handing its residual on. It carries the
    // request id in the same header the request used, so out-of-order or
    // duplicated frames still settle the waiter they belong to.
    if (typeof e.data !== 'string') {
      const { reqId, residual } = unframeStep(e.data as ArrayBuffer)
      settle(reqId, (w) => { hops++; hopMs += performance.now() - w.t0; w.resolve({ residual }) })
      return
    }
    const msg = JSON.parse(e.data) as { type: string; req: number; tokenId?: number; message?: string }
    settle(msg.req, (w) => {
      if (msg.type === 'stage-token' && typeof msg.tokenId === 'number') {
        hops++
        hopMs += performance.now() - w.t0
        w.resolve({ tokenId: msg.tokenId })
      } else {
        w.reject(new Error(msg.message ?? 'the other stage failed'))
      }
    })
  })
  dc.addEventListener('close', () => {
    for (const w of waiting.values()) w.reject(new Error('a stage of the model disconnected'))
    waiting.clear()
  })

  return {
    step(position, residual) {
      if (dc.readyState !== 'open') return Promise.reject(new Error('a stage of the model is not connected'))
      const reqId = nextId++
      const p = new Promise<StageReply>((resolve, reject) => {
        waiting.set(reqId, { resolve, reject, t0: performance.now() })
      })
      dc.send(frameStep(reqId, position, residual))
      return p
    },
    meanHopMs: () => (hops ? hopMs / hops : 0),
  }
}
