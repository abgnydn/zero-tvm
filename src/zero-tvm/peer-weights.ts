/**
 * PEER WEIGHTS — copy a model's weight cache from another browser instead of
 * downloading it from HuggingFace again.
 *
 * The unit of replication is the model's OPFS DIRECTORY, not the checkpoint.
 * That one choice is why this module is small and format-agnostic: whatever
 * the loader caches — MLC shard binaries keyed by flattened data path, MLX
 * BUILT buffers keyed by plan key (already concatenated and bf16-converted) —
 * lands as flat files in one per-model directory. Replicate the directory and
 * the receiving tab boots from cache with the loader untouched, never learning
 * the bytes came from a peer.
 *
 * Wire format, one message per piece:
 *
 *   [u32 headerLen LE][utf8 JSON header][payload bytes]
 *   header = { name, off, total, sha }   sha = SHA-256 of THIS piece, hex
 *
 * Pieces are 240 KB so a framed message stays under the ~256 KB every WebRTC
 * implementation accepts without negotiating maxMessageSize. Header and
 * payload ride in ONE message because the channel is ordered but interleaved
 * with control JSON — pairing two messages by position would be a race.
 *
 * WHAT THE HASH IS FOR: it catches transport corruption and a peer that
 * silently truncates. It does NOT make a dishonest host safe — the sender
 * computes it. That is deliberate for v1: this runs inside a share room,
 * where the guest already trusts the host to run the model honestly (the host
 * could return any tokens it liked). An open swarm with strangers needs a
 * manifest from an independent source, which is a different design.
 *
 * Backpressure is not optional. A 2.3 GB directory pushed without watching
 * bufferedAmount queues the whole thing in the sender's memory and the tab
 * dies; the sender waits on `bufferedamountlow` whenever it gets ahead.
 *
 * NO WebGPU ANYWHERE IN THIS MODULE. The receiving device may have none — it
 * might be pulling weights for a browser that will only later prove capable —
 * so this imports the light registry, never weight-loader.ts (which reads
 * GPUBufferUsage at module scope).
 */

import { opfsDirFor } from './model-registry.js'
import type { ModelSpec } from '../compiler/model-spec.js'

/** Payload bytes per framed message; header adds ~120 B. */
const PIECE = 240 * 1024
/** Bytes read from OPFS per read — sliced into PIECE-sized sends. */
const CHUNK = 8 * 1024 * 1024
/** Sender pauses above this many bytes queued, resumes at LOW. */
const BUFFER_HIGH = 8 * 1024 * 1024
const BUFFER_LOW = 1 * 1024 * 1024

export interface Inventory {
  files: number
  bytes: number
  /** File names already present, so a resumed pull skips them. */
  names: string[]
}

export interface PullProgress {
  filesDone: number
  filesTotal: number
  bytesDone: number
  bytesTotal: number
  /** Bytes/second over the transfer so far. */
  rate: number
}

// ============================================================
// shared helpers
// ============================================================

async function modelDir(spec: ModelSpec, create: boolean): Promise<FileSystemDirectoryHandle | null> {
  if (!navigator.storage?.getDirectory) return null
  try {
    const root = await navigator.storage.getDirectory()
    return await root.getDirectoryHandle(opfsDirFor(spec), { create })
  } catch {
    return null
  }
}

/** Every file in the model's OPFS dir, with sizes (no bytes read). */
async function listFiles(dir: FileSystemDirectoryHandle): Promise<{ name: string; size: number }[]> {
  const out: { name: string; size: number }[] = []
  for await (const [name, handle] of (dir as unknown as AsyncIterable<[string, FileSystemHandle]>)) {
    if (handle.kind !== 'file') continue
    const file = await (handle as FileSystemFileHandle).getFile()
    out.push({ name, size: file.size })
  }
  // Smallest first: the manifest and the little norm buffers land early, so an
  // interrupted pull leaves a cache that resumes cleanly rather than one giant
  // half-written blob.
  out.sort((a, b) => a.size - b.size)
  return out
}

const hex = (buf: ArrayBuffer): string =>
  [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('')

async function sha256(data: Uint8Array): Promise<string> {
  const copy = data.slice()   // detached-buffer safety: digest wants a stable view
  return hex(await crypto.subtle.digest('SHA-256', copy))
}

function frame(header: object, payload: Uint8Array): ArrayBuffer {
  const head = new TextEncoder().encode(JSON.stringify(header))
  const out = new Uint8Array(4 + head.byteLength + payload.byteLength)
  new DataView(out.buffer).setUint32(0, head.byteLength, true)
  out.set(head, 4)
  out.set(payload, 4 + head.byteLength)
  return out.buffer
}

function unframe(buf: ArrayBuffer): { header: PieceHeader; payload: Uint8Array } {
  const headLen = new DataView(buf).getUint32(0, true)
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headLen))) as PieceHeader
  return { header, payload: new Uint8Array(buf, 4 + headLen) }
}

interface PieceHeader { name: string; off: number; total: number; sha: string }

// ============================================================
// HOST — serve the local cache
// ============================================================

/**
 * Answer inventory and pull requests on `dc` for as long as it is open.
 *
 * Control messages are JSON strings; pieces are ArrayBuffers, so the two never
 * need distinguishing by content.
 */
export function serveWeights(
  dc: RTCDataChannel,
  spec: ModelSpec,
  onEvent?: (msg: string) => void,
): void {
  dc.bufferedAmountLowThreshold = BUFFER_LOW
  dc.binaryType = 'arraybuffer'

  /**
   * Wait until the send queue is short enough to push more.
   *
   * Re-CHECKS in a loop with a timeout rather than trusting one
   * `bufferedamountlow` event: that event only fires on the crossing, and the
   * queue can drain in the gap between reading bufferedAmount and attaching
   * the listener — which deadlocks the transfer permanently (observed: the
   * first e2e run stalled after one file). The timeout makes a missed edge
   * cost 50 ms instead of everything.
   */
  async function drain(): Promise<void> {
    while (dc.readyState === 'open' && dc.bufferedAmount > BUFFER_HIGH) {
      await new Promise<void>((resolve) => {
        const done = () => { clearTimeout(timer); dc.removeEventListener('bufferedamountlow', done); resolve() }
        const timer = setTimeout(done, 50)
        dc.addEventListener('bufferedamountlow', done, { once: true })
      })
    }
  }

  dc.addEventListener('message', (e) => {
    if (typeof e.data !== 'string') return
    let msg: { type: string; have?: string[] }
    try { msg = JSON.parse(e.data) } catch { return }
    if (msg.type === 'inventory') void sendInventory()
    else if (msg.type === 'pull') void sendAll(msg.have ?? [])
  })

  async function sendInventory(): Promise<void> {
    const dir = await modelDir(spec, false)
    const files = dir ? await listFiles(dir) : []
    const inv: Inventory = {
      files: files.length,
      bytes: files.reduce((a, f) => a + f.size, 0),
      names: files.map((f) => f.name),
    }
    dc.send(JSON.stringify({ type: 'inventory', ...inv }))
    onEvent?.(`inventory: ${inv.files} files, ${(inv.bytes / 1e9).toFixed(2)} GB`)
  }

  async function sendAll(have: string[]): Promise<void> {
    const dir = await modelDir(spec, false)
    if (!dir) { dc.send(JSON.stringify({ type: 'pull-error', message: 'no cached weights on the host' })); return }
    const skip = new Set(have)
    const files = (await listFiles(dir)).filter((f) => !skip.has(f.name))
    onEvent?.(`serving ${files.length} files (${(files.reduce((a, f) => a + f.size, 0) / 1e9).toFixed(2)} GB)`)
    try {
      for (const { name, size } of files) {
        const handle = await dir.getFileHandle(name)
        const file = await handle.getFile()
        if (size === 0) {   // empty file still needs its (single) frame
          dc.send(frame({ name, off: 0, total: 0, sha: await sha256(new Uint8Array(0)) }, new Uint8Array(0)))
          continue
        }
        // Read in CHUNKs and slice pieces out of that: one OPFS read per 8 MB
        // rather than per 240 KB cuts the async round trips 32x, and a 242 MB
        // buffer still never exists whole on the sender.
        for (let base = 0; base < size; base += CHUNK) {
          const chunk = new Uint8Array(await file.slice(base, Math.min(base + CHUNK, size)).arrayBuffer())
          for (let o = 0; o < chunk.byteLength; o += PIECE) {
            const payload = chunk.subarray(o, Math.min(o + PIECE, chunk.byteLength))
            const header: PieceHeader = { name, off: base + o, total: size, sha: await sha256(payload) }
            await drain()
            if (dc.readyState !== 'open') return
            dc.send(frame(header, payload))
          }
        }
      }
      dc.send(JSON.stringify({ type: 'pull-done' }))
      onEvent?.('serve complete')
    } catch (err) {
      dc.send(JSON.stringify({ type: 'pull-error', message: err instanceof Error ? err.message : String(err) }))
    }
  }
}

// ============================================================
// GUEST — pull into the local cache
// ============================================================

/** Ask the peer what it has cached. Rejects if the peer never answers. */
export function fetchInventory(dc: RTCDataChannel, timeoutMs = 15_000): Promise<Inventory> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { cleanup(); reject(new Error('peer did not answer the inventory request')) }, timeoutMs)
    const onMsg = (e: MessageEvent) => {
      if (typeof e.data !== 'string') return
      const msg = JSON.parse(e.data) as { type: string } & Inventory
      if (msg.type !== 'inventory') return
      cleanup()
      resolve({ files: msg.files, bytes: msg.bytes, names: msg.names })
    }
    const cleanup = () => { clearTimeout(timer); dc.removeEventListener('message', onMsg) }
    dc.addEventListener('message', onMsg)
    dc.send(JSON.stringify({ type: 'inventory' }))
  })
}

/** Files this device already holds for `spec` — the resume set. */
export async function localFiles(spec: ModelSpec): Promise<string[]> {
  const dir = await modelDir(spec, false)
  return dir ? (await listFiles(dir)).map((f) => f.name) : []
}

/**
 * Replicate the peer's cache into this device's OPFS.
 *
 * Each file is assembled in memory and written once complete, so peak host
 * memory is ONE file (242 MB for the largest MLX embedding buffer) — the same
 * ceiling the loader itself works under. A file whose pieces do not all arrive
 * is never written, so an interrupted pull can only leave whole files behind.
 */
export async function pullWeights(
  dc: RTCDataChannel,
  spec: ModelSpec,
  onProgress?: (p: PullProgress) => void,
  expected?: Inventory,
): Promise<{ files: number; bytes: number; seconds: number }> {
  dc.binaryType = 'arraybuffer'
  const dir = await modelDir(spec, true)
  if (!dir) throw new Error('this browser has no OPFS — cannot cache weights here')

  const have = new Set(await localFiles(spec))
  const bytesTotal = expected?.bytes ?? 0
  const filesTotal = expected?.files ?? 0
  const t0 = performance.now()
  let filesDone = have.size
  let bytesDone = 0

  // The file currently being assembled. Only one is open at a time — the
  // sender walks files in order and the channel is ordered.
  let open: { name: string; buf: Uint8Array<ArrayBuffer>; got: number } | null = null

  return await new Promise((resolve, reject) => {
    const finish = (err?: Error) => {
      dc.removeEventListener('message', onMsg)
      if (err) reject(err)
      else resolve({ files: filesDone, bytes: bytesDone, seconds: (performance.now() - t0) / 1000 })
    }

    // Pieces are processed STRICTLY in arrival order. `handle` awaits (hashing,
    // OPFS writes), so firing it directly from the event would let two pieces
    // interleave and race on `open` — chaining onto a queue keeps the ordered
    // channel's ordering all the way to disk.
    let queue: Promise<void> = Promise.resolve()
    const onMsg = (e: MessageEvent) => { queue = queue.then(() => handle(e)) }

    async function handle(e: MessageEvent): Promise<void> {
      if (typeof e.data === 'string') {
        const msg = JSON.parse(e.data) as { type: string; message?: string }
        if (msg.type === 'pull-done') finish()
        else if (msg.type === 'pull-error') finish(new Error(msg.message ?? 'peer failed to serve'))
        return
      }
      try {
        const { header, payload } = unframe(e.data as ArrayBuffer)
        if (await sha256(payload) !== header.sha) {
          throw new Error(`${header.name}+${header.off}: piece failed its SHA-256 — transfer corrupted`)
        }
        if (!open || open.name !== header.name) {
          if (open) throw new Error(`peer moved to ${header.name} with ${open.name} incomplete`)
          open = { name: header.name, buf: new Uint8Array(header.total), got: 0 }
        }
        open.buf.set(payload, header.off)
        open.got += payload.byteLength
        bytesDone += payload.byteLength
        if (open.got >= header.total) {
          const fh = await dir!.getFileHandle(open.name, { create: true })
          const w = await (fh as unknown as { createWritable(): Promise<FileSystemWritableFileStream> }).createWritable()
          await w.write(open.buf)
          await w.close()
          open = null
          filesDone++
        }
        onProgress?.({
          filesDone, filesTotal, bytesDone, bytesTotal,
          rate: bytesDone / Math.max((performance.now() - t0) / 1000, 0.001),
        })
      } catch (err) {
        finish(err instanceof Error ? err : new Error(String(err)))
      }
    }

    dc.addEventListener('message', onMsg)
    dc.send(JSON.stringify({ type: 'pull', have: [...have] }))
  })
}
