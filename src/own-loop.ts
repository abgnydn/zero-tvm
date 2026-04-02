/**
 * OWN DECODE LOOP — Drives GPU directly using TVM's compiled shaders.
 *
 * No TVM WASM in the hot path. We record ONE token via TVM, then take over:
 * - Use TVM's compiled pipelines (already on GPU)
 * - Use TVM's weight buffers (static, GPU-resident)
 * - Use TVM's KV cache buffers (GPU-resident, we manage pages)
 * - Own uniform buffers (we write position, token_id, page table)
 * - Own bind groups (rebuilt from captured layouts)
 * - Own token loop: write uniforms → submit dispatches → read token → repeat
 *
 * Build order:
 * 1. Static replay: capture one token's full state, replay it
 * 2. Add patches: parameterize by position/token_id
 * 3. Add page management
 */

// ============================================================
// Types
// ============================================================

interface CapturedBGEntry {
  binding: number
  resource:
    | { type: 'buffer'; buffer: GPUBuffer; offset: number; size: number }
    | { type: 'other'; raw: GPUBindingResource }
}

interface CapturedBindGroup {
  layout: GPUBindGroupLayout
  entries: CapturedBGEntry[]
}

interface DispatchOp {
  pipeline: GPUComputePipeline
  bgSlots: Array<{ index: number; group: GPUBindGroup; captured: CapturedBindGroup }>
  workgroups: [number, number, number]
}

interface CopyOp {
  src: GPUBuffer; srcOffset: number
  dst: GPUBuffer; dstOffset: number
  size: number
}

interface WriteSnapshot {
  buffer: GPUBuffer
  offset: number
  data: Uint8Array
}

/** A GPU-side snapshot of a large buffer's full contents */
interface LargeBufferSnapshot {
  original: GPUBuffer
  snapshot: GPUBuffer  // full copy, created at build time
  size: number
}

/** Everything captured from one TVM decode token */
interface TokenSnapshot {
  dispatches: DispatchOp[]
  copy: CopyOp | null
  writes: WriteSnapshot[]
}

/** Our own decode engine */
interface DecodeEngine {
  device: GPUDevice
  snapshot: TokenSnapshot
  stableBuffers: Map<GPUBuffer, GPUBuffer>
  bindGroups: Array<{ index: number; group: GPUBindGroup }[]>
  dispatchWriteMap: Array<Map<GPUBuffer, GPUBuffer>>
  largeBufferSnapshots: LargeBufferSnapshot[]  // full-buffer snapshots for partial-write buffers
  position: number
  tokenId: number
  attentionUniformIndices: number[]
  initialNnzPages: number
  pageSize: number
}

// ============================================================
// State
// ============================================================

type Phase = 'disabled' | 'waiting' | 'capturing' | 'ready'

let phase: Phase = 'disabled'
let tokenCount = 0
let capturedSnapshot: TokenSnapshot | null = null
export let engine: DecodeEngine | null = null
let capturedDevice: GPUDevice | null = null

// Intercept helpers
const bgMap = new WeakMap<GPUBindGroup, CapturedBindGroup>()
export let origSubmit: ((bufs: Iterable<GPUCommandBuffer>) => void) | null = null
export let origWriteBuffer: ((buf: GPUBuffer, off: number, data: BufferSource | SharedArrayBuffer, dOff?: number, sz?: number) => void) | null = null
export let origCreateEncoder: ((desc?: GPUCommandEncoderDescriptor) => GPUCommandEncoder) | null = null

// Callbacks
let lastCapturedOutputToken = 0
let onTokenCallback: ((tokenId: number) => void) | null = null
let onReadyCallback: (() => void) | null = null

export function setOwnLoopEnabled(on: boolean): void {
  phase = on ? 'waiting' : 'disabled'
  tokenCount = 0
  capturedSnapshot = null
  engine = null
}

export function getOwnLoopStats(): string {
  if (!engine) return `OwnLoop: ${phase}`
  return `OwnLoop: ${phase} | pos=${engine.position} | ${engine.snapshot.dispatches.length} dispatches | ${engine.stableBuffers.size} stable bufs`
}

export function onToken(cb: (tokenId: number) => void): void { onTokenCallback = cb }
export function onReady(cb: () => void): void { onReadyCallback = cb }

// ============================================================
// Helpers
// ============================================================

function snap(data: BufferSource | SharedArrayBuffer, dOff?: number, sz?: number): Uint8Array {
  if (data instanceof ArrayBuffer) {
    const o = dOff ?? 0, s = sz ?? (data.byteLength - o)
    return new Uint8Array(data.slice(o, o + s))
  }
  if (ArrayBuffer.isView(data)) {
    const o = dOff ?? 0, s = sz ?? (data.byteLength - o)
    return new Uint8Array(data.buffer.slice(data.byteOffset + o, data.byteOffset + o + s))
  }
  return new Uint8Array(0)
}

function writeU32(val: number): ArrayBuffer {
  const ab = new ArrayBuffer(4)
  new DataView(ab).setUint32(0, val, true)
  return ab
}

// ============================================================
// Build the decode engine from the captured snapshot
// ============================================================

function buildEngine(device: GPUDevice, snapshot: TokenSnapshot): DecodeEngine {
  const stableBuffers = new Map<GPUBuffer, GPUBuffer>()

  // Use TVM's original bind groups directly — simplest correct approach.
  // Write snapshot data to original buffers before each token.
  const bindGroups: Array<{ index: number; group: GPUBindGroup }[]> = snapshot.dispatches.map(d =>
    d.bgSlots.map(slot => ({ index: slot.index, group: slot.group }))
  )
  const dispatchWriteMap: Array<Map<GPUBuffer, GPUBuffer>> = []  // unused but required by type

  for (const w of snapshot.writes) {
    stableBuffers.set(w.buffer, w.buffer)
  }

  // Snapshot large buffers that received partial writes.
  // These buffers (4096B page tables, 1028B position maps) have data at offsets
  // we didn't capture via writeBuffer. TVM's next token may overwrite them
  // before we abort. GPU-side copy preserves the full state.
  const largeBufferSnapshots: LargeBufferSnapshot[] = []
  const snapshotted = new Set<GPUBuffer>()
  for (const w of snapshot.writes) {
    if (w.buffer.size > 256 && w.data.length < w.buffer.size && !snapshotted.has(w.buffer)) {
      snapshotted.add(w.buffer)
      const snap = device.createBuffer({
        size: w.buffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: `snap_${w.buffer.size}`,
      })
      // GPU-side copy: original → snapshot (captures full buffer state at build time)
      const enc = origCreateEncoder!()
      enc.copyBufferToBuffer(w.buffer, 0, snap, 0, w.buffer.size)
      origSubmit!([enc.finish()])
      largeBufferSnapshots.push({ original: w.buffer, snapshot: snap, size: w.buffer.size })
    }
  }
  if (largeBufferSnapshots.length > 0) {
    console.log(`[own-loop] Snapshotted ${largeBufferSnapshots.length} large buffers`)
  }

  // Extract initial position from the counter writes
  let position = 0
  for (const w of snapshot.writes) {
    if (w.data.length === 4 && w.buffer.size === 4096) {
      const val = w.data[0] | (w.data[1] << 8) | (w.data[2] << 16) | (w.data[3] << 24)
      if (val > 0 && val < 10000) {
        position = Math.max(position, val)
      }
    }
  }

  // Dump the 64B uniform writes to understand struct layout
  for (let i = 0; i < snapshot.writes.length; i++) {
    const w = snapshot.writes[i]
    if (w.buffer.size === 4096 && w.data.length <= 16) {
      const u32s: number[] = []
      for (let j = 0; j < w.data.length; j += 4) {
        u32s.push(w.data[j] | (w.data[j + 1] << 8) | (w.data[j + 2] << 16) | (w.data[j + 3] << 24))
      }
      console.log(`[own-loop] write[${i}] 4096B buf (${w.data.length}B): [${u32s.join(', ')}]`)
    }
    if (w.data.length >= 8 && w.data.length <= 20 && w.buffer.size <= 32) {
      const u32s: number[] = []
      for (let j = 0; j < w.data.length; j += 4) {
        u32s.push(w.data[j] | (w.data[j + 1] << 8) | (w.data[j + 2] << 16) | (w.data[j + 3] << 24))
      }
      console.log(`[own-loop] write[${i}] ${w.buffer.size}B buf (${w.data.length}B): [${u32s.join(', ')}]`)
    }
    if (w.data.length === 56) {
      const u32s: number[] = []
      for (let j = 0; j < w.data.length; j += 4) {
        u32s.push(w.data[j] | (w.data[j + 1] << 8) | (w.data[j + 2] << 16) | (w.data[j + 3] << 24))
      }
      console.log(`[own-loop] write[${i}] 56B uniform: [${u32s.join(', ')}]`)
    }
  }

  // Check for buffers that appear in both writeBuffer AND bind groups
  const writtenBufs = new Set(snapshot.writes.map(w => w.buffer))
  const bgBufs = new Set<GPUBuffer>()
  for (const d of snapshot.dispatches) {
    for (const slot of d.bgSlots) {
      for (const e of slot.captured.entries) {
        if (e.resource.type === 'buffer') bgBufs.add(e.resource.buffer)
      }
    }
  }
  const overlap = [...writtenBufs].filter(b => bgBufs.has(b))
  console.log(`[own-loop] Written bufs: ${writtenBufs.size}, BG bufs: ${bgBufs.size}, overlap: ${overlap.length}`)
  for (const b of overlap) {
    console.log(`[own-loop]   overlap: ${b.size}B buf, stable=${stableBuffers.has(b)}`)
  }

  console.log(`[own-loop] Engine built: ${snapshot.dispatches.length} dispatches, ${stableBuffers.size} stable bufs, initial pos=${position}`)

  // Identify the 56B attention uniform writes (one per layer, every 10 writes)
  // and the page table related values
  const PAGE_SIZE = 16
  const attentionUniformIndices: number[] = []
  let initialNnzPages = 0
  for (let i = 0; i < snapshot.writes.length; i++) {
    if (snapshot.writes[i].data.length === 56) {
      attentionUniformIndices.push(i)
      if (initialNnzPages === 0) {
        // nnz_pages is at u32 offset 4 (byte offset 16)
        const d = snapshot.writes[i].data
        initialNnzPages = d[16] | (d[17] << 8) | (d[18] << 16) | (d[19] << 24)
      }
    }
  }
  console.log(`[own-loop] Attention uniforms: ${attentionUniformIndices.length} writes, initial nnz_pages=${initialNnzPages}`)

  // Get the token_id that TVM used during the captured token (write[0])
  const capturedTokenId = snapshot.writes[0].data.length >= 4
    ? (snapshot.writes[0].data[0] | (snapshot.writes[0].data[1] << 8) | (snapshot.writes[0].data[2] << 16) | (snapshot.writes[0].data[3] << 24))
    : 0
  console.log(`[own-loop] Captured token_id=${capturedTokenId}`)

  return { device, snapshot, stableBuffers, bindGroups, dispatchWriteMap, largeBufferSnapshots, position, tokenId: capturedTokenId, attentionUniformIndices, initialNnzPages, pageSize: PAGE_SIZE }
}

// ============================================================
// The decode loop — NO TVM
// ============================================================

/**
 * Generate tokens. Call after engine is ready.
 * Returns array of token IDs.
 */
export async function generate(maxTokens: number): Promise<number[]> {
  if (!engine || !origCreateEncoder || !origSubmit || !origWriteBuffer) {
    throw new Error('Engine not ready')
  }

  console.log(`[own-loop] Starting generate with token_id=${engine.tokenId}, pos=${engine.position}`)

  const tokens: number[] = []
  // Phi-3-mini stop tokens
  const STOP_TOKENS = new Set([2, 32000, 32007])  // </s>, <|endoftext|>, <|end|>

  for (let i = 0; i < maxTokens; i++) {
    engine.position++

    // Restore large buffers from GPU-side snapshots (page tables, position maps)
    if (engine.largeBufferSnapshots.length > 0) {
      const enc = origCreateEncoder()
      for (const s of engine.largeBufferSnapshots) {
        enc.copyBufferToBuffer(s.snapshot, 0, s.original, 0, s.size)
      }
      origSubmit([enc.finish()])
    }

    // Write ALL uniforms from snapshot, patching the ones that change per token.
    // We must write all because TVM may have overwritten the original buffers
    // while starting the next token before we aborted.
    for (let w = 0; w < engine.snapshot.writes.length; w++) {
      const recorded = engine.snapshot.writes[w]
      let data: ArrayBuffer = recorded.data.slice().buffer

      if (w === 0) data = writeU32(engine.tokenId)
      else if (w === 4 || w === 11 || w === 12) data = writeU32(engine.position)

      origWriteBuffer(recorded.buffer, recorded.offset, data)

      if (recorded.data.length === 56 && engine.attentionUniformIndices.includes(w)) {
        const currentNnzPages = Math.floor(engine.position / engine.pageSize) + 1
        if (currentNnzPages !== engine.initialNnzPages) {
          origWriteBuffer(recorded.buffer, recorded.offset + 16, writeU32(currentNnzPages))
        }
      }
    }

    // Submit all dispatches in batches — safe because each dispatch reads from its own copy
    const BATCH = 32
    for (let d = 0; d < engine.snapshot.dispatches.length; d += BATCH) {
      const end = Math.min(d + BATCH, engine.snapshot.dispatches.length)
      const enc = origCreateEncoder()
      for (let j = d; j < end; j++) {
        const dispatch = engine.snapshot.dispatches[j]
        const pass = enc.beginComputePass()
        pass.setPipeline(dispatch.pipeline)
        for (const bg of engine.bindGroups[j]) pass.setBindGroup(bg.index, bg.group)
        pass.dispatchWorkgroups(...dispatch.workgroups)
        pass.end()
      }
      origSubmit([enc.finish()])
    }

    // Copy output to staging
    const stagingBuf = engine.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'own_staging',
    })
    if (engine.snapshot.copy) {
      const enc = origCreateEncoder()
      enc.copyBufferToBuffer(engine.snapshot.copy.src, engine.snapshot.copy.srcOffset, stagingBuf, 0, engine.snapshot.copy.size)
      origSubmit([enc.finish()])
    }

    // 5. Read token
    await stagingBuf.mapAsync(GPUMapMode.READ)
    const mapped = stagingBuf.getMappedRange()
    const tokenId = new DataView(mapped).getInt32(0, true)
    stagingBuf.unmap()
    stagingBuf.destroy()

    if (tokenId < 0 || tokenId >= 32064) {
      console.log(`[own-loop] Invalid token ${tokenId} at pos ${engine.position}, stopping`)
      break
    }

    engine.tokenId = tokenId
    tokens.push(tokenId)
    if (onTokenCallback) onTokenCallback(tokenId)

    if (STOP_TOKENS.has(tokenId)) {
      console.log(`[own-loop] EOS token ${tokenId} at pos ${engine.position}`)
      break
    }
  }

  return tokens
}

// ============================================================
// Device patching — capture ONE token then hand off
// ============================================================

export function patchDeviceOwnLoop(device: GPUDevice): void {
  capturedDevice = device
  origSubmit = device.queue.submit.bind(device.queue)
  origWriteBuffer = device.queue.writeBuffer.bind(device.queue)
  origCreateEncoder = device.createCommandEncoder.bind(device)

  let currentWrites: WriteSnapshot[] = []
  let currentDispatches: DispatchOp[] = []
  let currentCopy: CopyOp | null = null

  // === createBindGroup ===
  const prevCreateBG = device.createBindGroup.bind(device)
  device.createBindGroup = function(desc: GPUBindGroupDescriptor) {
    const bg = prevCreateBG(desc)
    const entries: CapturedBGEntry[] = []
    for (const e of desc.entries) {
      const r = e.resource
      if (r && typeof r === 'object' && 'buffer' in r) {
        const br = r as GPUBufferBinding
        entries.push({ binding: e.binding, resource: { type: 'buffer', buffer: br.buffer, offset: br.offset ?? 0, size: br.size ?? br.buffer.size } })
      } else {
        entries.push({ binding: e.binding, resource: { type: 'other', raw: r } })
      }
    }
    bgMap.set(bg, { layout: desc.layout as GPUBindGroupLayout, entries })
    return bg
  }

  // === buffer.destroy: prevent during capture ===
  const prevCreateBuf = device.createBuffer
  device.createBuffer = function(desc: GPUBufferDescriptor) {
    const buf = prevCreateBuf.call(device, desc)
    const realDestroy = buf.destroy.bind(buf)
    buf.destroy = function() {
      if (phase === 'capturing' || phase === 'ready') return
      return realDestroy()
    }

    if (desc.usage & GPUBufferUsage.MAP_READ) {
      // Token boundary on mapAsync
      const realMap = buf.mapAsync.bind(buf)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buf as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        onTokenBoundary()
        return realMap(...args)
      }

      // Capture output token from getMappedRange
      const realGetMapped = buf.getMappedRange.bind(buf)
      buf.getMappedRange = function(offset?: number, size?: number) {
        const range = realGetMapped(offset, size)
        if (range.byteLength >= 4) {
          const outputToken = new DataView(range).getInt32(0, true)
          if (outputToken >= 0 && outputToken < 100000) {
            // Only update during capture and the immediate readback after
            if (phase === 'capturing' || (phase === 'ready' && lastCapturedOutputToken === 0)) {
              lastCapturedOutputToken = outputToken
              console.log(`[own-loop] Captured output token=${outputToken}`)
            }
          }
        }
        return range
      }
    }
    return buf
  }

  // === writeBuffer ===
  const prevWriteBuf = device.queue.writeBuffer
  device.queue.writeBuffer = function(buf: GPUBuffer, off: number, data: BufferSource | SharedArrayBuffer, dOff?: number, sz?: number): undefined {
    if (phase === 'capturing') {
      currentWrites.push({ buffer: buf, offset: off, data: snap(data, dOff, sz) })
    }
    prevWriteBuf.call(device.queue, buf, off, data, dOff, sz)
  }

  // === createCommandEncoder ===
  const prevCreateEnc = device.createCommandEncoder
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const enc = prevCreateEnc.call(device, desc)
    if (phase !== 'capturing') return enc

    // Capture copies
    const prevCopy = enc.copyBufferToBuffer.bind(enc)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(enc as any).copyBufferToBuffer = function(...args: unknown[]) {
      if (args.length >= 5) {
        const [s, so, d, d2, sz] = args as [GPUBuffer, number, GPUBuffer, number, number]
        currentCopy = { src: s, srcOffset: so, dst: d, dstOffset: d2, size: sz }
      }
      return (prevCopy as Function)(...args)
    }

    // Capture dispatches
    const prevBeginPass = enc.beginComputePass
    enc.beginComputePass = function(pd?: GPUComputePassDescriptor) {
      const pass = prevBeginPass.call(enc, pd)
      const pending: DispatchOp = { pipeline: null as unknown as GPUComputePipeline, bgSlots: [], workgroups: [0, 0, 0] }

      const prevSP = pass.setPipeline.bind(pass)
      pass.setPipeline = function(p: GPUComputePipeline) { pending.pipeline = p; return prevSP(p) }

      const prevSBG = pass.setBindGroup.bind(pass)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(pass as any).setBindGroup = function(idx: number, grp: GPUBindGroup, ...rest: unknown[]) {
        const cap = bgMap.get(grp)
        if (cap) pending.bgSlots.push({ index: idx, group: grp, captured: cap })
        return (prevSBG as Function)(idx, grp, ...rest)
      }

      const prevDisp = pass.dispatchWorkgroups.bind(pass)
      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        pending.workgroups = [x, y ?? 1, z ?? 1]
        currentDispatches.push(pending)
        return prevDisp(x, y, z)
      }
      return pass
    }
    return enc
  }

  function onTokenBoundary(): void {
    if (phase === 'waiting') {
      tokenCount++
      if (tokenCount === 1) {
        // Prefill done → capture first decode token
        phase = 'capturing'
        currentWrites = []
        currentDispatches = []
        currentCopy = null
        console.log('[own-loop] Capturing first decode token...')
      }
    } else if (phase === 'capturing') {
      // Done capturing
      capturedSnapshot = {
        dispatches: currentDispatches,
        copy: currentCopy,
        writes: currentWrites,
      }
      console.log(`[own-loop] Captured: ${currentDispatches.length} dispatches, ${currentWrites.length} writes, copy=${currentCopy ? 'yes' : 'no'}`)

      // Build the engine — tokenId will be set from getMappedRange capture
      engine = buildEngine(capturedDevice!, capturedSnapshot)
      // lastCapturedOutputToken was set by getMappedRange during the captured token's readback
      // It gets the token that was the OUTPUT of the captured dispatch sequence
      phase = 'ready'

      if (onReadyCallback) onReadyCallback()
    }
  }
}
