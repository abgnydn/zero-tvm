/**
 * DISPATCH REPLAY v4 — 3-Token Recording + Mathematically Derived Offsets
 *
 * Records 3 consecutive decode tokens to compute exact per-token deltas.
 * No guessing. No trial and error.
 *
 * Token A (record): captures ops + writes, position=P
 * Token B (record): captures writes, position=P+1
 * Token C (record): captures writes, position=P+2
 *
 * From A→B→C we derive:
 *   - Which writes are counters (value increments by exactly 1 each token)
 *   - Which writes are token_id (value changes unpredictably)
 *   - The exact initial value and per-token delta for each counter
 *   - How many tokens ahead of the initial value the first replay should be
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

interface RecordedWrite {
  buffer: GPUBuffer
  offset: number
  data: Uint8Array
}

interface DispatchOp {
  type: 'dispatch'
  pipeline: GPUComputePipeline
  bgSlots: Array<{ index: number; captured: CapturedBindGroup; dynOffsets?: Uint32Array }>
  workgroups: [number, number, number]
}

interface CopyOp {
  type: 'copy'
  src: GPUBuffer; srcOffset: number
  dst: GPUBuffer; dstOffset: number
  size: number
}

type RecordedOp = DispatchOp | CopyOp

interface PatchSlot {
  role: 'token_id' | 'counter' | 'passthrough'
  writeIndex: number           // which writeBuffer call (0-356) this corresponds to
  stableBuffer: GPUBuffer
  offset: number
  size: number
  valueA: number
  valueB: number
  valueC: number
}

interface FrozenTape {
  ops: FrozenOp[]
  patches: PatchSlot[]
  copySrc: GPUBuffer | null
  copySrcOffset: number
  copySize: number
  // How many tokens ahead of valueA the first replay token is
  replayOffset: number  // = 3 (tokens A, B, C were recorded, replay starts at A+3)
}

type FrozenOp =
  | { type: 'dispatch'; pipeline: GPUComputePipeline; bindGroups: Array<{ index: number; group: GPUBindGroup; dynOffsets?: Uint32Array }>; workgroups: [number, number, number] }

// ============================================================
// State
// ============================================================

type Phase = 'disabled' | 'waiting' | 'record_a' | 'record_b' | 'record_c' | 'replaying'

let phase: Phase = 'disabled'
let tokenCount = 0
let replayCount = 0
let replayWriteIndex = 0  // tracks TVM's writeBuffer call position during replay

let stableBufferMap = new Map<GPUBuffer, GPUBuffer>()  // TVM buffer → our stable buffer
let recordedOps: RecordedOp[] = []
let writesA: RecordedWrite[] = []
let writesB: RecordedWrite[] = []
let writesC: RecordedWrite[] = []

let frozenTape: FrozenTape | null = null

let capturedDevice: GPUDevice | null = null
let origSubmit: ((bufs: Iterable<GPUCommandBuffer>) => void) | null = null
let origWriteBuffer: ((buf: GPUBuffer, off: number, data: BufferSource | SharedArrayBuffer, dOff?: number, sz?: number) => void) | null = null
let origCreateEncoder: ((desc?: GPUCommandEncoderDescriptor) => GPUCommandEncoder) | null = null

const bgMap = new WeakMap<GPUBindGroup, CapturedBindGroup>()
let lastTokenId: number | null = null

export function setReplayEnabled(on: boolean): void {
  phase = on ? 'waiting' : 'disabled'
  tokenCount = 0
  replayCount = 0
  frozenTape = null
  lastTokenId = null
}

export function getReplayStats(): string {
  if (!frozenTape) return `Replay: ${phase}, token ${tokenCount}`
  const dc = frozenTape.ops.filter(o => o.type === 'dispatch').length
  const counters = frozenTape.patches.filter(p => p.role === 'counter')
  const tids = frozenTape.patches.filter(p => p.role === 'token_id')
  return `Replay: ${phase} | ${dc} dispatches | ${tids.length} token_id + ${counters.length} counters | offset=${frozenTape.replayOffset} | ${replayCount} replayed`
}

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

function readU32(buf: Uint8Array): number {
  if (buf.length < 4) return -1
  return buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24)
}

function writeU32(val: number): ArrayBuffer {
  const ab = new ArrayBuffer(4)
  new DataView(ab).setUint32(0, val, true)
  return ab
}

function fakePass(): GPUComputePassEncoder {
  return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, insertDebugMarker() {}, pushDebugGroup() {}, popDebugGroup() {}, end() {}, label: '' } as unknown as GPUComputePassEncoder
}

function fakeEncoder(): GPUCommandEncoder {
  return { beginComputePass() { return fakePass() }, beginRenderPass() { return {} as GPURenderPassEncoder }, copyBufferToBuffer() {}, copyBufferToTexture() {}, copyTextureToBuffer() {}, copyTextureToTexture() {}, clearBuffer() {}, insertDebugMarker() {}, pushDebugGroup() {}, popDebugGroup() {}, resolveQuerySet() {}, finish() { return {} as GPUCommandBuffer }, label: '' } as unknown as GPUCommandEncoder
}

// ============================================================
// Tape Freezer (uses 3 tokens of data)
// ============================================================

function freezeTape(): void {
  if (!capturedDevice || !origWriteBuffer) return
  const device = capturedDevice

  console.log(`[replay] Freezing: ${recordedOps.length} ops, writes: A=${writesA.length} B=${writesB.length} C=${writesC.length}`)

  // Dump all writes for analysis
  console.log(`[replay] === ALL WRITES (token A) ===`)
  for (let i = 0; i < writesA.length; i++) {
    const w = writesA[i]
    const val = w.data.length >= 4 ? readU32(w.data) : -1
    if (w.buffer.size >= 256 || w.data.length > 4) {
      console.log(`[replay] write[${i}]: buf=${w.buffer.size}B, ${w.data.length}B at offset ${w.offset}, val=${val}`)
    }
  }

  stableBufferMap = new Map<GPUBuffer, GPUBuffer>()
  const bufRemap = stableBufferMap
  function stable(orig: GPUBuffer): GPUBuffer {
    let s = bufRemap.get(orig)
    if (!s) {
      s = device.createBuffer({ size: orig.size, usage: orig.usage | GPUBufferUsage.COPY_DST, label: `stable_${orig.size}` })
      bufRemap.set(orig, s)
    }
    return s
  }

  // Copy token A's data into stable buffers
  for (const w of writesA) {
    const s = stable(w.buffer)
    origWriteBuffer!(s, w.offset, w.data.buffer)
  }

  // Diff A vs B vs C to classify patches
  const patches: PatchSlot[] = []
  const min = Math.min(writesA.length, writesB.length, writesC.length)

  for (let i = 0; i < min; i++) {
    const wA = writesA[i], wB = writesB[i], wC = writesC[i]

    const s = bufRemap.get(wA.buffer)
    if (!s) continue

    // Check if data changed A→B
    let changedAB = wA.data.length !== wB.data.length
    if (!changedAB) { for (let b = 0; b < wA.data.length; b++) { if (wA.data[b] !== wB.data[b]) { changedAB = true; break } } }

    // Large buffers (page tables, position maps) may change at page boundaries
    // even though they were constant in our 3-token window. Always route them.
    const isLargeBuffer = wA.buffer.size >= 4096

    if (!changedAB && !isLargeBuffer) continue
    if (!changedAB && isLargeBuffer) {
      const val = readU32(wA.data)
      patches.push({ role: 'passthrough', writeIndex: i, stableBuffer: s, offset: wA.offset, size: wA.data.length, valueA: val, valueB: val, valueC: val })
      console.log(`[replay] patch[${patches.length - 1}] write[${i}]: LARGE_BUF (${wA.buffer.size}B buf, ${wA.data.length}B write at offset ${wA.offset}, val=${val})`)
      continue
    }

    const vA = readU32(wA.data)
    const vB = readU32(wB.data)
    const vC = readU32(wC.data)

    // Counter: A→B increments by 1 AND B→C increments by 1
    if (wA.data.length === 4 && vB === vA + 1 && vC === vB + 1) {
      patches.push({ role: 'counter', writeIndex: i, stableBuffer: s, offset: wA.offset, size: 4, valueA: vA, valueB: vB, valueC: vC })
      console.log(`[replay] patch[${patches.length - 1}] write[${i}]: COUNTER A=${vA} B=${vB} C=${vC}`)
    }
    // Token ID: small integers that change unpredictably
    else if (wA.data.length === 4 && vA >= 0 && vA < 100000 && vB >= 0 && vB < 100000) {
      patches.push({ role: 'token_id', writeIndex: i, stableBuffer: s, offset: wA.offset, size: 4, valueA: vA, valueB: vB, valueC: vC })
      console.log(`[replay] patch[${patches.length - 1}] write[${i}]: TOKEN_ID A=${vA} B=${vB} C=${vC}`)
    }
    // Passthrough: changes unpredictably, not a token ID (e.g., sampling random seed)
    else {
      patches.push({ role: 'passthrough', writeIndex: i, stableBuffer: s, offset: wA.offset, size: wA.data.length, valueA: vA, valueB: vB, valueC: vC })
      console.log(`[replay] patch[${patches.length - 1}] write[${i}]: PASSTHROUGH A=${vA} B=${vB} C=${vC} (${wA.data.length}B)`)
    }
  }

  // Find copy source (for staging buffer write-back)
  let copySrc: GPUBuffer | null = null, copySrcOffset = 0, copySize = 0
  for (const op of recordedOps) {
    if (op.type === 'copy') {
      copySrc = bufRemap.get(op.src) ?? op.src
      copySrcOffset = op.srcOffset
      copySize = op.size
    }
  }

  // Build frozen ops (dispatches only, copy handled separately)
  const frozenOps: FrozenOp[] = []
  for (const op of recordedOps) {
    if (op.type === 'copy') continue
    const bgs: Array<{ index: number; group: GPUBindGroup; dynOffsets?: Uint32Array }> = []
    for (const slot of op.bgSlots) {
      const entries: GPUBindGroupEntry[] = slot.captured.entries.map(e => {
        if (e.resource.type === 'buffer') {
          return { binding: e.binding, resource: { buffer: bufRemap.get(e.resource.buffer) ?? e.resource.buffer, offset: e.resource.offset, size: e.resource.size } }
        }
        return { binding: e.binding, resource: e.resource.raw }
      })
      bgs.push({ index: slot.index, group: device.createBindGroup({ layout: slot.captured.layout, entries }), dynOffsets: slot.dynOffsets })
    }
    frozenOps.push({ type: 'dispatch', pipeline: op.pipeline, bindGroups: bgs, workgroups: op.workgroups })
  }

  // replayOffset = 3: we recorded A, B, C. First replay is token D = A+3
  frozenTape = { ops: frozenOps, patches, copySrc, copySrcOffset, copySize, replayOffset: 3 }
  console.log(`[replay] Frozen: ${frozenOps.length} dispatches, ${patches.length} patches, copySrc=${copySrc ? 'yes' : 'NO'}, replayOffset=3`)
}

// ============================================================
// Replay
// ============================================================

function patchAndSubmit(tokenId: number | null, mappedBuffer: GPUBuffer): void {
  if (!frozenTape || !origWriteBuffer || !origCreateEncoder || !origSubmit) return

  const tokenOffset = frozenTape.replayOffset + replayCount  // how far ahead of valueA

  // Patch stable buffers with correct values
  for (const p of frozenTape.patches) {
    if (p.role === 'token_id' && tokenId !== null) {
      origWriteBuffer(p.stableBuffer, p.offset, writeU32(tokenId))
    } else if (p.role === 'counter') {
      const newVal = p.valueA + tokenOffset
      origWriteBuffer(p.stableBuffer, p.offset, writeU32(newVal))
    }
  }

  // Submit dispatches in pipelined batches
  const BATCH = 32
  for (let i = 0; i < frozenTape.ops.length; i += BATCH) {
    const end = Math.min(i + BATCH, frozenTape.ops.length)
    const enc = origCreateEncoder()
    for (let j = i; j < end; j++) {
      const op = frozenTape.ops[j]
      const pass = enc.beginComputePass()
      pass.setPipeline(op.pipeline)
      for (const bg of op.bindGroups) {
        bg.dynOffsets ? pass.setBindGroup(bg.index, bg.group, bg.dynOffsets) : pass.setBindGroup(bg.index, bg.group)
      }
      pass.dispatchWorkgroups(...op.workgroups)
      pass.end()
    }
    origSubmit([enc.finish()])
  }

  // Copy compute output to TVM's staging buffer
  if (frozenTape.copySrc) {
    try { mappedBuffer.unmap() } catch { /* ok */ }
    const enc = origCreateEncoder()
    enc.copyBufferToBuffer(frozenTape.copySrc, frozenTape.copySrcOffset, mappedBuffer, 0, frozenTape.copySize)
    origSubmit([enc.finish()])
  }

  replayCount++
}

// ============================================================
// Device Patching
// ============================================================

export function patchDeviceReplay(device: GPUDevice): void {
  capturedDevice = device
  origSubmit = device.queue.submit.bind(device.queue)
  origWriteBuffer = device.queue.writeBuffer.bind(device.queue)
  origCreateEncoder = device.createCommandEncoder.bind(device)

  // === createBindGroup: capture layout + entries ===
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

  // === buffer.destroy: prevent during record/replay ===
  const prevCreateBuf = device.createBuffer
  device.createBuffer = function(desc: GPUBufferDescriptor) {
    const buf = prevCreateBuf.call(device, desc)
    const realDestroy = buf.destroy.bind(buf)
    buf.destroy = function() {
      if (phase !== 'disabled' && phase !== 'waiting') return
      return realDestroy()
    }

    if (desc.usage & GPUBufferUsage.MAP_READ) {
      const realMap = buf.mapAsync.bind(buf)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buf as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        onTokenBoundary(buf)
        return realMap(...args)
      }
    }
    return buf
  }

  // === writeBuffer: snapshot during recording, route passthrough during replay ===
  const prevWriteBuf = device.queue.writeBuffer
  device.queue.writeBuffer = function(buf: GPUBuffer, off: number, data: BufferSource | SharedArrayBuffer, dOff?: number, sz?: number): undefined {
    if (phase === 'record_a') {
      writesA.push({ buffer: buf, offset: off, data: snap(data, dOff, sz) })
    } else if (phase === 'record_b') {
      writesB.push({ buffer: buf, offset: off, data: snap(data, dOff, sz) })
    } else if (phase === 'record_c') {
      writesC.push({ buffer: buf, offset: off, data: snap(data, dOff, sz) })
    } else if (phase === 'replaying' && origWriteBuffer) {
      // Route ALL writes to stable buffers — TVM's WASM computes correct
      // values for page tables, uniforms, etc. We just redirect them to
      // the buffers our frozen bind groups reference.
      const idx = replayWriteIndex++
      if (idx < writesA.length) {
        const recorded = writesA[idx]
        const stableBuf = stableBufferMap.get(recorded.buffer)
        if (stableBuf) {
          const snapshot = snap(data, dOff, sz)
          origWriteBuffer(stableBuf, recorded.offset, snapshot.buffer)
        } else if (replayCount < 40 && replayCount > 28) {
          console.log(`[replay] MISS write[${idx}]: no stable buf for ${recorded.buffer.size}B buffer`)
        }
      } else if (replayCount < 40 && replayCount > 28) {
        console.log(`[replay] EXTRA write idx=${idx} (recorded only ${writesA.length})`)
      }
    }
    // Always also pass through to TVM's own buffer
    prevWriteBuf.call(device.queue, buf, off, data, dOff, sz)
  }

  // === createCommandEncoder: record during record_a, fake during replay ===
  const prevCreateEnc = device.createCommandEncoder
  device.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    if (phase === 'replaying') return fakeEncoder()

    const enc = prevCreateEnc.call(device, desc)
    if (phase !== 'record_a') return enc

    // Record copies
    const prevCopy = enc.copyBufferToBuffer.bind(enc)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(enc as any).copyBufferToBuffer = function(...args: unknown[]) {
      if (args.length >= 5) {
        const [s, so, d, d2, sz] = args as [GPUBuffer, number, GPUBuffer, number, number]
        recordedOps.push({ type: 'copy', src: s, srcOffset: so, dst: d, dstOffset: d2, size: sz })
      } else {
        const [s, d, sz] = args as [GPUBuffer, GPUBuffer, number?]
        recordedOps.push({ type: 'copy', src: s, srcOffset: 0, dst: d, dstOffset: 0, size: sz ?? s.size })
      }
      return (prevCopy as Function)(...args)
    }

    // Record dispatches
    const prevBeginPass = enc.beginComputePass
    enc.beginComputePass = function(pd?: GPUComputePassDescriptor) {
      const pass = prevBeginPass.call(enc, pd)
      const pending: DispatchOp = { type: 'dispatch', pipeline: null as unknown as GPUComputePipeline, bgSlots: [], workgroups: [0, 0, 0] }

      const prevSP = pass.setPipeline.bind(pass)
      pass.setPipeline = function(p: GPUComputePipeline) { pending.pipeline = p; return prevSP(p) }

      const prevSBG = pass.setBindGroup.bind(pass)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(pass as any).setBindGroup = function(idx: number, grp: GPUBindGroup, ...rest: unknown[]) {
        const cap = bgMap.get(grp)
        if (cap) {
          let dyn: Uint32Array | undefined
          if (rest[0] instanceof Uint32Array) dyn = rest[0].slice()
          else if (Array.isArray(rest[0])) dyn = new Uint32Array(rest[0] as number[])
          pending.bgSlots.push({ index: idx, captured: cap, dynOffsets: dyn })
        }
        return (prevSBG as Function)(idx, grp, ...rest)
      }

      const prevDisp = pass.dispatchWorkgroups.bind(pass)
      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        pending.workgroups = [x, y ?? 1, z ?? 1]
        recordedOps.push(pending)
        return prevDisp(x, y, z)
      }
      return pass
    }
    return enc
  }

  // === submit: no-op during replay ===
  const prevSubmit = device.queue.submit
  device.queue.submit = function(bufs: Iterable<GPUCommandBuffer>) {
    if (phase === 'replaying') return
    return prevSubmit.call(device.queue, bufs)
  }
}

// ============================================================
// Token ID readback via getMappedRange
// ============================================================

export function patchStagingReadback(_device: GPUDevice): void {
  // Patch the PROTOTYPE once — catches ALL buffers regardless of creation order
  const origGetMapped = GPUBuffer.prototype.getMappedRange
  GPUBuffer.prototype.getMappedRange = function(offset?: number, size?: number) {
    const range = origGetMapped.call(this, offset, size)
    if (phase === 'replaying' && range.byteLength >= 4) {
      const val = new DataView(range).getInt32(0, true)
      if (replayCount < 5) console.log(`[replay] getMappedRange: ${range.byteLength}B, val=${val}`)
      if (val >= 0 && val < 100000) {
        lastTokenId = val
      }
    }
    return range
  }
}

// ============================================================
// State Machine
// ============================================================

function onTokenBoundary(mappedBuffer: GPUBuffer): void {
  if (phase === 'waiting') {
    tokenCount++
    if (tokenCount === 1) {
      // Prefill done → start recording token A
      phase = 'record_a'
      recordedOps = []
      writesA = []
      console.log('[replay] Recording token A...')
    }
  } else if (phase === 'record_a') {
    // Token A done → record token B (writes only)
    phase = 'record_b'
    writesB = []
    console.log(`[replay] Token A: ${recordedOps.length} ops, ${writesA.length} writes. Recording token B...`)
  } else if (phase === 'record_b') {
    // Token B done → record token C (writes only)
    phase = 'record_c'
    writesC = []
    console.log(`[replay] Token B: ${writesB.length} writes. Recording token C...`)
  } else if (phase === 'record_c') {
    // Token C done → freeze and switch to replay
    console.log(`[replay] Token C: ${writesC.length} writes. Freezing...`)
    freezeTape()
    phase = 'replaying'
    lastTokenId = null
  } else if (phase === 'replaying' && frozenTape) {
    if (replayCount < 15) {
      console.log(`[replay] token ${replayCount}: ${replayWriteIndex} writes (expected ${writesA.length}), lastTokenId=${lastTokenId}`)
    }
    replayWriteIndex = 0
    patchAndSubmit(lastTokenId, mappedBuffer)
  }
}
