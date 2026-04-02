/**
 * OWN DECODE LOOP
 *
 * Captures one TVM decode token's GPU dispatch tape, then drives
 * the GPU directly. Per token: patch 6 uniforms → 342 dispatches → read token.
 *
 * TVM handles: model load, shader compilation, prefill, first N tokens.
 * We handle: everything after that, 2x faster.
 */

// ============================================================
// Types
// ============================================================

interface Dispatch {
  pipeline: GPUComputePipeline
  bindGroups: Array<{ index: number; group: GPUBindGroup }>
  workgroups: [number, number, number]
}

interface Write {
  buffer: GPUBuffer
  offset: number
  data: Uint8Array
}

interface Copy {
  src: GPUBuffer
  srcOffset: number
  dst: GPUBuffer
  dstOffset: number
  size: number
}

interface Tape {
  dispatches: Dispatch[]
  writes: Write[]
  copy: Copy | null
}

// ============================================================
// State
// ============================================================

type Phase = 'off' | 'waiting' | 'capturing' | 'ready'

let phase: Phase = 'off'
let warmupCount = 0
const WARMUP = 3

let tape: Tape | null = null
let pendingDispatches: Dispatch[] = []
let pendingWrites: Write[] = []
let pendingCopy: Copy | null = null

let device: GPUDevice | null = null
let _origSubmit: ((b: Iterable<GPUCommandBuffer>) => void) | null = null
let _origWrite: ((b: GPUBuffer, o: number, d: BufferSource | SharedArrayBuffer, dO?: number, s?: number) => void) | null = null
let _origEncoder: ((d?: GPUCommandEncoderDescriptor) => GPUCommandEncoder) | null = null

let _onReady: (() => void) | null = null
let _onToken: ((id: number) => void) | null = null

// Bind group layout capture
const bgLayouts = new WeakMap<GPUBindGroup, unknown>()

// ============================================================
// Public API
// ============================================================

export function enable(): void { phase = 'waiting'; warmupCount = 0; tape = null }
export function disable(): void { phase = 'off' }
export function status(): string { return phase }
export function onReady(cb: () => void): void { _onReady = cb }
export function onToken(cb: (id: number) => void): void { _onToken = cb }

// ============================================================
// Generate
// ============================================================

export async function generate(maxTokens: number): Promise<number[]> {
  if (!tape || !device || !_origSubmit || !_origWrite || !_origEncoder) throw new Error('Not ready')

  const submit = _origSubmit
  const write = _origWrite
  const encoder = _origEncoder

  // Derive initial position and sequence base from captured writes
  const position0 = readU32(tape.writes[4]?.data)  // write[4] = position
  const seqBase = readU32(tape.writes[8]?.data)     // write[8] = sequence counter

  // Find attention uniform indices (56B writes) for nnz_pages patching
  const attnIndices: number[] = []
  let nnzPages0 = 0
  for (let i = 0; i < tape.writes.length; i++) {
    if (tape.writes[i].data.length === 56) {
      attnIndices.push(i)
      if (!nnzPages0) nnzPages0 = readU32(tape.writes[i].data.slice(16, 20))
    }
  }

  // Step 1: Pure replay to get first output token
  for (const w of tape.writes) write(w.buffer, w.offset, w.data.slice().buffer)
  for (let i = 0; i < tape.dispatches.length; i++) {
    const d = tape.dispatches[i]
    const enc = encoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(d.pipeline)
    for (const bg of d.bindGroups) pass.setBindGroup(bg.index, bg.group)
    pass.dispatchWorkgroups(...d.workgroups)
    pass.end()
    submit([enc.finish()])
  }
  let tokenId = await readToken(device, encoder, submit, tape.copy)
  let pos = position0 + 1

  // Step 2: Generate loop — patch 6 values, dispatch, read
  const STOP = new Set([2, 32000, 32007])
  const tokens: number[] = []

  for (let i = 0; i < maxTokens; i++) {
    pos++

    // 6 patches
    write(tape.writes[0].buffer, tape.writes[0].offset, u32(tokenId))          // token_id
    write(tape.writes[4].buffer, tape.writes[4].offset, u32(pos))              // position
    write(tape.writes[11].buffer, tape.writes[11].offset, u32(pos))            // position dup
    write(tape.writes[12].buffer, tape.writes[12].offset, u32(pos))            // position dup
    write(tape.writes[8].buffer, tape.writes[8].offset, u32(seqBase + (pos - position0)))  // seq counter

    // sampling seed
    if (tape.writes.length > 349) {
      const seed = new ArrayBuffer(4)
      new DataView(seed).setFloat32(0, Math.random(), true)
      write(tape.writes[349].buffer, tape.writes[349].offset, seed)
    }

    // nnz_pages at page boundaries
    const nnz = Math.floor(pos / 16) + 1
    if (nnz !== nnzPages0) {
      for (const idx of attnIndices) write(tape.writes[idx].buffer, tape.writes[idx].offset + 16, u32(nnz))
    }

    // Dispatch all
    for (let d = 0; d < tape.dispatches.length; d++) {
      const disp = tape.dispatches[d]
      const enc = encoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(disp.pipeline)
      for (const bg of disp.bindGroups) pass.setBindGroup(bg.index, bg.group)
      pass.dispatchWorkgroups(...disp.workgroups)
      pass.end()
      submit([enc.finish()])
    }

    // Read token
    tokenId = await readToken(device, encoder, submit, tape.copy)
    if (tokenId < 0 || tokenId >= 32064) break
    tokens.push(tokenId)
    if (_onToken) _onToken(tokenId)
    if (STOP.has(tokenId)) break
  }

  return tokens
}

// ============================================================
// Helpers
// ============================================================

function u32(val: number): ArrayBuffer {
  const ab = new ArrayBuffer(4)
  new DataView(ab).setUint32(0, val, true)
  return ab
}

function readU32(d: Uint8Array | undefined): number {
  if (!d || d.length < 4) return 0
  return d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24)
}

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

async function readToken(
  dev: GPUDevice,
  encoder: () => GPUCommandEncoder,
  submit: (b: Iterable<GPUCommandBuffer>) => void,
  copy: Copy | null,
): Promise<number> {
  const buf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  if (copy) {
    const enc = encoder()
    enc.copyBufferToBuffer(copy.src, copy.srcOffset, buf, 0, copy.size)
    submit([enc.finish()])
  }
  await buf.mapAsync(GPUMapMode.READ)
  const token = new DataView(buf.getMappedRange()).getInt32(0, true)
  buf.unmap()
  buf.destroy()
  return token
}

// ============================================================
// Device patching — capture one token's tape
// ============================================================

export function patch(dev: GPUDevice): void {
  device = dev
  _origSubmit = dev.queue.submit.bind(dev.queue)
  _origWrite = dev.queue.writeBuffer.bind(dev.queue)
  _origEncoder = dev.createCommandEncoder.bind(dev)

  // Capture bind groups
  const prevBG = dev.createBindGroup.bind(dev)
  dev.createBindGroup = function(desc: GPUBindGroupDescriptor) {
    const bg = prevBG(desc)
    bgLayouts.set(bg, desc.layout)
    return bg
  }

  // Prevent buffer destruction
  const prevBuf = dev.createBuffer
  dev.createBuffer = function(desc: GPUBufferDescriptor) {
    const buf = prevBuf.call(dev, desc)
    const realDestroy = buf.destroy.bind(buf)
    buf.destroy = function() {
      if (phase === 'capturing' || phase === 'ready') return
      return realDestroy()
    }
    // Token boundary detection
    if (desc.usage & GPUBufferUsage.MAP_READ) {
      const realMap = buf.mapAsync.bind(buf)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buf as any).mapAsync = function(...args: Parameters<GPUBuffer['mapAsync']>) {
        boundary()
        return realMap(...args)
      }
    }
    return buf
  }

  // Capture writes
  const prevWrite = dev.queue.writeBuffer
  dev.queue.writeBuffer = function(b: GPUBuffer, o: number, d: BufferSource | SharedArrayBuffer, dO?: number, s?: number): undefined {
    if (phase === 'capturing') pendingWrites.push({ buffer: b, offset: o, data: snap(d, dO, s) })
    prevWrite.call(dev.queue, b, o, d, dO, s)
  }

  // Capture dispatches + copies
  const prevEnc = dev.createCommandEncoder
  dev.createCommandEncoder = function(desc?: GPUCommandEncoderDescriptor) {
    const enc = prevEnc.call(dev, desc)
    if (phase !== 'capturing') return enc

    const prevCopy = enc.copyBufferToBuffer.bind(enc)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(enc as any).copyBufferToBuffer = function(...args: unknown[]) {
      if (args.length >= 5) {
        const [s, so, d, d2, sz] = args as [GPUBuffer, number, GPUBuffer, number, number]
        pendingCopy = { src: s, srcOffset: so, dst: d, dstOffset: d2, size: sz }
      }
      return (prevCopy as Function)(...args)
    }

    const prevPass = enc.beginComputePass
    enc.beginComputePass = function(pd?: GPUComputePassDescriptor) {
      const pass = prevPass.call(enc, pd)
      const disp: Dispatch = { pipeline: null as unknown as GPUComputePipeline, bindGroups: [], workgroups: [0, 0, 0] }

      const prevSP = pass.setPipeline.bind(pass)
      pass.setPipeline = function(p: GPUComputePipeline) { disp.pipeline = p; return prevSP(p) }

      const prevSBG = pass.setBindGroup.bind(pass)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(pass as any).setBindGroup = function(idx: number, grp: GPUBindGroup, ...rest: unknown[]) {
        disp.bindGroups.push({ index: idx, group: grp })
        return (prevSBG as Function)(idx, grp, ...rest)
      }

      const prevD = pass.dispatchWorkgroups.bind(pass)
      pass.dispatchWorkgroups = function(x: number, y?: number, z?: number) {
        disp.workgroups = [x, y ?? 1, z ?? 1]
        pendingDispatches.push(disp)
        return prevD(x, y, z)
      }
      return pass
    }
    return enc
  }
}

// ============================================================
// State machine
// ============================================================

function boundary(): void {
  if (phase === 'waiting') {
    warmupCount++
    if (warmupCount === 1) {
      phase = 'capturing'
      pendingWrites = []
      pendingDispatches = []
      pendingCopy = null
    }
  } else if (phase === 'capturing') {
    tape = { dispatches: pendingDispatches, writes: pendingWrites, copy: pendingCopy }
    pendingDispatches = []
    pendingWrites = []
    pendingCopy = null
    warmupCount++
    if (warmupCount >= WARMUP) {
      console.log(`[own-loop] Ready: ${tape.dispatches.length} dispatches, ${tape.writes.length} writes`)
      phase = 'ready'
      if (_onReady) _onReady()
    }
  }
}
