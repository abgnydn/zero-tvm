/**
 * ENGINE v3 — Minimal WebGPU LLM inference. ~100 lines.
 *
 * No TVM runtime. No pipeline registry. No step names.
 * Just two dispatch lists (prefill + decode) and 6 changing writes.
 *
 * Built from 100% reverse engineering of TVM's architecture:
 * - 343 prefill dispatches (batch) → 342 decode dispatches (single token)
 * - 3 scratch buffers ping-pong between layers
 * - 6 values change per token, everything else is static
 */

import { CaptureResult, setPositionMapBuffer } from './capture.js'

interface Dispatch {
  pipeline: GPUComputePipeline
  bg: GPUBindGroup
  wg: [number, number, number]
}

export function buildEngine(cap: CaptureResult) {
  const dev = cap.device

  // Build bind groups from captured dispatch entries
  const makeBGs = (dispatches: typeof cap.dispatches) =>
    dispatches.map(d => ({
      pipeline: d.pipeline,
      bg: dev.createBindGroup({
        layout: d.pipeline.getBindGroupLayout(0),
        entries: d.entries.map(e => ({
          binding: e.binding,
          resource: { buffer: e.buffer, offset: e.offset, size: e.size },
        })),
      }),
      wg: d.workgroups,
    } as Dispatch))

  const decode = makeBGs(cap.dispatches)
  const prefill = makeBGs(cap.prefillDispatches)

  // Token readback source (from captured copy operation)
  const copySrc = cap.copy!.src
  const copySrcOff = cap.copy!.srcOffset
  const copySz = cap.copy!.size

  // The 6 decode writes that change per token (identified by index in captured writes)
  const w = cap.writes
  const tokenIdBuf = w[0]                    // write[0]: token ID for embedding
  const posMapBuf = w[4]                     // write[4]: position in RoPE position_map

  // Register the exact position_map buffer for tracking across TVM calls
  setPositionMapBuffer(posMapBuf.buffer)
  const seqCounterBuf = w[8]                 // write[8]: sequence counter
  const qRopePosBuf = w[11]                  // write[11]: q_rope_position
  const lengthInfoBuf = w[12]                // write[12]: length_info (seq length)
  const seedBuf = w.length > 349 ? w[349] : null  // write[349]: sampling seed

  // Decode initial position from captured state
  const read32 = (d: Uint8Array) => d[0] | (d[1] << 8) | (d[2] << 16) | (d[3] << 24)
  const pos0 = read32(w[4]?.data ?? new Uint8Array(4))
  const seq0 = read32(w[8]?.data ?? new Uint8Array(4))

  // nnz_pages uniform offset — attention dispatches (every 10th starting at index 5)
  // Each has a 56B podArgs with nnz_pages at byte offset 16
  const attnUniformBufs: GPUBuffer[] = []
  for (let L = 0; L < 32; L++) {
    const d = cap.dispatches[2 + L * 10 + 3] // attention is step 3 in decode
    const uniformEntry = d.entries.find(e => e.buffer.size === 64)
    if (uniformEntry) attnUniformBufs.push(uniformEntry.buffer)
  }

  // Helpers
  const u32buf = (v: number) => { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, v, true); return a }
  const f32buf = (v: number) => { const a = new ArrayBuffer(4); new DataView(a).setFloat32(0, v, true); return a }

  function run(list: Dispatch[]): void {
    const enc = dev.createCommandEncoder()
    for (const d of list) {
      const pass = enc.beginComputePass()
      pass.setPipeline(d.pipeline)
      pass.setBindGroup(0, d.bg)
      pass.dispatchWorkgroups(...d.wg)
      pass.end()
    }
    dev.queue.submit([enc.finish()])
  }

  async function readToken(): Promise<number> {
    const buf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = dev.createCommandEncoder()
    enc.copyBufferToBuffer(copySrc, copySrcOff, buf, 0, copySz)
    dev.queue.submit([enc.finish()])
    await buf.mapAsync(GPUMapMode.READ)
    const id = new DataView(buf.getMappedRange()).getInt32(0, true)
    buf.unmap()
    buf.destroy()
    return id
  }

  function writeDecodeState(tokenId: number, pos: number, seqIdx: number): void {
    dev.queue.writeBuffer(tokenIdBuf.buffer, tokenIdBuf.offset, u32buf(tokenId))
    dev.queue.writeBuffer(posMapBuf.buffer, posMapBuf.offset, u32buf(pos))
    dev.queue.writeBuffer(seqCounterBuf.buffer, seqCounterBuf.offset, u32buf(seqIdx))
    dev.queue.writeBuffer(qRopePosBuf.buffer, qRopePosBuf.offset, u32buf(pos))
    dev.queue.writeBuffer(lengthInfoBuf.buffer, lengthInfoBuf.offset, u32buf(pos))
    if (seedBuf) dev.queue.writeBuffer(seedBuf.buffer, seedBuf.offset, f32buf(Math.random()))

    // Update nnz_pages when crossing a page boundary (page_size=16)
    const nnz = Math.floor(pos / 16) + 1
    for (const buf of attnUniformBufs) dev.queue.writeBuffer(buf, 16, u32buf(nnz))
  }

  const STOP = new Set([2, 32000, 32007])

  /**
   * Generate from captured state (message 1 only).
   * Replays all captured writes to establish activation state.
   */
  async function generate(maxTokens: number, onToken?: (id: number) => void): Promise<number[]> {
    for (const wr of w) dev.queue.writeBuffer(wr.buffer, wr.offset, wr.data.slice().buffer)
    run(decode)
    let tokenId = await readToken()

    let pos = pos0 + 1
    const tokens: number[] = []

    for (let i = 0; i < maxTokens; i++) {
      pos++
      writeDecodeState(tokenId, pos, seq0 + (pos - pos0))
      run(decode)
      tokenId = await readToken()
      if (tokenId < 0 || tokenId >= 32064 || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken?.(tokenId)
    }
    return tokens
  }

  /**
   * Continue decoding after TVM has already prefilled.
   * Does NOT replay captured writes — TVM's prefill set up the correct state.
   * Just dispatches decode kernels and reads tokens.
   */
  async function continueAfterPrefill(startPos: number, maxTokens: number, onToken?: (id: number) => void): Promise<number[]> {
    // TVM already set up uniforms + activations for first decode token
    run(decode)
    let tokenId = await readToken()

    let pos = startPos
    const tokens: number[] = []

    if (tokenId >= 0 && tokenId < 32064 && !STOP.has(tokenId)) {
      tokens.push(tokenId)
      onToken?.(tokenId)
    }

    for (let i = 0; i < maxTokens - 1; i++) {
      pos++
      writeDecodeState(tokenId, pos, i + 2)
      run(decode)
      tokenId = await readToken()
      if (tokenId < 0 || tokenId >= 32064 || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken?.(tokenId)
    }
    return tokens
  }

  console.log(`[engine-v3] Ready: ${decode.length} decode + ${prefill.length} prefill dispatches`)
  return { generate, continueAfterPrefill, decode, prefill }
}
