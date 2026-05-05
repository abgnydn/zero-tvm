/**
 * COMPILER CHAT v2 — Multi-token generation using our verified chain.
 *
 * Message 1: WebLLM generates (captures dispatches + weights)
 * Message 2+: Our engine generates tokens in a loop.
 *
 * Architecture: 279 of 342 dispatches use our shaders.
 * Only KV append + attention (64 dispatches) use TVM's compiled shaders.
 * TVM's runtime is completely eliminated.
 */

import { LLMEngine, MODELS } from '../engine.js'
import { ChatUI } from '../ui.js'
import { patchForCapture, getCaptureResult, CaptureResult } from '../capture.js'

import int4MatmulSrc from './shaders/int4_matmul.wgsl?raw'
import int4MatmulF32Src from './shaders/int4_matmul_f32.wgsl?raw'
import rmsNormSrc from './shaders/rms_norm.wgsl?raw'
import addNormSrc from './shaders/add_norm.wgsl?raw'
import ropeSrc from './shaders/rope.wgsl?raw'
import fusedFfnSrc from './shaders/fused_ffn.wgsl?raw'
import embeddingSrc from './shaders/embedding.wgsl?raw'
import argmaxSrc from './shaders/argmax.wgsl?raw'
import kvAppendSrc from './shaders/kv_append.wgsl?raw'
import attentionSrc from './shaders/attention.wgsl?raw'

// ============================================================
// Helpers
// ============================================================

function makePipeline(dev: GPUDevice, src: string, entry: string): GPUComputePipeline {
  return dev.createComputePipeline({ layout: 'auto', compute: { module: dev.createShaderModule({ code: src }), entryPoint: entry } })
}

function makeUniform(dev: GPUDevice, values: number[]): GPUBuffer {
  const buf = dev.createBuffer({ size: Math.max(values.length * 4, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  dev.queue.writeBuffer(buf, 0, new Uint32Array(values))
  return buf
}

function makeBuf(dev: GPUDevice, size: number): GPUBuffer {
  return dev.createBuffer({ size: Math.max(size, 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
}

function dispatchShader(dev: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[], wgX: number): void {
  const bg = dev.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b, offset: 0, size: b.size } })),
  })
  const enc = dev.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(wgX)
  pass.end()
  dev.queue.submit([enc.finish()])
}

function writeU32(dev: GPUDevice, buf: GPUBuffer, offset: number, val: number): void {
  dev.queue.writeBuffer(buf, offset, new Uint32Array([val]))
}

function writeF32(dev: GPUDevice, buf: GPUBuffer, offset: number, val: number): void {
  dev.queue.writeBuffer(buf, offset, new Float32Array([val]))
}

function readU32(data: Uint8Array): number {
  return data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)
}

// ============================================================
// Engine builder
// ============================================================

interface OurEngine {
  generate(promptTokenIds: number[], maxTokens: number, onToken: (id: number) => void): Promise<number[]>
  continueFromCapture(maxTokens: number, onToken: (id: number) => void): Promise<number[]>
}

function buildOurEngine(cap: CaptureResult): OurEngine {
  const dev = cap.device
  const w = cap.writes

  // Create our pipelines
  const P = {
    embed: makePipeline(dev, embeddingSrc, 'embedding'),
    norm: makePipeline(dev, rmsNormSrc, 'rms_norm'),
    matmul: makePipeline(dev, int4MatmulSrc, 'int4_matmul'),
    rope: makePipeline(dev, ropeSrc, 'rope_kernel'),
    addNorm: makePipeline(dev, addNormSrc, 'add_norm'),
    fusedFfn: makePipeline(dev, fusedFfnSrc, 'fused_ffn_kernel'),
    lmHead: makePipeline(dev, int4MatmulF32Src, 'int4_matmul_f32'),
    argmax: makePipeline(dev, argmaxSrc, 'argmax_kernel'),
    kvAppend: makePipeline(dev, kvAppendSrc, 'kv_append'),
    attention: makePipeline(dev, attentionSrc, 'attention'),
  }

  const tokenOut = makeBuf(dev, 4)
  let currentPos = 0  // tracked for attention nnz_pages calculation

  // All per-token write targets (from dump analysis)
  // [0]  token_id (4B)
  // [4]  position_map (4B in 4.1KB buf)
  // [5]  page_indptr [0, nnz] (8B)
  // [6]  page_values_offset [0, nnz] (8B)
  // [7]  length_info [kv_len, 0, 0] or [pos, seq_start, sliding_window_start] (12B in 1KB buf)
  // [8]  seq_counter (4B)
  // [10] kv_append ntoken/pos [0, ntoken] (8B)
  // [11] q_rope_position (4B)
  // [12] length_info for attention (4B in 4.1KB buf)
  // [13] init value = -1 (4B)
  // [14] init value = -1 (4B)
  // [349] sampling seed (f32)
  //
  // Plus: nnz_pages field at offset 16 in each attention uniform buffer (32 × 64B bufs)

  const tokenIdBuf    = w[0]   // BUF#728
  const posMapBuf     = w[4]   // BUF#13
  const pageIndptrBuf = w[5]   // BUF#1081
  const pageOffsetBuf = w[6]   // BUF#738
  const lengthInfoBuf = w[7]   // BUF#739 (1KB, 12B write)
  const seqCounterBuf = w[8]   // BUF#737
  const kvNtokenBuf   = w[10]  // BUF#18
  const qRopePosBuf   = w[11]  // BUF#17
  const attnLenBuf    = w[12]  // BUF#22
  // w[13] = BUF#1082 init=-1, w[14] = BUF#1083 init=-1 (replayed via staticWrites)
  const seedBuf       = w.length > 349 ? w[349] : null

  // Copy source for token readback
  const copySrc = cap.copy!.src
  const copySrcOff = cap.copy!.srcOffset

  async function readToken(): Promise<number> {
    const buf = dev.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const enc = dev.createCommandEncoder()
    enc.copyBufferToBuffer(copySrc, copySrcOff, buf, 0, 4)
    dev.queue.submit([enc.finish()])
    await buf.mapAsync(GPUMapMode.READ)
    const id = new DataView(buf.getMappedRange()).getInt32(0, true)
    buf.unmap()
    buf.destroy()
    return id
  }

  function forwardPass(): void {
    for (let i = 0; i < cap.dispatches.length; i++) {
      const d = cap.dispatches[i]
      const step = i >= 2 && i < 322 ? (i - 2) % 10 : -1

      if (i === 0) {
        // Embedding
        const u = makeUniform(dev, [1, 12])
        dispatchShader(dev, P.embed, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 12)
      } else if (i === 1) {
        // RMSNorm
        const u = makeUniform(dev, [1])
        dispatchShader(dev, P.norm, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, u], 1)
      } else if (step === 0) {
        // QKV matmul
        const u = makeUniform(dev, [384, 96, 9216])
        dispatchShader(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 9216)
      } else if (step === 1) {
        // RoPE (remap bindings)
        const u = makeUniform(dev, [1, 0, 1, 36])
        dispatchShader(dev, P.rope, [d.entries[2].buffer, d.entries[0].buffer, d.entries[4].buffer, d.entries[3].buffer, d.entries[1].buffer, u], 36)
      } else if (step === 2) {
        // KV Append — OUR shader
        // TVM bindings: 0=k_data, 1=pages, 2=position_map, 3=v_data, 4=uniform
        // Our bindings:  0=k_data, 1=v_data, 2=pages, 3=position_map, 4=uniform
        const kvU = makeUniform(dev, [1, 0, 0, 0, 12])  // ntoken=1, num_pages=0, pages/posmap offsets=0, packGridDimX=12
        const kvBG = dev.createBindGroup({
          layout: P.kvAppend.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: d.entries[0].buffer, offset: d.entries[0].offset, size: d.entries[0].size } },
            { binding: 1, resource: { buffer: d.entries[3].buffer, offset: d.entries[3].offset, size: d.entries[3].size } },
            { binding: 2, resource: { buffer: d.entries[1].buffer, offset: d.entries[1].offset, size: d.entries[1].size } },
            { binding: 3, resource: { buffer: d.entries[2].buffer, offset: d.entries[2].offset, size: d.entries[2].size } },
            { binding: 4, resource: { buffer: kvU, offset: 0, size: kvU.size } },
          ],
        })
        const kvEnc = dev.createCommandEncoder()
        const kvPass = kvEnc.beginComputePass()
        kvPass.setPipeline(P.kvAppend)
        kvPass.setBindGroup(0, kvBG)
        kvPass.dispatchWorkgroups(12)  // ceil(1 * 3072 / 256) = 12
        kvPass.end()
        dev.queue.submit([kvEnc.finish()])
      } else if (step === 3) {
        // Attention — OUR shader
        // TVM bindings: 0=Q, 1=k_rope_off, 2=length_info, 3=lse, 4=output, 5=page_indptr, 6=page_values, 7=pages, 8=q_rope_pos, 9=uniform
        // Our bindings:  0=Q, 1=page_table_indptr, 2=page_table_values, 3=pages, 4=length_info, 5=output, 6=uniform
        const smScale = 1.0 / Math.sqrt(96)
        const attnUBuf = dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
        const attnUData = new ArrayBuffer(36)
        const attnDV = new DataView(attnUData)
        attnDV.setInt32(0, 1, true)       // B = 1
        attnDV.setInt32(4, 0, true)       // max_num_pages (unused by shader)
        attnDV.setInt32(8, Math.floor(currentPos / 16) + 1, true)  // nnz_pages
        attnDV.setInt32(12, 0, true)      // pages_elem_offset
        attnDV.setInt32(16, 0, true)      // page_indptr_elem_offset
        attnDV.setInt32(20, 0, true)      // page_values_elem_offset
        attnDV.setInt32(24, 0, true)      // length_info_elem_offset
        attnDV.setFloat32(28, smScale, true)  // sm_scale
        attnDV.setUint32(32, 31, true)    // packGridDimX (unused)
        dev.queue.writeBuffer(attnUBuf, 0, attnUData)

        const attnBG = dev.createBindGroup({
          layout: P.attention.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: d.entries[0].buffer, offset: d.entries[0].offset, size: d.entries[0].size } },
            { binding: 1, resource: { buffer: d.entries[5].buffer, offset: d.entries[5].offset, size: d.entries[5].size } },
            { binding: 2, resource: { buffer: d.entries[6].buffer, offset: d.entries[6].offset, size: d.entries[6].size } },
            { binding: 3, resource: { buffer: d.entries[7].buffer, offset: d.entries[7].offset, size: d.entries[7].size } },
            { binding: 4, resource: { buffer: d.entries[2].buffer, offset: d.entries[2].offset, size: d.entries[2].size } },
            { binding: 5, resource: { buffer: d.entries[4].buffer, offset: d.entries[4].offset, size: d.entries[4].size } },
            { binding: 6, resource: { buffer: attnUBuf, offset: 0, size: 64 } },
          ],
        })
        const attnEnc = dev.createCommandEncoder()
        const attnPass = attnEnc.beginComputePass()
        attnPass.setPipeline(P.attention)
        attnPass.setBindGroup(0, attnBG)
        attnPass.dispatchWorkgroups(1, 32, 1)  // batch=1, heads=32
        attnPass.end()
        dev.queue.submit([attnEnc.finish()])
      } else if (step === 4) {
        // OProj
        const u = makeUniform(dev, [384, 96, 3072])
        dispatchShader(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
      } else if (step === 5 || step === 9) {
        // AddNorm
        const u = makeUniform(dev, [1])
        dispatchShader(dev, P.addNorm, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, d.entries[4].buffer, u], 1)
      } else if (step === 6) {
        // Fused FFN (replaces FFNUp + SiLU)
        const siluOutBuf = cap.dispatches[i + 1].entries[0].buffer
        const inputCopy = makeBuf(dev, d.entries[1].buffer.size)
        const cpEnc = dev.createCommandEncoder()
        cpEnc.copyBufferToBuffer(d.entries[1].buffer, 0, inputCopy, 0, d.entries[1].buffer.size)
        dev.queue.submit([cpEnc.finish()])
        const u = makeUniform(dev, [8192])
        dispatchShader(dev, P.fusedFfn, [siluOutBuf, inputCopy, d.entries[2].buffer, d.entries[3].buffer, u], 8192)
      } else if (step === 7) {
        // SiLU: skip (handled by fused FFN)
      } else if (step === 8) {
        // FFN Down
        const u = makeUniform(dev, [1024, 256, 3072])
        dispatchShader(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
      } else if (i === 322) {
        // LM Head (remap bindings)
        const u = makeUniform(dev, [384, 96, 32064])
        dispatchShader(dev, P.lmHead, [d.entries[0].buffer, d.entries[3].buffer, d.entries[1].buffer, d.entries[2].buffer, u], 32064)
      } else {
        // Tail sampling dispatches: skip (replaced by argmax)
      }
    }

    // Argmax
    const lmBuf = cap.dispatches[322].entries[0].buffer
    dispatchShader(dev, P.argmax, [lmBuf, tokenOut, makeUniform(dev, [32064])], 1)
  }

  const STOP = new Set([2, 32000, 32007])

  function writeState(tokenId: number, pos: number): void {
    currentPos = pos
    const nnz = Math.floor(pos / 16) + 1
    const lastPageLen = pos % 16

    writeU32(dev, tokenIdBuf.buffer, tokenIdBuf.offset, tokenId)
    writeU32(dev, posMapBuf.buffer, posMapBuf.offset, pos)
    writeU32(dev, seqCounterBuf.buffer, seqCounterBuf.offset, pos + 1)
    writeU32(dev, qRopePosBuf.buffer, qRopePosBuf.offset, pos)
    writeU32(dev, attnLenBuf.buffer, attnLenBuf.offset, pos)
    if (seedBuf) writeF32(dev, seedBuf.buffer, seedBuf.offset, Math.random())

    // Page table
    dev.queue.writeBuffer(pageIndptrBuf.buffer, pageIndptrBuf.offset, new Int32Array([0, nnz]))
    dev.queue.writeBuffer(pageOffsetBuf.buffer, pageOffsetBuf.offset, new Int32Array([0, nnz]))

    // length_info: [last_page_len, 0, 0] — 3 values for batch_decode
    dev.queue.writeBuffer(lengthInfoBuf.buffer, lengthInfoBuf.offset, new Int32Array([lastPageLen, 0, 0]))

    // KV append: ntoken offset
    dev.queue.writeBuffer(kvNtokenBuf.buffer, kvNtokenBuf.offset, new Int32Array([0, 1]))
  }

  // Classify writes: only replay UNIFORM writes (static config), not activations
  // From the dump: writes to buffers <=64B are uniforms, larger are activations
  const staticWrites = w.filter(wr => wr.buffer.size <= 64)

  // Page value table managed by TVM's KV cache

  /** Sequential prefill: run each prompt token through the decode pipeline */
  async function prefill(tokenIds: number[]): Promise<void> {
    // Replay only static uniform writes (pipeline configs)
    for (const wr of staticWrites) dev.queue.writeBuffer(wr.buffer, wr.offset, wr.data.slice().buffer)

    // Initialize page_table_values: pages 0,1,2,... (identity mapping)
    // TVM's attention reads page_table_values[page_indptr[batch]..page_indptr[batch+1]]
    // For sequential: page i maps to page i
    const pageValsArr = new Int32Array(257)
    for (let i = 0; i < 257; i++) pageValsArr[i] = i
    // Find the page_table_values buffer from attention dispatch
    // attnD would be cap.dispatches[5] but we handle page table via writeState
    // batch_decode_paged_kv: @6=page_table_values
    // Actually from dump: @1=k_rope_pos_offset, @2=length_info, @3=page_indptr,
    //   @4=output, @5=page_table_indptr, @6=temp, @7=kv_pages, @8=position_map
    // Let me find it from the dispatch entries by size
    // page_table_values buffer should be ~1KB (257 × 4B = 1028B)

    // Actually the page table is managed by TVM's KV cache system.
    // The page_indptr and page_values buffers are written by w[5] and w[6].
    // w[5] = BUF#1081 8B = [0, nnz_pages] — this IS page_indptr
    // w[6] = BUF#738 8B = [0, nnz_pages] — this is page offset
    // But the actual page_table_values (which pages are allocated) is a separate buffer.

    // From dump, attention dispatch [5] has:
    //   @3=BUF#19 KV_CACHE 131.1KB = page_table (indptr + values packed)
    // Wait no — BUF#19 is 131KB, that's the page table mapping buffer.
    // Let me look at what TVM wrote to it during prefill.

    // Actually, the page table is set up during TVM's PREFILL (message 1).
    // For message 2, the KV cache pages still contain message 1's data.
    // The attention kernel will read those old K,V values.

    // The REAL fix: we need to NOT attend to old positions.
    // The length_info controls how many positions attention scans.
    // If we set length_info correctly, attention only reads positions 0..pos.
    // But the KV CACHE PAGES at those positions contain message 1's K,V.
    // Our KV append OVERWRITES position 0,1,2... with new K,V.
    // So after prefill token 0: page 0, slot 0 has new K,V.
    // Attention at token 0 reads slot 0 → correct (just written).
    // Token 1: page 0, slot 1 has NEW K,V. But slot 0 was overwritten.
    // This should work IF we set length correctly.

    console.log(`[prefill] Starting with ${tokenIds.length} tokens`)

    for (let i = 0; i < tokenIds.length; i++) {
      writeState(tokenIds[i], i)
      forwardPass()
      await dev.queue.onSubmittedWorkDone()
    }

    console.log(`[prefill] Done, pos=${tokenIds.length}`)
  }

  async function generate(promptTokenIds: number[], maxTokens: number, onToken: (id: number) => void): Promise<number[]> {
    console.log(`[generate] prompt=${promptTokenIds.length} tokens: [${promptTokenIds.slice(0, 10).join(',')}...]`)

    // Prefill: sequential, one token at a time through our decode chain
    await prefill(promptTokenIds)

    // First generated token is already computed by last prefill step
    let tokenId = await readToken()
    console.log(`[generate] first token=${tokenId}`)
    let pos = promptTokenIds.length
    const tokens: number[] = []

    for (let i = 0; i < maxTokens; i++) {
      if (tokenId < 0 || tokenId >= 32064 || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken(tokenId)

      pos++
      writeState(tokenId, pos)
      forwardPass()
      tokenId = await readToken()
    }

    return tokens
  }

  /** Continue decoding from captured state (no prefill, just decode loop) */
  async function continueFromCapture(maxTokens: number, onToken: (id: number) => void): Promise<number[]> {
    // Replay all captured writes (restores exact state from capture)
    for (const wr of w) dev.queue.writeBuffer(wr.buffer, wr.offset, wr.data.slice().buffer)
    currentPos = readU32(w[4]?.data ?? new Uint8Array(4))
    forwardPass()
    let tokenId = await readToken()
    console.log(`[continueFromCapture] first token=${tokenId}, pos=${currentPos}`)

    let pos = currentPos + 1
    const tokens: number[] = []

    for (let i = 0; i < maxTokens; i++) {
      if (tokenId < 0 || tokenId >= 32064 || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken(tokenId)

      pos++
      writeState(tokenId, pos)
      forwardPass()
      tokenId = await readToken()
    }
    return tokens
  }

  return { generate, continueFromCapture }
}

// ============================================================
// Chat UI
// ============================================================

async function registerWeightsSW(): Promise<void> {
  if (!('serviceWorker' in navigator)) return
  try {
    await navigator.serviceWorker.register('/weights-cache-sw.js', { scope: '/' })
    await navigator.serviceWorker.ready
  } catch (e) {
    console.warn('[compiler-chat] weight-cache SW registration failed:', e)
  }
}

async function hasSharedWeightsCached(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return false
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle('zero-tvm-weights')
    await dir.getFileHandle('ndarray-cache.json')
    return true
  } catch {
    return false
  }
}

async function main(): Promise<void> {
  const ui = new ChatUI()
  const llm = new LLMEngine()

  // Register the shared weight-cache SW first. WebLLM's fetches to the
  // Phi-3 mirror are intercepted and served from / written to the same
  // OPFS dir zero-tvm.html uses, so visiting both pages downloads once.
  await registerWeightsSW()

  // Wait for user to click "Download & Start" before fetching ~2 GB of weights.
  // If the shared cache already has weights (e.g. the visitor came from
  // zero-tvm.html), skip the gate — load is instant.
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null
  const cached = await hasSharedWeightsCached()
  if (startBtn && !cached) {
    await new Promise<void>(resolve => {
      startBtn.addEventListener('click', () => {
        startBtn.disabled = true
        startBtn.textContent = 'Starting...'
        resolve()
      })
    })
    const startScreen = document.getElementById('start-screen')
    if (startScreen) startScreen.remove()
  } else if (cached) {
    // Cached path — drop the gate immediately.
    const startScreen = document.getElementById('start-screen')
    if (startScreen) startScreen.remove()
  }

  ui.appendLog('Loading model...')

  if (navigator.gpu) {
    const origRA = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function(...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRA(...args)
      if (!adapter) return adapter
      const origRD = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function(...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRD(...dArgs)
        patchForCapture(device)
        return device
      }
      return adapter
    }
  }

  await llm.load(MODELS.PHI3_MINI_Q4, (msg) => {
    ui.setBadge(msg, true)
    ui.appendLog(msg)
  })

  const tokenizer = llm.getTokenizer()
  ui.setBadge('Ready')
  ui.setEnabled(true)
  ui.setInitialMessage('Compiler Chat\n342/342 dispatches = my shaders\nMsg 1: TVM captures\nMsg 2+: My engine')

  let engine: OurEngine | null = null
  let generating = false

  async function send(): Promise<void> {
    if (generating) return
    const text = ui.consumeInput()
    if (!text) return

    generating = true
    ui.setEnabled(false)
    ui.addUserMessage(text)
    ui.startAiMessage()

    const t0 = performance.now()
    let count = 0
    const stats = () => {
      const s = (performance.now() - t0) / 1000
      ui.setStats(`${count} tok | ${(count / s).toFixed(1)} tok/s | ${s.toFixed(1)}s`)
    }

    try {
      if (!engine) {
        // Message 1: TVM generates first token only (to capture), then our engine takes over
        ui.appendLog('TVM generating first token (capturing)...')

        // Let TVM generate 2 tokens: prefill produces token 1, decode produces token 2
        // We need at least 1 decode pass to capture the decode dispatch graph
        let tvmTokens = 0
        await llm.chat(text, (tok) => {
          tvmTokens++
          ui.appendToken(tok)
          if (tvmTokens >= 2) throw new Error('capture done')
        }).catch(() => {}) // expected abort

        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0 && cap.writes.length > 0 && cap.copy) {
          engine = buildOurEngine(cap)
          ui.appendLog(`Our engine ready. Continuing with our engine...`)

          // Continue generating with our engine (KV cache has TVM's prefill + 1 decode token)
          if (tokenizer) {
            const allIds: number[] = []
            let prevText = ''
            await engine.continueFromCapture(500, (id) => {
              count++
              allIds.push(id)
              const full = tokenizer.decode(new Int32Array(allIds))
              ui.appendToken(full.slice(prevText.length))
              prevText = full
              stats()
            })
          }
        }
      } else if (tokenizer) {
        // Message 2+: our engine does everything — prefill + decode
        // Phi-3 chat template: <|system|>...<|end|><|user|>...<|end|><|assistant|>
        const chatPrompt = `<|system|>You are a helpful assistant.<|end|>\n<|user|>${text}<|end|>\n<|assistant|>\n`
        const promptIds = Array.from(tokenizer.encode(chatPrompt))
        ui.appendLog(`Encoding ${promptIds.length} tokens, our engine prefill + decode...`)

        const allIds: number[] = []
        let prevText = ''
        await engine.generate(promptIds, 500, (id) => {
          count++
          allIds.push(id)
          const full = tokenizer.decode(new Int32Array(allIds))
          ui.appendToken(full.slice(prevText.length))
          prevText = full
          stats()
        })
      } else {
        await llm.chat(text, (tok) => { count++; ui.appendToken(tok); stats() })
      }
    } catch (e) {
      ui.setError(e instanceof Error ? e.message : String(e))
      ui.appendLog(`Error: ${e}`)
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
