/**
 * ZERO-TVM CHAT — Phi-3 inference. No WebLLM. No TVM.
 *
 * Loads weights from browser cache (cached by a prior WebLLM session)
 * or fetches from HuggingFace. Uses our 10 WGSL shaders only.
 *
 * Correct buffer layout:
 *   B.residual  — persistent running residual (updated in-place by add_norm)
 *   B.hidden1   — normed scratch (matmul input)
 *   B.hidden2   — matmul output scratch (OProj, FFN down)
 */

import { loadWeights, LoadedWeights } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt, Tokenizer } from './tokenizer.js'
import { compile, PHI3 } from '../compiler/compiler.js'

// ============================================================
// GPU helpers
// ============================================================

function createBuf(device: GPUDevice, size: number, usage: number, label?: string): GPUBuffer {
  return device.createBuffer({ size: Math.max(size, 4), usage, label })
}

const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST

function makeBuf(device: GPUDevice, size: number, label: string): GPUBuffer {
  return createBuf(device, size, STORAGE, label)
}

function uniformBuf(device: GPUDevice, data: (number | ArrayBuffer)[]): GPUBuffer {
  const parts: ArrayBuffer[] = data.map(d =>
    d instanceof ArrayBuffer ? d : (() => { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, d, true); return a })()
  )
  const size = parts.reduce((s, p) => s + p.byteLength, 0)
  const padded = Math.ceil(size / 16) * 16
  const buf = device.createBuffer({ size: Math.max(padded, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const arr = new Uint8Array(padded)
  let off = 0
  for (const p of parts) { arr.set(new Uint8Array(p), off); off += p.byteLength }
  device.queue.writeBuffer(buf, 0, arr)
  return buf
}

function u32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setUint32(0, v, true); return a }
function i32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setInt32(0, v, true); return a }
function f32(v: number): ArrayBuffer { const a = new ArrayBuffer(4); new DataView(a).setFloat32(0, v, true); return a }

function bg(device: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[]): GPUBindGroup {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
  })
}

function dispatch(
  enc: GPUCommandEncoder,
  pipeline: GPUComputePipeline,
  bindGroup: GPUBindGroup,
  wgX: number, wgY = 1, wgZ = 1
): void {
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bindGroup)
  pass.dispatchWorkgroups(wgX, wgY, wgZ)
  pass.end()
}

// ============================================================
// KV cache allocation (one pages buffer per layer)
// ============================================================

function allocKVPages(device: GPUDevice): GPUBuffer[] {
  const bytesPerPage = 98304 * 2  // 32 heads * 16 slots * 96 dims * 2 (K+V) * 2 bytes
  const pages = PHI3.MAX_PAGES * bytesPerPage  // 257 * 196608 ≈ 50MB
  return Array.from({ length: PHI3.LAYERS }, (_, i) =>
    makeBuf(device, pages, `kvPages_${i}`)
  )
}

// ============================================================
// Decode engine
// ============================================================

interface DecodeEngine {
  generate(promptIds: number[], maxTokens: number, onToken: (id: number) => void): Promise<number[]>
}

function buildDecodeEngine(
  device: GPUDevice,
  weights: LoadedWeights,
  kvPages: GPUBuffer[]
): DecodeEngine {
  const { pipelines } = compile(device)
  const P = pipelines

  // Activation buffers
  const B = {
    residual:  makeBuf(device, PHI3.D * 2, 'residual'),      // running residual (ping)
    residual2: makeBuf(device, PHI3.D * 2, 'residual2'),     // running residual (pong)
    hidden1:   makeBuf(device, PHI3.D * 2, 'hidden1'),       // normed scratch
    hidden2:   makeBuf(device, PHI3.D * 2, 'hidden2'),       // matmul output scratch
    qkvOut:    makeBuf(device, 9216 * 2, 'qkvOut'),
    qOut:      makeBuf(device, PHI3.D * 2, 'qOut'),
    kOut:      makeBuf(device, PHI3.D * 2, 'kOut'),
    vOut:      makeBuf(device, PHI3.D * 2, 'vOut'),
    attnOut:   makeBuf(device, PHI3.D * 2, 'attnOut'),
    ffnOut:    makeBuf(device, PHI3.FFN * 2, 'ffnOut'),
    logits:    makeBuf(device, PHI3.VOCAB * 4, 'logits'),
    tokenOut:  makeBuf(device, 4, 'tokenOut'),
    inputIds:  makeBuf(device, 4, 'inputIds'),
    posMap:    makeBuf(device, 4, 'posMap'),
    pageIndptr: makeBuf(device, 8, 'pageIndptr'),
    pageValues: makeBuf(device, PHI3.MAX_PAGES * 4, 'pageValues'),
    lengthInfo: makeBuf(device, 12, 'lengthInfo'),
  }

  // Static uniforms
  const qkvU  = uniformBuf(device, [u32(384), u32(96), u32(9216)])
  const oProjU = uniformBuf(device, [u32(384), u32(96), u32(3072)])
  const ffnDnU = uniformBuf(device, [u32(1024), u32(256), u32(3072)])
  const lmHdU  = uniformBuf(device, [u32(384), u32(96), u32(PHI3.VOCAB)])
  const embU   = uniformBuf(device, [u32(1), u32(12)])
  const normU  = uniformBuf(device, [u32(1)])
  const ffnU   = uniformBuf(device, [u32(PHI3.FFN)])
  const argmaxU = uniformBuf(device, [u32(PHI3.VOCAB)])

  // Initialize identity page table (page i → physical page i)
  const pageVals = new Int32Array(PHI3.MAX_PAGES)
  for (let i = 0; i < PHI3.MAX_PAGES; i++) pageVals[i] = i
  device.queue.writeBuffer(B.pageValues, 0, pageVals)

  // Per-layer pre-created bind groups for the attention uniform (varies per position)
  // We create the attention uniform per-call since nnz_pages changes

  const SM_SCALE = 1.0 / Math.sqrt(PHI3.HEAD_DIM)

  async function decodeToken(tokenId: number, position: number): Promise<number> {
    const nnzPages = Math.floor(position / PHI3.PAGE_SIZE) + 1

    // --- Write per-token state ---
    device.queue.writeBuffer(B.inputIds, 0, new Int32Array([tokenId]))
    device.queue.writeBuffer(B.posMap, 0, new Int32Array([position]))
    device.queue.writeBuffer(B.pageIndptr, 0, new Int32Array([0, nnzPages]))
    // length_info: total tokens in sequence = position + 1
    device.queue.writeBuffer(B.lengthInfo, 0, new Int32Array([position + 1, 0, 0]))

    const enc = device.createCommandEncoder()

    // --- EMBEDDING → B.residual (ping) ---
    dispatch(enc, P.embedding, bg(device, P.embedding, [
      B.residual, B.inputIds, weights.embdScales, weights.embdWeights, embU,
    ]), 12)

    // --- INITIAL RMSNORM: B.residual → B.hidden1 (using layer 0's normGamma1) ---
    dispatch(enc, P.rmsNorm, bg(device, P.rmsNorm, [
      B.hidden1, B.residual, weights.layers[0].normGamma1, normU,
    ]), 1)

    // --- 32 TRANSFORMER LAYERS ---
    // Ping-pong between B.residual and B.residual2 so add_norm reads one and writes the other.
    // This avoids WebGPU's same-buffer read+write in a single dispatch.
    let resIn = B.residual   // current residual (read source for add_norm @1)
    let resOut = B.residual2 // next residual    (write dest  for add_norm @4)

    for (let L = 0; L < PHI3.LAYERS; L++) {
      const lw = weights.layers[L]

      // [0] QKV matmul: B.hidden1 → B.qkvOut
      dispatch(enc, P.qkvMatmul, bg(device, P.qkvMatmul, [
        B.qkvOut, B.hidden1, lw.qkvScales, lw.qkvWeights, qkvU,
      ]), 9216)

      // [1] RoPE: B.qkvOut → B.qOut, B.kOut, B.vOut
      const ropeU = uniformBuf(device, [i32(1), i32(0), i32(1), u32(36)])
      dispatch(enc, P.rope, bg(device, P.rope, [
        B.kOut, B.posMap, B.qOut, B.qkvOut, B.vOut, ropeU,
      ]), 36)

      // [2] KV append: kOut, vOut → kvPages[L]
      const kvAppU = uniformBuf(device, [i32(1), i32(PHI3.MAX_PAGES), i32(0), i32(0), u32(12)])
      dispatch(enc, P.kvAppend, bg(device, P.kvAppend, [
        B.kOut, B.vOut, kvPages[L], B.posMap, kvAppU,
      ]), 12)

      // [3] Attention: Q + kvPages[L] → B.attnOut
      const attnU = uniformBuf(device, [
        i32(1), i32(PHI3.MAX_PAGES), i32(nnzPages), i32(0),
        i32(0), i32(0), i32(0), f32(SM_SCALE), u32(1),
      ])
      dispatch(enc, P.attention, bg(device, P.attention, [
        B.qOut, B.pageIndptr, B.pageValues, kvPages[L],
        B.lengthInfo, B.attnOut, attnU,
      ]), 1, PHI3.HEADS)

      // [4] O projection: B.attnOut → B.hidden2
      dispatch(enc, P.oProjMatmul, bg(device, P.oProjMatmul, [
        B.hidden2, B.attnOut, lw.oProjScales, lw.oProjWeights, oProjU,
      ]), 3072)

      // [5] AddNorm (attention): A=hidden2(OProj), B=resIn, out=hidden1, residual=resOut
      //   resOut = A + resIn (raw new residual)
      //   hidden1 = RMSNorm(resOut, normGamma2) for FFN input
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, resIn, lw.normGamma2, B.hidden1, resOut, normU,
      ]), 1)
      ;[resIn, resOut] = [resOut, resIn]  // swap: resIn now has the updated residual

      // [6] Fused FFN gate+up+SiLU: B.hidden1 → B.ffnOut
      dispatch(enc, P.fusedFfn, bg(device, P.fusedFfn, [
        B.ffnOut, B.hidden1, lw.ffnScales, lw.ffnWeights, ffnU,
      ]), PHI3.FFN)

      // [7] FFN down: B.ffnOut → B.hidden2
      dispatch(enc, P.ffnDownMatmul, bg(device, P.ffnDownMatmul, [
        B.hidden2, B.ffnOut, lw.ffnDownScales, lw.ffnDownWeights, ffnDnU,
      ]), 3072)

      // [8] AddNorm (FFN): A=hidden2(FFN down), B=resIn, out=hidden1, residual=resOut
      //   For last layer: nextGamma = finalNormGamma → hidden1 ready for LM head
      const nextGamma = L < PHI3.LAYERS - 1
        ? weights.layers[L + 1].normGamma1
        : weights.finalNormGamma
      dispatch(enc, P.addNorm, bg(device, P.addNorm, [
        B.hidden2, resIn, nextGamma, B.hidden1, resOut, normU,
      ]), 1)
      ;[resIn, resOut] = [resOut, resIn]  // swap again
    }

    // --- LM HEAD: B.hidden1 (already normalized with model.norm) → B.logits ---
    dispatch(enc, P.lmHead, bg(device, P.lmHead, [
      B.logits, B.hidden1, weights.lmHeadScales, weights.lmHeadWeights, lmHdU,
    ]), PHI3.VOCAB)

    // --- ARGMAX: B.logits → B.tokenOut ---
    dispatch(enc, P.argmax, bg(device, P.argmax, [
      B.logits, B.tokenOut, argmaxU,
    ]), 1)

    device.queue.submit([enc.finish()])

    // --- Read token ---
    const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    const readEnc = device.createCommandEncoder()
    readEnc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
    device.queue.submit([readEnc.finish()])

    await readBuf.mapAsync(GPUMapMode.READ)
    const result = new DataView(readBuf.getMappedRange()).getInt32(0, true)
    readBuf.unmap()
    readBuf.destroy()

    return result
  }

  const STOP = new Set([2, 32000, 32007])

  async function generate(
    promptIds: number[],
    maxTokens: number,
    onToken: (id: number) => void
  ): Promise<number[]> {
    const tokens: number[] = []

    // Prefill: process each prompt token through the decode step to populate KV cache
    // Sequential prefill (simplest correct approach)
    for (let i = 0; i < promptIds.length; i++) {
      await decodeToken(promptIds[i], i)
    }

    // First generated token is the output of the last prefill step
    let tokenId = await (async () => {
      const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
      const readEnc = device.createCommandEncoder()
      readEnc.copyBufferToBuffer(B.tokenOut, 0, readBuf, 0, 4)
      device.queue.submit([readEnc.finish()])
      await readBuf.mapAsync(GPUMapMode.READ)
      const id = new DataView(readBuf.getMappedRange()).getInt32(0, true)
      readBuf.unmap()
      readBuf.destroy()
      return id
    })()

    // Decode loop
    let pos = promptIds.length
    for (let i = 0; i < maxTokens; i++) {
      if (tokenId < 0 || tokenId >= PHI3.VOCAB || STOP.has(tokenId)) break
      tokens.push(tokenId)
      onToken(tokenId)
      pos++
      tokenId = await decodeToken(tokenId, pos)
    }

    return tokens
  }

  return { generate }
}

// ============================================================
// Simple UI helpers
// ============================================================

const $ = (id: string) => document.getElementById(id)!

function log(msg: string) {
  const el = $('log') as HTMLPreElement
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

function setBadge(text: string, loading = false) {
  const badge = $('badge')
  badge.textContent = text
  badge.className = loading ? 'badge loading' : 'badge'
}

function setStats(text: string) {
  $('stats').textContent = text
}

function setEnabled(on: boolean) {
  const inp = $('inp') as HTMLInputElement
  const btn = $('btn') as HTMLButtonElement
  inp.disabled = !on
  btn.disabled = !on
  if (on) inp.focus()
}

function addMsg(role: 'user' | 'ai'): HTMLElement {
  const chat = $('chat')
  const div = document.createElement('div')
  div.className = `msg ${role}`
  chat.appendChild(div)
  chat.scrollTop = chat.scrollHeight
  return div
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  setBadge('Initializing...', true)
  log('Zero-TVM Phi-3 — No WebLLM, No TVM')
  log('10 hand-written WGSL shaders + HuggingFace weights')
  log('')

  if (!navigator.gpu) {
    setBadge('No WebGPU'); log('ERROR: WebGPU not supported'); return
  }

  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) {
    setBadge('No GPU adapter'); log('ERROR: No GPU adapter'); return
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16' as GPUFeatureName],
  })
  log(`GPU ready`)

  // Tokenizer
  setBadge('Loading tokenizer...', true)
  let tokenizer: Tokenizer
  try {
    tokenizer = await loadTokenizer((msg) => log(msg))
  } catch (e) {
    setBadge('Tokenizer failed'); log(`ERROR: ${e}`); return
  }

  // Weights
  setBadge('Loading weights...', true)
  log('')
  log('Loading weights (checks browser cache first)...')
  let weights: LoadedWeights
  try {
    weights = await loadWeights(device, (msg) => {
      log(msg)
      if (msg.startsWith('Loading layer')) setBadge(msg, true)
    })
  } catch (e) {
    setBadge('Weight load failed')
    log(`ERROR loading weights: ${e}`)
    log('')
    log('Tip: Visit compiler-chat.html first to cache the model via WebLLM.')
    return
  }

  // KV cache
  log('Allocating KV cache...')
  const kvPages = allocKVPages(device)

  // Build engine
  log('Building decode engine...')
  const engine = buildDecodeEngine(device, weights, kvPages)

  log('')
  log('Ready. Zero TVM. 10 WGSL shaders.')
  setBadge('Ready — Zero TVM')
  setEnabled(true)

  const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: 'You are a helpful assistant.' },
  ]
  let generating = false

  async function send(): Promise<void> {
    if (generating) return
    const inp = $('inp') as HTMLInputElement
    const text = inp.value.trim()
    if (!text) return
    inp.value = ''

    generating = true
    setEnabled(false)

    const userEl = addMsg('user')
    userEl.textContent = text
    const aiEl = addMsg('ai')

    history.push({ role: 'user', content: text })

    const promptIds = buildChatPrompt(history, tokenizer)
    log(`Prompt: ${promptIds.length} tokens`)

    const t0 = performance.now()
    let count = 0
    const allIds: number[] = []
    let prevText = ''

    try {
      await engine.generate(promptIds, 500, (id) => {
        count++
        allIds.push(id)
        const full = tokenizer.decode(allIds)
        const delta = full.slice(prevText.length)
        if (delta) { aiEl.textContent += delta; $('chat').scrollTop = $('chat').scrollHeight }
        prevText = full
        const elapsed = (performance.now() - t0) / 1000
        setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s`)
      })

      const fullResponse = tokenizer.decode(allIds)
      history.push({ role: 'assistant', content: fullResponse })
    } catch (e) {
      aiEl.textContent += `\n[Error: ${e}]`
      log(`Error: ${e}`)
    }

    generating = false
    setEnabled(true)
  }

  $('btn').addEventListener('click', send)
  ;($('inp') as HTMLInputElement).addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  })
}

main().catch((e) => { console.error(e); log(`FATAL: ${e}`) })
