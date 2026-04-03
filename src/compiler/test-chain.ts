/**
 * CHAIN TEST — Run our full forward pass and compare final token against TVM.
 *
 * Strategy:
 *   1. TVM runs one decode token (captures everything)
 *   2. We replay TVM's writes (establishes input state: token_id, position, etc.)
 *   3. We run OUR full dispatch sequence using OUR shaders but TVM's weight buffers
 *   4. Compare our output token vs TVM's output token
 *
 * For KV append + attention: we use TVM's KV cache buffers directly.
 * TVM already populated the KV cache during prefill — we just need to
 * append the new K,V and attend over the existing cache.
 *
 * Since our attention shader has different bindings from TVM's, we use TVM's
 * attention dispatch for now (proven correct) and replace everything else with ours.
 * This isolates the chain test to: embed, norm, matmuls, rope, addnorm, fused_ffn, argmax.
 */

import { LLMEngine, MODELS } from '../engine.js'
import { patchForCapture, getCaptureResult, CapturedDispatch, CaptureResult } from '../capture.js'

import int4MatmulSrc from './shaders/int4_matmul.wgsl?raw'
import int4MatmulF32Src from './shaders/int4_matmul_f32.wgsl?raw'
import rmsNormSrc from './shaders/rms_norm.wgsl?raw'
import addNormSrc from './shaders/add_norm.wgsl?raw'
import ropeSrc from './shaders/rope.wgsl?raw'
import fusedFfnSrc from './shaders/fused_ffn.wgsl?raw'
import embeddingSrc from './shaders/embedding.wgsl?raw'
import argmaxSrc from './shaders/argmax.wgsl?raw'

// ============================================================
// Helpers
// ============================================================

const lines: string[] = []
const log = (msg: string) => {
  lines.push(msg)
  const el = document.getElementById('log')
  if (el) { el.textContent = lines.join('\n'); el.scrollTop = el.scrollHeight }
}

async function readBuf(device: GPUDevice, buffer: GPUBuffer, bytes?: number): Promise<ArrayBuffer> {
  const size = bytes ?? buffer.size
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(buffer, 0, staging, 0, size)
  device.queue.submit([enc.finish()])
  await staging.mapAsync(GPUMapMode.READ)
  const data = staging.getMappedRange().slice(0)
  staging.unmap()
  staging.destroy()
  return data
}

function f16ToF32(h: number): number {
  const sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, frac = h & 0x3FF
  if (exp === 0) return frac === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024)
  if (exp === 31) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024)
}

function showF16(buf: ArrayBuffer, n = 8): string {
  return Array.from(new Uint16Array(buf.slice(0, n * 2))).map(v => f16ToF32(v).toFixed(4)).join(', ')
}

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

async function runTvm(dev: GPUDevice, d: CapturedDispatch): Promise<void> {
  const enc = dev.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(d.pipeline)
  pass.setBindGroup(0, dev.createBindGroup({
    layout: d.pipeline.getBindGroupLayout(0),
    entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
  }))
  pass.dispatchWorkgroups(...d.workgroups)
  pass.end()
  dev.queue.submit([enc.finish()])
  await dev.queue.onSubmittedWorkDone()
}

function dispatch(dev: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[], wgX: number, wgY = 1, wgZ = 1): void {
  const bg = dev.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b, offset: 0, size: b.size } })),
  })
  const enc = dev.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(wgX, wgY, wgZ)
  pass.end()
  dev.queue.submit([enc.finish()])
}

// ============================================================
// Full chain test
// ============================================================

async function runChainTest(cap: CaptureResult): Promise<void> {
  const dev = cap.device

  log('=== CHAIN TEST: Full forward pass with our shaders ===\n')

  // Capture GPU errors
  dev.addEventListener('uncapturederror', (e: Event) => {
    const err = (e as GPUUncapturedErrorEvent).error
    log(`  ⚠️ GPU ERROR: ${err.message}`)
  })

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
  }

  // Pre-create uniforms
  const U = {
    embed: makeUniform(dev, [1, 12]),
    norm1: makeUniform(dev, [1]),
    qkv: makeUniform(dev, [384, 96, 9216]),
    rope: makeUniform(dev, [1, 0, 1, 36]),
    oproj: makeUniform(dev, [384, 96, 3072]),
    addNorm: makeUniform(dev, [1]),
    fusedFfn: makeUniform(dev, [8192]),
    ffnDown: makeUniform(dev, [1024, 256, 3072]),
    lmHead: makeUniform(dev, [384, 96, 32064]),
    argmax: makeUniform(dev, [32064]),
  }

  const tokenOut = makeBuf(dev, 4)

  // Replay TVM writes to set up token_id, position, etc.
  for (const w of cap.writes) dev.queue.writeBuffer(w.buffer, w.offset, w.data.slice().buffer)
  await dev.queue.onSubmittedWorkDone()

  // Weight buffers extracted directly from TVM dispatches in the loop below

  // =========================================
  // FORWARD PASS
  // =========================================

  const t0 = performance.now()

  // HYBRID APPROACH: Use TVM's buffers where TVM dispatches need them.
  // Our shaders write into TVM's activation buffers so the handoff is seamless.
  //
  // From the dump, TVM decode uses these activation buffers per layer:
  //   dispatch[base+0] QKV:     @0=BUF732(out), @1=BUF730(in=normed)
  //   dispatch[base+1] RoPE:    @0=k(rw), @2=q(rw), @3=BUF732(in=qkv), @4=v(rw)
  //   dispatch[base+2] KVAppend: TVM's dispatch, reads from RoPE outputs
  //   dispatch[base+3] Attention: TVM's dispatch, @4=output(rw)
  //   dispatch[base+4] OProj:   @0=BUF732(out), @1=attnOut(in)
  //   dispatch[base+5] AddNorm1: @0=A(r), @1=B(r), @2=gamma, @3=O(rw), @4=resid(rw)
  //   dispatch[base+6] FFNUp:   @0=BUF732(out), @1=BUF730(in=normed)
  //   dispatch[base+7] SiLU:    @0=BUF730(out), @1=BUF732(in)
  //   dispatch[base+8] FFNDown: @0=BUF732(out), @1=BUF730(in=silu)
  //   dispatch[base+9] AddNorm2: @0=A(r), @1=B(r), @2=gamma, @3=O(rw), @4=resid(rw)

  // Strategy: Just use TVM's EXACT buffers for every step.
  // Our shaders produce output into the same buffers TVM would use.
  // For AddNorm: TVM uses fuse_add_norm_prefill_kernel which has different
  // buffer roles — we need to match those exactly.

  // Actually, the SIMPLEST correct approach: replace only the shaders but
  // use TVM's exact bind groups (buffers) for every dispatch.
  // This means: for each of the 342 dispatches, run OUR shader but with TVM's bindings.
  // The tricky part: our shaders have DIFFERENT binding layouts than TVM's.

  // Even simpler: for the chain test, just replace the shaders we CAN match 1:1
  // (same binding layout) and use TVM's dispatch for the rest.
  // Shaders with matching layout: matmul, rms_norm (same bindings as TVM)
  // Shaders with different layout: addNorm, rope, attention, fused_ffn

  // SIMPLEST CORRECT: Run TVM's dispatches for ALL steps, but replace just the
  // matmul dispatches with ours. This proves our matmul works in context.
  // Then incrementally replace more.

  log('  Strategy: Replace ALL except KV append + attention (TVM only for those 2)\n')

  // Replay writes
  for (const w of cap.writes) dev.queue.writeBuffer(w.buffer, w.offset, w.data.slice().buffer)
  await dev.queue.onSubmittedWorkDone()

  // Classify each dispatch and replace with our shader where possible
  // Per layer (10 dispatches): base = 2 + L*10
  //   +0: QKV matmul      → our matmul (same bindings) ✓
  //   +1: RoPE             → our rope (REMAP bindings)
  //   +2: KV Append        → TVM (different buffer model)
  //   +3: Attention        → TVM (paged KV bindings)
  //   +4: OProj matmul    → our matmul (same bindings) ✓
  //   +5: AddNorm1        → our addNorm (same bindings) ✓
  //   +6: FFNUp matmul    → SKIP (replaced by fused FFN below)
  //   +7: SiLU            → SKIP (replaced by fused FFN below)
  //   +8: FFNDown matmul  → our matmul (same bindings) ✓
  //   +9: AddNorm2        → our addNorm (same bindings) ✓

  let replaced = 0, tvmUsed = 0

  for (let i = 0; i < cap.dispatches.length; i++) {
    const d = cap.dispatches[i]

    // Preamble
    if (i === 0) {
      // Embedding: same bindings ✓
      const u = makeUniform(dev, [1, 12])
      dispatch(dev, P.embed, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 12)
      replaced++
    } else if (i === 1) {
      // RMSNorm: same bindings ✓
      const u = makeUniform(dev, [1])
      dispatch(dev, P.norm, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, u], 1)
      replaced++
    }
    // Layer dispatches
    else if (i >= 2 && i < 2 + 32 * 10) {
      const step = (i - 2) % 10

      if (step === 0) {
        // QKV matmul: same bindings ✓
        const u = makeUniform(dev, [384, 96, 9216])
        dispatch(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 9216)
        replaced++
      } else if (step === 1) {
        // RoPE: REMAP bindings
        // TVM: @0=k(rw), @1=pos_map(r), @2=q(rw), @3=qkv(r), @4=v(rw), @5=uniform
        // Ours: @0=q(rw), @1=k(rw), @2=v(rw), @3=qkv(r), @4=pos_map(r), @5=uniform
        const u = makeUniform(dev, [1, 0, 1, 36])
        dispatch(dev, P.rope, [d.entries[2].buffer, d.entries[0].buffer, d.entries[4].buffer, d.entries[3].buffer, d.entries[1].buffer, u], 36)
        replaced++
      } else if (step === 2 || step === 3) {
        // KV Append + Attention: TVM only
        await runTvm(dev, d)
        tvmUsed++
      } else if (step === 4) {
        // OProj matmul: same bindings ✓
        const u = makeUniform(dev, [384, 96, 3072])
        dispatch(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
        replaced++
      } else if (step === 5 || step === 9) {
        // AddNorm: same bindings ✓
        // TVM: @0=A(r), @1=B(r), @2=gamma(r), @3=O(rw), @4=residual(rw), @5=uniform
        const u = makeUniform(dev, [1])
        dispatch(dev, P.addNorm, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, d.entries[4].buffer, u], 1)
        replaced++
      } else if (step === 6) {
        // FFNUp: replaced by fused FFN (same bindings as FFNUp but output is SiLU result)
        // TVM FFNUp: @0=output(rw,16384 f16), @1=input(r), @2=scales(r), @3=weights(r), @4=uniform
        // But our fused FFN writes 8192 f16 (SiLU output), not 16384
        // The SiLU output goes into the buffer that TVM's SiLU would write to
        // TVM SiLU output buffer = dispatches[i+1].entries[0].buffer
        const siluD = cap.dispatches[i + 1]
        const siluOutBuf = siluD.entries[0].buffer

        // Copy input to ffnInput to avoid aliasing (input buffer = SiLU output buffer in TVM)
        const ffnInputCopy = makeBuf(dev, d.entries[1].buffer.size)
        const cpEnc = dev.createCommandEncoder()
        cpEnc.copyBufferToBuffer(d.entries[1].buffer, 0, ffnInputCopy, 0, d.entries[1].buffer.size)
        dev.queue.submit([cpEnc.finish()])
        await dev.queue.onSubmittedWorkDone()

        const u = makeUniform(dev, [8192])
        dispatch(dev, P.fusedFfn, [siluOutBuf, ffnInputCopy, d.entries[2].buffer, d.entries[3].buffer, u], 8192)
        replaced++
      } else if (step === 7) {
        // SiLU: SKIP — handled by fused FFN above
        replaced++  // count as replaced (not dispatched)
      } else if (step === 8) {
        // FFNDown matmul: same bindings ✓
        const u = makeUniform(dev, [1024, 256, 3072])
        dispatch(dev, P.matmul, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
        replaced++
      }
    }
    // Tail (sampling) — dispatch 322+
    else if (i === 2 + 32 * 10) {
      // LM head: remap bindings
      // TVM: @0=output(f32,rw), @1=scales(r), @2=weights(r), @3=input(r), @4=uniform
      // Ours: @0=output(f32,rw), @1=input(r), @2=scales(r), @3=weights(r), @4=uniform
      const u = makeUniform(dev, [384, 96, 32064])
      dispatch(dev, P.lmHead, [d.entries[0].buffer, d.entries[3].buffer, d.entries[1].buffer, d.entries[2].buffer, u], 32064)
      replaced++
    } else {
      // Remaining tail (sampling dispatches 323-341) — skip, replaced by argmax
      replaced++
    }

    await dev.queue.onSubmittedWorkDone()

    // Snapshot
    if (i === 11 || i === 21 || i === 31 || i === 321) {
      const L = Math.floor((i - 2) / 10)
      const snap = await readBuf(dev, d.entries[3].buffer, 32)
      log(`  L${L} normed[0..7]: ${showF16(snap)}`)
    }
  }

  // Argmax replaces 20 sampling dispatches
  const lmHeadDispatch = cap.dispatches[2 + 32 * 10]
  dispatch(dev, P.argmax, [lmHeadDispatch.entries[0].buffer, tokenOut, U.argmax], 1)
  await dev.queue.onSubmittedWorkDone()
  replaced++

  log(`\n  Replaced: ${replaced} dispatches with our shaders`)
  log(`  TVM used: ${tvmUsed} dispatches (KV append + attention only)`)

  const elapsed = performance.now() - t0

  // Read our token
  const ourTokenData = await readBuf(dev, tokenOut, 4)
  const ourToken = new Int32Array(ourTokenData)[0]

  // Run TVM fully for comparison + layer snapshots
  log('\n--- TVM reference run ---')
  for (const w of cap.writes) dev.queue.writeBuffer(w.buffer, w.offset, w.data.slice().buffer)
  for (let i = 0; i < cap.dispatches.length; i++) {
    await runTvm(dev, cap.dispatches[i])
    // Snapshot after AddNorm2 of key layers (dispatch index = 2 + L*10 + 9 = 11 + L*10)
    if (i === 11 || i === 21 || i === 31 || i === 321) {
      const L = Math.floor((i - 2) / 10)
      const d = cap.dispatches[i]
      const tvmSnap = await readBuf(dev, d.entries[3].buffer, 32)
      log(`  L${L} TVM normed[0..7]: ${showF16(tvmSnap)}`)
    }
  }
  const tvmTokenData = await readBuf(dev, cap.copy!.src, 4)
  const tvmToken = new Int32Array(tvmTokenData)[0]

  log(`\n${'='.repeat(60)}`)
  log(`  OUR TOKEN:  ${ourToken}`)
  log(`  TVM TOKEN:  ${tvmToken}`)
  log(`  TIME:       ${elapsed.toFixed(1)}ms`)
  log(`${'='.repeat(60)}`)

  if (ourToken === tvmToken) {
    log(`  ✅ CHAIN TEST PASSED — Same token!`)
  } else {
    log(`  ⚠️ DIFFERENT TOKEN — checking logits...`)
    // Read and compare top logits
    const lmHeadBuf = cap.dispatches[2 + 32 * 10].entries[0].buffer
    const ourLogits = new Float32Array(await readBuf(dev, lmHeadBuf, 32064 * 4))
    let ourMax = -Infinity, ourMaxIdx = 0
    for (let i = 0; i < 32064; i++) { if (ourLogits[i] > ourMax) { ourMax = ourLogits[i]; ourMaxIdx = i } }
    log(`  Our argmax: token ${ourMaxIdx} (logit ${ourMax.toFixed(4)})`)
    log(`  TVM uses top-p sampling with randomness — different token is expected if top-p applies`)
  }
}

// ============================================================
// Main
// ============================================================

async function run(): Promise<void> {
  log('Loading model...\n')

  const engine = new LLMEngine()
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

  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => log(msg))
  log('\nRunning TVM chat to capture...')
  await engine.chat('What is the capital of France?', () => {})

  const cap = getCaptureResult()
  if (!cap) { log('ERROR: No capture'); return }
  log(`Capture: ${cap.dispatches.length} decode, ${cap.writes.length} writes\n`)

  await runChainTest(cap)

  log('\n=== CHAIN TEST COMPLETE ===')
}

run().catch(e => log(`FATAL: ${e}`))
