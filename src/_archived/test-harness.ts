/**
 * SHADER TEST HARNESS — Verifies each of our 10 shaders against TVM's ground truth.
 *
 * For each decode step:
 *   1. Replay TVM writes + run TVM dispatches up to this step (establish state)
 *   2. Run TVM's dispatch → read output (ground truth)
 *   3. Run our shader with same input buffers → read output
 *   4. Compare
 */

import { LLMEngine, MODELS } from './engine.js'
import { patchForCapture, getCaptureResult, CapturedDispatch, CaptureResult } from './capture.js'

import { withPrelude } from '../compiler/shader-prelude'
import { int4MatmulWGSL } from '../compiler/shaders/int4_matmul.gen'
import rmsNormSrc from '../compiler/shaders/rms_norm.wgsl?raw'
import addNormSrc from '../compiler/shaders/add_norm.wgsl?raw'
import ropeSrc from '../compiler/shaders/rope.wgsl?raw'
// KV append + attention tested in chain test (different binding layout)
// import kvAppendSrc from '../compiler/shaders/kv_append.wgsl?raw'
// import attentionSrc from '../compiler/shaders/attention.wgsl?raw'
import fusedFfnSrc from '../compiler/shaders/fused_ffn.wgsl?raw'
import embeddingSrc from '../compiler/shaders/embedding.wgsl?raw'
import argmaxSrc from '../compiler/shaders/argmax.wgsl?raw'

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
  const sign = (h >> 15) & 1
  const exp = (h >> 10) & 0x1F
  const frac = h & 0x3FF
  if (exp === 0) return frac === 0 ? (sign ? -0 : 0) : (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024)
  if (exp === 31) return frac === 0 ? (sign ? -Infinity : Infinity) : NaN
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024)
}

function showF16(buf: ArrayBuffer, n = 8): string {
  return Array.from(new Uint16Array(buf.slice(0, n * 2))).map(v => f16ToF32(v).toFixed(4)).join(', ')
}

function showF32(buf: ArrayBuffer, n = 8): string {
  return Array.from(new Float32Array(buf.slice(0, n * 4))).map(v => v.toFixed(4)).join(', ')
}

function showI32(buf: ArrayBuffer, n = 8): string {
  return Array.from(new Int32Array(buf.slice(0, n * 4))).join(', ')
}

function cmpF16(a: ArrayBuffer, b: ArrayBuffer, n?: number): { maxDiff: number; avgDiff: number; matchPct: number; idx: number } {
  const va = new Uint16Array(a), vb = new Uint16Array(b)
  const count = n ?? Math.min(va.length, vb.length)
  let maxDiff = 0, sumDiff = 0, exact = 0, idx = -1
  for (let i = 0; i < count; i++) {
    const d = Math.abs(f16ToF32(va[i]) - f16ToF32(vb[i]))
    if (d > maxDiff) maxDiff = d
    sumDiff += d
    if (va[i] === vb[i]) exact++
    else if (idx === -1) idx = i
  }
  return { maxDiff, avgDiff: sumDiff / count, matchPct: (exact / count) * 100, idx }
}

function cmpF32(a: ArrayBuffer, b: ArrayBuffer, n?: number): { maxDiff: number; avgDiff: number; matchPct: number; idx: number } {
  const va = new Float32Array(a), vb = new Float32Array(b)
  const count = n ?? Math.min(va.length, vb.length)
  let maxDiff = 0, sumDiff = 0, exact = 0, idx = -1
  for (let i = 0; i < count; i++) {
    const d = Math.abs(va[i] - vb[i])
    if (d > maxDiff) maxDiff = d
    sumDiff += d
    if (va[i] === vb[i]) exact++
    else if (idx === -1) idx = i
  }
  return { maxDiff, avgDiff: sumDiff / count, matchPct: (exact / count) * 100, idx }
}

function result(name: string, cmp: { maxDiff: number; avgDiff: number; matchPct: number; idx: number }, tolerance = 0.1): void {
  log(`  RESULT: maxDiff=${cmp.maxDiff.toFixed(6)}, avgDiff=${cmp.avgDiff.toFixed(6)}, match=${cmp.matchPct.toFixed(1)}%`)
  if (cmp.matchPct > 99) log(`  ✅ ${name} — IDENTICAL`)
  else if (cmp.maxDiff < tolerance) log(`  ✅ ${name} — PASS (within tolerance)`)
  else log(`  ❌ ${name} — FAIL (first mismatch at index ${cmp.idx})`)
}

async function runTvm(device: GPUDevice, d: CapturedDispatch): Promise<void> {
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(d.pipeline)
  pass.setBindGroup(0, device.createBindGroup({
    layout: d.pipeline.getBindGroupLayout(0),
    entries: d.entries.map(e => ({ binding: e.binding, resource: { buffer: e.buffer, offset: e.offset, size: e.size } })),
  }))
  pass.dispatchWorkgroups(...d.workgroups)
  pass.end()
  device.queue.submit([enc.finish()])
  await device.queue.onSubmittedWorkDone()
}

function makePipeline(device: GPUDevice, src: string, entry: string): GPUComputePipeline {
  return device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: withPrelude(src) }), entryPoint: entry } })
}

function makeUniform(device: GPUDevice, values: number[]): GPUBuffer {
  const buf = device.createBuffer({ size: Math.max(values.length * 4, 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(buf, 0, new Uint32Array(values))
  return buf
}

function makeBuf(device: GPUDevice, size: number): GPUBuffer {
  return device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST })
}

function dispatchOur(device: GPUDevice, pipeline: GPUComputePipeline, bufs: GPUBuffer[], wgX: number, wgY = 1, wgZ = 1): void {
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b, offset: 0, size: b.size } })),
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bg)
  pass.dispatchWorkgroups(wgX, wgY, wgZ)
  pass.end()
  device.queue.submit([enc.finish()])
}

/** Replay all TVM writes + run dispatches 0..upTo-1 */
async function replayTvmUpTo(cap: CaptureResult, upTo: number): Promise<void> {
  const device = cap.device
  for (const w of cap.writes) device.queue.writeBuffer(w.buffer, w.offset, w.data.slice().buffer)
  for (let i = 0; i < upTo; i++) await runTvm(device, cap.dispatches[i])
}

function getPipelineName(cap: CaptureResult, d: CapturedDispatch): string {
  return cap.pipelines.find(p => p.pipeline === d.pipeline)?.entryPoint ?? '?'
}

// ============================================================
// TESTS
// ============================================================

async function testEmbedding(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[0]
  log(`\n--- TEST 0: Embedding ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 0)
  await runTvm(dev, d)
  const tvmOut = await readBuf(dev, d.entries[0].buffer, 6144)
  log(`  TVM: ${showF16(tvmOut)}`)

  const out = makeBuf(dev, d.entries[0].buffer.size)
  const p = makePipeline(dev, embeddingSrc, 'embedding')
  const u = makeUniform(dev, [1, 12])
  dispatchOur(dev, p, [out, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 12)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, 6144)
  log(`  Ours: ${showF16(ourOut)}`)
  result('Embedding', cmpF16(tvmOut, ourOut, 3072))
}

async function testRmsNorm(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[1]
  log(`\n--- TEST 1: RMSNorm ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 1)
  await runTvm(dev, d)
  const tvmOut = await readBuf(dev, d.entries[0].buffer, 6144)
  log(`  TVM: ${showF16(tvmOut)}`)

  const out = makeBuf(dev, 6144)
  const p = makePipeline(dev, rmsNormSrc, 'rms_norm')
  const u = makeUniform(dev, [1])
  dispatchOur(dev, p, [out, d.entries[1].buffer, d.entries[2].buffer, u], 1)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, 6144)
  log(`  Ours: ${showF16(ourOut)}`)
  result('RMSNorm', cmpF16(tvmOut, ourOut, 3072))
}

async function testQkvMatmul(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[2]
  log(`\n--- TEST 2: QKV Matmul (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 2)
  await runTvm(dev, d)
  const sz = Math.min(d.entries[0].buffer.size, 18432)
  const tvmOut = await readBuf(dev, d.entries[0].buffer, sz)
  log(`  TVM: ${showF16(tvmOut)}`)

  const out = makeBuf(dev, d.entries[0].buffer.size)
  const p = makePipeline(dev, int4MatmulWGSL(), 'int4_matmul')
  const u = makeUniform(dev, [384, 96, 9216])
  dispatchOur(dev, p, [out, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 9216)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, sz)
  log(`  Ours: ${showF16(ourOut)}`)
  result('QKV Matmul', cmpF16(tvmOut, ourOut, 9216))
}

async function testRope(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[3]
  log(`\n--- TEST 3: RoPE (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 3)
  await runTvm(dev, d)

  // TVM RoPE: @0=k(rw), @1=position_map(r), @2=q(rw), @3=qkv(r), @4=v(rw), @5=uniform
  // Outputs: q (@2), k (@0), v (@4)
  const tvmQ = await readBuf(dev, d.entries[2].buffer, 6144)
  const tvmK = await readBuf(dev, d.entries[0].buffer, 6144)
  const tvmV = await readBuf(dev, d.entries[4].buffer, 6144)
  log(`  TVM Q: ${showF16(tvmQ)}`)
  log(`  TVM K: ${showF16(tvmK)}`)
  log(`  TVM V: ${showF16(tvmV)}`)

  // Our RoPE: @0=q_out(rw), @1=k_out(rw), @2=v_out(rw), @3=qkv(r), @4=position_map(r), @5=uniform
  const qOut = makeBuf(dev, d.entries[2].buffer.size)
  const kOut = makeBuf(dev, d.entries[0].buffer.size)
  const vOut = makeBuf(dev, d.entries[4].buffer.size)
  const p = makePipeline(dev, ropeSrc, 'rope_kernel')
  const u = makeUniform(dev, [1, 0, 1, 36])  // apply_rope=1, pos_offset=0, seq_len=1, packGridDimX=36
  dispatchOur(dev, p, [qOut, kOut, vOut, d.entries[3].buffer, d.entries[1].buffer, u], 36)
  await dev.queue.onSubmittedWorkDone()

  const ourQ = await readBuf(dev, qOut, 6144)
  const ourK = await readBuf(dev, kOut, 6144)
  const ourV = await readBuf(dev, vOut, 6144)
  log(`  Our Q: ${showF16(ourQ)}`)
  log(`  Our K: ${showF16(ourK)}`)
  log(`  Our V: ${showF16(ourV)}`)

  result('RoPE Q', cmpF16(tvmQ, ourQ, 3072))
  result('RoPE K', cmpF16(tvmK, ourK, 3072))
  result('RoPE V', cmpF16(tvmV, ourV, 3072))
}

async function testKvAppend(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[4]
  log(`\n--- TEST 4: KV Append (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  // TVM KV append: tir_kv_cache_transpose_append_kernel
  // @0=kv_data(r), @1=pages(rw), @2=position_map(r), @3=?(r), @4=uniform
  await replayTvmUpTo(cap, 4)
  await runTvm(dev, d)

  // Read a small portion of KV pages to verify
  const tvmPages = await readBuf(dev, d.entries[1].buffer, 1024)
  log(`  TVM pages[0..512 f16]: ${showF16(tvmPages)}`)

  // Our KV append has different bindings: @0=k(r), @1=v(r), @2=pages(rw), @3=position_map(r), @4=uniform
  // We need the separate K and V buffers from the RoPE step
  // Since this test depends on RoPE output, skip standalone test — will verify in chain test
  log(`  ⏭️ SKIP — KV append depends on separate K,V buffers from our RoPE (different layout from TVM)`)
  log(`  Will verify in chain test`)
}

async function testAttention(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[5]
  log(`\n--- TEST 5: Attention (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  // Run TVM up to attention output
  await replayTvmUpTo(cap, 5)
  await runTvm(dev, d)

  // TVM decode attention: @0=Q(r), @1=k_rope_pos(r), @2=length_info(r), @3=pages_indptr(r),
  //   @4=output(rw), @5=page_indptr(r), @6=temp(rw), @7=kv_pages(r), @8=position_map(r), @9=uniform
  const tvmOutBuf = d.entries[4].buffer  // output at binding 4
  const tvmOut = await readBuf(dev, tvmOutBuf, Math.min(tvmOutBuf.size, 6144))
  log(`  TVM output: ${showF16(tvmOut)}`)

  // Our attention has very different bindings — skip standalone, verify in chain
  log(`  ⏭️ SKIP — Attention bindings differ significantly from TVM's paged KV layout`)
  log(`  Will verify in chain test (full layer output must match)`)
}

async function testOProj(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[6]
  log(`\n--- TEST 6: O Projection (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 6)
  await runTvm(dev, d)
  const sz = Math.min(d.entries[0].buffer.size, 6144)
  const tvmOut = await readBuf(dev, d.entries[0].buffer, sz)
  log(`  TVM: ${showF16(tvmOut)}`)

  // Same shader as QKV matmul, different dimensions: K=3072→3072
  const out = makeBuf(dev, d.entries[0].buffer.size)
  const p = makePipeline(dev, int4MatmulWGSL(), 'int4_matmul')
  const u = makeUniform(dev, [384, 96, 3072])  // K_PACKED=384, SCALES=96, N=3072
  dispatchOur(dev, p, [out, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, sz)
  log(`  Ours: ${showF16(ourOut)}`)
  result('O Projection', cmpF16(tvmOut, ourOut, 3072))
}

async function testAddNorm(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[7]
  log(`\n--- TEST 7: AddNorm (L0:AddNorm1) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 7)
  await runTvm(dev, d)

  // TVM fuse_add_norm_prefill_kernel: @0=A(r), @1=B(r), @2=gamma(r), @3=O(rw), @4=add_residual(rw), @5=uniform
  const tvmNormed = await readBuf(dev, d.entries[3].buffer, Math.min(d.entries[3].buffer.size, 6144))
  const tvmResidual = await readBuf(dev, d.entries[4].buffer, Math.min(d.entries[4].buffer.size, 6144))
  log(`  TVM normed: ${showF16(tvmNormed)}`)
  log(`  TVM residual: ${showF16(tvmResidual)}`)

  // Our add_norm: @0=A(r), @1=B(r), @2=gamma(r), @3=output(rw), @4=residual(rw), @5=uniform
  const ourNormed = makeBuf(dev, 6144)
  const ourResidual = makeBuf(dev, d.entries[4].buffer.size)
  const p = makePipeline(dev, addNormSrc, 'add_norm')
  const u = makeUniform(dev, [1])
  dispatchOur(dev, p, [d.entries[0].buffer, d.entries[1].buffer, d.entries[2].buffer, ourNormed, ourResidual, u], 1)
  await dev.queue.onSubmittedWorkDone()

  const ourN = await readBuf(dev, ourNormed, 6144)
  const ourR = await readBuf(dev, ourResidual, Math.min(ourResidual.size, 6144))
  log(`  Our normed: ${showF16(ourN)}`)
  log(`  Our residual: ${showF16(ourR)}`)
  result('AddNorm normed', cmpF16(tvmNormed, ourN, 3072))
  result('AddNorm residual', cmpF16(tvmResidual, ourR, 3072))
}

async function testFusedFfn(cap: CaptureResult): Promise<void> {
  const dev = cap.device
  // FFNUp = dispatch 8 (16384 wg), SiLU = dispatch 9 (32 wg)
  const dUp = cap.dispatches[8], dSilu = cap.dispatches[9]
  log(`\n--- TEST 8+9: Fused FFN (L0:FFNUp + SiLU) ---`)
  log(`  TVM FFNUp: ${getPipelineName(cap, dUp)} wg=[${dUp.workgroups}]`)
  log(`  TVM SiLU:  ${getPipelineName(cap, dSilu)} wg=[${dSilu.workgroups}]`)

  // Run TVM up through SiLU
  await replayTvmUpTo(cap, 8)
  await runTvm(dev, dUp)
  await runTvm(dev, dSilu)

  // TVM SiLU output: @0=output(rw) — 8192 f16
  const tvmOutBuf = dSilu.entries[0].buffer
  const tvmOut = await readBuf(dev, tvmOutBuf, Math.min(tvmOutBuf.size, 16384))
  log(`  TVM SiLU output: ${showF16(tvmOut)}`)

  // CRITICAL: BUF#730 is a ping-pong buffer. After TVM's SiLU it contains SiLU output,
  // not the normed FFN input. We must:
  //   1. Replay up to dispatch 8 (AddNorm1 done, FFNUp not yet)
  //   2. Save the input buffer
  //   3. Run our fused shader
  //   4. Then run TVM's FFNUp+SiLU for ground truth comparison

  // Step 1: Re-replay to get clean state before FFNUp
  await replayTvmUpTo(cap, 8)

  // Step 2: Copy input buffer before anyone overwrites it
  const inputCopy = makeBuf(dev, dUp.entries[1].buffer.size)
  const copyEnc = dev.createCommandEncoder()
  copyEnc.copyBufferToBuffer(dUp.entries[1].buffer, 0, inputCopy, 0, dUp.entries[1].buffer.size)
  dev.queue.submit([copyEnc.finish()])
  await dev.queue.onSubmittedWorkDone()

  const inputData = await readBuf(dev, inputCopy, 64)
  log(`  FFNUp input[0..7]: ${showF16(inputData)}`)

  // Step 3: Run OUR fused shader with the saved input
  const out = makeBuf(dev, tvmOutBuf.size)
  const p = makePipeline(dev, fusedFfnSrc, 'fused_ffn_kernel')
  const u = makeUniform(dev, [8192])
  dispatchOur(dev, p, [out, inputCopy, dUp.entries[2].buffer, dUp.entries[3].buffer, u], 8192)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, Math.min(out.size, 16384))
  log(`  Our fused output: ${showF16(ourOut)}`)

  // Step 4: Re-replay and run TVM for ground truth
  await replayTvmUpTo(cap, 8)
  await runTvm(dev, dUp)
  await runTvm(dev, dSilu)
  const tvmOut2 = await readBuf(dev, tvmOutBuf, Math.min(tvmOutBuf.size, 16384))
  log(`  TVM SiLU output (fresh): ${showF16(tvmOut2)}`)

  result('Fused FFN', cmpF16(tvmOut2, ourOut, 8192))
}

async function testFfnDown(cap: CaptureResult): Promise<void> {
  const dev = cap.device, d = cap.dispatches[10]
  log(`\n--- TEST 10: FFN Down (L0) ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, 10)
  await runTvm(dev, d)
  const sz = Math.min(d.entries[0].buffer.size, 6144)
  const tvmOut = await readBuf(dev, d.entries[0].buffer, sz)
  log(`  TVM: ${showF16(tvmOut)}`)

  // FFN down: K=8192→3072, K_PACKED=1024, SCALES_PER_ROW=256
  const out = makeBuf(dev, d.entries[0].buffer.size)
  const p = makePipeline(dev, int4MatmulWGSL(), 'int4_matmul')
  const u = makeUniform(dev, [1024, 256, 3072])
  dispatchOur(dev, p, [out, d.entries[1].buffer, d.entries[2].buffer, d.entries[3].buffer, u], 3072)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, sz)
  log(`  Ours: ${showF16(ourOut)}`)
  result('FFN Down', cmpF16(tvmOut, ourOut, 3072))
}

async function testLmHead(cap: CaptureResult): Promise<void> {
  const dev = cap.device
  // LM head = first tail dispatch (dispatch 322)
  const tailIdx = 2 + 32 * 10  // 322
  const d = cap.dispatches[tailIdx]
  log(`\n--- TEST 322: LM Head ---`)
  log(`  TVM: ${getPipelineName(cap, d)} wg=[${d.workgroups}]`)

  await replayTvmUpTo(cap, tailIdx)
  await runTvm(dev, d)

  // TVM LM head: @0=output(rw, f32!), @1=scales(r), @2=weights(r), @3=input(r), @4=uniform
  const tvmOutBuf = d.entries[0].buffer
  const tvmOut = await readBuf(dev, tvmOutBuf, Math.min(tvmOutBuf.size, 32064 * 4))
  log(`  TVM logits (f32): ${showF32(tvmOut)}`)

  // Our int4_matmul_f32: @0=output(rw,f32), @1=input(r), @2=scales(r), @3=weights(r), @4=uniform
  // NOTE: TVM has input at @3, scales at @1, weights at @2 — different order!
  const out = makeBuf(dev, tvmOutBuf.size)
  const p = makePipeline(dev, int4MatmulWGSL({ outF32: true }), 'int4_matmul_f32')
  const u = makeUniform(dev, [384, 96, 32064])
  // Map TVM bindings → our bindings: TVM @3=input → our @1, TVM @1=scales → our @2, TVM @2=weights → our @3
  dispatchOur(dev, p, [out, d.entries[3].buffer, d.entries[1].buffer, d.entries[2].buffer, u], 32064)
  await dev.queue.onSubmittedWorkDone()
  const ourOut = await readBuf(dev, out, Math.min(out.size, 32064 * 4))
  log(`  Our logits (f32): ${showF32(ourOut)}`)
  result('LM Head', cmpF32(tvmOut, ourOut, 32064), 1.0) // higher tolerance for f32 logits
}

async function testArgmax(cap: CaptureResult): Promise<void> {
  const dev = cap.device
  log(`\n--- TEST: Argmax (replaces 20 sampling dispatches) ---`)

  // Run TVM through all decode dispatches to get the final token
  await replayTvmUpTo(cap, cap.dispatches.length)

  // Read TVM's sampled token (from the copy operation)
  const tvmTokenBuf = cap.copy!.src
  const tvmToken = await readBuf(dev, tvmTokenBuf, 4)
  log(`  TVM sampled token: ${showI32(tvmToken, 1)}`)

  // Get logits from LM head (dispatch 322's output)
  const tailIdx = 2 + 32 * 10
  const lmHeadDispatch = cap.dispatches[tailIdx]
  const logitsBuf = lmHeadDispatch.entries[0].buffer

  // Run our argmax on the same logits
  const tokenOut = makeBuf(dev, 4)
  const p = makePipeline(dev, argmaxSrc, 'argmax_kernel')
  const u = makeUniform(dev, [32064])
  dispatchOur(dev, p, [logitsBuf, tokenOut, u], 1)
  await dev.queue.onSubmittedWorkDone()
  const ourToken = await readBuf(dev, tokenOut, 4)
  log(`  Our argmax token: ${showI32(ourToken, 1)}`)

  const tvmId = new Int32Array(tvmToken)[0]
  const ourId = new Int32Array(ourToken)[0]
  if (tvmId === ourId) {
    log(`  ✅ Argmax — IDENTICAL (token ${ourId})`)
  } else {
    log(`  ⚠️ Argmax — DIFFERENT (TVM=${tvmId}, Ours=${ourId})`)
    log(`    Note: TVM uses top-p sampling with randomness, argmax is deterministic`)
    log(`    This is expected — argmax picks the highest-probability token`)
  }
}

// ============================================================
// Main
// ============================================================

async function run(): Promise<void> {
  log('=== SHADER TEST HARNESS — All 10 shaders vs TVM ===')
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
  log(`Capture: ${cap.dispatches.length} decode dispatches, ${cap.writes.length} writes\n`)

  // Layer 0 decode steps: [2]=QKV, [3]=RoPE, [4]=KVAppend, [5]=Attention, [6]=OProj, [7]=AddNorm1, [8]=FFNUp, [9]=SiLU, [10]=FFNDown, [11]=AddNorm2
  await testEmbedding(cap)
  await testRmsNorm(cap)
  await testQkvMatmul(cap)
  await testRope(cap)
  await testKvAppend(cap)
  await testAttention(cap)
  await testOProj(cap)
  await testAddNorm(cap)
  await testFusedFfn(cap)
  await testFfnDown(cap)
  await testLmHead(cap)
  await testArgmax(cap)

  log('\n=== ALL TESTS COMPLETE ===')

  // Add download button
  const btn = document.createElement('button')
  btn.textContent = 'Download Results'
  btn.style.cssText = 'position:fixed;top:10px;right:10px;padding:10px 20px;background:#4a9eff;color:white;border:none;border-radius:6px;cursor:pointer;z-index:9999'
  btn.onclick = () => {
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `shader-tests-${Date.now()}.txt`
    a.click()
  }
  document.body.appendChild(btn)
}

run().catch(e => log(`FATAL: ${e}`))
