/**
 * DUMP TVM — 100% Reverse Engineering of TVM's WebGPU LLM Pipeline.
 *
 * This tool captures and analyzes EVERYTHING:
 *
 * 1. PIPELINE CATALOG — every shader module, entry point, bind group layout
 * 2. DISPATCH STRUCTURE — prefill vs decode, per-layer mapping, preamble + tail
 * 3. BUFFER LANDSCAPE — weights, activations, KV cache, uniforms with exact sizes
 * 4. UNIFORM DECODING — every field of every podArgs struct, interpreted as int/float/hex
 * 5. BUFFER DATA FLOW — which dispatches read/write which buffers (DAG)
 * 6. WRITE ANALYSIS — static vs per-token vs per-prompt writes
 * 7. PREFILL vs DECODE DIFF — side-by-side comparison of every layer step
 */

import { LLMEngine, MODELS } from './engine.js'
import { patchForCapture, getCaptureResult, CapturedDispatch, CapturedWrite, CaptureResult } from './capture.js'

// ============================================================
// Globals
// ============================================================

const lines: string[] = []
const log = (msg: string) => {
  lines.push(msg)
  const el = document.getElementById('log')
  if (el) { el.textContent = lines.join('\n'); el.scrollTop = el.scrollHeight }
}

const sep = (title: string) => {
  log(`\n${'='.repeat(90)}`)
  log(`  ${title}`)
  log(`${'='.repeat(90)}`)
}

const subsep = (title: string) => {
  log(`\n${'─'.repeat(70)}`)
  log(`  ${title}`)
  log(`${'─'.repeat(70)}`)
}

// Buffer identity tracking
const bufferIds = new Map<GPUBuffer, number>()
let nextBufId = 0
function bufId(buf: GPUBuffer): string {
  if (!bufferIds.has(buf)) bufferIds.set(buf, nextBufId++)
  return `BUF#${bufferIds.get(buf)!}`
}

function bufType(buf: GPUBuffer): string {
  const sz = buf.size
  if (sz > 1_000_000) return 'WEIGHT'
  if (sz > 10_000) return 'KV_CACHE'
  if (sz <= 64) return 'UNIFORM'
  return 'ACTIVATION'
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return (b / 1e9).toFixed(2) + 'GB'
  if (b >= 1e6) return (b / 1e6).toFixed(2) + 'MB'
  if (b >= 1e3) return (b / 1e3).toFixed(1) + 'KB'
  return b + 'B'
}

// ============================================================
// Uniform Decoders — interpret raw bytes as struct fields
// ============================================================

function decodeUniform(data: Uint8Array, pipelineName: string): string {
  if (!data || data.length === 0) return '(empty)'
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
  const fields: string[] = []

  // Read helpers
  const i32 = (off: number) => off + 4 <= data.length ? dv.getInt32(off, true) : '?'
  const u32 = (off: number) => off + 4 <= data.length ? dv.getUint32(off, true) : '?'
  const f32 = (off: number) => off + 4 <= data.length ? dv.getFloat32(off, true).toFixed(6) : '?'

  // Identify struct by pipeline name + data size
  if (pipelineName.includes('batch_decode_paged_kv') && data.length >= 52) {
    // Decode attention: 56B (14 fields × 4B)
    fields.push(`B=${i32(0)}`)
    fields.push(`k_rope_pos_offset_elem_off=${i32(4)}`)
    fields.push(`length_info_elem_off=${i32(8)}`)
    fields.push(`max_num_pages=${i32(12)}`)
    fields.push(`nnz_pages=${i32(16)}`)
    fields.push(`page_indptr_elem_off=${i32(20)}`)
    fields.push(`page_values_elem_off=${i32(24)}`)
    fields.push(`pages_elem_off=${i32(28)}`)
    fields.push(`q_rope_position_elem_off=${i32(32)}`)
    fields.push(`rope_scale=${f32(36)}`)
    fields.push(`rope_theta=${f32(40)}`)
    fields.push(`rotary_mode=${i32(44)}`)
    fields.push(`sm_scale=${f32(48)}`)
    if (data.length >= 56) fields.push(`packGridDimX=${u32(52)}`)
  } else if (pipelineName.includes('batch_prefill_ragged_kv') && data.length >= 48) {
    // Prefill attention — different struct
    fields.push(`causal=${i32(0)}`)
    fields.push(`kv_chunk_size=${i32(4)}`)
    fields.push(`nnz=${i32(8)}`)
    fields.push(`num_heads=${i32(12)}`)
    fields.push(`num_kv_heads=${i32(16)}`)
    fields.push(`page_size=${i32(20)}`)
    fields.push(`q_indptr_elem_off=${i32(24)}`)
    fields.push(`rope_scale=${f32(28)}`)
    fields.push(`rope_theta=${f32(32)}`)
    fields.push(`rotary_mode=${i32(36)}`)
    fields.push(`sm_scale=${f32(40)}`)
    if (data.length >= 48) fields.push(`packGridDimX=${u32(44)}`)
    if (data.length >= 52) fields.push(`extra1=${i32(48)}`)
  } else if (pipelineName.includes('batch_prefill_paged_kv') && data.length >= 48) {
    // Prefill paged KV sliding window — similar to ragged
    fields.push(`causal=${i32(0)}`)
    fields.push(`kv_chunk_size=${i32(4)}`)
    fields.push(`max_num_pages=${i32(8)}`)
    fields.push(`nnz=${i32(12)}`)
    fields.push(`num_heads=${i32(16)}`)
    fields.push(`num_kv_heads=${i32(20)}`)
    fields.push(`page_size=${i32(24)}`)
    fields.push(`q_indptr_elem_off=${i32(28)}`)
    fields.push(`rope_scale=${f32(32)}`)
    fields.push(`rope_theta=${f32(36)}`)
    fields.push(`rotary_mode=${i32(40)}`)
    fields.push(`sm_scale=${f32(44)}`)
    if (data.length >= 52) fields.push(`packGridDimX=${u32(48)}`)
  } else if (pipelineName.includes('fused_rope') && data.length >= 12) {
    fields.push(`apply_rope=${i32(0)}`)
    fields.push(`position_map_elem_off=${i32(4)}`)
    fields.push(`seq_len=${i32(8)}`)
    if (data.length >= 16) fields.push(`packGridDimX=${u32(12)}`)
  } else if (pipelineName.includes('kv_cache_transpose_append') && data.length >= 16) {
    fields.push(`ntoken=${i32(0)}`)
    fields.push(`num_pages=${i32(4)}`)
    fields.push(`pages_elem_off=${i32(8)}`)
    fields.push(`position_map_elem_off=${i32(12)}`)
    if (data.length >= 20) fields.push(`packGridDimX=${u32(16)}`)
  } else if (pipelineName.includes('rms_norm') && data.length >= 4) {
    fields.push(`packGridDimX=${u32(0)}`)
    if (data.length >= 8) fields.push(`field1=${u32(4)}`)
  } else if (pipelineName.includes('fuse_add_norm') && data.length >= 4) {
    fields.push(`packGridDimX=${u32(0)}`)
    if (data.length >= 8) fields.push(`field1=${u32(4)}`)
  } else if (pipelineName.includes('NT_matmul') && data.length >= 4) {
    fields.push(`packGridDimX=${u32(0)}`)
  } else if (pipelineName.includes('split2_silu2') && data.length >= 4) {
    fields.push(`packGridDimX=${u32(0)}`)
    if (data.length >= 8) fields.push(`field1=${u32(4)}`)
  } else if (pipelineName.includes('dequantize_take') && data.length >= 4) {
    fields.push(`packGridDimX=${u32(0)}`)
  } else if (pipelineName.includes('tree_attn') && data.length >= 4) {
    // Tree attention (prefill)
    for (let off = 0; off < Math.min(data.length, 64); off += 4) {
      fields.push(`f${off/4}=${i32(off)}`)
    }
  } else {
    // Generic: dump as both int32 and float32
    for (let off = 0; off < Math.min(data.length, 64); off += 4) {
      if (off + 4 <= data.length) {
        const iv = dv.getInt32(off, true)
        const fv = dv.getFloat32(off, true)
        // Heuristic: if float looks reasonable (not denorm/inf), show both
        if (Math.abs(fv) > 1e-6 && Math.abs(fv) < 1e10 && !Number.isNaN(fv) && iv > 100000) {
          fields.push(`[${off}]=${iv} (f32:${fv.toFixed(4)})`)
        } else {
          fields.push(`[${off}]=${iv}`)
        }
      }
    }
  }
  return `{${fields.join(', ')}}`
}

// ============================================================
// Main
// ============================================================

async function run(): Promise<void> {
  log('=== TVM 100% REVERSE ENGINEERING DUMP ===')
  log(`Date: ${new Date().toISOString()}`)
  log('')

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

  log('Loading model...')
  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => log(msg))

  log('Running chat to trigger capture...')
  await engine.chat('What is the capital of France?', () => {})

  const cap = getCaptureResult()
  if (!cap) { log('ERROR: No capture'); return }

  // Pipeline name lookup
  const pipelineNames = new Map<GPUComputePipeline, string>()
  for (const p of cap.pipelines) pipelineNames.set(p.pipeline, p.entryPoint)

  // Write data lookup by buffer
  const writesByBuffer = new Map<GPUBuffer, Array<{ offset: number; data: Uint8Array; phase: string; index: number }>>()
  cap.prefillWrites.forEach((w, i) => {
    if (!writesByBuffer.has(w.buffer)) writesByBuffer.set(w.buffer, [])
    writesByBuffer.get(w.buffer)!.push({ offset: w.offset, data: w.data, phase: 'prefill', index: i })
  })
  cap.writes.forEach((w, i) => {
    if (!writesByBuffer.has(w.buffer)) writesByBuffer.set(w.buffer, [])
    writesByBuffer.get(w.buffer)!.push({ offset: w.offset, data: w.data, phase: 'decode', index: i })
  })

  // ================================================================
  // SECTION 1: SUMMARY
  // ================================================================
  sep('1. SUMMARY')
  log(`  Model: Phi-3-mini-4k-instruct (Q4F16_1)`)
  log(`  Shaders: ${cap.shaders.length} modules`)
  log(`  Pipelines: ${cap.pipelines.length} compute pipelines`)
  log(`  Prefill dispatches: ${cap.prefillDispatches.length}`)
  log(`  Prefill writes: ${cap.prefillWrites.length}`)
  log(`  Decode dispatches: ${cap.dispatches.length}`)
  log(`  Decode writes: ${cap.writes.length}`)
  log(`  Copy: ${cap.copy ? `src=${fmtBytes(cap.copy.src.size)} offset=${cap.copy.srcOffset} → dst=${fmtBytes(cap.copy.dst.size)} offset=${cap.copy.dstOffset} size=${cap.copy.size}B` : 'none'}`)

  // ================================================================
  // SECTION 2: PIPELINE CATALOG
  // ================================================================
  sep('2. PIPELINE CATALOG — All ' + cap.pipelines.length + ' pipelines')
  for (const p of cap.pipelines) {
    const shader = cap.shaders[p.shaderIndex]
    const codeLines = shader ? shader.code.split('\n').length : 0
    log(`  [${p.index}] ${p.entryPoint}`)
    log(`       shader=${p.shaderIndex} (${codeLines} lines WGSL)`)
    // Count bindings from shader source
    if (shader) {
      const bindings = shader.code.match(/@binding\(\d+\)/g) || []
      const storageR = (shader.code.match(/var<storage, read>/g) || []).length
      const storageRW = (shader.code.match(/var<storage, read_write>/g) || []).length
      const uniforms = (shader.code.match(/var<uniform>/g) || []).length
      log(`       bindings=${bindings.length} (${storageR} read, ${storageRW} read_write, ${uniforms} uniform)`)
    }
  }

  // ================================================================
  // SECTION 3: DISPATCH STRUCTURE — Prefill
  // ================================================================
  sep('3. PREFILL DISPATCH STRUCTURE (' + cap.prefillDispatches.length + ' dispatches)')

  const prefillStructure = analyzeDispatchStructure(cap.prefillDispatches, 'PREFILL')
  log(`\n  Preamble: ${prefillStructure.preambleCount} dispatches`)
  log(`  Per-layer: ${prefillStructure.layerDispatchCount} dispatches × ${prefillStructure.layerCount} layers = ${prefillStructure.layerDispatchCount * prefillStructure.layerCount}`)
  log(`  Tail: ${prefillStructure.tailCount} dispatches`)
  log(`  Total: ${prefillStructure.preambleCount + prefillStructure.layerDispatchCount * prefillStructure.layerCount + prefillStructure.tailCount}`)

  // Dump all prefill dispatches with full detail
  dumpAllDispatches(cap.prefillDispatches, 'PREFILL', cap)

  // ================================================================
  // SECTION 4: DISPATCH STRUCTURE — Decode
  // ================================================================
  sep('4. DECODE DISPATCH STRUCTURE (' + cap.dispatches.length + ' dispatches)')

  const decodeStructure = analyzeDispatchStructure(cap.dispatches, 'DECODE')
  log(`\n  Preamble: ${decodeStructure.preambleCount} dispatches`)
  log(`  Per-layer: ${decodeStructure.layerDispatchCount} dispatches × ${decodeStructure.layerCount} layers = ${decodeStructure.layerDispatchCount * decodeStructure.layerCount}`)
  log(`  Tail: ${decodeStructure.tailCount} dispatches`)
  log(`  Total: ${decodeStructure.preambleCount + decodeStructure.layerDispatchCount * decodeStructure.layerCount + decodeStructure.tailCount}`)

  dumpAllDispatches(cap.dispatches, 'DECODE', cap)

  // ================================================================
  // SECTION 5: PREFILL vs DECODE SIDE-BY-SIDE COMPARISON
  // ================================================================
  sep('5. PREFILL vs DECODE — SIDE-BY-SIDE COMPARISON')
  dumpSideBySide(cap)

  // ================================================================
  // SECTION 6: BUFFER LANDSCAPE
  // ================================================================
  sep('6. BUFFER LANDSCAPE')
  dumpBufferLandscape(cap)

  // ================================================================
  // SECTION 7: BUFFER DATA FLOW (DAG)
  // ================================================================
  sep('7. BUFFER DATA FLOW — Which dispatches share which buffers')
  dumpBufferFlow(cap)

  // ================================================================
  // SECTION 8: WRITE ANALYSIS
  // ================================================================
  sep('8. WRITE ANALYSIS')
  dumpWriteAnalysis(cap)

  // ================================================================
  // SECTION 9: UNIFORM DEEP DIVE — Every field of every uniform
  // ================================================================
  sep('9. UNIFORM DEEP DIVE — Every podArgs struct decoded')
  dumpUniformDeepDive(cap)

  // ================================================================
  // SECTION 10: COPY OPERATION
  // ================================================================
  sep('10. COPY OPERATION (Token Readback)')
  if (cap.copy) {
    const c = cap.copy
    log(`  src: ${bufId(c.src)} ${fmtBytes(c.src.size)} offset=${c.srcOffset}`)
    log(`  dst: ${bufId(c.dst)} ${fmtBytes(c.dst.size)} offset=${c.dstOffset}`)
    log(`  size: ${c.size}B`)
    log(`  Purpose: copies sampled token ID from GPU → MAP_READ buffer for CPU readback`)
  } else {
    log(`  No copy operation captured`)
  }

  log('\n\n=== DUMP COMPLETE ===')
  ;(window as unknown as Record<string, unknown>).__tvmDump = cap
  ;(window as unknown as Record<string, unknown>).__tvmDumpText = lines.join('\n')

  // Add download button
  const btn = document.createElement('button')
  btn.textContent = 'Download Full Dump'
  btn.style.cssText = 'position:fixed;top:10px;right:10px;padding:10px 20px;background:#4a9eff;color:white;border:none;border-radius:6px;cursor:pointer;font-size:14px;z-index:9999'
  btn.onclick = () => {
    const blob = new Blob([lines.join('\n')], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `tvm-dump-${Date.now()}.txt`
    a.click()
  }
  document.body.appendChild(btn)
}

// ============================================================
// Analysis Functions
// ============================================================

interface DispatchStructure {
  preambleCount: number
  layerCount: number
  layerDispatchCount: number
  tailCount: number
}

function analyzeDispatchStructure(dispatches: CapturedDispatch[], _phase: string): DispatchStructure {
  // Strategy: find repeating pattern after preamble
  // Preamble = dispatches that use unique pipelines (embed + initial norm)
  // Layer = repeating block of N dispatches
  // Tail = remaining after 32 layers

  if (dispatches.length < 10) return { preambleCount: dispatches.length, layerCount: 0, layerDispatchCount: 0, tailCount: 0 }

  // Assume preamble = 2 (embed + norm), layer = 10, 32 layers, tail = rest
  const preamble = 2
  const layerSize = 10
  const layers = 32
  const expectedBody = preamble + layerSize * layers
  const tail = dispatches.length - expectedBody

  return { preambleCount: preamble, layerCount: layers, layerDispatchCount: layerSize, tailCount: tail }
}

function dumpAllDispatches(dispatches: CapturedDispatch[], phase: string, cap: CaptureResult): void {
  const pName = (p: GPUComputePipeline) => {
    for (const pp of cap.pipelines) { if (pp.pipeline === p) return pp.entryPoint }
    return 'UNKNOWN'
  }

  const writes = phase === 'PREFILL' ? cap.prefillWrites : cap.writes

  for (let i = 0; i < dispatches.length; i++) {
    const d = dispatches[i]
    const name = pName(d.pipeline)
    const wg = d.workgroups

    // Determine position in structure
    let label = ''
    if (i < 2) {
      label = i === 0 ? 'PREAMBLE:Embed' : 'PREAMBLE:InitNorm'
    } else if (i < 322) {
      const layer = Math.floor((i - 2) / 10)
      const step = (i - 2) % 10
      const stepNames = ['QKV', 'RoPE', 'KVAppend', 'Attention', 'OProj', 'AddNorm1', 'FFNUp', 'SiLU', 'FFNDown', 'AddNorm2']
      label = `L${layer}:${stepNames[step]}`
    } else {
      label = `TAIL:${i - 322}`
    }

    log(`\n  [${i}] ${label} — ${name}`)
    log(`       workgroups: [${wg.join(', ')}] = ${wg[0] * wg[1] * wg[2]} total`)

    // Bindings
    for (const e of d.entries) {
      const id = bufId(e.buffer)
      const type = bufType(e.buffer)
      const sz = fmtBytes(e.buffer.size)

      // Find write data for this buffer
      let uniformStr = ''
      if (type === 'UNIFORM') {
        const w = writes.find(w => w.buffer === e.buffer)
        if (w) {
          uniformStr = ' → ' + decodeUniform(w.data, name)
        }
      }

      log(`       @binding(${e.binding}): ${id} ${type} ${sz} [offset=${e.offset}, size=${fmtBytes(e.size)}]${uniformStr}`)
    }
  }
}

function dumpSideBySide(cap: CaptureResult): void {
  const pName = (p: GPUComputePipeline) => {
    for (const pp of cap.pipelines) { if (pp.pipeline === p) return pp.entryPoint }
    return 'UNKNOWN'
  }

  const prefill = cap.prefillDispatches
  const decode = cap.dispatches

  // Preamble comparison
  subsep('Preamble (dispatch 0-1)')
  for (let i = 0; i < 2; i++) {
    const pD = prefill[i]
    const dD = decode[i]
    if (!pD || !dD) continue
    const pN = pName(pD.pipeline)
    const dN = pName(dD.pipeline)
    const same = pN === dN ? '✓ SAME' : '✗ DIFFERENT'
    log(`  [${i}] ${same}`)
    log(`       Prefill: ${pN} wg=[${pD.workgroups.join(',')}]`)
    log(`       Decode:  ${dN} wg=[${dD.workgroups.join(',')}]`)
    if (pN !== dN) log(`       *** PIPELINE DIFFERS ***`)
    if (pD.workgroups[0] !== dD.workgroups[0]) {
      log(`       *** WORKGROUP X: ${pD.workgroups[0]} vs ${dD.workgroups[0]} (ratio: ${(pD.workgroups[0] / dD.workgroups[0]).toFixed(1)}x) ***`)
    }

    // Compare bindings
    const maxBindings = Math.max(pD.entries.length, dD.entries.length)
    for (let b = 0; b < maxBindings; b++) {
      const pe = pD.entries[b]
      const de = dD.entries[b]
      if (pe && de) {
        const sameBuf = pe.buffer === de.buffer ? 'SHARED' : 'DIFFERENT'
        const sameSize = pe.buffer.size === de.buffer.size ? '' : ` size:${fmtBytes(pe.buffer.size)} vs ${fmtBytes(de.buffer.size)}`
        log(`       @${pe.binding}: ${sameBuf}${sameSize} ${bufId(pe.buffer)} / ${bufId(de.buffer)}`)
      }
    }
  }

  // Per-layer comparison (show layer 0 in full, then summarize differences)
  subsep('Layer 0 — Full Comparison (dispatch 2-11)')
  const stepNames = ['QKV', 'RoPE', 'KVAppend', 'Attention', 'OProj', 'AddNorm1', 'FFNUp', 'SiLU', 'FFNDown', 'AddNorm2']
  for (let step = 0; step < 10; step++) {
    const pIdx = 2 + step
    const dIdx = 2 + step
    const pD = prefill[pIdx]
    const dD = decode[dIdx]
    if (!pD || !dD) continue

    const pN = pName(pD.pipeline)
    const dN = pName(dD.pipeline)
    const same = pN === dN ? '✓' : '✗'

    log(`\n  ${stepNames[step]} [${step}] ${same}`)
    log(`       Prefill: ${pN} wg=[${pD.workgroups.join(',')}]`)
    log(`       Decode:  ${dN} wg=[${dD.workgroups.join(',')}]`)

    // Workgroup ratio
    if (pD.workgroups[0] !== dD.workgroups[0]) {
      log(`       WG ratio: ${pD.workgroups[0]} / ${dD.workgroups[0]} = ${(pD.workgroups[0] / dD.workgroups[0]).toFixed(1)}x`)
    }

    // Binding comparison
    const maxB = Math.max(pD.entries.length, dD.entries.length)
    for (let b = 0; b < maxB; b++) {
      const pe = pD.entries.find(e => e.binding === b)
      const de = dD.entries.find(e => e.binding === b)
      if (pe && de) {
        const sameBuf = pe.buffer === de.buffer ? 'SHARED' : 'DIFF'
        const sameOff = pe.offset === de.offset ? '' : ` off:${pe.offset}/${de.offset}`
        const sameSize = pe.size === de.size ? '' : ` sz:${fmtBytes(pe.size)}/${fmtBytes(de.size)}`
        log(`       @${b}: ${sameBuf} ${bufType(pe.buffer)} ${fmtBytes(pe.buffer.size)}${sameOff}${sameSize}`)
      } else if (pe) {
        log(`       @${b}: PREFILL-ONLY ${bufType(pe.buffer)} ${fmtBytes(pe.buffer.size)}`)
      } else if (de) {
        log(`       @${b}: DECODE-ONLY ${bufType(de.buffer)} ${fmtBytes(de.buffer.size)}`)
      }
    }

    // Uniform comparison
    const pUniform = pD.entries.reduce((max, e) => e.binding > max.binding ? e : max, pD.entries[0])
    const dUniform = dD.entries.reduce((max, e) => e.binding > max.binding ? e : max, dD.entries[0])
    if (pUniform && dUniform && bufType(pUniform.buffer) === 'UNIFORM' && bufType(dUniform.buffer) === 'UNIFORM') {
      const pWrite = cap.prefillWrites.find(w => w.buffer === pUniform.buffer)
      const dWrite = cap.writes.find(w => w.buffer === dUniform.buffer)
      if (pWrite && dWrite) {
        log(`       Prefill uniform: ${decodeUniform(pWrite.data, pN)}`)
        log(`       Decode  uniform: ${decodeUniform(dWrite.data, dN)}`)
      }
    }
  }

  // Summarize layer differences across all 32 layers
  subsep('Layer Consistency Check (layers 0-31)')
  for (let step = 0; step < 10; step++) {
    const prefillPipelines = new Set<string>()
    const decodePipelines = new Set<string>()
    const prefillWGs: number[] = []
    const decodeWGs: number[] = []

    for (let L = 0; L < 32; L++) {
      const pD = prefill[2 + L * 10 + step]
      const dD = decode[2 + L * 10 + step]
      if (pD) { prefillPipelines.add(pName(pD.pipeline)); prefillWGs.push(pD.workgroups[0]) }
      if (dD) { decodePipelines.add(pName(dD.pipeline)); decodeWGs.push(dD.workgroups[0]) }
    }

    const pConsistent = prefillPipelines.size === 1 ? '✓' : `✗(${prefillPipelines.size} variants)`
    const dConsistent = decodePipelines.size === 1 ? '✓' : `✗(${decodePipelines.size} variants)`
    const pWGConsistent = new Set(prefillWGs).size === 1 ? `wg=${prefillWGs[0]}` : `wg varies: ${[...new Set(prefillWGs)].join(',')}`
    const dWGConsistent = new Set(decodeWGs).size === 1 ? `wg=${decodeWGs[0]}` : `wg varies: ${[...new Set(decodeWGs)].join(',')}`

    log(`  ${stepNames[step].padEnd(10)} | Prefill: ${pConsistent} ${pWGConsistent} | Decode: ${dConsistent} ${dWGConsistent}`)
  }

  // Tail comparison
  subsep('Tail Comparison')
  const prefillTailStart = 2 + 32 * 10
  const decodeTailStart = 2 + 32 * 10
  const prefillTail = prefill.slice(prefillTailStart)
  const decodeTail = decode.slice(decodeTailStart)
  log(`  Prefill tail: ${prefillTail.length} dispatches (starting at index ${prefillTailStart})`)
  log(`  Decode tail:  ${decodeTail.length} dispatches (starting at index ${decodeTailStart})`)

  const maxTail = Math.max(prefillTail.length, decodeTail.length)
  for (let i = 0; i < maxTail; i++) {
    const pD = prefillTail[i]
    const dD = decodeTail[i]
    const pN = pD ? pName(pD.pipeline) : '-'
    const dN = dD ? pName(dD.pipeline) : '-'
    const same = pN === dN ? '✓' : (pD && dD ? '✗' : ' ')
    log(`  [${i}] ${same} Prefill: ${pN}${pD ? ` wg=[${pD.workgroups}]` : ''} | Decode: ${dN}${dD ? ` wg=[${dD.workgroups}]` : ''}`)
  }
}

function dumpBufferLandscape(cap: CaptureResult): void {
  const bufInfo = new Map<GPUBuffer, {
    type: string
    size: number
    usedInPrefill: Set<number>
    usedInDecode: Set<number>
    bindings: Map<string, Set<number>>  // pipeline → set of binding indices
  }>()

  for (let i = 0; i < cap.prefillDispatches.length; i++) {
    const d = cap.prefillDispatches[i]
    const name = cap.pipelines.find(p => p.pipeline === d.pipeline)?.entryPoint ?? '?'
    for (const e of d.entries) {
      if (!bufInfo.has(e.buffer)) bufInfo.set(e.buffer, { type: bufType(e.buffer), size: e.buffer.size, usedInPrefill: new Set(), usedInDecode: new Set(), bindings: new Map() })
      const info = bufInfo.get(e.buffer)!
      info.usedInPrefill.add(i)
      if (!info.bindings.has(name)) info.bindings.set(name, new Set())
      info.bindings.get(name)!.add(e.binding)
    }
  }
  for (let i = 0; i < cap.dispatches.length; i++) {
    const d = cap.dispatches[i]
    const name = cap.pipelines.find(p => p.pipeline === d.pipeline)?.entryPoint ?? '?'
    for (const e of d.entries) {
      if (!bufInfo.has(e.buffer)) bufInfo.set(e.buffer, { type: bufType(e.buffer), size: e.buffer.size, usedInPrefill: new Set(), usedInDecode: new Set(), bindings: new Map() })
      const info = bufInfo.get(e.buffer)!
      info.usedInDecode.add(i)
      if (!info.bindings.has(name)) info.bindings.set(name, new Set())
      info.bindings.get(name)!.add(e.binding)
    }
  }

  // Group by type
  const groups: Record<string, Array<[GPUBuffer, typeof bufInfo extends Map<GPUBuffer, infer V> ? V : never]>> = {}
  for (const [buf, info] of bufInfo) {
    if (!groups[info.type]) groups[info.type] = []
    groups[info.type].push([buf, info])
  }

  for (const [type, bufs] of Object.entries(groups)) {
    bufs.sort((a, b) => b[1].size - a[1].size)
    const totalSize = bufs.reduce((s, [, info]) => s + info.size, 0)
    const shared = bufs.filter(([, info]) => info.usedInPrefill.size > 0 && info.usedInDecode.size > 0).length
    const prefillOnly = bufs.filter(([, info]) => info.usedInPrefill.size > 0 && info.usedInDecode.size === 0).length
    const decodeOnly = bufs.filter(([, info]) => info.usedInPrefill.size === 0 && info.usedInDecode.size > 0).length

    subsep(`${type}: ${bufs.length} buffers, ${fmtBytes(totalSize)} total`)
    log(`  Shared (prefill+decode): ${shared}`)
    log(`  Prefill-only: ${prefillOnly}`)
    log(`  Decode-only: ${decodeOnly}`)

    // Show each buffer (limit large groups)
    const showLimit = type === 'WEIGHT' ? 20 : type === 'UNIFORM' ? 50 : bufs.length
    for (let i = 0; i < Math.min(bufs.length, showLimit); i++) {
      const [buf, info] = bufs[i]
      const phase = info.usedInPrefill.size > 0 && info.usedInDecode.size > 0 ? 'SHARED'
        : info.usedInPrefill.size > 0 ? 'PREFILL-ONLY' : 'DECODE-ONLY'
      const pipeList = [...info.bindings.entries()].map(([p, bs]) => `${p}@${[...bs].join(',')}`).join(' | ')
      log(`  ${bufId(buf)} ${fmtBytes(info.size)} ${phase} dispatches:P${info.usedInPrefill.size}/D${info.usedInDecode.size}`)
      log(`       → ${pipeList}`)
    }
    if (bufs.length > showLimit) log(`  ... and ${bufs.length - showLimit} more`)
  }
}

function dumpBufferFlow(cap: CaptureResult): void {
  // For decode phase: show which dispatches WRITE (binding 0 = output) and READ each buffer
  const pName = (p: GPUComputePipeline) => {
    for (const pp of cap.pipelines) { if (pp.pipeline === p) return pp.entryPoint }
    return '?'
  }

  subsep('Decode Phase — Buffer Flow (first 2 layers + tail)')
  const bufFlowMap = new Map<GPUBuffer, { writers: string[]; readers: string[] }>()

  for (let i = 0; i < Math.min(cap.dispatches.length, 22 + 20); i++) {
    const d = cap.dispatches[i]
    const name = `[${i}]${pName(d.pipeline).substring(0, 30)}`
    for (const e of d.entries) {
      if (!bufFlowMap.has(e.buffer)) bufFlowMap.set(e.buffer, { writers: [], readers: [] })
      const info = bufFlowMap.get(e.buffer)!
      // Binding 0 is typically the output (read_write) in TVM convention
      if (e.binding === 0) info.writers.push(name)
      else info.readers.push(name)
    }
  }

  // Show buffers that are written by one dispatch and read by another (data flow edges)
  const flowEdges: Array<{ buffer: string; size: number; from: string; to: string }> = []
  for (const [buf, info] of bufFlowMap) {
    if (info.writers.length > 0 && info.readers.length > 0) {
      for (const w of info.writers) {
        for (const r of info.readers) {
          flowEdges.push({ buffer: bufId(buf), size: buf.size, from: w, to: r })
        }
      }
    }
  }

  // Show activation-sized flow edges (most interesting)
  const activationEdges = flowEdges.filter(e => e.size > 64 && e.size <= 1_000_000)
  activationEdges.sort((a, b) => {
    const aIdx = parseInt(a.from.match(/\[(\d+)\]/)?.[1] ?? '999')
    const bIdx = parseInt(b.from.match(/\[(\d+)\]/)?.[1] ?? '999')
    return aIdx - bIdx
  })

  for (const e of activationEdges) {
    log(`  ${e.from} → ${e.buffer} (${fmtBytes(e.size)}) → ${e.to}`)
  }
}

function dumpWriteAnalysis(cap: CaptureResult): void {
  subsep('Prefill Writes (' + cap.prefillWrites.length + ')')
  dumpWriteList(cap.prefillWrites, 'prefill')

  subsep('Decode Writes (' + cap.writes.length + ')')
  dumpWriteList(cap.writes, 'decode')

  // Compare: which buffers get written in BOTH prefill and decode?
  subsep('Write Buffer Overlap')
  const prefillBufs = new Set(cap.prefillWrites.map(w => w.buffer))
  const decodeBufs = new Set(cap.writes.map(w => w.buffer))
  let shared = 0, prefillOnly = 0, decodeOnly = 0
  const allBufs = new Set([...prefillBufs, ...decodeBufs])
  for (const buf of allBufs) {
    const inP = prefillBufs.has(buf)
    const inD = decodeBufs.has(buf)
    if (inP && inD) shared++
    else if (inP) prefillOnly++
    else decodeOnly++
  }
  log(`  Written in both: ${shared}`)
  log(`  Prefill-only writes: ${prefillOnly}`)
  log(`  Decode-only writes: ${decodeOnly}`)

  // Identify the 6 changing values (from phi3.ts analysis)
  subsep('Per-Token Changing Values (decode writes)')
  log('  These are the writes that MUST change per decode token:')
  log('  (Identified by position in write sequence + buffer purpose)')
  const w = cap.writes
  if (w.length > 12) {
    const show = (idx: number, label: string) => {
      if (idx >= w.length) return
      const wr = w[idx]
      const vals: number[] = []
      const dv = new DataView(wr.data.buffer, wr.data.byteOffset, wr.data.byteLength)
      for (let b = 0; b < Math.min(wr.data.length, 16); b += 4) {
        if (b + 4 <= wr.data.length) vals.push(dv.getInt32(b, true))
      }
      log(`  write[${idx}] ${label}: ${bufId(wr.buffer)} ${fmtBytes(wr.buffer.size)} offset=${wr.offset} ${wr.data.length}B → [${vals.join(', ')}]`)
    }
    show(0, 'TOKEN_ID (input token for embedding)')
    show(4, 'POSITION (RoPE position_map)')
    show(8, 'SEQUENCE_COUNTER (increments each token)')
    show(11, 'POSITION (attention q_rope_position)')
    show(12, 'POSITION (attention length_info)')
    if (w.length > 349) show(349, 'SAMPLING_SEED (random f32)')
  }
}

function dumpWriteList(writes: CapturedWrite[], _phase: string): void {
  // Group writes by buffer type
  const byType: Record<string, number> = {}
  let totalBytes = 0
  for (const w of writes) {
    const type = bufType(w.buffer)
    byType[type] = (byType[type] ?? 0) + 1
    totalBytes += w.data.length
  }
  log(`  Total: ${writes.length} writes, ${fmtBytes(totalBytes)}`)
  for (const [type, count] of Object.entries(byType)) {
    log(`    ${type}: ${count} writes`)
  }

  // Show first 30 and last 10
  const show = (w: CapturedWrite, idx: number) => {
    const dv = new DataView(w.data.buffer, w.data.byteOffset, w.data.byteLength)
    const vals: string[] = []
    for (let b = 0; b < Math.min(w.data.length, 24); b += 4) {
      if (b + 4 <= w.data.length) {
        const iv = dv.getInt32(b, true)
        const fv = dv.getFloat32(b, true)
        // Show float interpretation if it looks like a float
        if (Math.abs(fv) > 0.001 && Math.abs(fv) < 100000 && !Number.isNaN(fv) && (iv > 100000 || iv < -100000)) {
          vals.push(`${iv}(f:${fv.toFixed(4)})`)
        } else {
          vals.push(`${iv}`)
        }
      }
    }
    const trunc = w.data.length > 24 ? ` ...+${w.data.length - 24}B` : ''
    log(`  [${idx}] ${bufId(w.buffer)} ${bufType(w.buffer)} ${fmtBytes(w.buffer.size)} off=${w.offset} ${w.data.length}B → [${vals.join(', ')}]${trunc}`)
  }

  for (let i = 0; i < Math.min(writes.length, 30); i++) show(writes[i], i)
  if (writes.length > 40) log(`  ... (${writes.length - 40} more) ...`)
  for (let i = Math.max(30, writes.length - 10); i < writes.length; i++) show(writes[i], i)
}

function dumpUniformDeepDive(cap: CaptureResult): void {
  const pName = (p: GPUComputePipeline) => {
    for (const pp of cap.pipelines) { if (pp.pipeline === p) return pp.entryPoint }
    return 'UNKNOWN'
  }

  // For decode: dump every unique uniform
  subsep('Decode Uniforms — Every dispatch')
  const seenUniforms = new Set<string>()  // dedup by buffer identity

  for (let i = 0; i < cap.dispatches.length; i++) {
    const d = cap.dispatches[i]
    const name = pName(d.pipeline)

    // Find the uniform binding (highest binding number, small buffer)
    const uniformEntry = d.entries
      .filter(e => e.buffer.size <= 64)
      .sort((a, b) => b.binding - a.binding)[0]

    if (!uniformEntry) continue

    const bufKey = `${bufId(uniformEntry.buffer)}`
    if (seenUniforms.has(bufKey)) continue
    seenUniforms.add(bufKey)

    // Find write data
    const w = cap.writes.find(w => w.buffer === uniformEntry.buffer)
    if (!w) continue

    // Label
    let label = ''
    if (i < 2) label = i === 0 ? 'Embed' : 'InitNorm'
    else if (i < 322) {
      const layer = Math.floor((i - 2) / 10)
      const step = (i - 2) % 10
      const stepNames = ['QKV', 'RoPE', 'KVAppend', 'Attention', 'OProj', 'AddNorm1', 'FFNUp', 'SiLU', 'FFNDown', 'AddNorm2']
      label = `L${layer}:${stepNames[step]}`
    } else label = `Tail:${i - 322}`

    log(`\n  ${label} — ${name}`)
    log(`       ${bufKey} ${w.data.length}B`)
    log(`       ${decodeUniform(w.data, name)}`)

    // Also show raw hex
    const hex = Array.from(w.data.slice(0, 64)).map(b => b.toString(16).padStart(2, '0')).join(' ')
    log(`       hex: ${hex}`)
  }

  // Prefill uniforms that are DIFFERENT from decode
  subsep('Prefill Uniforms — Differences from Decode')
  const prefillSeenUniforms = new Set<string>()

  for (let i = 0; i < cap.prefillDispatches.length; i++) {
    const d = cap.prefillDispatches[i]
    const name = pName(d.pipeline)

    const uniformEntry = d.entries
      .filter(e => e.buffer.size <= 64)
      .sort((a, b) => b.binding - a.binding)[0]

    if (!uniformEntry) continue

    const bufKey = bufId(uniformEntry.buffer)
    if (prefillSeenUniforms.has(bufKey)) continue
    prefillSeenUniforms.add(bufKey)

    const w = cap.prefillWrites.find(w => w.buffer === uniformEntry.buffer)
    if (!w) continue

    // Only show if this buffer is NOT shared with decode
    if (seenUniforms.has(bufKey)) continue

    let label = ''
    if (i < 2) label = i === 0 ? 'Embed' : 'InitNorm'
    else if (i < 322) {
      const layer = Math.floor((i - 2) / 10)
      const step = (i - 2) % 10
      const stepNames = ['QKV', 'RoPE', 'KVAppend', 'Attention', 'OProj', 'AddNorm1', 'FFNUp', 'SiLU', 'FFNDown', 'AddNorm2']
      label = `L${layer}:${stepNames[step]}`
    } else label = `Tail:${i - 322}`

    log(`\n  ${label} — ${name}`)
    log(`       ${bufKey} ${w.data.length}B`)
    log(`       ${decodeUniform(w.data, name)}`)

    const hex = Array.from(w.data.slice(0, 64)).map(b => b.toString(16).padStart(2, '0')).join(' ')
    log(`       hex: ${hex}`)
  }
}

run().catch(e => {
  log(`FATAL: ${e}`)
  if (String(e).includes('Cache')) {
    log('')
    log('This is a WebLLM/tvmjs caching error — usually a transient network issue.')
    log('Try: reload the page, or clear site data and retry.')
  }
})
