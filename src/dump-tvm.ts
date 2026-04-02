/**
 * DUMP TVM — Complete reverse engineering of TVM's dispatch architecture.
 *
 * For each of the 342 dispatches, dump:
 * - Pipeline entry point name
 * - Every bind group entry (binding index, buffer size, data offset, data size)
 * - The uniform buffer contents (from captured writes)
 * - Workgroup dimensions
 *
 * Output: a complete map of what TVM does, in what order, with what data.
 */

import { LLMEngine, MODELS } from './engine.js'
import { patchForCapture, getCaptureResult } from './capture.js'

const log = (msg: string) => {
  const el = document.getElementById('log')!
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

async function run(): Promise<void> {
  log('=== TVM Architecture Dump ===')
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

  log(`\nCaptured: ${cap.dispatches.length} dispatches, ${cap.writes.length} writes, ${cap.pipelines.length} pipelines`)

  // Build buffer → write data map
  const writesByBuffer = new Map<GPUBuffer, Array<{ offset: number; data: Uint8Array }>>()
  for (const w of cap.writes) {
    if (!writesByBuffer.has(w.buffer)) writesByBuffer.set(w.buffer, [])
    writesByBuffer.get(w.buffer)!.push({ offset: w.offset, data: w.data })
  }

  // Map pipeline objects to entry point names
  const pipelineNames = new Map<GPUComputePipeline, string>()
  for (const p of cap.pipelines) {
    pipelineNames.set(p.pipeline, p.entryPoint)
  }

  log('\n' + '='.repeat(80))
  log('DISPATCH SEQUENCE')
  log('='.repeat(80))

  for (let i = 0; i < cap.dispatches.length; i++) {
    const d = cap.dispatches[i]
    const name = pipelineNames.get(d.pipeline) ?? 'unknown'
    const wg = d.workgroups

    // Determine layer context
    let context = ''
    if (i < 2) context = 'PREAMBLE'
    else if (i < 322) {
      const layer = Math.floor((i - 2) / 10)
      const step = (i - 2) % 10
      const stepNames = ['QKV', 'RoPE', 'KV_append', 'Attention', 'O_proj', 'AddNorm1', 'FFN_up', 'SiLU', 'FFN_down', 'AddNorm2']
      context = `Layer ${layer} / ${stepNames[step]}`
    } else {
      context = `TAIL (LM head + sampling) [${i - 322}]`
    }

    log(`\n--- Dispatch ${i}: ${context} ---`)
    log(`  Pipeline: ${name}`)
    log(`  Workgroups: [${wg.join(', ')}]`)
    log(`  Bindings: ${d.entries.length}`)

    for (const e of d.entries) {
      const bufSize = e.buffer.size
      const dataSize = e.size
      const offset = e.offset

      // Classify buffer
      let bufType = 'unknown'
      if (bufSize > 1_000_000) bufType = 'WEIGHT'
      else if (bufSize > 10_000) bufType = 'KV_CACHE'
      else if (bufSize <= 64) bufType = 'UNIFORM'
      else bufType = 'ACTIVATION'

      // Get uniform data from captured writes
      let uniformData = ''
      if (bufType === 'UNIFORM') {
        const writes = writesByBuffer.get(e.buffer)
        if (writes && writes.length > 0) {
          const w = writes[0]
          const vals: number[] = []
          for (let b = 0; b < w.data.length; b += 4) {
            if (b + 4 <= w.data.length) {
              vals.push(w.data[b] | (w.data[b+1] << 8) | (w.data[b+2] << 16) | (w.data[b+3] << 24))
            }
          }
          uniformData = ` DATA=[${vals.join(', ')}]`
        }
      }

      log(`    @binding(${e.binding}): ${bufType} buf=${bufSize}B offset=${offset} size=${dataSize}${uniformData}`)
    }

    // Only dump first 2 layers + preamble + first 3 tail in detail
    if (i === 1) log('\n  ... (layers 0-31 follow same pattern, showing layer 0 and 1) ...')
    if (i >= 22 && i < 322) {
      if (i === 22) log('\n  ... (layers 2-31 same pattern, skipping to tail) ...')
      continue
    }
    if (i >= 325 && i < cap.dispatches.length - 1) {
      if (i === 325) log('\n  ... (remaining tail dispatches) ...')
      continue
    }
  }

  // Dump the write sequence
  log('\n' + '='.repeat(80))
  log('WRITE SEQUENCE (357 writeBuffer calls)')
  log('='.repeat(80))

  for (let i = 0; i < cap.writes.length; i++) {
    const w = cap.writes[i]
    const vals: number[] = []
    for (let b = 0; b < Math.min(w.data.length, 32); b += 4) {
      if (b + 4 <= w.data.length) {
        vals.push(w.data[b] | (w.data[b+1] << 8) | (w.data[b+2] << 16) | (w.data[b+3] << 24))
      }
    }
    const truncated = w.data.length > 32 ? ` ...+${w.data.length - 32}B` : ''

    // Only show first 30 writes and last 10 (the per-layer ones repeat)
    if (i < 30 || i >= 347) {
      log(`  write[${i}]: buf=${w.buffer.size}B offset=${w.offset} ${w.data.length}B → [${vals.join(', ')}]${truncated}`)
    } else if (i === 30) {
      log(`  ... (writes 30-346 are per-layer uniforms, same pattern) ...`)
    }
  }

  // Dump copy operation
  if (cap.copy) {
    log(`\nCOPY: src=${cap.copy.src.size}B+${cap.copy.srcOffset} → dst=${cap.copy.dst.size}B+${cap.copy.dstOffset} size=${cap.copy.size}`)
  }

  log('\nDone.')

  // Also expose on window for devtools access
  ;(window as unknown as Record<string, unknown>).__tvmDump = cap
}

run().catch(e => log(`FATAL: ${e}`))
