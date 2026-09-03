/**
 * DUMP KEY SHADER SOURCES — outputs the actual WGSL for FFNUp + SiLU + FFNDown
 * so we can analyze TVM's weight layout and split pattern for fusion.
 */

import { LLMEngine, MODELS } from './engine.js'
import { patchForCapture, getCaptureResult } from './capture.js'

const log = (msg: string) => {
  const el = document.getElementById('log')!
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

async function run(): Promise<void> {
  log('Loading model to capture shaders...')

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
  await engine.chat('hi', () => {})

  const cap = getCaptureResult()
  if (!cap) { log('ERROR: No capture'); return }

  // Dump ALL 85 shaders
  for (const pipeline of cap.pipelines) {
    const shader = cap.shaders[pipeline.shaderIndex]
    if (!shader) { log(`\n--- pipeline ${pipeline.index}: shader ${pipeline.shaderIndex} NOT FOUND ---`); continue }

    log(`\n${'='.repeat(80)}`)
    log(`  [${pipeline.index}] ${pipeline.entryPoint} (shader #${pipeline.shaderIndex}, ${shader.code.split('\n').length} lines)`)
    log(`${'='.repeat(80)}`)
    log(shader.code)
  }

  log('\n\nDone. All key shader sources dumped.')

  // Download button
  const btn = document.createElement('button')
  btn.textContent = 'Download Shaders'
  btn.style.cssText = 'position:fixed;top:10px;right:10px;padding:10px 20px;background:#4a9eff;color:white;border:none;border-radius:6px;cursor:pointer;z-index:9999'
  btn.onclick = () => {
    const el = document.getElementById('log')!
    const blob = new Blob([el.textContent ?? ''], { type: 'text/plain' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `tvm-shaders-${Date.now()}.txt`
    a.click()
  }
  document.body.appendChild(btn)
}

run().catch(e => log(`FATAL: ${e}`))
