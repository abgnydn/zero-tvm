/**
 * Standalone Engine Test
 *
 * TVM loads model + prefills + generates 3 warmup tokens.
 * Capture intercepts everything.
 * Standalone engine builds with own buffers.
 * Generates 100 tokens and decodes to text.
 * No own-loop — standalone only.
 */

import { LLMEngine, MODELS } from './engine.js'
import { patchForCapture, getCaptureResult } from './capture.js'
import { buildPhi3Engine } from './phi3.js'

const log = (msg: string) => {
  const el = document.getElementById('log')!
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

async function run(): Promise<void> {
  log('=== Standalone Engine Test ===')

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

  const decode = engine.getDecoder()
  if (decode) log('Tokenizer grabbed')

  // Let TVM generate — this triggers capture
  log('')
  log('=== TVM chat (triggers capture) ===')
  const prompt = 'What is the capital of France?'
  const tvmTokens: string[] = []
  await engine.chat(prompt, (tok) => tvmTokens.push(tok))
  log(`TVM: ${tvmTokens.join('')}`)

  // Build standalone engine
  const cap = getCaptureResult()
  if (!cap || cap.dispatches.length === 0 || cap.writes.length === 0) {
    log('ERROR: Capture incomplete')
    return
  }
  log(`Captured: ${cap.dispatches.length} dispatches, ${cap.writes.length} writes, copy=${cap.copy ? 'yes' : 'no'}`)

  const standalone = buildPhi3Engine(cap)

  // Generate with standalone engine
  log('')
  log('=== Standalone generate ===')
  const t0 = performance.now()
  const tokens = await standalone.generate(200, (id) => {
    void id
  })
  const elapsed = performance.now() - t0
  const tokPerSec = tokens.length / (elapsed / 1000)
  log(`${tokens.length} tokens in ${(elapsed / 1000).toFixed(1)}s = ${tokPerSec.toFixed(1)} tok/s`)
  log(`IDs: [${tokens.slice(0, 30).join(', ')}${tokens.length > 30 ? '...' : ''}]`)

  if (decode && tokens.length > 0) {
    log(`\nText: ${decode(new Int32Array(tokens))}`)
  }

  log('\nDone.')
}

run().catch(e => log(`FATAL: ${e}`))
