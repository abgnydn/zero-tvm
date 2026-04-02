/**
 * Own Loop Test — Capture, generate, compare.
 *
 * Flow:
 * 1. Load model via WebLLM
 * 2. Chat once — capture first decode token, abort immediately
 * 3. Run own decode loop on clean KV cache
 * 4. Log results (no TVM baseline comparison — TVM is corrupted after our loop)
 */

import { LLMEngine, MODELS } from './engine.js'
import { setOwnLoopEnabled, patchDeviceOwnLoop, generate, onReady, getOwnLoopStats } from './own-loop.js'

const log = (msg: string) => {
  const el = document.getElementById('log')!
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

async function run(): Promise<void> {
  log('=== Own Decode Loop Test ===')
  log('')

  // Phase 1: Load model
  const engine = new LLMEngine()
  setOwnLoopEnabled(true)

  if (navigator.gpu) {
    const origRA = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function(...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRA(...args)
      if (!adapter) return adapter
      const origRD = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function(...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRD(...dArgs)
        patchDeviceOwnLoop(device)
        log('Device patched')
        return device
      }
      return adapter
    }
  }

  log('Loading model...')
  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => log(msg))

  // Phase 2: Capture — chat, abort after first decode token
  log('')
  log('=== Capturing first decode token ===')
  const prompt = 'Write a detailed paragraph about the history of the internet from its origins to the modern day.'

  const readyPromise = new Promise<void>(resolve => {
    onReady(() => {
      log('[own-loop] Engine ready — aborting TVM')
      resolve()
    })
  })

  try {
    await engine.chat(prompt, () => {
      if (getOwnLoopStats().includes('ready')) throw new Error('abort')
    })
  } catch {
    // Expected abort
  }

  await readyPromise
  log(getOwnLoopStats())

  // Phase 3: Run own decode loop
  log('')
  log('=== Generating ===')
  const t0 = performance.now()
  const tokens = await generate(200)
  const elapsed = performance.now() - t0
  const tokPerSec = tokens.length / (elapsed / 1000)

  log(`${tokens.length} tokens in ${(elapsed / 1000).toFixed(1)}s = ${tokPerSec.toFixed(1)} tok/s`)
  log(`Token IDs: [${tokens.slice(0, 30).join(', ')}${tokens.length > 30 ? '...' : ''}]`)

  // Decode tokens to text using a simple lookup
  // (We can't use TVM's tokenizer since TVM is corrupted)
  log('')
  log('Done. TVM baseline not available (state corrupted after own-loop).')
  log('To compare: run baseline separately at /?clean=1 with same prompt.')
}

run().catch(e => log(`FATAL: ${e}`))
