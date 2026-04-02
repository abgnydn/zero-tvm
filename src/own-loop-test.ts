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

  // Grab tokenizer before TVM gets corrupted
  const decoder = engine.getDecoder()
  if (decoder) log('Tokenizer grabbed')

  // Phase 2: Let TVM generate, capture tokens, abort after engine ready
  log('')
  log('=== TVM warmup + capture ===')
  const prompt = 'Write a detailed paragraph about the history of the internet from its origins to the modern day.'

  const tvmTokenIds: number[] = []
  // Capture TVM's token IDs via getMappedRange
  const origGetMapped = GPUBuffer.prototype.getMappedRange
  GPUBuffer.prototype.getMappedRange = function(offset?: number, size?: number) {
    const range = origGetMapped.call(this, offset, size)
    if (range.byteLength >= 4 && range.byteLength <= 8) {
      const val = new DataView(range).getInt32(0, true)
      if (val >= 0 && val < 100000) tvmTokenIds.push(val)
    }
    return range
  }

  const readyPromise = new Promise<void>(resolve => {
    onReady(() => {
      log(`[own-loop] Engine ready after ${tvmTokenIds.length} TVM tokens`)
      log(`TVM tokens: [${tvmTokenIds.join(', ')}]`)
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

  if (decoder && tokens.length > 0) {
    const text = decoder(new Int32Array(tokens))
    log(`\nGenerated text:\n${text}`)
  }
  log('')
  log('Done.')
}

run().catch(e => log(`FATAL: ${e}`))
