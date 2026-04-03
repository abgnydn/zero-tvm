/**
 * COMPILER CHAT — Chat UI powered by our own compiler.
 *
 * Message 1: WebLLM loads model + generates (we capture weights)
 * Message 2+: Our compiler drives everything. Zero TVM runtime.
 */

import { LLMEngine, MODELS } from '../engine.js'
import { ChatUI } from '../ui.js'
import { patchForCapture, getCaptureResult } from '../capture.js'
import { buildFromCapture } from './model.js'
import type { Runtime } from './runtime.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const llm = new LLMEngine()

  ui.appendLog('Loading model...')

  // Intercept WebGPU to capture shaders + weights
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
  ui.setInitialMessage('Own Compiler\nMsg 1: TVM (captures weights)\nMsg 2+: Our shaders, zero TVM')

  let runtime: Runtime | null = null
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
      if (!runtime) {
        // First message: TVM generates, we capture
        ui.appendLog('TVM generating (capturing weights)...')
        await llm.chat(text, (tok) => { count++; ui.appendToken(tok); stats() })

        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0) {
          try {
            const result = buildFromCapture(cap)
            runtime = result.runtime
            ui.appendLog('Own compiler ready — zero TVM from now on')
          } catch (e) {
            ui.appendLog(`Compiler build failed: ${e}`)
          }
        }
      } else if (tokenizer) {
        // Our compiler: encode prompt, run our runtime
        const promptIds = Array.from(tokenizer.encode(text))
        ui.appendLog(`Encoding ${promptIds.length} tokens, running own compiler...`)

        const allIds: number[] = []
        let prevText = ''
        await runtime.generate(promptIds, 500, (id) => {
          count++
          allIds.push(id)
          const full = tokenizer.decode(new Int32Array(allIds))
          ui.appendToken(full.slice(prevText.length))
          prevText = full
          stats()
        })
      } else {
        // Fallback
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
