/**
 * FAST CHAT — Phi3Engine powered chat.
 *
 * Message 1: TVM generates (captures shaders + pipelines)
 * Message 2+: Our Phi3Engine generates everything. No TVM.
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import { patchForCapture, getCaptureResult, resetCapture } from './capture.js'
import { buildPhi3Engine } from './phi3.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const llm = new LLMEngine()

  ui.appendLog('Loading model...')

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
  if (tokenizer) ui.appendLog('Tokenizer grabbed (encode + decode)')

  ui.setBadge('Ready')
  ui.setEnabled(true)
  ui.setInitialMessage('Phi-3 Engine\nMessage 1: TVM (captures shaders)\nMessage 2+: Our engine, no TVM')

  let phi3: ReturnType<typeof buildPhi3Engine> | null = null
  let generating = false
  let messageCount = 0

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
    const updateStats = () => {
      const elapsed = (performance.now() - t0) / 1000
      ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
    }

    try {
      messageCount++

      if (!phi3) {
        // No engine yet — TVM generates + we capture
        ui.appendLog('TVM generating (capturing)...')
        await llm.chat(text, (tok) => {
          count++
          ui.appendToken(tok)
          updateStats()
        })

        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0 && cap.writes.length > 0 && cap.copy) {
          phi3 = buildPhi3Engine(cap)
          ui.appendLog(`Phi3Engine ready — ${cap.dispatches.length} dispatches captured`)
        }
      } else {
        // Re-capture fresh dispatch tape for this message
        resetCapture()

        // TVM prefills + 3 warmup tokens, fresh capture builds
        let tvmCount = 0
        try {
          await llm.chat(text, (tok) => {
            tvmCount++
            count++
            ui.appendToken(tok)
            updateStats()
            if (tvmCount >= 3) throw new Error('switch')
          })
        } catch {
          // Expected: switch
        }

        // Rebuild engine with fresh capture
        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0 && cap.writes.length > 0 && cap.copy) {
          phi3 = buildPhi3Engine(cap)

          const allIds: number[] = []
          let prevText = ''
          await phi3.generate(500, (id) => {
            count++
            allIds.push(id)
            if (tokenizer) {
              const full = tokenizer.decode(new Int32Array(allIds))
              ui.appendToken(full.slice(prevText.length))
              prevText = full
            }
            updateStats()
          })
        } else {
          // Capture didn't complete — let TVM finish
          ui.appendLog('Re-capture incomplete, using TVM')
          await llm.chat(text, (tok) => { count++; ui.appendToken(tok); updateStats() })
        }
      }
    } catch (e) {
      ui.setError(e instanceof Error ? e.message : String(e))
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
