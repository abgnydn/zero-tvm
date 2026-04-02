/**
 * FAST CHAT — Phi3Engine powered chat.
 *
 * TVM loads model + handles prefill + first response (triggers capture).
 * Phi3Engine takes over for subsequent messages.
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import { patchForCapture, getCaptureResult } from './capture.js'
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

  const decode = llm.getDecoder()

  ui.setBadge('Ready')
  ui.setEnabled(true)
  ui.setInitialMessage('Phi-3 Engine — Own inference, no TVM runtime.\nFirst message uses TVM (captures shaders).\nAll subsequent messages use our engine.')

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

      if (messageCount === 1) {
        // First message: TVM generates (triggers capture)
        ui.appendLog('First message — TVM generating (capturing shaders)...')
        await llm.chat(text, (tok) => {
          count++
          ui.appendToken(tok)
          updateStats()
        })

        // Build Phi3Engine from captured data
        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0 && cap.writes.length > 0 && cap.copy) {
          phi3 = buildPhi3Engine(cap)
          ui.appendLog('Phi3Engine ready — next messages use our engine')
        } else {
          ui.appendLog('Capture incomplete — will keep using TVM')
        }
      } else if (phi3) {
        // Subsequent messages: Phi3Engine generates
        // TVM still does prefill (we need it for the prompt encoding)
        // But we intercept after prefill and use our engine for decode

        // For now: let TVM prefill + generate 1 token, then switch to our engine
        // The capture from first message has the dispatch tape
        // We re-use TVM for prefill each time (it handles tokenization + KV cache init)

        // Actually, simpler: just use our engine directly since TVM's chat
        // already populated the KV cache during the first message.
        // For a new message, we need TVM to prefill the new prompt.

        // Hybrid approach: TVM prefills, we decode
        let tvmTokenCount = 0
        try {
          await llm.chat(text, (tok) => {
            tvmTokenCount++
            count++
            ui.appendToken(tok)
            updateStats()
            // After 3 TVM tokens, abort and switch to our engine
            if (tvmTokenCount >= 3) throw new Error('switch')
          })
        } catch {
          // Expected: switch to our engine
        }

        // Our engine generates the rest
        const allIds: number[] = []
        let prevText = ''

        const tokens = await phi3.generate(500, (id) => {
          count++
          allIds.push(id)
          if (decode) {
            const full = decode(new Int32Array(allIds))
            ui.appendToken(full.slice(prevText.length))
            prevText = full
          }
          updateStats()
        })

        void tokens
      } else {
        // Fallback: TVM generates
        await llm.chat(text, (tok) => {
          count++
          ui.appendToken(tok)
          updateStats()
        })
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
