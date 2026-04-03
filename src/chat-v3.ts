/**
 * CHAT v3 — Minimal chat UI powered by engine-v3.
 *
 * Message 1: TVM generates (we capture shaders + dispatches)
 * Message 2+: Our engine generates. Zero TVM.
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import { patchForCapture, getCaptureResult, getLastPosition } from './capture.js'
import { buildEngine } from './engine-v3.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const llm = new LLMEngine()

  ui.appendLog('Loading model...')

  // Intercept WebGPU device creation to patch for capture
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
  ui.setInitialMessage('Engine v3\nMsg 1: TVM (captures)\nMsg 2+: Our engine, zero TVM')

  let engine: ReturnType<typeof buildEngine> | null = null
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
      if (!engine) {
        // First message: TVM generates, we capture
        ui.appendLog('TVM generating (capturing)...')
        await llm.chat(text, (tok) => { count++; ui.appendToken(tok); stats() })

        const cap = getCaptureResult()
        if (cap && cap.dispatches.length > 0 && cap.writes.length > 0 && cap.copy) {
          engine = buildEngine(cap)
          ui.appendLog(`Engine v3 ready — ${cap.dispatches.length} decode, ${cap.prefillDispatches.length} prefill`)
        }
      } else if (tokenizer) {
        // TVM prefills (builds KV cache for new prompt), our engine decodes
        try {
          await llm.chat(text, () => { throw new Error('prefill done') })
        } catch { /* expected — we abort after TVM's prefill */ }

        // TVM set up the state. Read actual position from intercepted writes.
        const pos = getLastPosition()
        ui.appendLog(`TVM prefilled to pos=${pos}, our engine decoding...`)

        const allIds: number[] = []
        let prevText = ''
        await engine.continueAfterPrefill(pos, 500, (id) => {
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
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
