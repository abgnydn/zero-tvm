/**
 * FAST CHAT — TVM prefill + own decode loop.
 *
 * TVM handles first few tokens (correct). Our loop handles the rest (fast).
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import * as loop from './own-loop.js'
import { patchForCapture, getCaptureResult } from './capture.js'
import { buildStandaloneEngine } from './standalone.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const engine = new LLMEngine()

  ui.appendLog('Loading model...')

  loop.enable()
  if (navigator.gpu) {
    const origRA = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function(...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRA(...args)
      if (!adapter) return adapter
      const origRD = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function(...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRD(...dArgs)
        patchForCapture(device)  // capture shaders + pipelines + weights
        loop.patch(device)       // capture dispatch tape (still used for now)
        return device
      }
      return adapter
    }
  }

  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => {
    ui.setBadge(msg, true)
    ui.appendLog(msg)
  })

  const decode = engine.getDecoder()

  // Log capture results (shaders + pipelines captured at load time)
  const capture = getCaptureResult()
  if (capture) {
    ui.appendLog(`Captured: ${capture.shaders.length} shaders, ${capture.pipelines.length} pipelines, ${capture.weights.length} weight writes`)
  }

  ui.setBadge('Ready')
  ui.setEnabled(true)
  ui.setInitialMessage('Fast Chat — 50 tok/s decode\nTVM prefill → own decode loop.\nEverything local on your GPU.')

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

    try {
      loop.enable()  // fresh capture each message

      // TVM generates — we capture, abort when ready
      const ready = new Promise<void>(r => loop.onReady(r))
      try {
        await engine.chat(text, (tok) => {
          count++
          ui.appendToken(tok)
          const elapsed = (performance.now() - t0) / 1000
          ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
          if (loop.status() === 'ready') throw new Error('switch')
        })
      } catch {
        // Expected: switch to own loop
      }
      await ready

      // Own loop — cumulative decode for correct spacing
      const allIds: number[] = []
      let prevText = ''
      loop.onToken((id) => {
        count++
        allIds.push(id)
        if (decode) {
          const full = decode(new Int32Array(allIds))
          ui.appendToken(full.slice(prevText.length))
          prevText = full
        }
        const elapsed = (performance.now() - t0) / 1000
        ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
      })

      await loop.generate(500)

      // After first chat: build standalone engine from captured dispatches
      const cap = getCaptureResult()
      if (cap && cap.dispatches.length > 0) {
        try {
          buildStandaloneEngine(cap)
        } catch (e) {
          ui.appendLog(`Standalone: ${e instanceof Error ? e.message : String(e)}`)
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
