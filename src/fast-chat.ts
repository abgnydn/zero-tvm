/**
 * FAST CHAT — Chat UI powered by own decode loop.
 *
 * Flow:
 * 1. Load model via WebLLM (gets TVM to do prefill + compile shaders)
 * 2. Grab tokenizer from WebLLM before corruption
 * 3. For each message: let TVM prefill, capture first decode, abort, run own loop
 * 4. Decode token IDs to text via grabbed tokenizer
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import { setOwnLoopEnabled, patchDeviceOwnLoop, generate, onReady, onToken, getOwnLoopStats } from './own-loop.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const engine = new LLMEngine()

  ui.appendLog('Loading model...')

  // Patch device for own-loop capture
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
        return device
      }
      return adapter
    }
  }

  // Load model
  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => {
    ui.setBadge(msg, true)
    ui.appendLog(msg)
  })

  // Grab tokenizer BEFORE any chat (TVM is clean at this point)
  const decode = engine.getDecoder()
  if (!decode) {
    ui.appendLog('WARNING: Could not grab tokenizer. Will show token IDs only.')
  } else {
    ui.appendLog('Tokenizer grabbed successfully')
  }

  ui.setBadge('Ready')
  ui.setEnabled(true)
  ui.setInitialMessage('Fast Chat — Own Decode Loop\n52 tok/s, no TVM in hot path.\nEverything runs locally on your GPU.')

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
      // Reset own-loop state for fresh capture
      setOwnLoopEnabled(true)

      // Let TVM do prefill + 3 warmup tokens, capture, then own loop
      const readyPromise = new Promise<void>(resolve => {
        onReady(() => resolve())
      })

      // TVM generates — we capture in background, abort when ready
      try {
        await engine.chat(text, (tok) => {
          count++
          ui.appendToken(tok)
          const elapsed = (performance.now() - t0) / 1000
          ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
          if (getOwnLoopStats().includes('ready')) throw new Error('abort')
        })
      } catch {
        // Expected abort after capture
      }

      await readyPromise

      // Stream token callback — decode cumulatively for correct spacing
      const allTokenIds: number[] = []
      let prevText = ''
      onToken((tokenId) => {
        count++
        allTokenIds.push(tokenId)
        if (decode) {
          const fullText = decode(new Int32Array(allTokenIds))
          const newText = fullText.slice(prevText.length)
          prevText = fullText
          ui.appendToken(newText)
        }
        const elapsed = (performance.now() - t0) / 1000
        ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
      })

      // Run own decode loop
      await generate(500)

    } catch (e) {
      ui.setError(e instanceof Error ? e.message : String(e))
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
