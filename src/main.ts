/**
 * Main App — Wires LLMEngine + ChatUI.
 * Clean separation: engine handles inference, UI handles DOM, main orchestrates.
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const engine = new LLMEngine()

  ui.appendLog('Starting...')

  // Load model
  try {
    await engine.load(MODELS.PHI3_MINI_Q4, (msg, _pct) => {
      ui.setBadge(msg, true)
      ui.appendLog(msg)
    })

    ui.setBadge('Ready')
    ui.setEnabled(true)
    ui.setInitialMessage('Phi-3-mini loaded. Ask me anything!\n\nEverything runs locally on your GPU. Zero server requests.')

  } catch (e) {
    ui.setBadge('Error')
    ui.appendLog(`Load error: ${e instanceof Error ? e.message : String(e)}`)
    return
  }

  // Handle chat
  let generating = false

  async function send(): Promise<void> {
    if (generating || !engine.ready) return
    const text = ui.consumeInput()
    if (!text) return

    generating = true
    ui.setEnabled(false)
    ui.addUserMessage(text)
    ui.startAiMessage()

    const t0 = performance.now()
    let count = 0

    try {
      const stats = await engine.chat(text, (token) => {
        ui.appendToken(token)
        count++
        const elapsed = (performance.now() - t0) / 1000
        ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
      })

      // Log detailed stats
      const runtimeStats = await engine.getStats()
      ui.appendLog(runtimeStats)
      ui.appendLog(`Generation: ${stats.decodeTokens} tokens, ${stats.tokPerSec.toFixed(1)} tok/s`)

    } catch (e) {
      ui.setError(e instanceof Error ? e.message : String(e))
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
