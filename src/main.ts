/**
 * Main App — Wires LLMEngine + ChatUI.
 * Clean separation: engine handles inference, UI handles DOM, main orchestrates.
 */

import { LLMEngine, MODELS } from './engine.js'
import { ChatUI } from './ui.js'
import { patchDevice, startProfile, stopProfile, summarizeProfile, getShaderSummary, captured } from './profiler.js'
import { dumpShader, dumpLayerShaders, dumpAllShaders } from './shader-dump.js'
import { setInterceptEnabled, patchDeviceIntercept, getInterceptStats } from './interceptor.js'
import { setRecorderEnabled, patchDeviceRecorder } from './recorder.js'
import { setBatcherEnabled, patchDeviceBatcher, getBatcherStats } from './batcher.js'
import { setReplayEnabled, patchDeviceReplay, patchStagingReadback, getReplayStats } from './replay.js'

async function main(): Promise<void> {
  const ui = new ChatUI()
  const engine = new LLMEngine()

  ui.appendLog('Starting...')

  const params = new URLSearchParams(window.location.search)

  // ?fuse=1 — shader interception
  const useFusion = params.has('fuse')
  if (useFusion) {
    setInterceptEnabled(true)
    ui.appendLog('FUSION MODE: shader interception enabled')
  }

  // ?record=1 — dispatch tape recording (auto-diffs two decode tokens)
  const useRecorder = params.has('record')
  if (useRecorder) {
    setRecorderEnabled(true)
    ui.appendLog('RECORDER MODE: capturing dispatch tape for diff analysis')
  }

  // ?batch=1 — submit batching (accumulate command buffers, submit once per token)
  const useBatcher = params.has('batch')
  if (useBatcher) {
    setBatcherEnabled(true)
    ui.appendLog('BATCH MODE: accumulating submits, flushing at mapAsync')
  }

  // ?replay=1 — dispatch tape replay (record first token, bypass TVM for rest)
  const useReplay = params.has('replay')
  if (useReplay) {
    setReplayEnabled(true)
    ui.appendLog('REPLAY MODE: will record first decode token, then bypass TVM')
  }

  // Monkey-patch GPU to intercept WebLLM's device
  if (navigator.gpu) {
    const origRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function(...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRequestAdapter(...args)
      if (!adapter) return adapter
      const origRequestDevice = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function(...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRequestDevice(...dArgs)
        // ?clean=1 — skip profiler for clean batcher measurement
        if (!params.has('clean')) {
          patchDevice(device)
          if (useFusion) patchDeviceIntercept(device)
          if (useRecorder) patchDeviceRecorder(device)
        }
        if (useBatcher) patchDeviceBatcher(device)
        if (useReplay) { patchDeviceReplay(device); patchStagingReadback(device) }
        ui.appendLog(params.has('clean') ? 'Clean mode (no profiler)' : 'GPU profiler attached')
        return device
      }
      return adapter
    }
  }

  // Load model
  try {
    await engine.load(MODELS.PHI3_MINI_Q4, (msg, _pct) => {
      ui.setBadge(msg, true)
      ui.appendLog(msg)
    })

    ui.setBadge('Ready')
    ui.setEnabled(true)
    ui.appendLog(getShaderSummary())
    ui.appendLog(`Total WGSL lines: ${captured.shaders.reduce((s, sh) => s + sh.lineCount, 0)}`)
    // Expose shaders on window for inspection in devtools
    ;(window as unknown as Record<string, unknown>).__capturedShaders = captured
    ;(window as unknown as Record<string, unknown>).__dumpShader = dumpShader
    ;(window as unknown as Record<string, unknown>).__dumpLayerShaders = dumpLayerShaders
    ;(window as unknown as Record<string, unknown>).__dumpAllShaders = dumpAllShaders
    if (useFusion) ui.appendLog(getInterceptStats())
    ui.appendLog('Devtools: __dumpShader(idx), __dumpLayerShaders(), __dumpAllShaders()')
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
      startProfile()
      const stats = await engine.chat(text, (token) => {
        ui.appendToken(token)
        count++
        const elapsed = (performance.now() - t0) / 1000
        ui.setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s | ${elapsed.toFixed(1)}s`)
      })

      // Profile report
      const profile = stopProfile()
      ui.appendLog(summarizeProfile(profile))

      const runtimeStats = await engine.getStats()
      ui.appendLog(runtimeStats)
      ui.appendLog(`Generation: ${stats.decodeTokens} tokens, ${stats.tokPerSec.toFixed(1)} tok/s`)
      ui.appendLog(`Dispatches per token: ${(profile.totalDispatches / stats.decodeTokens).toFixed(0)}`)
      if (useBatcher) ui.appendLog(getBatcherStats())
      if (useReplay) ui.appendLog(getReplayStats())

    } catch (e) {
      ui.setError(e instanceof Error ? e.message : String(e))
    }

    generating = false
    ui.setEnabled(true)
  }

  ui.onSend(send)
}

main().catch(console.error)
