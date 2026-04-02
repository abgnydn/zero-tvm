/**
 * Correctness Test Harness for Dispatch Replay
 *
 * Runs WebLLM baseline vs replay, captures token IDs from both,
 * and finds the first divergence point.
 */

import { LLMEngine, MODELS } from './engine.js'
import { setReplayEnabled, patchDeviceReplay, patchStagingReadback, getReplayStats } from './replay.js'

// ============================================================
// Types
// ============================================================

interface PatchSnapshot {
  tokenId: number | null
  counters: number[]
}

interface ReplayResult {
  tokenIds: number[]
  patchValues: PatchSnapshot[]
}

interface Divergence {
  index: number
  baselineToken: number
  replayToken: number
}

// ============================================================
// DOM Helpers
// ============================================================

const $ = (sel: string): HTMLElement => document.querySelector(sel)!

function log(target: string, msg: string): void {
  const el = $(target)
  el.textContent += msg + '\n'
  el.scrollTop = el.scrollHeight
}

function clearLog(target: string): void {
  $(target).textContent = ''
}

// ============================================================
// Token ID Capture via getMappedRange Interception
// ============================================================

let capturedTokenIds: number[] = []
let isCapturing = false

/**
 * Patches the device so every MAP_READ buffer's getMappedRange captures
 * the first int32 LE value as a token ID. This mirrors what replay.ts does
 * in patchStagingReadback.
 */
let tokenCaptureInstalled = false
function patchDeviceTokenCapture(_device: GPUDevice): void {
  if (tokenCaptureInstalled) return
  tokenCaptureInstalled = true
  // Chain on prototype — works with replay's prototype patch
  const prev = GPUBuffer.prototype.getMappedRange
  GPUBuffer.prototype.getMappedRange = function(offset?: number, size?: number): ArrayBuffer {
    const range = prev.call(this, offset, size)
    if (isCapturing && range.byteLength >= 4) {
      const val = new DataView(range).getInt32(0, true)
      if (val >= 0 && val < 100000) {
        capturedTokenIds.push(val)
      }
    }
    return range
  }
}

// ============================================================
// Replay Patch Value Capture
//
// We intercept writeBuffer during replay to observe what values
// get patched into stable buffers each token.
// ============================================================

let replayPatchValues: PatchSnapshot[] = []
let currentTokenPatches: { role: string; value: number }[] = []

function patchDeviceReplayCapture(device: GPUDevice): void {
  const origWriteBuffer = device.queue.writeBuffer.bind(device.queue)
  // Track mapAsync calls to detect token boundaries during replay
  const origCreateBuffer = device.createBuffer
  device.createBuffer = function (desc: GPUBufferDescriptor): GPUBuffer {
    const buf = origCreateBuffer.call(device, desc)
    if (desc.usage & GPUBufferUsage.MAP_READ) {
      const origMapAsync = buf.mapAsync.bind(buf)
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      ;(buf as any).mapAsync = function (...args: Parameters<GPUBuffer['mapAsync']>): Promise<undefined> {
        // Each mapAsync on a MAP_READ buffer = one token boundary
        if (isCapturing && currentTokenPatches.length > 0) {
          const counters = currentTokenPatches
            .filter(p => p.role === 'counter')
            .map(p => p.value)
          const tidPatch = currentTokenPatches.find(p => p.role === 'token_id')
          replayPatchValues.push({
            tokenId: tidPatch ? tidPatch.value : null,
            counters,
          })
          currentTokenPatches = []
        }
        return origMapAsync(...args)
      }
    }
    return buf
  }

  // Watch writeBuffer calls to stable buffers during replay
  // We detect patches by size (4 bytes) and value range
  device.queue.writeBuffer = function (
    buf: GPUBuffer,
    off: number,
    data: BufferSource | SharedArrayBuffer,
    dOff?: number,
    sz?: number,
  ): undefined {
    if (isCapturing) {
      // Snapshot the data being written
      let bytes: Uint8Array
      if (data instanceof ArrayBuffer) {
        const o = dOff ?? 0
        const s = sz ?? (data.byteLength - o)
        bytes = new Uint8Array(data.slice(o, o + s))
      } else if (ArrayBuffer.isView(data)) {
        const o = dOff ?? 0
        const s = sz ?? (data.byteLength - o)
        bytes = new Uint8Array(data.buffer.slice(data.byteOffset + o, data.byteOffset + o + s))
      } else {
        bytes = new Uint8Array(0)
      }
      if (bytes.length === 4) {
        const val = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)
        // Heuristic: counters are large-ish position values, token_ids are < 100000
        // We can't perfectly distinguish here, so we just log all 4-byte writes
        // and label them by whether they look like a counter or token_id
        const label = buf.label?.startsWith('stable_') ? 'counter' : 'unknown'
        currentTokenPatches.push({ role: label, value: val })
      }
    }
    return origWriteBuffer(buf, off, data, dOff, sz)
  }
}

// ============================================================
// Baseline Run
// ============================================================

// runBaseline removed — replaced by runBaselineDeterminism

// ============================================================
// Replay Run
// ============================================================

async function runReplay(prompt: string): Promise<ReplayResult> {
  log('#replay-log', 'Initializing replay engine...')
  capturedTokenIds = []
  replayPatchValues = []
  currentTokenPatches = []
  isCapturing = false

  const engine = new LLMEngine()

  setReplayEnabled(true)

  // Patch device for replay + token capture + patch capture
  if (navigator.gpu) {
    const origRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function (...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRequestAdapter(...args)
      if (!adapter) return adapter
      const origRequestDevice = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function (...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRequestDevice(...dArgs)
        patchDeviceReplayCapture(device)
        patchDeviceReplay(device)
        patchStagingReadback(device)
        patchDeviceTokenCapture(device)
        log('#replay-log', 'Device patched for replay + capture')
        return device
      }
      return adapter
    }
  }

  log('#replay-log', 'Loading model...')
  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => {
    log('#replay-log', msg)
  })

  log('#replay-log', `Generating with prompt: "${prompt}"`)
  isCapturing = true
  const tokens: string[] = []
  await engine.chat(prompt, (tok) => {
    tokens.push(tok)
  })
  isCapturing = false

  // Flush last token's patches
  if (currentTokenPatches.length > 0) {
    const counters = currentTokenPatches
      .filter(p => p.role === 'counter')
      .map(p => p.value)
    const tidPatch = currentTokenPatches.find(p => p.role === 'token_id')
    replayPatchValues.push({
      tokenId: tidPatch ? tidPatch.value : null,
      counters,
    })
  }

  log('#replay-log', `Done. Got ${tokens.length} text chunks, ${capturedTokenIds.length} token IDs`)
  log('#replay-log', `Text: ${tokens.join('')}`)
  log('#replay-log', `Token IDs: [${capturedTokenIds.join(', ')}]`)
  log('#replay-log', getReplayStats())

  // Log patch values
  for (let i = 0; i < replayPatchValues.length; i++) {
    const p = replayPatchValues[i]
    log('#replay-log', `  patch[${i}]: token_id=${p.tokenId}, counters=[${p.counters.join(',')}]`)
  }

  const result: ReplayResult = {
    tokenIds: [...capturedTokenIds],
    patchValues: [...replayPatchValues],
  }
  await engine.destroy()
  setReplayEnabled(false)
  return result
}

// ============================================================
// Comparison
// ============================================================

function compare(baseline: number[], replay: number[]): Divergence | null {
  const len = Math.min(baseline.length, replay.length)
  for (let i = 0; i < len; i++) {
    if (baseline[i] !== replay[i]) {
      return { index: i, baselineToken: baseline[i], replayToken: replay[i] }
    }
  }
  if (baseline.length !== replay.length) {
    const idx = len
    return {
      index: idx,
      baselineToken: idx < baseline.length ? baseline[idx] : -1,
      replayToken: idx < replay.length ? replay[idx] : -1,
    }
  }
  return null
}

// ============================================================
// UI Wiring
// ============================================================

const TEST_PROMPT = 'What is the capital of France?'

let baselineResult: number[] | null = null
let replayResult: ReplayResult | null = null

function showComparison(): void {
  if (!baselineResult || !replayResult) return

  clearLog('#compare-log')
  log('#compare-log', `Baseline: ${baselineResult.length} tokens`)
  log('#compare-log', `Replay:   ${replayResult.tokenIds.length} tokens`)

  const div = compare(baselineResult, replayResult.tokenIds)
  if (!div) {
    log('#compare-log', '\nRESULT: PERFECT MATCH - all tokens identical!')
  } else {
    log('#compare-log', `\nRESULT: DIVERGENCE at token index ${div.index}`)
    log('#compare-log', `  Baseline produced: ${div.baselineToken}`)
    log('#compare-log', `  Replay produced:   ${div.replayToken}`)

    // Show context around divergence
    const start = Math.max(0, div.index - 3)
    const end = Math.min(Math.max(baselineResult.length, replayResult.tokenIds.length), div.index + 5)
    log('#compare-log', '\nToken-by-token around divergence:')
    log('#compare-log', 'idx  | baseline | replay   | match')
    log('#compare-log', '-----|----------|----------|------')
    for (let i = start; i < end; i++) {
      const b = i < baselineResult.length ? String(baselineResult[i]) : '-'
      const r = i < replayResult.tokenIds.length ? String(replayResult.tokenIds[i]) : '-'
      const match = b === r ? 'OK' : 'DIFF'
      const marker = i === div.index ? ' <-- FIRST' : ''
      log('#compare-log', `${String(i).padStart(4)} | ${b.padStart(8)} | ${r.padStart(8)} | ${match}${marker}`)
    }

    // Show patch values around divergence if available
    if (replayResult.patchValues.length > 0) {
      log('#compare-log', '\nPatch values around divergence:')
      const pStart = Math.max(0, div.index - 3)
      const pEnd = Math.min(replayResult.patchValues.length, div.index + 5)
      for (let i = pStart; i < pEnd; i++) {
        const p = replayResult.patchValues[i]
        log('#compare-log', `  [${i}] token_id=${p.tokenId}, counters=[${p.counters.join(',')}]`)
      }
    }
  }
}

// baseline2 result stored inside runBaselineDeterminism

async function runAll(): Promise<void> {
  log('#compare-log', 'Starting automated test...')

  // Run baseline — same model, two chats, to check GPU determinism
  log('#compare-log', '1/3 Running baseline (2 chats on same model)...')
  try {
    const results = await runBaselineDeterminism(TEST_PROMPT)
    baselineResult = results.run1
    const baselineDiff = compare(results.run1, results.run2)
    if (!baselineDiff) {
      log('#compare-log', `DETERMINISM: Both runs match perfectly (${results.run1.length} tokens) ✓`)
    } else {
      log('#compare-log', `DETERMINISM: DIVERGE at index ${baselineDiff.index} (${baselineDiff.baselineToken} vs ${baselineDiff.replayToken})`)
      log('#compare-log', `  Run 1: [${results.run1.slice(Math.max(0, baselineDiff.index - 2), baselineDiff.index + 3).join(', ')}]`)
      log('#compare-log', `  Run 2: [${results.run2.slice(Math.max(0, baselineDiff.index - 2), baselineDiff.index + 3).join(', ')}]`)
      log('#compare-log', 'GPU is non-deterministic — replay cannot match better than this')
    }
  } catch (e) {
    log('#compare-log', `BASELINE FAILED: ${e instanceof Error ? e.message : String(e)}`)
    return
  }

  await new Promise(r => setTimeout(r, 1000))

  // Run replay
  log('#compare-log', '2/3 Running replay...')
  try {
    replayResult = await runReplay(TEST_PROMPT)
    log('#compare-log', `Replay: ${replayResult.tokenIds.length} tokens`)
  } catch (e) {
    log('#compare-log', `REPLAY FAILED: ${e instanceof Error ? e.message : String(e)}`)
    return
  }

  // Compare baseline vs replay
  log('#compare-log', '3/3 Comparing baseline vs replay...')
  showComparison()
}

/** Run two chats on the SAME loaded model to test GPU determinism */
async function runBaselineDeterminism(prompt: string): Promise<{ run1: number[]; run2: number[] }> {
  const engine = new LLMEngine()

  if (navigator.gpu) {
    const origRequestAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu)
    navigator.gpu.requestAdapter = async function (...args: Parameters<GPU['requestAdapter']>) {
      const adapter = await origRequestAdapter(...args)
      if (!adapter) return adapter
      const origRequestDevice = adapter.requestDevice.bind(adapter)
      adapter.requestDevice = async function (...dArgs: Parameters<GPUAdapter['requestDevice']>) {
        const device = await origRequestDevice(...dArgs)
        patchDeviceTokenCapture(device)
        return device
      }
      return adapter
    }
  }

  log('#baseline-log', 'Loading model...')
  await engine.load(MODELS.PHI3_MINI_Q4, (msg) => log('#baseline-log', msg))

  // Run 1
  capturedTokenIds = []
  isCapturing = true
  log('#baseline-log', `Chat 1: "${prompt}"`)
  const tokens1: string[] = []
  await engine.chat(prompt, (tok) => tokens1.push(tok))
  isCapturing = false
  const run1 = [...capturedTokenIds]
  log('#baseline-log', `Chat 1: ${run1.length} tokens — ${tokens1.join('')}`)

  // Reset conversation for run 2
  await engine.reset()

  // Run 2
  capturedTokenIds = []
  isCapturing = true
  log('#baseline-log', `Chat 2: "${prompt}"`)
  const tokens2: string[] = []
  await engine.chat(prompt, (tok) => tokens2.push(tok))
  isCapturing = false
  const run2 = [...capturedTokenIds]
  log('#baseline-log', `Chat 2: ${run2.length} tokens — ${tokens2.join('')}`)

  await engine.destroy()
  return { run1, run2 }
}

// Expose results on window for Puppeteer
const w = window as unknown as Record<string, unknown>
w.__testStatus = 'running'
w.__testResult = null

runAll()
  .then(() => {
    const div = baselineResult && replayResult ? compare(baselineResult, replayResult.tokenIds) : null
    w.__testResult = {
      baselineTokens: baselineResult?.length ?? 0,
      replayTokens: replayResult?.tokenIds.length ?? 0,
      divergence: div,
      baselineIds: baselineResult ?? [],
      replayIds: replayResult?.tokenIds ?? [],
      patchValues: replayResult?.patchValues ?? [],
    }
    w.__testStatus = div ? 'fail' : 'pass'
  })
  .catch(e => {
    w.__testStatus = 'error'
    w.__testResult = { error: String(e) }
    log('#compare-log', `FATAL: ${e}`)
  })
