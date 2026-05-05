/**
 * ZERO-TVM SLOTTED ENGINE — Multi-slot Phi-3 inference on WebGPU.
 *
 * Creates N independent DecodeEngine instances (from engine-core.ts), each with
 * its own KV cache. Slots share the same GPU device, weights, and compiled
 * pipelines. A mutex serializes GPU command submission since the per-engine
 * activation buffers would collide if two forward passes ran concurrently.
 *
 * Usage from the town sim:
 *   const engine = await initSlottedEngine({ maxSlots: 8 })
 *   const reply = await engine.complete(systemPrompt, userPrompt, 80)
 *
 * Each complete() call acquires a slot, runs prefill+decode, and releases.
 * If all slots are busy, callers queue and wait for one to free up.
 * Slots use reduced context (1K tokens) to fit more in GPU memory.
 */

import { loadWeights } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt, Tokenizer } from './tokenizer.js'
import { buildDecodeEngine, DecodeEngine } from './engine-core.js'
import { PHI3 } from '../compiler/compiler.js'

// ============================================================
// Reduced-context KV cache for agent slots
// ============================================================

const SLOT_MAX_PAGES = 128  // 2K context (128 pages × 16 tokens/page)

function allocSlotKV(device: GPUDevice): GPUBuffer[] {
  const bytesPerPage = 98304 * 2  // same layout as engine-core
  return Array.from({ length: PHI3.LAYERS }, (_, i) =>
    device.createBuffer({
      size: Math.max(SLOT_MAX_PAGES * bytesPerPage, 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: `slotKV_${i}`,
    })
  )
}

// ============================================================
// GPU mutex — serialize forward passes across slots
// ============================================================

function createMutex() {
  let tail: Promise<void> = Promise.resolve()
  return {
    async run<T>(fn: () => Promise<T>): Promise<T> {
      const prev = tail
      let unlock!: () => void
      tail = new Promise<void>(r => { unlock = r })
      await prev
      try { return await fn() }
      finally { unlock() }
    }
  }
}

// ============================================================
// Slot management
// ============================================================

interface Slot {
  id: number
  engine: DecodeEngine
  kvPages: GPUBuffer[]
  busy: boolean
}

// ============================================================
// Public API
// ============================================================

export interface SlottedEngine {
  /** Complete a prompt. Acquires a slot, generates, releases. Queues if full. */
  complete(systemPrompt: string, userPrompt: string, maxTokens: number): Promise<string>
  /** Number of slots allocated. */
  slotCount: number
  /** Number of slots currently idle. */
  idleSlots(): number
  /** The tokenizer instance. */
  tokenizer: Tokenizer
}

export interface SlottedEngineConfig {
  maxSlots?: number        // default 8
  onLog?: (msg: string) => void
}

export async function initSlottedEngine(config: SlottedEngineConfig = {}): Promise<SlottedEngine> {
  const { maxSlots = 8, onLog = () => {} } = config

  // --- GPU ---
  if (!navigator.gpu) throw new Error('WebGPU not supported')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('No GPU adapter')
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16' as GPUFeatureName],
  })
  onLog('GPU ready (shader-f16)')

  // --- Tokenizer ---
  onLog('Loading tokenizer...')
  const tokenizer = await loadTokenizer(onLog)

  // --- Weights (shared across all slots) ---
  onLog('Loading weights...')
  const weights = await loadWeights(device, onLog)

  // --- Allocate slots (each gets its own KV cache + DecodeEngine) ---
  const slots: Slot[] = []
  for (let i = 0; i < maxSlots; i++) {
    try {
      const kvPages = allocSlotKV(device)
      const engine = buildDecodeEngine(device, weights, kvPages)
      slots.push({ id: i, engine, kvPages, busy: false })
    } catch {
      onLog(`Allocated ${i}/${maxSlots} slots (GPU memory limit)`)
      break
    }
  }
  if (slots.length === 0) throw new Error('Could not allocate any inference slots')
  onLog(`${slots.length} inference slots ready (${SLOT_MAX_PAGES * PHI3.PAGE_SIZE} tok context each)`)

  // --- Warmup first slot's pipelines ---
  onLog('Warming up GPU pipelines...')
  const warmupIds = buildChatPrompt(
    [{ role: 'user', content: 'Hi.' }],
    tokenizer
  )
  await slots[0].engine.generate(warmupIds, 0, 1, () => {})
  onLog('Engine ready')

  // --- Mutex + wait queue ---
  const mutex = createMutex()
  const waiters: Array<(slot: Slot) => void> = []

  function acquireSlot(): Slot | null {
    for (const s of slots) {
      if (!s.busy) { s.busy = true; return s }
    }
    return null
  }

  function releaseSlot(slot: Slot) {
    slot.busy = false
    if (waiters.length > 0) {
      const next = waiters.shift()!
      slot.busy = true
      next(slot)
    }
  }

  async function waitForSlot(): Promise<Slot> {
    const s = acquireSlot()
    if (s) return s
    return new Promise<Slot>(resolve => { waiters.push(resolve) })
  }

  // --- complete() ---
  async function complete(systemPrompt: string, userPrompt: string, maxTokens: number): Promise<string> {
    const slot = await waitForSlot()
    try {
      const messages = [
        { role: 'system' as const, content: systemPrompt },
        { role: 'user' as const, content: userPrompt },
      ]
      const promptIds = buildChatPrompt(messages, tokenizer)

      // Truncate prompt if it exceeds slot context
      const maxPrompt = SLOT_MAX_PAGES * PHI3.PAGE_SIZE - maxTokens
      const ids = promptIds.length > maxPrompt
        ? promptIds.slice(promptIds.length - maxPrompt)
        : promptIds

      // Each agent call is independent — always prefill from 0
      slot.engine.resetKVTracking()

      const outputIds: number[] = []
      // Wrap generate in mutex so two slots don't run forward passes concurrently
      // (they share activation buffers inside buildDecodeEngine)
      await mutex.run(() =>
        slot.engine.generate(ids, 0, maxTokens, (id) => { outputIds.push(id) })
      )

      return tokenizer.decode(outputIds)
    } finally {
      releaseSlot(slot)
    }
  }

  return {
    complete,
    slotCount: slots.length,
    idleSlots: () => slots.filter(s => !s.busy).length,
    tokenizer,
  }
}
