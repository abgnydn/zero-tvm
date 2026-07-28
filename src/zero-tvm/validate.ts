/**
 * ZERO-TVM VALIDATION HARNESS
 *
 * Why this file exists:
 *   The per-shader test-harness (src/compiler/test-harness.ts) compares each
 *   of our 10 kernels (27 WGSL kernels) against TVM's ground truth — but only on a single
 *   hard-coded prompt ("What is the capital of France?"). That tests shader
 *   correctness in isolation; it does not test the live chat path on diverse
 *   inputs.
 *
 *   This page exercises the full Zero-TVM forward pass on a battery of
 *   diverse prompts and surfaces what the model actually predicts. For each
 *   prompt it shows:
 *     - The top-10 next-token candidates (token text + probability)
 *     - A short greedy continuation (≤24 tokens, stops on <|end|>)
 *
 *   A reader can scroll through the results and verify the outputs make sense
 *   for each prompt. Compared with WebLLM running the same prompts, the
 *   output shapes (top-1 token, probability mass concentration) should match
 *   modulo fp16 noise.
 */

import { buildChatPrompt, Tokenizer } from './tokenizer.js'
import { type DecodeEngine } from './engine-core.js'
import {
  bootEngine,
  hideProgress,
  log as bootLog,
  setBadge,
} from './loading-ui.js'

// ============================================================
// Test prompts — chosen to exercise different LM behaviors
// ============================================================

interface TestPrompt {
  label: string
  system: string
  user: string
  /** Optional sanity hint — string the top-1 continuation should plausibly contain.
   * This is informational only; the test doesn't fail if it's missing, since
   * a correct greedy continuation can legitimately phrase things differently. */
  hint?: string
}

const PROMPTS: TestPrompt[] = [
  {
    label: 'factual-recall',
    system: 'You are a helpful assistant.',
    user: 'What is the capital of France?',
    hint: 'Paris',
  },
  {
    label: 'arithmetic',
    system: 'You are a helpful assistant. Answer briefly.',
    user: 'What is 17 plus 25?',
    hint: '42',
  },
  {
    label: 'code-completion',
    system: 'You are a coding assistant.',
    user: 'Write a one-line Python expression that returns the length of a list called xs.',
    hint: 'len',
  },
  {
    label: 'instruction-follow',
    system: 'You are a helpful assistant.',
    user: 'Reply with exactly the word "yes" and nothing else.',
    hint: 'yes',
  },
  {
    label: 'open-ended',
    system: 'You are a helpful assistant.',
    user: 'Name three colors.',
  },
]

// ============================================================
// UI helpers
// ============================================================

function log(msg: string) {
  // Forward to the shared boot log so messages show up in #progress-log.
  bootLog(msg)
}

function setStatus(text: string) {
  const el = document.getElementById('status')
  if (el) el.textContent = text
}

function appendResult(html: string) {
  const wrap = document.getElementById('results-wrap')!
  const div = document.createElement('div')
  div.className = 'prompt-result'
  div.innerHTML = html
  wrap.appendChild(div)
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

// ============================================================
// Logit analysis
// ============================================================

interface TopK {
  id: number
  logit: number
  prob: number
  text: string
}

function softmaxTopK(logits: Float32Array, k: number, tokenizer: Tokenizer): TopK[] {
  // Numerical-stable softmax with subtraction of max.
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  let sum = 0
  const exps = new Float32Array(logits.length)
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max)
    sum += exps[i]
  }
  // Heap-free top-K via partial sort. K is small (10) so O(N*K) is fine.
  const taken = new Set<number>()
  const out: TopK[] = []
  for (let r = 0; r < k; r++) {
    let bestI = -1
    let bestV = -Infinity
    for (let i = 0; i < logits.length; i++) {
      if (taken.has(i)) continue
      if (logits[i] > bestV) {
        bestV = logits[i]
        bestI = i
      }
    }
    if (bestI < 0) break
    taken.add(bestI)
    out.push({
      id: bestI,
      logit: bestV,
      prob: exps[bestI] / sum,
      text: tokenizer.decode([bestI]),
    })
  }
  return out
}

function entropy(logits: Float32Array): number {
  let max = -Infinity
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i]
  let sum = 0
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max)
  const logZ = Math.log(sum) + max
  let h = 0
  for (let i = 0; i < logits.length; i++) {
    const p = Math.exp(logits[i] - logZ)
    if (p > 1e-12) h -= p * Math.log(p)
  }
  return h
}

// ============================================================
// Run a single prompt through Zero-TVM
// ============================================================

interface PromptResult {
  label: string
  prompt: string
  topK: TopK[]
  entropyNats: number
  continuation: string
  forwardMs: number
  decodeMs: number
  decodeTokens: number
}

async function runPrompt(
  engine: DecodeEngine,
  tokenizer: Tokenizer,
  p: TestPrompt
): Promise<PromptResult> {
  const promptIds = buildChatPrompt(
    [
      { role: 'system', content: p.system },
      { role: 'user', content: p.user },
    ],
    tokenizer
  )

  // (1) Forward pass — read logits at the final prompt position.
  // Always starts from position 0 — no KV reuse, fresh state per prompt.
  engine.resetKVTracking()
  const t0 = performance.now()
  const logits = await engine.forwardLogits(promptIds)
  const forwardMs = performance.now() - t0

  const topK = softmaxTopK(logits, 10, tokenizer)
  const ent = entropy(logits)

  // (2) Greedy decode — short continuation so the page stays snappy.
  // This shares the KV cache populated by forwardLogits above; the engine
  // only re-runs the final prompt token (which forwardLogits already did)
  // before entering the decode loop.
  const decoded: number[] = []
  const t1 = performance.now()
  await engine.generate(promptIds, promptIds.length, 24, (id) => {
    decoded.push(id)
  })
  const decodeMs = performance.now() - t1

  return {
    label: p.label,
    prompt: p.user,
    topK,
    entropyNats: ent,
    continuation: tokenizer.decode(decoded),
    forwardMs,
    decodeMs,
    decodeTokens: decoded.length,
  }
}

function renderResult(r: PromptResult): string {
  const topRows = r.topK
    .map((t, i) => {
      const pct = (t.prob * 100).toFixed(2)
      const bar = '█'.repeat(Math.max(1, Math.round(t.prob * 30)))
      const display = JSON.stringify(t.text).slice(1, -1) // shows escapes for \n etc
      return `  ${String(i + 1).padStart(2)}. ${display.padEnd(20)} ${pct.padStart(6)}%  ${bar}`
    })
    .join('\n')

  return `
    <div class="prompt-label">${escapeHtml(r.label)}</div>
    <div class="prompt-text">${escapeHtml(r.prompt)}</div>
    <div class="prompt-meta">
      forward: ${r.forwardMs.toFixed(0)} ms ·
      decode: ${r.decodeMs.toFixed(0)} ms (${r.decodeTokens} tok, ${(r.decodeTokens / (r.decodeMs / 1000)).toFixed(1)} tok/s) ·
      entropy: ${r.entropyNats.toFixed(2)} nats
    </div>
    <div class="prompt-section">Top-10 next tokens</div>
    <pre class="topk">${escapeHtml(topRows)}</pre>
    <div class="prompt-section">Greedy continuation (≤24 tokens)</div>
    <pre class="continuation">${escapeHtml(r.continuation)}</pre>
  `
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  if (!navigator.gpu) {
    setStatus('No WebGPU available')
    return
  }

  const startBtn = document.getElementById('start-btn') as HTMLButtonElement
  await new Promise<void>((resolve) => {
    startBtn.addEventListener('click', () => {
      startBtn.disabled = true
      startBtn.textContent = 'Loading...'
      resolve()
    })
  })

  document.getElementById('start-screen')?.remove()
  document.getElementById('results-wrap')?.classList.add('active')

  // Boot the engine through the shared loading flow — same progress bar +
  // byte counter the chat page uses, so the user can see how much download
  // is left on first run.
  setStatus('Booting engine...')
  const boot = await bootEngine()
  if (!boot.ok) {
    setStatus(boot.reason)
    return
  }
  const { tokenizer, engine } = boot
  hideProgress()

  // bootEngine already paid the GPU pipeline warmup, so the very first prompt
  // below sees steady-state numbers — same as every later one.
  setBadge('Running', 'loading')
  setStatus(`Running ${PROMPTS.length} test prompts...`)

  for (let i = 0; i < PROMPTS.length; i++) {
    const p = PROMPTS[i]
    setStatus(`[${i + 1}/${PROMPTS.length}] ${p.label}`)
    log(`\n--- ${p.label} ---`)
    log(`prompt: ${p.user}`)
    try {
      const r = await runPrompt(engine, tokenizer, p)
      log(`top-1: ${JSON.stringify(r.topK[0].text)} (${(r.topK[0].prob * 100).toFixed(1)}%)`)
      log(`entropy: ${r.entropyNats.toFixed(2)} nats`)
      log(`continuation: ${JSON.stringify(r.continuation)}`)
      appendResult(renderResult(r))
    } catch (e) {
      log(`ERROR: ${e}`)
      appendResult(`<div class="prompt-error">${escapeHtml(p.label)}: ${escapeHtml(String(e))}</div>`)
    }
  }

  setStatus('Done')
  setBadge('Ready', 'ready')
  log('\n=== validation complete ===')
}

main().catch((e) => {
  console.error(e)
  log(`FATAL: ${e}`)
  setStatus(`FATAL: ${e}`)
})
