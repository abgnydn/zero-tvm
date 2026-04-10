/**
 * ZERO-TVM CHAT — Phi-3 inference. No WebLLM. No TVM.
 *
 * Loads weights from browser cache (cached by a prior WebLLM session)
 * or fetches from HuggingFace. Uses our 10 WGSL shaders only.
 *
 * Correct buffer layout:
 *   B.residual  — persistent running residual (updated in-place by add_norm)
 *   B.hidden1   — normed scratch (matmul input)
 *   B.hidden2   — matmul output scratch (OProj, FFN down)
 */

import { buildChatPrompt } from './tokenizer.js'
import { bootEngine, hideProgress, log, setBadge } from './loading-ui.js'

// Engine internals (buildDecodeEngine, allocKVPages) live in ./engine-core.ts
// so the validate page can drive the same forward pass without booting this UI.
// The boot pipeline (GPU + tokenizer + weights + KV + shaders) lives in
// ./loading-ui.ts so chat and validate share one progress flow.


// ============================================================
// Chat-only UI helpers
// ============================================================

const $ = (id: string) => document.getElementById(id)!

function setStats(text: string) {
  $('stats').textContent = text
}

function setEnabled(on: boolean) {
  const inp = $('inp') as HTMLInputElement
  const btn = $('btn') as HTMLButtonElement
  inp.disabled = !on
  btn.disabled = !on
  if (on) inp.focus()
}

function addMsg(role: 'user' | 'ai'): HTMLElement {
  const chatWrap = document.querySelector('.chat-wrap')!
  const chat = $('chat')
  const div = document.createElement('div')
  div.className = `msg ${role}`
  if (role === 'ai') div.classList.add('generating')
  chat.appendChild(div)
  chatWrap.scrollTop = chatWrap.scrollHeight
  return div
}

// ============================================================
// Main
// ============================================================

async function main(): Promise<void> {
  if (!navigator.gpu) {
    setBadge('No WebGPU', 'error')
    const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null
    if (startBtn) {
      startBtn.textContent = 'WebGPU not available'
      startBtn.disabled = true
    }
    return
  }

  // Wait for user to click "Download & Start"
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement | null
  if (startBtn) {
    await new Promise<void>(resolve => {
      startBtn.addEventListener('click', () => {
        startBtn.disabled = true
        startBtn.textContent = 'Starting...'
        resolve()
      })
    })
  }

  document.getElementById('start-screen')?.remove()

  const boot = await bootEngine()
  if (!boot.ok) {
    log(`FATAL: ${boot.reason}`)
    return
  }
  const { tokenizer, engine } = boot

  // Transition from progress to chat
  hideProgress()
  setEnabled(true)

  const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
    { role: 'system', content: 'You are a helpful assistant.' },
  ]
  let generating = false
  // Tracks the full token sequence (prompt + generated) that the KV cache
  // currently reflects. Used to skip re-prefilling the conversation prefix
  // on multi-turn sends — slots [0, lcp) are already valid in KV pages.
  let kvCacheTokens: number[] = []

  function longestCommonPrefix(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length)
    let i = 0
    while (i < n && a[i] === b[i]) i++
    return i
  }

  async function send(): Promise<void> {
    if (generating) return
    const inp = $('inp') as HTMLInputElement
    const text = inp.value.trim()
    if (!text) return
    inp.value = ''

    generating = true
    setEnabled(false)

    const userEl = addMsg('user')
    userEl.textContent = text
    const aiEl = addMsg('ai')

    history.push({ role: 'user', content: text })

    const promptIds = buildChatPrompt(history, tokenizer)
    // KV slots already correct for any matching prefix — start prefill from
    // the first divergent token. For the typical chat flow this skips ~all of
    // the prior conversation on each turn.
    const startPos = longestCommonPrefix(kvCacheTokens, promptIds)

    const chatWrap = document.querySelector('.chat-wrap')!
    const t0 = performance.now()
    let count = 0
    const allIds: number[] = []
    let prevText = ''

    setBadge('Generating...', 'loading')

    try {
      await engine.generate(promptIds, startPos, 500, (id) => {
        count++
        allIds.push(id)
        const full = tokenizer.decode(allIds)
        const delta = full.slice(prevText.length)
        if (delta) {
          aiEl.textContent += delta
          chatWrap.scrollTop = chatWrap.scrollHeight
        }
        prevText = full
        const elapsed = (performance.now() - t0) / 1000
        setStats(`${count} tok | ${(count / elapsed).toFixed(1)} tok/s`)
      })

      const fullResponse = tokenizer.decode(allIds)
      history.push({ role: 'assistant', content: fullResponse })

      // The KV cache now reflects [prompt + generated tokens]. Recording this
      // lets the next turn skip re-prefilling everything that hasn't changed.
      kvCacheTokens = promptIds.concat(allIds)
    } catch (e) {
      aiEl.textContent += `\n[Error: ${e}]`
      // Generation failed mid-stream — KV cache state is undefined. Reset
      // tracking so the next turn does a full prefill from scratch.
      kvCacheTokens = []
      engine.resetKVTracking()
    }

    aiEl.classList.remove('generating')
    setBadge('Ready', 'ready')
    generating = false
    setEnabled(true)
  }

  $('btn').addEventListener('click', send)
  ;($('inp') as HTMLInputElement).addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  })
}

main().catch((e) => { console.error(e); log(`FATAL: ${e}`) })
