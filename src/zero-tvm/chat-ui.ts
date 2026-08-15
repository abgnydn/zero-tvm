/**
 * CHAT-UI — the conversational surface shared by chat.ts (the local engine
 * page) and share.ts (the remote guest): message construction, streaming
 * Markdown rendering, the copy/regenerate/stats action row, the cut-short
 * banner, and scroll management. Styles live in public/chat-ui.css — the two
 * files moved out of chat.ts / zero-tvm.html together, verbatim, so the local
 * and remote conversations cannot drift apart visually.
 *
 * What is deliberately NOT here: generation orchestration (prefix reuse,
 * resume, regenerate policy — chat.ts's business) and the transport
 * (DataChannel — share.ts's business). This module renders whatever its
 * caller streams into it, from any source.
 *
 * Expected markup (ids, both pages): #chat-main > #messages, optional
 * #scroll-fab and #welcome.
 */

import { renderMarkdown, wireCopyButton, ICON_REFRESH } from './markdown.js'
import type { ModelSpec } from '../compiler/model-spec.js'

const $ = (id: string) => document.getElementById(id)

/** Message-header identity (role name + quant tag). Set once at boot by the
 *  chat page, or when the `info` frame arrives on the guest. */
let IDENT = { name: '…', tag: '' }
export function setChatIdentity(name: string, tag: string): void {
  IDENT = { name, tag }
}

/** Quant label for the message header — from the spec, not a literal: MLX
 *  checkpoints are affine group-64, not MLC's q4f16_1. The host sends this
 *  over the wire so the GUEST shows the truth about a model it never loads. */
export function quantTagFor(spec: ModelSpec): string {
  return spec.weightFormat === 'mlx-safetensors'
    ? (spec.moe?.bits === 3 ? 'MLX 3-bit experts' : 'MLX 4-bit')
    : 'q4f16_1'
}

export function hideWelcome(): void {
  const w = $('welcome')
  if (w && !w.classList.contains('hidden')) w.classList.add('hidden')
}

export function autoGrow(ta: HTMLTextAreaElement): void {
  ta.style.height = 'auto'
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px'
}

export function scrollToBottom(force = false): void {
  const main = $('chat-main')
  if (!main) return
  const distance = main.scrollHeight - main.scrollTop - main.clientHeight
  if (force || distance < 120) main.scrollTop = main.scrollHeight
}

let scrollFabWired = false
export function wireScrollFab(): void {
  if (scrollFabWired) return
  const main = $('chat-main'); const fab = $('scroll-fab')
  if (!main || !fab) return
  scrollFabWired = true
  const update = () => {
    const distance = main.scrollHeight - main.scrollTop - main.clientHeight
    fab.classList.toggle('visible', distance > 140)
  }
  main.addEventListener('scroll', update, { passive: true })
  fab.addEventListener('click', () => scrollToBottom(true))
  update()
}

// ---------- Message construction ----------

export function addUserMsg(text: string): void {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg user'
  const bubble = document.createElement('div')
  bubble.className = 'bubble'
  bubble.textContent = text
  wrap.appendChild(bubble)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)
}

export interface AiMsgHandle {
  wrap: HTMLElement
  body: HTMLElement
  cursor: HTMLElement
  /** Replace the role tag with a thinking indicator (before first token). */
  showThinking(): void
  /** Re-render the full text (Markdown) into body on next animation frame. */
  render(text: string): void
  /** Remove cursor, add copy + stats + regenerate actions, stop streaming.
   *  `notice` renders a persistent banner under the reply — used when the
   *  reply was cut short, so it never just stops mid-word in silence. */
  finish(opts: {
    fullText: string
    tokens: number
    tokPerS: number
    onRegenerate?: () => void
    notice?: { text: string; onContinue?: () => void }
  }): void
}

export function addAiMsg(): AiMsgHandle {
  hideWelcome()
  const wrap = document.createElement('div')
  wrap.className = 'msg ai'
  wrap.innerHTML = `
    <div class="role">
      <span class="role-dot">Z</span>
      <span class="role-name"></span>
      <span class="model-tag">${IDENT.tag}</span>
    </div>
    <div class="body"></div>
    <div class="actions"></div>
  `
  ;(wrap.querySelector('.role-name') as HTMLElement).textContent = IDENT.name
  const body = wrap.querySelector('.body') as HTMLElement
  const cursor = document.createElement('span')
  cursor.className = 'cursor'
  body.appendChild(cursor)
  $('messages')?.appendChild(wrap)
  scrollToBottom(true)

  let rafPending = false
  let latestText = ''
  const doRender = () => {
    rafPending = false
    const frag = renderMarkdown(latestText)
    body.replaceChildren(frag, cursor)
    scrollToBottom(false)
  }

  return {
    wrap, body, cursor,
    showThinking() {
      // Replace the empty body (just cursor) with a thinking indicator.
      body.replaceChildren()
      const t = document.createElement('span')
      t.className = 'thinking'
      t.innerHTML = '<span></span><span></span><span></span>'
      body.appendChild(t)
    },
    render(text: string) {
      latestText = text
      if (!rafPending) {
        rafPending = true
        requestAnimationFrame(doRender)
      }
    },
    finish({ fullText, tokens, tokPerS, onRegenerate, notice }) {
      // Ensure final markdown render without cursor.
      const frag = renderMarkdown(fullText)
      body.replaceChildren(frag)
      // A resumed reply re-runs finish() on the same message — drop the
      // banner that made it resumable before rebuilding.
      wrap.querySelector('.truncation')?.remove()
      const actions = wrap.querySelector('.actions') as HTMLElement | null
      if (!actions) return
      actions.replaceChildren()

      const copy = document.createElement('button')
      copy.type = 'button'
      copy.className = 'action-btn'
      wireCopyButton(copy, () => fullText)
      actions.appendChild(copy)

      if (onRegenerate) {
        const regen = document.createElement('button')
        regen.type = 'button'
        regen.className = 'action-btn'
        regen.innerHTML = `${ICON_REFRESH}<span>Regenerate</span>`
        regen.addEventListener('click', onRegenerate)
        actions.appendChild(regen)
      }

      // A restored (or refused) reply has no rate to claim — show only what
      // is true: the count when there is one, the rate when it was measured.
      if (tokens > 0) {
        const stats = document.createElement('span')
        stats.className = 'msg-stats'
        stats.textContent = tokPerS > 0
          ? `${tokens} tok · ${tokPerS.toFixed(1)} tok/s`
          : `${tokens} tok`
        actions.appendChild(stats)
      }

      // .actions is hover-revealed; the cut-short banner must not be, so it
      // gets its own always-visible row between the body and the actions.
      if (notice) {
        const el = document.createElement('div')
        el.className = 'truncation'
        const label = document.createElement('span')
        label.textContent = notice.text
        el.appendChild(label)
        if (notice.onContinue) {
          const btn = document.createElement('button')
          btn.type = 'button'
          btn.className = 'truncation-btn'
          btn.textContent = 'Continue'
          btn.addEventListener('click', notice.onContinue)
          el.appendChild(btn)
        }
        wrap.insertBefore(el, actions)
      }
    },
  }
}
