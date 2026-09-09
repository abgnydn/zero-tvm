/**
 * Chat UI — Typed DOM manipulation.
 * No raw getElementById calls scattered around.
 */

export class ChatUI {
  private badge: HTMLElement
  private stats: HTMLElement
  private log: HTMLElement
  private chat: HTMLElement
  private input: HTMLInputElement
  private btn: HTMLButtonElement
  private currentAiMsg: HTMLElement | null = null

  constructor() {
    this.badge = this.el('badge')
    this.stats = this.el('stats')
    this.log = this.el('log')
    this.chat = this.el('chat')
    this.input = this.el('inp') as HTMLInputElement
    this.btn = this.el('btn') as HTMLButtonElement
  }

  private el(id: string): HTMLElement {
    const e = document.getElementById(id)
    if (!e) throw new Error(`Element #${id} not found`)
    return e
  }

  /** Log to the debug panel */
  appendLog(msg: string): void {
    this.log.textContent += msg + '\n'
    this.log.scrollTop = this.log.scrollHeight
  }

  /** Update badge text and style */
  setBadge(text: string, loading = false): void {
    this.badge.textContent = text
    this.badge.className = loading ? 'badge loading' : 'badge'
  }

  /** Update stats display */
  setStats(text: string): void {
    this.stats.textContent = text
  }

  /** Set chat enabled/disabled */
  setEnabled(enabled: boolean): void {
    this.input.disabled = !enabled
    this.btn.disabled = !enabled
    if (enabled) this.input.focus()
  }

  /** Get and clear input text */
  consumeInput(): string {
    const text = this.input.value.trim()
    this.input.value = ''
    return text
  }

  /** Clear chat and show initial message */
  setInitialMessage(msg: string): void {
    this.chat.innerHTML = ''
    this.addAiMessage(msg)
  }

  /** Add a user message bubble */
  addUserMessage(text: string): void {
    const div = document.createElement('div')
    div.className = 'msg user'
    div.textContent = text
    this.chat.appendChild(div)
    this.chat.scrollTop = this.chat.scrollHeight
  }

  /** Start a new AI message (returns nothing — use appendToken) */
  startAiMessage(): void {
    const div = document.createElement('div')
    div.className = 'msg ai'
    this.chat.appendChild(div)
    this.currentAiMsg = div
    this.chat.scrollTop = this.chat.scrollHeight
  }

  /** Append token text to current AI message */
  appendToken(text: string): void {
    if (this.currentAiMsg) {
      this.currentAiMsg.textContent += text
      this.chat.scrollTop = this.chat.scrollHeight
    }
  }

  /** Set AI message to error */
  setError(msg: string): void {
    if (this.currentAiMsg) {
      this.currentAiMsg.textContent += ` [${msg}]`
    }
  }

  /** Register send handler */
  onSend(handler: () => void): void {
    this.btn.onclick = handler
    this.input.addEventListener('keydown', (e: KeyboardEvent) => {
      if (e.key === 'Enter') handler()
    })
  }

  private addAiMessage(text: string): void {
    const div = document.createElement('div')
    div.className = 'msg ai'
    div.textContent = text
    this.chat.appendChild(div)
  }
}
