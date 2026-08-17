/**
 * LANDING ROOM — hosting as part of the game, not a separate page.
 *
 * The ⟁ Room tool in the entrance chat opens a strip inside the panel:
 * consent copy first (hosting runs guests' prompts on THIS GPU — the click
 * that opens the room is the click that accepts that, with the words right
 * above it), then the live room: invite link, membership, and the request
 * feed — your machine never runs someone's prompt without you seeing it.
 *
 * Guests appear ON STAGE: one small mascot per connected guest, flanking the
 * character on its ring. The link they open is share.html#room — the guest
 * surface stays the thin no-WebGPU page; what moved here is the HOST's seat.
 * The loop itself is room-host.ts, shared with share.html, so the two
 * hosting surfaces cannot drift.
 *
 * Dynamically imported by landing-chat.ts once the engine is live.
 */

import type { ModelSpec } from './compiler/model-spec.js'
import type { DecodeEngine } from './zero-tvm/engine-core.js'
import type { Tokenizer } from './zero-tvm/tokenizer.js'
import type { MascotHandle } from './mascot.js'
import { mountMascot } from './mascot.js'
import { modelBranding } from './zero-tvm/model-registry.js'
import { buildChatPromptFor } from './zero-tvm/model-select.js'
import { hostRoom, type RoomHandle } from './zero-tvm/room-host.js'
import { recordFeat } from './feats.js'

export interface RoomToolOptions {
  root: HTMLElement
  panel: HTMLElement
  spec: ModelSpec
  param: string
  engine: DecodeEngine
  tokenizer: Tokenizer
  mascot: MascotHandle | null
  /** Open the strip immediately (the "Enter & open a room" path) — on the
   *  CONSENT step; hosting still takes the explicit click. */
  openStrip?: boolean
}

export function mountRoomTool(o: RoomToolOptions): void {
  const head = o.panel.querySelector('.cs-chat-head')
  if (!head) return
  const brand = modelBranding(o.spec)

  const btn = document.createElement('button')
  btn.type = 'button'
  btn.className = 'cs-chat-tool'
  btn.id = 'room-btn'
  btn.textContent = '⟁ Room'
  btn.title = 'Serve this model to other machines'
  head.insertBefore(btn, head.querySelector('#new-chat-btn'))

  const strip = document.createElement('div')
  strip.className = 'cs-room'
  strip.hidden = !o.openStrip
  strip.innerHTML = `
    <div class="cs-room-consent">
      <p>Open a room and whoever has the link chats with <b>${brand.name}</b> running on
      THIS machine. Their prompts run on your GPU; every request is listed here as it
      arrives. Keep this tab in the foreground while serving.</p>
      <button type="button" class="cs-chat-tool" id="room-open">Open room →</button>
    </div>
    <div class="cs-room-live" hidden>
      <div class="cs-room-linkrow">
        <input id="room-link" readonly aria-label="Room link">
        <button type="button" class="cs-chat-tool" id="room-copy">Copy</button>
        <button type="button" class="cs-chat-tool" id="room-close">Close room</button>
      </div>
      <div class="cs-room-members" id="room-members">waiting for guests…</div>
      <ul class="cs-room-log" id="room-log"></ul>
    </div>`
  head.insertAdjacentElement('afterend', strip)

  const $ = <T extends HTMLElement>(sel: string): T => strip.querySelector(sel) as T
  let room: RoomHandle | null = null
  let wake: { release(): Promise<void> } | null = null
  /** One small mascot per connected guest, flanking the character. */
  const guestMascots: { canvas: HTMLCanvasElement; handle: MascotHandle | null }[] = []

  function syncGuestMascots(n: number): void {
    const stage = o.root.querySelector('.mb-art')
    if (!stage) return
    while (guestMascots.length < Math.min(n, 6)) {
      const slot = guestMascots.length
      const c = document.createElement('canvas')
      c.className = `cs-guest-mascot cs-guest-${slot}`
      c.setAttribute('aria-hidden', 'true')
      stage.appendChild(c)
      const entry = { canvas: c, handle: null as MascotHandle | null }
      guestMascots.push(entry)
      void mountMascot(c, o.spec).then((m) => {
        if (!m) { c.remove(); return }
        entry.handle = m
      })
    }
    while (guestMascots.length > n) {
      const g = guestMascots.pop()
      g?.handle?.destroy()
      g?.canvas.remove()
    }
  }

  btn.addEventListener('click', () => { strip.hidden = !strip.hidden })

  $('#room-open').addEventListener('click', () => {
    if (room) return
    room = hostRoom({
      spec: o.spec,
      brand,
      param: o.param,
      engine: o.engine,
      tokenizer: o.tokenizer,
      encode: (messages) => buildChatPromptFor(o.spec, messages, o.tokenizer),
      ui: {
        row: (who, text) => {
          const log = $('#room-log')
          const li = document.createElement('li')
          li.innerHTML = `<b></b><span></span><i></i>`
          ;(li.querySelector('b') as HTMLElement).textContent = who.slice(0, 8)
          ;(li.querySelector('span') as HTMLElement).textContent = text
          log.prepend(li)
          while (log.children.length > 6) log.lastChild?.remove()
          const st = li.querySelector('i') as HTMLElement
          return (s: string) => { st.textContent = s }
        },
        onMembers: ({ hosts, guests, connected }) => {
          $('#room-members').textContent = connected === 0
            ? 'waiting for guests…'
            : `${hosts} ${hosts === 1 ? 'machine' : 'machines'} serving · `
              + `${guests} ${guests === 1 ? 'guest' : 'guests'} in the room`
          syncGuestMascots(connected)
        },
        // The character speaks for its guests too — same mouth, same
        // honest cadence, whoever asked the question.
        onToken: () => o.mascot?.pulse(),
        onBusy: (busy) => {
          o.mascot?.setMood(busy ? 'talking' : 'idle')
          o.root.classList.toggle('cs-generating', busy)
        },
      },
    })
    ;($('#room-link') as unknown as HTMLInputElement).value = room.link
    $('.cs-room-consent').hidden = true
    $('.cs-room-live').hidden = false
    btn.classList.add('cs-tool-live')
    recordFeat('room')
    // Best-effort: hosting from a sleeping screen serves nobody.
    void (navigator as unknown as { wakeLock?: { request(t: string): Promise<{ release(): Promise<void> }> } })
      .wakeLock?.request('screen').then((l) => { wake = l }).catch(() => {})
  })

  $('#room-copy').addEventListener('click', () => {
    if (!room) return
    void navigator.clipboard.writeText(room.link)
    $('#room-copy').textContent = 'Copied'
    setTimeout(() => { $('#room-copy').textContent = 'Copy' }, 1200)
  })

  $('#room-close').addEventListener('click', () => {
    room?.close()
    room = null
    void wake?.release().catch(() => {})
    wake = null
    syncGuestMascots(0)
    $('.cs-room-live').hidden = true
    $('.cs-room-consent').hidden = false
    btn.classList.remove('cs-tool-live')
    $('#room-members').textContent = 'waiting for guests…'
  })
}
