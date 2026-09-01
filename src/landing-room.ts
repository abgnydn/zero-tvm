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
import { swarmUrls } from './zero-tvm/room-url.js'
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
  /** The engine's single-owner latch, shared with the local chat surface. */
  lock: import('./zero-tvm/engine-lock.js').EngineLock
  /** Expert-pool slots the engine booted with (0 = full) — a pooled host
   *  must not advertise the full model's measured rate to guests. */
  poolSlots: number
  /** This tab holds only these layers — the room serves a STAGE, and the
   *  helper links it writes start where this one ends. */
  stageRange?: { start: number; end: number }
  /** The split this stage belongs to — bounds, which stage, the room context.
   *  Present exactly when the entrance opened a room for a SPLIT. */
  split?: { bounds: number[]; index: number; ctx: number }
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
      arrives. Guests can also copy the model's cached weights from this machine to
      run it locally. Keep this tab in the foreground while serving.</p>
      <button type="button" class="cs-chat-tool" id="room-open">Open room →</button>
    </div>
    <div class="cs-room-live" hidden>
      <div class="cs-room-linkrow">
        <input id="room-link" readonly aria-label="Room link">
        <button type="button" class="cs-chat-tool" id="room-copy">Copy</button>
        <button type="button" class="cs-chat-tool" id="room-close">Close room</button>
      </div>
      <div class="cs-room-split" id="room-split" hidden></div>
      <div class="cs-room-members" id="room-members" aria-live="polite">waiting for guests…</div>
      <ul class="cs-room-log" id="room-log" role="log" aria-live="polite" aria-label="Guest requests"></ul>
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
        // The guest may have LEFT while the mascot was mounting — a canvas
        // no longer in the DOM must not keep a live render loop.
        if (!c.isConnected) { m.destroy(); return }
        entry.handle = m
      })
    }
    while (guestMascots.length > n) {
      const g = guestMascots.pop()
      g?.handle?.destroy()
      g?.canvas.remove()
    }
  }

  btn.setAttribute('aria-expanded', String(!strip.hidden))
  btn.addEventListener('click', () => {
    strip.hidden = !strip.hidden
    btn.setAttribute('aria-expanded', String(!strip.hidden))
  })

  /**
   * The split, as it stands, inside the room. The Split panel could not write
   * these links — it had no room id yet, which is why it asked you to open the
   * host in another tab and paste the link back. Hosting here means the id
   * exists the moment the room does, so every other machine's link can simply
   * be shown. Ranges come from the same swarmUrls the panel used, so the two
   * surfaces cannot disagree about who holds what.
   */
  function paintSplit(roomId: string): void {
    const sp = o.split
    const box = $('#room-split')
    if (!sp || sp.bounds.length < 3) return
    const machines = sp.bounds.length - 1
    const stops = swarmUrls({
      origin: location.origin,
      param: o.param,
      layers: o.spec.layers,
      machines,
      room: roomId,
      ctx: sp.ctx,
      cuts: sp.bounds.slice(1, -1),
    })
    const rows = stops.map((st, i) => {
      const mine = st.role !== 'guest' && i === sp.index
      const who = st.role === 'guest' ? 'Anyone else' : `Machine ${i + 1}`
      const range = st.range ? ` · layers ${st.range.start}–${st.range.end}` : ''
      const link = mine
        ? `<span class="cs-split-mine">running in this tab</span>`
        : `<input readonly value="${st.url ?? ''}" aria-label="${who} link">
           <button type="button" class="cs-chat-tool cs-split-copy">Copy</button>`
      return `<div class="cs-split-row${mine ? ' is-mine' : ''}">
        <div class="cs-split-who"><b>${who}</b><i>${st.role}${range}</i></div>
        <div class="cs-split-link">${link}</div>
      </div>`
    }).join('')
    box.innerHTML = `
      <div class="cs-split-head">Split across ${machines} machines · ${sp.ctx} tokens each</div>
      ${rows}
      <p class="cs-split-note">Changing the split means re-holding different
        layers, so it reopens the room. Close this one first.</p>`
    box.hidden = false
  }

  // Copy on any stage link, same behaviour as the room link's own button.
  strip.addEventListener('click', (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>('.cs-split-copy')
    if (!b) return
    const input = b.parentElement?.querySelector('input')
    if (!input) return
    void navigator.clipboard.writeText(input.value)
    b.textContent = 'Copied'
    setTimeout(() => { b.textContent = 'Copy' }, 1200)
  })

  $('#room-open').addEventListener('click', () => {
    if (room) return
    room = hostRoom({
      spec: o.spec,
      // A pooled host must not tell guests the full model's measured rate —
      // the pooled builds are slower and the guest sees the label as truth.
      brand: o.poolSlots ? { ...brand, rateLabel: '' } : brand,
      param: o.param,
      stageRange: o.stageRange,
      engine: o.engine,
      tokenizer: o.tokenizer,
      lock: o.lock,
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
    paintSplit(room.roomId)
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
