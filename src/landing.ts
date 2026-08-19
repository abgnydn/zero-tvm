/**
 * MODEL BROWSER — one model at a time, rendered from the engine's own table.
 *
 * The old page hand-wrote the list into HTML; after Qwen3.6 landed, twelve
 * separate claims on it were stale. Everything here comes from
 * model-registry.ts, so adding a model adds a screen and nothing on the page
 * can disagree with what the code loads.
 *
 * VARIANTS ARE GROUPED. Qwen3-4B shipped twice (MLC q4f16_1 and MLX 4-bit) and
 * Qwen3.6-35B-A3B ships at two expert widths. Those are the same model at
 * different quantisation, not different models, and listing them as separate
 * cards asked the reader to work that out. They are one screen with a picker
 * now, and the mascot re-skins when the picker changes.
 *
 * Progressive enhancement, strictly: it renders from the LIGHT registry with no
 * WebGPU touched, so a browser without it still sees every model plus a note.
 * The mascot and the cached badge are additive; both fail to silence.
 */

import { SHIPPED_MODELS, modelBranding, specWithCtx } from './zero-tvm/model-registry.js'
import type { ModelSpec } from './compiler/model-spec.js'
import { mountMascot, mascotPalette, type MascotHandle } from './mascot.js'
import { LANE_SIGIL, laneOf, loreOf, abilitiesOf } from './landing-lore.js'
import { FEATS, feats } from './feats.js'

interface Variant { param: string; spec: ModelSpec; label: string }
interface Group { name: string; params: string; variants: Variant[] }

// (the CTA URL is composed in paint() — model + pool + ctx, one query string)

/** Context ceiling, derived from the spec (maxPages x pageSize) — a fact about
 *  the build rather than a figure typed into the page. */
const ctxLabel = (t: number): string => (t >= 1024 ? `${Math.round(t / 1024)}k` : String(t))

/** Whether the engine this page hands off to will quantise its KV cache.
 *  Default ON; `?kv8=0` opts out — the SAME rule variants.ts applies, read
 *  here rather than assumed, so the price shown is the price charged. */
const INT8_KV = new URLSearchParams(location.search).get('kv8') !== '0'

/** KV bytes per token AS ALLOCATED. int8 KV is the default (`?kv8=0` opts
 *  out), so the cache is half the f16 width plus one f16 scale per
 *  (token, kv head, side) PER ATTENTION LAYER — allocKVPagesInt8 makes one
 *  scales buffer per attention layer, and leaving the layer count out understated
 *  Phi-3 by 2% (771.5 MB shown against 787.1 MB allocated) and Llama-3.2 by
 *  more. A figure headed "will it fit" must not round away the thing it
 *  allocates, and must not round it in the reassuring direction.
 *  MLA caches a latent and the int8 path does not cover it, so it stays f16. */
const kvBytesPerToken = (spec: ModelSpec, int8: boolean): number =>
  int8 && !spec.mla
    ? spec.kvBytesPerToken / 2
      + 4 * spec.kvHeads * spec.layerKinds.filter((k) => k === 'attn').length
    : spec.kvBytesPerToken

/** What a context window costs: KV bytes, from the spec's own per-token rate.
 *  Computed, never typed — the same rule as every other figure here. */
const kvPrice = (spec: ModelSpec, tokens: number, int8: boolean): string => {
  const b = tokens * kvBytesPerToken(spec, int8)
  return b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(1)} GB KV` : `${Math.round(b / 2 ** 20)} MB KV`
}

interface CtxMode { name: string; tokens: number }

/** Context builds, derived from the spec: the compiled default (a ~1 GiB KV
 *  budget choice, not a limit), a middle step, and the checkpoint's own
 *  trained window. One entry when the default already IS the trained window
 *  (Phi-3) — then the row does not render. */
function ctxModesOf(spec: ModelSpec): CtxMode[] {
  const out: CtxMode[] = [{ name: 'Standard', tokens: spec.maxContext }]
  const long = Math.min(spec.maxContext * 4, spec.maxSeq)
  if (long > spec.maxContext) out.push({ name: 'Long', tokens: long })
  if (spec.maxSeq > long) out.push({ name: 'Full', tokens: spec.maxSeq })
  return out
}

/** Branding writes "4B dense · q4f16_1": the family is the first part and the
 *  quantisation is the rest. With no separator there is one variant and the
 *  picker does not render. */
function splitParams(s: string): { family: string; variant: string } {
  const i = s.indexOf(' · ')
  return i < 0 ? { family: s, variant: '' } : { family: s.slice(0, i), variant: s.slice(i + 3) }
}

function buildGroups(): Group[] {
  const out = new Map<string, Group>()
  for (const { param, spec } of SHIPPED_MODELS) {
    // The entrance is a CHAT roster. The embedding model returns a vector and
    // does not speak — it stays in the registry (validate + ?model= still
    // work) but earns no place on a character-select screen for conversation.
    if (spec.embeddingOnly) continue
    const b = modelBranding(spec)
    // Pending = generated but not yet numerics-validated. A roster card is a
    // claim the model runs; the claim waits for validate-model.
    if (b.pending) continue
    const { family, variant } = splitParams(b.params)
    let g = out.get(b.name)
    if (!g) { g = { name: b.name, params: family, variants: [] }; out.set(b.name, g) }
    g.variants.push({ param, spec, label: variant || family })
  }
  return [...out.values()]
}

const GROUPS = buildGroups()

/** Class sigils — one per architecture lane, same circuit-rune language as
 *  the /entrance assets. currentColor, so they sit in the accent for free. */
const STAT_ICON: Record<string, string> = {
  Weights: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 1l3 3-3 3-3-3z"/><path d="M4 8l3 3-3 3-3-3z" opacity="0.6"/><path d="M12 8l3 3-3 3-3-3z" opacity="0.6"/></svg>',
  Context: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 2h8l2 2v10H3z"/><path d="M5 6h6M5 9h6M5 12h4" opacity="0.7"/></svg>',
  Speed: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M9 1L3 9h4l-1 6 6-8H8z"/></svg>',
  Footprint: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 11h12v3H2z"/><path d="M4 7h8v4H4z" opacity="0.75"/><path d="M6 3h4v4H6z" opacity="0.5"/></svg>',
}

/** Weights GB parsed from a registry label ('~16.4 GB', 'Half · ~8.4 GB · …').
 *  The labels are the single source for resident size; NaN hides the row. */
const weightsGb = (label: string): number => {
  const m = /([\d.]+)\s*GB/.exec(label)
  return m ? parseFloat(m[1]) : NaN
}

/** A saved conversation exists for this spec (chat-flow.ts's localStorage
 *  key). Presence only — the save-slot glyph, not a tally. */
const hasSave = (specId: string): boolean => {
  try { return (localStorage.getItem(`zt-chat-${specId}`)?.length ?? 0) > 2 } catch { return false }
}

// LANE_SIGIL / laneOf / loreOf live in landing-lore.ts — shared with the
// in-place chat panel (landing-chat.ts), which wears the same sigil.

/** Spec ids whose weights are already in OPFS. Filled asynchronously; the
 *  picker renders without waiting, and a model that turns out to be cached
 *  simply gains a line when the probe lands. Restoring this — the carousel
 *  rewrite dropped it — matters most on exactly the models where it is worth
 *  knowing, since a 16 GB download is not something to start twice. */
const CACHED = new Set<string>()

async function probeCached(repaint: () => void): Promise<void> {
  if (!('gpu' in navigator) || !navigator.storage?.getDirectory) return
  try {
    // Dynamic because the probe pulls in the loaders, which read
    // `GPUBufferUsage` at module scope — a static import would throw on a
    // browser with no WebGPU, which is a browser this page still has to render
    // for. The chat page shares this exact probe: two copies of it disagreed
    // about every MLX model.
    const { isModelCached } = await import('./zero-tvm/cache-probe.js')
    for (const { spec } of SHIPPED_MODELS) {
      if (await isModelCached(spec)) CACHED.add(spec.id)
    }
    repaint()
  } catch { /* no WebGPU / import failed — badges stay hidden */ }
}

function render(): void {
  const host: HTMLElement | null = document.getElementById('model-browser')
  if (host === null) return
  const root: HTMLElement = host

  // THE SELECT SCREEN, WoW layout: nameplate top-centre, the character
  // centre-stage on its light, the roster a vertical rail on the right, the
  // sheet on the left, ENTER CHAT bottom-centre. The splash is the logo
  // opening — once per session, skipped for reduced-motion.
  host.innerHTML = `
    <div class="cs-splash" aria-hidden="true">
      <div class="cs-splash-ring" aria-hidden="true"></div>
      <svg class="cs-emblem" viewBox="0 0 96 96" fill="none">
        <path d="M48 6 L90 48 L48 90 L6 48 Z" stroke="currentColor" stroke-width="3"/>
        <path d="M48 20 L76 48 L48 76 L20 48 Z" stroke="currentColor" stroke-width="1.4" opacity="0.55"/>
        <path d="M30 48 h12 l6 -10 l6 20 l6 -10 h6" stroke="currentColor" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="M48 2 l4 4 l-4 4 l-4 -4 Z M48 86 l4 4 l-4 4 l-4 -4 Z M2 48 l4 -4 l4 4 l-4 4 Z M86 48 l4 -4 l4 4 l-4 4 Z" fill="currentColor"/>
      </svg>
      <span class="cs-logo">zero<b>-tvm</b></span>
      <span class="cs-sub">LLM inference in the browser · hand-written WGSL</span>
    </div>
    <div class="cs-spires" aria-hidden="true"></div>
    <div class="cs-col cs-col-l" aria-hidden="true"></div>
    <div class="cs-col cs-col-r" aria-hidden="true"></div>
    <div class="cs-comet" aria-hidden="true"></div>
    <div class="cs-sigilbg" aria-hidden="true"></div>
    <div class="cs-stream" aria-hidden="true"><i>◆</i><i>◇</i><i>◆</i><i>◇</i><i>◆</i><i>◇</i></div>
    <div class="cs-fog" aria-hidden="true"><i></i><i></i></div>
    <div class="cs-dust" aria-hidden="true"></div>
    <div class="mb-plate">
      <div class="cs-banner" aria-hidden="true"></div>
      <div class="mb-name"></div>
      <div class="mb-params"><span class="mb-sigil" aria-hidden="true"></span><span class="mb-params-text"></span></div>
      <div class="cs-lore"></div>
    </div>
    <div class="mb-stage">
      <button class="mb-arrow" data-dir="-1" aria-label="Previous model">&lsaquo;</button>
      <div class="mb-art">
        <div class="mb-pedestal" aria-hidden="true"></div>
        <canvas class="mb-mascot" aria-hidden="true"></canvas>
      </div>
      <button class="mb-arrow" data-dir="1" aria-label="Next model">&rsaquo;</button>
    </div>
    <aside class="mb-info">
      <div class="mb-panel">
        <div class="mb-row-label mb-variants-label">Quantisation</div>
        <div class="mb-variants" role="tablist" aria-label="Quantisation"></div>
        <dl class="mb-stats"></dl>
        <ul class="mb-abilities"></ul>
        <div class="mb-row-label mb-modes-label" hidden>Memory build</div>
        <div class="mb-modes" role="tablist" aria-label="Memory build"></div>
        <div class="mb-row-label mb-ctxs-label" hidden>Context</div>
        <div class="mb-modes mb-ctxs" role="tablist" aria-label="Context"></div>
        <p class="mb-cached" hidden>Already on this device — opens in seconds</p>
        <p class="mb-save" hidden>◈ Saved conversation — it continues where you left it</p>
        <p class="mb-ram"></p>
      </div>
    </aside>
    ${'gpu' in navigator ? '<div class="cs-deeds" role="list" aria-label="Deeds"></div>' : ''}
    <div class="cs-roster-head" aria-hidden="true">Roster</div>
    <div class="mb-dots mb-roster" role="tablist" aria-label="Models"></div>
    <div class="cs-enter">
      <a class="mb-cta btn btn-primary">Enter chat ▸</a>
      <a class="mb-cta-room">⟁ Enter &amp; open a room — serve this model to other machines</a>
    </div>
    <div class="cs-live" aria-live="polite"></div>
    <div class="cs-wipe" aria-hidden="true"></div>
    <div class="cs-engage" aria-hidden="true"></div>
    <div class="cs-borderline-t" aria-hidden="true"></div>
    <div class="cs-borderline-b" aria-hidden="true"></div>
    <div class="cs-corner cs-corner-l" aria-hidden="true">registry-rendered · measured, or marked est</div>
    <div class="cs-corner cs-corner-r" aria-hidden="true">WebGPU · hand-written WGSL</div>
    ${'gpu' in navigator ? '' :
      '<p class="note cs-note">This browser has no WebGPU — the chat needs Chrome, Edge, or another WebGPU-enabled browser.</p>'}
  `
  {
    const splash = host.querySelector<HTMLElement>('.cs-splash')!
    const still2 = matchMedia('(prefers-reduced-motion: reduce)').matches
    if (still2 || sessionStorage.getItem('zt-intro')) splash.remove()
    else {
      sessionStorage.setItem('zt-intro', '1')
      // Child animations (emblem, logo) bubble their animationend — only the
      // splash's OWN fade-out may remove it, or it vanishes at full opacity.
      splash.addEventListener('animationend', (e) => { if (e.target === splash) splash.remove() })
      const skip = (): void => splash.remove()
      splash.addEventListener('pointerdown', skip)
      window.addEventListener('keydown', skip, { once: true })
    }
  }

  const el = <T extends Element>(s: string): T => host.querySelector<T>(s)!
  const canvas = el<HTMLCanvasElement>('.mb-mascot')
  const dots = el<HTMLElement>('.mb-dots')
  let gi = 0
  let vi = 0
  /** Selected memory build (index into the spec's poolModes; 0 = full). Reset
   *  on every model/variant change — a build belongs to a character. */
  let mi = 0
  /** Selected context build (index into ctxModesOf; 0 = the compiled default).
   *  Context is a KV-memory dial, not a model property — the number the sheet
   *  shows is whichever build is chosen here, priced in KV bytes. */
  let xi = 0
  let mascot: MascotHandle | null = null
  // Denominator counts the shared expert only where one EXISTS — qwen30b has
  // none, and a blanket +1 understated its residency fraction.
  const poolFracOf = (spec: ModelSpec, slots: number): number =>
    slots && spec.moe
      ? slots / (spec.moe.experts + (spec.sharedExpertIndex >= 0 ? 1 : 0))
      : 0

  dots.innerHTML = GROUPS.map((g, i) => {
    const spec0 = g.variants[0].spec
    const pal0 = mascotPalette(spec0)
    return `<button class="mb-dot" data-i="${i}" role="tab" title="${g.name}" aria-label="${g.name}">
       <span class="mb-dot-face" aria-hidden="true"></span>
       <span class="mb-dot-text"><b>${g.name.replace(/-Instruct.*$/, '')}</b><i>${g.params}</i></span>
       <span class="mb-dot-sigil" style="color:${pal0.accent}" aria-hidden="true">${LANE_SIGIL[laneOf(spec0)] ?? ''}</span>
     </button>`
  }).join('')

  /** Roster portraits: one hidden mascot cycles the roster and snapshots each
   *  face — eight PNGs, not eight live render loops. Re-run when the cache
   *  probe lands so unlocked (cached) characters get their lit portrait. */
  async function paintRoster(): Promise<void> {
    const off = document.createElement('canvas')
    off.style.cssText = 'position:absolute;left:-9999px;width:96px;height:96px'
    document.body.appendChild(off)
    const m = await mountMascot(off, GROUPS[0].variants[0].spec)
    if (!m) { off.remove(); return }
    for (let i = 0; i < GROUPS.length; i++) {
      const spec = GROUPS[i].variants[0].spec
      m.setSpec(spec, GROUPS[i].variants.some((x) => CACHED.has(x.spec.id)))
      await new Promise((r) => setTimeout(r, 40))
      // Posed: every portrait the same front-facing frame — a rail of
      // random sway/blink instants read as misaligned cards.
      const url = await m.snapshot(true)
      const dot = root.querySelector<HTMLElement>(`.mb-dot[data-i="${i}"] .mb-dot-face`)
      if (dot) dot.style.setProperty('--thumb', `url("${url}")`)
    }
    m.destroy()
    off.remove()
  }
  void paintRoster()

  function paint(): void {
    const g = GROUPS[gi]
    const v = g.variants[vi]
    const b = modelBranding(v.spec)

    el('.mb-name').textContent = g.name
    el('.mb-params-text').textContent = g.params
    el('.mb-sigil').innerHTML = LANE_SIGIL[laneOf(v.spec)] ?? ''
    el('.cs-lore').textContent = loreOf(v.spec)
    // The constellation: the lane's sigil, huge and faint, behind the stage —
    // as a mask, so the accent wash colours it. The geometry carries the
    // alpha; the SVG's currentColor resolves opaque, which is all a mask needs.
    {
      const bg = root.querySelector<HTMLElement>('.cs-sigilbg')
      const svg = LANE_SIGIL[laneOf(v.spec)]
      if (bg && svg) {
        const uri = `url("data:image/svg+xml,${encodeURIComponent(svg)}")`
        bg.style.webkitMaskImage = uri
        bg.style.maskImage = uri
      }
    }
    // Aura tempo from the measured rate — the same number the mascot's idle
    // clock uses. Unmeasured models get 1.0 rather than a guess.
    const rate2 = parseFloat(b.rateLabel.replace(/[^0-9.]/g, ''))
    root.style.setProperty('--cs-tempo', String(Number.isFinite(rate2) ? Math.min(2.2, 0.7 + rate2 / 140) : 1))
    // A cached character's summoning circle is ARMED: the ring gains a
    // rotating highlight. Real OPFS state, same source as the READY tag.
    el('.mb-art').toggleAttribute('data-armed', CACHED.has(v.spec.id))
    const live = root.querySelector<HTMLElement>('.cs-live')
    if (live) live.textContent = `${g.name}, ${g.params}`
    // Class-coloured chrome: the pedestal, plate and CTA take the lane accent
    // of the model on stage — the same colour its chat page runs.
    const pal = mascotPalette(v.spec)
    root.style.setProperty('--cs-accent', pal.accent)
    root.style.setProperty('--cs-accent-hi', pal.accentHi)
    // The CHARACTER SCREEN contract: the chat link carries the chosen builds,
    // so what you picked here is what boots there — ?pool= and ?ctx= are the
    // same numbers the registry row and the engine read.
    const modes = b.poolModes ?? []
    const mode = modes[mi] ?? modes[0]
    const ctxs = ctxModesOf(v.spec)
    const cx = ctxs[xi] ?? ctxs[0]
    {
      const qs: string[] = []
      if (v.param) qs.push(`model=${v.param}`)
      if (mode && mode.slots) qs.push(`pool=${mode.slots}`)
      if (cx.tokens !== v.spec.maxContext) qs.push(`ctx=${cx.tokens}`)
      el<HTMLAnchorElement>('.mb-cta').href = `zero-tvm.html${qs.length ? '?' + qs.join('&') : ''}`
      // The room path carries the SAME build choices — share.html reads
      // ?pool= and ?ctx= too, so a modified click hosts the build that was
      // chosen, not silently the full model.
      el<HTMLAnchorElement>('.mb-cta-room').href = `share.html${qs.length ? '?' + qs.join('&') : ''}`
    }

    el('.mb-modes').innerHTML = modes.length < 2 ? '' : modes.map((x, i) =>
      `<button class="mb-variant mb-mode" data-m="${i}" role="tab" aria-selected="${i === mi}">`
      + `<span class="mb-gauge" aria-hidden="true">${'◆'.repeat(modes.length - i)}${'◇'.repeat(i)}</span>${x.label}</button>`).join('')
    ;(el('.mb-modes-label') as HTMLElement).hidden = modes.length < 2

    // Context is a KV-memory dial, not a model constant — the engine's page
    // table is a budget choice, and specWithCtx rebuilds it at boot. Each chip
    // carries its true KV price, computed from the spec's own per-token rate.
    el('.mb-ctxs').innerHTML = ctxs.length < 2 ? '' : ctxs.map((c, i) =>
      `<button class="mb-variant mb-ctx" data-x="${i}" role="tab" aria-selected="${i === xi}">`
      + `${c.name} · ${ctxLabel(c.tokens)} · ~${kvPrice(v.spec, c.tokens, INT8_KV)}</button>`).join('')
    ;(el('.mb-ctxs-label') as HTMLElement).hidden = ctxs.length < 2

    el('.mb-variants').innerHTML = g.variants.length < 2 ? '' : g.variants.map((x, i) =>
      `<button class="mb-variant" data-v="${i}" role="tab" aria-selected="${i === vi}"`
      + `${CACHED.has(x.spec.id) ? ' data-cached' : ''}>${x.label}</button>`).join('')
    // A section label over an empty picker reads as a rendering fault — 5 of
    // 7 groups ship one quantisation.
    ;(el('.mb-variants-label') as HTMLElement).hidden = g.variants.length < 2

    // Every figure is read from the spec or from a measured rate label. A model
    // with no measurement renders no speed row rather than a guess.
    const rows: Array<[string, string]> = [
      ['Weights', b.sizeLabel],
      ['Context', `${ctxLabel(cx.tokens)} tokens`],
    ]
    // Suppressed when a pooled build is chosen: the measured rate belongs to
    // the FULL model, and the sheet must not contradict the chip beside it.
    if (b.rateLabel && !(mode && mode.slots)) rows.push(['Speed', `${b.rateLabel} <span class="mb-hw">M2 Max</span>`])
    // Boot footprint: resident weights (the chosen build's label) + the
    // chosen window's KV — the number that answers "will it fit?", assembled
    // from the same sources every other figure reads. Kept in scope because
    // the RAM warning below has to agree with it.
    let footprintGb = NaN
    {
      const w = weightsGb(mode && mode.slots ? mode.label : b.sizeLabel)
      if (Number.isFinite(w)) {
        // Hybrids also allocate the GDN rewind ring — four snapshots of the
        // recurrent state so a late prompt divergence replays from the nearest
        // chunk boundary instead of re-reading the conversation. It is real
        // resident memory (0.19-0.24 GB), allocated whenever prefix reuse is
        // on, which is the default. A row headed "will it fit?" that omits it
        // is short by exactly the amount that decides a close call.
        const sp = v.spec
        const gdnLayers = sp.layerKinds.filter((k) => k === 'gdn').length
        const ring = gdnLayers
          ? (4 * gdnLayers * ((sp.gdnConvK - 1) * sp.gdnQkvDim * 2 + sp.gdnVHeads * sp.gdnStatePerHead * 4)) / 2 ** 30
          : 0
        footprintGb = w + (cx.tokens * kvBytesPerToken(sp, INT8_KV)) / 2 ** 30 + ring
        rows.push(['Footprint', `~${footprintGb.toFixed(1)} GB <span class="mb-hw">weights + KV${ring ? ' + state' : ''}</span>`])
      }
    }
    // Context bar: the CHOSEN build over the checkpoint's own maxSeq — a
    // within-character fraction that fills as the picker climbs to Full.
    // Memory bar: the chosen build's slots over the full expert count.
    const ctxFrac = Math.min(1, cx.tokens / v.spec.maxSeq)
    const memFrac = mode && mode.slots ? (poolFracOf(v.spec, mode.slots) || 1) : 1
    const bar = (frac: number, title: string): string =>
      `<span class="mb-bar" title="${title}"><i style="width:${Math.round(frac * 100)}%"></i></span>`
    el('.mb-stats').innerHTML = rows.map(([k, val]) => {
      const icon = STAT_ICON[k] ? `<span class="mb-stat-ico" aria-hidden="true">${STAT_ICON[k]}</span>` : ''
      const sub = k === 'Context' ? bar(ctxFrac, `${Math.round(ctxFrac * 100)}% of the checkpoint's trained ${ctxLabel(v.spec.maxSeq)} window`)
        : k === 'Weights' && memFrac < 1 ? bar(memFrac, `${Math.round(memFrac * 100)}% of experts resident in this build`)
        : ''
      return `<div><dt>${icon}${k}</dt><dd>${val}${sub}</dd></div>`
    }).join('')
    // Count-up: numbers roll to their real value in ~300 ms. The final frame
    // is always the exact registry string; the roll is presentation only.
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      for (const dd of root.querySelectorAll<HTMLElement>('.mb-stats dd')) {
        const final = dd.innerHTML
        const m2 = /^([^0-9]*)([0-9]+(?:\.[0-9]+)?)([\s\S]*)$/.exec(dd.textContent ?? '')
        if (!m2) continue
        const target = parseFloat(m2[2])
        const dec = m2[2].includes('.') ? 1 : 0
        const t0 = performance.now()
        const tick = (): void => {
          const f = Math.min(1, (performance.now() - t0) / 300)
          const eased = 1 - (1 - f) * (1 - f)
          dd.textContent = `${m2[1]}${(target * (0.4 + 0.6 * eased)).toFixed(dec)}${m2[3]}`
          if (f < 1) requestAnimationFrame(tick)
          else dd.innerHTML = final
        }
        requestAnimationFrame(tick)
      }
    }

    // Abilities: the spec's mechanics as passives — rune bullets, true clauses.
    el('.mb-abilities').innerHTML = abilitiesOf(v.spec).map((a) =>
      `<li><b>${a.name}</b><span>${a.desc}</span></li>`).join('')

    const ram = el<HTMLElement>('.mb-ram')
    // A pooled build's note replaces the full model's RAM warning — picking
    // less memory IS the answer to that warning. A long-context build adds
    // its own: the KV is allocated EAGERLY at boot, so the price is upfront.
    // DERIVED from the footprint, not a fixed string. ramNote is authored per
    // model and cannot know which context the reader just picked, so at 256k
    // it kept promising the 16k answer — the one row headed "will it fit?"
    // disagreeing with the warning beside it. 1.15x rounded reproduces the
    // authored notes at their default windows (Qwen3.8: 15.7 -> 18) and keeps tracking
    // when the picker moves. Any prose in the authored note survives; only its
    // number is replaced.
    // max(derived, authored): the derived figure tracks the context picker, but
    // int8 halving the cache must never WEAKEN a hand-authored RAM warning. It
    // did — qwen36q3 rendered "needs ~19 GB" under an authored ~20, qwen36 ~23
    // against ~24, qwen38 ~17 against ~18 — so the entrance was quietly telling
    // people a model fits in less than the author had measured it needing.
    const derivedGb = Number.isFinite(footprintGb) ? Math.round(footprintGb * 1.15) : NaN
    const authoredGb = Number.parseFloat(
      /needs\s*~?([\d.]+)\s*GB free RAM/i.exec(
        (mode && mode.slots ? (mode.note ?? '') : (b.ramNote ?? '')),
      )?.[1] ?? '')
    const needGb = Number.isFinite(derivedGb) && Number.isFinite(authoredGb)
      ? Math.max(derivedGb, Math.round(authoredGb))
      : Number.isFinite(derivedGb) ? derivedGb : Math.round(authoredGb)
    const authored = mode && mode.slots ? (mode.note ?? '') : (b.ramNote ?? '')
    const qualifier = authored.replace(/needs\s*~?[\d.]+\s*GB free RAM/i, '').replace(/^[\s—-]+/, '').trim()
    const baseNote = Number.isFinite(needGb)
      ? `needs ~${needGb} GB free RAM${qualifier ? ` — ${qualifier}` : ''}`
      : authored
    const ctxNote = cx.tokens > v.spec.maxContext
      ? `long context allocates ~${kvPrice(v.spec, cx.tokens, INT8_KV)} at boot, on top of the weights`
      : ''
    ram.textContent = [baseNote, ctxNote, b.qualityNote].filter(Boolean).join(' — ')
    ram.hidden = !(baseNote || ctxNote || b.qualityNote)

    for (const d of root.querySelectorAll<HTMLElement>('.mb-dot')) {
      d.setAttribute('aria-selected', String(Number(d.dataset.i) === gi))
      // ANY variant counts: the user who downloaded the 3-bit 35B has the
      // group ready even though variants[0] is the 4-bit — keying the badge
      // to variants[0] alone hid exactly the downloads worth showing.
      const grp = GROUPS[Number(d.dataset.i)]
      d.toggleAttribute('data-cached', grp.variants.some((x) => CACHED.has(x.spec.id)))
      // Save slot: a conversation stored on this device continues on enter.
      d.toggleAttribute('data-save', grp.variants.some((x) => hasSave(x.spec.id)))
    }
    const cached = el<HTMLElement>('.mb-cached')
    cached.hidden = !CACHED.has(v.spec.id)
    el<HTMLElement>('.mb-save').hidden = !hasSave(v.spec.id)

    // The character wears the chosen builds: pool leans the figure, and the
    // context build sets the EARS (earCtx is derived from maxContext) — pick
    // Full and they grow.
    mascot?.setSpec(specWithCtx(v.spec, cx.tokens), CACHED.has(v.spec.id),
      mode ? poolFracOf(v.spec, mode.slots) : 0)
  }

  /** DEEDS: things that happened on this device — lit runes, earned names.
   *  Unearned deeds keep their glyph and tooltip (the trigger is stated, so
   *  the rail is a quest list, not decoration); earned ones show the name. */
  function paintDeeds(): void {
    const rail = root.querySelector<HTMLElement>('.cs-deeds')
    if (!rail) return
    const have = feats()
    // aria-label as well as title: an unearned deed's only visible content is
    // a glyph, so screen readers had nothing to read at all.
    rail.innerHTML = FEATS.map((f) => {
      const label = `${have.has(f.id) ? 'Earned' : 'Not yet earned'}: ${f.name} — ${f.desc}`
      return `<span class="cs-deed" role="listitem"${have.has(f.id) ? ' data-on' : ''} `
        + `title="${f.name} — ${f.desc}" aria-label="${label}">◆<i>${f.name}</i></span>`
    }).join('')
  }
  document.addEventListener('zt-feat', paintDeeds)
  paintDeeds()

  /** Selection transition: retrigger the stage flash + plate/sheet entrance
   *  animations by yanking the class off for a frame. */
  function selectFx(): void {
    const wipe = root.querySelector<HTMLElement>('.cs-wipe')
    if (wipe) { wipe.classList.remove('cs-go'); void wipe.offsetWidth; wipe.classList.add('cs-go') }
    for (const sel of ['.mb-art', '.mb-plate', '.mb-panel']) {
      const n = root.querySelector<HTMLElement>(sel)
      if (!n) continue
      n.classList.remove('cs-in')
      void n.offsetWidth
      n.classList.add('cs-in')
    }
  }

  function go(nextG: number): void {
    gi = (nextG + GROUPS.length) % GROUPS.length
    vi = 0
    mi = 0
    xi = 0
    paint()
    selectFx()
  }

  host.addEventListener('click', (e) => {
    const t = e.target as HTMLElement
    const arrow = t.closest<HTMLElement>('.mb-arrow')
    if (arrow) { go(gi + Number(arrow.dataset.dir)); return }
    const dot = t.closest<HTMLElement>('.mb-dot')
    if (dot) { go(Number(dot.dataset.i)); return }
    // paint() re-creates the chips, destroying the button that was just
    // activated — restore focus to its successor or a keyboard user lands on
    // <body> after every choice.
    const refocus = (sel: string): void => { root.querySelector<HTMLElement>(sel)?.focus() }
    const modeBtn = t.closest<HTMLElement>('.mb-mode')
    if (modeBtn) { mi = Number(modeBtn.dataset.m); paint(); refocus(`.mb-mode[data-m="${mi}"]`); return }
    const ctxBtn = t.closest<HTMLElement>('.mb-ctx')
    if (ctxBtn) { xi = Number(ctxBtn.dataset.x); paint(); refocus(`.mb-ctx[data-x="${xi}"]`); return }
    const variant = t.closest<HTMLElement>('.mb-variant')
    if (variant) { vi = Number(variant.dataset.v); mi = 0; xi = 0; paint(); refocus(`.mb-variant[data-v="${vi}"]`) }
  })
  host.tabIndex = 0
  host.setAttribute('role', 'application')
  host.setAttribute('aria-label', 'Character select — Up and Down arrows change model, Enter opens the chat')
  host.addEventListener('keydown', (e) => {
    // In chat mode the keyboard belongs to the composer — arrows must not
    // switch characters under a conversation.
    if (root.classList.contains('cs-chatting')) return
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); go(gi + 1) }
    else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); go(gi - 1) }
    else if (e.key === 'Enter' && (e.target as HTMLElement).tagName !== 'BUTTON' && (e.target as HTMLElement).tagName !== 'A') {
      root.querySelector<HTMLAnchorElement>('.mb-cta')?.click()
    }
  })
  const art = el<HTMLElement>('.mb-art')
  art.addEventListener('mouseenter', () => mascot?.setHover(true))
  art.addEventListener('mouseleave', () => mascot?.setHover(false))
  // The character notices the roster: hovering the rail turns the stage a
  // few degrees toward it and wakes the mascot's hover state — attention,
  // not animation for its own sake.
  dots.addEventListener('mouseenter', () => {
    art.style.rotate = 'y -6deg'
    mascot?.setHover(true)
  }, true)
  dots.addEventListener('mouseleave', () => {
    art.style.rotate = 'none'
    mascot?.setHover(false)
  }, true)

  // Depth: the fog and dust track the pointer at different rates — the cheap
  // half of a camera. Skipped entirely under reduced motion.
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const layers: [HTMLElement | null, number, number][] = [
      [root.querySelector('.cs-spires'), 6, 2],
      [root.querySelector('.cs-col-l'), 34, 8],
      [root.querySelector('.cs-col-r'), 34, 8],
      [root.querySelector('.cs-fog'), 26, 10],
      [root.querySelector('.cs-dust'), 12, 5],
      [root.querySelector('.cs-sigilbg'), 9, 4],
      [root.querySelector('.mb-art'), -7, -3],
    ]
    root.addEventListener('pointermove', (e) => {
      const r = root.getBoundingClientRect()
      const x = (e.clientX - r.left) / r.width - 0.5
      const y = (e.clientY - r.top) / r.height - 0.5
      for (const [n, fx, fy] of layers) {
        if (n) n.style.transform = `translate(${x * fx}px, ${y * fy}px)`
      }
      // The character watches the hand — same normalized pointer the
      // parallax reads, straight into the irises.
      mascot?.setGaze(x * 2, y * 2)
    })
  }

  // The realm line is the REAL adapter. adapter.info ships in current
  // Chromium; anywhere it does not, the static truth stays.
  void (async () => {
    try {
      const ad = await navigator.gpu?.requestAdapter()
      const info = (ad as { info?: { vendor?: string; architecture?: string } } | null)?.info
      const cornerR = root.querySelector<HTMLElement>('.cs-corner-r')
      if (cornerR && info && (info.vendor || info.architecture)) {
        cornerR.textContent = `realm: ${[info.vendor, info.architecture].filter(Boolean).join(' · ')} — online`
      }
    } catch { /* the static line stands */ }
  })()

  // ENTER stays home. The ceremony flashes, the roster steps aside, and the
  // chat mounts IN the entrance — the character never leaves the stage
  // (landing-chat.ts, imported only now because its chain touches WebGPU
  // globals). The href still points at zero-tvm.html?model=&pool= — that is
  // what modified clicks, middle-click, and browsers without WebGPU get.
  const engage = (e: MouseEvent, openRoom: boolean): void => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    if (root.classList.contains('cs-chatting')) { e.preventDefault(); return }
    if (!('gpu' in navigator)) return  // navigate — the fallback page explains what is missing
    const href = (e.currentTarget as HTMLAnchorElement).href
    e.preventDefault()
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      root.querySelector<HTMLElement>('.cs-engage')?.classList.add('cs-go')
    }
    const v = GROUPS[gi].variants[vi]
    const modes = modelBranding(v.spec).poolModes ?? []
    const mode = modes[mi] ?? modes[0]
    const cx = ctxModesOf(v.spec)[xi] ?? ctxModesOf(v.spec)[0]
    import('./landing-chat.js').then(({ enterChat }) => enterChat({
      root,
      spec: v.spec,
      param: v.param,
      poolSlots: mode?.slots ?? 0,
      poolLabel: mode && mode.slots ? mode.label : '',
      ctxTokens: cx.tokens !== v.spec.maxContext ? cx.tokens : 0,
      openRoom,
      mascot,
    })).catch((err) => {
      // The panel could not even mount — fall back to the standalone page.
      console.error('[landing] in-place chat failed, navigating:', err)
      location.href = href
    })
  }
  root.querySelector<HTMLAnchorElement>('.mb-cta')?.addEventListener('click', (e) => engage(e, false))
  // The room path is the same summoning with the room strip already open on
  // its CONSENT step — discoverable from the select screen, never auto-armed.
  root.querySelector<HTMLAnchorElement>('.mb-cta-room')?.addEventListener('click', (e) => engage(e, true))

  // Idle: 18 s without input and the realm starts breathing on its own; 50 s
  // and the character falls ASLEEP — heavy lids, slow breath, dim ring — but
  // never mid-work (summoning/thinking/talking defer the nap). Any movement
  // hands the camera back and wakes it. Reduced motion never idles.
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    let idleTimer = 0
    let sleepTimer = 0
    const busyNow = (): boolean => ['cs-summoning', 'cs-thinking', 'cs-generating']
      .some((c) => root.classList.contains(c))
    const sleep = (): void => {
      if (busyNow()) { sleepTimer = window.setTimeout(sleep, 50000); return }
      root.classList.add('cs-asleep')
      mascot?.setMood('sleepy')
    }
    const wake = (): void => {
      if (root.classList.contains('cs-asleep')) {
        root.classList.remove('cs-asleep')
        if (!busyNow()) mascot?.setMood('idle')
      }
      root.classList.remove('cs-idle')
      clearTimeout(idleTimer)
      clearTimeout(sleepTimer)
      idleTimer = window.setTimeout(() => root.classList.add('cs-idle'), 18000)
      sleepTimer = window.setTimeout(sleep, 50000)
    }
    for (const ev of ['pointermove', 'pointerdown', 'keydown']) root.addEventListener(ev, wake)
    wake()
  }

  paint()
  selectFx()
  void probeCached(() => { paint(); void paintRoster() })
  void mountMascot(canvas, GROUPS[gi].variants[vi].spec).then((m) => {
    if (!m) { art.style.display = 'none'; return }
    mascot = m
    m.setSpec(GROUPS[gi].variants[vi].spec, CACHED.has(GROUPS[gi].variants[vi].spec.id))
  })
}

render()
