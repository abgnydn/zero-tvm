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

import { SHIPPED_MODELS, modelBranding } from './zero-tvm/model-registry.js'
import type { ModelSpec } from './compiler/model-spec.js'
import { mountMascot, mascotPalette, type MascotHandle } from './mascot.js'

interface Variant { param: string; spec: ModelSpec; label: string }
interface Group { name: string; params: string; variants: Variant[] }

const chatUrl = (param: string): string => (param ? `zero-tvm.html?model=${param}` : 'zero-tvm.html')

/** Context ceiling, derived from the spec (maxPages x pageSize) — a fact about
 *  the build rather than a figure typed into the page. */
const ctxLabel = (t: number): string => (t >= 1024 ? `${Math.round(t / 1024)}k` : String(t))

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
    const b = modelBranding(spec)
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
const LANE_SIGIL: Record<string, string> = {
  moe: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3 3-3 3-3-3zM5 9l2.4 2.4L5 13.8 2.6 11.4zM19 9l2.4 2.4L19 13.8l-2.4-2.4zM12 16l3 3-3 3-3-3z" opacity="0.9"/><circle cx="12" cy="11.4" r="1.6"/></svg>',
  hybrid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 9c3 0 3-4 6-4s3 4 6 4 3-4 6-4"/><path d="M2 16h6l3-5 3 8 2-3h6" opacity="0.85"/></svg>',
  dense: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1" opacity="0.55"/><rect x="4" y="13" width="7" height="7" rx="1" opacity="0.55"/><rect x="13" y="13" width="7" height="7" rx="1"/></svg>',
  mla: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l8 9-8 9-8-9z"/><path d="M12 8l4 4-4 4-4-4z" fill="currentColor" stroke="none" opacity="0.7"/></svg>',
  embed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12c2.5-4 6-6 9-6s6.5 2 9 6c-2.5 4-6 6-9 6s-6.5-2-9-6z"/><path d="M8 12h8" opacity="0.8"/></svg>',
}
function laneOf(spec: ModelSpec): string {
  return spec.embeddingOnly ? 'embed' : spec.mla ? 'mla' : spec.moe ? 'moe'
    : spec.layerKinds.some((k) => k === 'gdn') ? 'hybrid' : 'dense'
}

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
      <svg class="cs-emblem" viewBox="0 0 96 96" fill="none">
        <path d="M48 6 L90 48 L48 90 L6 48 Z" stroke="currentColor" stroke-width="3"/>
        <path d="M48 20 L76 48 L48 76 L20 48 Z" stroke="currentColor" stroke-width="1.4" opacity="0.55"/>
        <path d="M30 48 h12 l6 -10 l6 20 l6 -10 h6" stroke="currentColor" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>
        <path d="M48 2 l4 4 l-4 4 l-4 -4 Z M48 86 l4 4 l-4 4 l-4 -4 Z M2 48 l4 -4 l4 4 l-4 4 Z M86 48 l4 -4 l4 4 l-4 4 Z" fill="currentColor"/>
      </svg>
      <span class="cs-logo">zero<b>-tvm</b></span>
      <span class="cs-sub">LLM inference in the browser · hand-written WGSL</span>
    </div>
    <div class="cs-fog" aria-hidden="true"><i></i><i></i></div>
    <div class="cs-dust" aria-hidden="true"></div>
    <div class="mb-plate">
      <div class="mb-name"></div>
      <div class="mb-params"><span class="mb-sigil" aria-hidden="true"></span><span class="mb-params-text"></span></div>
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
        <div class="mb-row-label">Quantisation</div>
        <div class="mb-variants" role="tablist" aria-label="Quantisation"></div>
        <dl class="mb-stats"></dl>
        <div class="mb-row-label mb-modes-label" hidden>Memory build</div>
        <div class="mb-modes" role="tablist" aria-label="Memory build"></div>
        <p class="mb-cached" hidden>Already on this device — opens in seconds</p>
        <p class="mb-ram"></p>
      </div>
    </aside>
    <div class="cs-roster-head" aria-hidden="true">Roster · ${GROUPS.length}</div>
    <div class="mb-dots mb-roster" role="tablist" aria-label="Models"></div>
    <div class="cs-enter"><a class="mb-cta btn btn-primary">Enter chat ▸</a></div>
    <div class="cs-corner cs-corner-l" aria-hidden="true">registry-rendered · numbers are measured</div>
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
      splash.addEventListener('animationend', () => splash.remove())
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
  let mascot: MascotHandle | null = null
  const poolFracOf = (spec: ModelSpec, slots: number): number =>
    slots && spec.moe ? slots / (spec.moe.experts + 1) : 0

  dots.innerHTML = GROUPS.map((g, i) =>
    `<button class="mb-dot" data-i="${i}" role="tab" title="${g.name}" aria-label="${g.name}">
       <span class="mb-dot-face" aria-hidden="true"></span>
       <span class="mb-dot-text"><b>${g.name.replace(/-Instruct.*$/, '')}</b><i>${g.params}</i></span>
     </button>`).join('')

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
      m.setSpec(spec, CACHED.has(spec.id))
      await new Promise((r) => setTimeout(r, 40))
      const url = await m.snapshot()
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
    // Class-coloured chrome: the pedestal, plate and CTA take the lane accent
    // of the model on stage — the same colour its chat page runs.
    const pal = mascotPalette(v.spec)
    root.style.setProperty('--cs-accent', pal.accent)
    root.style.setProperty('--cs-accent-hi', pal.accentHi)
    // The CHARACTER SCREEN contract: the chat link carries the chosen build,
    // so what you picked here is what boots there — ?pool= is the same number
    // the registry row and the engine read.
    const modes = b.poolModes ?? []
    const mode = modes[mi] ?? modes[0]
    el<HTMLAnchorElement>('.mb-cta').href =
      chatUrl(v.param) + (mode && mode.slots ? `${v.param ? '&' : '?'}pool=${mode.slots}` : '')

    el('.mb-modes').innerHTML = modes.length < 2 ? '' : modes.map((x, i) =>
      `<button class="mb-variant mb-mode" data-m="${i}" role="tab" aria-selected="${i === mi}">${x.label}</button>`).join('')
    ;(el('.mb-modes-label') as HTMLElement).hidden = modes.length < 2

    el('.mb-variants').innerHTML = g.variants.length < 2 ? '' : g.variants.map((x, i) =>
      `<button class="mb-variant" data-v="${i}" role="tab" aria-selected="${i === vi}">${x.label}</button>`).join('')

    // Every figure is read from the spec or from a measured rate label. A model
    // with no measurement renders no speed row rather than a guess.
    const rows: Array<[string, string]> = [
      ['Weights', b.sizeLabel],
      ['Context', `${ctxLabel(v.spec.maxContext)} tokens`],
    ]
    if (b.rateLabel) rows.push(['Speed', `${b.rateLabel} <span class="mb-hw">M2 Max</span>`])
    el('.mb-stats').innerHTML = rows.map(([k, val]) =>
      `<div><dt>${k}</dt><dd>${val}</dd></div>`).join('')

    const ram = el<HTMLElement>('.mb-ram')
    // A pooled build's note replaces the full model's RAM warning — picking
    // less memory IS the answer to that warning.
    ram.textContent = mode && mode.slots ? (mode.note ?? '') : (b.ramNote ?? '')
    ram.hidden = !(mode && mode.slots ? mode.note : b.ramNote)

    for (const d of root.querySelectorAll<HTMLElement>('.mb-dot')) {
      d.setAttribute('aria-selected', String(Number(d.dataset.i) === gi))
      const gSpec = GROUPS[Number(d.dataset.i)].variants[0].spec
      d.toggleAttribute('data-cached', CACHED.has(gSpec.id))
    }
    const cached = el<HTMLElement>('.mb-cached')
    cached.hidden = !CACHED.has(v.spec.id)

    mascot?.setSpec(v.spec, CACHED.has(v.spec.id), mode ? poolFracOf(v.spec, mode.slots) : 0)
  }

  /** Selection transition: retrigger the stage flash + plate/sheet entrance
   *  animations by yanking the class off for a frame. */
  function selectFx(): void {
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
    paint()
    selectFx()
  }

  host.addEventListener('click', (e) => {
    const t = e.target as HTMLElement
    const arrow = t.closest<HTMLElement>('.mb-arrow')
    if (arrow) { go(gi + Number(arrow.dataset.dir)); return }
    const dot = t.closest<HTMLElement>('.mb-dot')
    if (dot) { go(Number(dot.dataset.i)); return }
    const modeBtn = t.closest<HTMLElement>('.mb-mode')
    if (modeBtn) { mi = Number(modeBtn.dataset.m); paint(); return }
    const variant = t.closest<HTMLElement>('.mb-variant')
    if (variant) { vi = Number(variant.dataset.v); mi = 0; paint() }
  })
  host.tabIndex = 0
  host.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') go(gi + 1)
    else if (e.key === 'ArrowLeft') go(gi - 1)
  })
  const art = el<HTMLElement>('.mb-art')
  art.addEventListener('mouseenter', () => mascot?.setHover(true))
  art.addEventListener('mouseleave', () => mascot?.setHover(false))

  // Depth: the fog and dust track the pointer at different rates — the cheap
  // half of a camera. Skipped entirely under reduced motion.
  if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
    const fog = root.querySelector<HTMLElement>('.cs-fog')
    const dust = root.querySelector<HTMLElement>('.cs-dust')
    const art = root.querySelector<HTMLElement>('.mb-art')
    root.addEventListener('pointermove', (e) => {
      const r = root.getBoundingClientRect()
      const x = (e.clientX - r.left) / r.width - 0.5
      const y = (e.clientY - r.top) / r.height - 0.5
      if (fog) fog.style.transform = `translate(${x * 26}px, ${y * 10}px)`
      if (dust) dust.style.transform = `translate(${x * 12}px, ${y * 5}px)`
      if (art) art.style.transform = `translate(${x * -7}px, ${y * -3}px)`
    })
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
