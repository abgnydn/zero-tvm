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

  host.innerHTML = `
    <div class="mb-stage">
      <button class="mb-arrow" data-dir="-1" aria-label="Previous model">&lsaquo;</button>
      <div class="mb-slide">
        <div class="mb-art">
          <div class="mb-pedestal" aria-hidden="true"></div>
          <canvas class="mb-mascot" aria-hidden="true"></canvas>
          <div class="mb-plate">
            <div class="mb-name"></div>
            <div class="mb-params"></div>
          </div>
        </div>
        <div class="mb-info">
          <div class="mb-panel">
            <div class="mb-row-label">Quantisation</div>
            <div class="mb-variants" role="tablist" aria-label="Quantisation"></div>
            <dl class="mb-stats"></dl>
            <div class="mb-row-label mb-modes-label" hidden>Memory build</div>
            <div class="mb-modes" role="tablist" aria-label="Memory build"></div>
            <p class="mb-cached" hidden>Already on this device — opens in seconds</p>
            <p class="mb-ram"></p>
          </div>
          <a class="mb-cta btn btn-primary">Enter chat ▸</a>
        </div>
      </div>
      <button class="mb-arrow" data-dir="1" aria-label="Next model">&rsaquo;</button>
    </div>
    <div class="mb-dots mb-roster" role="tablist" aria-label="Models"></div>
    ${'gpu' in navigator ? '' :
      '<p class="note">This browser has no WebGPU — the chat needs Chrome, Edge, or another WebGPU-enabled browser.</p>'}
  `

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
    `<button class="mb-dot" data-i="${i}" role="tab" title="${g.name}" aria-label="${g.name}"><span class="mb-dot-name">${g.name.replace(/-Instruct.*$/, '')}</span></button>`).join('')

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
      const dot = root.querySelector<HTMLElement>(`.mb-dot[data-i="${i}"]`)
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
    el('.mb-params').textContent = g.params
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
    }
    const cached = el<HTMLElement>('.mb-cached')
    cached.hidden = !CACHED.has(v.spec.id)

    mascot?.setSpec(v.spec, CACHED.has(v.spec.id), mode ? poolFracOf(v.spec, mode.slots) : 0)
  }

  function go(nextG: number): void {
    gi = (nextG + GROUPS.length) % GROUPS.length
    vi = 0
    mi = 0
    paint()
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

  paint()
  void probeCached(() => { paint(); void paintRoster() })
  void mountMascot(canvas, GROUPS[gi].variants[vi].spec).then((m) => {
    if (!m) { art.style.display = 'none'; return }
    mascot = m
    m.setSpec(GROUPS[gi].variants[vi].spec, CACHED.has(GROUPS[gi].variants[vi].spec.id))
  })
}

render()
