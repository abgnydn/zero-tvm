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
import { mountMascot, type MascotHandle } from './mascot.js'

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

function render(): void {
  const host: HTMLElement | null = document.getElementById('model-browser')
  if (host === null) return
  const root: HTMLElement = host

  host.innerHTML = `
    <div class="mb-stage">
      <button class="mb-arrow" data-dir="-1" aria-label="Previous model">&lsaquo;</button>
      <div class="mb-slide">
        <div class="mb-art"><canvas class="mb-mascot" aria-hidden="true"></canvas></div>
        <div class="mb-info">
          <div class="mb-name"></div>
          <div class="mb-params"></div>
          <div class="mb-variants" role="tablist" aria-label="Quantisation"></div>
          <dl class="mb-stats"></dl>
          <p class="mb-ram"></p>
          <a class="mb-cta btn btn-primary">Open chat</a>
        </div>
      </div>
      <button class="mb-arrow" data-dir="1" aria-label="Next model">&rsaquo;</button>
    </div>
    <div class="mb-dots" role="tablist" aria-label="Models"></div>
    ${'gpu' in navigator ? '' :
      '<p class="note">This browser has no WebGPU — the chat needs Chrome, Edge, or another WebGPU-enabled browser.</p>'}
  `

  const el = <T extends Element>(s: string): T => host.querySelector<T>(s)!
  const canvas = el<HTMLCanvasElement>('.mb-mascot')
  const dots = el<HTMLElement>('.mb-dots')
  let gi = 0
  let vi = 0
  let mascot: MascotHandle | null = null

  dots.innerHTML = GROUPS.map((g, i) =>
    `<button class="mb-dot" data-i="${i}" role="tab" title="${g.name}" aria-label="${g.name}"></button>`).join('')

  function paint(): void {
    const g = GROUPS[gi]
    const v = g.variants[vi]
    const b = modelBranding(v.spec)

    el('.mb-name').textContent = g.name
    el('.mb-params').textContent = g.params
    el<HTMLAnchorElement>('.mb-cta').href = chatUrl(v.param)

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
    ram.textContent = b.ramNote ?? ''
    ram.hidden = !b.ramNote

    for (const d of root.querySelectorAll<HTMLElement>('.mb-dot')) {
      d.setAttribute('aria-selected', String(Number(d.dataset.i) === gi))
    }
    mascot?.setSpec(v.spec)
  }

  function go(nextG: number): void {
    gi = (nextG + GROUPS.length) % GROUPS.length
    vi = 0
    paint()
  }

  host.addEventListener('click', (e) => {
    const t = e.target as HTMLElement
    const arrow = t.closest<HTMLElement>('.mb-arrow')
    if (arrow) { go(gi + Number(arrow.dataset.dir)); return }
    const dot = t.closest<HTMLElement>('.mb-dot')
    if (dot) { go(Number(dot.dataset.i)); return }
    const variant = t.closest<HTMLElement>('.mb-variant')
    if (variant) { vi = Number(variant.dataset.v); paint() }
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
  void mountMascot(canvas, GROUPS[gi].variants[vi].spec).then((m) => {
    if (!m) { art.style.display = 'none'; return }
    mascot = m
    m.setSpec(GROUPS[gi].variants[vi].spec)
  })
}

render()
