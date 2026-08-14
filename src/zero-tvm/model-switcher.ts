/**
 * MODEL SWITCHER — the header dropdown that lets a visitor move between the
 * shipped models (five, as of Qwen3.6) without hand-editing `?model=` into
 * the URL.
 *
 * Switching reloads the page. That's deliberate, not a shortcut: the engine
 * owns gigabytes of GPU buffers (2.6 for the 4B models, 15.7 for the MoE),
 * a compiled pipeline set and a paged KV cache
 * sized from the spec's constants, and there is no teardown path that frees
 * them. A reload is the one operation guaranteed to release the device.
 *
 * The menu probes OPFS so each entry says whether those weights are already
 * on disk. Without that, picking "Qwen3.5-4B" out of curiosity silently
 * starts a 2.6 GB download — the label turns a surprise into a choice.
 */

import { type ModelSpec } from '../compiler/model-spec.js'
import { modelBranding, SHIPPED_MODELS } from './model-registry.js'
// Dynamic, like every other importer of cache-probe: it reaches the loaders,
// which read GPUBufferUsage at module scope. chat.ts imports THIS module
// statically, so a static import here would drag that chain into the chat
// page's entry and break it on any browser without WebGPU.

const MODELS = SHIPPED_MODELS

/**
 * Reload onto `param`, keeping every other query flag. Bench runs drive this
 * page with shader-variant flags (`?vec4=1`), so a switch must not silently
 * drop them — and rebuilding the URL avoids the trailing `?` that assigning
 * `location.search = ''` leaves behind.
 */
function switchTo(param: string): void {
  const url = new URL(window.location.href)
  if (param) url.searchParams.set('model', param)
  else url.searchParams.delete('model')
  window.location.href = url.toString()
}

export function initModelSwitcher(current: ModelSpec): void {
  const trigger = document.getElementById('model-switch') as HTMLButtonElement | null
  const menu = document.getElementById('model-menu')
  if (!trigger || !menu) return

  const cached = new Map<string, boolean>()

  function render(): void {
    menu!.innerHTML = ''
    for (const { param, spec } of MODELS) {
      const brand = modelBranding(spec)
      const isCurrent = spec.id === current.id
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.role = 'option'
      btn.setAttribute('aria-selected', String(isCurrent))
      // Cached entries advertise it; everything else shows the download cost.
      const meta = cached.get(spec.id)
        ? '<span class="mm-meta mm-cached">cached</span>'
        : `<span class="mm-meta">${brand.sizeLabel} download</span>`
      btn.innerHTML = `<span class="mm-name">${brand.name}</span>${meta}`
      if (!isCurrent) {
        btn.addEventListener('click', () => switchTo(param))
      }
      menu!.appendChild(btn)
    }
  }

  function close(): void {
    menu!.hidden = true
    trigger!.setAttribute('aria-expanded', 'false')
  }

  trigger.addEventListener('click', (e) => {
    e.stopPropagation()
    const open = menu.hidden
    menu.hidden = !open
    trigger.setAttribute('aria-expanded', String(open))
  })
  document.addEventListener('click', close)
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') close()
  })
  menu.addEventListener('click', (e) => e.stopPropagation())

  render()
  renderGateChips(current, cached)
  // Re-render once the OPFS probes land, so labels flip to "cached" without
  // blocking the header on three filesystem round-trips.
  void import('./cache-probe.js').then(async ({ isModelCached }) => {
    await Promise.all(
      MODELS.map(async ({ spec }) => cached.set(spec.id, await isModelCached(spec))),
    )
    render()
    renderGateChips(current, cached)
  }).catch(() => { /* no WebGPU — labels stay as download sizes */ })
}

/**
 * The download gate is a *modal* dialog, so the header switcher is inert while
 * it's up — and the gate is exactly where a first-time visitor commits 2+ GB.
 * Without a picker here, everyone who lands on /zero-tvm.html with no `?model=`
 * downloads Phi-3 whether or not that's the one they wanted.
 */
function renderGateChips(current: ModelSpec, cached: Map<string, boolean>): void {
  const host = document.getElementById('dialog-models')
  if (!host) return
  host.innerHTML = ''
  for (const { param, spec } of MODELS) {
    const brand = modelBranding(spec)
    const isCurrent = spec.id === current.id
    const chip = document.createElement('button')
    chip.type = 'button'
    chip.className = 'dm-chip'
    chip.setAttribute('aria-pressed', String(isCurrent))
    const sub = cached.get(spec.id) ? 'cached' : brand.sizeLabel
    chip.innerHTML = `<span class="n">${brand.name}</span><span class="s">${sub}</span>`
    if (!isCurrent) {
      chip.addEventListener('click', () => switchTo(param))
    }
    host.appendChild(chip)
  }
}
