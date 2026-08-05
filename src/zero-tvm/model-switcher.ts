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

import { PHI3, QWEN3_4B, QWEN35_4B, QWEN36_35B_A3B, QWEN36_35B_A3B_Q3, type ModelSpec } from '../compiler/model-spec.js'
import { modelBranding } from './model-select.js'
import { opfsDirFor } from './weight-loader.js'

/** URL `?model=` value for each spec; Phi-3 is the no-flag default. */
const MODELS: { param: string; spec: ModelSpec }[] = [
  { param: '', spec: PHI3 },
  { param: 'qwen3', spec: QWEN3_4B },
  { param: 'qwen35', spec: QWEN35_4B },
  // The 35B MoE ships twice: 3-bit experts first (the one most machines can
  // actually run — ~20 GB free RAM), full 4-bit for the boxes that can.
  { param: 'qwen36q3', spec: QWEN36_35B_A3B_Q3 },
  { param: 'qwen36', spec: QWEN36_35B_A3B },
]

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

/** True if this spec's weights are already in OPFS (same sentinel as chat.ts). */
async function isCached(spec: ModelSpec): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.storage?.getDirectory) return false
  try {
    const root = await navigator.storage.getDirectory()
    const dir = await root.getDirectoryHandle(opfsDirFor(spec))
    // The SPEC's manifest, not a hardcoded name — Qwen3.5 renamed it and the
    // MLX checkpoints use the safetensors index (same fix as chat.ts).
    await dir.getFileHandle(spec.manifestName ?? 'ndarray-cache.json')
    return true
  } catch {
    return false
  }
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
  void Promise.all(
    MODELS.map(async ({ spec }) => cached.set(spec.id, await isCached(spec))),
  ).then(() => {
    render()
    renderGateChips(current, cached)
  })
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
