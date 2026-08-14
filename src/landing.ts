/**
 * LANDING CARDS — renders the model picker on index.html from the engine's own
 * model table (model-registry.ts), so the landing page cannot disagree with
 * what actually ships. The old page hand-wrote this list into the HTML; after
 * Qwen3.6 landed, twelve separate claims on it were stale.
 *
 * Progressive enhancement, strictly:
 *   - cards render from the LIGHT registry (no WebGPU touched), so browsers
 *     without WebGPU still see the full picker plus a "needs WebGPU" note;
 *   - the "on this device ✓" badge comes from an OPFS probe behind a dynamic
 *     import of weight-loader — that module reads GPUBufferUsage at module
 *     scope and would throw here at import time, so the import failure is the
 *     feature-detect.
 */

import { SHIPPED_MODELS, modelBranding } from './zero-tvm/model-registry.js'

function chatUrl(param: string): string {
  return param ? `zero-tvm.html?model=${param}` : 'zero-tvm.html'
}

function render(): void {
  const host = document.getElementById('model-cards')
  if (!host) return

  const webgpu = 'gpu' in navigator
  host.innerHTML = ''
  for (const { param, spec } of SHIPPED_MODELS) {
    const b = modelBranding(spec)
    const a = document.createElement('a')
    a.className = 'card'
    a.href = chatUrl(param)
    // rateLabel is empty until a measurement exists, and an empty chip is
    // simply not rendered. That is the landing's rule: a number appears here
    // because it was measured, or it does not appear.
    a.innerHTML = `
      <div class="card-name">${b.name}</div>
      <div class="card-params">${b.params}</div>
      <div class="chips">
        <span class="chip">${b.sizeLabel}</span>
        ${b.rateLabel ? `<span class="chip rate">${b.rateLabel}</span>` : ''}
      </div>
      ${b.ramNote ? `<div class="card-ram">${b.ramNote}</div>` : ''}
      <div class="card-go">Open chat →<span data-model-cached="${spec.id}"></span></div>
    `
    host.appendChild(a)
  }

  if (!webgpu) {
    const note = document.createElement('p')
    note.className = 'note'
    note.textContent = 'This browser has no WebGPU — the chat needs Chrome, Edge, or another WebGPU-enabled browser.'
    host.appendChild(note)
  }
}

/** Fill the "on this device ✓" badges. Failure is silence, never a break. */
async function markCached(): Promise<void> {
  if (!('gpu' in navigator) || !navigator.storage?.getDirectory) return
  try {
    const { opfsDirFor } = await import('./zero-tvm/weight-loader.js')
    const root = await navigator.storage.getDirectory()
    for (const { spec } of SHIPPED_MODELS) {
      try {
        const dir = await root.getDirectoryHandle(opfsDirFor(spec))
        await dir.getFileHandle(spec.manifestName ?? 'ndarray-cache.json')
        const el = document.querySelector(`[data-model-cached="${spec.id}"]`)
        if (el) el.textContent = ' · on this device ✓'
      } catch { /* not cached — leave blank */ }
    }
  } catch { /* no WebGPU / import failed — badges stay blank */ }
}

render()
void markCached()
