/**
 * SAMPLING CONTROLS — the header popover for temperature, top-p, min-p and seed.
 *
 * The sampler shipped reachable only as `?temp=0.8&topp=0.95`, which is a
 * developer idiom, not a feature: nobody hand-edits a URL to make a model less
 * repetitive. These are the same four knobs, in the header.
 *
 * NO RELOAD, unlike the model switcher next to it. Switching models has to
 * reload because the engine owns gigabytes of GPU buffers with no teardown
 * path; sampling is four floats in a uniform, so `engine.setSampling()` writes
 * them and the next token uses them. A control that reloaded — and so
 * re-downloaded the weights — every time a slider moved would be worse than no
 * control at all.
 *
 * Greedy stays GREEDY. temperature 0 does not mean "the sampler, with a
 * degenerate temperature": the engine returns to dispatching argmax.wgsl, which
 * is what every reference comparison in this repo pins. The panel says so,
 * because "0" reading as "off" rather than "divide by zero" is not obvious.
 *
 * State lives in the URL as well as the engine, so a session stays shareable
 * and reproducible — the draw is a pure function of (seed, position), so the
 * same link really does replay the same text.
 */

import type { DecodeEngine } from './engine-core.js'

type Sampling = NonNullable<Parameters<DecodeEngine['setSampling']>[0]>

const $ = (id: string) => document.getElementById(id) as HTMLElement

/** Reads the same flags chat.ts booted with, so opening the panel shows the
 *  state the URL asked for rather than defaults. */
function fromUrl(): Sampling {
  const q = new URLSearchParams(location.search)
  const num = (k: string, d: number) => {
    const v = Number(q.get(k))
    return Number.isFinite(v) && q.get(k) !== null ? v : d
  }
  return {
    temperature: num('temp', 0),
    topP: num('topp', 1),
    minP: num('minp', 0),
    seed: num('seed', 0),
  }
}

/** Mirror into the URL without reloading. history.replaceState keeps the link
 *  copyable and the back button unpolluted — a slider is not a navigation. */
function toUrl(s: Sampling): void {
  const url = new URL(location.href)
  const set = (k: string, v: number, dflt: number) => {
    if (v === dflt) url.searchParams.delete(k)
    else url.searchParams.set(k, String(v))
  }
  set('temp', s.temperature, 0)
  set('topp', s.topP ?? 1, 1)
  set('minp', s.minP ?? 0, 0)
  set('seed', s.seed ?? 0, 0)
  history.replaceState(null, '', url.toString())
}

const FIELDS = [
  {
    key: 'temperature' as const, id: 'samp-temp', label: 'Temperature',
    min: 0, max: 2, step: 0.05, dflt: 0,
    note: '0 = greedy (argmax). Higher is more varied.',
  },
  {
    key: 'topP' as const, id: 'samp-topp', label: 'Top-p',
    min: 0.05, max: 1, step: 0.01, dflt: 1,
    note: 'Keep the smallest set of tokens covering this much probability. 1 = off.',
  },
  {
    key: 'minP' as const, id: 'samp-minp', label: 'Min-p',
    min: 0, max: 0.5, step: 0.005, dflt: 0,
    note: 'Drop tokens below this fraction of the top token. 0 = off.',
  },
]

/**
 * Mounts the popover. `getEngine` rather than an engine because this runs
 * before bootEngine — there is no engine yet at mount time, and the panel must
 * pick one up later without being re-mounted.
 *
 * The button is NOT reachable during the download: #start-dialog is a modal
 * <dialog>, so everything outside it is inert (measured — elementFromPoint over
 * the button returns the dialog). That is fine, since a URL that already
 * carries ?temp= boots the engine with it; the panel only has to serve changes
 * made after the gate closes.
 */
export function initSamplingControls(getEngine: () => DecodeEngine | null): void {
  const btn = $('sampling-btn') as HTMLButtonElement
  const menu = $('sampling-menu')
  if (!btn || !menu) return

  let state = fromUrl()

  const rows = FIELDS.map((f) => `
    <label class="samp-row" for="${f.id}">
      <span class="samp-label">${f.label}<b id="${f.id}-val"></b></span>
      <input id="${f.id}" type="range" min="${f.min}" max="${f.max}" step="${f.step}">
      <span class="samp-note">${f.note}</span>
    </label>`).join('')
  menu.innerHTML = `
    <div class="samp-head"><b>Sampling</b><span id="samp-mode"></span></div>
    ${rows}
    <label class="samp-row" for="samp-seed">
      <span class="samp-label">Seed<b id="samp-seed-val"></b></span>
      <div class="samp-seed-row">
        <input id="samp-seed" type="number" min="0" step="1">
        <button type="button" id="samp-reseed" class="samp-btn">New</button>
      </div>
      <span class="samp-note">The same seed replays the same text, token for token.</span>
    </label>
    <button type="button" id="samp-reset" class="samp-btn samp-reset">Back to greedy</button>`

  const render = (): void => {
    for (const f of FIELDS) {
      const v = (state[f.key] ?? f.dflt) as number
      ;($(f.id) as HTMLInputElement).value = String(v)
      $(`${f.id}-val`).textContent = f.key === 'temperature' && v === 0 ? 'off' : String(v)
    }
    ;($('samp-seed') as HTMLInputElement).value = String(state.seed ?? 0)
    const greedy = state.temperature <= 0
    $('samp-mode').textContent = greedy ? 'greedy · argmax' : 'sampling'
    // top-p and min-p do nothing at temperature 0; saying so beats letting
    // someone move them and conclude the feature is broken.
    menu.classList.toggle('samp-greedy', greedy)
    btn.classList.toggle('active', !greedy)
    btn.title = greedy ? 'Sampling: off (greedy)' : `Sampling: temp ${state.temperature}`
  }

  const apply = (): void => {
    toUrl(state)
    render()
    getEngine()?.setSampling(state.temperature > 0 ? state : null)
  }

  for (const f of FIELDS) {
    $(f.id).addEventListener('input', (e) => {
      state = { ...state, [f.key]: Number((e.target as HTMLInputElement).value) }
      apply()
    })
  }
  $('samp-seed').addEventListener('input', (e) => {
    state = { ...state, seed: Math.max(0, Math.floor(Number((e.target as HTMLInputElement).value) || 0)) }
    apply()
  })
  $('samp-reseed').addEventListener('click', () => {
    state = { ...state, seed: Math.floor(Math.random() * 2 ** 31) }
    apply()
  })
  $('samp-reset').addEventListener('click', () => {
    state = { temperature: 0, topP: 1, minP: 0, seed: state.seed ?? 0 }
    apply()
  })

  const close = (): void => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false') }
  btn.addEventListener('click', (e) => {
    e.stopPropagation()
    menu.hidden = !menu.hidden
    btn.setAttribute('aria-expanded', String(!menu.hidden))
  })
  menu.addEventListener('click', (e) => e.stopPropagation())
  document.addEventListener('click', close)
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close() })

  render()
  // The URL may already carry ?temp=; chat.ts booted the engine with it, so
  // this only has to agree, not re-apply. Verified end to end by driving these
  // controls in a real browser: same seed reproduces the reply exactly, a
  // different seed diverges, and both differ from greedy.
}
