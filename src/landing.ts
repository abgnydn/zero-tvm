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

import { SHIPPED_MODELS, canSplitAcrossDevices, kvBytesPerTokenShown as kvBytesPerToken, modelBranding, quantLabel, specForParam, specWithCtx } from './zero-tvm/model-registry.js'
// `?ctx=` is read HERE by the one reader every other surface uses. The
// entrance used to run its own `Number(Q.get('ctx'))`, which is how it came to
// disagree with share.html and zero-tvm.html about the same link — see the
// note on the ctx picker below.
import { ctxFrom } from './zero-tvm/room-url.js'
import type { ModelSpec } from './compiler/model-spec.js'
import { mountMascot, mascotPalette, type MascotHandle } from './mascot.js'
import { LANE_SIGIL, laneOf, loreOf, abilitiesOf } from './landing-lore.js'
import { FEATS, feats } from './feats.js'
import type { SwarmHandle } from './landing-swarm.js'

interface Variant { param: string; spec: ModelSpec; label: string }
interface Group { name: string; params: string; variants: Variant[] }

// (the CTA URL is composed in paint() — model + pool + ctx, one query string)

/** Context ceiling, derived from the spec (maxPages x pageSize) — a fact about
 *  the build rather than a figure typed into the page. */
const ctxLabel = (t: number): string => (t >= 1024 ? `${Math.round(t / 1024)}k` : String(t))

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
 *  (Phi-3) — then the row does not render.
 *
 *  `linked` is a window a LINK named that this list does not enumerate.
 *  Without it the entrance honoured `?ctx=` only on an exact match to one of
 *  the three, and every other value fell back to the spec default in silence —
 *  while `landing-room.ts` writes room chips carrying any ctx in
 *  [256, maxContext] and hands the SAME `?ctx=` to the other machines. Two
 *  tabs then run different KV budgets for one conversation and neither prints
 *  the number: the failure ctxFrom's own docstring exists to prevent, arriving
 *  through the one surface that was not reading ctxFrom.
 *
 *  The value is passed through `specWithCtx` first, so the chip quotes the
 *  window that is actually ALLOCATED — KV pages round up, so `?ctx=5000` on a
 *  16-token page is a 5008-token cache — and the CTA writes that number back
 *  into the link. Appended last, never sorted in: `go()` resets the picker to
 *  index 0 and index 0 has to stay the compiled default. */
function ctxModesOf(spec: ModelSpec, linked?: number | null): CtxMode[] {
  const out: CtxMode[] = [{ name: 'Standard', tokens: spec.maxContext }]
  const long = Math.min(spec.maxContext * 4, spec.maxSeq)
  if (long > spec.maxContext) out.push({ name: 'Long', tokens: long })
  if (spec.maxSeq > long) out.push({ name: 'Full', tokens: spec.maxSeq })
  if (linked != null) {
    const t = specWithCtx(spec, linked).maxContext
    if (!out.some((c) => c.tokens === t)) out.push({ name: 'From the link', tokens: t })
  }
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

/** The roster the entrance renders. Exported so a headless test can resolve a
 *  slot the URL grammar returns to the spec it will actually boot. */
export const GROUPS = buildGroups()

// ─────────────────────────────────────────────────────────────────────────
// THE ENTRANCE URL GRAMMAR
// ─────────────────────────────────────────────────────────────────────────

/** The stage of a split this tab holds. */
export interface EntranceSplit {
  /** [0, ...cuts, spec.layers] — the WHOLE split, so the room strip can write
   *  the other machines' links; this machine's slice alone cannot say where
   *  the others begin. */
  bounds: number[]
  /** Which stage this tab is. */
  index: number
}

/** Everything a link may ask this page to do. */
export interface EntranceIntent {
  /** Roster group and variant the URL names. */
  gi: number
  vi: number
  /** The split this tab holds, or null for no split (boot the whole model). */
  split: EntranceSplit | null
  /** The URL asked to enter the chat. There is deliberately no verb here
   *  meaning "enter now": a URL is not a click, so the only thing a link can
   *  ask for is to be ASKED — see openUrlGate() in render(). */
  enter: { room: boolean; act: 'consent' } | null
  /** The URL asked for swarm mode. */
  swarm: boolean
}

/**
 * The roster slot the URL names. TWO QUESTIONS, deliberately answered
 * differently — do not fold them back into one.
 *
 * NO `model` KEY AT ALL is not a model request. It is "which character does
 * the select screen OPEN on", and that is a presentation choice this page
 * owns alone: the standalone pages have no roster, so `specForParam` has no
 * opinion about it. The answer is the roster's first card — the strongest
 * model this project runs. Phi-3 used to lead the roster and the whole
 * reorder (b37e849) exists because the weakest shipped model was the
 * project's first impression.
 *
 * A `model` KEY THAT IS PRESENT but names nothing is a COMPATIBILITY
 * question — "what does an unrecognised ?model= mean" — and there the two
 * surfaces must not disagree. `specForParam` is where `?model=` is resolved,
 * and /zero-tvm.html, share.html and validate.html have always asked it.
 * The entrance ran its own findIndex over the roster instead and fell back to
 * slot 0 on a miss, which was harmless while slot 0 was Phi-3 and became a
 * 14.1 GB liability the day the roster was reordered:
 * `/zero-tvm.html?model=not-a-model` booted Phi-3 while `/?model=not-a-model`
 * booted Qwen3.8-27B. Every pre-registry URL depends on that fall-through.
 *
 * So the split is on PRESENCE (`q.get('model') !== null`), not on whether the
 * value resolves. `?model=bogus`, `?model=` and `?model=embed` are all Phi-3;
 * a bare `/` is the flagship.
 *
 * A spec the ROSTER does not carry — the embedding model, which answers
 * nothing a visitor typed, or a build that is generated but not yet
 * numerics-validated — takes the registry's own fallback rather than slot 0.
 * The entrance cannot put a character on stage that it deliberately does not
 * ship, and slot 0 is whichever model leads the roster this month.
 */
function rosterSlotFor(param: string | null): { gi: number; vi: number } {
  // Presentation: no ?model= key, so nothing was asked for. Open on the
  // roster's first card.
  if (param === null) return { gi: 0, vi: 0 }
  const slotOf = (id: string): { gi: number; vi: number } | null => {
    for (let g = 0; g < GROUPS.length; g++) {
      const v = GROUPS[g].variants.findIndex((x) => x.spec.id === id)
      if (v >= 0) return { gi: g, vi: v }
    }
    return null
  }
  // Compatibility: the registry's answer, whatever it is.
  return slotOf(specForParam(param).id) ?? slotOf(specForParam(null).id) ?? { gi: 0, vi: 0 }
}

/**
 * `?split=0,18,36&stage=0` — the stage of a split this tab holds, or null.
 *
 * A MALFORMED SPLIT IS NO SPLIT: the entrance boots the whole model rather
 * than a broken stage. The old guard checked only that there were three
 * boundaries, that the index was in range, and that the last boundary equalled
 * the layer count, and every rule below is one that was measured breaking
 * something downstream of that:
 *
 *  - Boundaries must be integers, START AT 0, ascend strictly and end at
 *    `spec.layers`. `split=4,8,16&stage=0` on a 16-layer checkpoint passed and
 *    served layers 4-8 from a HOST stage — a stage with no embedding, which
 *    share.ts refuses outright ("a hosting stage must start at layer 0").
 *    `split=0,999,16` passed too and died inside the weight loader on
 *    `Cannot read properties of undefined (reading 'normGamma1')`, before
 *    engine-core's own `layerRange ... is not a range` guard could fire.
 *  - The stage index must be an INTEGER in range. `stage=0.5` passed the range
 *    check, `bounds[0.5]` is undefined, and the room plan then read
 *    "This tab holds layers undefined-undefined".
 *  - The stage must be the one that STARTS the model. Besides share.ts's own
 *    rule, `swarmUrls` writes machine 1's row with `room: null` on purpose —
 *    it assumes this tab is machine 1 — so from a later stage the room strip
 *    hands out a serving link with no room fragment, which `roleFor` routes to
 *    the host branch: a brand new room, the silent failure room-url.ts exists
 *    to prevent.
 *  - The checkpoint must be one the loader can CUT. `canSplitAcrossDevices` is
 *    the registry's own predicate and it exists so the swarm link builder
 *    cannot hand out a `?layers=` URL that throws at boot; this reader was a
 *    second builder that could. `?model=&split=0,16,32` sent Phi-3 into
 *    serving mode to die on "loadWeights: layerRange needs an MLX checkpoint;
 *    phi3-mini ships MLC shards".
 */
function splitFor(spec: ModelSpec, splitParam: string | null, stageParam: string | null): EntranceSplit | null {
  if (splitParam === null || splitParam === '') return null
  if (!canSplitAcrossDevices(spec)) return null
  const bounds = splitParam.split(',').map(Number)
  if (bounds.length < 3 || !bounds.every((n) => Number.isInteger(n))) return null
  if (bounds[0] !== 0 || bounds[bounds.length - 1] !== spec.layers) return null
  if (!bounds.every((n, i) => i === 0 || n > bounds[i - 1])) return null
  const index = stageParam === null ? 0 : Number(stageParam)
  if (!Number.isInteger(index) || index < 0 || index >= bounds.length - 1) return null
  // A later stage joins a room as a HELPER, through share.html — never here.
  if (bounds[index] !== 0) return null
  return { bounds, index }
}

/**
 * The URL a reload should land on once a link's request has been honoured:
 * the same page and the same character, minus the keys that mean "act without
 * a click".
 *
 * "⟨ Roster" and the /roster command both `location.reload()`, and a reload
 * keeps the query — so on any `?chat=1` URL the way OUT of the conversation
 * walked straight back into it. Every link the room plan writes carries
 * `chat=1`, so that was the way out of every split. Cleaning the address bar
 * at the moment consent is given fixes it here, where the link is read,
 * without landing-chat.ts having to know the entrance's grammar.
 */
export function urlAfterEnter(search: string): string {
  const q = new URLSearchParams(search)
  q.delete('chat')
  q.delete('room')
  const rest = q.toString()
  return rest ? `?${rest}` : ''
}

/** What this URL asks the entrance to do. Pure, so the headless suite can
 *  hold the whole grammar. */
export function entranceIntent(search: string, hash: string): EntranceIntent {
  const q = new URLSearchParams(search)
  const model = q.get('model')
  const { gi, vi } = rosterSlotFor(model)
  const v = GROUPS[gi].variants[vi]
  // The split applies only to the character the URL actually NAMES, so walking
  // the roster afterwards cannot carry a stale set of bounds onto a model with
  // a different layer count.
  const split = model !== null && model === v.param
    ? splitFor(v.spec, q.get('split'), q.get('stage'))
    : null
  return {
    gi,
    vi,
    split,
    // A split has to open the room strip: the other stages need links.
    enter: q.get('chat') === '1'
      ? { room: q.get('room') === '1' || split !== null, act: 'consent' }
      : null,
    // README publishes https://zerotvm.com/#swarm. Only a CLICK on
    // `a[href="#swarm"]` was ever wired, and render() hides the #swarm
    // fallback section — so the published link opened nothing at all.
    swarm: hash === '#swarm',
  }
}

// ─────────────────────────────────────────────────────────────────────────
// WHAT WAS CHOSEN, AND WHAT THAT COSTS — pure, so the suite can hold it
// ─────────────────────────────────────────────────────────────────────────

/** A choice on the select screen. Four indices, and only together do they
 *  mean anything: character, quantisation, memory build, context build. */
export interface Selection { gi: number; vi: number; mi: number; xi: number }

/**
 * Everything that follows from a Selection: what boots, and what the page is
 * allowed to SAY about it.
 *
 * ONE resolver, and the boot takes its answer BY VALUE. That is the whole
 * repair for a gate that read "This link asks to run Llama-3.2-1B … ~528 MB
 * KV", then booted Qwen3.6-35B-A3B — 16.4 GB, ~20 GB of free RAM — because
 * the roster had moved behind it and the boot re-read the live selection at
 * click time. The sentence and the boot are two views of one BootPlan now, so
 * there is no live state left for them to disagree about.
 */
export interface BootPlan {
  spec: ModelSpec
  /** `?model=` value. `''` is the DEFAULT model, not "unset". */
  param: string
  name: string
  sizeLabel: string
  ramNote: string
  /** Expert slots for a pooled MoE build; 0 is the full model. */
  poolSlots: number
  poolLabel: string
  /** The window this plan boots at, in tokens — post-clamp, the number that
   *  is actually allocated. */
  ctxTokens: number
  /** `?model=&pool=&ctx=` for this plan: the CTA's href, the room verb's
   *  href, and the page a failed in-place mount falls back to. */
  query: string
}

/** The plan a Selection resolves to. `linkedCtx` is the window the URL named
 *  (see ctxModesOf) or null. */
export function bootPlanFor(sel: Selection, linkedCtx: number | null): BootPlan {
  const v = GROUPS[sel.gi].variants[sel.vi]
  const b = modelBranding(v.spec)
  const modes = b.poolModes ?? []
  const mode = modes[sel.mi] ?? modes[0]
  const ctxs = ctxModesOf(v.spec, linkedCtx)
  const cx = ctxs[sel.xi] ?? ctxs[0]
  const qs: string[] = []
  if (v.param) qs.push(`model=${v.param}`)
  if (mode && mode.slots) qs.push(`pool=${mode.slots}`)
  if (cx.tokens !== v.spec.maxContext) qs.push(`ctx=${cx.tokens}`)
  return {
    spec: v.spec,
    param: v.param,
    name: b.name,
    sizeLabel: b.sizeLabel,
    ramNote: b.ramNote ?? '',
    poolSlots: mode?.slots ?? 0,
    poolLabel: mode && mode.slots ? mode.label : '',
    ctxTokens: cx.tokens,
    query: qs.length ? `?${qs.join('&')}` : '',
  }
}

/** Every word the consent gate says, derived from the plan it will boot and
 *  from nothing else. Same voice as confirmDownload() in share.ts. */
export function gateCopy(plan: BootPlan, o: {
  room: boolean
  cached: boolean
  /** The layers this device would fetch, when the link named a stage — then
   *  the download is a SLICE, and quoting the whole checkpoint's size for it
   *  is the figure that once promised a phone 14.1 GB for a fraction of it. */
  stage: { start: number; end: number } | null
  int8: boolean
}): { title: string; what: string; cost: string; go: string } {
  const s = o.stage
  const weights = o.cached
    ? (s ? `Layers ${s.start}–${s.end} are already cached on this device.`
      : 'The weights are already cached on this device.')
    : (s ? `Layers ${s.start}–${s.end} of ${plan.spec.layers} download once — a slice of `
        + `the full ${plan.sizeLabel}, not all of it — and are cached locally.`
      : `The weights download once (${plan.sizeLabel}) and are cached locally; every later `
        + 'visit starts from disk.')
  return {
    title: `Run ${plan.name} on this machine?`,
    what: `This link asks to run ${plan.name} on this machine. ${weights} Nothing has downloaded yet.`,
    // The second half of the price: the KV cache is allocated EAGERLY at boot,
    // and ?ctx= moves it. ramNote is a whole-checkpoint figure and is false
    // for a stage, so a stage does not carry it.
    cost: [
      `${ctxLabel(plan.ctxTokens)} context · ~${kvPrice(plan.spec, plan.ctxTokens, o.int8)} allocated at boot`,
      s ? '' : plan.ramNote,
    ].filter(Boolean).join(' — '),
    go: o.room
      ? (o.cached ? 'Enter & open a room →' : 'Download & open a room →')
      : (o.cached ? 'Enter chat →' : 'Download & enter →'),
  }
}

/** What a keypress on the scene means. */
export type SceneKey = 'ignore' | 'prev' | 'next' | 'enter' | 'exit-swarm'

/**
 * THE SCENE'S KEYBOARD, as a decision rather than as a listener body.
 *
 * `display:none` HIDES PIXELS; IT IS NOT A CONTROL. The consent gate moved
 * the verbs out of view and left this handler alone, and this handler
 * synthesises a click on `.mb-cta` for Enter. `#model-browser` has
 * `tabIndex = 0`, so any click on the scene focuses it — including the click
 * that dismisses the splash — and one Enter afterwards started a 14.1 GB
 * download with the gate still on screen and `?chat=1` still in the address
 * bar (so the `⟨ Roster` escape the last round fixed came back on that path
 * too). Arrows were worse: they are what a visitor presses to scroll a
 * full-screen scene, and they walked the roster behind a dialog that had
 * already been read.
 *
 * `gated` is checked FIRST and refuses everything. The gate is a real
 * `<dialog>` opened with `showModal()`, so the document behind it is inert
 * and no pointer can reach the roster either — but the dialog is a DESCENDANT
 * of `#model-browser`, so keystrokes aimed at its own buttons still bubble
 * through this listener. Escape belongs to the dialog's `cancel` event, which
 * is the decline path.
 */
export function keyIntent(key: string, targetTag: string, s: {
  chatting: boolean; gated: boolean; swarm: boolean
}): SceneKey {
  if (s.gated) return 'ignore'
  // In chat mode the keyboard belongs to the composer — arrows must not
  // switch characters under a conversation. Nor under a text field: the swarm
  // sheet has one, and typing a room link into it moved the roster.
  if (s.chatting) return 'ignore'
  if (targetTag === 'INPUT' || targetTag === 'TEXTAREA') return 'ignore'
  if (key === 'Escape') return s.swarm ? 'exit-swarm' : 'ignore'
  if (key === 'ArrowRight' || key === 'ArrowDown') return 'next'
  if (key === 'ArrowLeft' || key === 'ArrowUp') return 'prev'
  if (key === 'Enter' && targetTag !== 'BUTTON' && targetTag !== 'A') return 'enter'
  return 'ignore'
}

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
    <!-- The reach caveat, in the realm's corner-line register rather than as a
         paragraph of prose: two mono lines that only show in swarm mode. NOT
         aria-hidden like the decorative corners — this is the one thing a
         reader has to know before they spend an evening on it. The full prose
         stays in the #swarm fallback for a browser with no JavaScript. -->
    <div class="sw-reach-note" role="note">
      <span>Reach · STUN only, no TURN — same network or an ordinary home router; corporate and hotel usually will not</span>
      <span>Splitting needs an MLX checkpoint · every serving tab has to stay awake</span>
    </div>
    <div class="cs-roster-head" aria-hidden="true">Roster</div>
    <div class="mb-dots mb-roster" role="tablist" aria-label="Models"></div>
    <div class="cs-enter">
      <!-- TWO verbs, equal weight. Serving a room works on EVERY model — only
           splitting one across machines needs an MLX checkpoint, and 8 of the
           11 shipped specs are MLX. Hiding the room behind a line of grey text
           under the button, below the fold, buried the most shareable thing
           the project does; the split doorway on the RAM line made it worse by
           framing the swarm as a consolation for not having enough memory. -->
      <div class="cs-verbs">
        <a class="mb-cta btn btn-primary">Enter chat ▸</a>
        <a class="mb-cta-room btn btn-room">⟁ Open a room</a>
      </div>
      <p class="cs-verbs-note">A room serves this model to other machines. <span class="cs-verbs-split"></span></p>
      <!-- A LINK IS NOT CONSENT. ?chat=1 used to click ENTER for you. This is
           the same question share.html's confirmDownload() asks, in the same
           voice, and it is asked EVEN WHEN THE WEIGHTS ARE CACHED — being
           cached changes the wording, not whether anybody agreed.

           A REAL <dialog>, opened with showModal(). The first version was a
           <div> that set display:none on the verbs, which hid pixels and
           controlled nothing: the roster stayed clickable, the arrow keys
           still walked it, Enter still synthesised a click on the hidden CTA,
           and a screen reader was told none of it. showModal() makes the rest
           of the document inert — roster, sheet and site nav — traps focus,
           and gives Escape a meaning, which is what the three findings under
           it were all consequences of. Styled inline because the entrance's
           stylesheet is not this file's to edit; the palette is the same
           token set the rest of the scene reads. -->
      <dialog class="cs-room-consent cs-url-gate" role="dialog" aria-modal="true"
              aria-labelledby="cs-gate-title" aria-describedby="cs-gate-what cs-gate-cost"
              style="max-width:52ch;text-align:center;padding:20px 22px;border-radius:6px;
                     color:var(--text);background:var(--surface-2);
                     border:1px solid color-mix(in srgb, var(--cs-accent, var(--accent)) 34%, var(--border))">
        <h2 id="cs-gate-title" style="margin:0 0 12px;font-family:var(--mono);font-size:0.62rem;
            letter-spacing:0.14em;text-transform:uppercase;color:var(--dim);font-weight:600"></h2>
        <!-- role=status so the probe's refinement ("already cached on this
             device") is ANNOUNCED. The first wording is written before the
             dialog opens, so aria-describedby carries it into the opening
             announcement; without both, a blind visitor was asked to approve
             a download whose size was never read out. -->
        <div role="status">
          <p id="cs-gate-what"></p>
          <p class="cs-plan-line" id="cs-gate-cost"></p>
        </div>
        <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px">
          <button type="button" class="cs-chat-tool" id="cs-gate-go" disabled>Checking this device…</button>
          <button type="button" class="cs-chat-tool" id="cs-gate-no">Not now</button>
        </div>
      </dialog>
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
  // The scene rendered, so the static #swarm section below it has done its
  // job: it is the no-JS fallback and nothing else. Hidden HERE rather than in
  // the markup, so a browser that never reaches this line still gets the prose
  // and the plain share.html link.
  document.getElementById('swarm')?.setAttribute('hidden', '')
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

  /** Whether the engine this page hands off to will quantise its KV cache.
   *  Default ON; `?kv8=0` opts out — the SAME rule variants.ts applies, read
   *  here rather than assumed, so the price shown is the price charged. */
  const INT8_KV = new URLSearchParams(location.search).get('kv8') !== '0'

  const el = <T extends Element>(s: string): T => host.querySelector<T>(s)!
  const canvas = el<HTMLCanvasElement>('.mb-mascot')
  const dots = el<HTMLElement>('.mb-dots')
  /* The entrance reads its own URL. It used to ignore it entirely, so there
     was no way back into a chosen build without walking the roster again —
     which is what made the build controls unreachable once chat had covered
     the sheet. ?model= picks the character and its quantisation, ?ctx= and
     ?pool= the two builds, and ?chat=1 goes straight in. That is exactly the
     shape the chat panel's own build strip writes when you change one. */
  const Q = new URLSearchParams(location.search)
  const wanted = Q.get('model')
  const intent = entranceIntent(location.search, location.hash)
  let gi = intent.gi
  let vi = intent.vi
  /** Selected memory build (index into the spec's poolModes; 0 = full). Reset
   *  on every model/variant change — a build belongs to a character. */
  let mi = 0
  /** Selected context build (index into ctxModesOf; 0 = the compiled default).
   *  Context is a KV-memory dial, not a model property — the number the sheet
   *  shows is whichever build is chosen here, priced in KV bytes. */
  let xi = 0
  /** The window the LINK asked for, through the one reader of `?ctx=` — which
   *  refuses anything that is not an integer of at least 256 tokens, so a
   *  `?ctx=0.5` cannot reach the picker at all. */
  const LINKED_CTX = ctxFrom(location.search)
  /** …and it applies only while the character the link NAMED is the one on
   *  stage, the same rule the split follows: walking the roster must not
   *  carry a stale window onto a model with a different trained length. */
  const linkedFor = (param: string): number | null => (Q.get('model') === param ? LINKED_CTX : null)
  /** The context builds offered for a slot, link included. */
  const ctxsFor = (g: number, v2: number): CtxMode[] =>
    ctxModesOf(GROUPS[g].variants[v2].spec, linkedFor(GROUPS[g].variants[v2].param))
  /** THE ONE RESOLVER — see BootPlan. Every href, every sentence and the boot
   *  itself come through here. */
  const planFor = (sel: Selection): BootPlan =>
    bootPlanFor(sel, linkedFor(GROUPS[sel.gi].variants[sel.vi].param))
  if (wanted !== null) {
    const spec0 = GROUPS[gi].variants[vi].spec
    const wp = Q.get('pool')
    if (wp !== null) {
      const j = (modelBranding(spec0).poolModes ?? []).findIndex((m) => String(m.slots) === wp)
      if (j >= 0) mi = j
    }
    if (LINKED_CTX !== null) {
      // ctxModesOf appends the link's window when the three enumerated builds
      // do not already carry it, so this findIndex hits for ANY in-range
      // value now — it used to hit only on an exact match to one of three,
      // and `?ctx=5000` silently booted at the spec default.
      const t = specWithCtx(spec0, LINKED_CTX).maxContext
      const j = ctxsFor(gi, vi).findIndex((c) => c.tokens === t)
      if (j >= 0) xi = j
    }
  }
  let mascot: MascotHandle | null = null
  /** The swarm stage-mode, when it is on. Non-null exactly while the root
   *  carries `cs-swarm` — the second mode this screen has, after cs-chatting. */
  let swarm: SwarmHandle | null = null
  /**
   * THE AGREEMENT ON SCREEN, or null when the gate is down.
   *
   * Non-null exactly while the consent dialog is open, and it holds the plan
   * that was DESCRIBED — not a pointer at the live selection. Every other
   * input path asks this variable first (keyIntent, the click delegate,
   * engage), and the accept button boots `gate.plan`, so the thing that boots
   * and the thing that was agreed to are the same object.
   */
  let gate: { plan: BootPlan; room: boolean } | null = null
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
       <span class="mb-dot-text"><b>${g.name}</b><i>${g.params}</i></span>
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
    const ctxs = ctxsFor(gi, vi)
    const cx = ctxs[xi] ?? ctxs[0]
    {
      // The SAME query the boot uses — one resolver, so a middle-click and an
      // in-place ENTER cannot land on different builds.
      const q2 = planFor({ gi, vi, mi, xi }).query
      el<HTMLAnchorElement>('.mb-cta').href = `zero-tvm.html${q2}`
      // The room path carries the SAME build choices — share.html reads
      // ?pool= and ?ctx= too, so a modified click hosts the build that was
      // chosen, not silently the full model.
      el<HTMLAnchorElement>('.mb-cta-room').href = `share.html${q2}`
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
      // Stated for EVERY model, not only the ones offering a choice. The
      // picker above renders only when a group ships two builds, so without
      // this row five of seven characters showed no quantisation and read as
      // full precision. None of them are.
      ['Quantisation', quantLabel(v.spec)],
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
    // THE DOORWAY. This amber line is the page's only statement that a model
    // may not fit the machine reading it, and until now nothing on the scene
    // connected it to the one thing that answers it. The control sits IN the
    // sentence, on the models the registry says can actually be cut — offering
    // it on an MLC checkpoint would hand out a ?layers= URL that throws at boot.
    // The RAM line states a fact and stops there. It used to carry the only
    // door to the swarm, which put a primary action inside a caveat, on a
    // sheet that is itself below the fold on the larger characters.
    const splittable = canSplitAcrossDevices(v.spec)
    ram.hidden = !(baseNote || ctxNote || b.qualityNote)

    // The split offer lives with the room verb, where it reads as a capability
    // rather than a fallback. Absent on the three MLC specs, whose loader
    // refuses a layerRange.
    const splitNote = el<HTMLElement>('.cs-verbs-split')
    // This button is destroyed on every paint(), and the keydown handler lives
    // on #model-browser — so a keyboard user who pressed an arrow while it held
    // focus landed on <body>, out of the handler's reach, and every later arrow
    // did nothing at all. Same refocus the .mb-mode / .mb-ctx / .mb-variant
    // chips get, falling back to the scene itself when the character walked to
    // has no split button to restore focus to.
    const hadSplitFocus = splitNote.contains(document.activeElement)
    splitNote.replaceChildren()
    if (splittable) {
      const b = document.createElement('button')
      b.type = 'button'
      b.className = 'cs-verbs-split-btn'
      b.textContent = 'Too big for one machine? Split it ▸'
      splitNote.append(b)
      if (hadSplitFocus) b.focus()
    } else if (hadSplitFocus) root.focus()

    for (const d of root.querySelectorAll<HTMLElement>('.mb-dot')) {
      const on = Number(d.dataset.i) === gi
      d.setAttribute('aria-selected', String(on))
      // The rail scrolls, so the arrows could walk the stage onto a character
      // whose card was below the fold — the roster then showed no selection at
      // all and the model on stage looked like one the roster did not carry.
      // 'nearest' scrolls only when the card is actually out of view.
      if (on) d.scrollIntoView({ block: 'nearest' })
      // ANY variant counts: the user who downloaded the 3-bit 35B has the
      // group ready even though variants[0] is the 4-bit — keying the badge
      // to variants[0] alone hid exactly the downloads worth showing.
      const grp = GROUPS[Number(d.dataset.i)]
      d.toggleAttribute('data-cached', grp.variants.some((x) => CACHED.has(x.spec.id)))
      // Save slot: a conversation stored on this device continues on enter.
      d.toggleAttribute('data-save', grp.variants.some((x) => hasSave(x.spec.id)))
      // Who can be split. The old section hid the answer inside a filtered
      // picker, so the MLX-only rule read as an arbitrary short list; in swarm
      // mode the rail dims the characters that cannot, which states the limit
      // on the roster where the reader is already looking.
      d.toggleAttribute('data-splittable', grp.variants.some((x) => canSplitAcrossDevices(x.spec)))
    }
    const cached = el<HTMLElement>('.mb-cached')
    cached.hidden = !CACHED.has(v.spec.id)
    el<HTMLElement>('.mb-save').hidden = !hasSave(v.spec.id)

    // The character wears the chosen builds: pool leans the figure, and the
    // context build sets the EARS (earCtx is derived from maxContext) — pick
    // Full and they grow.
    mascot?.setSpec(specWithCtx(v.spec, cx.tokens), CACHED.has(v.spec.id),
      mode ? poolFracOf(v.spec, mode.slots) : 0)

    // In swarm mode the character on stage IS the model being split, so a
    // repaint has to re-point the links at it. refresh() is a no-op when the
    // model has not changed — the cache probe repaints, and it must not wipe a
    // room link somebody just pasted.
    swarm?.refresh(v.spec, v.param)
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

  /** Leave swarm mode. The sheet's own rows are still in the DOM underneath —
   *  the mode hides them rather than replacing them — so a repaint restores
   *  the stat sheet exactly. */
  function exitSwarm(): void {
    swarm?.destroy()
    swarm = null
    root.classList.remove('cs-swarm')
    paint()
    selectFx()
  }

  /**
   * Enter swarm mode on the character currently on stage. Imported on demand
   * for the same reason the chat is: the entrance must render before either
   * module is fetched, and neither is needed to look at a roster.
   *
   * The transition is the SELECTION transition — cs-wipe plus the sheet's
   * row-stagger — because that is what this is: a second thing you can choose
   * about the character you are already looking at.
   */
  async function enterSwarm(): Promise<void> {
    const panel = root.querySelector<HTMLElement>('.mb-panel')
    if (!panel || swarm) return
    const { mountSwarm } = await import('./landing-swarm.js')
    const v = GROUPS[gi].variants[vi]
    swarm = mountSwarm({
      root,
      panel,
      spec: v.spec,
      param: v.param,
      onExit: exitSwarm,
      // The first machine is THIS machine. Sending it to share.html meant
      // leaving the game for a second page to do the one thing this page is
      // already set up to do; the other stages are other machines and still
      // need a URL. Booting here also removes the paste-the-room-link step:
      // the room strip writes the helper link once the stage is live.
      onHostHere: (bounds, index, ctxTokens) => {
        exitSwarm()
        void import('./landing-chat.js').then(({ enterChat }) => enterChat({
          root,
          spec: v.spec,
          param: v.param,
          poolSlots: 0,
          poolLabel: '',
          ctxTokens: ctxTokens !== v.spec.maxContext ? ctxTokens : 0,
          openRoom: true,
          layerRange: { start: bounds[index], end: bounds[index + 1] },
          // The room strip writes a link per stage, so it needs the whole
          // split — this machine's slice alone cannot say where the others
          // begin, and with moved cuts it cannot even be inferred.
          split: { bounds: [...bounds], index, ctx: ctxTokens },
          mascot,
        })).catch((err) => console.error('[landing] hosting a stage here failed:', err))
      },
    })
    root.classList.add('cs-swarm')
    const live = root.querySelector<HTMLElement>('.cs-live')
    if (live) live.textContent = `${GROUPS[gi].name} — split across machines`
    selectFx()
  }

  function go(nextG: number): void {
    gi = (nextG + GROUPS.length) % GROUPS.length
    vi = 0
    mi = 0
    xi = 0
    if (swarm) {
      // Qwen3-4B ships an MLC build and an MLX one under one name, and only
      // the MLX one can be cut — land on the build the mode is about rather
      // than on variants[0] and then hand out links for a different one.
      const k = GROUPS[gi].variants.findIndex((x) => canSplitAcrossDevices(x.spec))
      if (k < 0) {
        // Nothing to split here. Give the sheet back FIRST, then say why:
        // exitSwarm() repaints, and paint()'s first act is to write the model
        // name into this same aria-live region. Two writes in one task announce
        // only the last, so saying it first meant the builder vanished with no
        // reason — neither on screen nor announced.
        exitSwarm()
        const live = root.querySelector<HTMLElement>('.cs-live')
        if (live) live.textContent = `${GROUPS[gi].name} cannot be split — it is not an MLX checkpoint`
        return
      }
      vi = k
    }
    paint()
    selectFx()
  }

  host.addEventListener('click', (e) => {
    const t = e.target as HTMLElement
    // THE GATE IS MODAL, AND POINTER INPUT IS HALF OF WHAT THAT HAS TO MEAN.
    // showModal() already makes the document behind the dialog inert, so a
    // roster card cannot be clicked through it — but the dialog is a
    // DESCENDANT of #model-browser, so its own clicks bubble here, and a
    // future move of the markup would silently restore the bypass. The gate's
    // two buttons carry their own listeners and do not need this delegate.
    if (gate) return
    // The RAM line's doorway. First, because it lives inside the sheet and
    // every other branch below would otherwise have to know about it.
    if (t.closest('.cs-verbs-split-btn')) { void enterSwarm(); return }
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
    // data-v guards the branch: the swarm mode's machine-count chips wear
    // .mb-variant for the chip styling but carry no data-v, and Number(undefined)
    // is NaN — which would index the variant list out of existence.
    const variant = t.closest<HTMLElement>('.mb-variant')
    if (variant && variant.dataset.v !== undefined) {
      vi = Number(variant.dataset.v); mi = 0; xi = 0; paint(); refocus(`.mb-variant[data-v="${vi}"]`)
    }
  })
  host.tabIndex = 0
  host.setAttribute('role', 'application')
  host.setAttribute('aria-label', 'Character select — Up and Down arrows change model, Enter opens the chat')
  host.addEventListener('keydown', (e) => {
    // The rule is keyIntent()'s, not this listener's — it is unit-tested, and
    // the bug it closes (Enter walking past an open consent gate) is only
    // decidable headlessly because the decision left the listener body.
    const act = keyIntent(e.key, (e.target as HTMLElement).tagName, {
      chatting: root.classList.contains('cs-chatting'),
      gated: gate !== null,
      swarm: swarm !== null,
    })
    if (act === 'ignore') return
    if (act === 'exit-swarm') { e.preventDefault(); exitSwarm(); return }
    if (act === 'next') { e.preventDefault(); go(gi + 1); return }
    if (act === 'prev') { e.preventDefault(); go(gi - 1); return }
    root.querySelector<HTMLAnchorElement>('.mb-cta')?.click()
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
  /**
   * THE ONE BOOT PATH, and it takes a PLAN — never the live selection.
   *
   * A click on ENTER builds a plan from what is on stage at that instant; the
   * consent dialog builds one when it OPENS and hands back the same object on
   * accept. Neither can read `gi/vi/mi/xi` from in here, which is what let a
   * gate describing Llama-3.2-1B boot Qwen3.6-35B-A3B: the accept used to
   * synthesise a click on `.mb-cta`, and that click resolved the roster
   * afresh, two ArrowDowns later.
   */
  function enter(plan: BootPlan, openRoom: boolean): void {
    // ENTER is still on screen in swarm mode — it is the "run it here instead"
    // path. Chat mode owns the same sheet and the same stage, so the swarm has
    // to hand both back before the panel mounts.
    if (swarm) exitSwarm()
    if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
      root.querySelector<HTMLElement>('.cs-engage')?.classList.add('cs-go')
    }
    // ?split=0,8,16&stage=0 — the room strip's way of changing the split. It
    // only applies to the character the URL actually names, so walking the
    // roster afterwards cannot carry a stale set of bounds onto a model with
    // a different layer count.
    const st = intent.split && Q.get('model') === plan.param ? intent.split : null
    const urlRange = st ? { start: st.bounds[st.index], end: st.bounds[st.index + 1] } : undefined
    const urlSplit = st ? { bounds: st.bounds, index: st.index, ctx: plan.ctxTokens } : undefined
    import('./landing-chat.js').then(({ enterChat }) => enterChat({
      root,
      spec: plan.spec,
      param: plan.param,
      poolSlots: plan.poolSlots,
      poolLabel: plan.poolLabel,
      ctxTokens: plan.ctxTokens !== plan.spec.maxContext ? plan.ctxTokens : 0,
      openRoom: openRoom || urlSplit !== undefined,
      layerRange: urlRange,
      split: urlSplit,
      mascot,
    })).catch((err) => {
      // The panel could not even mount — fall back to the standalone page,
      // on the SAME plan rather than on whatever the roster now reads.
      console.error('[landing] in-place chat failed, navigating:', err)
      location.href = `zero-tvm.html${plan.query}`
    })
  }

  const engage = (e: MouseEvent, openRoom: boolean): void => {
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
    if (root.classList.contains('cs-chatting')) { e.preventDefault(); return }
    // The verbs are behind an open modal and cannot be clicked; a synthesised
    // click still could, so the state is checked rather than assumed.
    if (gate) { e.preventDefault(); return }
    if (!('gpu' in navigator)) return  // navigate — the fallback page explains what is missing
    e.preventDefault()
    enter(planFor({ gi, vi, mi, xi }), openRoom)
  }
  root.querySelector<HTMLAnchorElement>('.mb-cta')?.addEventListener('click', (e) => engage(e, false))
  // The room path is the same summoning with the room strip already open on
  // its CONSENT step — discoverable from the select screen, never auto-armed.
  root.querySelector<HTMLAnchorElement>('.mb-cta-room')?.addEventListener('click', (e) => engage(e, true))
  // ?chat=1 — how the chat panel's build strip comes back after changing a
  // build. It goes through the CTA's own click rather than a second entry
  // point, so the auto path and the human path cannot drift.
  /**
   * ASK BEFORE SPENDING GIGABYTES — the URL path only.
   *
   * `?chat=1` used to click ENTER for you: any link a stranger sent started a
   * multi-gigabyte download and an eager KV allocation (up to 16 GiB at the
   * top of the ?ctx= range) before the page had finished painting, with no
   * click anywhere. share.html grew a gate for exactly this incident — a
   * visitor who clicked "Rooms" in the site nav was ~2 GB into a download
   * before reading a word about it — and the entrance is the surface people
   * actually link to.
   *
   * Same rule as confirmDownload() in share.ts: the question is ALWAYS put,
   * and an already-cached model changes the WORDING, not whether anyone
   * agreed. Clicking ENTER on this page is a click already and is not gated —
   * this is only the door a URL opens.
   */
  /**
   * WHY THE SELECTION IS CAPTURED HERE AND THE GATE DOES NOT REPAINT.
   *
   * Two designs answer "the gate named a model it will not boot": repaint the
   * gate from paint() so its words track the roster, or capture the selection
   * when the gate opens and boot from the capture. This is the capture, and
   * the roster is FROZEN behind it — those two halves are one decision, not
   * two.
   *
   * Repainting keeps the words in sync but not the READING. The visitor reads
   * a sentence about 528 MB, presses ArrowDown twice to scroll a full-screen
   * scene, and the sentence they have already read silently becomes a
   * different sentence about 16.4 GB and ~20 GB of free RAM. Consent then
   * means "whatever the text said at the instant of the click", which is the
   * property that made `?chat=1` unsafe in the first place. It also cannot be
   * made honest cheaply: the wording turns on an AWAITED cache probe, so
   * every roster move needs a re-probe, and until one lands the dialog either
   * shows the previous model's "already cached" clause or flickers its own
   * button between "Enter chat" and "Download & enter".
   *
   * Capturing is only honest if changing your mind is POSSIBLE, which is why
   * the dialog carries "Not now". A deliberate change of character while the
   * gate is up is neither silently ignored nor silently obeyed — it cannot
   * happen. Decline (or press Escape), the scene comes back, the
   * act-without-a-click keys leave the address bar, and choosing another
   * character and pressing ENTER is a click, which is consent on its own
   * terms.
   */
  async function openUrlGate(room: boolean): Promise<void> {
    const dlg = el<HTMLDialogElement>('.cs-url-gate')
    const go = el<HTMLButtonElement>('#cs-gate-go')
    const no = el<HTMLButtonElement>('#cs-gate-no')
    // `hidden` loses to `#models .cs-verbs { display: flex }`, so the verbs
    // are moved by the property that actually wins. BOTH directions live in
    // setGate, never in a handler's success path: the first version restored
    // them only when the accept button was clicked, so a `?chat=1` link that
    // was never accepted left "Enter chat" and "⟁ Open a room" gone from the
    // page for good, and editing the address bar was the only way back into
    // the site.
    const setGate = (g: { plan: BootPlan; room: boolean } | null): void => {
      gate = g
      for (const sel of ['.cs-verbs', '.cs-verbs-note']) {
        const n = root.querySelector<HTMLElement>(sel)
        if (n) n.style.display = g ? 'none' : ''
      }
      if (g === null) {
        // The link has been ANSWERED — either way — so take its verbs out of
        // the address bar: "⟨ Roster" and /roster both reload, a reload keeps
        // the query, and `?chat=1` then walks straight back into the gate.
        history.replaceState(null, '', `${location.pathname}${urlAfterEnter(location.search)}${location.hash}`)
      }
    }

    const plan = planFor({ gi, vi, mi, xi })
    // The URL's stage, if it named one — what this device fetches is a SLICE
    // then, and quoting the whole checkpoint's size for it is the figure that
    // once promised an iPhone 14.1 GB for a fraction of it.
    const st = intent.split && Q.get('model') === plan.param ? intent.split : null
    const stage = st ? { start: st.bounds[st.index], end: st.bounds[st.index + 1] } : null

    /** The dialog's words, from the plan it will boot and nothing else. */
    const say = (cached: boolean): void => {
      const c = gateCopy(plan, { room, cached, stage, int8: INT8_KV })
      el<HTMLElement>('#cs-gate-title').textContent = c.title
      el<HTMLElement>('#cs-gate-what').textContent = c.what
      el<HTMLElement>('#cs-gate-cost').textContent = c.cost
      go.textContent = c.go
    }
    // Written COLD and shown BEFORE the probe runs, so the dialog's opening
    // announcement (aria-labelledby + aria-describedby) already carries the
    // download size and the RAM note. Cold is also the honest side to be
    // wrong on.
    say(false)
    setGate({ plan, room })
    dlg.showModal()

    // EVERY WAY OUT IS WIRED BEFORE THE PROBE IS AWAITED. Attaching these
    // after it means that for as long as the probe runs, "Not now" does
    // nothing and Escape closes the dialog with no handler — leaving the
    // scene gated, the verbs hidden and no dialog on screen: worse than the
    // stranding this replaces. The accept does not need the probe's answer
    // either; `go` is disabled until it lands, which is what makes it wait.
    let accepted = false
    go.addEventListener('click', () => {
      accepted = true
      go.disabled = true
      dlg.close()
      setGate(null)
      if (!('gpu' in navigator)) { location.href = `zero-tvm.html${plan.query}`; return }
      // The PLAN, not the roster. This used to synthesise a click on the
      // hidden CTA, and that click resolved the live selection all over again.
      enter(plan, room)
    }, { once: true })
    no.addEventListener('click', () => dlg.close())
    // Escape fires `cancel` and then `close`, and the decline path is the
    // same either way — so it is written once, on `close`.
    dlg.addEventListener('close', () => {
      if (accepted) return
      setGate(null)
      root.querySelector<HTMLElement>('.mb-cta')?.focus()
    })

    // The SAME probe share.html's gate uses, on the same spec and the same
    // stage, so the two surfaces cannot disagree about whether this device
    // already holds the weights. Dynamic: it pulls in the loaders, which read
    // GPUBufferUsage at module scope.
    //
    // BOUNDED. A probe that never settles used to leave the button disabled
    // on "Checking this device…" with the verbs already hidden — a page with
    // no way forward and no way back. Asking without the cached wording is
    // strictly better than never asking.
    let cached = false
    try {
      if ('gpu' in navigator) {
        const { isModelCached } = await import('./zero-tvm/cache-probe.js')
        cached = await Promise.race([
          isModelCached(plan.spec, stage ?? undefined),
          new Promise<boolean>((r) => { setTimeout(() => r(false), 2500) }),
        ])
      }
    } catch { /* treat as cold — the honest side to be wrong on */ }
    if (gate === null) return            // declined while the probe was out
    if (cached) say(true)                // role=status announces the change
    go.disabled = false
    // NO go.focus() HERE. showModal() has already placed focus inside the
    // dialog; stealing it when an async probe happens to land yanks a
    // keyboard user mid-Tab onto a button whose label is all they are told.
    // The focus showModal() gives is "Not now", because it is the first
    // enabled control — the safe default for a question about gigabytes.
  }

  // "Swarm" in the nav and the footer pointed at the section below. That
  // section is the no-JS fallback now, so with JS the same href opens the
  // stage mode instead — and a browser that never runs this still scrolls to
  // the prose, which is the whole point of leaving the href alone.
  /** Open the swarm mode. THREE doors land here — the nav and footer links'
   *  click, the hash on a cold load, and a later hashchange — so they cannot
   *  drift. Only the click was ever wired, and README publishes
   *  https://zerotvm.com/#swarm, so the published link opened nothing. */
  function openSwarm(): void {
    window.scrollTo({ top: 0, behavior: 'smooth' })
    if (swarm) return
    // The character on stage may be one that cannot be cut — walk to the
    // first that can rather than opening a mode with nothing in it.
    if (!canSplitAcrossDevices(GROUPS[gi].variants[vi].spec)) {
      const g = GROUPS.findIndex((x) => x.variants.some((y) => canSplitAcrossDevices(y.spec)))
      if (g < 0) return
      gi = g
      vi = GROUPS[g].variants.findIndex((x) => canSplitAcrossDevices(x.spec))
      mi = 0
      xi = 0
      paint()
    }
    void enterSwarm()
  }
  for (const link of document.querySelectorAll<HTMLAnchorElement>('a[href="#swarm"]')) {
    link.addEventListener('click', (e) => {
      if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return
      e.preventDefault()
      openSwarm()
    })
  }
  // The click preventDefaults, so the hash only ever arrives from OUTSIDE:
  // a published link, a bookmark, someone editing the address bar.
  window.addEventListener('hashchange', () => {
    if (entranceIntent(location.search, location.hash).swarm) openSwarm()
  })

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

  // ── What the URL asked for, now that the scene is on screen ──────────────
  // Neither may happen silently: the gate asks before a byte is spent, and
  // #swarm opens a mode that starts nothing.
  if (intent.enter) void openUrlGate(intent.enter.room)
  else if (intent.swarm) openSwarm()
}

// The URL grammar above is exported and unit-tested; the scene renders only
// where there is a page to render it into, so the headless suite can import
// this module without a DOM.
if (typeof document !== 'undefined') render()
