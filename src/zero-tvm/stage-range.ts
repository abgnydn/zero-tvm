/**
 * `?layers=` BOUNDED AGAINST THE CHECKPOINT IT NAMES.
 *
 * room-url.ts's `stageRangeFrom` is a PARSER, not a validator: it accepts any
 * `\d+-\d+` because it has no spec to check a range against, and it is shared
 * with the swarm link builder, which has no spec either. So the bound lives
 * where a spec exists — at share.html's routing point, before any role is
 * handed a range.
 *
 * It had no bound at all. share.ts checked `start !== 0` and nothing checked
 * `end`. Measured on qwen38 (64 layers):
 *
 *   ?model=qwen38               "The weights download once (~14.1 GB)…"
 *                               "needs ~18 GB free RAM"
 *   ?model=qwen38&layers=0-64   "Layers 0-64 of 64 download once — a slice of
 *                               the full ~14.1 GB, not all of it", RAM note
 *                               HIDDEN
 *   ?model=qwen38&layers=0-9999 the same sentence for a range past the end;
 *                               planModel then emitted plans past spec.layers
 *                               and the loader died after fetching all 64
 *
 * 0-64 IS the whole model. The crafted link understated a 14.1 GB download as a
 * slice of itself AND deleted the RAM warning — `confirmDownload` suppresses
 * `brand.ramNote` whenever a stage is set, correctly, because a whole-checkpoint
 * RAM figure is false for a real stage. So a fake stage buys silence on the one
 * limit a visitor cannot undo by waiting.
 *
 * A HOST's stage must end BELOW `layers`: a hosting stage that ends the model is
 * the whole model, and must not be described as a slice of it. A HELPER's may
 * end AT `layers` — `?layers=k-N#room` is the far half of a split, and that is
 * the whole point of a helper.
 *
 * NULL MEANS NO STAGE, not a refusal to boot: the caller falls back to the whole
 * model and the honest whole-model gate. That is the ruling the entrance's
 * `splitFor` already made for the sibling keys — "a malformed split is no
 * split" — and this is the same class of link on the surface the entrance's own
 * room CTA points at.
 *
 * A CHECKPOINT THE LOADER CANNOT CUT HAS NO STAGE, which is why the layer
 * count is nullable: `null` is a caller saying "this model has no cuttable
 * layers at all". `?model=&layers=0-31` on Phi-3 (32 layers, MLC shards)
 * painted an entire consent screen for a split that cannot exist — title
 * "Phi-3-mini · layers 0–31 of 32", the whole model sold as "a slice … not all
 * of it", the RAM note gone, and a paragraph about hosting a split — and then
 * died inside loadWeights, which refuses a layerRange on anything but an MLX
 * checkpoint before a byte is fetched. `splitFor` (landing.ts) has asked
 * `canSplitAcrossDevices` since it was written; this module inherited the two
 * bounds and neither of the other two rules.
 *
 * A HOST STAGE STARTS AT LAYER 0, the other one. A hosting stage that skips
 * the start of the model has no embedding, and share.ts THREW for it — from
 * inside runHost, after the title, the nameplate and the consent paragraph had
 * already been painted with the stage it was about to refuse. Same ruling as
 * everything else here: a malformed range is no range, and the fallback is the
 * whole model with the honest whole-model gate.
 *
 * Imports nothing, and takes a layer COUNT rather than a spec, so the rules can
 * be held by the headless suite. share.ts cannot: `signalEnv()` reads
 * `location` at module scope, so importing that file outside a browser throws.
 */
export function stageFor(
  range: { start: number; end: number } | null,
  /** The checkpoint's layer count, or `null` when it cannot be cut at all. */
  layers: number | null,
  kind: 'host' | 'helper',
): { start: number; end: number } | null {
  if (!range) return null
  if (layers === null) return null
  const { start, end } = range
  // Self-contained rather than leaning on `stageRangeFrom`'s regex to have
  // already ruled these out. A validator that is correct only for one caller's
  // parser is a check waiting to be wrong when a second caller appears.
  if (!Number.isInteger(start) || !Number.isInteger(end)) return null
  if (start < 0 || end <= start) return null
  if (kind === 'host' && start !== 0) return null
  return end <= (kind === 'host' ? layers - 1 : layers) ? range : null
}

/**
 * WHAT A CONSENT GATE MAY SAY ABOUT A STAGE — one generator, both surfaces.
 *
 * The bound above stops `0-64` of 64. It does not stop `0-63`, and it must
 * not: that is the host's half of a two-machine split. But `gateCopy`
 * (landing.ts) and `confirmDownload` (share.ts) each dropped the
 * whole-checkpoint RAM note for ANY non-null stage, so `?layers=0-63` fetched
 * ~13.9 of 14.1 GB and read, word for word, like `?layers=0-8`:
 *
 *   ?model=qwen38               #gate-ram "needs ~18 GB free RAM"
 *   ?model=qwen38&layers=0-64   #gate-ram "needs ~18 GB free RAM"  (bounded)
 *   ?model=qwen38&layers=0-63   #gate-ram HIDDEN
 *   ?model=qwen38&layers=0-8    #gate-ram HIDDEN, same sentence
 *
 * A stage that is 98% of a checkpoint must not read like one that is 12%, and
 * neither may buy silence on the one limit a visitor cannot undo by waiting.
 *
 * TWO THINGS, AND NEITHER IS AN INVENTED NUMBER. The share is arithmetic on
 * the two layer counts the sentence already prints. The RAM figure is the
 * registry's own whole-checkpoint one, LABELLED as the whole checkpoint's
 * rather than deleted — which is what keeps the fix the suppression was
 * written for (the iPhone asked to approve "~14.1 GB" for one layer of the
 * 27B, real device, 2026-08-29): nothing here claims a per-stage byte count,
 * because a stage's real size is not knowable until the safetensors headers
 * are read, after consent. It states the checkpoint's figures, says what
 * fraction of it this device takes, and lets the reader divide.
 *
 * Both gates call this, so the two cannot drift. What each does with the
 * strings is its own: the entrance folds `ram` into its cost line, share.html
 * gives it its own paragraph and appends the progress-panel sentence.
 */
export function stageGateCopy(o: {
  stage: { start: number; end: number } | null
  layers: number
  /** The bytes are already on this device — changes the WORDING, never
   *  whether the question is put. */
  cached: boolean
  /** The registry's whole-checkpoint download size, e.g. '~14.1 GB'. */
  sizeLabel: string
  /** The registry's whole-checkpoint RAM note, or '' where a model has none. */
  ramNote: string
}): { weights: string; ram: string } {
  const s = o.stage
  if (!s) {
    return {
      weights: o.cached
        ? 'The weights are already cached on this device.'
        : `The weights download once (${o.sizeLabel}) and are cached locally; every later visit starts from disk.`,
      ram: o.ramNote,
    }
  }
  const share = `${Math.round((100 * (s.end - s.start)) / o.layers)}%`
  return {
    weights: o.cached
      ? `Layers ${s.start}–${s.end} of ${o.layers} — ${share} of this checkpoint's layers — are already cached on this device.`
      : `Layers ${s.start}–${s.end} of ${o.layers} download once — ${share} of this checkpoint's layers, `
        + `out of a full ${o.sizeLabel} — and are cached locally.`,
    ram: o.ramNote ? `the whole checkpoint ${o.ramNote}` : '',
  }
}
