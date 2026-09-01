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
 * Imports nothing, and takes a layer COUNT rather than a spec, so the rule can
 * be held by the headless suite. share.ts cannot: `signalEnv()` reads
 * `location` at module scope, so importing that file outside a browser throws.
 */
export function stageFor(
  range: { start: number; end: number } | null,
  layers: number,
  kind: 'host' | 'helper',
): { start: number; end: number } | null {
  if (!range) return null
  const { start, end } = range
  // Self-contained rather than leaning on `stageRangeFrom`'s regex to have
  // already ruled these out. A validator that is correct only for one caller's
  // parser is a check waiting to be wrong when a second caller appears.
  if (!Number.isInteger(start) || !Number.isInteger(end)) return null
  if (start < 0 || end <= start) return null
  return end <= (kind === 'host' ? layers - 1 : layers) ? range : null
}
