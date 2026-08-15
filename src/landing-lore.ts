/**
 * LANDING LORE — the entrance's spec-derived flavour layer, shared by the
 * character-select screen (landing.ts) and the in-place chat it opens into
 * (landing-chat.ts). Pure data + pure functions over ModelSpec: no DOM, no
 * GPU imports, safe to load on any browser.
 */

import type { ModelSpec } from './compiler/model-spec.js'

/** Class sigils — one per architecture lane, same circuit-rune language as
 *  the /entrance assets. currentColor, so they sit in the accent for free. */
export const LANE_SIGIL: Record<string, string> = {
  moe: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l3 3-3 3-3-3zM5 9l2.4 2.4L5 13.8 2.6 11.4zM19 9l2.4 2.4L19 13.8l-2.4-2.4zM12 16l3 3-3 3-3-3z" opacity="0.9"/><circle cx="12" cy="11.4" r="1.6"/></svg>',
  hybrid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M2 9c3 0 3-4 6-4s3 4 6 4 3-4 6-4"/><path d="M2 16h6l3-5 3 8 2-3h6" opacity="0.85"/></svg>',
  dense: '<svg viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1" opacity="0.55"/><rect x="4" y="13" width="7" height="7" rx="1" opacity="0.55"/><rect x="13" y="13" width="7" height="7" rx="1"/></svg>',
  mla: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l8 9-8 9-8-9z"/><path d="M12 8l4 4-4 4-4-4z" fill="currentColor" stroke="none" opacity="0.7"/></svg>',
  embed: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 12c2.5-4 6-6 9-6s6.5 2 9 6c-2.5 4-6 6-9 6s-6.5-2-9-6z"/><path d="M8 12h8" opacity="0.8"/></svg>',
}

export function laneOf(spec: ModelSpec): string {
  return spec.embeddingOnly ? 'embed' : spec.mla ? 'mla' : spec.moe ? 'moe'
    : spec.layerKinds.some((k) => k === 'gdn') ? 'hybrid' : 'dense'
}

/** One line of lore per character — every clause computed from the spec, so
 *  the flavour text is as registry-true as the stat rows. */
export function loreOf(spec: ModelSpec): string {
  if (spec.embeddingOnly) return 'Returns a vector. It does not speak.'
  if (spec.mla) return 'Attends through a compressed latent — the cache is 7× smaller than it looks.'
  if (spec.moe) {
    const gdn = spec.layerKinds.filter((k) => k === 'gdn').length
    return `Routes every token through ${spec.moe.topK} of ${spec.moe.experts} experts`
      + (gdn > 0 ? `, ${gdn} of ${spec.layers} layers recurrent.` : '.')
  }
  const gdn = spec.layerKinds.filter((k) => k === 'gdn').length
  if (gdn > 0) return `${gdn} recurrent layers, ${spec.layers - gdn} attention — memory that does not grow.`
  return `${spec.layers} layers straight through, ${spec.heads} heads each.`
}
