/**
 * MODEL SELECT — URL → ModelSpec, plus the per-spec tokenizer factory and
 * chat-template dispatch.
 *
 * Both shipped pages accept `?model=qwen3` (Qwen3-4B) and `?model=qwen35`
 * (Qwen3.5-4B hybrid GDN+attention); anything else (including no flag) boots
 * Phi-3, so all existing URLs keep their exact behavior. This module is the
 * one place that maps a spec to:
 *
 *   - the tokenizer implementation (SPM for Phi-3, byte-level BPE for Qwen3),
 *     fetching tokenizer.json from the spec's own HF repo
 *   - the chat template (Phi-3 `<|user|>` vs ChatML `<|im_start|>`); Qwen3
 *     chat runs the NON-thinking template in v1 (the HF template's
 *     `enable_thinking is false` branch, `<think>\n\n</think>\n\n` suffix) —
 *     the chat UX has no <think>-block rendering yet
 *   - display copy (model name + approximate download size) for gate/header UI
 */

import { PHI3, QWEN3_4B, QWEN35_4B, type ModelSpec } from '../compiler/model-spec.js'
import { modelBaseUrl } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt as buildPhi3ChatPrompt, type Tokenizer } from './tokenizer.js'
import { loadByteLevelTokenizer, buildChatPrompt as buildChatMLPrompt } from './tokenizer-bpe.js'

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

/** `?model=qwen3` selects QWEN3_4B, `?model=qwen35` selects QWEN35_4B;
 *  default (and anything else) is PHI3. */
export function specFromSearch(search: string): ModelSpec {
  const model = new URLSearchParams(search).get('model')
  if (model === 'qwen35') return QWEN35_4B
  if (model === 'qwen3') return QWEN3_4B
  return PHI3
}

/** Page-facing copy for the active model (gate dialog, header, hints). */
export function modelBranding(spec: ModelSpec): { name: string; sizeLabel: string; rateLabel: string } {
  // rateLabel is the TOTAL wall-clock median (prefill + decode) from the
  // 2026-07-30 corrected protocol — the conservative number a user actually
  // experiences. Decode-only rates run higher (83 / 75 / 73 t/s); see BENCH.md.
  if (spec.id === QWEN35_4B.id) {
    return { name: 'Qwen3.5-4B', sizeLabel: '~2.6 GB', rateLabel: '~65 t/s' }  // 65.28 total, M2 Max (BENCH.md 2026-07-30)
  }
  return spec.id === QWEN3_4B.id
    ? { name: 'Qwen3-4B', sizeLabel: '~2.3 GB', rateLabel: '~60 t/s' }  // 59.85 total, M2 Max (BENCH.md 2026-07-30)
    : { name: 'Phi-3-mini', sizeLabel: '~2 GB', rateLabel: '~70 t/s' }  // 69.55 total, M2 Max (BENCH.md 2026-07-30)
}

/**
 * Load the tokenizer the spec declares (spec.tokenizerKind), fetching
 * tokenizer.json from the spec's HF repo. The byte-level tokenizer is
 * structurally a superset of the SPM Tokenizer interface, so both return as
 * `Tokenizer` and every consumer (engine warmup, chat, validate) is agnostic.
 */
export async function loadTokenizerFor(
  spec: ModelSpec,
  onProgress?: (msg: string) => void,
): Promise<Tokenizer> {
  const base = modelBaseUrl(spec)
  return spec.tokenizerKind === 'byteLevel'
    ? loadByteLevelTokenizer(onProgress, base)
    : loadTokenizer(onProgress, base)
}

/** Encode `messages` with the spec's chat template (plus generation prompt). */
export function buildChatPromptFor(
  spec: ModelSpec,
  messages: ChatMessage[],
  tokenizer: Tokenizer,
): number[] {
  if (spec.chatTemplateId === 'chatml') {
    // Qwen3 v1 chat is non-thinking: emit the template's explicit
    // `enable_thinking is false` suffix so the model skips the <think> block.
    return buildChatMLPrompt(messages, tokenizer, { thinking: false })
  }
  return buildPhi3ChatPrompt(messages, tokenizer)
}
