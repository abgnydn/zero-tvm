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

import type { ModelSpec } from '../compiler/model-spec.js'
import { specForParam } from './model-registry.js'
import { resolveModelBase } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt as buildPhi3ChatPrompt, type Tokenizer } from './tokenizer.js'
import { loadByteLevelTokenizer, buildChatPrompt as buildChatMLPrompt, buildDeepSeekChatPrompt, buildLlama3ChatPrompt } from './tokenizer-bpe.js'

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

/** `?model=<param>` → spec, straight from SHIPPED_MODELS — registering a model
 *  there is what creates its URL. Default (and anything unknown) is Phi-3. */
export function specFromSearch(search: string): ModelSpec {
  return specForParam(new URLSearchParams(search).get('model'))
}

export { modelBranding, SHIPPED_MODELS } from './model-registry.js'
export type { ModelBrand } from './model-registry.js'

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
  const base = await resolveModelBase(spec)
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
  if (spec.chatTemplateId === 'deepseek') {
    // Prose turns, not header tokens — see buildDeepSeekChatPrompt for the two
    // details the template hides (unlabelled system, no space after
    // "Assistant:"). Pinned against the vendor jinja in
    // tests/unit/chat-template-deepseek.test.ts.
    return buildDeepSeekChatPrompt(messages, tokenizer)
  }
  if (spec.chatTemplateId === 'llama3') {
    // Llama-3 header template (pinned token-exact vs mlx_lm apply_chat_template
    // — tests/tokenizer/tokenizer-llama3.test.ts).
    return buildLlama3ChatPrompt(messages, tokenizer)
  }
  return buildPhi3ChatPrompt(messages, tokenizer)
}
