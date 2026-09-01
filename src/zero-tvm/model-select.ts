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
import { specForParam, specWithCtx } from './model-registry.js'
import { ctxFrom } from './room-url.js'
import { resolveModelBase } from './weight-loader.js'
import { loadTokenizer, buildChatPrompt as buildPhi3ChatPrompt, type Tokenizer } from './tokenizer.js'
import { loadByteLevelTokenizer, buildChatPrompt as buildChatMLPrompt, buildDeepSeekChatPrompt, buildLlama3ChatPrompt, buildMistralChatPrompt, buildTuluChatPrompt } from './tokenizer-bpe.js'

export type ChatMessage = { role: 'system' | 'user' | 'assistant'; content: string }

/** `?model=<param>` → spec, straight from SHIPPED_MODELS — registering a model
 *  there is what creates its URL. Default (and anything unknown) is Phi-3. */
export function specFromSearch(search: string): ModelSpec {
  const q = new URLSearchParams(search)
  const base = specForParam(q.get('model'))
  // `?ctx=` is documented on specWithCtx (model-registry.ts) — the logic lives
  // there so it can be unit-tested without this file's weight-loader import.
  // The link is read with room-url.ts's ctxFrom, the same parser share.ts
  // routes on, so a helper booting from a host's link cannot size its KV cache
  // off a different rule.
  //
  // NO ctx in the link → the spec UNTOUCHED, not a round-trip through
  // specWithCtx. This was `specWithCtx(base, ctxFor(search, base.maxContext))`
  // — feeding a spec's own maxContext back through its clamp, which is a no-op
  // on every spec whose maxContext sits below maxSeq, and a SHRINK on the one
  // where it does not. KV pages round UP: Phi-3's 257 pages hold 4112 tokens
  // against a 4096-token trained window, so clamping to floor(maxSeq/pageSize)
  // took a page off the DEFAULT model of zero-tvm.html and validate.html
  // silently. scripts/station.mjs states the rule ("a naive clamp to maxSeq
  // would SHRINK the shipped default") and computes its ceiling as
  // max(maxSeq, maxContext) for exactly this reason.
  //
  // An EXPLICIT ctx still goes through specWithCtx and is still clamped —
  // asking for a budget is a request to re-size, and past maxSeq the model
  // degrades without erroring. Not asking is not a request.
  const ctx = ctxFrom(search)
  return ctx === null ? base : specWithCtx(base, ctx)
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
  /** Out-param: receives the token index where the trailing generation prompt
   *  begins — the exact point the NEXT turn's prompt diverges from this one.
   *  generatePipelined takes it as `rewindAt` and puts a GDN rewind snapshot
   *  there, which is what makes hybrid cross-turn reuse work at all. Only the
   *  ChatML builders report it today; the others leave it untouched, and the
   *  engine reads an unset value as "no rewind point". */
  split?: { genStart: number },
): number[] {
  // ModelSpec.chatTemplateId (model-spec.ts) still declares only the four
  // original ids, and `===` against a literal outside a union is a type ERROR
  // rather than a branch that is never taken — so the dispatch reads the raw
  // string until that union is widened with the ids below.
  const id: string = spec.chatTemplateId
  if (id === 'chatml' || id === 'chatml-q35' || id === 'chatml-q38') {
    // Non-thinking: emit the template's explicit `enable_thinking is false`
    // suffix so the model skips the <think> block on THIS turn. `generation`
    // picks which past turns keep one — the two Qwen generations disagree, and
    // rendering 3.6 with the 3.0 rule breaks tool calling at depth without
    // changing anything visible.
    return buildChatMLPrompt(messages, tokenizer, {
      thinking: false,
      generation: id === 'chatml-q38' ? 'qwen38' : id === 'chatml-q35' ? 'qwen35' : 'qwen3',
      ...(split ? { split } : {}),
    })
  }
  if (id === 'deepseek') {
    // Prose turns, not header tokens — see buildDeepSeekChatPrompt for the two
    // details the template hides (unlabelled system, no space after
    // "Assistant:"). Pinned against the vendor jinja in
    // tests/unit/chat-template-deepseek.test.ts.
    return buildDeepSeekChatPrompt(messages, tokenizer)
  }
  if (id === 'llama3') {
    // Llama-3 header template (pinned token-exact vs mlx_lm apply_chat_template
    // — tests/tokenizer/tokenizer-llama3.test.ts).
    return buildLlama3ChatPrompt(messages, tokenizer)
  }
  if (id === 'mistral' || id === 'mistral-nemo' || id === 'ministral') {
    // One `[INST]` family, three published spacings — one id per checkpoint
    // because they are not interchangeable. See buildMistralChatPrompt for
    // which repo each came from and for what the system-message fold costs the
    // 2407/2410 pair in cross-turn reuse. Pinned in
    // tests/unit/chat-template-mistral.test.ts.
    return buildMistralChatPrompt(messages, tokenizer, {
      variant: id === 'mistral' ? 'v1' : id === 'mistral-nemo' ? 'nemo' : 'ministral',
    })
  }
  if (id === 'olmo2' || id === 'falcon3') {
    // The same Tulu-3 template both ways; OLMo-2 opens with its bos token and
    // Falcon3 has none (bos_token: null). OLMoE-1B-7B would land here too, but
    // its eos is `|||IP_ADDRESS|||`, so registering it means passing eosToken —
    // the spec carries no template-token strings today. Pinned in
    // tests/unit/chat-template-tulu.test.ts.
    return buildTuluChatPrompt(messages, tokenizer, id === 'olmo2' ? { bosToken: '<|endoftext|>' } : undefined)
  }
  return buildPhi3ChatPrompt(messages, tokenizer)
}
