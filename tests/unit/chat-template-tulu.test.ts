// TULU-3 CHAT TEMPLATE (OLMo-2, OLMoE, Falcon3) — our builder against the
// vendor's own jinja.
//
// Three RED repos in docs/PORTING.md share one template, and it is the SOLE
// blocker on two of them (OLMo-2-1124-7B-Instruct, Falcon3-7B-Instruct). Every
// expected string below was derived BY HAND from the chat_template in
// mlx-community/OLMo-2-1124-7B-Instruct-4bit and
// mlx-community/Falcon3-7B-Instruct-4bit tokenizer_config.json (fetched
// 2026-08-10 from huggingface.co/<repo>/resolve/main/tokenizer_config.json),
// then cross-checked by rendering those same vendor templates through jinja2 —
// the method behind tests/fixtures/deepseek-chat-template.json. None of these
// strings came out of the code under test.
//
// The two templates are character-identical except for the leading
// `{{ bos_token }}`, which OLMo-2/OLMoE have and Falcon3 (bos_token: null) does
// not — reproduced verbatim, with the jinja source whitespace that is NOT
// output marked where it misleads:
//
//   {{ bos_token }}{% for message in messages %}
//     {% if message['role'] == 'system' %}{{ '<|system|>\n' + message['content'] + '\n' }}
//     {% elif message['role'] == 'user' %}{{ '<|user|>\n' + message['content'] + '\n' }}
//     {% elif message['role'] == 'assistant' %}
//        {% if not loop.last %}{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}
//        {% else %}{{ '<|assistant|>\n'  + message['content'] + eos_token }}{% endif %}
//     {% endif %}
//     {% if loop.last and add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}
//   {% endfor %}
//
// (the double space in `'<|assistant|>\n'  + message['content']` sits between
// the closing quote and the `+`, so it is jinja source, not output. Reading it
// as a space in the transcript is the easiest mistake to make here.)
//
// The tokenizer is a stub that returns the string it was handed — what is under
// test is the ASSEMBLY. (The dispatch in model-select.ts cannot be reached from
// a unit test: it imports tokenizer.ts → weight-loader.ts, which reads
// GPUBufferUsage at module scope.)

import { describe, expect, it } from 'vitest'
import { buildTuluChatPrompt } from '../../src/zero-tvm/tokenizer-bpe.ts'

type Msg = { role: 'system' | 'user' | 'assistant'; content: string }

/** Hands back the text instead of ids, so the assertion reads as the prompt. */
const capture = { encode: (t: string) => [t] as unknown as number[] }
/** Falcon3: no bos, eos <|endoftext|> — the builder's defaults. */
const falcon3 = (msgs: Msg[]) => buildTuluChatPrompt(msgs, capture) as unknown as string[]
/** OLMo-2: the same template opened with its bos token, which is also its eos. */
const olmo2 = (msgs: Msg[]) =>
  buildTuluChatPrompt(msgs, capture, { bosToken: '<|endoftext|>' }) as unknown as string[]

describe('tulu chat template', () => {
  it('renders a single user turn plus the generation prompt', () => {
    expect(falcon3([{ role: 'user', content: 'Hi' }])[0]).toBe('<|user|>\nHi\n<|assistant|>\n')
  })

  it('labels the system turn (unlike DeepSeek, which emits it bare)', () => {
    expect(falcon3([{ role: 'system', content: 'You are terse.' }, { role: 'user', content: 'Hi' }])[0])
      .toBe('<|system|>\nYou are terse.\n<|user|>\nHi\n<|assistant|>\n')
  })

  it('puts eos INSIDE the transcript after an assistant turn, then a newline', () => {
    expect(falcon3([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: '  padded  ' },
    ])[0]).toBe('<|user|>\nHi\n<|assistant|>\nHello!<|endoftext|>\n<|user|>\n  padded  \n<|assistant|>\n')
  })

  it('does not trim content', () => {
    // Llama-3's template trims, this one interpolates verbatim — copying the
    // neighbouring builder would silently eat the user's whitespace.
    expect(falcon3([{ role: 'user', content: '  padded  ' }])[0])
      .toBe('<|user|>\n  padded  \n<|assistant|>\n')
  })

  it('drops the newline after eos when the assistant turn is LAST', () => {
    // `{% if not loop.last %}` — a prefill/continuation history ends with the
    // eos and goes straight into the generation prompt, no separator.
    expect(falcon3([{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }])[0])
      .toBe('<|user|>\nHi\n<|assistant|>\nHello!<|endoftext|><|assistant|>\n')
  })

  it('emits NO generation prompt for an empty history', () => {
    // The `<|assistant|>\n` is inside the for-loop, guarded by loop.last, so
    // an empty message list renders as the bos alone. Hoisting it out of the
    // loop — the obvious way to write this — would not.
    expect(falcon3([])[0]).toBe('')
    expect(olmo2([])[0]).toBe('<|endoftext|>')
  })

  it('opens with bos for OLMo-2 and with nothing for Falcon3', () => {
    // OLMo-2's template starts `{{ bos_token }}`; Falcon3's does not, and its
    // bos_token is null. Emitting <|endoftext|> at position 0 of a Falcon3
    // prompt would be an end-of-document token opening the document.
    expect(olmo2([{ role: 'user', content: 'Hi' }])[0]).toBe('<|endoftext|><|user|>\nHi\n<|assistant|>\n')
  })

  it('takes eos as a parameter — OLMoE\'s is not <|endoftext|>', () => {
    // OLMoE-1B-7B-0125-Instruct sets both bos_token and eos_token to
    // `|||IP_ADDRESS|||`, the OLMo tokenizer's PII placeholder. Hardcoding
    // <|endoftext|> would put a literal, unrelated token in every transcript.
    const olmoe = buildTuluChatPrompt(
      [{ role: 'user', content: 'Hi' }, { role: 'assistant', content: 'Hello!' }, { role: 'user', content: 'x' }],
      capture,
      { bosToken: '|||IP_ADDRESS|||', eosToken: '|||IP_ADDRESS|||' },
    ) as unknown as string[]
    expect(olmoe[0]).toBe(
      '|||IP_ADDRESS|||<|user|>\nHi\n<|assistant|>\nHello!|||IP_ADDRESS|||\n<|user|>\nx\n<|assistant|>\n',
    )
  })
})

// -------------------------------------------------- prefix stability
//
// tokenizer-bpe.ts:446 claims a re-rendered ChatML transcript makes each turn
// an exact extension of the previous one, and the engine's cross-turn KV reuse
// (engine-core.ts computeReuseStart) is built on it;
// scripts/prefix-stability-test.mjs measured it on nine real agent transcripts
// for chatml/llama3/deepseek. A template that re-renders past turns differently
// destroys that reuse silently, so every template added here is checked for it.
//
// This is the STRING-level property (the stub tokenizer is the identity), which
// is the half the builder controls; the token-level half is what the script
// measures against a real vocab, and this template is not in its TEMPLATES map
// yet. Unlike the Mistral family, nothing here moves between renders — the
// system turn is emitted in place — so the property holds with a system message
// too, which is the case chat.ts actually sends.

describe('prefix stability', () => {
  it('is append-only across turns, with and without a system message', () => {
    const chat: Msg[] = [
      { role: 'user', content: 'first' },
      { role: 'assistant', content: 'one' },
      { role: 'user', content: 'second' },
      { role: 'assistant', content: 'two' },
      { role: 'user', content: 'third' },
      { role: 'assistant', content: 'three' },
    ]
    for (const msgs of [chat, [{ role: 'system', content: 'You are terse.' } as Msg, ...chat]]) {
      // The prompt the engine sends at each assistant turn: every message up to
      // — not including — that reply. Same construction as
      // scripts/prefix-stability-test.mjs.
      const prompts: string[] = []
      for (let i = 0; i < msgs.length; i++) if (msgs[i].role === 'assistant') prompts.push(olmo2(msgs.slice(0, i))[0])
      expect(prompts.length).toBe(3)
      for (let k = 1; k < prompts.length; k++) expect(prompts[k].startsWith(prompts[k - 1])).toBe(true)
    }
  })
})
