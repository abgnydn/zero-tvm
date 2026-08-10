// MISTRAL `[INST]` CHAT TEMPLATES — our builder against the vendor's own jinja.
//
// Three checkpoints in one family ship three DIFFERENT spacings. Getting one
// wrong does not error — the model just answers slightly off distribution,
// forever, and nothing points back here. So every expected string below was
// derived BY HAND from the chat_template in that repo's own
// tokenizer_config.json (fetched 2026-08-10 from
// huggingface.co/mlx-community/<repo>/resolve/main/tokenizer_config.json) and
// then cross-checked by rendering that same vendor template through jinja2 —
// the method behind tests/fixtures/deepseek-chat-template.json. None of these
// strings came out of the code under test.
//
// The relevant jinja is quoted above each block so the derivation can be redone
// without the network. Unlike the DeepSeek suite these live inline rather than
// in a fixture: three templates × a handful of cases is small, and having the
// spacing sit next to the jinja it came from is the whole point.
//
// The tokenizer is a stub that returns the string it was handed — what is under
// test is the ASSEMBLY. (The dispatch in model-select.ts cannot be reached from
// a unit test at all: it imports tokenizer.ts → weight-loader.ts, which reads
// GPUBufferUsage at module scope.)

import { describe, expect, it } from 'vitest'
import { buildMistralChatPrompt } from '../../src/zero-tvm/tokenizer-bpe.ts'

type Msg = { role: 'system' | 'user' | 'assistant'; content: string }

/** Hands back the text instead of ids, so the assertion reads as the prompt. */
const capture = { encode: (t: string) => [t] as unknown as number[] }
const render = (variant: 'v1' | 'nemo' | 'ministral') => (msgs: Msg[]) =>
  buildMistralChatPrompt(msgs, capture, { variant }) as unknown as string[]

const MULTI_TURN: Msg[] = [
  { role: 'user', content: 'Hi' },
  { role: 'assistant', content: 'Hello!' },
  { role: 'user', content: '  padded  ' },
]

// ------------------------------------------------------------------ v1
//
// mlx-community/Mistral-7B-Instruct-v0.3-4bit, chat_template[name="default"]:
//
//   {{bos_token}}{% for message in messages %}
//     {% if message['role'] == 'user' %}
//       {{ ' [INST] ' + message['content'] + ' [/INST]' }}
//     {% elif message['role'] == 'assistant' %}
//       {{ ' ' + message['content'] + ' ' + eos_token}}
//     {% else %}{{ raise_exception('Only user and assistant roles are supported!') }}
//   {% endfor %}

describe('mistral v1 chat template (Mistral-7B-Instruct-v0.3)', () => {
  const r = render('v1')

  it('brackets the user turn with a space on BOTH sides of the content', () => {
    // ' [INST] ' + content + ' [/INST]' — a leading space before [INST] too,
    // which the two later templates dropped.
    expect(r([{ role: 'user', content: 'Hi' }])[0]).toBe('<s> [INST] Hi [/INST]')
  })

  it('renders a full turn pair verbatim, with no trimming', () => {
    // assistant is ' ' + content + ' ' + eos — the space BEFORE </s> is unique
    // to this variant. The user content keeps its padding, so "[INST]" is
    // followed by three spaces (one from the template, two from the content).
    expect(r(MULTI_TURN)[0]).toBe('<s> [INST] Hi [/INST] Hello! </s> [INST]   padded   [/INST]')
  })

  it('does not trim the assistant content either', () => {
    expect(r([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: '  Hello!  ' },
      { role: 'user', content: 'x' },
    ])[0]).toBe('<s> [INST] Hi [/INST]   Hello!   </s> [INST] x [/INST]')
  })

  it('folds a system message into the FIRST user turn', () => {
    // This template raises on a system role. The fold is Mistral's own answer:
    // mistral_common InstructTokenizerV1.encode_user_message does
    // `if is_first and system_prompt: content = system_prompt + "\n\n" + content`.
    expect(r([{ role: 'system', content: 'You are terse.' }, { role: 'user', content: 'Hi' }])[0])
      .toBe('<s> [INST] You are terse.\n\nHi [/INST]')
  })

  it('ends after [/INST] — this family has no generation prompt', () => {
    // Every other template in this engine appends an assistant header. These
    // do not: `add_generation_prompt` appears nowhere in the jinja, and the
    // model continues straight from the closing tag.
    const out = r(MULTI_TURN)[0]
    expect(out.endsWith('[/INST]')).toBe(true)
  })
})

// ---------------------------------------------------------------- nemo
//
// mlx-community/Mistral-Nemo-Instruct-2407-4bit, chat_template:
//
//   {%- if messages[0]['role'] == 'system' %}
//       {%- set system_message = messages[0]['content'] %}
//       {%- set loop_messages = messages[1:] %}
//   {%- endif %}
//   {{- bos_token }}
//   {%- for message in loop_messages %}
//       {%- if message['role'] == 'user' %}
//           {%- if loop.last and system_message is defined %}
//               {{- '[INST] ' + system_message + '\n\n' + message['content'] + '[/INST]' }}
//           {%- else %}
//               {{- '[INST] ' + message['content'] + '[/INST]' }}
//       {%- elif message['role'] == 'assistant' %}
//           {{- ' ' + message['content'] + eos_token}}
//   {%- endfor %}

describe('mistral nemo chat template (Mistral-Nemo-Instruct-2407)', () => {
  const r = render('nemo')

  it('drops the leading space and the one before [/INST]', () => {
    expect(r([{ role: 'user', content: 'Hi' }])[0]).toBe('<s>[INST] Hi[/INST]')
  })

  it('renders a full turn pair verbatim', () => {
    // assistant is ' ' + content + eos — no space before </s>, unlike v1.
    expect(r(MULTI_TURN)[0]).toBe('<s>[INST] Hi[/INST] Hello!</s>[INST]   padded  [/INST]')
  })

  it('folds a system message into the LAST user turn', () => {
    expect(r([{ role: 'system', content: 'You are terse.' }, { role: 'user', content: 'Hi' }])[0])
      .toBe('<s>[INST] You are terse.\n\nHi[/INST]')
    expect(r([
      { role: 'system', content: 'You are terse.' },
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: 'Hello!' },
      { role: 'user', content: 'Again' },
    ])[0]).toBe('<s>[INST] Hi[/INST] Hello!</s>[INST] You are terse.\n\nAgain[/INST]')
  })
})

// ----------------------------------------------------------- ministral
//
// mlx-community/Ministral-8B-Instruct-2410-4bit, chat_template:
//
//   {%- if loop.last and system_message is defined %}
//       {{- "[INST]" + system_message + "\n\n" + message["content"] + "[/INST]" }}
//   {%- else %}
//       {{- "[INST]" + message["content"] + "[/INST]" }}
//   {%- elif message["role"] == "assistant" %}
//       {{- " " + message["content"]|trim + eos_token}}

describe('ministral chat template (Ministral-8B-Instruct-2410)', () => {
  const r = render('ministral')

  it('puts no space inside [INST] at all', () => {
    expect(r([{ role: 'user', content: 'Hi' }])[0]).toBe('<s>[INST]Hi[/INST]')
  })

  it('trims the ASSISTANT content and only the assistant content', () => {
    // `message["content"]|trim` on the assistant branch; the user branch has
    // no filter, so its padding survives. This is the only one of the three
    // that trims anything.
    expect(r(MULTI_TURN)[0]).toBe('<s>[INST]Hi[/INST] Hello!</s>[INST]  padded  [/INST]')
    expect(r([
      { role: 'user', content: 'Hi' },
      { role: 'assistant', content: '  Hello!  ' },
      { role: 'user', content: 'x' },
    ])[0]).toBe('<s>[INST]Hi[/INST] Hello!</s>[INST]x[/INST]')
  })

  it('folds a system message into the LAST user turn', () => {
    expect(r([{ role: 'system', content: 'You are terse.' }, { role: 'user', content: 'Hi' }])[0])
      .toBe('<s>[INST]You are terse.\n\nHi[/INST]')
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
// measures against a real vocab, and these two templates are not in its
// TEMPLATES map yet.

/** The prompt the engine sends at each assistant turn: every message up to —
 *  not including — that reply. Same construction as prefix-stability-test.mjs. */
function turnPrompts(msgs: Msg[], build: (m: Msg[]) => string[]): string[] {
  const out: string[] = []
  for (let i = 0; i < msgs.length; i++) if (msgs[i].role === 'assistant') out.push(build(msgs.slice(0, i))[0])
  return out
}

const CHAT: Msg[] = [
  { role: 'user', content: 'first' },
  { role: 'assistant', content: 'one' },
  { role: 'user', content: 'second' },
  { role: 'assistant', content: 'two' },
  { role: 'user', content: 'third' },
  { role: 'assistant', content: 'three' },
]
const SYSTEM: Msg = { role: 'system', content: 'You are terse.' }

describe('prefix stability', () => {
  it('v1 is append-only across turns, with or without a system message', () => {
    for (const msgs of [CHAT, [SYSTEM, ...CHAT]]) {
      const prompts = turnPrompts(msgs, render('v1'))
      expect(prompts.length).toBe(3)
      for (let k = 1; k < prompts.length; k++) expect(prompts[k].startsWith(prompts[k - 1])).toBe(true)
    }
  })

  it('nemo and ministral are append-only ONLY without a system message', () => {
    for (const variant of ['nemo', 'ministral'] as const) {
      const prompts = turnPrompts(CHAT, render(variant))
      for (let k = 1; k < prompts.length; k++) expect(prompts[k].startsWith(prompts[k - 1])).toBe(true)
    }
  })

  it('nemo moves the system prompt to the newest turn, breaking reuse', () => {
    // Not a bug to fix — it is the vendor's template. It is asserted so the
    // cost is recorded: chat.ts always sends a system message, so a 2407/2410
    // model re-prefills from the first turn every time. The shared prefix is
    // "<s>[INST] " and nothing more.
    const prompts = turnPrompts([SYSTEM, ...CHAT], render('nemo'))
    expect(prompts[1].startsWith(prompts[0])).toBe(false)
    let shared = 0
    while (prompts[0][shared] === prompts[1][shared]) shared++
    expect(prompts[0].slice(0, shared)).toBe('<s>[INST] ')
  })
})
