// DEEPSEEK CHAT TEMPLATE — our builder against the vendor's own jinja.
//
// The fixture is produced by rendering the chat_template shipped in
// DeepSeek-V2-Lite's tokenizer_config.json through jinja2, so this pins the
// builder to the template rather than to my reading of it. A chat template that
// is subtly wrong does not error — the model just answers slightly off
// distribution, forever, and nothing points back here.
//
// The tokenizer is a stub that returns the string it was handed. What is under
// test is the ASSEMBLY; byte-level BPE has its own suites.

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { buildDeepSeekChatPrompt } from '../../src/zero-tvm/tokenizer-bpe.ts'

const fx = JSON.parse(
  readFileSync(join(import.meta.dirname, '../fixtures/deepseek-chat-template.json'), 'utf8'),
) as {
  bos: string
  eos: string
  cases: Record<string, Array<{ role: 'system' | 'user' | 'assistant'; content: string }>>
  rendered: Record<string, string>
}

/** Hands back the text instead of ids, so the assertion reads as the prompt. */
const capture = { encode: (t: string) => [t] as unknown as number[] }
const render = (msgs: Parameters<typeof buildDeepSeekChatPrompt>[0]) =>
  buildDeepSeekChatPrompt(msgs, capture, { bosToken: fx.bos, eosToken: fx.eos }) as unknown as string[]

describe('deepseek chat template', () => {
  for (const name of Object.keys(fx.cases)) {
    it(`renders ${name} exactly as the vendor jinja does`, () => {
      expect(render(fx.cases[name])[0]).toBe(fx.rendered[name])
    })
  }

  it('emits a system message with no label', () => {
    // The template writes message['content'] alone for role 'system' — no
    // "System:" prefix, unlike both other roles. Inventing one would be a
    // reasonable guess and wrong.
    const out = render([{ role: 'system', content: 'RULE' }])[0]
    expect(out).toBe(`${fx.bos}RULE\n\nAssistant:`)
    expect(out).not.toContain('System:')
  })

  it('opens the turn with "Assistant:" and no trailing space', () => {
    // The model produces the leading space itself; adding one here puts its
    // very first token off distribution.
    const out = render([{ role: 'user', content: 'x' }])[0]
    expect(out.endsWith('Assistant:')).toBe(true)
    expect(out.endsWith('Assistant: ')).toBe(false)
  })

  it('does not trim content', () => {
    // Llama-3's template trims, this one interpolates verbatim — copying the
    // neighbouring builder would silently eat the user's whitespace.
    expect(render([{ role: 'user', content: '  padded  ' }])[0])
      .toBe(`${fx.bos}User:   padded  \n\nAssistant:`)
  })
})
