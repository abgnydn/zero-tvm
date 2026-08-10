// WHICH CHAT TEMPLATE IS THIS? — the detector, against real vendor files.
//
// A wrong answer here does not error. The engine renders a plausible prompt
// with the wrong spacing or the wrong bos token, and the model answers slightly
// off distribution forever with nothing pointing back here. That is why the
// detector returns 'unknown' rather than guessing, and why this test exists.
//
// tests/fixtures/chat-template-detect.json holds each repo's `chat_template`
// and `bos_token` VERBATIM from its tokenizer_config.json (fetched 2026-08-10),
// so this runs offline. Expectations are the family each repo belongs to, which
// is a fact about the repo, not an output of the code under test.
//
// Two traps recorded here because the detector got both wrong first:
//
//   1. All three Mistral variants are '[INST]' templates and none of them says
//      which. They differ only in spacing. Detected on the tools block
//      (Ministral) and the exception text (Nemo vs v0.3).
//   2. OLMo-2 and OLMoE ship BYTE-IDENTICAL templates. The only difference is
//      the bos token — OLMoE's is `|||IP_ADDRESS|||`. Keying on "is bos null"
//      silently mapped OLMoE onto olmo2, which would have rendered
//      <|endoftext|> as its bos on every turn.

import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { detectChatTemplate } from '../../src/compiler/constraints.ts'

const fixture = JSON.parse(
  readFileSync(join(import.meta.dirname, '../fixtures/chat-template-detect.json'), 'utf8'),
) as {
  cases: { repo: string; expect: string; bos_token: unknown; chat_template: string }[]
}

const bosOf = (b: unknown): string | null =>
  typeof b === 'string' ? b
  : b && typeof b === 'object' ? ((b as { content?: string }).content ?? null)
  : null

describe('detectChatTemplate', () => {
  for (const c of fixture.cases) {
    it(`${c.repo.split('/')[1]} → ${c.expect}`, () => {
      expect(detectChatTemplate(c.chat_template, bosOf(c.bos_token))).toBe(c.expect)
    })
  }

  it('returns unknown for a template it does not recognise', () => {
    expect(detectChatTemplate('{{ messages[0].content }}', null)).toBe('unknown')
    expect(detectChatTemplate('', null)).toBe('unknown')
  })

  it('does not map an [INST] template it cannot place', () => {
    // A Mistral-shaped template with none of the three discriminators must NOT
    // inherit a spacing — that is the silent-wrong-prompt case.
    expect(detectChatTemplate('{% for m in messages %}[INST] {{m.content}} [/INST]{% endfor %}', '<s>'))
      .toBe('unknown')
  })

  it('does not map a Tulu-shaped template whose bos it does not know', () => {
    const tulu = fixture.cases.find((c) => c.expect === 'olmo2')!.chat_template
    expect(detectChatTemplate(tulu, '|||IP_ADDRESS|||')).toBe('unknown')
    expect(detectChatTemplate(tulu, '<|endoftext|>')).toBe('olmo2')
    expect(detectChatTemplate(tulu, null)).toBe('falcon3')
  })
})
