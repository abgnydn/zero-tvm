// Generate reference fixtures for the hand-rolled Qwen3 byte-level BPE
// tokenizer (src/zero-tvm/tokenizer-bpe.ts).
//
// Uses @huggingface/transformers (pure-JS BPE, runs in Node) to load the
// EXACT tokenizer the app ships (mlc-ai/Qwen3-4B-q4f16_1-MLC) and emits:
//
//   tests/tokenizer/fixtures-qwen3.json    {text, ids, decoded}[] reference cases
//   tests/tokenizer/tokenizer-qwen3.json   the app's own tokenizer.json,
//                                          minified (~5.4 MB; the HF original
//                                          is ~11 MB pretty-printed — same
//                                          data, all fields kept, Apache-2.0)
//
// Both files are committed so `npm run test:unit` is fully offline.
// Re-run with `npm run gen:tokenizer-fixtures-qwen` (needs network).
//
// The script also cross-checks the buildChatPrompt() template mirror against
// tokenizer.apply_chat_template() — the ground truth from
// tokenizer_config.json — and fails loudly on any drift.

import { writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const REPO = 'mlc-ai/Qwen3-4B-q4f16_1-MLC'
const BASE = `https://huggingface.co/${REPO}/resolve/main/`
const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../tests/tokenizer')

// Mirror of buildChatPrompt() in src/zero-tvm/tokenizer-bpe.ts — the fixtures
// must cover the exact strings the app feeds to encode(). Thinking is ON by
// default (no suffix); thinking: false appends the chat_template's
// `enable_thinking is false` branch verbatim.
function chatPrompt(messages, { thinking = true } = {}) {
  let text = ''
  for (const msg of messages) {
    text += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`
  }
  text += '<|im_start|>assistant\n'
  if (!thinking) text += '<think>\n\n</think>\n\n'
  return text
}

const CHAT_MESSAGES = [
  [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' },
  ],
  [
    { role: 'user', content: 'Write a haiku about GPUs.' },
    { role: 'assistant', content: 'Warm silicon hums' },
    { role: 'user', content: 'Continue it' },
  ],
  [{ role: 'user', content: 'multi\nline\n\nuser content with  spaces' }],
]

const CASES = [
  // Plain English
  'Hello world',
  'The quick brown fox jumps over the lazy dog.',
  // Contractions — the (?i:'s|'t|'re|'ve|'m|'ll|'d) branch, incl. uppercase
  "it's John's dog and we're sure they've left, I'll go, he'd agree, don't",
  "DON'T SHOUT, I'M HERE, YOU'RE LOUD, WE'VE HEARD, SHE'LL KNOW, HE'D SAY, IT'S FINE",
  // Numbers — \p{N} splits every digit into its own pre-token
  'pi is 3.14159 and the answer is 42',
  '1234567890',
  'year 2026, price $1,234.56 or 1.000.000₺',
  // Leading / trailing / multiple spaces (the \s+(?!\S) branch)
  ' leading space',
  'trailing space ',
  'multiple   spaces   between',
  '    four-space indent',
  // Newlines (the \s*[\r\n]+ branch), tabs, code
  'line one\nline two',
  'paragraph one\n\nparagraph two',
  'def f(x):\n    return x * 2\n',
  '```\ncode block\n\tindented line\n```',
  '\n',
  '\n\n',
  'tab\there',
  '\t\n mixed \r\n endings',
  'crlf\r\nline',
  // Emoji (multi-byte, split across byte-level tokens) + ZWJ sequences
  'emoji 😀 test',
  'rockets 🚀🔥 and flags 🇹🇷',
  'family 👨‍👩‍👧‍👦 and pride 🏳️‍🌈',
  // CJK
  '中文分词测试',
  'こんにちは世界',
  '한국어 테스트',
  // Turkish
  'Türkçe karakterler: ğüşıöçİĞÜŞÖÇ',
  "İstanbul'da yağmur yağıyor",
  // Accents; the decomposed form pins the NFC normalizer (e + U+0301)
  'naïve café résumé',
  'café déjà vu',
  // Punctuation runs (the ` ?[^\s\p{L}\p{N}]+[\r\n]*` branch)
  'wait... what?!?! (really); [yes] {no} <maybe>',
  '!!!???,,,;;;:::',
  'she said "hi" and \'bye\' — “curly” ‘quotes’',
  // Identifiers
  'MixedCASE camelCase snake_case kebab-case',
  // Byte-level exotica (no byte-fallback tokens — raw UTF-8 bytes via Ġ-map)
  '\u{1D54C}nicode math \u{1D54D}',
  'null \u0000 byte and bell \u0007',
  'zero\u200Bwidth\uFEFFjoiners',
  // Special tokens embedded in text (+ non-special added tokens like <think>)
  '<|endoftext|>',
  'before <|im_end|> after',
  '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n',
  '<think>\nreasoning here\n</think>\n\nanswer',
  '<tool_call>\n{"name": "f"}\n</tool_call>',
  // Single char / empty-ish
  'a',
  ' ',
  '',
  // Very long word (single pre-token, deep BPE merge chain)
  'Pneumonoultramicroscopicsilicovolcanoconiosis' + 'a'.repeat(200),
  // The exact chat-template strings buildChatPrompt emits
  chatPrompt(CHAT_MESSAGES[0]),
  chatPrompt(CHAT_MESSAGES[1]),
  chatPrompt(CHAT_MESSAGES[2]),
  chatPrompt(CHAT_MESSAGES[0], { thinking: false }),
]

async function main() {
  const { AutoTokenizer, env } = await import('@huggingface/transformers')
  env.allowLocalModels = false

  console.log(`loading reference tokenizer: ${REPO}`)
  const tok = await AutoTokenizer.from_pretrained(REPO)

  // Pin the chat template against the real tokenizer_config.json chat_template
  // (via minja) before trusting the mirrored strings in CASES.
  for (const messages of CHAT_MESSAGES) {
    const expected = tok.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
    })
    const ours = chatPrompt(messages)
    if (ours !== expected) {
      throw new Error(
        `chat template drift!\n--- apply_chat_template ---\n${JSON.stringify(expected)}\n--- chatPrompt mirror ---\n${JSON.stringify(ours)}`,
      )
    }
  }
  try {
    const expected = tok.apply_chat_template(CHAT_MESSAGES[0], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
    })
    const ours = chatPrompt(CHAT_MESSAGES[0], { thinking: false })
    if (ours !== expected) {
      throw new Error(
        `non-thinking template drift!\n--- apply_chat_template ---\n${JSON.stringify(expected)}\n--- chatPrompt mirror ---\n${JSON.stringify(ours)}`,
      )
    }
    console.log('chat template verified against apply_chat_template (thinking + non-thinking)')
  } catch (err) {
    if (String(err).includes('drift')) throw err
    // Older transformers.js may not forward extra template kwargs.
    console.log('chat template verified (thinking); enable_thinking kwarg unsupported here:', err.message)
  }

  const fixtures = CASES.map((text) => {
    const ids = tok.encode(text, { add_special_tokens: false })
    const decoded = ids.length > 0 ? tok.decode(ids, { skip_special_tokens: true }) : ''
    return { text, ids, decoded }
  })

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(resolve(OUT_DIR, 'fixtures-qwen3.json'), JSON.stringify(fixtures, null, 1) + '\n')
  console.log(`wrote ${fixtures.length} fixtures → tests/tokenizer/fixtures-qwen3.json`)

  // The app's own tokenizer.json (full pipeline + vocab + merges), so the
  // unit test builds the tokenizer from the same data the app uses — offline.
  // Minified only (HF pretty-prints it to ~11 MB); every field is kept.
  const resp = await fetch(BASE + 'tokenizer.json')
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching tokenizer.json`)
  const raw = JSON.stringify(JSON.parse(await resp.text()))
  writeFileSync(resolve(OUT_DIR, 'tokenizer-qwen3.json'), raw)
  console.log(`wrote tokenizer-qwen3.json (${(raw.length / 1e6).toFixed(1)} MB) → tests/tokenizer/tokenizer-qwen3.json`)
}

main()
