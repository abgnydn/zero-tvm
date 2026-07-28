// Generate reference fixtures for the hand-rolled Phi-3 tokenizer.
//
// Uses @huggingface/transformers (pure-JS BPE, runs in Node) to load the
// EXACT tokenizer the app ships (mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC —
// same repo weight-loader.ts points at) and emits:
//
//   tests/tokenizer/fixtures.json    {text, ids, decoded}[] reference cases
//   tests/tokenizer/tokenizer.json   the app's own vocab/merges (~1.8 MB, MIT)
//
// Both files are committed so `npm run test:unit` is fully offline.
// Re-run with `npm run gen:tokenizer-fixtures` (needs network).

import { writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const REPO = 'mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC'
const BASE = `https://huggingface.co/${REPO}/resolve/main/`
const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../tests/tokenizer')

// Mirror of buildChatPrompt() in src/zero-tvm/tokenizer.ts — the fixtures
// must cover the exact strings the app feeds to encode().
function chatPrompt(messages) {
  let text = ''
  for (const msg of messages) {
    text += `<|${msg.role}|>\n${msg.content}<|end|>\n`
  }
  text += '<|assistant|>\n'
  return text
}

const CASES = [
  // Plain English
  'Hello world',
  'The quick brown fox jumps over the lazy dog.',
  // Leading / trailing / multiple spaces
  ' leading space',
  'trailing space ',
  'multiple   spaces   between',
  '    four-space indent',
  // Newlines
  'line one\nline two',
  'paragraph one\n\nparagraph two',
  'def f(x):\n    return x * 2\n',
  '```\ncode block\n\tindented line\n```',
  '\n',
  '\n\n',
  // Tabs and mixed whitespace
  'tab\there',
  '\t\n mixed \r\n endings',
  // Emoji (multi-byte, byte-fallback territory)
  'emoji 😀 test',
  'rockets 🚀🔥 and flags 🇹🇷',
  // CJK
  '中文分词测试',
  'こんにちは世界',
  // Turkish
  'Türkçe karakterler: ğüşıöçİĞÜŞÖÇ',
  'İstanbul\'da yağmur yağıyor',
  // Accents / NFC vs decomposed-ish input
  'naïve café résumé',
  // Punctuation runs
  'wait... what?!?! (really); [yes] {no} <maybe>',
  '!!!???,,,;;;:::',
  // Numbers
  'pi is 3.14159 and the answer is 42',
  // Identifiers
  'MixedCASE camelCase snake_case kebab-case',
  // Quotes
  'she said "hi" and \'bye\'',
  // Byte-fallback edge cases
  '\u{1D54C}nicode math \u{1D54D}',
  'null \u0000 byte and bell \u0007',
  // Special tokens embedded in text
  '<s>bos then text</s>',
  'before <|end|> after',
  '<|user|>\nHi<|end|>\n<|assistant|>\n',
  // Single char / empty-ish
  'a',
  ' ',
  '',
  // The exact chat-template strings buildChatPrompt emits
  chatPrompt([
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is the capital of France?' },
  ]),
  chatPrompt([
    { role: 'user', content: 'Write a haiku about GPUs.' },
    { role: 'assistant', content: 'Warm silicon hums' },
    { role: 'user', content: 'Continue it' },
  ]),
  chatPrompt([{ role: 'user', content: 'multi\nline\n\nuser content with  spaces' }]),
]

async function main() {
  const { AutoTokenizer, env } = await import('@huggingface/transformers')
  env.allowLocalModels = false

  console.log(`loading reference tokenizer: ${REPO}`)
  const tok = await AutoTokenizer.from_pretrained(REPO)

  const fixtures = CASES.map((text) => {
    const ids = tok.encode(text, { add_special_tokens: false })
    const decoded = ids.length > 0 ? tok.decode(ids, { skip_special_tokens: true }) : ''
    return { text, ids, decoded }
  })

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(resolve(OUT_DIR, 'fixtures.json'), JSON.stringify(fixtures, null, 1) + '\n')
  console.log(`wrote ${fixtures.length} fixtures → tests/tokenizer/fixtures.json`)

  // The app's own tokenizer.json (vocab + merges), so the unit test builds
  // the tokenizer from the same data the app uses — offline.
  const resp = await fetch(BASE + 'tokenizer.json')
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching tokenizer.json`)
  const raw = await resp.text()
  writeFileSync(resolve(OUT_DIR, 'tokenizer.json'), raw)
  console.log(`wrote tokenizer.json (${(raw.length / 1e6).toFixed(1)} MB) → tests/tokenizer/tokenizer.json`)
}

main()
