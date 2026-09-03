// Generate reference fixtures for the hand-rolled byte-level BPE tokenizer
// (src/zero-tvm/tokenizer-bpe.ts) for any ChatML-generation checkpoint.
//
// Uses @huggingface/transformers (pure-JS BPE, runs in Node) to load the
// EXACT tokenizer the app ships and emits:
//
//   tests/tokenizer/fixtures-<tag>.json    {text, ids, decoded}[] reference cases
//   tests/tokenizer/tokenizer-<tag>.json   the app's own tokenizer.json,
//                                          minified (all fields kept)
//
// Both files are committed so `npm run test:unit` is fully offline.
//
//   node scripts/gen-tokenizer-fixtures-bpe.mjs --tag qwen35 \
//     --repo mlc-ai/Qwen3.5-4B-q4f16_1-MLC --generation qwen35
//   node scripts/gen-tokenizer-fixtures-bpe.mjs --tag qwen38 \
//     --repo mlx-community/Qwen3.8-27B-4bit --generation qwen38
//   # ... --local-json <path> reads tokenizer.json from disk instead of HF.
//
// The script cross-checks the per-generation chatPrompt() mirror below against
// the checkpoint's own template (apply_chat_template) and fails loudly on any
// drift — the mirror must encode the SAME generation rule the app implements,
// or the fixtures pin the wrong strings.

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, all) =>
    a.startsWith('--') ? [[a.slice(2), all[i + 1] ?? '']] : [],
  ),
)
const TAG = args.tag
const REPO = args.repo
const GENERATION = args.generation ?? 'qwen3'
const LOCAL_JSON = args['local-json'] ?? ''
// Some checkpoints (Qwen3.8) ship the template as a standalone
// chat_template.jinja instead of inside tokenizer_config.json —
// transformers.js then has no template to apply. Pass it explicitly.
const TEMPLATE_FILE = args['template-file'] ?? ''
if (!TAG || !REPO) {
  console.error('usage: gen-tokenizer-fixtures-bpe.mjs --tag <tag> --repo <hf-repo> [--generation qwen3|qwen35|qwen38] [--local-json <path>] [--template-file <chat_template.jinja>]')
  process.exit(2)
}
const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../tests/tokenizer')

// ---- mirror of buildChatPrompt() for one generation ----------------------
// Must stay identical to src/zero-tvm/tokenizer-bpe.ts (buildChatPrompt +
// lastQueryIndex + splitReasoning + lstripNl/rstripNl). Verified below
// against apply_chat_template, so drift fails here, not in the test.
const lstripNl = (s) => s.replace(/^\n+/, '')
const rstripNl = (s) => s.replace(/\n+$/, '')
function lastQueryIndex(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== 'user') continue
    const c = messages[i].content.trim()
    if (c.startsWith('<tool_response>') && c.endsWith('</tool_response>')) continue
    return i
  }
  return messages.length - 1
}
function splitReasoning(content) {
  if (!content.includes('</think>')) return { content, reasoning: '' }
  const parts = content.split('</think>')
  const before = parts[0]
  return {
    content: lstripNl(parts[parts.length - 1]),
    reasoning: lstripNl(rstripNl(before).split('<think>').pop() ?? ''),
  }
}
function chatPrompt(messages, { thinking = true, generation = GENERATION } = {}) {
  const trims = generation !== 'qwen3'
  // Mirrors withQwen38Instructions: qwen38 thinking mode prepends the default
  // (xhigh) effort preamble to the system turn, or synthesizes the turn.
  const QWEN38_INSTRUCTIONS =
    'Reasoning effort is set to xhigh. Please think carefully through the task, '
    + 'validate key assumptions, consider plausible alternatives, and prioritize '
    + 'correctness, consistency, and clarity in the final answer.'
  let msgs = messages
  if (generation === 'qwen38' && thinking && messages.length > 0) {
    const first = messages[0]
    msgs = first.role === 'system' && first.content.trim()
      ? [{ ...first, content: `${QWEN38_INSTRUCTIONS}\n\n${first.content.trim()}` }, ...messages.slice(1)]
      : [{ role: 'system', content: QWEN38_INSTRUCTIONS },
        ...(first.role === 'system' ? messages.slice(1) : messages)]
  }
  const q = lastQueryIndex(msgs)
  let text = ''
  for (let i = 0; i < msgs.length; i++) {
    const msg = msgs[i]
    const raw = trims ? msg.content.trim() : msg.content
    if (msg.role !== 'assistant') {
      text += `<|im_start|>${msg.role}\n${raw}<|im_end|>\n`
      continue
    }
    const split = generation === 'qwen38' ? { content: raw, reasoning: '' } : splitReasoning(raw)
    const inRound = i > q
    const think = generation === 'qwen38'
      || (inRound && (generation === 'qwen35' || i === msgs.length - 1 || !!split.reasoning))
    const reasoning = generation === 'qwen3' ? lstripNl(rstripNl(split.reasoning)) : split.reasoning.trim()
    const body = think
      ? `<think>\n${reasoning}\n</think>\n\n${generation === 'qwen3' ? lstripNl(split.content) : split.content}`
      : split.content
    text += `<|im_start|>${msg.role}\n${body}<|im_end|>\n`
  }
  text += '<|im_start|>assistant\n'
  if (!thinking) {
    text += '<think>\n\n</think>\n\n'
  } else if (generation !== 'qwen3') {
    // Mirrors buildChatPrompt: Qwen3.5+ opens the think block in thinking mode.
    text += '<think>\n'
  }
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
  // Past assistant turn carrying a </think> block — pins the per-generation
  // split/keep rule (qwen3 splits, qwen35 splits, qwen38 keeps whole).
  [
    { role: 'user', content: 'Think step by step.' },
    { role: 'assistant', content: '<think>\nfirst principles\n</think>\n\nParis.' },
    { role: 'user', content: 'And Spain?' },
  ],
  [{ role: 'user', content: 'multi\nline\n\nuser content with  spaces' }],
]

// Same adversarial battery as the qwen3 generator: the pretokenizer regex,
// NFC folding, byte alphabet, and special-token splitting are shared
// machinery, so the same strings pin them on every vocab.
const CASES = [
  'Hello world',
  'The quick brown fox jumps over the lazy dog.',
  "it's John's dog and we're sure they've left, I'll go, he'd agree, don't",
  "DON'T SHOUT, I'M HERE, YOU'RE LOUD, WE'VE HEARD, SHE'LL KNOW, HE'D SAY, IT'S FINE",
  'pi is 3.14159 and the answer is 42',
  '1234567890',
  'year 2026, price $1,234.56 or 1.000.000₺',
  ' leading space',
  'trailing space ',
  'multiple   spaces   between',
  '    four-space indent',
  'line one\nline two',
  'paragraph one\n\nparagraph two',
  'def f(x):\n    return x * 2\n',
  '```\ncode block\n\tindented line\n```',
  '\n',
  '\n\n',
  'tab\there',
  '\t\n mixed \r\n endings',
  'crlf\r\nline',
  'emoji 😀 test',
  'rockets 🚀🔥 and flags 🇹🇷',
  'family 👨‍👩‍👧‍👦 and pride 🏳️‍🌈',
  '中文分词测试',
  'こんにちは世界',
  '한국어 테스트',
  'Türkçe karakterler: ğüşıöçİĞÜŞÖÇ',
  "İstanbul'da yağmur yağıyor",
  'naïve café résumé',
  'café déjà vu',
  'wait... what?!?! (really); [yes] {no} <maybe>',
  '!!!???,,,;;;:::',
  'she said "hi" and \'bye\' — “curly” ‘quotes’',
  'MixedCASE camelCase snake_case kebab-case',
  '\u{1D54C}nicode math \u{1D54D}',
  'null \u0000 byte and bell \u0007',
  'zero\u200Bwidth\uFEFFjoiners',
  '<|endoftext|>',
  'before <|im_end|> after',
  '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n',
  '<think>\nreasoning here\n</think>\n\nanswer',
  '<tool_call>\n{"name": "f"}\n</tool_call>',
  'a',
  ' ',
  '',
  'Pneumonoultramicroscopicsilicovolcanoconiosis' + 'a'.repeat(200),
  chatPrompt(CHAT_MESSAGES[0]),
  chatPrompt(CHAT_MESSAGES[1]),
  chatPrompt(CHAT_MESSAGES[2]),
  chatPrompt(CHAT_MESSAGES[3]),
  chatPrompt(CHAT_MESSAGES[0], { thinking: false }),
]

async function main() {
  const { AutoTokenizer, env } = await import('@huggingface/transformers')
  env.allowLocalModels = false

  console.log(`loading reference tokenizer: ${REPO}`)
  const tok = await AutoTokenizer.from_pretrained(REPO)
  const templateArg = TEMPLATE_FILE ? { chat_template: readFileSync(TEMPLATE_FILE, 'utf8') } : {}
  if (TEMPLATE_FILE) console.log(`using standalone template: ${TEMPLATE_FILE}`)

  // Pin the per-generation mirror against the checkpoint's own template.
  for (const messages of CHAT_MESSAGES) {
    const expected = tok.apply_chat_template(messages, {
      tokenize: false,
      add_generation_prompt: true,
      ...templateArg,
    })
    const ours = chatPrompt(messages)
    if (ours !== expected) {
      throw new Error(
        `chat template drift [${GENERATION}]!\n--- apply_chat_template ---\n${JSON.stringify(expected)}\n--- chatPrompt mirror ---\n${JSON.stringify(ours)}`,
      )
    }
  }
  try {
    const expected = tok.apply_chat_template(CHAT_MESSAGES[0], {
      tokenize: false,
      add_generation_prompt: true,
      enable_thinking: false,
      ...templateArg,
    })
    const ours = chatPrompt(CHAT_MESSAGES[0], { thinking: false })
    if (ours !== expected) {
      throw new Error(
        `non-thinking template drift [${GENERATION}]!\n--- apply_chat_template ---\n${JSON.stringify(expected)}\n--- chatPrompt mirror ---\n${JSON.stringify(ours)}`,
      )
    }
    console.log(`chat template verified against apply_chat_template [${GENERATION}] (thinking + non-thinking)`)
  } catch (err) {
    if (String(err).includes('drift')) throw err
    console.log(`chat template verified (thinking) [${GENERATION}]; enable_thinking kwarg unsupported here:`, err.message)
  }

  const fixtures = CASES.map((text) => {
    const ids = tok.encode(text, { add_special_tokens: false })
    const decoded = ids.length > 0 ? tok.decode(ids, { skip_special_tokens: true }) : ''
    return { text, ids, decoded }
  })

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(resolve(OUT_DIR, `fixtures-${TAG}.json`), JSON.stringify(fixtures, null, 1) + '\n')
  console.log(`wrote ${fixtures.length} fixtures → tests/tokenizer/fixtures-${TAG}.json`)

  // The app's own tokenizer.json (minified, all fields kept), from disk when
  // --local-json points at the shipped checkout, else from the HF repo.
  let raw
  if (LOCAL_JSON) {
    raw = JSON.stringify(JSON.parse(readFileSync(LOCAL_JSON, 'utf8')))
    console.log(`read tokenizer.json from ${LOCAL_JSON}`)
  } else {
    const resp = await fetch(`https://huggingface.co/${REPO}/resolve/main/tokenizer.json`)
    if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching tokenizer.json`)
    raw = JSON.stringify(JSON.parse(await resp.text()))
  }
  writeFileSync(resolve(OUT_DIR, `tokenizer-${TAG}.json`), raw)
  console.log(`wrote tokenizer-${TAG}.json (${(raw.length / 1e6).toFixed(1)} MB)`)
}

main()
