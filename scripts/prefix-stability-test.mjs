#!/usr/bin/env node
// PREFIX-STABILITY — is an agent's prompt append-only in TOKEN IDS?
//
//   node scripts/prefix-stability-test.mjs <transcript.jsonl>
//       [--template chatml|llama3|deepseek] [--max 24000] [--mutate]
//   node scripts/prefix-stability-test.mjs --list
//
// This is the gate in docs/PAGING_PLAN.md §0.2, and it decides whether the
// prefix pool is worth five weeks. The pool can only reuse a prefix that is
// still a PREFIX, token for token. Everything about the engine's cross-turn
// reuse already assumes it — tokenizer-bpe.ts:446 says a re-rendered ChatML
// transcript "makes each turn's prompt an exact token-level extension of the
// previous turn's absorbed sequence" — but that is a comment, tested only
// against short synthetic chats.
//
// The reason it might be false: buildChatPromptFor renders the WHOLE
// transcript to one string and calls encode() once. BPE merges across
// boundaries. chat.ts:305-308 stores each assistant turn's exact emitted ids
// with a comment saying re-encoding the rendered text is not guaranteed to
// reproduce them — and chat.ts:505 re-encodes anyway. BENCH.md has the crack
// already: Qwen3.5 ChatML reuses 922/950 cleanly, Phi-3 1086/1105 with a
// documented boundary merge at `<|assistant|>\n`.
//
// Tool-call transcripts are where this should break if it breaks: JSON
// punctuation runs, file paths, digit runs, and trailing whitespace before a
// turn boundary are exactly the contexts where BPE re-segments.
//
// REAL TRANSCRIPTS ONLY, and they are NOT committed. This repo is public and a
// session transcript is the user's own work; token ids are the text. Point it
// at a local file. A synthetic fixture would pass regardless, which is the one
// outcome that teaches nothing.

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { homedir } from 'node:os'
import { fileURLToPath } from 'node:url'
import {
  createByteLevelTokenizer,
  buildChatPrompt as buildChatML,
  buildLlama3ChatPrompt,
  buildDeepSeekChatPrompt,
} from '../src/zero-tvm/tokenizer-bpe.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
// Byte-level BPE templates only. Phi-3's is SPM and lives in tokenizer.ts,
// which imports weight-loader and reads GPUBufferUsage at module scope — not
// importable here. That gap matters: BENCH.md records Phi-3 reusing 1086/1105
// with a boundary merge at `<|assistant|>\n`, so Phi-3 is the one template
// already known to break and the one this script cannot reach.
const tIdx = args.indexOf('--template')
const TEMPLATE = tIdx >= 0 ? args[tIdx + 1] : 'chatml'
const TEMPLATES = {
  chatml: { dir: 'Qwen3-4B-4bit', build: (m, t) => buildChatML(m, t, { thinking: false }) },
  llama3: { dir: 'Llama-3.2-1B-Instruct-4bit', build: buildLlama3ChatPrompt },
  deepseek: { dir: 'Qwen3-4B-4bit', build: buildDeepSeekChatPrompt },
}
const T = TEMPLATES[TEMPLATE]
if (!T) { console.log(`unknown --template ${TEMPLATE}; have ${Object.keys(TEMPLATES).join(', ')}`); process.exit(2) }
const TOK_DIR = T.dir

// ---------------------------------------------------------------- transcripts

/** Claude Code and pi both write JSONL, one record per line, but disagree on
 *  every field name. Both are read here because a single agent's habits are
 *  not evidence about agents. */
function readTranscript(path) {
  const out = []
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue
    let j
    try { j = JSON.parse(line) } catch { continue }

    // Claude Code: {type: 'user'|'assistant', message: {role, content}}
    const m = j.message
    if (m && (j.type === 'user' || j.type === 'assistant')) {
      const text = flatten(m.content)
      if (text) out.push({ role: j.type, content: text })
      continue
    }
    // pi: {type: 'message', role, content}
    if (j.type === 'message' && (j.role === 'user' || j.role === 'assistant')) {
      const text = flatten(j.content)
      if (text) out.push({ role: j.role, content: text })
    }
  }
  return out
}

/** Content blocks flattened the way a chat UI would render them. Tool calls and
 *  their results are KEPT, and rendered as JSON — they are the whole point.
 *  A transcript with the tool traffic stripped is a plain chat, and plain chats
 *  are already known to work. */
function flatten(content) {
  if (typeof content === 'string') return content
  if (!Array.isArray(content)) return ''
  const parts = []
  for (const b of content) {
    if (typeof b === 'string') { parts.push(b); continue }
    if (b?.type === 'text') parts.push(b.text ?? '')
    else if (b?.type === 'thinking') parts.push(b.thinking ?? '')
    else if (b?.type === 'tool_use') parts.push(`<tool_call>\n${JSON.stringify({ name: b.name, arguments: b.input })}\n</tool_call>`)
    else if (b?.type === 'tool_result') {
      const c = typeof b.content === 'string' ? b.content : flatten(b.content)
      parts.push(`<tool_response>\n${c}\n</tool_response>`)
    }
  }
  return parts.join('\n').trim()
}

/** Alternation the templates assume: merge consecutive same-role turns (a tool
 *  result arrives as a second `user` record) and drop a leading assistant. */
function normalize(msgs) {
  const out = []
  for (const m of msgs) {
    if (out.length === 0 && m.role === 'assistant') continue
    const last = out[out.length - 1]
    if (last && last.role === m.role) last.content += `\n${m.content}`
    else out.push({ ...m })
  }
  return out
}

if (args[0] === '--list' || args.length === 0) {
  const dirs = [join(homedir(), '.claude/projects'), join(homedir(), '.pi/agent/sessions')]
  const found = []
  const walk = (d, depth = 0) => {
    if (depth > 2 || !existsSync(d)) return
    for (const e of readdirSync(d)) {
      const p = join(d, e)
      const s = statSync(p)
      if (s.isDirectory()) walk(p, depth + 1)
      else if (e.endsWith('.jsonl')) found.push({ p, size: s.size })
    }
  }
  for (const d of dirs) walk(d)
  found.sort((a, b) => b.size - a.size)
  console.log('local agent transcripts, largest first (nothing here is committed):\n')
  for (const f of found.slice(0, 12)) console.log(`  ${(f.size / 1e6).toFixed(1).padStart(7)} MB  ${f.p}`)
  console.log('\nrun:  node scripts/prefix-stability-test.mjs <path>')
  process.exit(0)
}

// The positional argument, skipping any value that belongs to a flag.
const flagValues = new Set([TEMPLATE, args[args.indexOf('--max') + 1]])
const path = args.find((a) => !a.startsWith('--') && !flagValues.has(a))
if (!path || !existsSync(path)) {
  console.log(`SKIP  no transcript at ${path ?? '<none given>'} — run with --list to find one`)
  process.exit(0)
}
const tokPath = join(ROOT, '.weights-local', TOK_DIR, 'tokenizer.json')
if (!existsSync(tokPath)) {
  console.log(`SKIP  no tokenizer at ${tokPath}`)
  process.exit(0)
}

// ---------------------------------------------------------------------- run

const tok = createByteLevelTokenizer(JSON.parse(readFileSync(tokPath, 'utf8')))
const msgs = normalize(readTranscript(path))
if (msgs.length < 4) {
  console.log(`SKIP  only ${msgs.length} turns in ${path} — need at least 4`)
  process.exit(0)
}

// One prompt per assistant turn: what the engine would actually send at turn k,
// which is every message up to (not including) that assistant reply.
//
// Capped, and the cap is the workload rather than a shortcut. A 147 MB session
// transcript is millions of tokens; no engine here holds one (Qwen3.5's
// maxContext is 7168, Qwen3.6-35B's 6144). What an agent actually carries is a
// window, so stop at --max and report how much of the file that covered.
// Without the cap this is O(n^2) re-encodes and does not terminate.
const maxIdx = args.indexOf('--max')
const MAX = maxIdx >= 0 ? Number(args[maxIdx + 1]) : 24_000

// --mutate rewrites one token deep inside an early message, the way a clock or
// a re-rendered id in a system prompt would. It must turn this test RED. A
// stability test that has only ever passed cannot distinguish "append-only" from
// "not actually comparing anything" — the same trap PAGING_PLAN §0.4 sets for
// the page-table falsifier.
const MUTATE = args.includes('--mutate')

const prompts = []
let truncatedAt = null
let mutationApplied = false
for (let i = 0; i < msgs.length; i++) {
  if (msgs[i].role !== 'assistant') continue
  const view = msgs.slice(0, i)
  if (MUTATE && prompts.length >= 2) {
    // Turn 3 onward sees a different turn-1 — a few characters, deep in the
    // past, where a clock or a re-rendered id would sit. Unconditional: a
    // regex replace that silently matches nothing makes the falsifier pass,
    // which is the failure it exists to prevent (it did, the first time).
    view[0] = { ...view[0], content: `[MUTATED] ${view[0].content}` }
    mutationApplied = true
  }
  const ids = T.build(view, tok)
  if (ids.length > MAX) { truncatedAt = i; break }
  prompts.push({ turn: prompts.length + 1, ids })
}

const lcp = (a, b) => { let i = 0; while (i < a.length && i < b.length && a[i] === b[i]) i++; return i }

let breaks = 0
let firstBreak = null
let lostTotal = 0
for (let k = 1; k < prompts.length; k++) {
  const prev = prompts[k - 1].ids
  const cur = prompts[k].ids
  const shared = lcp(prev, cur)
  if (shared !== prev.length) {
    breaks++
    lostTotal += prev.length - shared
    if (!firstBreak) firstBreak = { turn: prompts[k].turn, shared, prevLen: prev.length, cur, prev }
  }
}

const longest = prompts[prompts.length - 1].ids.length
console.log(`\ntranscript  ${path}`)
console.log(`            ${msgs.length} turns -> ${prompts.length} prompts, longest ${longest} tokens`
  + ` (tokenizer ${TOK_DIR}, template ${TEMPLATE})`)
if (truncatedAt !== null) {
  console.log(`            stopped at turn ${truncatedAt}/${msgs.length}: the next prompt exceeds --max ${MAX}`)
}
if (prompts.length < 2) {
  console.log(`\nSKIP  fewer than 2 prompts under --max ${MAX} — nothing to compare`)
  process.exit(0)
}
// Self-evidencing: a PASS over a transcript with no tool traffic is a PASS over
// a plain chat, which is already known to work and is not what is being asked.
const window = msgs.slice(0, truncatedAt ?? msgs.length).map((m) => m.content).join('\n')
const nCalls = (window.match(/<tool_call>/g) ?? []).length
const nResults = (window.match(/<tool_response>/g) ?? []).length
console.log(`            ${nCalls} tool calls, ${nResults} tool results inside the compared window`)
if (nCalls === 0) console.log('            WARNING: no tool traffic here — this is a plain-chat result')
if (MUTATE) {
  console.log(`            --mutate: turn 1 rewritten from turn 3 onward (applied: ${mutationApplied}); this MUST fail`)
  if (!mutationApplied) { console.log('\nBROKEN  the mutation never applied — the run below proves nothing'); process.exit(2) }
}

// The number the pool actually lives on: how far a cached prefix survives.
// A single break at token 3000 of a 20000-token transcript caps the pool at
// 3000 no matter how many later turns are clean.
let reusableCeiling = prompts[0].ids.length
for (let k = 1; k < prompts.length; k++) {
  const shared = lcp(prompts[k - 1].ids, prompts[k].ids)
  reusableCeiling = Math.min(reusableCeiling === prompts[k - 1].ids.length ? prompts[k].ids.length : reusableCeiling,
    shared === prompts[k - 1].ids.length ? prompts[k].ids.length : shared)
}

if (breaks === 0) {
  console.log(`\nPASS  every turn extends the previous one exactly — ${prompts.length - 1} transitions, 0 breaks`)
  console.log(`      a pooled prefix survives the whole transcript (${longest} tokens)`)
} else {
  const f = firstBreak
  console.log(`\nFAIL  ${breaks}/${prompts.length - 1} transitions are NOT append-only`)
  console.log(`      first break at turn ${f.turn}: shared ${f.shared} of ${f.prevLen} tokens`
    + ` (lost ${f.prevLen - f.shared}); ${lostTotal} tokens lost across all breaks`)
  console.log(`      pooled prefix ceiling ~${reusableCeiling} tokens, not ${longest}`)
  const at = f.shared
  const show = (ids) => ids.slice(Math.max(0, at - 6), at + 6)
    .map((id, i) => `${Math.max(0, at - 6) + i === at ? '>' : ' '}${id}`).join(' ')
  console.log(`\n      prev: ${show(f.prev)}`)
  console.log(`      cur : ${show(f.cur)}`)
  console.log(`      text: ${JSON.stringify(tok.decode(f.prev.slice(Math.max(0, at - 6), at + 6)))}`)
}

// The second half of the gate: does re-encoding an assistant turn reproduce the
// ids the model emitted? The engine re-encodes rendered text (chat.ts:505) even
// though it stores the exact ids (chat.ts:305-308).
let roundTripFails = 0
for (const m of msgs) {
  if (m.role !== 'assistant') continue
  const ids = tok.encode(m.content)
  if (tok.decode(ids) !== m.content) roundTripFails++
}
console.log(`\n      decode(encode(x)) === x on ${msgs.filter((m) => m.role === 'assistant').length}`
  + ` assistant turns: ${roundTripFails === 0 ? 'all clean' : `${roundTripFails} FAILED`}`)

process.exit(breaks === 0 ? 0 : 1)
