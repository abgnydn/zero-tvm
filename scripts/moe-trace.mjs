#!/usr/bin/env node
// MOE-TRACE — what does the router ACTUALLY choose?
//
//   npm run dev                                        # weights mirror
//   node --experimental-strip-types scripts/moe-trace.mjs qwen30b --tokens 400
//
// Streaming experts from disk is the only technique that would put a 35B MoE on
// a machine that cannot hold it: 93% of that model is expert weights and only
// top-8 of 257 run per token. Whether it lands at a usable rate or crawls turns
// entirely on how SKEWED routing is — if a few experts take most tokens a small
// resident pool almost never misses; if routing is flat it misses constantly.
//
// Published numbers do not settle it. LRU recall on Mixtral is reported at
// 58.2%, and at least one paper finds LFU beats LRU because expert activation
// is skewed rather than time-local. Nobody has measured OUR models.
//
// So this records the ids and scripts/moe-replay.mjs scores cache policies over
// them offline. One day of measurement against 2-4 weeks of building, and this
// repo has already paid for that lesson once: MoE chunking Phase B looked
// obviously right and measured 0.39x.
//
// The engine writes the trace itself (`traceMoe`), copying moeIds straight
// after each layer's top-k, so the record is exactly what the expert matmul
// then read — not a reconstruction.

import { writeFileSync } from 'node:fs'
import { installShims } from './native/shims.mjs'

const model = process.argv[2] ?? 'qwen30b'
const arg = (k, d) => {
  const i = process.argv.indexOf(`--${k}`)
  return i > 0 ? Number(process.argv[i + 1]) : d
}
const TOKENS = arg('tokens', 400)
const OUT = process.argv.includes('--out')
  ? process.argv[process.argv.indexOf('--out') + 1]
  : `bench/quality/moe-trace-${model}.json`

await installShims({ unsafe: !process.argv.includes('--safe') })
const { createEngineRaw } = await import('../dist-lib/index.js')

const t0 = Date.now()
const { engine, tokenizer, spec, buildChatPromptFor } = await createEngineRaw({ model, traceMoe: true })
if (!spec.moe) throw new Error(`${model} is not a MoE spec — nothing to trace`)
console.log(`[boot] ${spec.id} in ${((Date.now() - t0) / 1000).toFixed(1)}s`)
console.log(`[spec] ${spec.layers} layers · ${spec.moe.experts} experts · top-${spec.moe.topK}`
  + `${spec.sharedExpertIndex >= 0 ? ' + shared' : ''} · ${spec.moeSlots} slots`)

// Varied prompts: routing that looks skewed on one topic and flat across many
// is not skew, it is a short trace. These span registers deliberately.
const PROMPTS = [
  'Write a function that merges two sorted arrays in Rust.',
  'Explain why the ocean is salty, for a ten year old.',
  'Translate to French: the meeting has been moved to Thursday afternoon.',
  'What are the tradeoffs between mutexes and channels for shared state?',
  'Summarise the causes of the 1929 crash in one paragraph.',
  'Write a haiku about a submarine.',
  'Given 3x + 7 = 22, solve for x and show the steps.',
  'Describe how a refrigerator moves heat against a temperature gradient.',
]

let generated = 0
for (const prompt of PROMPTS) {
  if (generated >= TOKENS) break
  engine.resetKVTracking()
  const ids = buildChatPromptFor(spec, [{ role: 'user', content: prompt }], tokenizer)
  const want = Math.min(Number(process.env.PER_PROMPT) || 80, TOKENS - generated)
  await engine.generatePipelined(ids, want, () => { generated++ }, () => false)
  process.stdout.write(`  ${generated} decode steps traced\r`)
}

// ONE readback at the end. The engine keeps a ring of decode steps, so every
// token's routing is still there — a per-token readback would have dominated
// the run and changed what it measured.
const t = await engine.readMoeTrace()
if (!t) throw new Error('readMoeTrace returned null — traceMoe did not take effect')

const steps = []
for (let i = 0; i < t.steps; i++) steps.push(Array.from(t.ids.subarray(i * t.stride, (i + 1) * t.stride)))

const out = {
  model, specId: spec.id,
  layers: spec.layers, experts: spec.moe.experts, topK: spec.moe.topK,
  slots: spec.moeSlots, sharedExpertIndex: spec.sharedExpertIndex,
  bits: spec.moe.bits ?? 4, d: spec.d, ffn: spec.ffn,
  steps: steps.length,
  note: 'one row per DECODE step, laid out [layer][slot]; prefill is excluded '
      + 'because a chunk touches essentially every expert and needs no cache',
  trace: steps,
}
writeFileSync(OUT, JSON.stringify(out))
console.log(`\nwrote ${OUT} — ${steps.length} decode steps x ${spec.layers} layers x ${spec.moeSlots} slots`)

await new Promise((r) => process.stdout.write('', r))
process.exit(0)
