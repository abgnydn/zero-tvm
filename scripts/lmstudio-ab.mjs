#!/usr/bin/env node
// LMSTUDIO-AB — zero-tvm (browser WebGPU) vs LM Studio (MLX native), SAME
// machine, SAME session, interleaved runs, all runs reported.
//
//   node scripts/lmstudio-ab.mjs        # expects LM Studio serving on :1234
//
// Same model family and size both sides: Qwen3.5-4B. NOT the same bytes —
// ours is the MLC q4f16_1 checkpoint, LM Studio's the MLX 4-bit. Same
// architecture, same parameter count, both 4-bit; quantization details
// differ, and the caveat rides every number this prints.
//
// The prompt is ~800 tokens of THIS REPO's own documentation — in-domain for
// the agentic target and identical text to both sides. Thinking is disabled
// on the LM Studio side (reasoning_effort: none — anything else burns the
// decode budget on a <think> block and the comparison dies).

import { readFileSync } from 'node:fs'
import { resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const LMS = 'http://127.0.0.1:1234'
const RUNS = 3
const DECODE = 128

const text = readFileSync(`${ROOT}/CLAUDE.md`, 'utf8').slice(0, 3200)
const userMsg = `Summarise the following notes in one paragraph:\n\n${text}`

async function lmstudio() {
  const t0 = performance.now()
  const res = await fetch(`${LMS}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: 'small', stream: true, max_tokens: DECODE, temperature: 0,
      reasoning_effort: 'none',
      stream_options: { include_usage: true },
      messages: [{ role: 'user', content: userMsg }],
    }),
  })
  let tFirst = 0, nOut = 0, usage = null, buf = ''
  const dec = new TextDecoder()
  for await (const part of res.body) {
    buf += dec.decode(part, { stream: true })
    const lines = buf.split('\n'); buf = lines.pop() ?? ''
    for (const line of lines) {
      if (!line.startsWith('data: ') || line.includes('[DONE]')) continue
      const j = JSON.parse(line.slice(6))
      if (j.usage) usage = j.usage
      const d = j.choices?.[0]?.delta?.content
      if (d) { if (!tFirst) tFirst = performance.now(); nOut++ }
    }
  }
  const tEnd = performance.now()
  const pt = usage?.prompt_tokens ?? 0
  const ct = usage?.completion_tokens ?? nOut
  return {
    promptTokens: pt, tokens: ct,
    ttftMs: tFirst - t0,
    prefillTokS: pt > 1 ? (pt - 1) / ((tFirst - t0) / 1000) : 0,
    decodeTokS: ct > 1 ? (ct - 1) / ((tEnd - tFirst) / 1000) : 0,
  }
}

const fmt = (r) => `prompt ${String(r.promptTokens).padStart(4)}  ttft ${(r.ttftMs / 1000).toFixed(2)}s`
  + `  prefill ${r.prefillTokS.toFixed(0).padStart(4)} tok/s  decode ${r.decodeTokS.toFixed(1).padStart(6)} tok/s`

await startHarness()
try {
  const page = await newPage('/model-smoke.html?model=qwen35')
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  // Our prompt ids: the SAME text through our own ChatML template.
  const ids = await page.evaluate(async (msg) => {
    const { loadByteLevelTokenizer, buildChatPrompt } = await import('/src/zero-tvm/tokenizer-bpe.ts')
    const tok = await loadByteLevelTokenizer()
    window.__tok = tok
    return buildChatPrompt([{ role: 'user', content: msg }], tok, { thinking: false })
  }, userMsg)
  console.log(`prompt: ours ${ids.length} ids (theirs reports its own count)\n`)

  // Warm both once (uncounted): first run pays shader warm-up / cache fill.
  await page.evaluate((i) => window.__timedGenerateChat(i, 8), ids)
  await lmstudio()

  const ours = [], theirs = []
  for (let r = 0; r < RUNS; r++) {
    const a = await page.evaluate((i, n) => window.__timedGenerateChat(i, n), ids, DECODE)
    ours.push(a)
    console.log(`  zero-tvm  ${fmt(a)}`)
    const b = await lmstudio()
    theirs.push(b)
    console.log(`  lmstudio  ${fmt(b)}`)
  }
  const med = (xs, k) => [...xs].map((x) => x[k]).sort((a, b) => a - b)[Math.floor(xs.length / 2)]
  console.log(`\nmedians of ${RUNS} interleaved runs:`)
  console.log(`  zero-tvm   prefill ${med(ours, 'prefillTokS').toFixed(0)} tok/s   decode ${med(ours, 'decodeTokS').toFixed(1)} tok/s`)
  console.log(`  lmstudio   prefill ${med(theirs, 'prefillTokS').toFixed(0)} tok/s   decode ${med(theirs, 'decodeTokS').toFixed(1)} tok/s`)
  console.log(`  ratio      prefill ${(med(ours, 'prefillTokS') / med(theirs, 'prefillTokS')).toFixed(2)}x        decode ${(med(ours, 'decodeTokS') / med(theirs, 'decodeTokS')).toFixed(2)}x`)
  console.log(`\ncaveats: different checkpoints (MLC q4f16_1 vs MLX 4-bit), browser tab focused,`)
  console.log(`one machine, one day. Context: ours 262,144 max (?ctx=), theirs fitted 198,400.`)
} finally {
  await stopHarness()
}
