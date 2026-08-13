#!/usr/bin/env node
// LMSTUDIO-AB — zero-tvm (browser WebGPU) vs LM Studio (MLX native), SAME
// machine, SAME session, interleaved runs, all runs reported.
//
//   node scripts/lmstudio-ab.mjs                    # browser arm, qwen35
//   MODEL=qwen3mlx node --experimental-strip-types \
//     scripts/lmstudio-ab.mjs --native               # native arm, same bytes
//
// Default (qwen35): same family and size both sides but NOT the same bytes —
// ours MLC q4f16_1, theirs MLX 4-bit. That caveat rides every number.
//
// --native removes it. Our native host reads .weights-local/<repo-tail>, and
// symlinking that same directory into ~/.lmstudio/models/mlx-community/ makes
// LM Studio load the IDENTICAL FILES (verified by byte count both sides), so
// the only difference left is the runtime. It also swaps our arm from the
// browser to dawn.node, which is the faster of our two hosts.
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
// LM Studio does not necessarily listen on loopback. With "serve on local
// network" on it binds ONE interface — here a Tailscale address — and
// 127.0.0.1 is refused outright. `lsof -nP -iTCP:1234 -sTCP:LISTEN` prints the
// real one; LMS_URL overrides.
const LMS = process.env.LMS_URL || 'http://127.0.0.1:1234'
const NATIVE = process.argv.includes('--native')
const OUR_MODEL = process.env.MODEL || (NATIVE ? 'qwen3mlx' : 'qwen35')
const LMS_MODEL = process.env.LMS_MODEL || 'small'
const GEMM = process.env.GEMM || undefined
const RUNS = 3
const DECODE = 128

const text = readFileSync(`${ROOT}/CLAUDE.md`, 'utf8').slice(0, 3200)
// A DISTINCT prompt per round, identical text on both sides. With one fixed
// prompt our engine's cross-turn prefix reuse serves rounds 2..N from cache and
// reports ~16,000 tok/s "prefill" — a cache hit, not a measurement.
//
// `/no_think` is the switch that actually works on this checkpoint. Qwen3-4B is
// a thinking model; LM Studio ignored BOTH `reasoning_effort: none` and
// `chat_template_kwargs.enable_thinking`, spending every one of 128 tokens on
// reasoning and emitting no content at all, while our side rendered the
// non-thinking template. Verified by hand before being trusted here.
//
// The prompt also carries a PER-PROCESS nonce. Distinct-per-round was not
// enough: LM Studio's server outlives our process and cached round 0-2 from the
// previous invocation, so a second run read 3,751 tok/s against our cold 491 —
// their cache, measured as their prefill. Our own reuse guard could not see it,
// because it only ever watched our side. Both engines must be cold, and the
// only way to guarantee that from here is a prompt neither has seen.
const NONCE = process.env.NONCE || `${Date.now().toString(36)}`
const msgFor = (r) => `Summarise the following notes in one paragraph`
  + ` (ref ${NONCE}${r >= 0 ? `-${r + 1}` : ''}):\n\n${text} /no_think`

async function lmstudio(round) {
  const t0 = performance.now()
  const res = await fetch(`${LMS}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: LMS_MODEL, stream: true, max_tokens: DECODE, temperature: 0,
      stream_options: { include_usage: true },
      messages: [{ role: 'user', content: msgFor(round) }],
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
  // No content delta means tFirst never moved, and `tFirst - t0` then reports a
  // NEGATIVE ttft and a negative prefill rate. That printed as a result once;
  // it must be an error instead.
  if (!tFirst) throw new Error('lmstudio: no content deltas — the model produced only reasoning, or the request failed')
  const pt = usage?.prompt_tokens ?? 0
  const ct = usage?.completion_tokens ?? nOut
  return {
    promptTokens: pt, tokens: ct,
    ttftMs: tFirst - t0,
    prefillTokS: pt > 1 ? (pt - 1) / ((tFirst - t0) / 1000) : 0,
    decodeTokS: ct > 1 ? (ct - 1) / ((tEnd - tFirst) / 1000) : 0,
  }
}

// Fail with the fix, not a stack trace: a refused connection here is almost
// always the interface binding, not a stopped server.
try {
  const probe = await fetch(`${LMS}/v1/models`)
  if (!probe.ok) throw new Error(`HTTP ${probe.status}`)
} catch (e) {
  throw new Error(`cannot reach LM Studio at ${LMS} (${e.message}).\n`
    + `  It may be bound to a non-loopback interface — find it with:\n`
    + `    lsof -nP -iTCP:1234 -sTCP:LISTEN\n`
    + `  then re-run with LMS_URL=http://<addr>:1234`)
}

const fmt = (r) => `prompt ${String(r.promptTokens).padStart(4)}  ttft ${(r.ttftMs / 1000).toFixed(2)}s`
  + `  prefill ${r.prefillTokS.toFixed(0).padStart(4)} tok/s  decode ${r.decodeTokS.toFixed(1).padStart(6)} tok/s`

// --- our arm ---------------------------------------------------------------
// Two implementations of ONE contract: given ids and n, return the same record
// the LM Studio arm returns. Everything downstream (interleaving, medians,
// ratios) is shared, so the two arms cannot drift apart in protocol.
let ourRun, ourIds, teardown = async () => {}

if (NATIVE) {
  const { installShims } = await import('./native/shims.mjs')
  await installShims({ unsafe: true })
  const { createEngineRaw } = await import('../dist-lib/index.js')
  const { engine, tokenizer, spec, buildChatPromptFor } =
    await createEngineRaw({ model: OUR_MODEL, ...(GEMM ? { chunkGemm: GEMM } : {}) })
  ourIds = (r) => buildChatPromptFor(spec, [{ role: 'user', content: msgFor(r) }], tokenizer)
  ourRun = async (ids, n) => {
    const t0 = performance.now()
    let tFirst = 0, count = 0
    await engine.generatePipelined(ids, n, () => { if (!count) tFirst = performance.now(); count++ }, () => false)
    const tEnd = performance.now()
    const pre = engine.getLastPrefill()
    // A few reused ids are the chat template's own opening, shared by every
    // prompt and worth ~0.3%. A round served from the cache reuses nearly all
    // of them (974 of 975 was the run that reported 16,443 tok/s "prefill").
    if (pre && pre.reused > ids.length / 20) {
      throw new Error(`prefill reused ${pre.reused} of ${ids.length} ids — that round measured the prefix cache, not prefill`)
    }
    return {
      promptTokens: ids.length, tokens: count,
      ttftMs: tFirst - t0,
      prefillTokS: ids.length / ((tFirst - t0) / 1000),
      decodeTokS: count > 1 ? (count - 1) / ((tEnd - tFirst) / 1000) : 0,
    }
  }
} else {
  const { startHarness, stopHarness, newPage } = await import('../tests/e2e/harness.ts')
  await startHarness()
  teardown = stopHarness
  const page = await newPage(`/model-smoke.html?model=${OUR_MODEL}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  ourIds = async (r) => page.evaluate(async (msg) => {
    const { loadByteLevelTokenizer, buildChatPrompt } = await import('/src/zero-tvm/tokenizer-bpe.ts')
    const tok = await loadByteLevelTokenizer()
    return buildChatPrompt([{ role: 'user', content: msg }], tok, { thinking: false })
  }, msgFor(r))
  ourRun = (ids, n) => page.evaluate((i, k) => window.__timedGenerateChat(i, k), ids, n)
}

try {
  const warmIds = await ourIds(-1)
  console.log(`ours: ${OUR_MODEL} on ${NATIVE ? 'dawn.node' : 'browser'}${GEMM ? ` (gemm ${GEMM})` : ''}`
    + `, ~${warmIds.length} ids | theirs: LM Studio '${LMS_MODEL}'\n`)

  // Warm both once (uncounted): first run pays shader warm-up / cache fill.
  await ourRun(warmIds, 8)
  await lmstudio(-1)

  const ours = [], theirs = []
  for (let r = 0; r < RUNS; r++) {
    const a = await ourRun(await ourIds(r), DECODE)
    ours.push(a)
    console.log(`  zero-tvm  ${fmt(a)}`)
    const b = await lmstudio(r)
    theirs.push(b)
    console.log(`  lmstudio  ${fmt(b)}`)
  }
  const med = (xs, k) => [...xs].map((x) => x[k]).sort((a, b) => a - b)[Math.floor(xs.length / 2)]
  console.log(`\nmedians of ${RUNS} interleaved runs:`)
  console.log(`  zero-tvm   prefill ${med(ours, 'prefillTokS').toFixed(0)} tok/s   decode ${med(ours, 'decodeTokS').toFixed(1)} tok/s`)
  console.log(`  lmstudio   prefill ${med(theirs, 'prefillTokS').toFixed(0)} tok/s   decode ${med(theirs, 'decodeTokS').toFixed(1)} tok/s`)
  console.log(`  ratio      prefill ${(med(ours, 'prefillTokS') / med(theirs, 'prefillTokS')).toFixed(2)}x        decode ${(med(ours, 'decodeTokS') / med(theirs, 'decodeTokS')).toFixed(2)}x`)
  console.log(`\nraw ours   prefill ${ours.map((x) => x.prefillTokS.toFixed(0)).join(' / ')}  decode ${ours.map((x) => x.decodeTokS.toFixed(1)).join(' / ')}`)
  console.log(`raw theirs prefill ${theirs.map((x) => x.prefillTokS.toFixed(0)).join(' / ')}  decode ${theirs.map((x) => x.decodeTokS.toFixed(1)).join(' / ')}`)
  console.log(NATIVE
    ? `\ncaveats: SAME checkpoint bytes both sides; one machine, one day.`
    : `\ncaveats: different checkpoints (MLC q4f16_1 vs MLX 4-bit), browser tab focused,`
      + `\none machine, one day. Context: ours 262,144 max (?ctx=), theirs fitted 198,400.`)
} finally {
  await teardown()
}

if (NATIVE) {
  // dawn.node holds the loop open; drain stdout before leaving.
  await new Promise((r) => process.stdout.write('', r))
  process.exit(process.exitCode ?? 0)
}
