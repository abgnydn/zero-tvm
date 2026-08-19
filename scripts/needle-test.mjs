#!/usr/bin/env node
// NEEDLE TEST — can the model still see the START of its context from the END?
//
// Written for a specific failure: qwen38 at ~24k tokens read every tool result
// (which sit near the end of the prompt) and then invented tool names that were
// never in the tools block (which sits at the start). mlx_lm, given the same
// conversation through the same template, emitted the correct call at 0, 8k and
// 24k. The prompt we send is byte-identical to the checkpoint's own template at
// both depths, so the fault is below the prompt — and "ignores the beginning,
// uses the end" is a retrieval failure, not general degradation.
//
// So this asks the narrowest question that separates them: put one fact at the
// very start, pad, ask for it at the very end. A model whose attention over
// distance is intact answers at every depth. One that cannot reach position 0
// from position 24,000 answers at short depths and fails at long ones, and the
// depth where it flips is where to look.
//
// Deliberately NOT a tool loop: no tool schema, no multi-step, nothing that can
// fail for a second reason. A wrong answer here is retrieval or nothing.
//
//   npm run station                                  # other shell, load a model
//   node scripts/needle-test.mjs                     # default sweep
//   DEPTHS=0,4000,16000 node scripts/needle-test.mjs
import http from 'node:http'

const post = (port, path, body) => new Promise((res, rej) => {
  const d = JSON.stringify(body)
  const r = http.request({ host: '127.0.0.1', port, path, method: 'POST',
    headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(d) } }, (x) => {
    let b = ''; x.on('data', (c) => b += c); x.on('end', () => { try { res(JSON.parse(b)) } catch (e) { rej(new Error(b.slice(0, 300))) } })
  })
  r.setTimeout(0); r.on('error', rej); r.end(d)
})
const get = (p, q) => new Promise((res, rej) => {
  const r = http.get({ host: '127.0.0.1', port: p, path: q }, (x) => {
    let b = ''; x.on('data', (c) => b += c); x.on('end', () => { try { res(JSON.parse(b)) } catch (e) { rej(new Error(b.slice(0, 300))) } })
  }); r.on('error', rej)
})

// The needle is arbitrary and unguessable: nothing in the padding or the
// question implies it, so a correct answer can only come from reading position
// ~0. A number the model might reach for by chance would make a pass meaningless.
const NEEDLE = 'plum-47-harbor'
const SYSTEM = `Remember this access code exactly: ${NEEDLE}. You will be asked for it later.`
const ASK = 'What is the access code from the beginning of this conversation? Reply with the code and nothing else.'

/** Inert filler. It must not mention the code, a code, or anything that reads as
 *  one, or the test measures pattern-matching instead of retrieval. */
function padding(target) {
  const out = []
  let approx = 0
  for (let i = 0; approx < target; i++) {
    const q = `Item ${i}: describe the deployment steps for release 2.${i % 9}.${i % 5}.`
    const a = `Release 2.${i % 9}.${i % 5} follows the standard procedure: build the artifact, run the `
      + `regression suite, promote the staging environment, then flip the traffic weight. No `
      + `configuration keys move and no credentials rotate. Rollback is promoting the previous build.`
    out.push({ role: 'user', content: q }, { role: 'assistant', content: a })
    approx += Math.ceil((q.length + a.length) / 3.6)
  }
  return out
}

const DEPTHS = (process.env.DEPTHS ?? '0,2000,8000,16000,24000').split(',').map(Number)

const state = await get(8017, '/api/state').catch((e) => {
  if (e.code === 'ECONNREFUSED') { console.error('no station on 127.0.0.1:8017 — start it with: npm run station\n'); process.exit(1) }
  throw e
})
if (state.phase !== 'ready') { console.error(`station is "${state.phase}", not ready — load a model first`); process.exit(1) }
const h = await get(8019, '/health')
console.log(`${h.hosting} · ctx ${h.ctx} · kv8=${h.kv8} · reuse=${h.reuse}`)
console.log(`needle "${NEEDLE}" at position ~0, asked at the end\n`)
console.log(`  ${'depth'.padStart(7)}  ${'prompt tok'.padStart(10)}  found?  answer`)

const rows = []
for (const d of DEPTHS) {
  const messages = [{ role: 'system', content: SYSTEM }, ...padding(d), { role: 'user', content: ASK }]
  const t0 = Date.now()
  const j = await post(8017, '/v1/chat/completions', {
    model: 'ztvm', messages, max_tokens: 40, temperature: 0, stream: false,
  })
  const text = (j.choices?.[0]?.message?.content ?? '').trim()
  const tok = j.usage?.prompt_tokens ?? 0
  const ok = text.toLowerCase().includes(NEEDLE)
  rows.push({ d, ok })
  console.log(`  ${String(d).padStart(7)}  ${String(tok).padStart(10)}  ${ok ? 'YES ' : 'NO  '}   ${JSON.stringify(text.slice(0, 60))}  ${((Date.now() - t0) / 1000).toFixed(0)}s`)
}

const firstFail = rows.find((r) => !r.ok)
console.log('')
if (!firstFail) {
  console.log('  Retrieval is intact at every depth tested. Whatever the agentic loop is')
  console.log('  failing on, it is not the model losing sight of the start of its context.')
} else if (firstFail.d === DEPTHS[0]) {
  console.log('  It fails at the SHALLOWEST depth, so this is not about distance at all —')
  console.log('  check the prompt reaches the model intact before reading anything else.')
} else {
  console.log(`  Retrieval breaks between ${rows[rows.indexOf(firstFail) - 1].d} and ${firstFail.d} tokens.`)
  console.log('  That is attention over distance, below the prompt layer: same rendering,')
  console.log('  same weights, only the position range changed. Compare our final-position')
  console.log('  logits against mlx_lm at that depth next — a short-prompt validate-model')
  console.log('  run cannot see it.')
}
