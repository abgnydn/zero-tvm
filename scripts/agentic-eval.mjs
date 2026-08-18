// AGENTIC EVAL — can the model drive a tool loop to completion, or does it
// read and read and never act?
//
// Not a chat benchmark. It scores the loop itself: does it use what a tool
// returned, does it stop reading once it has the answer, does it finish. The
// reported failure was "it reads and reads, fills the context, then I compact"
// — so REDUNDANT READS and NEVER FINISHING are the two things measured, not
// prose quality.
//
// The workspace is fake but consistent: the answer exists, is reachable in
// three steps, and cannot be guessed from the task text.
// Needs the station on :8017. It loads the model itself, so the station can be
// idle when this starts.
//
//   npm run station                          # in another shell
//   node scripts/agentic-eval.mjs qwen36q3            # short task
//   PAD=24000 node scripts/agentic-eval.mjs qwen36q3  # the failing arm
//   PAD=24000 KV8=0 node scripts/agentic-eval.mjs qwen36q3    # f16 control
//   PAD=24000 REUSE=0 node scripts/agentic-eval.mjs qwen36q3  # no-reuse control
//
// Measured 2026-08-18 on qwen36q3, ctx 32768: short = SOLVED in 20s, 3/3 files,
// no wasted reads. PAD=24000 = reads all three files, computes the right
// number, then answers in PROSE ("7 * 12 = 84.\n\n84") instead of calling
// attempt_completion. KV8=0 fails identically, so the int8 cache is not it.
// qwen38 at the same depth is worse: it invents mcp__tools__list_directory and
// calls it nine times.
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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function load(param) {
  // KV8=0 loads an f16 cache — the control for "is this the model or our
  // quantizer?". int8 was measured at 1k and 4k windows only, and the failure
  // being chased sits at 24k.
  const kv8 = process.env.KV8 !== '0'
  // REUSE=0 turns off cross-turn prefix reuse AND the GDN rewind ring, so every
  // turn prefills the whole prompt from zero. It is the control for "does our
  // cross-turn machinery corrupt a deep conversation?" — the failing turn is
  // the FOURTH one, and turns 2-4 all ride reused state. Slow on purpose: a
  // 24k-token turn re-prefills 24k tokens.
  const reuse = process.env.REUSE !== '0'
  await post(8017, '/api/load', { param, ctx: 32768, pool: 0, kv8, reuse })
  for (let i = 0; i < 200; i++) {
    await sleep(2000)
    const s = await get(8017, '/api/state')
    if (s.phase === 'ready') return
    if (s.phase === 'failed') throw new Error(`${param}: ${s.failure}`)
  }
  throw new Error(`${param}: no boot`)
}

// ── the workspace ───────────────────────────────────────────────────────────
// 24 files, 3 of which matter. The reported failure is "it reads and reads,
// fills the context, then I compact" — so the workspace has to REWARD not
// reading. Every distractor is plausible: same naming, same shape, several
// carry numbers that look like they could be the answer.
const FILES = {
  'README.md': 'Service workspace. Pipeline stages live under src/. See docs/ for notes.',
  'config.json': '{ "entry": "src/pipeline.ts", "version": 7, "timeout": 300 }',
  'package.json': '{ "name": "svc", "version": "2.4.1" }',
  'src/pipeline.ts': 'import { WIDTH } from "./dims.ts"\nimport { scale } from "./scale.ts"\n'
    + 'export function capacity() { return scale(WIDTH) }',
  'src/dims.ts': 'export const WIDTH = 12\nexport const HEIGHT = 40   // unused by capacity()',
  'src/scale.ts': 'export const FACTOR = 7\nexport function scale(n: number) { return n * FACTOR }',
  'src/legacy.ts': 'export const WIDTH = 99   // old copy, nothing imports this',
  'src/util.ts': 'export const clamp = (n: number) => Math.max(0, n)',
  'src/index.ts': 'export * from "./pipeline.ts"',
  'src/cache.ts': 'export const TTL = 84',
  'src/queue.ts': 'export const DEPTH = 21',
  'src/retry.ts': 'export const LIMIT = 3',
  'src/log.ts': 'export const LEVEL = "info"',
  'src/http.ts': 'export const PORT = 8080',
  'src/db.ts': 'export const POOL = 16',
  'src/auth.ts': 'export const ROUNDS = 12',
  'docs/architecture.md': 'The pipeline computes capacity from the configured width.',
  'docs/notes.md': 'Old note: capacity used to be 99. This is out of date.',
  'docs/faq.md': 'Q: what is capacity? A: see src/pipeline.ts',
  'tests/pipeline.test.ts': 'test("capacity", () => { expect(capacity()).toBe(EXPECTED) })',
  'tests/dims.test.ts': 'test("width", () => { expect(WIDTH).toBeGreaterThan(0) })',
  'CHANGELOG.md': '2.4.1 — bumped FACTOR. 2.4.0 — initial.',
  '.gitignore': 'node_modules\ndist',
  'LICENSE': 'MIT',
}
// capacity() = scale(WIDTH) = 12 * 7 = 84. Needs pipeline.ts, dims.ts, scale.ts.
// Traps: legacy.ts says WIDTH=99, notes.md says capacity was 99, cache.ts
// carries the literal 84 so a guesser can land on it without earning it.
const ANSWER = '84'

const TOOLS = [
  { type: 'function', function: { name: 'list_files', description: 'List every file in the workspace',
    parameters: { type: 'object', properties: {}, required: [] } } },
  { type: 'function', function: { name: 'read_file', description: 'Read one file by path',
    parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'search', description: 'Search file contents for a string',
    parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } },
  { type: 'function', function: { name: 'write_file', description: 'Write content to a file',
    parameters: { type: 'object', properties: { path: { type: 'string' }, content: { type: 'string' } }, required: ['path', 'content'] } } },
  { type: 'function', function: { name: 'run_command', description: 'Run a shell command in the workspace',
    parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } } },
  { type: 'function', function: { name: 'attempt_completion', description: 'Report the final answer and finish the task',
    parameters: { type: 'object', properties: { result: { type: 'string' } }, required: ['result'] } } },
]

/** Files a correct solution must read. Anything else is a wasted step. */
const NEEDED = ['src/pipeline.ts', 'src/dims.ts', 'src/scale.ts']

const TASK = 'What number does capacity() return? Read only the files you need, '
  + 'then call attempt_completion with just the number. Do not guess.'

/** Prior conversation, to put the task at REAL depth. The short version of
 *  this eval both models ace; the reported failure only shows up on sessions
 *  tens of thousands of tokens deep, so depth is the variable to move. Content
 *  is inert on purpose — it must not contain the answer or hint at it, or the
 *  test measures retrieval rather than whether the loop survives length. */
function padding(targetTokens) {
  const msgs = []
  if (!targetTokens) return msgs
  let approx = 0
  for (let i = 0; approx < targetTokens; i++) {
    const q = `Step ${i}: summarise what changed in release 2.${i % 9}.${i % 5}.`
    const a = `Release 2.${i % 9}.${i % 5} adjusted logging thresholds, renamed two internal `
      + `helpers, and left public behaviour unchanged. No configuration keys moved, and the `
      + `deployment procedure is identical to the previous release. Nothing here affects `
      + `pipeline capacity or the dimension constants used elsewhere in the service.`
    msgs.push({ role: 'user', content: q }, { role: 'assistant', content: a })
    approx += Math.ceil((q.length + a.length) / 3.6)
  }
  return msgs
}

async function runEpisode(label, maxSteps = 12, padTokens = 0) {
  const msgs = [
    { role: 'system', content: 'You are a coding agent. Call exactly one tool per step. '
      + 'Use what each tool returns. When you know the answer, call attempt_completion. '
      + 'Never read a file you have already read.' },
    ...padding(padTokens),
    { role: 'user', content: TASK },
  ]
  const calls = []
  const reads = []
  let finished = null
  let steps = 0
  const t0 = Date.now()

  for (; steps < maxSteps; steps++) {
    const j = await post(8017, '/v1/chat/completions', {
      model: 'ztvm', messages: msgs, tools: TOOLS, max_tokens: 300, temperature: 0, stream: false,
    })
    const m = j.choices?.[0]?.message ?? {}
    const c = (m.tool_calls ?? [])[0]
    if (!c) {
      calls.push('TEXT')
      // What it said instead matters: prose CONTAINING the answer means the
      // model solved it and only failed to emit the call — instruction
      // following, not arithmetic. Prose without it means it lost the thread.
      //
      // Print the END as well as the start, and say whether the budget ran
      // out. This model's own template invites reasoning BEFORE the call, so
      // the call is the LAST thing generated — a run cut off at max_tokens
      // looks, from the front, exactly like a model that chose prose. The
      // reference harness was misread that way once already.
      const text = m.content ?? ''
      const fin = j.choices?.[0]?.finish_reason ?? '?'
      console.log(`    prose instead of a call (finish_reason=${fin}, ${text.length} chars)`)
      console.log(`      starts: ${JSON.stringify(text.slice(0, 180))}`)
      if (text.length > 180) console.log(`      ends:   ${JSON.stringify(text.slice(-180))}`)
      break
    }
    const args = (() => { try { return JSON.parse(c.function.arguments) } catch { return {} } })()
    const sig = `${c.function.name}(${args.path ?? ''})`
    calls.push(sig)

    let out
    if (c.function.name === 'list_files') out = Object.keys(FILES).join('\n')
    else if (c.function.name === 'read_file') {
      reads.push(args.path)
      out = FILES[args.path] ?? `Error: no such file: ${args.path}`
    } else if (c.function.name === 'search') {
      const q = String(args.query ?? '')
      const hits = Object.entries(FILES).filter(([, v]) => v.includes(q)).map(([k]) => k)
      out = hits.length ? hits.join('\n') : 'no matches'
    } else if (c.function.name === 'write_file' || c.function.name === 'run_command') {
      out = 'Error: this task is read-only'
    } else if (c.function.name === 'attempt_completion') {
      finished = String(args.result ?? '')
      break
    } else out = `Error: unknown tool ${c.function.name}`

    msgs.push({ role: 'assistant', content: m.content ?? '', tool_calls: m.tool_calls })
    msgs.push({ role: 'tool', tool_call_id: c.id, content: out })
  }

  const dupReads = reads.length - new Set(reads).size
  const wasted = [...new Set(reads)].filter((f) => !NEEDED.includes(f)).length
  const missed = NEEDED.filter((f) => !reads.includes(f)).length
  const correct = finished !== null && finished.includes(ANSWER)
  const secs = ((Date.now() - t0) / 1000).toFixed(0)
  console.log(`\n  ${label}`)
  console.log(`    steps        ${steps + (finished !== null ? 1 : 0)} of ${maxSteps}`)
  console.log(`    finished     ${finished !== null ? `yes → ${JSON.stringify(finished.slice(0, 60))}` : 'NO — ran out of steps or gave prose'}`)
  console.log(`    correct      ${correct ? 'YES' : 'no'}   (expected ${ANSWER})`)
  console.log(`    files read   ${reads.length} (${dupReads} repeats, ${wasted} unnecessary of ${new Set(reads).size} distinct)`)
  console.log(`    needed files ${NEEDED.length - missed}/${NEEDED.length} read`)
  console.log(`    trace        ${calls.join(' → ')}`)
  console.log(`    wall         ${secs}s`)
  return { correct, finished: finished !== null, steps, dupReads, wasted, secs }
}

const PAD = Number(process.env.PAD ?? 0)
const models = process.argv.slice(2)
if (!models.length) { console.error('usage: agentic-eval.mjs <param> [param...]'); process.exit(1) }
const results = {}
for (const p of models) {
  console.log(`\n=== ${p} ===`)
  await load(p)
  const h = await get(8019, '/health')
  console.log(`  loaded ${h.hosting} · ctx ${h.ctx} · kv8=${h.kv8} · reuse=${h.reuse}`)
  const arm = `kv8=${h.kv8} reuse=${h.reuse}`
  results[p] = await runEpisode(`${p} · ${PAD ? `~${Math.round(PAD / 1000)}k-token history` : 'short'} · ${arm}`, 12, PAD)
}
console.log('\n=== VERDICT ===')
for (const [p, r] of Object.entries(results)) {
  console.log(`  ${p.padEnd(10)} ${r.correct ? 'SOLVED' : r.finished ? 'finished but WRONG' : 'DID NOT FINISH'}`
    + ` · ${r.steps} steps · ${r.wasted} unnecessary reads · ${r.dupReads} repeats · ${r.secs}s`)
}
