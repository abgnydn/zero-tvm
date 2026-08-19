#!/usr/bin/env node
// DEPTH BISECT — which subsystem breaks the agentic loop at long context?
//
// The finding this exists to close: qwen38 solves the eval at ~600 tokens and
// at 8k, invents tool names at 16k, and is total nonsense at 24k. mlx_lm, given
// a byte-identical prompt from the same checkpoint, calls the right tool at all
// three depths. So the divergence is ours and it grows with position.
//
// Exactly three things in this engine behave differently as a prompt gets
// longer:
//
//   int8 KV        every cached K/V is quantized per row; more tokens means
//                  more quantized values inside one softmax
//   chunked prefill  batched GEMMs over token blocks instead of per-token
//                  matvecs — a different arithmetic ORDER, and only reachable
//                  on prompts long enough to chunk
//   prefix reuse   restored KV plus, on a hybrid, a rewound GDN state
//
// Everything else — the kernels, the weights, the rendering — is identical at
// 600 tokens and at 24,000, and the 600-token run is correct. So turning each
// off in turn at a depth that FAILS tells you which one, and a run where all
// three are off and it still fails tells you it is none of them.
//
// Runs the arms in a fixed order and prints them together, because the failure
// is graded rather than binary: an arm that invents one tool name and recovers
// is different from one that never issues a real call, and reading four
// separate scrollbacks loses that.
//
//   npm run station                              # other shell
//   node scripts/depth-bisect.mjs qwen38         # default depth 16000
//   DEPTH=24000 node scripts/depth-bisect.mjs qwen38
import { spawn } from 'node:child_process'
import http from 'node:http'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const PARAM = process.argv[2]
const DEPTH = Number(process.env.DEPTH ?? 16000)
if (!PARAM) { console.error('usage: node scripts/depth-bisect.mjs <param>   [DEPTH=16000]'); process.exit(2) }

const get = (p, q) => new Promise((res, rej) => {
  const r = http.get({ host: '127.0.0.1', port: p, path: q }, (x) => {
    let b = ''; x.on('data', (c) => b += c); x.on('end', () => { try { res(JSON.parse(b)) } catch (e) { rej(new Error(b.slice(0, 200))) } })
  }); r.on('error', rej)
})

await get(8017, '/api/state').catch((e) => {
  if (e.code === 'ECONNREFUSED') { console.error('no station on 127.0.0.1:8017 — start it with: npm run station\n'); process.exit(1) }
  throw e
})

// Baseline FIRST. If the shipped configuration happens to pass today, every
// other arm is uninterpretable and the run should be read as "did not
// reproduce" rather than as evidence about any subsystem.
const ARMS = [
  { name: 'baseline (everything on)', env: {} },
  { name: 'int8 KV off', env: { KV8: '0' } },
  { name: 'chunked prefill off', env: { CHUNK: '0' } },
  { name: 'cross-turn reuse off', env: { REUSE: '0' } },
  { name: 'all three off', env: { KV8: '0', CHUNK: '0', REUSE: '0' } },
]

const run = (env) => new Promise((res) => {
  const p = spawn('node', ['scripts/agentic-eval.mjs', PARAM], {
    cwd: ROOT, env: { ...process.env, ...env, PAD: String(DEPTH) },
  })
  let out = ''
  p.stdout.on('data', (c) => { out += c })
  p.stderr.on('data', (c) => { out += c })
  p.on('close', () => res(out))
})

console.log(`${PARAM} · depth ${DEPTH.toLocaleString()} · ${ARMS.length} arms\n`)
const rows = []
for (const arm of ARMS) {
  process.stdout.write(`  ${arm.name.padEnd(26)} … `)
  const out = await run(arm.env)
  const solved = /SOLVED/.test(out)
  const trace = /trace\s+(.*)/.exec(out)?.[1] ?? ''
  // A tool name we never offered is the specific failure here, and it is not
  // the same as giving prose or running out of steps.
  const invented = [...trace.matchAll(/(\w+)\(/g)].map((m) => m[1])
    .filter((n) => !['list_files', 'read_file', 'search', 'write_file', 'run_command', 'attempt_completion', 'TEXT'].includes(n))
  const secs = /wall\s+(\d+)s/.exec(out)?.[1] ?? '?'
  rows.push({ arm: arm.name, solved, invented: [...new Set(invented)], secs })
  console.log(`${solved ? 'SOLVED' : 'failed'}  ${invented.length ? `invented: ${[...new Set(invented)].join(', ')}` : ''}  ${secs}s`)
}

console.log('\n=== VERDICT ===')
const base = rows[0]
if (base.solved && !base.invented.length) {
  console.log('  The BASELINE passed, so nothing was reproduced and no arm below means anything.')
  console.log('  Raise DEPTH until the baseline fails, then re-run.')
} else {
  const fixed = rows.slice(1, 4).filter((r) => r.solved && !r.invented.length)
  if (fixed.length === 1) {
    console.log(`  Turning off "${fixed[0].arm}" fixes it and the others do not.`)
    console.log('  That subsystem is the cause. Its kernels are where to look.')
  } else if (fixed.length > 1) {
    console.log(`  More than one arm fixes it (${fixed.map((r) => r.arm).join('; ')}).`)
    console.log('  They interact, or the failure is marginal enough that any perturbation')
    console.log('  moves it. Re-run before concluding — a graded failure can flip on noise.')
  } else if (rows[4].solved && !rows[4].invented.length) {
    console.log('  No single arm fixes it, but all three off does. It is a combination.')
  } else {
    console.log('  NONE of them fix it, including all three off. The depth-dependent')
    console.log('  subsystems are cleared, and what remains is the base decode path at')
    console.log('  long positions: attention accumulation and RoPE at high indices.')
    console.log('  Compare final-position logits against mlx_lm at this depth next.')
  }
}
