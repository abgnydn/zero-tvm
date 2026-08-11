#!/usr/bin/env node
// AGENT — one command from zero to "pi is talking to a model in a browser tab".
//
//   npm run agent                      # qwen3mlx, the gated default
//   npm run agent -- qwen35 --ctx 65536
//   npm run agent -- llama32 --pool 0
//
// Does, in order, skipping anything already running:
//   1. vite dev server (:5173) and agent-server (:8017), spawned detached —
//      they survive this script exiting.
//   2. Opens Chrome on agent-host.html for the chosen model.
//   3. Patches ~/.pi/agent/models.json: provider "zerotvm", model id "ztvm",
//      contextWindow matched to the ACTUAL engine context (including --ctx).
//      One-time backup at models.json.zerotvm.bak. Only the "zerotvm"
//      provider is ever touched.
//   4. Waits until the tab reports hosting, then prints the pi command.
//
// The model id is always "ztvm" and always means "whatever this launcher
// started" — relaunching with a different model rewrites the entry, so pi
// never silently talks to the wrong model (the server also 400s on a
// mismatched model name — the LM Studio failure mode, deliberately closed).

import { spawn } from 'node:child_process'
import { existsSync, readFileSync, writeFileSync, copyFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { specForParam, specWithCtx, SHIPPED_MODELS } from '../src/zero-tvm/model-registry.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
const flag = (n) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : null }
// Presets, because "npm run agent -- qwen35 --ctx 65536" is not a UX.
//   ztvm        -> qwen3mlx, the gated tool-caller
//   ztvm big    -> qwen35 at 64k context
//   ztvm max    -> qwen35 at its native 262k (8 GiB of KV — quiet machine)
//   ztvm fast   -> llama32 (chat only; it does not call tools)
//   ztvm code   -> like default, then DROPS YOU INTO pi when ready
const PRESETS = {
  big: { param: 'qwen35', ctx: 65536 },
  max: { param: 'qwen35', ctx: 262144 },
  fast: { param: 'llama32', ctx: 0 },
  code: { param: 'qwen3mlx', ctx: 0, pi: true },
}
const word = args.find((a) => !a.startsWith('--') && a !== flag('ctx') && a !== flag('pool'))
const preset = word ? PRESETS[word] : null
const param = preset?.param ?? word ?? 'qwen3mlx'
const ctx = Number(flag('ctx')) || preset?.ctx || 0
const pool = flag('pool') !== '0'
const intoPi = preset?.pi === true || args.includes('--pi')

const known = SHIPPED_MODELS.map((m) => m.param).filter(Boolean)
if (!known.includes(param)) {
  console.error(`unknown model "${param}" — one of: ${known.join(', ')}`)
  process.exit(1)
}
const spec = ctx ? specWithCtx(specForParam(param), ctx) : specForParam(param)
if (param === 'llama32') {
  console.log('note: llama32 does not CALL tools (it echoes the schema) — fine for chat,')
  console.log('      wrong for agentic use. qwen3mlx is the gated tool-caller.\n')
}

const up = async (url) => { try { return (await fetch(url, { signal: AbortSignal.timeout(1500) })).ok } catch { return false } }
const detach = (cmd, argv, name) => {
  const child = spawn(cmd, argv, { cwd: ROOT, detached: true, stdio: 'ignore' })
  child.unref()
  console.log(`started ${name} (pid ${child.pid})`)
}

// 1. servers — reuse anything already listening.
if (await up('http://localhost:5173/agent-host.html')) console.log('vite already up (:5173)')
else detach('npm', ['run', 'dev'], 'vite dev server')
if (await up('http://127.0.0.1:8017/health')) console.log('agent-server already up (:8017)')
else detach('node', [join(ROOT, 'scripts/agent-server.mjs')], 'agent-server')

for (let i = 0; i < 40; i++) {
  if (await up('http://localhost:5173/agent-host.html') && await up('http://127.0.0.1:8017/health')) break
  await new Promise((r) => setTimeout(r, 500))
}

// 2. the hosting tab.
const url = `http://localhost:5173/agent-host.html?model=${param}`
  + (ctx ? `&ctx=${ctx}` : '') + (pool ? '' : '&pool=0')
spawn('open', ['-a', 'Google Chrome', url], { stdio: 'ignore' }).on('error', () => spawn('open', [url]))
console.log(`opening ${url}`)

// 3. pi config. Touches ONLY providers.zerotvm; everything else verbatim.
const piPath = join(homedir(), '.pi/agent/models.json')
if (existsSync(piPath)) {
  const bak = piPath.replace(/\.json$/, '.zerotvm.bak.json')
  if (!existsSync(bak)) copyFileSync(piPath, bak)
  const cfg = JSON.parse(readFileSync(piPath, 'utf8'))
  cfg.providers ??= {}
  cfg.providers.zerotvm = {
    baseUrl: 'http://127.0.0.1:8017/v1',
    api: 'openai-completions',
    apiKey: 'zerotvm',
    models: [{
      id: 'ztvm',
      name: `zero-tvm ${param} (browser WebGPU, ctx ${spec.maxContext.toLocaleString()})`,
      contextWindow: spec.maxContext,
      // Leave the model room to answer inside small windows.
      maxTokens: spec.maxContext >= 16384 ? 4096 : 1024,
      reasoning: false,
      samplingParams: { temperature: 0.7, top_p: 0.8 },
    }],
  }
  writeFileSync(piPath, JSON.stringify(cfg, null, 2) + '\n')
  console.log(`pi config: providers.zerotvm -> ztvm = ${param}, ctx ${spec.maxContext} (backup: ${bak.split('/').pop()})`)
} else {
  console.log(`pi config not found at ${piPath} — skipping (curl/Cline still work)`)
}

// 4. wait for the model.
process.stdout.write('waiting for the tab to host (first run downloads weights)')
let hosting = null
for (let i = 0; i < 600; i++) {
  try {
    const h = await (await fetch('http://127.0.0.1:8017/health')).json()
    if (h.tabConnected) { hosting = h.hosting; break }
  } catch { /* server booting */ }
  process.stdout.write('.')
  await new Promise((r) => setTimeout(r, 1000))
}
console.log('')
if (!hosting) {
  console.log('tab never connected — check the Chrome window for an error, then rerun.')
  process.exit(1)
}
console.log(`
READY — ${hosting} hosting on http://127.0.0.1:8017/v1

  pi:     PI_OFFLINE=1 pi --model ztvm
  Cline:  OpenAI Compatible · Base URL http://127.0.0.1:8017/v1 · model ztvm
  curl:   curl 127.0.0.1:8017/v1/chat/completions -H 'content-type: application/json' \\
            -d '{"messages":[{"role":"user","content":"hi"}]}'

  keep the Chrome tab VISIBLE (background = ~3x slower)
  ${pool ? 'pool ON: kill the tab, relaunch, context restores in ~0.5s' : 'pool OFF (--pool 0)'}
  switch models: rerun with another name (${known.filter((k) => k !== param).slice(0, 4).join(', ')}…)`)

if (intoPi) {
  console.log('\ndropping into pi — Ctrl+C twice to leave\n')
  spawn('pi', ['--model', 'ztvm'], {
    stdio: 'inherit',
    env: { ...process.env, PI_OFFLINE: '1' },
  }).on('exit', (code) => process.exit(code ?? 0))
}
