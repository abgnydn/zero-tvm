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
  //   ztvm native -> NO BROWSER: the engine in-process on dawn.node,
  //                  robustness off, same OpenAI surface. The fastest mode.
  native: { param: 'qwen3mlx', native: true },
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

// 2a. NATIVE mode: no tab at all — agent-native.mjs runs the engine
// in-process on dawn.node (see scripts/dawn-probe.mjs for the numbers).
if (preset?.native || args.includes('--native')) {
  // 8017 may be held by a BROWSER-mode server from an earlier launch — the
  // native host then dies on EADDRINUSE under detached stdio and the wait
  // below spins forever (found live). Evict it; the two modes are exclusive.
  try {
    const h = await (await fetch('http://127.0.0.1:8017/health', { signal: AbortSignal.timeout(1500) })).json()
    if (!h.native) {
      console.log('stopping the browser-mode agent-server on :8017')
      spawn('pkill', ['-f', 'scripts/agent-server.mjs']).on('error', () => {})
      await new Promise((r) => setTimeout(r, 800))
    }
  } catch { /* nothing on 8017 — good */ }
  try { await import('webgpu') } catch {
    console.log('installing dawn.node prebuilt (once)…')
    await new Promise((r) => spawn('npm', ['i', '--no-save', 'webgpu'], { cwd: ROOT, stdio: 'inherit' }).on('exit', r))
  }
  detach('node', [join(ROOT, 'scripts/agent-native.mjs'), param,
    ...(ctx ? ['--ctx', String(ctx)] : []), ...(pool ? [] : ['--pool', '0'])], 'agent-native (dawn.node)')
  patchPi()
  process.stdout.write('waiting for the native host (first run builds the weight cache)')
  let ok = false
  for (let i = 0; i < 600; i++) {
    try { const h = await (await fetch('http://127.0.0.1:8017/health')).json(); if (h.native) { ok = true; break } } catch { /* booting */ }
    process.stdout.write('.'); await new Promise((r) => setTimeout(r, 1000))
  }
  console.log('')
  if (!ok) { console.log('native host never came up — check `node scripts/agent-native.mjs` directly'); process.exit(1) }
  console.log(`
READY (native, no browser) — http://127.0.0.1:8017/v1

  pi:     PI_OFFLINE=1 pi --model ztvm
  Cline:  OpenAI Compatible · Base URL http://127.0.0.1:8017/v1 · model ztvm
`)
  if (intoPi) {
    spawn('pi', ['--model', 'ztvm'], { stdio: 'inherit', env: { ...process.env, PI_OFFLINE: '1' } })
      .on('exit', (code) => process.exit(code ?? 0))
  }
  process.exit(0)
}

// 2. the hosting tab — in ZTVM'S OWN Chrome instance, not the user's browser.
// A Chrome we launch can take Dawn flags the public web never gets, and one of
// them is worth +22% prefill end-to-end: disable_robustness removes the
// mandatory bounds-check on every GPU buffer access (measured here: 482 vs
// 394 tok/s on qwen3mlx, tokens identical; kernel-level +26-56%, E2 in
// docs/PREFILL_RESEARCH.md). Safe for THIS surface because the separate
// profile only ever loads our localhost page — do not browse in that window.
// --safe opts back into a stock browser tab.
const url = `http://localhost:5173/agent-host.html?model=${param}`
  + (ctx ? `&ctx=${ctx}` : '') + (pool ? '' : '&pool=0')
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
if (!args.includes('--safe') && existsSync(CHROME)) {
  const profile = join(homedir(), '.zerotvm', 'chrome-profile')
  const c = spawn(CHROME, [
    `--user-data-dir=${profile}`,
    '--enable-unsafe-webgpu',
    '--enable-dawn-features=allow_unsafe_apis,disable_robustness',
    '--no-first-run', '--no-default-browser-check',
    `--app=${url}`,
  ], { detached: true, stdio: 'ignore' })
  c.unref()
  console.log(`opening ${url} (ztvm Chrome, robustness off: +22% prefill)`)
} else {
  spawn('open', ['-a', 'Google Chrome', url], { stdio: 'ignore' }).on('error', () => spawn('open', [url]))
  console.log(`opening ${url}${args.includes('--safe') ? ' (stock browser, --safe)' : ''}`)
}

// 3. pi config. Touches ONLY providers.zerotvm; everything else verbatim.
function patchPi() {
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
}
patchPi()

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
