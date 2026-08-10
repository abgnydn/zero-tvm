#!/usr/bin/env node
// MLA-ENGINE-TEST — the ENGINE running an MLA layer, not the kernels.
//
//   node scripts/mla-engine-test.mjs
//
// tests/kernels/real-weights.mjs drives the MLA kernels by hand: it builds its
// own bind groups, dispatches in its own order, and seeds intermediates from
// numpy. Every one of those is a thing the engine also has to get right, and
// none of them is under test there. This runs buildDecodeEngine and lets the
// engine do it — allocKVPages' MLA branch, kvIndex, the hoisted bind groups,
// the per-token position threading, SM_SCALE, the ten-dispatch order and the
// residual tail.
//
// IN A BROWSER, because that is the only place the engine runs: engine-core
// imports weight-loader, which reads GPUBufferUsage at module scope and spells
// its imports `.js`. Node type-stripping resolves neither. So this follows
// pipeline-split-test.mjs — harness, real Chrome, real WebGPU — and does its
// work inside page.evaluate against modules the dev server transforms.
//
// THREE layers, and the count is load-bearing. The stage under test is layer 1
// of a 3-layer spec with layerRange {1, 2}:
//   L0 = 1 != 0        -> no embedding bind group
//   L1 = 2 != 3        -> no LM head, no argmax, no sampler
// With `layers: 2` the stage would end the model, build an LM head and demand
// weights the bundle does not contain. pipelineStep is then exactly
// residual-in / residual-out, which is the shape the bundle's reference has.
//
// Positions are fed 0..19 IN ORDER so the engine's own mla_kv_write builds the
// whole cache. Only the last is compared against ref_out — but it can only be
// right if every earlier write landed at the right address, which is the part
// no other test covers.

import { createServer } from 'node:http'
import { readFileSync, existsSync } from 'node:fs'
import { join, resolve, dirname, normalize } from 'node:path'
import { fileURLToPath } from 'node:url'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'
import { f16Array, f16BitsToF32 } from '../tests/kernels/half.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const DIR = join(ROOT, '.weights-local/kernel-refs/dsv2layer0')
if (!existsSync(join(DIR, 'meta.json'))) {
  console.log('SKIP  no dsv2layer0 bundle — build it with scripts/pull-tensors.mjs + scripts/make-dsv2-layer-ref.py')
  process.exit(0)
}
const meta = JSON.parse(readFileSync(join(DIR, 'meta.json'), 'utf8'))
const { d: D, tokens: T, query_at: QI, dense_ffn: FFN } = meta

// The bundle is ~40 MB of int4 records; serving it beats marshalling it through
// evaluate(). Its own port, not the dev mirror, so this does not depend on how
// /local-weights maps a nested first path segment.
const PORT = 5199
const files = createServer((req, res) => {
  const rel = normalize(decodeURIComponent(req.url.split('?')[0])).replace(/^(\.\.[/\\])+/, '')
  const path = join(DIR, rel)
  if (!path.startsWith(DIR) || !existsSync(path)) { res.writeHead(404).end(); return }
  res.writeHead(200, { 'access-control-allow-origin': '*', 'content-type': 'application/octet-stream' })
  res.end(readFileSync(path))
}).listen(PORT)

// COPY the exact range, never `.buffer` — readFileSync hands back a view into a
// shared pool, so `.buffer` is the pool and a Float32Array over it silently
// reads its neighbours. (mlx-repack.mjs carries the same warning; ref_out came
// back 8x too long here before this.)
const f32 = (name) => {
  const b = readFileSync(join(DIR, name))
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length))
}
const x = f32('ref_x.bin')
const refOut = f32('ref_out.bin')
// f16 at the boundary, so the page receives exactly the bits a real stage would
// get off the wire — the hand-off is f16, and rounding here rather than there
// keeps the comparison against the same input the reference used.
const inputs = Array.from({ length: T }, (_, t) => Array.from(f16Array(x.slice(t * D, (t + 1) * D))))

let failed = false
await startHarness()
try {
  const page = await newPage('/docs.html')
  const out = await page.evaluate(async (base, inputs, cfg) => {
    const MS = await import('/src/compiler/model-spec.ts')
    const W = await import('/src/zero-tvm/mlx-weights.ts')
    const EC = await import('/src/zero-tvm/engine-core.ts')
    const C = await import('/src/compiler/compiler.ts')
    const V = await import('/src/zero-tvm/variants.ts')

    if (!navigator.gpu) return { error: 'no WebGPU in this browser' }
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter.requestDevice({
      requiredFeatures: ['shader-f16', ...(adapter.features.has('subgroups') ? ['subgroups'] : [])],
      requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
                        maxBufferSize: adapter.limits.maxBufferSize },
    })

    const S = MS.makeModelSpec({
      ...MS.DEEPSEEK_V2_LITE,
      id: 'dsv2-engine-test',
      layers: 3,
      ffn: cfg.ffn,          // layer 0 of the real model is DENSE at 10944
      moe: undefined,        // ...so a MoE spec would demand experts not in the bundle
      maxPages: 4,           // 64 positions, comfortably past the bundle's 20
      paramNaming: MS.mlxParamNaming(''),
    })

    // The real loader path over the bundle — buildBuffer is what tests/kernels/
    // mlx-repack.mjs verifies byte-exact, so a failure here is the engine's.
    const meta = await (await fetch(`${base}/meta.json`)).json()
    const rec = (n) => { const r = meta.tensors[n]; if (!r) throw new Error(`not in bundle: ${n}`); return r }
    const bytes = new Map()
    await Promise.all(Object.values(meta.tensors).map(async (r) => {
      bytes.set(r.file, new Uint8Array(await (await fetch(`${base}/${r.file}`)).arrayBuffer()))
    }))
    const readRecord = (n) => bytes.get(rec(n).file)
    const dtypeOf = (n) => rec(n).dtype

    // Plan for layer 0 — the plan's L picks RECORD NAMES (and the FFN kind,
    // dense everywhere in this spec), not the stage index. The stage is still
    // layer 1; the bundle just holds the real model's layer 0.
    const plans = Object.fromEntries(W.planLayer(S, 0, 'model.').map((p) => [p.name, p]))
    const buf = (name) => {
      const data = W.buildBuffer(plans[name], readRecord, dtypeOf).data
      const b = device.createBuffer({
        size: Math.ceil(data.byteLength / 4) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: name,
      })
      device.queue.writeBuffer(b, 0, data)
      return b
    }

    const layer = {
      normGamma1: buf('norm1'), normGamma2: buf('norm2'),
      oProjWeights: buf('o_proj_w'), oProjScales: buf('o_proj_s'), oProjBiases: buf('o_proj_b'),
      ffnWeights: buf('ffn_w'), ffnScales: buf('ffn_s'), ffnBiases: buf('ffn_b'),
      ffnDownWeights: buf('ffn_down_w'), ffnDownScales: buf('ffn_down_s'), ffnDownBiases: buf('ffn_down_b'),
      mla: {
        qWeights: buf('mla_q_w'), qScales: buf('mla_q_s'), qBiases: buf('mla_q_b'),
        kvaWeights: buf('mla_kva_w'), kvaScales: buf('mla_kva_s'), kvaBiases: buf('mla_kva_b'),
        kvaNormGamma: buf('mla_kva_norm'),
        kvbF16: buf('mla_kvb'),
      },
    }
    // Layers 0 and 2 are never touched by a {1,2} range, but the array has to
    // be the right LENGTH — kvIndex and the layer loop both index by absolute L.
    const weights = { root: {}, layers: [{}, layer, {}] }

    const variants = { ...V.SCALAR_VARIANTS, subgroups: device.features.has('subgroups') }
    const engine = EC.buildDecodeEngine(device, weights, EC.allocKVPages(device, S), {
      spec: S, variants, layerRange: { start: 1, end: 2 },
      pipelines: C.compile(device, S, variants),
    })

    let last = null
    for (let t = 0; t < inputs.length; t++) {
      const residual = Uint16Array.from(inputs[t]).buffer
      const r = await engine.pipelineStep({ residual }, t)
      if (!r || !('residual' in r)) return { error: `position ${t}: stage ended the model, expected a residual` }
      last = r.residual
    }
    return { bits: Array.from(new Uint16Array(last)), subgroups: variants.subgroups }
  }, `http://localhost:${PORT}`, inputs, { ffn: FFN })

  if (out.error) throw new Error(out.error)

  const got = out.bits.map(f16BitsToF32)
  if (got.length !== refOut.length || got.some(Number.isNaN)) {
    console.log(`      got ${got.length} values, ref ${refOut.length};`
      + ` ${got.filter(Number.isNaN).length} NaN; head ${got.slice(0, 4).map((v) => v.toFixed(4))}`
      + ` vs ref ${Array.from(refOut.slice(0, 4)).map((v) => v.toFixed(4))}`)
  }
  let scale = 0
  for (const v of refOut) scale = Math.max(scale, Math.abs(v))
  let err = 0
  for (let i = 0; i < refOut.length; i++) err = Math.max(err, Math.abs(got[i] - refOut[i]) / scale)

  // The kernel-level bundle holds each stage to ~3e-4. The whole layer through
  // the engine accumulates ten dispatches of f16 rounding on top of that.
  const pass = err < 2e-3
  failed = !pass
  console.log(`${pass ? 'PASS' : 'FAIL'}  engine MLA layer   max rel err ${err.toExponential(2)} at position ${QI}`)
  console.log(`      ${T} positions written by the engine's own mla_kv_write · dense FFN ${FFN}`
    + ` · subgroups ${out.subgroups ? 'on' : 'off'}`)
  if (pass) console.log('\nallocKVPages MLA branch, kvIndex, hoisted bind groups, position threading, '
    + 'SM_SCALE and the ten-dispatch order all agree with mlx')
} finally {
  files.close()
  await stopHarness()
}
process.exit(failed ? 1 : 0)
