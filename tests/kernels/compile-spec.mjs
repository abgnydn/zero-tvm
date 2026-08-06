// COMPILE-SPEC — the generic compile-all gate: every .wgsl file and every
// int4_matmul generator variant must build under an arbitrary spec's dims.
//
//   node tests/kernels/compile-spec.mjs QWEN3_4B          # any model-spec export
//   node tests/kernels/compile-spec.mjs QWEN3_4B_4BIT     # generated specs too
//
// This is the per-model half of what compile-qwen(35).mjs pioneered, split out
// so scripts/add-model.mjs can gate GENERATED specs without a hand-written
// suite per model. It proves the shader prelude's const block is consistent
// (dims divide the workgroup layouts, address math stays in u32 range) — the
// cheapest strong check that exists without weights. Numerical trust still
// comes from validate.html's logits diff against mlx_lm.

import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { getDevice } from './gpu.mjs'
import { withPrelude } from '../../src/compiler/shader-prelude.ts'
import * as specs from '../../src/compiler/model-spec.ts'
import {
  int4MatmulWGSL,
  int4MatmulEntry,
  INT4_MATMUL_VARIANTS,
} from '../../src/compiler/shaders/int4_matmul.gen.ts'

const name = process.argv[2]
const S = specs[name]
if (!S || typeof S !== 'object' || !S.id) {
  console.error(`usage: node tests/kernels/compile-spec.mjs <MODEL_SPEC_EXPORT>`)
  console.error(`known: ${Object.keys(specs).filter((k) => specs[k]?.id).join(', ')}`)
  process.exit(2)
}

const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')

async function main() {
  let env
  try {
    env = await getDevice()
  } catch (e) {
    console.error(`ERROR: ${e.message} — no WebGPU adapter for the compile gate`)
    process.exit(1)
  }
  const { device, info, f16, subgroups } = env
  console.log(`adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16} | subgroups: ${subgroups} | spec: ${S.id}`)
  if (!f16) {
    console.error('FAIL: adapter lacks shader-f16')
    process.exit(1)
  }

  const sources = readdirSync(SHADERS)
    .filter((f) => f.endsWith('.wgsl'))
    .sort()
    .map((f) => ({ name: f, code: withPrelude(readFileSync(resolve(SHADERS, f), 'utf8'), S) }))
  for (const v of INT4_MATMUL_VARIANTS) {
    sources.push({ name: `gen:${int4MatmulEntry(v)}`, code: withPrelude(int4MatmulWGSL(v), S) })
  }

  const errors = []
  let compiled = 0
  let skippedSg = 0
  for (const s of sources) {
    if (!subgroups && /^\s*enable subgroups;/m.test(s.code)) { skippedSg++; continue }
    const entryMatch = s.code.match(/@compute[\s\S]*?fn\s+([A-Za-z0-9_]+)/)
    if (!entryMatch) { errors.push(`${s.name}: no @compute entry point found`); continue }
    device.pushErrorScope('validation')
    const module = device.createShaderModule({ code: s.code })
    const compInfo = module.getCompilationInfo ? await module.getCompilationInfo() : { messages: [] }
    const msgs = [...compInfo.messages].filter((m) => m.type === 'error')
    if (msgs.length) {
      errors.push(`${s.name}: ${msgs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ')}`)
      await device.popErrorScope()
      continue
    }
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entryMatch[1] } })
    const scopeErr = await device.popErrorScope()
    if (scopeErr) { errors.push(`${s.name}: ${String(scopeErr.message).split('\n')[0]}`); continue }
    compiled++
  }
  for (const e of errors) console.error(`  ${e}`)
  const skipNote = skippedSg ? `, ${skippedSg} sg-only skipped` : ''
  console.log(`${errors.length ? 'FAIL' : 'PASS'}  compile_all  ${compiled} shaders compile under ${S.id} (${sources.length - INT4_MATMUL_VARIANTS.length} .wgsl + ${INT4_MATMUL_VARIANTS.length} generated${skipNote})`)
  process.exit(errors.length ? 1 : 0)
}

main()
