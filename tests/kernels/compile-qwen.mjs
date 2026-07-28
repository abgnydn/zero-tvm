// Compile-only gate for the QWEN3_4B ModelSpec (phase A of the Qwen3 port).
//
// Injects shaderPrelude(QWEN3_4B) — GQA dims: D=2560, HEADS=32, KV_HEADS=8,
// HEAD_DIM=128, FFN=9728, VOCAB=151936 — into every .wgsl file and every
// int4_matmul generator variant, and checks they COMPILE on the local Dawn
// adapter. This is a shape-consistency gate only: shaders that still assume
// heads == kvHeads or qDim == d are semantically wrong under these constants
// until the phase-B GQA port, but they must at least build.
//
// Files that legitimately cannot compile under GQA dims yet are listed in
// EXPECTED_INCOMPATIBLE and reported loudly as the phase-B worklist instead
// of failing the gate. An UNEXPECTED failure (or an entry that unexpectedly
// starts compiling) exits non-zero.
//
//   npm run test:kernels:qwen

import { readFileSync, readdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import { getDevice } from './gpu.mjs'
import { withPrelude, QWEN3_4B } from '../../src/compiler/shader-prelude.ts'
import {
  int4MatmulWGSL,
  int4MatmulEntry,
  INT4_MATMUL_VARIANTS,
} from '../../src/compiler/shaders/int4_matmul.gen.ts'

const SHADERS = resolve(dirname(fileURLToPath(import.meta.url)), '../../src/compiler/shaders')

// Shaders that cannot compile under QWEN3_4B constants yet. Empty today —
// every shader is shape-consistent enough to build (semantic GQA correctness
// is phase B's job). Add names here (with a reason) only if a GQA-dim change
// genuinely breaks compilation.
const EXPECTED_INCOMPATIBLE = new Map([
  // 'example.wgsl': 'reason',
])

async function main() {
  let env
  try {
    env = await getDevice()
  } catch (e) {
    console.error(`ERROR: ${e.message}`)
    process.exit(1)
  }
  const { device, info, f16, subgroups } = env
  console.log(
    `adapter: ${info.description || info.vendor || 'unknown'} | shader-f16: ${f16} | subgroups: ${subgroups ? 'yes' : 'no'}`,
  )
  if (!f16) {
    console.error('FAIL: adapter lacks shader-f16; cannot compile the f16 kernels')
    process.exit(1)
  }

  const sources = readdirSync(SHADERS)
    .filter((f) => f.endsWith('.wgsl'))
    .sort()
    .map((f) => ({ name: f, code: withPrelude(readFileSync(resolve(SHADERS, f), 'utf8'), QWEN3_4B) }))
  for (const v of INT4_MATMUL_VARIANTS) {
    sources.push({ name: `gen:${int4MatmulEntry(v)}`, code: withPrelude(int4MatmulWGSL(v), QWEN3_4B) })
  }

  const failures = []
  let compiled = 0
  let skippedSg = 0
  for (const s of sources) {
    if (!subgroups && /^\s*enable subgroups;/m.test(s.code)) {
      skippedSg++
      continue
    }
    const entryMatch = s.code.match(/@compute[\s\S]*?fn\s+([A-Za-z0-9_]+)/)
    if (!entryMatch) {
      failures.push([s.name, 'no @compute entry point found'])
      continue
    }
    device.pushErrorScope('validation')
    const module = device.createShaderModule({ code: s.code })
    const infoC = module.getCompilationInfo ? await module.getCompilationInfo() : { messages: [] }
    const msgs = [...infoC.messages].filter((m) => m.type === 'error')
    if (msgs.length) {
      failures.push([s.name, msgs.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join(' | ')])
      await device.popErrorScope()
      continue
    }
    device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: entryMatch[1] },
    })
    const scopeErr = await device.popErrorScope()
    if (scopeErr) {
      failures.push([s.name, String(scopeErr.message).split('\n')[0]])
      continue
    }
    compiled++
  }

  const unexpected = failures.filter(([name]) => !EXPECTED_INCOMPATIBLE.has(name))
  const expectedHit = failures.filter(([name]) => EXPECTED_INCOMPATIBLE.has(name))
  const expectedMiss = [...EXPECTED_INCOMPATIBLE.keys()].filter(
    (name) => !failures.some(([f]) => f === name),
  )

  if (expectedHit.length) {
    console.log('\nSKIP (known GQA-incompatible — the phase-B worklist):')
    for (const [name, detail] of expectedHit) {
      console.log(`  ${name}: ${EXPECTED_INCOMPATIBLE.get(name)}\n    ${detail}`)
    }
  }
  if (unexpected.length) {
    console.error('\nFAIL: unexpected compile errors under QWEN3_4B constants:')
    for (const [name, detail] of unexpected) console.error(`  ${name}: ${detail}`)
  }
  if (expectedMiss.length) {
    console.error('\nFAIL: listed as incompatible but now compile (remove from EXPECTED_INCOMPATIBLE):')
    for (const name of expectedMiss) console.error(`  ${name}`)
  }

  const skipNote = skippedSg ? `, ${skippedSg} sg-only skipped (no 'subgroups' feature)` : ''
  console.log(
    `\n${compiled}/${sources.length - skippedSg} shaders compile under QWEN3_4B ` +
    `(${sources.length - INT4_MATMUL_VARIANTS.length} .wgsl + ${INT4_MATMUL_VARIANTS.length} generated${skipNote})`,
  )
  process.exit(unexpected.length || expectedMiss.length ? 1 : 0)
}

main()
