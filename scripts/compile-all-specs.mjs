#!/usr/bin/env node
// COMPILE-ALL-SPECS — the compile gate for EVERY spec export in model-spec.ts.
//
//   node scripts/compile-all-specs.mjs
//
// Each spec gets its own tests/kernels/compile-spec.mjs run: every .wgsl
// file and every int4_matmul generator variant must build under that spec's
// dims (prelude consts, workgroup layouts, u32 address math). No weights, no
// numerics — a new spec (or a prelude edit) that breaks another spec's
// shaders fails here instead of in someone's browser. Runs on lavapipe:
// sg-only shaders skip loudly there, like run.mjs.

import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'
import * as specs from '../src/compiler/model-spec.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const names = Object.keys(specs).filter((k) => specs[k]?.id).sort()

let failed = 0
for (const name of names) {
  console.log(`--- ${name} (${specs[name].id})`)
  const r = spawnSync('node', ['tests/kernels/compile-spec.mjs', name],
    { cwd: ROOT, stdio: 'inherit' })
  if (r.status !== 0) {
    failed++
    console.log(`SPEC-FAIL  ${name}`)
  }
}
console.log(failed
  ? `\n${failed} spec(s) FAILED to compile`
  : `\nall ${names.length} specs compile`)
process.exit(failed ? 1 : 0)
