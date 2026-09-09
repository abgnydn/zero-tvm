#!/usr/bin/env node
// PAGING-TEST — does the engine actually honour its KV page table?
//
//   node scripts/paging-test.mjs [qwen3mlx|llama32|qwen35] [--expect-fail]
//
// docs/PAGING_PLAN.md §0.4. The page table has been the identity since the day
// it was written, so no code in this repo has ever run with a permuted one.
// Before any kernel is taught to use it, this establishes what actually happens
// today — and the answer is the thing worth writing down.
//
// The asymmetry: the READERS (attention.wgsl:70-72 and its _sg / _splitk /
// _int8 siblings) resolve every page through page_table_values. The WRITERS
// (kv_append.wgsl, qkv_fused.wgsl, qk_norm_rope_append.wgsl,
// kv_quantize_int8.wgsl, mla_kv_write.wgsl) do not take the table at all —
// they compute position / PAGE_SIZE and write there. Under the identity those
// agree. Under a permutation they cannot.
//
// So --expect-fail is the DEFAULT and it is the honest assertion for HEAD:
// every non-identity permutation MUST change the logits, and the identity
// control MUST not. Once the writers take the table, drop the flag and the
// same script asserts bit-identical logits instead. Written in this order on
// purpose — §0.2's falsifier passed on its first run because its mutation
// silently did nothing, and a test authored after the fix cannot tell "the
// kernels are right" from "the permutation never reached them".

import { readFileSync } from 'node:fs'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const args = process.argv.slice(2)
const param = args.find((a) => !a.startsWith('--')) ?? 'llama32'
// Default expectation is HEAD's behaviour. Pass --expect-pass once the writers
// take the table.
const EXPECT_PASS = args.includes('--expect-pass')

// Five pages of prompt at pageSize 16, so a permutation has something to
// permute. Arbitrary ids in range for every spec's vocab.
const IDS = Array.from({ length: 88 }, (_, i) => 1000 + ((i * 37) % 900))

let failed = false
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(26)} ${detail}`)
  if (!ok) failed = true
}

await startHarness()
try {
  const page = await newPage(`/model-smoke.html?model=${param}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  const maxPages = await page.evaluate(() => window.__maxPages())
  const nUsed = Math.ceil(IDS.length / 16)
  console.log(`spec ${param} | ${IDS.length} tokens = ${nUsed} pages of ${maxPages}\n`)

  // Permutations. Only the pages the prompt actually touches matter; the rest
  // stay put so the table remains a permutation of 0..maxPages-1.
  const ident = Array.from({ length: maxPages }, (_, i) => i)
  const perms = {
    'identity (control)': ident,
    'reverse used pages': ident.map((v, i) => (i < nUsed ? nUsed - 1 - i : v)),
    'rotate used by 1': ident.map((v, i) => (i < nUsed ? (i + 1) % nUsed : v)),
    'page 0 -> high page': (() => {
      const p = ident.slice()
      const hi = maxPages - 1
      ;[p[0], p[hi]] = [p[hi], p[0]]
      return p
    })(),
  }

  // Both variant sets, because they select DIFFERENT write kernels — the fused
  // QKV path writes K inline, the unfused one goes through kv_append.
  const modes = [
    ['scalar', {}],
    ['shipped', { subgroups: true, matmul: 'tiled', splitK: 8 }],
  ]

  for (const [modeName, v] of modes) {
    for (const [permName, perm] of Object.entries(perms)) {
      const { identity, permuted } = await page.evaluate(
        (ids, p, vv) => window.__pageTableCheck(ids, p, vv), IDS, perm, v)
      let maxAbs = 0
      let diffs = 0
      for (let i = 0; i < identity.length; i++) {
        const d = Math.abs(identity[i] - permuted[i])
        if (d !== 0) diffs++
        maxAbs = Math.max(maxAbs, d)
      }
      const isControl = permName.startsWith('identity')
      // Control must always be exactly 0 — same engine, same table, same
      // dispatches. If it is not, nothing else in this run means anything.
      const ok = isControl ? maxAbs === 0
        : EXPECT_PASS ? maxAbs === 0 : maxAbs !== 0
      check(`${modeName} / ${permName}`, ok,
        `max|Δlogit| ${maxAbs.toExponential(2)}  (${diffs}/${identity.length} differ)`
        + (isControl ? '  <- must be 0' : EXPECT_PASS ? '  <- must be 0' : '  <- must be nonzero on HEAD'))
    }
  }

  console.log(`\n${EXPECT_PASS
    ? 'asserted the writers HONOUR the page table'
    : 'asserted HEAD\'s real behaviour: readers honour the table, writers ignore it.\n'
      + 'A permuted table silently corrupts the cache today — which is exactly why\n'
      + 'the pool cannot ship before the writers take it (PAGING_PLAN §2.2).'}`)
} finally {
  await stopHarness()
}
process.exit(failed ? 1 : 0)
