#!/usr/bin/env node
// CHUNK-PREFILL-TEST — does chunked prefill compute what per-token prefill does?
//
//   node scripts/chunk-prefill-test.mjs qwen3mlx
//
// Chunking replaces n matvecs with one batched GEMM over n tokens. That is a
// different ARITHMETIC ORDER, not a different algorithm, so f16 rounding can
// differ — but the tokens must not. Two engines over the same weights, one with
// ?chunk=0, same prompt, greedy.
//
// Until 2026-08-11 this path was hybrid-and-MLC only. The gate now admits
// plain-attention and MLX-affine specs, which is what this checks.

import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const param = process.argv.slice(2).find((a) => !a.startsWith('--')) ?? 'qwen3mlx'
const GEMM = process.env.GEMM || ''
const TOKENS = Number(process.env.TOKENS) || 24
// Long enough to span several chunks (CHUNK_CAP is 64) and to make the batched
// GEMM's ragged final block non-trivial.
const PROMPT = Number(process.env.PROMPT) || 150
const CAP = Number(process.env.CAP) || 0
const IDS = Array.from({ length: PROMPT }, (_, i) => 1000 + ((i * 37) % 900))

let failed = false
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name.padEnd(26)} ${detail}`)
  if (!ok) failed = true
}

await startHarness()
try {
  const page = await newPage(`/model-smoke.html?model=${param}${GEMM ? `&gemm=${GEMM}` : ''}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  const logs = []
  let sawChunks = 0
  page.on('console', (m) => {
    const t = m.text()
    if (t.includes('chunked prefill')) logs.push(t)
    // CHUNK_MIN gates chunking, so a short prompt runs the per-token path in
    // BOTH arms and passes vacuously. The first version of this test reported
    // PASS at 4 and 8 tokens for exactly that reason while the engine was
    // broken at every length that actually chunked.
    const m2 = /prefill: .*?(\d+) chunks? of/.exec(t)
    if (m2) sawChunks += Number(m2[1])
  })

  for (const [label, v] of [['scalar', {}], ['shipped', { subgroups: true, matmul: 'tiled', splitK: 8 }]]) {
    const { chunked, perToken, chunkedMs, perTokenMs } = await page.evaluate(
      (ids, n, vv, cap) => window.__chunkCheck(ids, n, vv, cap), IDS, TOKENS, v, CAP)
    if (chunkedMs && perTokenMs) {
      console.log(`      ${label}: chunked ${(chunkedMs / 1000).toFixed(2)}s vs per-token ${(perTokenMs / 1000).toFixed(2)}s`
        + `  (${(perTokenMs / chunkedMs).toFixed(2)}x, ${IDS.length}-token prompt + ${TOKENS} decode)`)
    }
    // A run that produced NO tokens must not be able to pass. Two empty lists
    // are elementwise identical, so the length is checked against what was
    // asked for, not just against the other arm.
    const same = chunked.length === TOKENS && chunked.length === perToken.length
      && chunked.every((t, i) => t === perToken[i])
    const at = chunked.findIndex((t, i) => t !== perToken[i])
    check(`${label} tokens identical`, same,
      same ? `${chunked.length} tokens over ${IDS.length}-token prompt`
        : `diverges at ${at}: ${chunked[at]} vs ${perToken[at]}`)
  }
  // SCALE IS PART OF THE CLAIM. This gate asserts that chunked prefill equals
  // per-token, and it has passed for months at PROMPT=150 — which, with
  // CHUNK_CAP at 1024, is a SINGLE chunk. One chunk crosses no boundary and
  // exercises no per-chunk accumulation, so the property most likely to break
  // was never under test. It broke: qwen38 at 16k emits a tool name that was
  // never offered, and per-token on the same prompt emits the right one.
  //
  // So a single chunk no longer counts as having tested chunking. Two is the
  // minimum that crosses a boundary; the size of each chunk matters
  // independently, which is what CAP is for.
  check('chunking actually ran', sawChunks >= 2,
    sawChunks >= 2 ? `${sawChunks} chunks recorded (${IDS.length} tokens)`
      : sawChunks === 1
        ? `ONE chunk — crosses no boundary, so this proves nothing about chunking. `
          + `Raise PROMPT above the cap (or lower CAP): PROMPT=${IDS.length} needs ~${IDS.length * 2} to split.`
        : 'ZERO chunks — this run proves nothing, raise PROMPT')
  const gpuErrs = await page.evaluate(() => window.__gpuErrs())
  check('gpu errors', gpuErrs === 0, String(gpuErrs))
  for (const l of [...new Set(logs)]) console.log(`      ${l}`)
} finally {
  await stopHarness()
}
console.log(failed
  ? '\nchunked prefill DIVERGES from per-token'
  : `\nchunked prefill agrees with per-token at ${IDS.length} tokens / ${sawChunks} chunks`)
if (!failed && IDS.length < 4000) {
  console.log('  NOTE: this is a SHORT prompt. The defect found on 2026-08-19 needed ~16k')
  console.log('  tokens and got worse with chunk SIZE, so a short pass does not cover it:')
  console.log(`    PROMPT=16000 TOKENS=8 node scripts/chunk-prefill-test.mjs ${process.argv[2] ?? '<spec>'}`)
}
process.exit(failed ? 1 : 0)
