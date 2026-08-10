#!/usr/bin/env node
// QUALITY-EVAL — perplexity of the WGSL engine, which is the only number here
// that can see a model quantized into uselessness.
//
//   node scripts/quality-eval.mjs llama32 --tokens 256
//   node scripts/quality-eval.mjs llama32 --corpus code --tokens 512
//   node scripts/quality-eval.mjs llama32 --dump-ids /tmp/ids.json   # no GPU
//   node scripts/quality-eval.mjs llama32 --ref /tmp/ppl-llama32.json
//
// THE GAP THIS CLOSES. Everything else in this repo is a FIDELITY check:
// validate-model.mjs compares the engine against mlx_lm, but mlx-ref.py:29 is
// `mlx_lm.load(args.model)` — the SAME quantized checkpoint. Quantize a model
// until it emits gibberish and MLX emits the same gibberish; cosine stays
// 0.9999, argmax matches, greedy is token-exact, every gate is green. That is
// not a bug in those gates, it is what they are for. But it means nothing in
// this repo has ever measured whether a quantized build is any GOOD.
//
// Perplexity is absolute. It needs no reference model, only held-out text.
//
// TWO CORPORA, because they answer different questions:
//   --corpus prose  (default) natural English. Comparable to published
//                   perplexity numbers, and the standard signal for "is this
//                   model broken".
//   --corpus code   this repository's own TypeScript. In-domain for the
//                   agentic-coding target, on disk already, and not memorised
//                   in any specific form. This is the number that matters for
//                   "can this model write code".
//
// What a result means:
//   - Absolute ppl is only meaningful RELATIVE to another build on the SAME
//     tokens. Compare 4-bit vs 3-bit, not this number against a paper.
//   - `--ref` compares against scripts/mlx-perplexity.py over the identical
//     ids. Close agreement is a fidelity result (a stronger one than a single
//     prompt's logits, since it covers hundreds of positions). It says nothing
//     about quality on its own.

import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'
import { createByteLevelTokenizer } from '../src/zero-tvm/tokenizer-bpe.ts'
import { specForParam } from '../src/zero-tvm/model-registry.ts'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const args = process.argv.slice(2)
const flag = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 ? args[i + 1] : d }
// The positional arg is the first token that is neither a flag NOR the value
// of one. The old test (`!args.includes('--'+a)`) let a flag's value through:
// `quality-eval.mjs --tokens 256` took "256" as the model name and printed
// `SKIP no tokenizer for 256`.
const FLAG_VALUE_INDICES = new Set(
  args.flatMap((a, i) => (a.startsWith('--') ? [i + 1] : [])))
const param = args.find((a, i) => !a.startsWith('--') && !FLAG_VALUE_INDICES.has(i)) ?? 'llama32'
const N = Number(flag('tokens', 256))
const CORPUS = flag('corpus', 'prose')
const DUMP = flag('dump-ids', null)
const REF = flag('ref', null)

// ------------------------------------------------------------------ corpus

/** Repo source as one blob — the in-domain corpus, already on disk.
 *  Files are taken in sorted order so the token sequence is reproducible. */
function codeCorpus(budgetChars) {
  const dirs = ['src/zero-tvm', 'src/compiler']
  const files = []
  for (const d of dirs) {
    const full = join(ROOT, d)
    if (!existsSync(full)) continue
    for (const e of readdirSync(full).sort()) {
      const p = join(full, e)
      if (statSync(p).isFile() && e.endsWith('.ts')) files.push(p)
    }
  }
  let out = ''
  for (const f of files) {
    out += readFileSync(f, 'utf8')
    if (out.length >= budgetChars) break
  }
  return out.slice(0, budgetChars)
}

/** Held-out prose. Committed rather than downloaded (bandwidth), and written
 *  for this purpose rather than lifted from a corpus — text from wikitext or
 *  Project Gutenberg is in every model's training set, which deflates
 *  perplexity by an unknown amount and makes the absolute number a fiction.
 *  It is still only useful COMPARATIVELY; see the header. */
const PROSE_PATH = join(ROOT, 'tests/fixtures/quality-corpus.txt')

// ------------------------------------------------------------------- run

const spec = specForParam(param)
const tokDir = {
  llama32: 'Llama-3.2-1B-Instruct-4bit',
  qwen3mlx: 'Qwen3-4B-4bit',
  qwen30b: 'Qwen3-30B-A3B-4bit',
  qwen36q3: 'Qwen3.6-35B-A3B-MLX-q3exp',
  qwen36: 'Qwen3.6-35B-A3B-MLX-4bit',
}[param]
const tokPath = tokDir && join(ROOT, '.weights-local', tokDir, 'tokenizer.json')
if (!tokPath || !existsSync(tokPath)) {
  console.log(`SKIP  no tokenizer for ${param} at ${tokPath ?? '(unmapped)'}`)
  process.exit(0)
}
const tok = createByteLevelTokenizer(JSON.parse(readFileSync(tokPath, 'utf8')))

const text = CORPUS === 'code' ? codeCorpus(N * 8)
  : existsSync(PROSE_PATH) ? readFileSync(PROSE_PATH, 'utf8')
  : (() => { console.log(`SKIP  no corpus at ${PROSE_PATH}`); process.exit(0) })()

const ids = tok.encode(text).slice(0, N + 1)
if (ids.length < N + 1) {
  console.log(`note: corpus yielded only ${ids.length} tokens, wanted ${N + 1}`)
}
if (ids.length > spec.maxContext) {
  console.log(`SKIP  ${ids.length} tokens exceeds ${param}'s ${spec.maxContext}-token context`)
  process.exit(0)
}

if (DUMP) {
  writeFileSync(DUMP, JSON.stringify({ ids, corpus: CORPUS, model: param }))
  console.log(`wrote ${ids.length} ids -> ${DUMP}`)
  console.log(`reference:\n  cd ~/dev/ml-research && uv run python ${ROOT}/scripts/mlx-perplexity.py \\\n`
    + `    --model ${ROOT}/.weights-local/${tokDir} --ids ${DUMP} --out /tmp/ppl-${param}.json`)
  process.exit(0)
}

console.log(`model ${param} | corpus ${CORPUS} | ${ids.length - 1} positions to score`)

await startHarness()
let ppl = null
try {
  const page = await newPage(`/model-smoke.html?model=${param}`)
  await page.waitForFunction(() => window.__phase === 'loaded' || window.__phase === 'error',
    { timeout: 8 * 60_000, polling: 1000 })
  if (await page.evaluate(() => window.__phase) === 'error') {
    throw new Error(await page.evaluate(() => window.__error))
  }
  const t0 = Date.now()
  const nll = await page.evaluate((x) => window.__score(x), ids)
  const secs = (Date.now() - t0) / 1000

  const mean = nll.reduce((a, b) => a + b, 0) / nll.length
  ppl = Math.exp(mean)
  const gpuErrs = await page.evaluate(() => window.__gpuErrs())

  // Standard error over BLOCKS, not over positions. Adjacent positions in one
  // prefix are strongly correlated — position 3 and position 4 share almost
  // all their context — so a per-position sigma understates the real spread by
  // a lot. Blocks of 64 are roughly independent.
  const BLK = 64
  const blocks = []
  for (let i = 0; i + BLK <= nll.length; i += BLK) {
    let s = 0
    for (let j = i; j < i + BLK; j++) s += nll[j]
    blocks.push(s / BLK)
  }
  let sem = NaN
  if (blocks.length >= 2) {
    const bm = blocks.reduce((a, b) => a + b, 0) / blocks.length
    const v = blocks.reduce((a, b) => a + (b - bm) ** 2, 0) / (blocks.length - 1)
    sem = Math.sqrt(v / blocks.length)
  }

  console.log(`\n  tokens scored  ${nll.length}`)
  console.log(`  mean NLL       ${mean.toFixed(4)}`)
  console.log(`  PERPLEXITY     ${ppl.toFixed(3)}`
    + (Number.isFinite(sem) ? `  [${Math.exp(mean - sem).toFixed(3)}, ${Math.exp(mean + sem).toFixed(3)}]` : ''))
  console.log(`  ${(nll.length / secs).toFixed(1)} positions/s, ${secs.toFixed(0)}s, gpu errors ${gpuErrs}`)

  // THE trap with a single-prefix score: early positions have almost no
  // context and enormous NLL, so the mean falls steeply as the window grows.
  // Measured on Llama-3.2-1B over the same text: 128 -> 151.15, 256 -> 65.57,
  // 512 -> 41.31, 1198 -> 32.18. Two runs at different --tokens are not
  // comparable, and nothing but this warning says so.
  console.log(`\n  NOT COMPARABLE ACROSS --tokens: this scores one contiguous prefix, so the`)
  console.log(`  number falls steeply with window length (5x between 128 and 1198 on a 1B).`)
  console.log(`  Quote it only against another run at --tokens ${N} on --corpus ${CORPUS}.`)
  console.log(`  To compare two CHECKPOINTS, use scripts/quality-ab.py — independent`)
  console.log(`  windows, error bars, and it runs on the reference in seconds.`)

  // A model that has been quantized into noise sits near uniform over the
  // vocabulary. That bound is not a quality bar — it is a floor, and a model
  // can be useless well above it — but crossing it is unambiguous.
  const uniform = spec.vocab
  console.log(`\n  uniform-over-vocab perplexity would be ${uniform} (${param} vocab)`)
  if (ppl > uniform * 0.5) {
    console.log(`  *** ppl is within 2x of uniform — this model is not producing language ***`)
  }

  if (REF && existsSync(REF)) {
    const ref = JSON.parse(readFileSync(REF, 'utf8'))
    if (ref.nll.length !== nll.length) {
      console.log(`\n  reference scored ${ref.nll.length} positions, engine ${nll.length} — not comparable`)
    } else {
      let maxAbs = 0
      for (let i = 0; i < nll.length; i++) maxAbs = Math.max(maxAbs, Math.abs(nll[i] - ref.nll[i]))
      console.log(`\n  vs mlx_lm on the SAME ids (${ref.dtype}):`)
      console.log(`    mlx perplexity     ${ref.perplexity.toFixed(3)}`)
      console.log(`    engine perplexity  ${ppl.toFixed(3)}`)
      console.log(`    ratio              ${(ppl / ref.perplexity).toFixed(4)}`)
      console.log(`    max |ΔNLL| at any position  ${maxAbs.toExponential(2)}`)
      console.log(`  (fidelity, over ${nll.length} positions rather than one prompt's final logits.`)
      console.log(`   It says nothing about whether the model is good — both sides are the same`)
      console.log(`   quantized checkpoint.)`)
    }
  } else if (REF) {
    console.log(`\n  --ref ${REF} not found; run scripts/mlx-perplexity.py first (see --dump-ids)`)
  }
} finally {
  await stopHarness()
}
