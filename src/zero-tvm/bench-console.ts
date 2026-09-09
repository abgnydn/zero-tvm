/**
 * DEVTOOLS BENCH HARNESSES (chat page)
 *
 * Wires window.bench / window.benchTtft / window.benchBatched / window.specSim
 * so the chat page can be benchmarked from the console (or preview_eval)
 * without touching the chat UI. Imported by chat.ts for its side effects via
 * installBenchConsole().
 */

import { Tokenizer } from './tokenizer.js'
import { buildChatPromptFor } from './model-select.js'
import { summarize as summarizePLD } from './spec-sim.js'
import type { DecodeEngine } from './engine-core.js'
import { buildPromptOfLength, TTFT_PROMPT_LENGTHS } from '../bench-prompts.js'

export interface BenchConsoleCtx {
  engine: DecodeEngine
  tokenizer: Tokenizer
  /** True while a chat decode is in flight (bench refuses to overlap). */
  isBusy: () => boolean
  /** Toggle the chat UI's busy state around a bench run. */
  setBusy: (busy: boolean) => void
  /** Called after a run to restore the send button's enabled state. */
  onIdle: () => void
}

export function installBenchConsole(ctx: BenchConsoleCtx): void {
  const { engine, tokenizer } = ctx

  // Benchmark harness: `await window.bench(nTokens=128, nRuns=3)`.
  // Uses a fixed one-turn prompt and runs `nRuns + 1` decodes; the first is
  // discarded as warmup. Reports per-run and aggregate tok/s (min/median/mean/max).
  ;(window as Window & typeof globalThis & { bench?: unknown }).bench =
    async (nTokens: number = 128, nRuns: number = 3, profile: boolean = false): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[bench] decode already in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        const benchPrompt = 'Write a four-sentence explanation of how photosynthesis works.'
        const benchHist: Array<{ role: 'system' | 'user'; content: string }> = [
          { role: 'system', content: 'You are a helpful assistant.' },
          { role: 'user', content: benchPrompt },
        ]
        const promptIds = buildChatPromptFor(engine.spec, benchHist, tokenizer)

        if (profile) {
          const prof = await engine.profileStep(promptIds)
          if (!prof) {
            console.warn('[bench] profile requested but timestamp-query feature unavailable')
            return null
          }
          console.log(`[profile] total = ${prof.totalMs.toFixed(3)} ms/token across ${prof.kernels.reduce((s, k) => s + k.calls, 0)} dispatches`)
          console.table(prof.kernels.map(k => ({
            kernel: k.label,
            calls: k.calls,
            'ms/token': +k.totalMs.toFixed(3),
            'ms/call': +(k.totalMs / k.calls).toFixed(4),
            '% total': +k.pctOfTotal.toFixed(1),
          })))
          return prof
        }

        const runs: { tokens: number; seconds: number; tokPerS: number; ttftMs: number; decodeTokPerS: number }[] = []
        for (let r = 0; r <= nRuns; r++) {
          const label = r === 0 ? 'warmup' : `run${r}/${nRuns}`
          // Every run must do a FULL prefill. Without this, cross-turn prefix
          // reuse makes runs 2..N prefill a single token, so our half of the
          // A/B measures decode-only while the WebLLM half (a fresh chat
          // completion per run) still pays prefill inside its wall clock.
          engine.resetKVTracking()
          const t0 = performance.now()
          let count = 0
          let tFirst = 0
          await engine.generatePipelined(promptIds, nTokens, () => {
            if (count === 0) tFirst = performance.now()
            count++
          })
          const tEnd = performance.now()
          const seconds = (tEnd - t0) / 1000
          const tokPerS = count / seconds
          // TTFT covers prefill + the first decode step; decode rate excludes it,
          // so the two halves can be compared like for like.
          const ttftMs = tFirst ? tFirst - t0 : seconds * 1000
          const decodeTokPerS = count > 1 && tFirst ? (count - 1) / ((tEnd - tFirst) / 1000) : tokPerS
          console.log(
            `[bench] ${label}: ${count} tok / ${seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s total` +
            ` · ttft ${ttftMs.toFixed(0)}ms · decode ${decodeTokPerS.toFixed(2)} tok/s`
          )
          if (r > 0) runs.push({ tokens: count, seconds, tokPerS, ttftMs, decodeTokPerS })
        }
        const medianOf = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]
        const sorted = runs.map(r => r.tokPerS).sort((a, b) => a - b)
        const median = sorted[Math.floor(sorted.length / 2)]
        const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
        const min = sorted[0]
        const max = sorted[sorted.length - 1]
        const medianDecode = medianOf(runs.map(r => r.decodeTokPerS))
        const medianTtftMs = medianOf(runs.map(r => r.ttftMs))
        const summary = { runs, median, mean, min, max, medianDecode, medianTtftMs }
        console.log(
          `[bench] summary: median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s` +
          ` · decode=${medianDecode.toFixed(2)} tok/s · ttft=${medianTtftMs.toFixed(0)}ms`
        )
        return summary
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  // TTFT as a function of prompt length: `await window.benchTtft()`.
  //
  // The counterpart of src/tjs-bench/main.ts's ?ttft=1 sweep, sharing its
  // prompt construction (src/bench-prompts.ts) so both engines prefill the
  // same text at the same measured length. Exists to answer a specific
  // question — huggingface/transformers.js#1599 reports a 16-second TTFT on
  // Qwen3.5-4B, and a single prompt length cannot separate "prefill has a
  // large constant cost" from "prefill scales badly with length".
  //
  // Only 8 tokens are generated per point; everything past the first token is
  // decode, which window.bench() already measures. `chunks` comes from the
  // engine's own getLastPrefill() and is the chunked-prefill evidence: it
  // should rise with prompt length while ms/prompt-token stays flat.
  ;(window as Window & typeof globalThis & { benchTtft?: unknown }).benchTtft =
    async (
      targets: number[] = TTFT_PROMPT_LENGTHS,
      nGen: number = 8,
      nRuns: number = 3,
    ): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[ttft] decode already in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        const medianOf = (xs: number[]) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)]
        const encodeTemplated = (userMessage: string) =>
          buildChatPromptFor(engine.spec, [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: userMessage },
          ], tokenizer)

        const points: {
          targetTokens: number
          promptTokens: number
          ttftMsRuns: number[]
          ttftMs: number
          decodeTokPerS: number
          chunks: number
        }[] = []

        for (const target of targets) {
          const built = buildPromptOfLength(target, encodeTemplated)
          const promptIds = encodeTemplated(built.userMessage)
          if (promptIds.length > engine.maxContext) {
            console.warn(`[ttft] target ${target}: ${promptIds.length} tok exceeds maxContext ${engine.maxContext} — skipped`)
            continue
          }
          const ttfts: number[] = []
          const decodes: number[] = []
          let chunks = 0
          // One unmeasured run at this length first, mirroring the
          // transformers.js half (a new sequence length can trigger fresh
          // pipeline specialisation, which is not the prefill cost).
          engine.resetKVTracking()
          await engine.generatePipelined(promptIds, nGen, () => {})
          for (let r = 0; r < nRuns; r++) {
            // Every run pays a FULL prefill — without this, cross-turn prefix
            // reuse makes runs 2..N prefill a single token and the whole
            // experiment measures nothing.
            engine.resetKVTracking()
            const t0 = performance.now()
            let count = 0
            let tFirst = 0
            await engine.generatePipelined(promptIds, nGen, () => {
              if (count === 0) tFirst = performance.now()
              count++
            })
            const tEnd = performance.now()
            const seconds = (tEnd - t0) / 1000
            const ttftMs = tFirst ? tFirst - t0 : seconds * 1000
            const decodeTokPerS = count > 1 && tFirst ? (count - 1) / ((tEnd - tFirst) / 1000) : count / seconds
            const pf = engine.getLastPrefill()
            if (pf) chunks = pf.chunks
            ttfts.push(ttftMs)
            decodes.push(decodeTokPerS)
            console.log(
              `[ttft] prompt=${promptIds.length} tok (target ${target}) run${r + 1}: ` +
              `ttft ${ttftMs.toFixed(0)}ms · ${count} gen tok · decode ${decodeTokPerS.toFixed(2)} tok/s` +
              (pf ? ` · prefill chunks ${pf.chunks} (reused ${pf.reused})` : ''),
            )
          }
          points.push({
            targetTokens: target,
            promptTokens: promptIds.length,
            ttftMsRuns: ttfts,
            ttftMs: medianOf(ttfts),
            decodeTokPerS: medianOf(decodes),
            chunks,
          })
        }

        console.log('')
        console.log('  prompt tok |  TTFT ms |  ms/prompt-token |  vs shortest | chunks')
        console.log('  -----------+----------+------------------+--------------+-------')
        const base = points[0]
        for (const p of points) {
          console.log(
            `  ${String(p.promptTokens).padStart(10)} | ${p.ttftMs.toFixed(0).padStart(8)} | ` +
            `${(p.ttftMs / p.promptTokens).toFixed(3).padStart(16)} | ` +
            `${(p.ttftMs / base.ttftMs).toFixed(2).padStart(12)}x | ${String(p.chunks).padStart(6)}`,
          )
        }
        return { mode: 'ttft-sweep', engine: 'zero-tvm', runsPerPoint: nRuns, genTokensPerPoint: nGen, points }
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  ;(window as Window & typeof globalThis & { benchBatched?: unknown }).benchBatched =
    async (
      M: number = 4,
      iters: number = 500,
      target: 'ffnDown' | 'oproj' = 'ffnDown',
    ): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[batched-bench] decode in flight — skipping'); return null }
      return engine.benchBatchedFfnDown(M, iters, target)
    }

  ;(window as Window & typeof globalThis & { specSim?: unknown }).specSim =
    async (nTokens: number = 160, N: number = 3, K: number = 3): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[specSim] decode in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        const prompts: Array<{ name: string; messages: Array<{ role: 'system' | 'user'; content: string }> }> = [
          {
            name: 'prose',
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: 'Write a four-sentence explanation of how photosynthesis works.' },
            ],
          },
          {
            name: 'code',
            messages: [
              { role: 'system', content: 'You are a helpful coding assistant. Reply with only code, no prose.' },
              { role: 'user', content: 'Write a TypeScript function that computes the nth Fibonacci number using memoization. Include a short test block that prints fib(10), fib(20), and fib(30).' },
            ],
          },
          {
            name: 'summary',
            messages: [
              { role: 'system', content: 'You are a helpful assistant.' },
              { role: 'user', content: 'List the top five considerations when designing a REST API, with a one-sentence explanation for each.' },
            ],
          },
        ]

        const results = []
        for (const p of prompts) {
          const promptIds = buildChatPromptFor(engine.spec, p.messages, tokenizer)
          const generated: number[] = []
          await engine.generatePipelined(promptIds, nTokens, (id) => { generated.push(id) })
          // Concatenate prompt + generated; PLD keys search over the full
          // sequence (prompt is the richest lookup source).
          const full = [...promptIds, ...generated]
          const sim = summarizePLD(p.name, full, promptIds.length, N, K)
          results.push({ ...sim, generatedTokens: generated.length })
          console.log(
            `[specSim] ${p.name}: ${generated.length} gen tok, ` +
            `hit=${(sim.hitRate * 100).toFixed(0)}%, ` +
            `α=${(sim.acceptanceRate * 100).toFixed(0)}%, ` +
            `accepted/step=${sim.meanAcceptedPerStep.toFixed(2)}/${K}, ` +
            `theoretical speedup=${sim.theoreticalSpeedup.toFixed(2)}×`
          )
        }
        console.table(results.map(r => ({
          prompt: r.name,
          gen: r.generatedTokens,
          'hit %': +(r.hitRate * 100).toFixed(1),
          'accept α': +(r.acceptanceRate * 100).toFixed(1),
          'accepted/step': +r.meanAcceptedPerStep.toFixed(2),
          speedup: +r.theoreticalSpeedup.toFixed(2),
        })))
        return results
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  // ── Prefill throughput: `await benchPrefill(800, 3)` ──────────────────────
  // Builds a chat prompt of ≈ nTokens (repeating a fixed paragraph), then for
  // each run drops the engine's absorbed-token record (resetKVTracking → full
  // prefill) and measures time-to-first-token. A/B chunked prefill with
  // ?chunk=0 in the URL; the same numbers are comparable across builds.
  ;(window as Window & typeof globalThis & { benchPrefill?: unknown }).benchPrefill =
    async (nTokens: number = 800, nRuns: number = 3): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[benchPrefill] decode in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        const para =
          'Photosynthesis is the process by which green plants convert light energy into chemical energy, ' +
          'storing it in the bonds of glucose molecules that fuel growth, respiration, and reproduction. '
        // Grow the user message until the templated prompt reaches nTokens.
        let reps = 4
        let promptIds: number[] = []
        const build = (r: number) =>
          buildChatPromptFor(engine.spec, [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: para.repeat(r) + '\nSummarize the passage above in one sentence.' },
          ], tokenizer)
        promptIds = build(reps)
        while (promptIds.length < nTokens) {
          reps = Math.max(reps + 1, Math.ceil((reps * nTokens) / promptIds.length))
          promptIds = build(reps)
        }
        console.log(`[benchPrefill] prompt: ${promptIds.length} tokens (target ${nTokens})`)
        const runs: { ttftMs: number; tokPerS: number }[] = []
        for (let r = 0; r <= nRuns; r++) {
          engine.resetKVTracking()   // force a full prefill every run
          const t0 = performance.now()
          await engine.generatePipelined(promptIds, 1, () => {})
          const ttftMs = performance.now() - t0
          const tokPerS = promptIds.length / (ttftMs / 1000)
          const label = r === 0 ? 'warmup' : `run${r}/${nRuns}`
          console.log(`[benchPrefill] ${label}: ${ttftMs.toFixed(0)} ms to first token = ${tokPerS.toFixed(1)} prefill tok/s`)
          if (r > 0) runs.push({ ttftMs, tokPerS })
        }
        const sorted = runs.map((x) => x.tokPerS).sort((a, b) => a - b)
        const median = sorted[Math.floor(sorted.length / 2)]
        console.log(`[benchPrefill] median: ${median.toFixed(1)} prefill tok/s (${engine.getLastPrefill()?.chunks ?? 0} chunks/run)`)
        return { promptTokens: promptIds.length, runs, median, lastPrefill: engine.getLastPrefill() }
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  // ── Multi-turn TTFT: `await benchTurns()` ─────────────────────────────────
  // Simulates a 3-turn conversation (~800 absorbed tokens entering turn 3) and
  // reports each turn's time-to-first-token plus the engine's reuse stats.
  // A/B cross-turn prefix reuse with ?reuse=0 in the URL.
  ;(window as Window & typeof globalThis & { benchTurns?: unknown }).benchTurns =
    async (genTokens: number = 120): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[benchTurns] decode in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        const filler =
          'The city library reopened after a two-year renovation that added a glass atrium, ' +
          'a children\'s wing, quiet study pods, and a rooftop reading garden overlooking the river. '
        const turns = [
          `Here is some background material: ${filler.repeat(9)}\nWhat changed in the renovation? Answer in detail.`,
          `More context: ${filler.repeat(9)}\nHow do the study pods differ from the reading garden?`,
          'Now, in one short sentence: what overlooks the river?',
        ]
        engine.resetKVTracking()   // fresh conversation
        const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
          { role: 'system', content: 'You are a helpful, concise assistant.' },
        ]
        const results: unknown[] = []
        for (let i = 0; i < turns.length; i++) {
          history.push({ role: 'user', content: turns[i] })
          const promptIds = buildChatPromptFor(engine.spec, history, tokenizer)
          const generated: number[] = []
          const t0 = performance.now()
          let ttftMs = 0
          await engine.generatePipelined(promptIds, genTokens, (id) => {
            if (generated.length === 0) ttftMs = performance.now() - t0
            generated.push(id)
          })
          history.push({ role: 'assistant', content: tokenizer.decode(generated) })
          const info = engine.getLastPrefill()
          console.log(
            `[benchTurns] turn ${i + 1}: prompt ${promptIds.length} tok, ` +
            `reused ${info?.reused ?? 0}, chunks ${info?.chunks ?? 0}, ` +
            `TTFT ${ttftMs.toFixed(0)} ms, generated ${generated.length}`,
          )
          results.push({ turn: i + 1, promptTokens: promptIds.length, ttftMs, ...info })
        }
        return results
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  // ── Prefix-reuse debug assertion: `await checkReuse()` ────────────────────
  // Runs a short turn 1, then compares turn-2 logits computed via the reused
  // prefix vs a fresh full prefill (engine.debugCompareReuse). Both passes
  // replay per token, so when turn 1's prefill was also per-token the diff is
  // exactly 0. With the hybrid CHUNKED prefill enabled, the absorbed prefix
  // was built by the batched-matmul kernels, whose reduction order differs
  // from the per-token GEMV — a small nonzero diff there measures
  // chunked-vs-per-token numerics (same class as the sg/tiled variant
  // differences), NOT a reuse-rule violation. For the bit-exact reuse
  // assertion on hybrid specs, run with ?chunk=0.
  ;(window as Window & typeof globalThis & { checkReuse?: unknown }).checkReuse =
    async (): Promise<unknown> => {
      if (ctx.isBusy()) { console.warn('[checkReuse] decode in flight — skipping'); return null }
      ctx.setBusy(true)
      try {
        engine.resetKVTracking()
        const history: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
          { role: 'system', content: 'You are a helpful, concise assistant.' },
          { role: 'user', content: 'Name three primary colors.' },
        ]
        const t1 = buildChatPromptFor(engine.spec, history, tokenizer)
        const generated: number[] = []
        await engine.generatePipelined(t1, 48, (id) => generated.push(id))
        history.push({ role: 'assistant', content: tokenizer.decode(generated) })
        history.push({ role: 'user', content: 'Which of those is the warmest?' })
        const t2 = buildChatPromptFor(engine.spec, history, tokenizer)
        const res = await engine.debugCompareReuse(t2)
        const chunked = (engine.getLastPrefill()?.chunks ?? 0) > 0
        const verdict = res.maxAbsDiff === 0
          ? ' (bit-exact ✓)'
          : chunked
            ? ' (nonzero: prefix built by CHUNKED prefill vs per-token replay — kernel-variant numerics; rerun with ?chunk=0 for the bit-exact assertion)'
            : ' (MISMATCH — reuse rules broken!)'
        console.log(
          `[checkReuse] prompt ${res.promptLen} tok, reused prefix ${res.startPos} — ` +
          `logits max|Δ| = ${res.maxAbsDiff} mean|Δ| = ${res.meanAbsDiff}` + verdict,
        )
        return { ...res, chunkedPrefix: chunked }
      } finally {
        ctx.setBusy(false)
        ctx.onIdle()
      }
    }

  console.log('[bench] harness ready — call `await bench(128, 3)` or `await bench(0, 0, true)` for per-kernel profile')
  console.log('[bench] prefill: `await benchPrefill(800, 3)` · multi-turn TTFT: `await benchTurns()` · reuse assertion: `await checkReuse()`')
  console.log('[bench] batched primitive: `await benchBatched(4, 500)` — ffnDown weight-reuse falsifiability test')
  console.log('[bench] PLD acceptance sim: `await specSim(160, 3, 3)` — measures spec-decode upside without GPU changes')
}
