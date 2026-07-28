/**
 * DEVTOOLS BENCH HARNESSES (chat page)
 *
 * Wires window.bench / window.benchBatched / window.specSim so the chat page
 * can be benchmarked from the console (or preview_eval) without touching the
 * chat UI. Imported by chat.ts for its side effects via installBenchConsole().
 */

import { Tokenizer } from './tokenizer.js'
import { buildChatPromptFor } from './model-select.js'
import { summarize as summarizePLD } from './spec-sim.js'
import type { DecodeEngine } from './engine-core.js'

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

        const runs: { tokens: number; seconds: number; tokPerS: number }[] = []
        for (let r = 0; r <= nRuns; r++) {
          const label = r === 0 ? 'warmup' : `run${r}/${nRuns}`
          const t0 = performance.now()
          let count = 0
          await engine.generatePipelined(promptIds, nTokens, () => { count++ })
          const seconds = (performance.now() - t0) / 1000
          const tokPerS = count / seconds
          console.log(`[bench] ${label}: ${count} tok / ${seconds.toFixed(2)}s = ${tokPerS.toFixed(2)} tok/s`)
          if (r > 0) runs.push({ tokens: count, seconds, tokPerS })
        }
        const sorted = runs.map(r => r.tokPerS).sort((a, b) => a - b)
        const median = sorted[Math.floor(sorted.length / 2)]
        const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length
        const min = sorted[0]
        const max = sorted[sorted.length - 1]
        const summary = { runs, median, mean, min, max }
        console.log(`[bench] summary: median=${median.toFixed(2)} mean=${mean.toFixed(2)} min=${min.toFixed(2)} max=${max.toFixed(2)} tok/s`)
        return summary
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

  console.log('[bench] harness ready — call `await bench(128, 3)` or `await bench(0, 0, true)` for per-kernel profile')
  console.log('[bench] batched primitive: `await benchBatched(4, 500)` — ffnDown weight-reuse falsifiability test')
  console.log('[bench] PLD acceptance sim: `await specSim(160, 3, 3)` — measures spec-decode upside without GPU changes')
}
