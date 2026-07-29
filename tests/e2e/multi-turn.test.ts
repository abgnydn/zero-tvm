/**
 * E2E TESTS — multi-turn chat with cross-turn prefix reuse (perf round A)
 *
 * Exercises the engine's absorbed-token tracking end-to-end: turn 2 of a
 * conversation must (a) stay coherent — the model still sees turn 1's
 * content through the reused KV/GDN state — and (b) actually REUSE a
 * non-zero prefix (asserted via the engine's `[engine] prefill:` console
 * line), on both engine families:
 *
 *   - Phi-3 (default page): pure-attention, fused pipeline — KV-slot reuse.
 *   - Qwen3.5 (?model=qwen35): hybrid GDN — reuse additionally requires the
 *     non-rewindable recurrent state to sit exactly at the absorbed
 *     boundary, and the prompt path runs the chunked GDN prefill.
 *
 * The Qwen3.5 half is GATED on the local weight mirror like qwen35.test.ts.
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import type { Page } from 'puppeteer'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const QWEN35_MIRROR = resolve(process.cwd(), '.weights-local/Qwen3.5-4B-q4f16_1-MLC')
const HAVE_QWEN35 = existsSync(resolve(QWEN35_MIRROR, 'tensor-cache.json'))

const BOOT_TIMEOUT_MS = 10 * 60 * 1000
const GEN_TIMEOUT_MS = 180 * 1000

/** Send a chat message and wait for the reply to finish; returns the reply text. */
async function sendTurn(page: Page, text: string): Promise<string> {
  const before = await page.$$eval('.msg.ai', (els) => els.length)
  await page.type('#inp', text)
  await page.click('#btn')
  await page.waitForFunction(
    (n: number) => {
      const btn = document.getElementById('btn') as HTMLButtonElement | null
      return document.querySelectorAll('.msg.ai').length > n && !!btn && !btn.hidden
    },
    { timeout: GEN_TIMEOUT_MS, polling: 100 },
    before,
  )
  return page.$eval('.msg.ai:last-of-type .body', (el) => el.textContent || '')
}

/** Collect the engine's prefill log lines ("[engine] prefill: N tokens, reused prefix R…"). */
function capturePrefillLogs(page: Page): string[] {
  const lines: string[] = []
  page.on('console', (m) => {
    const t = m.text()
    if (t.includes('[engine] prefill:')) lines.push(t)
  })
  return lines
}

const reusedFrom = (line: string): number =>
  Number(/reused prefix (\d+)/.exec(line)?.[1] ?? -1)

describe('multi-turn prefix reuse (phi-3)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('turn 2 stays coherent and reuses the turn-1 prefix', async () => {
    const page = await newPage('/zero-tvm.html')
    const prefillLogs = capturePrefillLogs(page)
    await bootAndWaitReady(page, BOOT_TIMEOUT_MS)

    const reply1 = await sendTurn(
      page,
      'My name is Alice. Please remember it. Now: what is the capital of France? Answer in one short sentence.',
    )
    console.log(`[multi-turn phi3] turn 1: ${JSON.stringify(reply1.slice(0, 80))}`)
    expect(reply1.toLowerCase()).toContain('paris')

    const reply2 = await sendTurn(page, 'What is my name? Answer in one short sentence.')
    console.log(`[multi-turn phi3] turn 2: ${JSON.stringify(reply2.slice(0, 80))}`)
    expect(reply2.toLowerCase()).toContain('alice')

    // Turn 2 must have reused a non-trivial absorbed prefix (at minimum the
    // system prompt + turn 1; the warmup generation makes turn 1's own reuse
    // small, so only turn 2 is asserted).
    const turn2Log = prefillLogs.filter((l) => reusedFrom(l) > 0).pop()
    console.log(`[multi-turn phi3] prefill logs: ${JSON.stringify(prefillLogs)}`)
    expect(turn2Log, 'expected a prefill log with reused prefix > 0').toBeDefined()
    expect(reusedFrom(turn2Log!)).toBeGreaterThan(20)

    await page.close()
  })

  test.skipIf(!HAVE_QWEN35)('qwen3.5 hybrid: turn 2 coherent, GDN state reused, chunked prefill', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen35')
    const prefillLogs = capturePrefillLogs(page)
    await bootAndWaitReady(page, BOOT_TIMEOUT_MS)

    const reply1 = await sendTurn(
      page,
      'My name is Alice. Please remember it. Now: what is the capital of France? Answer in one short sentence.',
    )
    console.log(`[multi-turn qwen35] turn 1: ${JSON.stringify(reply1.slice(0, 80))}`)
    expect(reply1.toLowerCase()).toContain('paris')
    expect(reply1).not.toContain('<think>')

    const reply2 = await sendTurn(page, 'What is my name? Answer in one short sentence.')
    console.log(`[multi-turn qwen35] turn 2: ${JSON.stringify(reply2.slice(0, 80))}`)
    expect(reply2.toLowerCase()).toContain('alice')

    // Hybrid reuse is all-or-nothing: turn 2 either extends EVERY absorbed
    // token (reused == full turn-1 sequence incl. the stop-id overrun) or
    // falls back to 0. Assert the reuse actually engaged — this is the
    // regression gate for the GDN state-boundary rules and the non-thinking
    // template's history rendering.
    console.log(`[multi-turn qwen35] prefill logs: ${JSON.stringify(prefillLogs)}`)
    const turn2Log = prefillLogs.filter((l) => reusedFrom(l) > 0).pop()
    expect(turn2Log, 'expected a prefill log with reused prefix > 0 (GDN state reuse)').toBeDefined()
    expect(reusedFrom(turn2Log!)).toBeGreaterThan(20)

    await page.close()
  })
})
