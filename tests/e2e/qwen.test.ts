/**
 * E2E TESTS — Qwen3-4B (?model=qwen3) chat + validate pages
 *
 * Same shape as the Phi-3 suite (zero-tvm.test.ts), but boots the Qwen3-4B
 * spec: GQA 32/8, QK-norm (unfused path), byte-level BPE, ChatML template.
 *
 * GATED on the local weight mirror: these tests need the ~2.3 GB Qwen3-4B
 * snapshot in .weights-local/ (there is no download gate fallback budgeted
 * for CI). If the mirror isn't primed the whole file skips LOUDLY instead
 * of failing, so the Phi-3 e2e suite stays green on machines without it.
 *
 * Prime with:  node scripts/download-weights.mjs --model qwen3
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const QWEN_MIRROR = resolve(process.cwd(), '.weights-local/Qwen3-4B-q4f16_1-MLC')
const HAVE_QWEN = existsSync(resolve(QWEN_MIRROR, 'ndarray-cache.json'))

if (!HAVE_QWEN) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen e2e] SKIPPING — Qwen3-4B local mirror not found at\n' +
      `  ${QWEN_MIRROR}\n` +
      '  Prime it with: node scripts/download-weights.mjs --model qwen3\n',
  )
}

// First run copies ~2.3 GB from the local mirror into OPFS; cached after.
const BOOT_TIMEOUT_MS = 10 * 60 * 1000
const GEN_TIMEOUT_MS = 120 * 1000

describe.skipIf(!HAVE_QWEN)('qwen3-4b (?model=qwen3)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen3')
    await bootAndWaitReady(page, BOOT_TIMEOUT_MS)

    await page.type('#inp', 'What is the capital of France? Answer in one short sentence.')
    await page.click('#btn')

    await page.waitForFunction(
      () => {
        const ais = document.querySelectorAll('.msg.ai')
        const last = ais[ais.length - 1]
        return !!last && (last.textContent?.length ?? 0) > 0
      },
      { timeout: GEN_TIMEOUT_MS, polling: 30 },
    )

    // End-of-generation signal: the composer re-shows #btn (same contract as
    // the Phi-3 tests). This is the stop-behavior assertion — the ChatML stop
    // ids (<|im_end|>/<|endoftext|>) must actually terminate the decode loop
    // well before the 500-token cap.
    await page.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).hidden,
      { timeout: GEN_TIMEOUT_MS, polling: 100 },
    )

    const aiText = await page.$eval('.msg.ai:last-of-type', (el) => el.textContent || '')
    console.log(`[qwen chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    // Stop tokens are control ids, never surfaced as text; and v1 chat runs
    // the NON-thinking ChatML template, so no <think> block should leak.
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })

  test('validate battery: 5 prompts, lexically sensible outputs', async () => {
    const page = await newPage('/validate.html?model=qwen3')

    await page.click('#start-btn')

    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'Done',
      { timeout: BOOT_TIMEOUT_MS, polling: 500 },
    )

    const results = await page.$$eval('.prompt-result', (els) =>
      els.map((el) => ({
        label: el.querySelector('.prompt-label')?.textContent?.trim() || '',
        cont: el.querySelector('pre.continuation')?.textContent || '',
      })),
    )

    expect(results.length).toBe(5)

    const byLabel = Object.fromEntries(results.map((r) => [r.label, r.cont.toLowerCase()]))
    console.log('[qwen validate] continuations:')
    for (const r of results) {
      console.log(`  ${r.label}: ${JSON.stringify(r.cont.slice(0, 60))}`)
    }

    // Same lexical bar as the Phi-3 battery: only the load-bearing token,
    // no exact-string pinning (greedy decode phrasing varies).
    expect(byLabel['factual-recall'], 'should mention Paris').toContain('paris')
    expect(byLabel['arithmetic'], 'should contain 42').toMatch(/42|forty-?two/)
    expect(byLabel['code-completion'], 'should mention len').toContain('len')
    expect(byLabel['instruction-follow'], 'should reply yes').toMatch(/\byes\b/)
    const colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white']
    const found = colors.filter((c) => (byLabel['open-ended'] || '').includes(c))
    expect(found.length, `expected at least 2 colors, found ${found}`).toBeGreaterThanOrEqual(2)

    await page.close()
  })
})
