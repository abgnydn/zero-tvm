/**
 * E2E TESTS — Qwen3.5-4B (?model=qwen35) chat + validate pages
 *
 * Same shape as the Qwen3 suite (qwen.test.ts), but boots the hybrid
 * GatedDeltaNet + gated-attention spec: 24 GDN layers + 8 attention layers
 * (GQA 16/4, head_dim 256, partial RoPE 64, sigmoid attention gate),
 * byte-level BPE with the renumbered 248k-vocab specials, ChatML template.
 *
 * GATED on the local weight mirror: these tests need the ~2.6 GB Qwen3.5-4B
 * snapshot in .weights-local/ (there is no download gate fallback budgeted
 * for CI). If the mirror isn't primed the whole file skips LOUDLY instead
 * of failing, so the Phi-3/Qwen3 e2e suites stay green on machines without it.
 *
 * Prime with:  node scripts/download-weights.mjs --model qwen35
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const QWEN35_MIRROR = resolve(process.cwd(), '.weights-local/Qwen3.5-4B-q4f16_1-MLC')
// Newer MLC repos ship tensor-cache.json (the renamed ndarray-cache.json).
const HAVE_QWEN35 = existsSync(resolve(QWEN35_MIRROR, 'tensor-cache.json'))

if (!HAVE_QWEN35) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen35 e2e] SKIPPING — Qwen3.5-4B local mirror not found at\n' +
      `  ${QWEN35_MIRROR}\n` +
      '  Prime it with: node scripts/download-weights.mjs --model qwen35\n',
  )
}

// First run copies ~2.6 GB from the local mirror into OPFS; cached after.
const BOOT_TIMEOUT_MS = 10 * 60 * 1000
const GEN_TIMEOUT_MS = 180 * 1000

describe.skipIf(!HAVE_QWEN35)('qwen3.5-4b (?model=qwen35)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen35')
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
    // the Phi-3/Qwen3 tests). This is the stop-behavior assertion — the
    // renumbered ChatML stop ids (<|im_end|>=248046 / <|endoftext|>=248044)
    // must actually terminate the decode loop well before the 500-token cap.
    await page.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).hidden,
      { timeout: GEN_TIMEOUT_MS, polling: 100 },
    )

    const aiText = await page.$eval('.msg.ai:last-of-type', (el) => el.textContent || '')
    console.log(`[qwen35 chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
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
    const page = await newPage('/validate.html?model=qwen35')

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
    console.log('[qwen35 validate] continuations:')
    for (const r of results) {
      console.log(`  ${r.label}: ${JSON.stringify(r.cont.slice(0, 60))}`)
    }

    // Same lexical bar as the Phi-3/Qwen3 batteries: only the load-bearing
    // token, no exact-string pinning (greedy decode phrasing varies).
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
