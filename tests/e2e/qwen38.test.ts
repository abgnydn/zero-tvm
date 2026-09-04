/**
 * E2E TESTS — Qwen3.8-27B (?model=qwen38) chat + validate pages
 *
 * The quarantined spec: 27B hybrid GDN, chunk cap held at 256 by
 * maxChunkCap while the corruption threshold sits somewhere in (256, 1024].
 * This suite is the END of that story — a model that boots, generates, and
 * answers sanely at its shipped configuration. It does not replace the
 * long-context sweep (PROMPT=16000 chunk-prefill-test --long), which is the
 * gate that watches the quarantine boundary itself.
 *
 * GATED on the local weight mirror (.weights-local/Qwen3.8-27B-4bit). If
 * the mirror isn't primed the whole file skips LOUDLY instead of failing.
 * Needs a machine that holds ~14 GB of weights (27B at 4-bit).
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3.8-27B-4bit')
const HAVE_QWEN38 = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_QWEN38) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen38 e2e] SKIPPING — checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it by downloading mlx-community/Qwen3.8-27B-4bit there.\n',
  )
}

// 27B of weights take a long while to copy into OPFS on first run; cached after.
const BOOT_TIMEOUT_MS = 20 * 60 * 1000
const GEN_TIMEOUT_MS = 180 * 1000

describe.skipIf(!HAVE_QWEN38)('qwen3.8-27b (?model=qwen38)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen38')
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

    await page.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).hidden,
      { timeout: GEN_TIMEOUT_MS, polling: 100 },
    )

    const aiText = await page.$eval('.msg.ai:last-of-type', (el) => el.textContent || '')
    console.log(`[qwen38 chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })

  test('validate battery: 5 prompts, lexically sensible outputs', async () => {
    const page = await newPage('/validate.html?model=qwen38')

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
    console.log('[qwen38 validate] continuations:')
    for (const r of results) {
      console.log(`  ${r.label}: ${JSON.stringify(r.cont.slice(0, 60))}`)
    }

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
