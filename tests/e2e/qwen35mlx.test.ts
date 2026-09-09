/**
 * E2E TESTS — Qwen3.5-9B-MLX (?model=qwen35mlx) chat + validate pages
 *
 * Same shape as the Qwen3.5 suite (qwen35.test.ts): the 9B MLX-affine
 * checkpoint runs the hybrid GDN architecture at a different scale and
 * quant layout, so it gets the same lexical battery. Scale- or
 * format-specific regressions (GDN state, int8 KV, chunked prefill at 9B
 * widths) surface here as wrong words, not wrong bytes.
 *
 * GATED on the local weight mirror (.weights-local/Qwen3.5-9B-MLX-4bit). If
 * the mirror isn't primed the whole file skips LOUDLY instead of failing.
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3.5-9B-MLX-4bit')
const HAVE_MLX = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_MLX) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen35mlx e2e] SKIPPING — MLX checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it by downloading lmstudio-community/Qwen3.5-9B-MLX-4bit there.\n',
  )
}

// 9B of weights take a while to copy into OPFS on first run; cached after.
const BOOT_TIMEOUT_MS = 15 * 60 * 1000
const GEN_TIMEOUT_MS = 180 * 1000

describe.skipIf(!HAVE_MLX)('qwen3.5-9b-mlx (?model=qwen35mlx)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen35mlx')
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
    console.log(`[qwen35mlx chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })

  test('validate battery: 5 prompts, lexically sensible outputs', async () => {
    const page = await newPage('/validate.html?model=qwen35mlx')

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
    console.log('[qwen35mlx validate] continuations:')
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
