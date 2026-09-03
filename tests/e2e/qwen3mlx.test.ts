/**
 * E2E TESTS — Qwen3-4B-MLX (?model=qwen3mlx) chat + validate pages
 *
 * Same shape as the Qwen3 suite (qwen.test.ts): the MLX-affine checkpoint
 * runs the same architecture through a different quant layout (group-64
 * affine with biases vs symmetric group-32), so it gets the same lexical
 * battery. Fixture drift between the two formats shows up here, not in a
 * kernel dump.
 *
 * GATED on the local weight mirror (.weights-local/Qwen3-4B-4bit). If the
 * mirror isn't primed the whole file skips LOUDLY instead of failing.
 *
 * Prime with:  node scripts/download-weights.mjs --model qwen3mlx
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3-4B-4bit')
const HAVE_MLX = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_MLX) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen3mlx e2e] SKIPPING — MLX checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it with: node scripts/download-weights.mjs --model qwen3mlx\n',
  )
}

// First run copies the checkpoint from the local mirror into OPFS; cached after.
const BOOT_TIMEOUT_MS = 10 * 60 * 1000
const GEN_TIMEOUT_MS = 120 * 1000

describe.skipIf(!HAVE_MLX)('qwen3-4b-mlx (?model=qwen3mlx)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen3mlx')
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
    console.log(`[qwen3mlx chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })

  test('validate battery: 5 prompts, lexically sensible outputs', async () => {
    const page = await newPage('/validate.html?model=qwen3mlx')

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
    console.log('[qwen3mlx validate] continuations:')
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
