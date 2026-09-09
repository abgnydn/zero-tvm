/**
 * E2E TESTS — Qwen3-30B-A3B (?model=qwen30b) chat page, single prompt.
 *
 * Chat-only by design: a 30B MoE boot copies gigabytes into OPFS on first
 * run and decodes slowly, so the full 5-prompt validate battery belongs on
 * the smaller specs. This file proves the MoE path boots end to end and
 * answers sanely — expert routing, shared expert, MoE combine — in the one
 * configuration no kernel test reaches: the whole model, one real prompt.
 *
 * GATED on the local weight mirror (.weights-local/Qwen3-30B-A3B-4bit). If
 * the mirror isn't primed the whole file skips LOUDLY instead of failing.
 * Needs a machine that holds ~17 GB of weights.
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3-30B-A3B-4bit')
const HAVE_QWEN30B = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_QWEN30B) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen30b e2e] SKIPPING — checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it by downloading mlx-community/Qwen3-30B-A3B-4bit there.\n',
  )
}

// Tens of GB into OPFS on first run; cached after. Chat-only: one prompt.
const BOOT_TIMEOUT_MS = 30 * 60 * 1000
const GEN_TIMEOUT_MS = 300 * 1000

describe.skipIf(!HAVE_QWEN30B)('qwen3-30b-a3b (?model=qwen30b)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen30b')
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
    console.log(`[qwen30b chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })
})
