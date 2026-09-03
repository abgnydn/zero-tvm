/**
 * E2E TESTS — Qwen3.6-35B-A3B 3-bit experts (?model=qwen36q3) chat page.
 *
 * Chat-only by design, same reasoning as qwen30b.test.ts: the resident
 * build most machines can actually run (~20 GB free RAM), proven end to
 * end on one real prompt — GDN recurrence, MoE routing over 3-bit expert
 * stacks, shared expert — in the configuration no kernel test reaches.
 *
 * GATED on the local weight mirror
 * (.weights-local/Qwen3.6-35B-A3B-MLX-q3exp). If the mirror isn't primed
 * the whole file skips LOUDLY instead of failing.
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Qwen3.6-35B-A3B-MLX-q3exp')
const HAVE_QWEN36Q3 = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_QWEN36Q3) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[qwen36q3 e2e] SKIPPING — checkpoint not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it from the converted 3-bit expert build (see CLAUDE.md).\n',
  )
}

// Tens of GB into OPFS on first run; cached after. Chat-only: one prompt.
const BOOT_TIMEOUT_MS = 30 * 60 * 1000
const GEN_TIMEOUT_MS = 300 * 1000

describe.skipIf(!HAVE_QWEN36Q3)('qwen3.6-35b-a3b-q3exp (?model=qwen36q3)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=qwen36q3')
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
    console.log(`[qwen36q3 chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    expect(aiText).not.toContain('<|im_end|>')
    expect(aiText).not.toContain('<|endoftext|>')
    expect(aiText).not.toContain('<think>')

    await page.close()
  })
})
