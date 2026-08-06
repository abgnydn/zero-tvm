/**
 * E2E TEST — Llama-3.2-1B (?model=llama32) chat page
 *
 * Same waiting contract as the Qwen suites: a .msg.ai appears with text, and
 * end-of-generation re-shows #btn (the stop-id assertion — <|eot_id|> must
 * actually terminate the decode loop). First Llama on the engine: llama3
 * header template, llama3-scaled RoPE table, MLX-affine unfused composition.
 *
 * GATED on the local weight mirror (~700 MB):
 *   hf download mlx-community/Llama-3.2-1B-Instruct-4bit \
 *       --local-dir .weights-local/Llama-3.2-1B-Instruct-4bit
 */

import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const MIRROR = resolve(process.cwd(), '.weights-local/Llama-3.2-1B-Instruct-4bit')
const HAVE_LLAMA = existsSync(resolve(MIRROR, 'model.safetensors.index.json'))

if (!HAVE_LLAMA) {
  // eslint-disable-next-line no-console
  console.warn(
    '\n[llama32 e2e] SKIPPING — Llama-3.2-1B local mirror not found at\n' +
      `  ${MIRROR}\n` +
      '  Prime it with: hf download mlx-community/Llama-3.2-1B-Instruct-4bit --local-dir .weights-local/Llama-3.2-1B-Instruct-4bit\n',
  )
}

// First run copies ~700 MB from the local mirror into OPFS; cached after.
const BOOT_TIMEOUT_MS = 10 * 60 * 1000
const GEN_TIMEOUT_MS = 120 * 1000

describe.skipIf(!HAVE_LLAMA)('llama-3.2-1b (?model=llama32)', () => {
  beforeAll(async () => {
    await startHarness()
  }, 60_000)

  afterAll(async () => {
    await stopHarness()
  })

  test('chat boots, generates a reply, and stops cleanly', async () => {
    const page = await newPage('/zero-tvm.html?model=llama32')
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

    // End-of-generation signal: the composer re-shows #btn. This is the
    // stop-behavior assertion — <|eot_id|> (128009) must terminate the decode
    // loop well before the 500-token cap.
    await page.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).hidden,
      { timeout: GEN_TIMEOUT_MS, polling: 100 },
    )

    const aiText = await page.$eval('.msg.ai:last-of-type', (el) => el.textContent || '')
    console.log(`[llama32 chat] reply: ${JSON.stringify(aiText.slice(0, 80))}`)
    expect(aiText.length).toBeGreaterThan(5)
    expect(aiText.toLowerCase()).toContain('paris')
    // Stop/header tokens are control ids, never surfaced as text.
    expect(aiText).not.toContain('<|eot_id|>')
    expect(aiText).not.toContain('<|start_header_id|>')

    await page.close()
  })
})
