/**
 * E2E TESTS — Zero-TVM chat + validate pages
 *
 * These exercise the same code paths a real user hits: load the HTML, click
 * the start button, wait for the boot pipeline (GPU + tokenizer + weights +
 * KV + shaders + warmup), then drive the page and assert on what it shows.
 *
 * What each test proves:
 *   1. chat boots and produces a non-trivial first reply  → engine is alive
 *   2. chat second message is steady-state                → no per-message warmup
 *      and warmup in bootEngine actually killed the cold-start outlier
 *   3. validate runs all 5 prompts and the outputs make   → live forward pass
 *      lexical sense ("Paris", "42", "len", "Yes")          generalizes beyond the
 *                                                            single hard-coded
 *                                                            per-shader test prompt
 */

import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import type { Page } from 'puppeteer'
import { bootAndWaitReady, newPage, startHarness, stopHarness } from './harness.js'

const BOOT_TIMEOUT_MS = 5 * 60 * 1000 // first run downloads ~2 GB
const GEN_TIMEOUT_MS = 90 * 1000

beforeAll(async () => {
  await startHarness()
}, 60_000)

afterAll(async () => {
  await stopHarness()
})

// Shared between the two chat tests so the second one can compare against the
// first one's measurement and reuse the booted page.
const chatState: { page: Page | null; coldFirstTokenMs: number | null } = {
  page: null,
  coldFirstTokenMs: null,
}

describe('zero-tvm.html', () => {
  test('boots, sends "hi", produces a reply', async () => {
    const page = await newPage('/zero-tvm.html')
    chatState.page = page

    await bootAndWaitReady(page, BOOT_TIMEOUT_MS)

    await page.type('#inp', 'hi')
    const t0 = Date.now()
    await page.click('#btn')

    // First-token latency: time from submit to first non-empty AI message.
    await page.waitForFunction(
      () => {
        const ais = document.querySelectorAll('.msg.ai')
        const last = ais[ais.length - 1]
        return !!last && (last.textContent?.length ?? 0) > 0
      },
      { timeout: GEN_TIMEOUT_MS, polling: 30 }
    )
    chatState.coldFirstTokenMs = Date.now() - t0
    console.log(`[chat] cold first-token: ${chatState.coldFirstTokenMs} ms`)

    // Wait for full generation to finish (send button re-enabled).
    await page.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).disabled,
      { timeout: GEN_TIMEOUT_MS, polling: 100 }
    )

    const aiText = await page.$eval(
      '.msg.ai:last-of-type',
      (el) => el.textContent || ''
    )
    console.log(`[chat] reply: ${JSON.stringify(aiText.slice(0, 60))}`)
    expect(aiText.length).toBeGreaterThan(5)
  })

  test('second message is steady-state (no per-message coldstart)', async () => {
    const page = chatState.page
    expect(page, 'first chat test must run before this').not.toBeNull()
    expect(chatState.coldFirstTokenMs).not.toBeNull()

    const before = await page!.$$eval('.msg.ai', (els) => els.length)

    await page!.type('#inp', 'what is 2+2?')
    const t0 = Date.now()
    await page!.click('#btn')

    await page!.waitForFunction(
      (n: number) => {
        const ais = document.querySelectorAll('.msg.ai')
        return ais.length > n && (ais[ais.length - 1].textContent?.length ?? 0) > 0
      },
      { timeout: GEN_TIMEOUT_MS, polling: 30 },
      before
    )
    const warmFirstTokenMs = Date.now() - t0
    console.log(
      `[chat] warm first-token: ${warmFirstTokenMs} ms (cold was ${chatState.coldFirstTokenMs} ms)`
    )

    // The post-warmup numbers we saw by hand: cold ~1.4s, warm ~1.4s, both
    // sub-2s on an M2 Pro. Since bootEngine now warms up the pipelines before
    // the badge flips to Ready, BOTH chat messages should be steady-state.
    // Allow generous slack for slower machines.
    expect(warmFirstTokenMs).toBeLessThan(8_000)

    // And it should not be dramatically slower than the first (which would
    // indicate the warmup pass regressed).
    expect(warmFirstTokenMs).toBeLessThan(chatState.coldFirstTokenMs! * 3)

    await page!.waitForFunction(
      () => !(document.getElementById('btn') as HTMLButtonElement).disabled,
      { timeout: GEN_TIMEOUT_MS, polling: 100 }
    )
    await page!.close()
    chatState.page = null
  })
})

describe('validate.html', () => {
  test('runs all 5 prompts and the outputs are sensible', async () => {
    const page = await newPage('/validate.html')

    await page.click('#start-btn')

    // Wait for the validation harness to finish all prompts (status flips to
    // "Done"). Generous timeout because this includes boot + warmup + 5
    // prompts × ~1s each.
    await page.waitForFunction(
      () => document.getElementById('status')?.textContent === 'Done',
      { timeout: BOOT_TIMEOUT_MS, polling: 500 }
    )

    const results = await page.$$eval('.prompt-result', (els) =>
      els.map((el) => ({
        label: el.querySelector('.prompt-label')?.textContent?.trim() || '',
        cont: el.querySelector('pre.continuation')?.textContent || '',
      }))
    )

    expect(results.length).toBe(5)

    const byLabel = Object.fromEntries(
      results.map((r) => [r.label, r.cont.toLowerCase()])
    )
    console.log('[validate] continuations:')
    for (const r of results) {
      console.log(`  ${r.label}: ${JSON.stringify(r.cont.slice(0, 60))}`)
    }

    // Each assertion checks the model produced lexically reasonable output.
    // We don't pin exact strings — Phi-3 greedy decode is occasionally weird
    // and tiny fp16 noise can shift token boundaries. We only check the
    // load-bearing token is present.
    expect(byLabel['factual-recall'], 'should mention Paris').toContain('paris')
    expect(byLabel['arithmetic'], 'should contain 42').toMatch(/42|forty-?two/)
    expect(byLabel['code-completion'], 'should mention len').toContain('len')
    expect(byLabel['instruction-follow'], 'should reply yes').toMatch(/\byes\b/)
    // open-ended just needs to mention at least 2 colors.
    const colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white']
    const found = colors.filter((c) => (byLabel['open-ended'] || '').includes(c))
    expect(found.length, `expected at least 2 colors, found ${found}`).toBeGreaterThanOrEqual(2)

    await page.close()
  })
})
