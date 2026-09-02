/**
 * E2E — THE RITE CARD MUST NOT PUSH THE COMPOSER OUT OF THE PANEL.
 *
 * `.cs-boot` is the second row of the phone chat grid, and it was `auto`: a
 * row is whatever its content asks for, and a boot FAILURE is the tallest the
 * card ever gets — the title turns into 'The summoning failed', the real
 * reason is printed, and showBootError appends a Retry button. Measured at
 * 360x740 before the fix: card 274.08px of a 416.91px panel, `.chat-main`
 * 0.00px, and the composer 24.45px BELOW the panel's bottom edge. Not only the
 * failure path — mid-rite with the Rite log open it was 30.66px out. Shorter
 * viewports were worse (97.45px at 360x667, 214.59px at 320x568).
 *
 * This is the SAME defect the room strip had, in the row beside it, and it
 * predates the grid rework: identical on 902a4e4^ (the old flex layout) and on
 * the grid. The rework fixed `.cs-room` and left this row `auto`.
 *
 * The fix caps the row (fit-content of the panel minus the head+composer
 * budget) and scrolls the card, so this file asserts the two things that make
 * a failed boot RECOVERABLE rather than merely visible:
 *
 *   1. the composer stays inside the panel — you can still type, and on the
 *      success path that is the only way out of the panel;
 *   2. the Retry button can be brought on screen, and nothing is hidden to
 *      achieve 1 (the card scrolls; it is not clipped and not display:none).
 *
 * NO MODEL BOOTS HERE. The weight requests are aborted at the network layer,
 * which is what drives the page into its real boot-failed state through the
 * real showBootError — so this runs in seconds and needs no primed mirror.
 */

import { afterAll, beforeAll, describe, expect, test } from 'vitest'
import type { Page } from 'puppeteer'
import { newPage, startHarness, stopHarness } from './harness.js'

beforeAll(async () => {
  await startHarness()
}, 60_000)

afterAll(async () => {
  await stopHarness()
})

/** The entrance at a phone size, driven into its boot-FAILED state. */
async function failedBoot(width: number, height: number): Promise<Page> {
  const page = await newPage('/')
  await page.setViewport({ width, height, deviceScaleFactor: 1 })
  await page.setRequestInterception(true)
  page.on('request', (req) => {
    // The dev mirror and the CDN — everything a boot needs and nothing else.
    if (/local-weights|huggingface\.co/.test(req.url())) {
      void req.abort('failed').catch(() => {})
      return
    }
    void req.continue().catch(() => {})
  })
  await page.reload({ waitUntil: 'networkidle0' })
  // ENTER through the CTA's own listener — the same door a person uses. A
  // trusted click is not needed and puppeteer's would have to hit-test an
  // element the ceremony overlay sits on.
  await page.waitForSelector('#models .mb-cta', { timeout: 20_000 })
  await page.evaluate(() => document.querySelector<HTMLElement>('#models .mb-cta')?.click())
  // The Retry button is appended by showBootError itself: waiting for it is
  // waiting for the real failure surface, not for a class.
  await page.waitForSelector('#retry-download-btn', { timeout: 60_000 })
  return page
}

describe('a failed boot leaves the panel usable on a phone', () => {
  for (const [w, h] of [[360, 740], [390, 844]] as const) {
    test(`${w}x${h}: the rite card cannot push the composer out`, async () => {
      const page = await failedBoot(w, h)
      try {
        const m = await page.evaluate(() => {
          const q = <T extends HTMLElement>(s: string) => document.querySelector<T>(s)!
          const boot = q('#models .cs-boot')
          // The tallest the card ever is: failed, with the Rite log open.
          q<HTMLDetailsElement>('.cs-boot-log').open = true
          const panel = q('#models .cs-chat').getBoundingClientRect()
          const composer = q('#models .composer-wrap').getBoundingClientRect()
          const mainH = q('#models .chat-main').getBoundingClientRect().height
          const cs = getComputedStyle(boot)
          // Can a person reach Retry? Scroll the card the way a thumb would.
          boot.scrollTop = boot.scrollHeight
          const card = boot.getBoundingClientRect()
          const retry = q('#retry-download-btn').getBoundingClientRect()
          return {
            bootH: card.height, mainH,
            composerPastPanel: composer.bottom - panel.bottom,
            display: cs.display, overflowY: cs.overflowY,
            retryReadable: retry.top >= card.top - 1 && retry.bottom <= card.bottom + 1,
            errorShown: q('#loading-error').classList.contains('visible'),
          }
        })
        // The failure surface really is on screen — otherwise the rest of this
        // test is measuring an empty card.
        expect(m.errorShown, 'the boot error is not displayed').toBe(true)
        expect(m.display, 'the rite card was hidden rather than fitted').not.toBe('none')
        expect(m.bootH, 'the rite card did not render').toBeGreaterThan(0)

        expect(
          m.composerPastPanel,
          `the composer left the panel by ${m.composerPastPanel.toFixed(2)}px `
          + `(rite card ${m.bootH.toFixed(2)}, .chat-main ${m.mainH.toFixed(2)})`,
        ).toBeLessThanOrEqual(1)
        expect(m.mainH, 'the conversation row was starved to nothing').toBeGreaterThan(0)
        // Capping without scrolling would trade one broken state for another:
        // the reason and the Retry button would be cut off instead.
        expect(m.overflowY, 'a clipped rite card cannot be read to the end').not.toBe('hidden')
        expect(m.retryReadable, 'Retry cannot be scrolled into view').toBe(true)
      } finally {
        await page.close()
      }
    }, 120_000)
  }
})
