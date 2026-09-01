/**
 * E2E — three things the ENTRANCE'S SCENE must keep, all of which broke
 * silently and none of which any existing test could see. No model boots here:
 * every assertion is about the scene as it renders, so the whole file runs in
 * seconds and never touches a weight.
 *
 * 1. REDUCED MOTION ACTUALLY WINS. `@media (prefers-reduced-motion: reduce)`
 *    blocks in landing.css were losing specificity fights to rules further
 *    down the same file — `.cs-fog i` (0,1,1) to `.cs-fog i:nth-child(1)`
 *    (0,2,1), and `#models .mb-pedestal::after` (1,1,1) to
 *    `#models .mb-art.cs-in .mb-pedestal::after` (1,2,1), where .cs-in is the
 *    RESTING state, applied unconditionally by selectFx(). Measured under
 *    reduce: fogA (46s), fogB (58s) and ringPulse (5s) running on / — and on
 *    /share.html, which inherited all three when it adopted the scene.
 *    document.getAnimations() is the only honest check: it reads what the
 *    cascade actually resolved to, not what a rule says.
 *
 * 2. THE FLOOR RING IS THE CHARACTER'S RING AT EVERY VIEWPORT HEIGHT. The ring
 *    is sized as a share of .mb-art and anchored to its bottom, which is only
 *    correct while the canvas fills that box. It stopped filling it above
 *    ~926px of viewport height, where the 500px half of min(54dvh,500px)
 *    binds and the 1fr stage row keeps growing: at 1920x1080 the box was
 *    618.39 tall against a 500px canvas, so the ring grew to 210.25 (0.42 of
 *    the character rather than 0.34) and its bottom sat 46.85px BELOW the
 *    feet. The invariant is one line — the canvas fills .mb-art — and it must
 *    hold below the cap, at it, and above it.
 *
 * 3. A .cs-room STRIP MUST NOT COST THE CONVERSATION ITS HEIGHT ON A PHONE.
 *    In flow the strip is a block above .chat-main, and on a phone it is
 *    bigger than the panel: on the entrance at 360x740 it took 333.48px of a
 *    416.91px panel, leaving .chat-main 0px and pushing the composer 83px out
 *    of the panel's bottom edge. share.html's guest panel carries the same
 *    component in the same place, boots nothing to render it, and so is where
 *    the rule can be pinned honestly.
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

async function sized(path: string, width: number, height: number): Promise<Page> {
  const page = await newPage(path)
  await page.setViewport({ width, height, deviceScaleFactor: 1 })
  await page.reload({ waitUntil: 'networkidle0' })
  return page
}

describe('prefers-reduced-motion is not a suggestion', () => {
  for (const path of ['/', '/share.html']) {
    test(`${path} runs no animation under reduce`, async () => {
      const page = await newPage(path)
      try {
        const cdp = await page.createCDPSession()
        await cdp.send('Emulation.setEmulatedMedia', {
          features: [{ name: 'prefers-reduced-motion', value: 'reduce' }],
        })
        await page.reload({ waitUntil: 'networkidle0' })
        // The scene mounts, then animates. Give it a beat so an animation that
        // starts late is caught rather than missed.
        await new Promise((r) => setTimeout(r, 2_000))
        const running = await page.evaluate(() => {
          const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
          const names = document.getAnimations().map(
            (a) => (a as unknown as { animationName?: string }).animationName ?? '?'
          )
          return { reduce, names }
        })
        // If the emulation did not take, the rest of this test proves nothing.
        expect(running.reduce, 'CDP media emulation did not apply').toBe(true)
        expect(running.names, `${path} animates under reduce`).toEqual([])
      } finally {
        await page.close()
      }
    }, 60_000)
  }
})

describe('the floor ring belongs to the character, not to the viewport', () => {
  // 900 is below the 500px cap, 982 is just above it, 1080 is well above —
  // the three heights where the old rule read 149.05, 176.92 and 210.25 for a
  // character that was 438.39, 500 and 500 tall.
  for (const [w, h] of [[1512, 900], [1512, 982], [1920, 1080]] as const) {
    test(`${w}x${h}: the canvas fills .mb-art and the ring is 0.34 of it`, async () => {
      const page = await sized('/', w, h)
      try {
        await page.waitForSelector('#models .mb-pedestal', { timeout: 20_000 })
        const m = await page.evaluate(() => {
          const rect = (sel: string): DOMRect | null =>
            document.querySelector(sel)?.getBoundingClientRect() ?? null
          const art = rect('#models .mb-art')
          const canvas = rect('#models .mb-mascot')
          const ring = rect('#models .mb-pedestal')
          if (!art || !canvas || !ring) return null
          return {
            artH: art.height, canvasH: canvas.height, ringH: ring.height,
            // Positive = the canvas hangs below its own box, which is what
            // detaches the ring from the feet.
            canvasPastArt: canvas.bottom - art.bottom,
          }
        })
        expect(m, 'the stage did not render').not.toBeNull()
        const { artH, canvasH, ringH, canvasPastArt } = m!
        expect(canvasH, 'the canvas no longer fills .mb-art').toBeCloseTo(artH, 1)
        expect(canvasPastArt, 'the canvas hangs outside .mb-art').toBeCloseTo(0, 1)
        expect(ringH / canvasH, 'the ring is no longer 34% of the character').toBeCloseTo(0.34, 2)
      } finally {
        await page.close()
      }
    }, 60_000)
  }
})

describe('a room strip costs the conversation nothing on a phone', () => {
  for (const [w, h] of [[390, 844], [360, 740]] as const) {
    test(`${w}x${h}: showing the strip does not shrink .chat-main`, async () => {
      const page = await sized('/share.html#0123456789abcdef0123456789abcdef', w, h)
      try {
        await page.waitForSelector('.cs-chat .cs-room', { timeout: 20_000 })
        const m = await page.evaluate(() => {
          const panel = document.querySelector('.cs-chat')!
          const strip = panel.querySelector<HTMLElement>('.cs-room')!
          const main = panel.querySelector('.chat-main')!
          const composer = panel.querySelector('.composer-wrap')!
          const hiddenH = main.getBoundingClientRect().height
          strip.hidden = false
          strip.classList.remove('hidden')
          const shownH = main.getBoundingClientRect().height
          return {
            hiddenH, shownH,
            stripH: strip.getBoundingClientRect().height,
            panelBottom: panel.getBoundingClientRect().bottom,
            composerBottom: composer.getBoundingClientRect().bottom,
          }
        })
        expect(m.stripH, 'the strip did not render').toBeGreaterThan(0)
        expect(m.shownH, 'the strip took height from the conversation')
          .toBeCloseTo(m.hiddenH, 1)
        expect(m.composerBottom, 'the composer left the panel')
          .toBeLessThanOrEqual(m.panelBottom + 1)
      } finally {
        await page.close()
      }
    }, 60_000)
  }
})
