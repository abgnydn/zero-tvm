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
 *    the rule can be pinned cheaply.
 *
 *    But share.html's strip is the SHORT case — the weight-copy offer is two
 *    elements, 61px at 360x740 against a 181.71px body row, so it never asks
 *    the rule for anything. The half the rule was written for is a strip
 *    TALLER than the cell it sits in, and the fix has two clauses for it:
 *    `max-height: 100%` and `overflow-y: auto`. Dropping the clamp leaves
 *    share.html's numbers byte-identical and every assertion above green
 *    (measured), while the entrance's four-machine plan grows to 372.27px in
 *    a 249.05px row and hangs 15.98px past the panel's bottom edge, over the
 *    composer. So the tall case is rendered too: mountRoomTool driven
 *    directly, the way landing-chat.ts drives it after a boot, which needs no
 *    engine, no tokenizer and no weights.
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

/**
 * A page at a viewport that has SETTLED — asserted, not assumed.
 *
 * `1512x982` failed once, intermittently, with `canvas=482.5` against
 * `art=500`, and it is the TEST that was wrong, not the CSS. The review that
 * chased it did not reproduce it in 34 attempts, nor a second pass in 109
 * observations, and anchored the mechanism: `--cs-char-h: min(54dvh, 500px)`
 * feeds both boxes, so `canvas=482.5` needs 54dvh resolved at 893.5px while
 * `art=500` needs it resolved at >= 926px. A settled layout cannot hold both —
 * forcing dvh=894 gave `art=432.39 canvas=432.39`, coupled, never split. (Those
 * numbers are that review's; what is verified here is that the file passes with
 * the check below in place.) So the two were read either side of a viewport the
 * page had not finished resolving `dvh` against, and a `setViewport` + `reload`
 * that returns before that is a race every assertion after it inherits.
 *
 * The check is on `100dvh` as well as `innerHeight`, because `dvh` is the unit
 * the stylesheet actually resolves against and the two are what disagreed. Both
 * are polled first and then ASSERTED, so a viewport that never settles fails
 * here, naming the numbers, instead of surfacing later as one flaky ratio.
 */
async function sized(path: string, width: number, height: number): Promise<Page> {
  const page = await newPage(path)
  await page.setViewport({ width, height, deviceScaleFactor: 1 })
  await page.reload({ waitUntil: 'networkidle0' })
  const read = (): Promise<{ w: number; h: number; dvh: number }> => page.evaluate(() => {
    // What `min(54dvh, …)` in landing.css resolves to right now, measured the
    // only way a page can measure it: give an element the unit and read it back.
    const probe = document.createElement('div')
    probe.style.cssText =
      'position:fixed;top:0;left:0;width:0;height:100dvh;visibility:hidden;pointer-events:none'
    document.body.appendChild(probe)
    const dvh = probe.getBoundingClientRect().height
    probe.remove()
    return { w: innerWidth, h: innerHeight, dvh }
  })
  let m = await read()
  const t0 = Date.now()
  while (Date.now() - t0 < 5_000
    && (m.w !== width || m.h !== height || Math.abs(m.dvh - height) > 0.5)) {
    await new Promise((r) => setTimeout(r, 100))
    m = await read()
  }
  const asked = `${width}x${height}`
  expect(m.w, `innerWidth is ${m.w}, not the ${width} this test asked for`).toBe(width)
  expect(m.h, `innerHeight is ${m.h}, not the ${height} this test asked for — `
    + `every box measured below would be read against a viewport that is not ${asked}`).toBe(height)
  expect(m.dvh, `100dvh resolves to ${m.dvh}, not ${height} — the stylesheet sizes the `
    + 'character off this unit, so a box read now can disagree with a box read a frame later')
    .toBeCloseTo(height, 1)
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

describe("...and the entrance's own strip is the one that is too tall", () => {
  // The case above is short enough to fit whatever it is given, so it asks the
  // phone rule for nothing. This one does not fit: consent paragraph + four
  // machine-count chips + plan line is 371px of content in a 249.05px body row
  // at 360x740 — the shape that broke, measured at 333.48px for the entrance's
  // plain plan and 371.14px for a four-machine split against a 416.91px panel.
  //
  // mountRoomTool is called the way landing-chat.ts calls it once the engine is
  // live. Everything the strip needs to RENDER is the spec and the split;
  // engine, tokenizer and lock are read only inside the "Open room →" handler,
  // so a stub reaches the geometry without a weight ever being fetched.
  //
  // The HEIGHT is what makes these the tall case, so the pair above's 844 is
  // not reused: at 390x844 the strip measures 353px against a 352.97px row —
  // a tie, and a test that passed on 0.03px of margin would be claiming
  // coverage it does not have. At 740 the same panel gives the conversation
  // ~249px against the same content. Measured, all four:
  //   390x844 -> 353 in 352.97   390x740 -> 353 in 248.97
  //   360x740 -> 371 in 249.05   360x640 -> 371 in 149.05
  for (const [w, h] of [[390, 740], [360, 740]] as const) {
    test(`${w}x${h}: a four-machine plan scrolls inside the body row`, async () => {
      const page = await sized('/', w, h)
      try {
        await page.waitForSelector('#model-browser', { timeout: 20_000 })
        const m = await page.evaluate(async () => {
          // These are dev-server URLs the page resolves, not modules this test
          // file can resolve at compile time. Built with the Function
          // constructor because vitest transforms a literal `import()` in this
          // callback into `__vite_ssr_dynamic_import__`, which does not exist
          // in the browser — a string body is passed through untouched.
          const load = new Function('p', 'return import(p)') as
            (p: string) => Promise<Record<string, unknown>>
          const { panelMarkup } = await load('/src/landing-chat.ts') as unknown as {
            panelMarkup: (s: unknown, b: unknown, l: string, r?: unknown) => string
          }
          const { specForParam, modelBranding } =
            await load('/src/zero-tvm/model-registry.ts') as unknown as {
              specForParam: (p: string) => { layers: number }
              modelBranding: (s: unknown) => { params: string }
            }
          const { splitBounds } = await load('/src/zero-tvm/room-url.ts') as unknown as {
            splitBounds: (layers: number, machines: number) => number[]
          }
          const { mountRoomTool } = await load('/src/landing-room.ts') as unknown as {
            mountRoomTool: (o: Record<string, unknown>) => void
          }

          const root = document.getElementById('model-browser')
          if (!root) return null
          // qwen38 is an MLX checkpoint, so canSplitAcrossDevices() is true and
          // paintPlan draws the whole 1/2/3/4-machine row. On an MLC spec that
          // row is a single dead chip and this stops being the tall case.
          const spec = specForParam('qwen38')
          const brand = modelBranding(spec)
          const bounds = splitBounds(spec.layers, 4)
          const range = { start: bounds[0], end: bounds[1] }

          const panel = document.createElement('section')
          panel.className = 'cs-chat'
          panel.innerHTML = panelMarkup(spec, brand, brand.params, range)
          root.appendChild(panel)
          root.classList.add('cs-chatting')
          // enterChat marks the boot card done once the engine is ready. The
          // cell the strip has to fit in is the one the conversation gets
          // AFTER that, not the one left while the download is on screen.
          panel.querySelector('.cs-boot')?.classList.add('cs-done')

          mountRoomTool({
            root, panel, spec, param: 'qwen38',
            engine: null, tokenizer: null, mascot: null, lock: null,
            poolSlots: 0, openStrip: true,
            stageRange: range,
            split: { bounds: [...bounds], index: 0, ctx: 4096 },
          })

          const strip = panel.querySelector<HTMLElement>('.cs-room')
          const main = panel.querySelector('.chat-main')
          const composer = panel.querySelector('.composer-wrap')
          if (!strip || !main || !composer) return null
          strip.hidden = true
          const hiddenH = main.getBoundingClientRect().height
          strip.hidden = false
          const rect = strip.getBoundingClientRect()
          const mainRect = main.getBoundingClientRect()
          return {
            chips: strip.querySelectorAll('.cs-room-plan .mb-variant').length,
            hiddenH,
            shownH: mainRect.height,
            // What the strip WANTS, before the cell clamps it.
            contentH: strip.scrollHeight,
            stripH: rect.height,
            stripBottom: rect.bottom,
            // The strip shares the `body` grid area with .chat-main, so the
            // conversation's box IS the cell the strip has to stay inside.
            mainBottom: mainRect.bottom,
            panelBottom: panel.getBoundingClientRect().bottom,
            composerBottom: composer.getBoundingClientRect().bottom,
          }
        })
        expect(m, 'the entrance panel or the room strip did not mount').not.toBeNull()
        const s = m!
        // THE PREMISE. Without both of these this is the short case again and
        // the file is back to not covering the failure it was written for.
        expect(s.chips, 'the plan drew no machine-count row').toBe(4)
        // By a MARGIN, not by a pixel: the clamp is only under test while the
        // content genuinely does not fit, and 390x844 sits within 0.03px of
        // its row. A tie must read as "wrong viewport", not as a pass.
        expect(s.contentH,
          `the strip wants ${s.contentH}px in a ${s.shownH}px row — not the tall case`)
          .toBeGreaterThan(s.shownH + 40)

        expect(s.shownH, 'the strip took height from the conversation')
          .toBeCloseTo(s.hiddenH, 1)
        // Both clauses of the rule, and the body row is the bound that matters:
        // a taller panel can absorb the overflow without the strip ever leaving
        // it, and the composer would still be underneath a strip that covers
        // the cell it was told to scroll inside.
        expect(s.stripH, 'the strip is taller than the row it shares with the conversation')
          .toBeLessThanOrEqual(s.shownH + 1)
        expect(s.stripBottom, 'the strip spills out of the body row instead of scrolling inside it')
          .toBeLessThanOrEqual(s.mainBottom + 1)
        expect(s.composerBottom, 'the composer left the panel')
          .toBeLessThanOrEqual(s.panelBottom + 1)
      } finally {
        await page.close()
      }
    }, 60_000)
  }
})
