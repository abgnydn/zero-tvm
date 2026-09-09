#!/usr/bin/env node
// CAPTURE MASCOT — render the entrance character to a PNG for the README.
//
// The mascot is not an asset. It is drawn per frame by src/mascot.ts through a
// WebGPU pipeline, re-skinned from the selected model's palette, so there is no
// file to point a README at — which is why README.md still shows a screenshot
// of a landing page that no longer exists.
//
// So this renders it the only way it can be rendered: a real browser with a
// real adapter. Needs a GPU, and it competes with anything else using one.
//
// Writes two files:
//   docs/mascot.png   the character alone, transparent, 2x
//   docs/banner.png   a wide banner — character, wordmark, tagline — composed
//                     in the page so it uses the site's own fonts and palette
//                     rather than an approximation of them
//
// docs/banner.svg is the committed FALLBACK the README points at, authored by
// hand from public/tokens.css so it ships from any machine. Point the README at
// banner.png only once this has actually been run — the README referenced a
// banner.png that did not exist for about an hour, and rendered a broken image
// on the repo's front page.
//
//   node scripts/capture-mascot.mjs                 # default model's palette
//   MODEL=qwen36q3 node scripts/capture-mascot.mjs
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
// The group NAME to select, e.g. "Qwen3-30B-A3B". Not a ?model= param: the
// entrance starts at `gi = 0` and ignores the URL entirely (landing.ts:231), so
// a capture run with MODEL=qwen30b silently rendered the DEFAULT character and
// the banner shipped with the wrong one. Selection has to be a click.
const WANT = process.env.MASCOT ?? process.env.MODEL ?? ''
const OUT = join(ROOT, 'docs')

await startHarness()
try {
  const page = await newPage('/index.html')
  await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 2 })

  // The entrance draws the mascot on a canvas inside the model sheet. Wait for
  // it to have painted rather than for a timeout: an empty canvas screenshots
  // perfectly happily and the failure would be a blank banner in the README.
  await page.waitForFunction(() => {
    const c = document.querySelector('canvas.mb-mascot')
    return c instanceof HTMLCanvasElement && c.width > 0 && c.height > 0
  }, { timeout: 120_000, polling: 250 })
  // Two animation frames after first paint, so the idle animation is settled.
  await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))))

  // Pick the character by clicking its roster dot. Each dot carries the group
  // name in its title, so this matches on what the UI itself displays.
  if (WANT) {
    const picked = await page.evaluate((want) => {
      const dots = [...document.querySelectorAll('.mb-dot')]
      const hit = dots.find((d) => (d.getAttribute('title') || '').toLowerCase().includes(want.toLowerCase()))
      if (!hit) return { ok: false, have: dots.map((d) => d.getAttribute('title')) }
      hit.click()
      return { ok: true, name: hit.getAttribute('title') }
    }, WANT)
    if (!picked.ok) {
      throw new Error(`no roster entry matching "${WANT}". Available: ${picked.have.join(', ')}`)
    }
    console.log(`selected: ${picked.name}`)
    // The mascot re-skins on selection; give it frames to settle before the shot.
    await page.evaluate(() => new Promise((r) => setTimeout(() => requestAnimationFrame(() => requestAnimationFrame(r)), 400)))
  }

  const canvas = await page.$('canvas.mb-mascot')
  if (!canvas) throw new Error('no mascot canvas on the entrance')
  mkdirSync(OUT, { recursive: true })
  // omitBackground does nothing here — the mascot canvas PAINTS its own
  // background, so the screenshot is opaque and the banner showed the character
  // inside a visible dark rectangle. The fix is to composite over that exact
  // colour rather than to try to remove it, so the seam disappears.
  const mascot = await canvas.screenshot({ encoding: 'binary' })
  writeFileSync(join(OUT, 'mascot.png'), mascot)
  console.log(`docs/mascot.png  ${(mascot.length / 1024).toFixed(0)} KB`)

  // Compose the banner IN the page: it already has the fonts loaded and the
  // palette in CSS custom properties, so nothing here has to guess at either.
  const dataUrl = `data:image/png;base64,${Buffer.from(mascot).toString('base64')}`
  await page.evaluate(async (src) => {
    const img = new Image()
    img.src = src
    await img.decode()
    const W = 1600, H = 500
    const c = document.createElement('canvas')
    c.width = W; c.height = H
    c.id = 'banner-capture'
    const g = c.getContext('2d')
    const css = getComputedStyle(document.documentElement)
    const ink = css.getPropertyValue('--text').trim() || '#edf0f6'
    const accent = css.getPropertyValue('--accent').trim() || '#f0a860'
    // Sample the mascot canvas's OWN corner pixel and paint the whole banner
    // that colour. Reading --bg gave a different value and left the character
    // sitting in a visible box; matching the source makes the seam vanish
    // without touching the image.
    const probe = document.createElement('canvas')
    probe.width = probe.height = 1
    probe.getContext('2d').drawImage(img, 0, 0, 1, 1, 0, 0, 1, 1)
    const [pr, pg, pb] = probe.getContext('2d').getImageData(0, 0, 1, 1).data
    const bg = `rgb(${pr}, ${pg}, ${pb})`
    g.fillStyle = bg; g.fillRect(0, 0, W, H)
    // Character on the right, text on the left — the same reading order the
    // entrance uses.
    const h = H * 0.86, w = img.width * (h / img.height)
    g.drawImage(img, W - w - 80, (H - h) / 2, w, h)
    const mono = css.getPropertyValue('--mono').trim() || 'ui-monospace, monospace'
    g.fillStyle = ink
    g.font = `600 96px ${mono}`
    g.textBaseline = 'alphabetic'
    g.fillText('zero-tvm', 96, 210)
    g.fillStyle = accent
    g.fillRect(96, 244, 260, 4)
    g.fillStyle = ink
    g.font = `400 30px ${mono}`
    g.globalAlpha = 0.82
    g.fillText('LLM inference in the browser,', 96, 316)
    g.fillText('on hand-written WGSL.', 96, 360)
    document.body.appendChild(c)
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', zIndex: '99999' })
  }, dataUrl)

  const bannerEl = await page.$('#banner-capture')
  const banner = await bannerEl.screenshot({ encoding: 'binary' })
  writeFileSync(join(OUT, 'banner.png'), banner)
  console.log(`docs/banner.png  ${(banner.length / 1024).toFixed(0)} KB`)
  await page.close()
} finally {
  await stopHarness()
}
