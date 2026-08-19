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
//   node scripts/capture-mascot.mjs                 # default model's palette
//   MODEL=qwen36q3 node scripts/capture-mascot.mjs
import { mkdirSync, writeFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { startHarness, stopHarness, newPage } from '../tests/e2e/harness.ts'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const MODEL = process.env.MODEL ?? ''
const OUT = join(ROOT, 'docs')

await startHarness()
try {
  const page = await newPage(`/index.html${MODEL ? `?model=${MODEL}` : ''}`)
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

  const canvas = await page.$('canvas.mb-mascot')
  if (!canvas) throw new Error('no mascot canvas on the entrance')
  mkdirSync(OUT, { recursive: true })
  const mascot = await canvas.screenshot({ omitBackground: true, encoding: 'binary' })
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
    const ink = css.getPropertyValue('--ink').trim() || '#e8e6e3'
    const accent = css.getPropertyValue('--accent').trim() || '#e8955a'
    const bg = css.getPropertyValue('--bg').trim() || '#0b1020'
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
