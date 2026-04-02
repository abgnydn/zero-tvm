/**
 * Automated Replay Correctness Test
 *
 * Launches Chrome with WebGPU, runs baseline vs replay,
 * compares token IDs, reports pass/fail.
 *
 * Usage: node test.mjs
 * Requires: vite dev server running on port 5180
 */

import puppeteer from 'puppeteer'

const PORT = 5180
const URL = `http://localhost:${PORT}/test.html`
const TIMEOUT = 300_000 // 5 min max (model load + 2 inference runs)

async function run() {
  console.log('Launching Chrome with WebGPU...')

  const browser = await puppeteer.launch({
    headless: false, // WebGPU needs real GPU
    channel: 'chrome', // use installed Chrome, not Puppeteer's bundled Chromium
    userDataDir: '/tmp/replay-test-chrome-profile', // separate profile to avoid conflicts
    args: [
      '--enable-unsafe-webgpu',
      '--enable-features=Vulkan',
      '--disable-gpu-sandbox',
    ],
  })

  const page = await browser.newPage()

  // Forward console logs
  page.on('console', msg => {
    const type = msg.type()
    const text = msg.text()
    if (type === 'error') console.error(`  [browser] ${text}`)
    else if (text.includes('[replay]') || text.includes('DIVERGENCE') || text.includes('MATCH'))
      console.log(`  [browser] ${text}`)
  })

  console.log(`Opening ${URL}...`)
  await page.goto(URL, { timeout: 60_000 })

  // Poll from Node side (avoids CDP protocol timeout)
  console.log('Waiting for test to complete (baseline + replay + compare)...')
  const startTime = Date.now()

  let result
  while (true) {
    await new Promise(r => setTimeout(r, 3000))
    const status = await page.evaluate(() => window.__testStatus)
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
    if (status !== 'running') {
      result = await page.evaluate(() => ({
        status: window.__testStatus,
        result: window.__testResult,
      }))
      break
    }
    if (Date.now() - startTime > TIMEOUT) {
      result = { status: 'timeout', result: null }
      break
    }
    process.stdout.write(`\r  ...${elapsed}s elapsed`)
  }
  console.log()

  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log(`\nTest completed in ${elapsed}s`)
  console.log('─'.repeat(60))

  const { status, result: testResult } = result

  if (status === 'error') {
    console.error('TEST ERROR:', testResult?.error)
    await browser.close()
    process.exit(2)
  }

  if (status === 'timeout') {
    console.error('TEST TIMEOUT: exceeded', TIMEOUT / 1000, 'seconds')
    await browser.close()
    process.exit(2)
  }

  if (!testResult) {
    console.error('TEST ERROR: no result')
    await browser.close()
    process.exit(2)
  }

  const baselineCount = testResult.baselineTokens
  const replayCount = testResult.replayTokens
  const divergence = testResult.divergence
  const baselineIds = testResult.baselineIds
  const replayIds = testResult.replayIds

  console.log(`Baseline: ${baselineCount} tokens`)
  console.log(`Replay:   ${replayCount} tokens`)

  if (!divergence) {
    console.log('\n  PASS: All tokens match!')
    await browser.close()
    process.exit(0)
  }

  console.log(`\n  FAIL: First divergence at token index ${divergence.index}`)
  console.log(`    Baseline: ${divergence.baselineToken}`)
  console.log(`    Replay:   ${divergence.replayToken}`)

  // Show context
  const start = Math.max(0, divergence.index - 3)
  const end = Math.min(Math.max(baselineCount, replayCount), divergence.index + 8)
  console.log('\n  idx  | baseline | replay   | match')
  console.log('  -----|----------|----------|------')
  for (let i = start; i < end; i++) {
    const b = i < baselineIds.length ? String(baselineIds[i]) : '-'
    const r = i < replayIds.length ? String(replayIds[i]) : '-'
    const match = b === r ? 'OK' : 'DIFF'
    const marker = i === divergence.index ? ' <--' : ''
    console.log(`  ${String(i).padStart(4)} | ${b.padStart(8)} | ${r.padStart(8)} | ${match}${marker}`)
  }

  // Show patch values around divergence
  const patches = testResult.patchValues
  if (patches.length > 0) {
    console.log('\n  Patch values around divergence:')
    const pStart = Math.max(0, divergence.index - 2)
    const pEnd = Math.min(patches.length, divergence.index + 5)
    for (let i = pStart; i < pEnd; i++) {
      const p = patches[i]
      const marker = i === divergence.index ? ' <--' : ''
      console.log(`    [${i}] token_id=${p.tokenId}, counters=[${p.counters.join(',')}]${marker}`)
    }
  }

  await browser.close()
  process.exit(1)
}

run().catch(e => {
  console.error('Fatal:', e)
  process.exit(2)
})
