/**
 * Reproduces where onnxruntime-web's WebGPU backend places an ONNX `Scan`.
 *
 * Serves this directory, loads it in Chrome with WebGPU enabled, and prints
 * ORT's own verbose node-placement log. The interesting lines are emitted by
 * ORT itself (session_state.cc / js_execution_provider.cc), not by this script.
 *
 *   python3 build-toy.py                       # writes toy_scan.onnx
 *   cp -r node_modules/onnxruntime-web/dist ./ort-dist
 *   node run.mjs                 # placement
 *   node run.mjs scan-bench.html  # cost of the real Scan on WebGPU
 */
import http from 'node:http'
import { readFile } from 'node:fs/promises'
import { extname, join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import puppeteer from 'puppeteer'

const ROOT = dirname(fileURLToPath(import.meta.url))
const MIME = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript',
               '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' }

const server = http.createServer(async (req, res) => {
  try {
    const p = join(ROOT, decodeURIComponent(req.url.split('?')[0]))
    const buf = await readFile(p)
    res.writeHead(200, { 'Content-Type': MIME[extname(p)] || 'application/octet-stream' })
    res.end(buf)
  } catch { res.writeHead(404); res.end('not found') }
})
await new Promise((r) => server.listen(8731, r))

const browser = await puppeteer.launch({
  headless: 'shell',
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=metal', '--no-sandbox'],
})
const page = await browser.newPage()
const logs = []
page.on('console', (m) => logs.push(m.text()))
page.on('pageerror', (e) => logs.push('PAGEERROR: ' + e.message))

const PAGE = process.argv[2] || 'index.html'
await page.goto(`http://127.0.0.1:8731/${PAGE}`, { waitUntil: 'domcontentloaded' })
try { await page.waitForFunction('window.__done === true', { timeout: 120_000 }) }
catch { logs.push('TIMEOUT') }
await new Promise((r) => setTimeout(r, 2000))

console.log('=== page ===')
logs.filter((l) => l.startsWith('[T]')).forEach((l) => console.log(l))
console.log('\n=== ORT placement log ===')
logs.filter((l) => /Scan|placement|Placed|Provider|Memcpy/i.test(l))
    .forEach((l) => console.log(l.slice(0, 300)))

await browser.close()
server.close()
