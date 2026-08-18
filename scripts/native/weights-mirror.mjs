/**
 * Serve .weights-local/ over HTTP so the native host can read checkpoints it
 * already has instead of re-downloading them.
 *
 * The browser gets this free: vite mirrors the snapshot dir at
 * /local-weights/<repo>/ and the loader tries it first. But that tier is a
 * RELATIVE url behind `import.meta.env.DEV`, and the native host runs the
 * production library build — so it could never use it. The result was 15 GB of
 * Qwen3.8 on disk and a 40-minute download from HuggingFace to get the same
 * bytes onto the same machine.
 *
 * Range support is not optional here: weight-loader-mlx fetches BYTE RANGES out
 * of multi-gigabyte safetensors shards rather than whole files, so a server
 * that ignores Range would hand back 5 GB per request and defeat the point.
 */
import { createServer } from 'node:http'
import { createReadStream, existsSync, statSync, readdirSync } from 'node:fs'
import { join, normalize } from 'node:path'

/** Repo dirs that look primed — one of the three manifest names, same rule as
 *  vite.config.ts uses. A half-downloaded dir is not a mirror. */
const MANIFESTS = ['ndarray-cache.json', 'tensor-cache.json', 'model.safetensors.index.json']
const primed = (dir) => MANIFESTS.some((m) => existsSync(join(dir, m)))

export function startWeightsMirror(weightsRoot) {
  if (!existsSync(weightsRoot)) return null
  const repos = readdirSync(weightsRoot, { withFileTypes: true })
    .filter((d) => d.isDirectory() && primed(join(weightsRoot, d.name)))
    .map((d) => d.name)
  if (repos.length === 0) return null

  const server = createServer((req, res) => {
    // /local-weights/<repo>/<file...>
    const url = new URL(req.url, 'http://x')
    const rel = decodeURIComponent(url.pathname).replace(/^\/local-weights\//, '')
    // Refuse traversal outright rather than normalising and hoping.
    const safe = normalize(rel)
    if (!rel || safe.startsWith('..') || safe.includes('../')) { res.writeHead(400).end(); return }
    const file = join(weightsRoot, safe)
    if (!file.startsWith(weightsRoot) || !existsSync(file) || !statSync(file).isFile()) {
      res.writeHead(404).end(); return
    }

    const size = statSync(file).size
    const range = req.headers.range
    if (range) {
      const m = /^bytes=(\d+)-(\d*)$/.exec(range)
      if (!m) { res.writeHead(416, { 'content-range': `bytes */${size}` }).end(); return }
      const start = Number(m[1])
      const end = m[2] ? Math.min(Number(m[2]), size - 1) : size - 1
      if (start > end || start >= size) { res.writeHead(416, { 'content-range': `bytes */${size}` }).end(); return }
      res.writeHead(206, {
        'content-range': `bytes ${start}-${end}/${size}`,
        'content-length': end - start + 1,
        'accept-ranges': 'bytes',
      })
      if (req.method === 'HEAD') { res.end(); return }
      createReadStream(file, { start, end }).pipe(res)
      return
    }
    res.writeHead(200, { 'content-length': size, 'accept-ranges': 'bytes' })
    if (req.method === 'HEAD') { res.end(); return }
    createReadStream(file).pipe(res)
  })

  return new Promise((resolve) => {
    // Port 0: never collide with the station, the engine, or a dev server.
    server.listen(0, '127.0.0.1', () => {
      const base = `http://127.0.0.1:${server.address().port}/local-weights`
      globalThis.__ZT_WEIGHTS_MIRROR__ = base
      resolve({ base, repos, close: () => server.close() })
    })
  })
}
