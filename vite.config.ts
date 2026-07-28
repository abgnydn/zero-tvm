import { defineConfig } from 'vite'
import { existsSync, readdirSync, statSync, createReadStream } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Local MLC-weights mirror for e2e testing without re-downloading 2 GB.
 *
 * Expects:   node scripts/download-weights.mjs [--model qwen3]
 *            (or huggingface-cli download mlc-ai/<repo>)
 * Serves:    /local-weights/<repo-name>/*  →  .weights-local/<repo-name>/
 *            (or ~/mlc-weights/<repo-name>, or the HF hub snapshot dir)
 *
 * The bare legacy form /local-weights/* (no repo segment) still serves the
 * Phi-3 snapshot — WebLLM's compiler-chat / webllm-bench pages depend on that
 * exact URL shape, and Phi-3 mirror behavior must stay byte-compatible.
 *
 * Weight loader tries /local-weights/<repo-name>/<file> first (tier 0) when
 * import.meta.env.DEV. Falls through to OPFS / browser cache / HuggingFace
 * CDN if the local mirror isn't primed.
 */
const PHI3_REPO_NAME = 'Phi-3-mini-4k-instruct-q4f16_1-MLC'

function findMlcSnapshotDir(repoName: string = PHI3_REPO_NAME): string | null {
  // Repo-local: where scripts/download-weights.mjs puts the snapshot.
  const repoLocal = join(__dirname, '.weights-local', repoName)
  if (existsSync(join(repoLocal, 'ndarray-cache.json'))) return repoLocal

  // Preferred: a flat mirror dir populated via parallel curl (fastest to prime).
  const flatMirror = join(homedir(), 'mlc-weights', repoName)
  if (existsSync(join(flatMirror, 'ndarray-cache.json'))) return flatMirror

  // Fallback: HF hub snapshot dir (hf download populates this).
  const cacheRoot = join(
    homedir(),
    '.cache',
    'huggingface',
    'hub',
    `models--mlc-ai--${repoName}`,
    'snapshots',
  )
  if (!existsSync(cacheRoot)) return null
  const entries = readdirSync(cacheRoot)
    .map((name) => join(cacheRoot, name))
    .filter((p) => statSync(p).isDirectory())
  // Pick the newest snapshot so an updated download wins.
  entries.sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs)
  return entries[0] ?? null
}

function localWeightsPlugin() {
  return {
    name: 'local-mlc-weights',
    configureServer(server: any) {
      const defaultSnapshot = findMlcSnapshotDir()
      if (defaultSnapshot) {
        // eslint-disable-next-line no-console
        console.log(`[local-weights] Serving /local-weights/* from ${defaultSnapshot}`)
      } else {
        // eslint-disable-next-line no-console
        console.log('[local-weights] No Phi-3 snapshot found — bare /local-weights/* will 404')
      }
      // Per-model snapshot lookup, memoized per repo-name segment.
      const snapshotCache = new Map<string, string | null>()
      const snapshotFor = (repoName: string): string | null => {
        if (!snapshotCache.has(repoName)) {
          const dir = findMlcSnapshotDir(repoName)
          if (dir) {
            // eslint-disable-next-line no-console
            console.log(`[local-weights] Serving /local-weights/${repoName}/* from ${dir}`)
          }
          snapshotCache.set(repoName, dir)
        }
        return snapshotCache.get(repoName) ?? null
      }
      server.middlewares.use('/local-weights', (req: any, res: any, next: any) => {
        // Strip query string, resolve safely inside snapshot dir.
        let urlPath = decodeURIComponent((req.url ?? '/').split('?')[0])
        // WebLLM's cleanModelUrl() rewrites any bare model URL by appending
        // `resolve/main/` to it (HF-hub convention). Strip that prefix so the
        // same mirror serves both Zero-TVM (bare) and WebLLM (HF-shaped) URLs.
        urlPath = urlPath.replace(/^\/resolve\/[^/]+\//, '/')
        // Model routing: /<repo-name>/<file> serves <file> from that repo's
        // mirror dir. Unknown first segments fall through to the legacy
        // single-model (Phi-3) behavior, byte-compatible with before.
        let snapshot = defaultSnapshot
        const seg = /^\/([^/]+)\/(.+)$/.exec(urlPath)
        if (seg) {
          const modelSnap = snapshotFor(seg[1])
          if (modelSnap) {
            snapshot = modelSnap
            urlPath = '/' + seg[2]
          }
        }
        // WebLLM v0.2.80 renamed ndarray-cache.json → tensor-cache.json; our
        // mirrors still have the old name. Contents/shape are identical.
        if (urlPath === '/tensor-cache.json') urlPath = '/ndarray-cache.json'
        if (!snapshot) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const target = resolve(snapshot, '.' + urlPath)
        if (!target.startsWith(snapshot)) {
          res.statusCode = 403
          res.end('Forbidden')
          return
        }
        if (!existsSync(target) || !statSync(target).isFile()) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const st = statSync(target)
        res.setHeader('Content-Length', String(st.size))
        res.setHeader(
          'Content-Type',
          target.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        )
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        createReadStream(target).pipe(res)
        void next
      })
    },
  }
}

export default defineConfig({
  plugins: [localWeightsPlugin()],
  server: {
    fs: {
      // Allow serving files from the HF cache / flat mirror outside the repo root.
      allow: [
        resolve(__dirname),
        join(homedir(), '.cache', 'huggingface'),
        join(homedir(), 'mlc-weights'),
      ],
    },
  },
  build: {
    // Deploy all user-facing entries, including the shader-inspection pages
    // (dump, shaders) which README documents as part of the build. Dev-only
    // pages (test-shaders, test-chain) are intentionally excluded — they're
    // in-repo for debugging and not linked from the site.
    rollupOptions: {
      input: {
        index:         resolve(__dirname, 'index.html'),
        'zero-tvm':    resolve(__dirname, 'zero-tvm.html'),
        'compiler-chat': resolve(__dirname, 'compiler-chat.html'),
        demo:          resolve(__dirname, 'demo.html'),
        validate:      resolve(__dirname, 'validate.html'),
        'webllm-bench':resolve(__dirname, 'webllm-bench.html'),
        architecture:  resolve(__dirname, 'architecture.html'),
        docs:          resolve(__dirname, 'docs.html'),
        dump:          resolve(__dirname, 'dump.html'),
        shaders:       resolve(__dirname, 'shaders.html'),
      },
    },
    chunkSizeWarningLimit: 8000,
  },
})
