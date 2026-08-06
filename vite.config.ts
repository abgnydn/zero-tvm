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

// A primed snapshot carries one of the three weight-manifest names: the two
// MLC ones (newer MLC repos, e.g. Qwen3.5, renamed ndarray-cache.json →
// tensor-cache.json) or a safetensors index for an MLX checkpoint (Qwen3.6).
const MANIFESTS = ['ndarray-cache.json', 'tensor-cache.json', 'model.safetensors.index.json']
const hasManifest = (dir: string): boolean => MANIFESTS.some((m) => existsSync(join(dir, m)))

function findMlcSnapshotDir(repoName: string = PHI3_REPO_NAME): string | null {
  // Repo-local: where scripts/download-weights.mjs puts the snapshot.
  const repoLocal = join(__dirname, '.weights-local', repoName)
  if (hasManifest(repoLocal)) return repoLocal

  // Preferred: a flat mirror dir populated via parallel curl (fastest to prime).
  const flatMirror = join(homedir(), 'mlc-weights', repoName)
  if (hasManifest(flatMirror)) return flatMirror

  // Fallback: HF hub snapshot dir (hf download populates this).
  // MLC repos live under mlc-ai/; MLX ones under whichever org published them,
  // so try both org spellings before giving up.
  const cacheRoot = ['mlc-ai', 'lmstudio-community']
    .map((org) => join(homedir(), '.cache', 'huggingface', 'hub', `models--${org}--${repoName}`, 'snapshots'))
    .find(existsSync) ?? ''
  if (!cacheRoot) return null
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
            // Per-model WebLLM URLs are /<repo-name>/resolve/main/<file>
            // (webllm-bench points AppConfig.model at the repo route, and
            // cleanModelUrl() leaves the resolve/main/ suffix alone) — strip
            // the HF-shape prefix again now that the repo segment is gone.
            urlPath = ('/' + seg[2]).replace(/^\/resolve\/[^/]+\//, '/')
          }
        }
        // WebLLM v0.2.80 renamed ndarray-cache.json → tensor-cache.json;
        // mirrors may carry either name (older repos ship ndarray-cache.json,
        // newer ones like Qwen3.5 ship tensor-cache.json). Contents/shape are
        // identical — serve whichever the snapshot actually has.
        if (snapshot) {
          if (urlPath === '/tensor-cache.json' && !existsSync(resolve(snapshot, './tensor-cache.json'))) {
            urlPath = '/ndarray-cache.json'
          } else if (urlPath === '/ndarray-cache.json' && !existsSync(resolve(snapshot, './ndarray-cache.json'))) {
            urlPath = '/tensor-cache.json'
          }
        }
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
        res.setHeader(
          'Content-Type',
          target.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        )
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        res.setHeader('Accept-Ranges', 'bytes')

        // RANGE support. The MLX loader reads BYTE RANGES, never whole shards —
        // a Qwen3.6 shard is 5.3 GB and a single ArrayBuffer caps near 4.29 GB —
        // and it refuses a 200 answer to a Range request rather than risk that
        // allocation. Without this the dev mirror is unusable for safetensors,
        // which is the whole point of the mirror.
        const range = /^bytes=(\d+)-(\d*)$/.exec(req.headers?.range ?? '')
        if (range) {
          const start = Number(range[1])
          const end = range[2] === '' ? st.size - 1 : Math.min(Number(range[2]), st.size - 1)
          if (start >= st.size || end < start) {
            res.statusCode = 416
            res.setHeader('Content-Range', `bytes */${st.size}`)
            res.end()
            return
          }
          res.statusCode = 206
          res.setHeader('Content-Range', `bytes ${start}-${end}/${st.size}`)
          res.setHeader('Content-Length', String(end - start + 1))
          createReadStream(target, { start, end }).pipe(res)
          return
        }

        res.setHeader('Content-Length', String(st.size))
        createReadStream(target).pipe(res)
        void next
      })
    },
  }
}

/**
 * Local GGUF mirror for the wllama baseline (src/wllama-bench/main.ts).
 *
 * Expects:   .weights-local/gguf/{phi3-q4,qwen3-4b-q4km,qwen35-4b-q4km}.gguf
 * Serves:    /local-gguf/<file>.gguf  →  .weights-local/gguf/<file>.gguf
 *
 * Separate from localWeightsPlugin because these are a DIFFERENT quantization
 * family (GGUF Q4_K_M) of the same base models, not the q4f16_1 tensors the
 * Zero-TVM/WebLLM same-bytes A/B shares. Keeping the routes distinct keeps
 * that distinction visible in the URL.
 *
 * Must answer HEAD with Content-Length (wllama's ModelManager sizes the
 * download that way) and must honour Range (llama.cpp/wllama may fetch
 * partially; a plain 200 for a range request would corrupt the load).
 */
const GGUF_DIR = join(__dirname, '.weights-local', 'gguf')

function localGgufPlugin() {
  return {
    name: 'local-gguf-weights',
    configureServer(server: any) {
      // eslint-disable-next-line no-console
      console.log(
        existsSync(GGUF_DIR)
          ? `[local-gguf] Serving /local-gguf/* from ${GGUF_DIR}`
          : `[local-gguf] No GGUF dir at ${GGUF_DIR} — /local-gguf/* will 404`,
      )
      // Cross-origin isolation for the wllama page only. Production already
      // sets these site-wide (public/_headers), but the dev server doesn't,
      // and without them SharedArrayBuffer is gone and wllama silently drops
      // to single-thread WASM. Scoped to this one document so the Zero-TVM
      // and WebLLM pages keep serving byte-identical headers to before —
      // their recorded numbers must stay comparable.
      server.middlewares.use((req: any, res: any, next: any) => {
        if ((req.url ?? '').startsWith('/wllama-bench')) {
          res.setHeader('Cross-Origin-Opener-Policy', 'same-origin')
          res.setHeader('Cross-Origin-Embedder-Policy', 'credentialless')
        }
        next()
      })
      server.middlewares.use('/local-gguf', (req: any, res: any, next: any) => {
        const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0])
        const target = resolve(GGUF_DIR, '.' + urlPath)
        if (!target.startsWith(GGUF_DIR)) {
          res.statusCode = 403
          res.end('Forbidden')
          return
        }
        if (!existsSync(target) || !statSync(target).isFile()) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const size = statSync(target).size
        res.setHeader('Content-Type', 'application/octet-stream')
        res.setHeader('Accept-Ranges', 'bytes')
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        if (req.method === 'HEAD') {
          res.setHeader('Content-Length', String(size))
          res.end()
          return
        }
        const m = /^bytes=(\d*)-(\d*)$/.exec(String(req.headers.range ?? '').trim())
        if (m) {
          // `bytes=-N` is the suffix form (last N bytes); `bytes=N-` runs to EOF.
          const start = m[1] === '' ? Math.max(0, size - Number(m[2])) : Number(m[1])
          const end = m[1] === '' ? size - 1 : m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1)
          if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
            res.statusCode = 416
            res.setHeader('Content-Range', `bytes */${size}`)
            res.end()
            return
          }
          res.statusCode = 206
          res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
          res.setHeader('Content-Length', String(end - start + 1))
          createReadStream(target, { start, end }).pipe(res)
          return
        }
        res.setHeader('Content-Length', String(size))
        createReadStream(target).pipe(res)
        void next
      })
    },
  }
}

/**
 * Local ONNX mirror for the transformers.js baseline (src/tjs-bench/main.ts).
 *
 * Expects:   .weights-local/onnx/<hf-repo-name>/{config.json,tokenizer.json,onnx/*}
 *            laid out exactly as the HuggingFace repo is, e.g.
 *            .weights-local/onnx/Qwen3.5-4B-ONNX/onnx/decoder_model_merged_q4f16.onnx
 * Serves:    /local-onnx/<repo-name>/<path>  →  .weights-local/onnx/<repo-name>/<path>
 *
 * The page sets `env.localModelPath = '/local-onnx/'` + `allowLocalModels`,
 * which makes transformers.js build exactly this URL shape (hub.js pathJoin of
 * localModelPath + repo id + filename) with no HF-hub `resolve/main/` segment
 * — so unlike localWeightsPlugin this needs no prefix rewriting.
 *
 * A THIRD quantization family: ONNX q4f16 (MatMulNBits int4 + fp16 activations),
 * which is neither the MLC q4f16_1 of the Zero-TVM/WebLLM same-bytes A/B nor
 * the GGUF Q4_K_M of the wllama baseline. Distinct route, distinct directory,
 * so the distinction stays visible in the URL — same reasoning as /local-gguf.
 *
 * Range support matters more here than for GGUF: ORT's external-data loader
 * fetches the multi-gigabyte `.onnx_data*` blobs, and a plain 200 answering a
 * range request would silently corrupt the weights rather than error.
 */
const ONNX_DIR = join(__dirname, '.weights-local', 'onnx')

function localOnnxPlugin() {
  return {
    name: 'local-onnx-weights',
    configureServer(server: any) {
      // eslint-disable-next-line no-console
      console.log(
        existsSync(ONNX_DIR)
          ? `[local-onnx] Serving /local-onnx/* from ${ONNX_DIR}`
          : `[local-onnx] No ONNX dir at ${ONNX_DIR} — /local-onnx/* will 404`,
      )
      server.middlewares.use('/local-onnx', (req: any, res: any, next: any) => {
        const urlPath = decodeURIComponent((req.url ?? '/').split('?')[0])
        const target = resolve(ONNX_DIR, '.' + urlPath)
        if (!target.startsWith(ONNX_DIR)) {
          res.statusCode = 403
          res.end('Forbidden')
          return
        }
        if (!existsSync(target) || !statSync(target).isFile()) {
          res.statusCode = 404
          res.end('Not found')
          return
        }
        const size = statSync(target).size
        res.setHeader(
          'Content-Type',
          target.endsWith('.json') ? 'application/json' : 'application/octet-stream',
        )
        res.setHeader('Accept-Ranges', 'bytes')
        res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
        if (req.method === 'HEAD') {
          res.setHeader('Content-Length', String(size))
          res.end()
          return
        }
        const m = /^bytes=(\d*)-(\d*)$/.exec(String(req.headers.range ?? '').trim())
        if (m) {
          const start = m[1] === '' ? Math.max(0, size - Number(m[2])) : Number(m[1])
          const end = m[1] === '' ? size - 1 : m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1)
          if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
            res.statusCode = 416
            res.setHeader('Content-Range', `bytes */${size}`)
            res.end()
            return
          }
          res.statusCode = 206
          res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`)
          res.setHeader('Content-Length', String(end - start + 1))
          createReadStream(target, { start, end }).pipe(res)
          return
        }
        res.setHeader('Content-Length', String(size))
        createReadStream(target).pipe(res)
        void next
      })
    },
  }
}

export default defineConfig({
  plugins: [localWeightsPlugin(), localGgufPlugin(), localOnnxPlugin()],
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
      // 10-char content hashes (default 8). Changed once on 2026-08-06 to
      // rotate EVERY asset URL past a poisoned edge-cache entry: the zone had
      // cached an HTML fallback for a chunk URL under /assets/*'s immutable
      // header (alias flipped mid-deploy), and nothing on this machine holds
      // zone-purge permission. Content addressing is unchanged.
      output: {
        chunkFileNames: 'assets/[name]-[hash:10].js',
        entryFileNames: 'assets/[name]-[hash:10].js',
        assetFileNames: 'assets/[name]-[hash:10][extname]',
      },
      input: {
        index:         resolve(__dirname, 'index.html'),
        'zero-tvm':    resolve(__dirname, 'zero-tvm.html'),
        'compiler-chat': resolve(__dirname, 'compiler-chat.html'),
        demo:          resolve(__dirname, 'demo.html'),
        validate:      resolve(__dirname, 'validate.html'),
        'webllm-bench':resolve(__dirname, 'webllm-bench.html'),
        // wllama-bench and tjs-bench are DEV-ONLY benchmark harnesses. They are
        // deliberately not built or deployed: they pull in llama.cpp's 7.7 MB
        // wasm and ONNX Runtime's >10 MB wasm respectively, which bloat the
        // deploy and exceed HuggingFace's 10 MiB non-LFS file limit. bench/run.mjs
        // drives them through the vite DEV server, which serves any root HTML
        // regardless of build inputs, so benchmarking is unaffected.
        architecture:  resolve(__dirname, 'architecture.html'),
        docs:          resolve(__dirname, 'docs.html'),
        dump:          resolve(__dirname, 'dump.html'),
        shaders:       resolve(__dirname, 'shaders.html'),
      },
    },
    chunkSizeWarningLimit: 8000,
  },
})
