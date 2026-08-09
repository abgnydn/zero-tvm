import { defineConfig } from 'vite'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * LIBRARY BUILD — `npm run build:lib`, the package `exports` entry.
 *
 * Separate from vite.config.ts (the multi-page SITE build) on purpose: that
 * config's HTML inputs, dev-server middlewares and dist/ output are exactly
 * what a package consumer must not get, and `npm run build` must keep working
 * untouched.
 *
 * Vite is what builds this rather than tsc, because the engine imports its
 * WGSL through `?raw` — plain tsc would emit those specifiers verbatim and
 * only a Vite-flavoured consumer could resolve them.
 *
 * Code splitting MUST stay on. src/lib/index.ts loads the engine through a
 * dynamic import so that importing the package never touches GPUBufferUsage
 * (see its header); rollup's inlineDynamicImports would hoist that module
 * into the entry chunk and evaluate it eagerly, which reintroduces the exact
 * throw-on-import the lazy import exists to prevent. `es` format with a
 * single entry keeps the chunks separate.
 */
export default defineConfig({
  // public/ is the SITE's static root (favicon, _headers, sitemap, chat-ui.css).
  // Left on, vite copies all of it beside the bundle and the package ships a
  // 404.html and a Cloudflare _headers file.
  publicDir: false,
  build: {
    outDir: 'dist-lib',
    emptyOutDir: true,
    lib: {
      entry: resolve(__dirname, 'src/lib/index.ts'),
      formats: ['es'],
      fileName: () => 'index.js',
    },
    // The engine bundle is one big chunk by nature (55 WGSL kernels inlined).
    chunkSizeWarningLimit: 8000,
  },
})
