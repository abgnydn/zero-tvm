import { defineConfig } from 'vite'
import { resolve } from 'path'

// Multi-page build — the repo is a narrative arc across several HTML entry points.
// See README.md for what each page means.
export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        index:          resolve(__dirname, 'index.html'),
        'zero-tvm':     resolve(__dirname, 'zero-tvm.html'),
        'compiler-chat':resolve(__dirname, 'compiler-chat.html'),
        demo:           resolve(__dirname, 'demo.html'),
        dump:           resolve(__dirname, 'dump.html'),
        shaders:        resolve(__dirname, 'shaders.html'),
        'test-shaders': resolve(__dirname, 'test-shaders.html'),
        'test-chain':   resolve(__dirname, 'test-chain.html'),
        'standalone-test': resolve(__dirname, 'standalone-test.html'),
        validate:       resolve(__dirname, 'validate.html'),
      },
    },
    chunkSizeWarningLimit: 8000,
  },
})
