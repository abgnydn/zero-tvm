import { defineConfig } from 'vitest/config'

// Vitest runs two very different suites:
//   tests/tokenizer — offline unit tests for the hand-rolled BPE tokenizer,
//                     pinned against HuggingFace-generated fixtures
//                     (`npm run test:unit`).
//   tests/e2e       — Puppeteer-driven e2e tests that boot a real Vite dev
//                     server + Chrome with WebGPU and exercise the full
//                     chat + validate pages (`npm run test:e2e`).
// Per-kernel GPU correctness lives outside vitest in tests/kernels
// (`npm run test:kernels`).
export default defineConfig({
  test: {
    include: ['tests/tokenizer/**/*.test.ts', 'tests/e2e/**/*.test.ts'],
    // First e2e run downloads ~2GB of Phi-3 weights from HuggingFace into the
    // test browser profile. Subsequent runs use the cached profile and finish
    // in well under a minute. (The tokenizer unit tests finish in seconds.)
    testTimeout: 6 * 60 * 1000,
    hookTimeout: 6 * 60 * 1000,
    // The e2e tests share a single browser + dev server, so they cannot run
    // in parallel against each other.
    fileParallelism: false,
    sequence: { concurrent: false },
    // Print console.log from tests so cold-start timings are visible in the
    // terminal output, not just in failed-assertion messages.
    silent: false,
  },
})
