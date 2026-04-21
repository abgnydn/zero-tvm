import { defineConfig } from 'vitest/config'

// Vitest is used here exclusively as a runner for Puppeteer-driven e2e tests
// (see tests/e2e). There are no unit tests in this repo — the model is too
// stateful to test in isolation, and the per-shader correctness harness lives
// in test-shaders.html. The e2e tests boot a real Vite dev server, launch real
// Chrome with WebGPU, and exercise the full chat + validate pages.
export default defineConfig({
  test: {
    include: ['tests/e2e/**/*.test.ts'],
    // First run downloads ~2GB of Phi-3 weights from HuggingFace into the
    // test browser profile. Subsequent runs use the cached profile and finish
    // in well under a minute.
    testTimeout: 6 * 60 * 1000,
    hookTimeout: 6 * 60 * 1000,
    // The tests share a single browser + dev server, so they cannot run in
    // parallel against each other.
    fileParallelism: false,
    sequence: { concurrent: false },
    // Print console.log from tests so cold-start timings are visible in the
    // terminal output, not just in failed-assertion messages.
    silent: false,
  },
})
