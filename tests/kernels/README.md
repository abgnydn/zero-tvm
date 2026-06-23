# Headless kernel-correctness tests

Runs the engine's **real** WGSL kernels (from `src/compiler/shaders/`) on a
WebGPU adapter and checks each against a plain-JS reference. This is the
automated net the project previously lacked — before this, per-kernel
correctness was only checkable by hand via `test-shaders.html`.

```bash
npm run test:kernels
```

Covers 8 of the 10 kernel roles: `int4_matmul`, `rms_norm`, `argmax`,
`embedding`, `rope`, `add_norm`, `kv_append`, and `fused_ffn`. Each reference
is an independent JS reimplementation (not derived from the WGSL), and
`fused_ffn`'s reference mirrors the kernel's f16 accumulation structure so the
tolerance can stay tight.

Still uncovered: `qkv_fused` and paged `attention` — they carry enough KV-cache
/ softmax state that they're better exercised end-to-end (`npm test`) than in
isolation. Add a kernel by writing one `testX(device)` function in `run.mjs`
and listing it in `TESTS`.

## WebGPU without a GPU (CI / this sandbox)

Chrome's GPU process **blocklists WebGPU on software rasterizers**, so headless
Chrome never exposes `navigator.gpu` on a machine without a real GPU. We use
the Dawn-native binding (`@kmamal/gpu`) instead, which has no such blocklist and
talks to whatever Vulkan adapter is present — including Mesa's **lavapipe**, a
CPU software device that supports `shader-f16`.

On a dev machine with a real GPU, the same suite uses the real adapter and
needs no setup. On a Linux CI runner:

```bash
apt-get install -y mesa-vulkan-drivers
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
npm run test:kernels
```

lavapipe is a software rasterizer, so this suite asserts **correctness only** —
never throughput. Performance numbers belong in `BENCH.md`, measured on real
hardware.

## GitHub Actions

Adding this to CI needs a workflow edit (separate `workflow`-scoped commit):

```yaml
  kernels:
    name: kernel correctness (lavapipe)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22, cache: npm }
      - run: sudo apt-get update && sudo apt-get install -y mesa-vulkan-drivers
      - run: npm ci
      - run: npm run test:kernels
        env:
          VK_ICD_FILENAMES: /usr/share/vulkan/icd.d/lvp_icd.json
```
