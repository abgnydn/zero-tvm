# Headless kernel-correctness tests

Runs the engine's **real** WGSL kernels (from `src/compiler/shaders/`) on a
WebGPU adapter and checks each against a plain-JS reference. This is the
automated net the project previously lacked — before this, per-kernel
correctness was only checkable by hand via `test-shaders.html`.

```bash
npm run test:kernels
```

Covers all 10 kernel roles: `int4_matmul`, `rms_norm`, `argmax`, `embedding`,
`rope`, `add_norm`, `kv_append`, `fused_ffn`, paged `attention` (shuffled
page table vs a CPU softmax-attention reference), and `qkv_fused` (QKV matmul
+ RoPE + paged KV append vs a CPU reference). Each reference is an
independent JS reimplementation (not derived from the WGSL).

The shipped subgroup variants are covered too: `int4_matmul_sg`, `argmax_sg`,
`attention_sg`, `qkv_fused_sg`, and `fused_ffn_tiled_sg`. These require the
WebGPU `subgroups` feature and (for all but `argmax_sg`) subgroup size >= 32
— the same gate chat.ts applies. The suite probes the adapter's actual
subgroup size on-device; when the requirement isn't met (lavapipe in CI may
lack it) each gated test prints a loud `SKIP <name> <reason>` line rather
than passing silently. On a real GPU (e.g. Apple Metal, subgroup size 32)
everything runs.

The experiment variants (measured 2026-07-25 on M2 Max — see BENCH.md
"Measured 2026-07-25 (M2 Max)") are correctness-tested here too: the four
`_vec4` int4 matmuls and `qkv_fused_sg_vec4` (vs the same CPU references —
now the default path, +7.1% measured), `attention_splitk[_sg]` +
`attention_combine` (N=2 and N=4, shuffled page table, page count not
divisible by N, empty partitions, kv_len < N — still opt-in, ~+3% at short
context), and the prologue-fusion pair (`fused_ffn[_tiled_sg]_prologue` vs a
sequential add_norm-then-FFN reference, plus `add3_norm` — measured −13.7%
on M2 Max, kept for A/B on other GPUs). Passing here says nothing about
throughput — that lives in BENCH.md.

Every shader is compiled through the same prelude path the app uses
(`src/compiler/shader-prelude.ts` injects the Phi-3 shape constants), and the
int4_matmul family comes from the generator
(`src/compiler/shaders/int4_matmul.gen.ts`) — the tests exercise the exact
WGSL the engine ships.

A final `compile_all` gate creates a shader module **and** a compute pipeline
for every `.wgsl` file plus every generator variant, failing loudly on any
compilation or validation error. This protects the shaders the correctness
tests above don't execute (int8-KV path, the tiled/regressed experiments, the
tiled matmul variants).

Add a kernel by writing one `testX(device)` function (or a `makeXTest`
factory when a scalar/`_sg` pair shares its reference) in `run.mjs` and
listing it in `TESTS`.

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

This suite runs in CI as the `kernels` job in `.github/workflows/ci.yml`
(lavapipe via `mesa-vulkan-drivers`, `PUPPETEER_SKIP_DOWNLOAD=1` on `npm ci`
since no browser is needed). Subgroup-gated tests print `SKIP` lines there
when lavapipe can't satisfy the feature/lane-width requirement; the full
matrix runs on a real GPU (`npm run test:kernels` on a dev machine).
