---
title: Zero-TVM
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
short_description: Phi-3-mini in 10 hand-written WGSL kernels (no TVM)
tags:
  - llm
  - phi-3
  - phi3
  - microsoft-phi
  - webgpu
  - wgsl
  - kernel-fusion
  - in-browser
  - mlc
  - webllm
  - inference
models:
  - microsoft/Phi-3-mini-4k-instruct
  - mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC
---

# Zero-TVM

**Phi-3-mini running in a browser on hand-written WGSL shaders. No TVM. No WebLLM runtime. No compiler.**

The standard way to run a modern LLM in a browser is [WebLLM / MLC-LLM](https://webllm.mlc.ai/), which ships an Apache-TVM compiler pipeline that emits **85 autotuned WGSL kernels**. This Space replaces that entire stack with **10 kernel roles (27 WGSL implementations, counting subgroup / tiled / int8 variants) and about 2,000 lines of TypeScript**, using the same Phi-3-mini-q4f16_1 weights.

The whole forward pass — 32 transformer layers, paged KV cache, int4-dequant matmul, RoPE, fused FFN, RMSNorm, paged attention, argmax sampling — is readable end-to-end in a single sitting. That is the point.

## Headline result

Measured on an Apple M2 Max (Chrome, WebLLM v0.2.80, identical Phi-3-mini-q4f16_1 weights, same-run head-to-head):

| | WebLLM (TVM) | Zero-TVM |
|---|---|---|
| Unique WGSL kernels | **85** | **10 roles / 27 files** |
| Total WGSL lines | **12,962** (generated) | **3,078** (hand-written) |
| Dispatches per decode step | **342** | **228** |
| Decode throughput | baseline | **28–31% faster** (same weights, same run) |
| JS bundle (excl. weights) | **5.9 MB** / 2.1 MB gz | **157 kB** / 33 kB gz |

**The hand-written kernels don't just keep up with the autotuning compiler — on this hardware they beat it.** For decoder-only LLMs of this shape, most of the compiler's complexity budget isn't buying much — the expensive parts are matmul, attention, and int4 dequant. Everything else is plumbing. ~10 kernels of plumbing, instead of 85. Exact medians, full methodology, and the optimization experiments that were measured and *dropped* live in [BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md). (An earlier measurement on an M2 Pro with an older, buggier engine read 22% *behind* — that history is preserved there too, not retconned.)

## Run it

After this Space loads, open **[zero-tvm.html](./zero-tvm.html)** to launch the chat UI.

On first load you'll see a "Download & Start" gate — clicking it streams ~1.8 GB of Phi-3-mini-q4f16_1 weights from the Hugging Face mirror at [`mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC) into OPFS (Origin Private File System).

Subsequent loads are instant. **Both `zero-tvm.html` and `compiler-chat.html` share the same cached weights** via a Service Worker that intercepts the Phi-3 mirror URLs — visiting either page after the first download is gate-free.

**Requirements**: Chrome / Edge with WebGPU enabled and the `shader-f16` feature available (default on macOS Apple-Silicon, enabled on most modern Windows / Linux GPUs).

## Pages in this Space

- [`index.html`](./index.html) — landing page, shader catalog, compare table
- [`zero-tvm.html`](./zero-tvm.html) — **the chat demo** (start here)
- [`webllm-bench.html`](./webllm-bench.html) — head-to-head harness vs WebLLM on identical weights
- [`compiler-chat.html`](./compiler-chat.html) — same Phi-3-mini weights, run via WebLLM (for direct comparison; reuses the shared cache, no extra download)
- [`architecture.html`](./architecture.html) — kernel architecture explainer
- [`shaders.html`](./shaders.html) — browseable WGSL source
- [`demo.html`](./demo.html) — dispatch visualization
- [`validate.html`](./validate.html) — multi-prompt smoke test
- [`docs.html`](./docs.html) — annotated reference

## URL flags

`zero-tvm.html` accepts query flags for A/B-ing shader variants without rebuilding:

- `?sg=0` — disable all subgroup shaders (argmax / attention / QKV matmul)
- `?sgqkv=0` / `?sgattn=0` / `?sgargmax=0` — disable one at a time
- `?qkvtile=1` / `?qkvtile2=1` — opt into tiled QKV variants
- `?ffnsg=1` — opt into the tiled-subgroup fused FFN
- `?kv8=1` — opt into the int8 KV cache path

## How it relates to a published paper

Zero-TVM is the LLM-decoding application of a broader thesis: **per-dispatch overhead is the dominant tax on browser GPU compute, and most workloads collapse from "needs a server" to "runs in a browser tab" once you fuse the loop.** Two preprints document the underlying technique:

- [Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders](https://doi.org/10.5281/zenodo.19342888) (the foundational result)
- [Single-Kernel Fusion for Autoregressive Transformer Decoding via WebGPU Compute Shaders](https://doi.org/10.5281/zenodo.19344277) (the LLM application — the work this Space embodies)

## Source / canonical site

- **GitHub**: [github.com/abgnydn/zero-tvm](https://github.com/abgnydn/zero-tvm)
- **Canonical site**: [zerotvm.com](https://zerotvm.com)
- **Research umbrella**: [kernelfusion.dev](https://kernelfusion.dev)
- **Companion benchmarks**: [gpubench.dev](https://gpubench.dev)

## License

MIT. Reference comparison code (WebLLM compiler-chat path) belongs to its respective authors and is governed by their licenses.

## Author

Ahmet Barış Günaydın · [barisgunaydin.com](https://barisgunaydin.com) · senior full-stack consultant by day, independent researcher by night.
