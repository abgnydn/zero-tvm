---
title: Zero-TVM
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
thumbnail: https://huggingface.co/spaces/abgunaydin/zero-tvm/resolve/main/og.png
short_description: Phi-3 to a 35B sparse MoE, in the browser on hand-written WGSL
tags:
  - webgpu
  - llm
  - inference
  - on-device
  - wgsl
models:
  - Qwen/Qwen3.5-4B
  - Qwen/Qwen3-4B
  - microsoft/Phi-3-mini-4k-instruct
  - mlc-ai/Qwen3.5-4B-q4f16_1-MLC
  - mlc-ai/Qwen3-4B-q4f16_1-MLC
  - mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC
---

# Zero-TVM

Models from a 3.8B dense to a 35B sparse MoE, running in the browser on hand-written WGSL. No TVM, no ONNX, no WASM runtime.

The standard way to run a modern LLM in a browser is [WebLLM / MLC-LLM](https://webllm.mlc.ai/), which ships an Apache-TVM compiler pipeline that emits **85 autotuned WGSL kernels**. This Space replaces that entire stack with **10 kernel roles** of hand-written WGSL — and runs models WebLLM ships (same weights, measured faster) as well as one it does not: Qwen3.6-35B-A3B, a 256-expert sparse MoE.

The whole forward pass — 32 transformer layers, paged KV cache, int4-dequant matmul, RoPE, fused FFN, RMSNorm, paged attention, argmax sampling — is readable end-to-end in a single sitting. That is the point.

## Headline result

Measured on an Apple M2 Max (Chrome 150, WebLLM v0.2.80, identical Phi-3-mini-q4f16_1 weights, same-session head-to-head, 2026-07-30). Two numbers, always both: **total** is wall-clock throughput with prefill included (both engines doing identical work), **decode** excludes prefill.

| | WebLLM (TVM) | Zero-TVM |
|---|---|---|
| WGSL kernels (decode path) | **~11 TVM-generated** (85 captured across a session) | **10 hand-written roles** |
| Dispatches per decode step | **342** | **260** default (228 with `?splitk=0`) |
| Total throughput (prefill + decode) | **59.95 tok/s** | **69.55 tok/s** — **+16.0%** |
| Decode only | **63.23 tok/s** (self-reported) | **83.10 tok/s** — **+31.4%** |
| Time to first token (~35-tok prompt) | **~150 ms** (implied) | **291 ms** — *WebLLM wins this one* |
| JS bundle (excl. weights) | **~6.0 MB** / ~2.2 MB gz | **~460 kB** / ~126 kB gz |

**The hand-written kernels don't just keep up with the autotuning compiler — on this hardware they beat it on throughput.** For decoder-only LLMs of this shape, most of the compiler's complexity budget isn't buying much — the expensive parts are matmul, attention, and int4 dequant. Everything else is plumbing, and ten hand-written roles cover it.

**The gap grows with how recent the architecture is** — +16% / +31% (total / decode) on 2024's Phi-3, +32% / +58% on 2025's Qwen3-4B, +100% / +114% on 2026's Qwen3.5-4B. Consistent with compiler stacks having had less time to tune newer architectures; an observation across three points on one machine, not a proven law.

**Honest negative, published not buried:** WebLLM reaches the first token faster than we do on short prompts (~150 ms implied vs our 291 ms on Phi-3; 453 ms on Qwen3-4B). We win sustained throughput and lose the first-token sprint on short inputs — our chunked prefill is strong on long ones (202 tok/s at 816 tokens) and prefix reuse makes follow-up turns free.

Exact medians, full methodology, and the optimization experiments that were measured and *dropped* live in [BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md). (An earlier measurement on an M2 Pro with an older, buggier engine read 22% *behind* — that history is preserved there too, not retconned. So are two 2026-07-29 Qwen pairs that a bench-harness defect made non-comparable; they are marked withdrawn with the reason, not deleted.)

## Run it

The Space opens on a character-select entrance: pick a model and it loads and chats in place. [`zero-tvm.html`](./zero-tvm.html) is the same chat as a direct link, if you want to deep-link a specific model.

Picking a character starts the download immediately — the sheet shows the weight size before you click, and there is no second confirmation. It streams ~2 GB of Phi-3-mini-q4f16_1 weights from the Hugging Face mirror at [`mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC) into OPFS (Origin Private File System).

Subsequent loads are instant: a Service Worker intercepts the mirror URLs, so every page sharing those weights is gate-free after the first download.

**Requirements**: Chrome / Edge with WebGPU enabled and the `shader-f16` feature available (default on macOS Apple-Silicon, enabled on most modern Windows / Linux GPUs).

## Models

All three pairs re-measured 2026-07-30 under the corrected protocol — same session, identical local weight bytes, both engines paying a full prefill on every run, total and decode reported side by side.

- **Phi-3-mini-4k-instruct (3.8B, q4f16_1)** — the default; every headline number above is Phi-3, no URL flag needed. Zero-TVM **69.55 tok/s total** (TTFT 291 ms, decode 83.10) vs WebLLM 0.2.80's **59.95 tok/s total** (self-reported decode 63.23) — **+16.0% total, +31.4% decode**.
- **Qwen3-4B (q4f16_1)** — append `?model=qwen3` to `zero-tvm.html` or `validate.html`; weights stream from [`mlc-ai/Qwen3-4B-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Qwen3-4B-q4f16_1-MLC) (~2.3 GB). A port on the spec-parameterized engine: GQA 32/8, per-head QK-norm, byte-level BPE, tied lm_head, with the tuning round's fused post-matmul chain (`qk_norm_rope_append`, 8 dispatches/layer) and K%512 wide loads. Zero-TVM **59.85 tok/s total** (TTFT 453 ms, decode 75.49) vs WebLLM 0.2.84's prebuilt Qwen3-4B at **45.46 tok/s total** (self-reported decode 47.77) — **+31.7% total, +58.0% decode**. This is the model where WebLLM most clearly beats us on first-token latency (~150 ms implied vs our 453 ms). The previously published "75.7 vs 43.8, +73%" pair is **withdrawn** — a bench-harness defect had stopped our half paying prefill; and the 2026-07-28 pair (25.43 vs 14.15) was a degraded session that did not reproduce. Both are kept as dated history with the reason in [BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md).
- **Qwen3.5-4B (q4f16_1)** — append `?model=qwen35` to `zero-tvm.html` or `validate.html`; weights stream from [`mlc-ai/Qwen3.5-4B-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Qwen3.5-4B-q4f16_1-MLC) (~2.6 GB). The first *hybrid* on the engine: 24 gated-DeltaNet (linear-attention) layers + 8 gated-attention layers (GQA 16/4, head_dim 256, partial RoPE, sigmoid attention gate) — to our knowledge the first hand-written-kernel int4 gated-DeltaNet hybrid in a browser. Zero-TVM **65.28 tok/s total** (TTFT 171 ms, decode 73.30) vs WebLLM 0.2.84's prebuilt Qwen3.5-4B at **32.56 tok/s total** (self-reported decode 34.32) — **+100.5% total, +113.6% decode**, and the one model where first-token latency is a wash rather than a loss. The GDN decode kernels are still scalar (non-subgroup), so the decode number is a floor; prefill runs in chunks of ≤64 (202 tok/s on an 816-token prompt) and every model reuses its cross-turn prefix (turn-3 first token 14.3 s → 0.19 s, logits verified bit-identical to a fresh prefill). The previously published "65.7 vs 34.0, +93%" cross-check is **withdrawn** (same harness defect); the earlier 53.07/32.36 (+64%) and 47.99/31.99 (+50%) pairs were like-for-like but are superseded. One machine, one pair each; caveats in [BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md).

## Pages in this Space

- [`index.html`](./index.html) — the entrance: pick a model, chat in place (start here)
- [`zero-tvm.html`](./zero-tvm.html) — the chat surface as a direct link
- [`validate.html`](./validate.html) — multi-prompt smoke test
- [`docs.html`](./docs.html) — annotated reference, including the kernel walkthrough

Six pages were removed on 2026-08-14 — `architecture`, `demo`, `compiler-chat`,
`dump`, `shaders`, `webllm-bench`. Three of them started multi-gigabyte
downloads on page load. `docs.html` carries what the first two explained.

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
