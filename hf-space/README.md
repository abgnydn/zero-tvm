---
title: Zero-TVM
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: static
pinned: true
license: mit
thumbnail: https://huggingface.co/spaces/abgunaydin/zero-tvm/resolve/main/og.png
short_description: 1B dense to a 35B MoE in the browser, on hand-written WGSL
tags:
  - webgpu
  - llm
  - inference
  - on-device
  - wgsl
models:
  - microsoft/Phi-3-mini-4k-instruct
  - Qwen/Qwen3-4B
  - Qwen/Qwen3.5-4B
  - meta-llama/Llama-3.2-1B-Instruct
  - mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC
  - mlc-ai/Qwen3-4B-q4f16_1-MLC
  - mlc-ai/Qwen3.5-4B-q4f16_1-MLC
  - mlx-community/Qwen3-4B-4bit
  - mlx-community/Qwen3-30B-A3B-4bit
  - mlx-community/Llama-3.2-1B-Instruct-4bit
  - mlx-community/Qwen3.8-27B-4bit
  - mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
  - lmstudio-community/Qwen3.5-9B-MLX-4bit
  - lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit
  - abgunaydin/Qwen3.6-35B-A3B-MLX-q3exp
---

# Zero-TVM

Models from a 1B dense up to a 35B sparse mixture-of-experts, running in your
browser on hand-written WGSL. No TVM, no ONNX, no WASM runtime.

The standard way to run an LLM in a browser is
[WebLLM / MLC-LLM](https://webllm.mlc.ai/), which autotunes its GPU shaders
with a compiler. This Space replaces that stack with shaders written by hand —
**10 kernel roles** for the Phi-3 forward pass, plus the gated-DeltaNet and
mixture-of-experts roles newer architectures need. It runs models WebLLM ships
(same weights, measured faster) and models it does not — Qwen3.6-35B-A3B, a
256-expert sparse MoE, was the first of those.

The whole forward pass is readable end-to-end in a single sitting. That is the
point.

## Headline result

Measured on an Apple M2 Max (Chrome 150, WebLLM v0.2.80, identical
Phi-3-mini-q4f16_1 weights, same-session head-to-head, 2026-07-30). Two
numbers, always both: **total** includes prompt processing, **decode**
excludes it.

| | WebLLM (TVM) | Zero-TVM |
|---|---|---|
| WGSL kernels (decode path) | **~11 TVM-generated** | **10 hand-written roles** |
| Dispatches per decode step | **342** | **260** default (228 with `?kv8=0&splitk=0`) |
| Total throughput (prefill + decode) | **59.95 tok/s** | **69.55 tok/s** — **+16.0%** |
| Decode only | **63.23 tok/s** (self-reported) | **83.10 tok/s** — **+31.4%** |
| Time to first token (~35-tok prompt) | **~150 ms** (implied) | **291 ms** — *WebLLM wins this one* |
| JS bundle (excl. weights) | **~6.0 MB** / ~2.2 MB gz | **~460 kB** / ~126 kB gz |

The gap grows with how recent the architecture is: +16% / +31% (total /
decode) on 2024's Phi-3, +32% / +58% on 2025's Qwen3-4B, +100% / +114% on
2026's Qwen3.5-4B. An observation across three points on one machine, not a
proven law. The honest negative: WebLLM reaches the first token faster on
short prompts.

Exact medians, methodology, and every withdrawn claim — dated, with reasons,
not deleted — live in
[BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md).

## Run it

The Space opens on a character-select entrance: pick a model and it loads and
chats in place. The roster runs strongest first, so it opens on Qwen3.8-27B.
[`zero-tvm.html`](./zero-tvm.html) is the same chat as a direct link; with no
`?model=` flag it boots Phi-3-mini.

Clicking ENTER starts the download — the sheet states the weight size and the
free RAM needed before you click, so the click is the consent. A link that
enters for you (`?chat=1`) is gated instead: a dialog names the model, its
download size and the memory allocated at boot, and nothing downloads until
you accept. [`share.html`](./share.html) asks the same question before a room
fetches anything.

Weights stream from Hugging Face into browser storage (OPFS) once; returning
to a model you already have starts it straight away. Every model's size, RAM
need and `?model=` flag comes from one table,
[`model-registry.ts`](https://github.com/abgnydn/zero-tvm/blob/main/src/zero-tvm/model-registry.ts) —
the cards render from it, so a card and the engine cannot disagree.

A model too big for one machine can be split by layer range across several:
each machine holds only its own layers, tokens hop over WebRTC, and a guest
with the room link just chats — no download, no WebGPU needed. The entrance
builds the links under *Too big for one machine? Split it*. Limits: MLX
checkpoints only, serving tabs must stay awake (there is a toggle), and no
TURN relay — home networks yes, corporate ones usually not. Details in the
[repo README](https://github.com/abgnydn/zero-tvm#the-swarm).

**Requirements**: Chrome / Edge with WebGPU and `shader-f16` (default on
Apple-Silicon macOS, available on most modern Windows / Linux GPUs). MoE
models also need subgroups. The KV cache is allocated at boot, so a model
needs the free RAM its card names.

## Models

The three below are the ones measured head-to-head against WebLLM on
identical weights (2026-07-30, corrected protocol: same session, both engines
paying full prefill). They are not the whole roster — the registry above is
the shipped list; its DeltaNet hybrids and sparse MoEs have no WebLLM build
to sit beside.

- **Phi-3-mini-4k-instruct (3.8B)** — `zero-tvm.html`'s no-flag default, and
  the source of every headline number above: **69.55 vs 59.95 tok/s total,
  +16.0%** (decode +31.4%).
- **Qwen3-4B** — `?model=qwen3` (~2.3 GB). **59.85 vs 45.46 tok/s total,
  +31.7%** (decode +58.0%). The model where WebLLM most clearly wins
  first-token latency.
- **Qwen3.5-4B** — `?model=qwen35` (~2.6 GB). The first hybrid on the engine:
  24 gated-DeltaNet layers + 8 attention layers. **65.28 vs 32.56 tok/s
  total, +100.5%** (decode +113.6%).

Earlier published pairs for Qwen3 and Qwen3.5, and a cross-turn-reuse figure
published here until 2026-08-19, are **withdrawn** — a bench-harness defect
and a prompt-rendering bug respectively.
[BENCH.md](https://github.com/abgnydn/zero-tvm/blob/main/BENCH.md) keeps each
with its reason.

## Pages in this Space

- [`index.html`](./index.html) — the entrance: pick a model, chat in place (start here)
- [`zero-tvm.html`](./zero-tvm.html) — the chat surface as a direct link
- [`share.html`](./share.html) — host a model for other machines, join a split, or chat as a guest
- [`validate.html`](./validate.html) — multi-prompt smoke test
- [`docs.html`](./docs.html) — annotated reference, including the kernel walkthrough
- [`agent-host.html`](./agent-host.html) — front door for local agent tools; needs a server on `127.0.0.1`, so it does nothing on its own

## URL flags

`zero-tvm.html` accepts query flags for A/B-ing shader variants without
rebuilding:

- `?sg=0` — disable all subgroup shaders (argmax / attention / QKV matmul)
- `?sgqkv=0` / `?sgattn=0` / `?sgargmax=0` — disable one at a time
- `?qkvtile=1` / `?qkvtile2=1` — opt into tiled QKV variants
- `?sgffn=0` — opt OUT of the tiled-subgroup fused FFN, on by default
- `?kv8=0` — opt OUT of the int8 KV cache, the default since 2026-08-18

One flag that is not a shader variant: `?ctx=N` sets the context budget in
tokens, clamped to the model's trained window. It is allocated at boot, so
raising it can fail there; the entrance offers the same knob as a picker of
named windows (Standard / Long / Full).

## How it relates to a published paper

Zero-TVM applies a broader thesis: per-dispatch overhead is the dominant tax
on browser GPU compute, and many workloads collapse from "needs a server" to
"runs in a browser tab" once the loop is fused. Two preprints document the
technique:

- [Single-Kernel Fusion for Sequential Fitness Evaluation via WebGPU Compute Shaders](https://doi.org/10.5281/zenodo.19342888) (the foundational result)
- [Single-Kernel Fusion for Autoregressive Transformer Decoding via WebGPU Compute Shaders](https://doi.org/10.5281/zenodo.19344277) (the LLM application — the work this Space embodies)

## Source / canonical site

- **GitHub**: [github.com/abgnydn/zero-tvm](https://github.com/abgnydn/zero-tvm)
- **Canonical site**: [zerotvm.com](https://zerotvm.com)
- **Research umbrella**: [kernelfusion.dev](https://kernelfusion.dev)
- **Companion benchmarks**: [gpubench.dev](https://gpubench.dev)

## License

MIT. Reference comparison code (WebLLM compiler-chat path) belongs to its
respective authors and is governed by their licenses.

## Author

Ahmet Barış Günaydın · [barisgunaydin.com](https://barisgunaydin.com) ·
senior full-stack consultant by day, independent researcher by night.
