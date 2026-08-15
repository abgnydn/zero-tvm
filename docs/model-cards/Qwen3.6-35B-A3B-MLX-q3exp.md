---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- image-text-to-text
- conversational
- webgpu
- mixed-precision
---

# Qwen3.6-35B-A3B — MLX, 3-bit experts / 4-bit everything else

**▶ Try it in your browser — no install: [zerotvm.com](https://zerotvm.com/zero-tvm.html?model=qwen36q3)** · [same demo as a Space](https://huggingface.co/spaces/abgunaydin/zero-tvm)
(WebGPU; needs ~20 GB free RAM as shipped; weights stream from this repo and cache in OPFS)

The checkpoint keeps the base model's vision tower, so `mlx_lm` loads it as the
multimodal model it is. The browser runtime linked above is text-only — it does
not build the vision path.

A mixed-precision requantization of
[lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit)
(itself a 4-bit MLX build of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)):

- **expert stacks** (`switch_mlp.*` and the `shared_expert.*` folded into them): **3-bit**, group 64, MLX affine
- attention, gated-DeltaNet, embeddings, lm_head: **4-bit** (unchanged from the source)
- router and shared-expert gate: **8-bit** (unchanged)

Download: **16.36 GB** (from 19.5). Resident at decode: **15.7 GB** (from
19.7), which is the difference between running and not running beside a browser
on a 32 GB Mac. Experts were chosen because they are 16.2 GB of that 19.7 GB
resident set, and because only 8 of 256 run per token — each expert's error is
diluted, so expert-only 3-bit is a far milder intervention than 3-bit anywhere
dense.

## The cost: +10.4% perplexity vs the 4-bit build

Measured 2026-08-14 with
[`scripts/quality-ab.py`](https://github.com/abgnydn/zero-tvm/blob/main/scripts/quality-ab.py):
both builds score the identical 24 independent windows of 512 tokens and the
test differences per window, which cancels between-window difficulty — the
same data scored unpaired gives z = 1.3, i.e. nothing.

| build | perplexity | 95% CI |
|---|---:|---|
| Qwen3.6-35B-A3B MLX **4-bit** (source) | **26.179** | [24.762, 27.677] |
| **this build** (3-bit experts) | **28.908** | [27.364, 30.540] |

**+10.4%, paired z = 18.5, worse on 24 of 24 windows.** The harness's verdict
is MARGINAL: the regression is real, and perplexity is the wrong instrument to
decide whether it matters — published work puts the math-accuracy drop at
3-bit near 3x the perplexity drop, and no task benchmark (MMLU, GSM8K, …) has
been run on either build. If you have the RAM, use the 4-bit build; pick this
one because it fits.

An earlier revision of this card carried a layer-0 block-cosine (0.936) and
called quality unmeasured. The A/B above supersedes that.

## Speed, measured in the browser

**65.56 total tok/s** (decode 74.87 tok/s, TTFT 194 ms, run spread 65.3–65.6)
in zero-tvm's WebGPU engine — Apple M2 Max, Chrome, 128-token target, median
of 5 runs plus warmup, total = prefill + decode wall clock, 2026-08-13. This
supersedes the ~55 tok/s single owner-session figure an earlier revision of
this card carried.

## Running in less memory — the expert pool (library API)

zero-tvm's engine can hold a subset of the 256 experts in an LRU slot pool and
stream the rest from the on-disk weight cache on demand
(`DecodeEngineOptions.expertPool`, with weights loaded via the matching
`loadWeights` option). This is the library API only — the website does not
wire it. Measured 2026-08-15 on this checkpoint, blocking `generate()` path,
AC power, 2 rounds × 512 tokens, every row token-identical to unpooled:

| experts resident | resident memory | speed | warm LRU hit rate |
|---|---:|---:|---:|
| 256/256 (as shipped) | 15.7 GB | 58.6 tok/s (this harness; see Speed above for the browser protocol figure) | — |
| 128/256 | ~8.4 GB | 15.3 tok/s | 93.5% |
| 96/256 | ~6.6 GB | 15.0 tok/s | 90.4% |
| 64/256 | ~4.8 GB | 11.7 tok/s | 83.8% |

The flat 96→128 step says the cost is the per-layer readback, not the misses
— which is also why the quarter pool is nearly as fast as the half. With the
engine's speculative prefetch on, warm hit rates rise to 97.9–98.9%, at
similar speed.

Every pooled configuration is **token-exact** on both generate paths: pooled
generation is token-identical to unpooled over 512-token generations
(verified on the blocking path for this checkpoint and on both paths for the
30B sibling), and speculative prefetch cannot change a token — predicted ids
only warm the pool and never reach a dispatch.

Why it is slow today: each MoE layer's router ids must be read back GPU→CPU
before that layer's experts can be fetched (WebGPU has no GPU-side wait), so
the per-layer submit and readback — not bandwidth — is the stall. Details and
the measurement trail: `docs/MOE_CHUNK_PLAN.md` in the source repo.

## Use with mlx_lm

`config.json` carries per-tensor quantization overrides, so this loads like
any other MLX checkpoint — the perplexity A/B above ran on it through
`mlx_lm`:

```python
from mlx_lm import load, generate
model, tokenizer = load("abgunaydin/Qwen3.6-35B-A3B-MLX-q3exp")
print(generate(model, tokenizer, "List the planets of the solar system.", max_tokens=64))
```

## Provenance

Produced by
[`scripts/convert-q3-experts.py`](https://github.com/abgnydn/zero-tvm/blob/main/scripts/convert-q3-experts.py):
per expert tensor, `mx.dequantize(bits=4)` → `mx.quantize(bits=3, group_size=64)`;
everything else copied verbatim. Experts are selected by path
(`.mlp.switch_mlp.`, `.mlp.shared_expert.` — the shared expert is included
because the engine stacks it as an extra expert index, and one stacked tensor
must be one format), and the script refuses a run that converts nothing rather
than writing a byte-identical copy an A/B would misread as "no difference".
`config.json` gains per-path quantization overrides in the same style the
source checkpoint already uses for its 8-bit router, which is why `mlx_lm`
loads the result directly.

Requantizing from the 4-bit build (rather than the bf16 original) adds only a
small error on top of 3-bit's own: snapping an already-quantized value to a
coarser grid mostly lands where the original would have.

## Related

The engine that loads this checkpoint, and the other browser-native simulators
built on the same hand-written WGSL approach:

| Space | What it runs |
|---|---|
| [zero-tvm](https://huggingface.co/spaces/abgunaydin/zero-tvm) | this checkpoint and the rest of the model roster, on hand-written WGSL |
| [neuropulse](https://huggingface.co/spaces/abgunaydin/neuropulse) | a full LLM forward pass rendered 1:1 from live activations |
| [webgpu-dna](https://huggingface.co/spaces/abgunaydin/webgpu-dna) | Geant4-DNA Monte Carlo track structure |
| [webgpu-q](https://huggingface.co/spaces/abgunaydin/webgpu-q) | quantum chemistry — HF/UHF, DFT, MP2, CCSD(T), EOM-CCSD |
| [webgpu-fly](https://huggingface.co/spaces/abgunaydin/webgpu-fly) | FlyWire connectome and MANC spine, realtime |
| [enter-the-painting](https://huggingface.co/spaces/abgunaydin/enter-the-painting) | any image lifted into a 3D Gaussian-splat cloud |
| [draw-instant](https://huggingface.co/spaces/abgunaydin/draw-instant) | Stable Diffusion with a fused U-Net pass |

Source: [github.com/abgnydn/zero-tvm](https://github.com/abgnydn/zero-tvm)
