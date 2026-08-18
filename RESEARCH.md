# WebGPU Shader Interception for TVM-Compiled LLMs

**Date:** 2026-04-01
**Author:** Ahmet Baris Gunaydin
**Hardware:** Apple M2 Pro, macOS, Chrome WebGPU
**Model:** Phi-3-mini-4k-instruct (3.6B params, Q4 quantized)
**Runtime:** WebLLM (@mlc-ai/web-llm) backed by Apache TVM

## Abstract

We demonstrate the first runtime interception and replacement of TVM-compiled WebGPU compute shaders inside a production browser LLM (WebLLM). By monkey-patching the WebGPU API at the `createShaderModule`, `createComputePipeline`, and `dispatchWorkgroups` levels, we capture all 85 WGSL shader modules, map them to the model's execution graph, and successfully replace 3 core matmul shaders with tiled variants — producing numerically correct, coherent text output.

## Key Findings

### 1. TVM's Per-Token Execution Profile

Each decode token requires exactly **342 GPU dispatches**:

| Component | Dispatches | Shader | Entry Point |
|-----------|-----------|--------|-------------|
| Embedding lookup | 1 | [77] | `fused_dequantize_take1_kernel` |
| Initial RMSNorm | 1 | [66] | `rms_norm2_kernel` |
| **Per layer (×32):** | | | |
| QKV projection | 1 | [6] | `fused_dequantize1_NT_matmul10_kernel` |
| RoPE | 1 | [40] | `fused_rope_kernel` |
| KV cache append | 1 | [26] | `tir_kv_cache_transpose_append_kernel` |
| Paged attention | 1 | [29] | `batch_decode_paged_kv_kernel` |
| Output projection | 1 | [46] | `fused_dequantize2_NT_matmul11_kernel` |
| Residual + RMSNorm | 1 | [22] | `fuse_add_norm_decode_kernel` |
| FFN gate+up | 1 | [47] | `fused_dequantize3_NT_matmul12_kernel` |
| SiLU activation | 1 | [19] | `fused_split2_silu2_multiply2_kernel` |
| FFN down | 1 | [27] | `fused_dequantize4_NT_matmul13_kernel` |
| Residual + RMSNorm | 1 | [22] | `fuse_add_norm_decode_kernel` |
| **LM head + sampling** | 20 | various | argsort, cumsum, sample |
| **Total** | **342** | | |

Additionally, **357 `writeBuffer` calls** per token update uniforms between every dispatch. This means a `writeBuffer` precedes every single dispatch — there are **zero consecutive dispatches** that could be batched without modifying the shader pipeline.

### 2. TVM Already Fuses Elementwise Operations

TVM is not naive — it already fuses across elementwise boundaries:
- `fused_dequantize + NT_matmul` — int4 dequantization fused into matmul
- `fuse_add_norm_decode` — residual addition fused with RMSNorm
- `fused_split_silu_multiply` — gate split + SiLU + elementwise multiply

It does NOT fuse across matmul boundaries, which is where cross-operation fusion would need to happen.

### 3. Submit-Level Optimization Is Impossible

We attempted two submit-level optimizations:

**Attempt 1: setTimeout batching** — Accumulate command buffers, flush on next event loop tick.
- Result: 2x decode speedup (47→99 tok/s) but **corrupted output**. WebLLM's `mapAsync` read-backs executed before the deferred submit.

**Attempt 2: Flush-on-read coalescing** — Accumulate command buffers, flush synchronously before `mapAsync`/`writeBuffer`/`onSubmittedWorkDone`.
- Result: 1.003x reduction (essentially 1:1). TVM calls `writeBuffer` between every submit, so there is nothing to coalesce.

**Conclusion:** Submit-level optimization cannot work because TVM writes unique uniforms before every dispatch.

### 4. Shader Interception Works

We successfully:
1. Captured all 85 WGSL shader modules (13,042 lines total) via `createShaderModule` interception
2. Mapped all 85 compute pipelines to their shader modules via `createComputePipeline`/`createComputePipelineAsync` interception
3. Detected int4 matmul shaders by structural pattern matching (workgroup reduction buffer, dequant formula, weight stride)
4. Generated tiled replacement shaders (4 rows per workgroup, shared memory input caching)
5. Replaced 3 decode-path matmul shaders at runtime
6. Adjusted dispatch workgroup counts via `dispatchWorkgroups` interception
7. **Produced coherent text output** with the replaced shaders — responses are valid English but terminate prematurely (~12-16 tokens vs ~50-70 baseline), indicating small numerical drift from f16 accumulation order differences that compounds across 32 layers and shifts the sampling distribution toward early EOS

### 5. Tiled Matmul Does Not Help on Apple Silicon

The tiled matmul (shared memory input caching) showed **no speedup on M2 Pro** (10-22 tok/s vs 27-48 tok/s baseline — actually slower due to shared memory loading overhead).

**Root cause:** Apple M2's unified memory architecture has a large L2 cache that automatically caches the 6KB input vector across workgroups. The redundant global memory reads that tiling eliminates are served from L2 cache at near-shared-memory speeds. The shared memory loading adds overhead for zero bandwidth benefit.

This optimization would likely help on discrete GPUs (NVIDIA/AMD) where:
- L2 cache is smaller relative to working set
- Global memory has higher latency (GDDR6/HBM vs unified)
- Shared memory provides a meaningful latency advantage

## Technical Architecture

```
Browser (Chrome)
  ↓
navigator.gpu.requestAdapter() ← intercepted
  ↓
adapter.requestDevice() ← intercepted
  ↓
GPUDevice ← monkey-patched:
  ├── createShaderModule()     → capture WGSL source, optionally replace
  ├── createComputePipeline()  → map pipeline to shader, tag if replaced
  ├── createComputePipelineAsync() → same
  ├── createCommandEncoder()   → patch beginComputePass
  │   └── beginComputePass()   → patch setPipeline + dispatchWorkgroups
  │       ├── setPipeline()    → detect replaced pipeline
  │       └── dispatchWorkgroups() → adjust workgroup count if replaced
  ├── queue.submit()           → count submits
  ├── queue.writeBuffer()      → count writes, track per-dispatch
  └── createBuffer()           → intercept mapAsync for token boundary detection
```

## Matmul Shader Analysis

TVM's int4 matmul (117 lines WGSL):
- **Workgroup size:** 64 threads
- **Output:** 1 element per workgroup
- **Strategy:** Each thread loads 1 packed u32 (8 int4 weights), performs 8 fma ops, repeats 6 times (6 × 64 × 8 = 3072 dot product). Workgroup reduction via 6-stage barrier tree (64→32→16→8→4→2→1).
- **Dequant:** `(f16(nibble) - 7.0) * group_scale`, group size = 32 elements
- **Bindings:** output (f16 rw), input (f16 r), scales (f16 r), weights (u32 r), uniforms

Our tiled replacement (92 lines WGSL):
- **Workgroup size:** 256 threads (4 rows × 64 threads/row)
- **Output:** 4 elements per workgroup
- **Strategy:** Cooperative shared memory load of input vector (3072 f16), then per-row dot product with same reduction.
- **Same bindings, same dequant, near-identical results.** Small numerical differences from f16 accumulation order (shared memory round-trip + different loop structure) compound across 32 layers, causing premature EOS (~12-16 tokens vs ~50-70 baseline). The output is coherent English but not bit-identical to TVM's original.

## Files

| File | Purpose |
|------|---------|
| `src/profiler.ts` | GPU dispatch profiler + shader/pipeline capture |
| `src/tiled-matmul.ts` | Tiled int4 matmul WGSL generator + detection |
| `src/interceptor.ts` | Runtime shader replacement via WebGPU monkey-patching |
| `src/shader-dump.ts` | WGSL source extraction utilities |
| `src/engine.ts` | WebLLM wrapper |
| `src/main.ts` | App orchestration, `?fuse=1` enables interception |

### 6. Dispatch Replay — Bypass TVM's Runtime (Failed)

Attempted to record one decode token's dispatch tape (pipelines, bind groups, workgroups, copies), freeze it with stable buffers, and replay on subsequent tokens while no-oping TVM's GPU calls.

**Phase 1 (Tape Freezing):** Successfully captured 343 ops (342 dispatches + 1 copy), created stable buffer copies, rebuilt bind groups with stable refs, identified 6 patch slots from write diff. ✓

**Phase 2 (Replay):** Failed. Two fundamental issues:
1. TVM's WASM write sequence is non-deterministic during replay — buffer allocation order changes when processing fake GPU results, so write-by-index patching targets wrong buffers
2. TVM's WASM grinds to 0.1 tok/s when receiving fake GPU operations — internal error/timeout paths activate

**Root cause:** Replay requires replacing TVM's runtime loop entirely (our own JS token loop: GPU→detokenize→re-encode→submit), not just no-oping GPU calls. TVM's WASM maintains complex internal state that diverges immediately when GPU results don't match expectations.

### 7. The Definitive Finding

The batching experiment (Phase 6) is the most conclusive result: **337x fewer submits, correct output, 30% slower.** This proves:

- Chrome's GPU driver on Apple M2 already pipelines CPU→GPU work optimally
- There is no scheduling overhead to eliminate — the driver hides it through pipelining
- The only way to go faster is to reduce the actual compute/memory work (smaller weights, fewer layers, speculative decoding)

WebLLM at ~26-48 tok/s on M2 Pro is operating at **~50-85% of the theoretical memory bandwidth limit** (1.8GB weights / 200GB/s = 9ms = 111 tok/s theoretical max).

### 8. Own Decode Loop — No TVM in the Hot Path (SUCCESS)

Built a standalone decode loop that drives TVM's compiled GPU shaders directly, with zero TVM WASM in the token generation path.

**Architecture:**
1. Let WebLLM/TVM load model normally (prefill + first decode token)
2. Capture one decode token's full GPU state: 342 dispatches, 357 writeBuffer calls, all bind group layouts/entries
3. Build stable buffer copies and reconstructed bind groups
4. **Own token loop:** write uniforms → 342 dispatches in pipelined batches → read token via mapAsync → repeat

**Results:**
- **35-55 tok/s own loop vs ~27 tok/s baseline = 1.3-2x speedup**
- First 6 tokens match baseline perfectly
- GPU determinism test confirms: **baseline diverges from itself at token 7** (f16 parallel reduction non-determinism). Our own loop diverges at token 6 — within 1 token of the hardware limit.
- **Our decode loop is as correct as the GPU hardware allows.**
- Clean EOS detection (tokens 2, 32000, 32007)
- Zero TVM WASM calls during generation

**What this proves:**
- TVM's WASM runtime overhead accounts for ~30-50% of per-token latency
- The GPU compute itself (342 dispatches at ~9ms) is only half the story
- Eliminating the JS↔WASM↔GPU bridge yields a real, measurable speedup
- GPU f16 non-determinism makes bit-exact replay impossible beyond ~6-7 tokens regardless of approach

**Token-0 divergence:** Output differs from TVM baseline at token 0 (`4685` vs `4955`). Root cause: batched encoder submission (32 compute passes per encoder) changes f16 parallel reduction ordering vs TVM's 1:1 encoder:submit. The output is coherent English — valid but different sampling path. Same class as baseline-vs-baseline non-determinism (measured at index 7-156 depending on prompt).

**Key insight — the whiteboard problem:** TVM writes uniform data to buffers immediately before each dispatch, then TVM's next token overwrites those buffers. Our approach: snapshot all 357 writes during capture, replay snapshot data to original buffers before each token, then batch 342 dispatches. This preserves correct uniform values per dispatch.

**Architecture:** Uses TVM's original bind groups (captured during recording, kept alive via destroy prevention). Writes only to original buffers — no copies. Patches token_id (write[0]), position (writes[4,11,12]), and nnz_pages at page boundaries.

**Current limitation:** Page table management needed for responses longer than ~16 tokens past the recording position. The `nnz_pages` uniform in the 56B attention struct needs incrementing at page boundaries.

**Files:**
- `src/own-loop.ts` — The decode engine: capture, build, generate
- `src/own-loop-test.ts` — Automated test comparing own loop vs baseline
- `own-loop-test.html` — Browser test page

## What's Next

Potential optimization paths for Apple silicon:
1. **Vectorized loads** — Use `vec4<f16>` for weight/input access (4x fewer load instructions)
2. **Wider workgroups** — 128+ threads per row to reduce instruction count via ILP
3. **Fused RoPE+KV** — Combine RoPE and KV cache append (saves 32 dispatches/token)
4. **Fused matmul+activation** — Combine FFN gate+up matmul with SiLU (saves 32 dispatches/token)
5. **Sampling optimization** — Replace 20-dispatch hierarchical argsort with single top-k shader

## Reproduction

```bash
cd webgpu-fusion-webllm
npm install
npx vite --port 5180

# Baseline (no interception):
# Open http://localhost:5180/

# Fusion mode (shader replacement):
# Open http://localhost:5180/?fuse=1

# DevTools console:
# __dumpShader(6)         — view QKV matmul WGSL
# __dumpLayerShaders()    — view all 11 decode shaders
# __capturedShaders       — full shader/pipeline data
```

---

# Zero-TVM: The Minimal Counter-Implementation

**Date:** 2026-04-20

Where the sections above interrogated TVM's compiled runtime from the outside, Zero-TVM answers a narrower question from the other direction:

> For **one specific model** (Phi-3 Mini, Q4F16_1) on **one specific runtime** (the browser, via WebGPU), how much of the compiler / runtime stack is actually load-bearing?

The answer this repo ships: ten hand-written WGSL kernel roles ([`src/compiler/shaders/`](src/compiler/shaders/)), ~14 kB of TypeScript runtime ([`src/compiler/runtime.ts`](src/compiler/runtime.ts) + [`src/zero-tvm/`](src/zero-tvm/)), and no ML framework runtime in the hot path. Decode uses 228 dispatches/token on this repo's forward pass, versus the 342 dispatches/token measured from TVM-WebLLM running the same Q4 weights.

## Timing Context

In **February 2026**, Hugging Face shipped Transformers.js v4 — a C++ WebGPU runtime co-developed with Microsoft's ONNX Runtime team — as the production answer for browser LLM inference. It covers ~200 architectures, ships a substantial JS + native bundle, and is the right choice when you want "run any model in the browser."

In **April 2026**, this repo takes the opposite bet: for Phi-3 Mini specifically, the correct implementation is 10 WGSL files and a few hundred lines of TypeScript. The argument is the llm.c / nanoGPT argument applied to the browser: most of the stack isn't load-bearing for one model.

## Why Fewer Dispatches Matter: Maczan's Overhead Numbers

Maczan, J. (2026). *Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors, Three Backends, and Three Browsers.* arXiv:2604.02344.

Key measurements from that paper:

- **Vulkan:** 24–36 µs per dispatch
- **Metal:** 32–71 µs per dispatch
- **~95 µs total per-operation overhead** once CPU/JS plumbing is included
- Naive microbenchmarks that dispatch one op in a tight loop **overestimate peak throughput by ~20×** because they measure a warm cache / pipelined case that never occurs in a real forward pass

Maczan measured the cost. Zero-TVM is one demonstration of the corresponding remedy: fewer, fatter kernels.

**Worked back-of-envelope for this repo on Metal:** `(342 − 228) × 32 µs ≈ 3.6 ms` saved per token from kernel-count alone. The decode path fuses QKV projection, RoPE, and KV-cache append into a single shader ([`qkv_fused.wgsl`](src/compiler/shaders/qkv_fused.wgsl)), so each layer costs 7 dispatches instead of 9.

The per-decode dispatch table for this repo ([`src/zero-tvm/chat.ts`](src/zero-tvm/chat.ts)):

| Component | Dispatches | Shader file |
|-----------|-----------:|-------------|
| Embedding | 1 | `embedding.wgsl` |
| Initial RMSNorm | 1 | `rms_norm.wgsl` |
| **Per layer (×32):** | | |
| Fused QKV proj + RoPE + KV append | 1 | `qkv_fused.wgsl` |
| Paged attention | 1 | `attention.wgsl` |
| O projection | 1 | `int4_matmul` (generated) |
| AddNorm #1 | 1 | `add_norm.wgsl` |
| Fused FFN (gate+up+SiLU) | 1 | `fused_ffn.wgsl` |
| FFN down | 1 | `int4_matmul` (generated) |
| AddNorm #2 | 1 | `add_norm.wgsl` |
| Final RMSNorm | 1 | `rms_norm.wgsl` |
| LM head | 1 | `int4_matmul_f32` (generated) |
| Argmax | 1 | `argmax.wgsl` |
| **Total** | **228** | |

Eleven unique kernels (`int4_matmul` and `int4_matmul_f32` are two variants of the same generated kernel shape — see `src/compiler/shaders/int4_matmul.gen.ts` — one f16-out, one f32-out for stable argmax; `qkv_fused` is the decode-path fusion; `rope.wgsl` and `kv_append.wgsl` are retained for prefill and parity testing).

## Three-Way Comparison (April 2026)

| | **Zero-TVM** (this repo) | **TVM / WebLLM** | **Transformers.js v4 + ORT-WebGPU** |
|---|---|---|---|
| Unique WGSL kernels | 11 | 85 captured; ~11 active on decode path | ONNX Runtime-generated (not user-visible) |
| Total hand-written WGSL LOC | ~966 | 0 (compiler-generated, 13,042 lines captured) | 0 (runtime-generated) |
| Dispatches / decode token | 228 | 342 (measured) | not measured here |
| Runtime language | TypeScript | Apache TVM WASM runtime | C++ via ONNX Runtime WebGPU EP |
| Approx. runtime footprint (non-weight) | ~14 kB TS | ~50 MB WASM | ~10 MB JS + native runtime |
| Model scope | Phi-3 Mini (Q4F16_1) only | ~20 models via MLC compilation | ~200 architectures |
| Designed for | "readable in one sitting" | general browser LLM production | general browser LLM production |

Transformers.js v4 is the right choice for most production work. Zero-TVM is not trying to replace it. The point of the contrast is that when a stack can assume a single model/precision, most of the compiler and runtime surface is optional.

Speed (2026-07-30 same-session head-to-head): Zero-TVM measures 69.55 tok/s total wall-clock (prefill + decode) and 83.10 tok/s decode-only on an Apple M2 Max; WebLLM on the same machine, same session, same weights measures 59.95 tok/s total and 63.23 tok/s decode (via `webllm-bench.html` / `npm run bench`, recorded in `BENCH.md`). Zero-TVM is +16.0% on total and +31.4% on decode — one machine, one model, one browser, short-context, one pair. WebLLM has the better time-to-first-token on short prompts (~150 ms implied vs our 291 ms); that negative is published in `BENCH.md`, not buried. An earlier M2 Pro measurement with a since-fixed engine read ~22% behind, and a 2026-07-30 harness fix withdrew two 2026-07-29 Qwen pairs that were not like-for-like; both histories are preserved in `BENCH.md`.

## What This Repo Is NOT

- **Not a compiler.** Nothing here emits WGSL. The ten kernel roles are human-written.
- **Not cross-architecture.** Hard-coded to Phi-3 Mini with Q4F16_1 weights. No Llama / Qwen / Gemma path without shader work.
- **Not cross-runtime.** Browser WebGPU only. No Node / Bun / Deno target.
- **Not a production replacement for Transformers.js v4.** It does fewer things, on purpose.
- **Not currently faster than WebLLM on Apple silicon.** That's the subject of Milestones 2–5.

It IS a minimal-surface-area demonstration that for one fixed model shape, the compiler/runtime complexity is not required to do browser LLM inference end-to-end.

## Experimental: int8 KV Cache

An opt-in int8 KV cache halves KV memory. As of 2026-08-17/18 it runs on the
unfused path, on hybrid (GDN) specs and through chunked prefill — it was gated
to the fused QKV composition before that, which meant Phi-3 alone and so no
model where context is the constraint. MLA is the one exclusion: it caches a
latent rather than per-head K/V.

Shaders: [`kv_quantize_int8.wgsl`](src/compiler/shaders/kv_quantize_int8.wgsl)
(quantise on append, one f16 scale per (page, slot, head, K|V)),
[`attention_int8.wgsl`](src/compiler/shaders/attention_int8.wgsl) (decode) and
[`attention_prefill_int8.wgsl`](src/compiler/shaders/attention_prefill_int8.wgsl)
(chunked prefill); the fused path additionally uses
[`qkv_fused_scratch.wgsl`](src/compiler/shaders/qkv_fused_scratch.wgsl).

Enable with `?kv8=1` on the chat page, or `int8KV` via the library. Cost is one
extra dispatch per layer and ~5-8% of prefill throughput; memory is 1.98x
smaller.

VALIDATED, which the earlier version of this section said it was not: paired
perplexity against an f16 cache on Qwen3.6-35B-A3B measured -0.09% at 1k
windows and +0.10% at 4k, both within noise; greedy output is token-identical
to f16 on llama32 (5/6 prompts, sixth differs by a trailing period), qwen35
(6/6) and qwen36q3 (5/6, sixth a correct paraphrase). See
docs/TURBOQUANT_PLAN.md for the measurements and for why 4-bit was rejected.

## Planned: Multi-Token Batched Forward

**Motivation.** Decode is memory-bandwidth bound: each forward reads ~1.86 GB
of weights. Two user-visible problems follow from the current
one-token-at-a-time engine:

1. **Prefill is slow.** A 200-token prompt fires 200 sequential forwards =
   372 GB of weight reads. At ~80 GB/s effective bandwidth on M1 that's ~4.7 s
   before the first generated token appears, despite each forward being well
   pipelined by the GPU command chain.
2. **Speculative decoding can't help.** Prompt-lookup spec decoding wants to
   verify N candidates in parallel, but sequential N-token forwards pay
   N × weight bytes — with typical accept rates (~1.5–2) that's **slower**
   than plain decode on this hardware. Spec decoding only wins if the N
   forwards share a single weight read — i.e. M-dim batching at the kernel
   level.

**Proposal.** Generalise every activation buffer and every compute shader
from `[D]` / `[1, K] × [K, N]` to `[M, D]` / `[M, K] × [K, N]`, where M is
the batch dim in the query axis. Weights stay `[K, N]` (or packed int4 of
same shape) — they are read once per tile and reused across all M rows.

**Shader changes** (keeping current f16-KV path; int8-KV can follow):

| shader | change |
|---|---|
| `embedding.wgsl` | lookup M token ids, write `[M, D]` |
| `rms_norm.wgsl` | per-row norm across M rows |
| `qkv_fused*.wgsl` | matmul yields `[M, QKV_DIM]`; RoPE reads per-row position; writes M new slots to KV pages |
| `attention.wgsl` / `_sg` | each of M queries attends to its own history slice; naturally a grid of (M, HEADS) workgroups |
| `int4_matmul*` (generated) | GEMM `[M, K] × [K, N]` instead of GEMV — the key shape change. Each output WG tiles both M and N rows |
| `fused_ffn*.wgsl` | gate/up GEMM `[M, D] × [D, FFN]`; per-row SiLU and elementwise multiply |
| `add_norm.wgsl` | per-row residual add + norm across M rows |
| `argmax*.wgsl` | per-row argmax → M token ids |
| LM head matmul | already uses the same int4 GEMV kernel — gets M-dim automatically when the matmul is generalised |

**Engine changes** (`src/zero-tvm/chat.ts`):

- Activation buffer size → `PIPELINE_DEPTH × M × D × 2` bytes (dynamic M up
  to some `M_MAX`, e.g. 8 for prefill chunks + spec decode).
- `submitStep` takes `tokenIds: Int32Array` and `positions: Int32Array` of
  length M; writes both to their respective uniforms.
- Prefill path: chunk the prompt into groups of `M_MAX`, submit one batched
  forward per chunk, only read back the last argmax of the last chunk.
- Decode path: still M=1 by default, plus an opt-in spec-decode mode that
  runs M=4 or M=5 with candidates from a prompt-lookup lookback.

**Expected wins** (on M1, against the current 42 tok/s / ~4 s prefill):

- Prefill for a 200-token prompt: ~4.7 s → ~0.3 s (batched = 1 weight read
  instead of 200; activation work scales linearly but is tiny vs weights).
- Decode with working spec decoding + 2× accept rate: ~42 → ~70 tok/s.
- Decode with 3× accept rate (repetitive prompts): ~42 → ~95 tok/s.

**Risk**. This is a ~10-shader rewrite with new bind layouts and a new
engine path. Correctness risk is non-trivial — attention masking per-row
is the subtlest piece. Landing strategy: gate the whole batched path behind
a URL flag, keep the current M=1 engine untouched, ship one shader at a
time, A/B test each with the existing tooling.

## Parallel Work: bitnet.js

[qwatts-dev/bitnet.js](https://github.com/qwatts-dev/bitnet.js) pursues a very similar philosophy — no TVM, no WASM runtime, hand-written WGSL — but targets BitNet b1.58 ternary weights rather than standard int4. Different architectural bet (ternary vs int4), same llm.c-style argument about surface area. Worth knowing about when evaluating whether "hand-written WGSL for one model family" is a niche or a pattern.
