# Zero-TVM

**Phi-3-mini running in a browser on 10 hand-written WGSL shaders. No TVM. No WebLLM runtime. No compiler.**

The standard way to run a modern LLM in a browser is [WebLLM / MLC-LLM](https://webllm.mlc.ai/), which ships an Apache-TVM compiler pipeline that emits 85 autotuned WGSL kernels and drives them from a WASM scheduler. This repo replaces that entire stack with **10 WGSL compute shaders and about 1,250 lines of TypeScript**, using the same model and the same quantized weights.

The whole forward pass — 32 transformer layers, paged KV cache, int4-dequant matmul, RoPE, fused FFN, RMSNorm, attention, argmax sampling — is readable end-to-end in a single sitting. That is the point.

## What's actually in the box

All numbers below are measured from the source and the build output in this repo. Performance numbers are intentionally omitted until they are benchmarked; see [Benchmarks](#benchmarks).

| | WebLLM (TVM) | Zero-TVM (this repo) |
|---|---|---|
| Unique WGSL shaders | **85** | **10** |
| Total WGSL lines | **12,962** (generated) | **792** (hand-written) |
| Dispatches per forward pass | **342** | **292** (–50 via fusion) |
| Runtime | TVM → WASM scheduler | Plain TypeScript, no runtime |
| Tokenizer | bundled from WebLLM | BPE from scratch (`tokenizer.ts`) |
| Weight loader | MLC's | Direct HuggingFace fetch |
| JS bundle (index.html, excl. model weights) | **6.0 MB** / 2.1 MB gz | — |
| JS bundle (zero-tvm.html, excl. model weights) | — | **14 kB** / 5.5 kB gz |

Zero-TVM issues **fewer** dispatches than TVM because it fuses operations TVM's default pipeline doesn't: `attention.wgsl` combines attention + paged-KV read, `fused_ffn.wgsl` combines gate + up + SiLU, `add_norm.wgsl` combines residual add + RMSNorm.

Every FLOP the model executes is in a file you can open. Every GPU buffer has a human label. Every dispatch in `src/zero-tvm/chat.ts` has a numbered comment explaining what it does.

## Why this might be interesting

Hand-written GPU kernels usually lose significantly to an autotuning compiler. The claim this repo is designed to test is: **for a decoder-only LLM of this shape, most of the compiler's complexity budget isn't buying much.** The expensive parts are matmul, attention, and int4 dequant. Everything else is plumbing. 10 shaders of plumbing, instead of 85.

Whether that's true in practice is a *benchmarking question* — the shader count and bundle size are objectively smaller, but whether the end-to-end throughput is competitive is something you should measure on your own hardware before believing. See below.

It also makes the stack *auditable*. If you want to instrument a layer, add a new fusion, test a different attention pattern, or teach someone how browser LLM inference works at the metal, there is no compiler in the way — just 792 lines of WGSL and 462 lines of TypeScript orchestrating them.

The closest reference point is Karpathy's [llm.c](https://github.com/karpathy/llm.c) (hand-written CUDA/C GPT-2). This is that thesis — *you don't need the giant framework* — ported to browser / WebGPU / int4 / paged KV / modern arch, for a model people actually use.

## How to run

**Requirements:** A recent Chrome or Edge with WebGPU enabled and the `shader-f16` feature available. Tested on macOS (M2 Pro, Chrome 120+). Other platforms should work but are untested.

```bash
npm install
npm run dev
```

Then open <http://localhost:5173/zero-tvm.html>. On first load the browser downloads the Phi-3-mini-4k-instruct Q4 weights from HuggingFace (several hundred MB) and caches them in the browser's storage. Subsequent loads are instant.

To build a deployable bundle:

```bash
npm run build   # → dist/
```

The build produces a multi-page Vite output with all demo pages (`index.html`, `compiler-chat.html`, `zero-tvm.html`, `dump.html`, `demo.html`, tests).

## Benchmarks

**Not yet measured in this tree.** Earlier commits on the compiler-based chat path (`phi3v2.ts`) reported ~34 tok/s on an M2 Pro. Zero-TVM uses the same 10 shaders and the same decode loop structure, so throughput is expected to be in the same ballpark, but this has **not been independently benchmarked** for the `zero-tvm.html` page. If you run it, file an issue with your hardware and numbers; I'll paste them here.

## The repository as an argument

The directory layout is the narrative arc of the project. Each page is a milestone.

```
index.html              → src/main.ts              (1) Baseline: WebLLM, untouched
compiler-chat.html      → src/compiler/chat-v2.ts  (2) Intermediate: WebLLM captures
                                                       dispatches, our shaders replay
                                                       279 of 342 of them
zero-tvm.html           → src/zero-tvm/chat.ts     (3) The result: all 342 replaced,
                                                       WebLLM never touched

dump.html               → src/dump-tvm.ts          Captures all 85 TVM-emitted WGSL
shaders.html            → src/dump-shaders.ts      Browses the captured shaders
demo.html               → src/demo.ts              Dispatch timeline visualization
test-shaders.html       → src/compiler/test-harness.ts
test-chain.html         → src/compiler/test-chain.ts
standalone-test.html    → src/standalone-test.ts
```

```
src/
  zero-tvm/             THE RESULT
    chat.ts               462 lines — full chat loop, prefill, decode, KV cache
    tokenizer.ts          248 lines — BPE tokenizer from scratch
    weight-loader.ts      318 lines — direct HuggingFace Phi-3-MLC fetch + parse

  compiler/             THE 10 SHADERS
    compiler.ts           225 lines — pipeline creation, PHI3 model constants,
                          weight buffer allocation. Not an optimizing compiler —
                          the name is historical.
    shaders/              10 hand-written WGSL files, 792 lines total:
      int4_matmul.wgsl         Used by QKV, OProj, and FFN-down matmuls
      int4_matmul_f32.wgsl     LM head (f32 output for stable argmax)
      rms_norm.wgsl
      add_norm.wgsl            Fused residual add + RMSNorm
      rope.wgsl                Splits fused QKV and applies rotary
      kv_append.wgsl           Writes into paged KV cache
      attention.wgsl           Paged attention (vLLM-style page table)
      fused_ffn.wgsl           Gate + up + SiLU, fused
      embedding.wgsl
      argmax.wgsl

  tvm-shaders/          THE EVIDENCE — all 85 TVM-emitted WGSL kernels,
                        captured from a running WebLLM session by
                        src/dump-tvm.ts. Keep this next to compiler/shaders/
                        and the replacement is auditable.
```

`RESEARCH.md` is a writeup of how the shader capture worked and what reading TVM's output revealed about its kernel set. `SHADER-ANALYSIS.md` is currently a placeholder for per-shader notes and is not yet populated.

## Known caveats

- **Phi-3-mini-4k-instruct Q4 only.** The constants in `src/compiler/compiler.ts` hard-code `D=3072`, `HEADS=32`, `HEAD_DIM=96`, `LAYERS=32`, `FFN=8192`, `VOCAB=32064`, `PAGE_SIZE=16`, `MAX_PAGES=257`. Porting to another architecture means editing `PHI3` and re-reading the shaders that bake those dimensions in as dispatch sizes.
- **Greedy decoding only.** Sampling is a single `argmax.wgsl` dispatch. No temperature, top-k, or top-p. Adding them is a new shader or a CPU-side sampler; deliberately left out to keep the stack minimal.
- **Sequential prefill.** Each prompt token is run through the full decode path. Fine for chat-length prompts, not optimized for long-context ingest.
- **Requires the `shader-f16` WebGPU feature.** Matmuls run in f16. The LM head uses an f32 output buffer (`int4_matmul_f32.wgsl`) so the 32064-wide argmax is stable.
- **Buffer aliasing workaround.** WebGPU disallows read+write to the same buffer in one dispatch, so `chat.ts` ping-pongs the residual between two buffers (`B.residual` / `B.residual2`). See the comment at `src/zero-tvm/chat.ts:160`.
- **Weights come from HuggingFace on first run.** The loader expects the [`mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC) repo layout (MLC's Q4f16_1 quantization, parameter names `transformer.h.N.mixer.*`). If MLC changes either, the loader needs an update.
- **Benchmarks not yet in-tree.** See the [Benchmarks](#benchmarks) section.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If this repo is useful to your research or writing, cite it as:

```
Gunaydin, A. B. (2026). Zero-TVM: Phi-3 in a browser on 10 WGSL shaders.
https://github.com/abgnydn/zero-tvm
```
