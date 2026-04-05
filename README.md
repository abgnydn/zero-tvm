# Zero-TVM

**Phi-3-mini running in a browser on 10 hand-written WGSL shaders. No TVM. No WebLLM runtime. No compiler.**

The standard way to run a modern LLM in a browser is [WebLLM/MLC-LLM](https://webllm.mlc.ai/), which ships an Apache-TVM compiler pipeline that emits ~85 autotuned WGSL kernels and drives them from a WASM scheduler. This repo replaces that entire stack with **10 WGSL compute shaders and ~700 lines of TypeScript**, keeps the same model and the same quantized weights, and runs at roughly the same speed (~34 tok/s on an M2 Pro).

The whole forward pass — 32 transformer layers, paged KV cache, int4-dequant matmul, RoPE, fused FFN, RMSNorm, attention, argmax sampling — is readable end-to-end in one afternoon. That is the point.

## The claim, precisely

| | WebLLM (TVM) | Zero-TVM (this repo) |
|---|---|---|
| Unique WGSL shaders | 85 | **10** |
| Dispatches per forward pass | 342 | **342** |
| Runtime | TVM → WASM scheduler | Plain TypeScript |
| Total WGSL lines | ~5000+ (generated) | **792** (hand-written) |
| Tokenizer | WebLLM's | BPE from scratch |
| Weight loader | MLC's | Direct HuggingFace fetch |
| JS bundle size (excluding model weights) | **6.0 MB** (2.1 MB gz) | **14 kB** (5.5 kB gz) |
| Phi-3-mini-4k Q4 decode speed, M2 Pro | ~40 tok/s | **~34 tok/s** |

Every FLOP the model executes is in a file you can open. Every buffer is labeled. Every dispatch has a comment explaining what it does.

## Why this is interesting

Hand-written GPU kernels usually lose 2–5× to an autotuning compiler. Coming within ~20% of TVM's tuned output with 10 shaders is evidence that — for a decoder-only LLM of this shape — **the compiler's complexity budget isn't buying as much as it looks like it is**. The expensive parts are matmul, attention, and int4 dequant. Everything else is plumbing.

It also makes the stack *auditable*. If you want to instrument a layer, add a new fusion, test a different attention pattern, or teach someone how browser LLM inference actually works at the metal, there is no compiler in the way.

The closest reference points are Karpathy's [llm.c](https://github.com/karpathy/llm.c) (hand-written CUDA/C GPT-2) and the various from-scratch GPT-2 ports. This is that thesis — *you don't need the giant framework* — ported to the hardest inference target (WebGPU, in a browser, int4, paged KV, modern arch, real chat loop) for a model people actually use.

## How to run

**Requirements:** Chrome/Edge 121+ (WebGPU + `shader-f16`), ~2.5 GB free disk for model cache.

```bash
npm install
npm run dev
```

Then open <http://localhost:5173/zero-tvm.html>. First load downloads Phi-3-mini-4k-instruct (Q4, ~2.3 GB) from HuggingFace and caches it in the browser. Subsequent loads are instant.

To build a deployable bundle:

```bash
npm run build   # → dist/
```

## The repository as an argument

The directory layout is the narrative arc of the project. Each page is a milestone, and each milestone is a checkpoint in the reverse-engineering of WebLLM's shader set.

```
index.html              → src/main.ts              (1) Baseline: WebLLM, untouched
compiler-chat.html      → src/compiler/chat-v2.ts  (2) Our compiler + TVM weights
zero-tvm.html           → src/zero-tvm/chat.ts     (3) Zero TVM, 10 shaders — the result

dump.html               → src/dump-tvm.ts          Captures all 85 TVM-emitted WGSL
shaders.html            → src/dump-shaders.ts      Browses the captured shaders
demo.html               → src/demo.ts              Interactive dispatch visualization
test-shaders.html       → src/compiler/test-harness.ts
test-chain.html         → src/compiler/test-chain.ts
standalone-test.html    → src/standalone-test.ts
```

```
src/
  zero-tvm/             THE RESULT — 462 lines, runs on the 10 shaders
    chat.ts               Full chat loop, prefill, decode, KV cache
    tokenizer.ts          BPE tokenizer from scratch
    weight-loader.ts      Direct HuggingFace Phi-3-MLC fetch + parse

  compiler/             THE 10 SHADERS
    compiler.ts           Pipeline creation, model constants, weight allocation
    shaders/              10 hand-written WGSL files, 792 lines total
      int4_matmul.wgsl         QKV / OProj / FFN down (f16)
      int4_matmul_f32.wgsl     LM head (f32 accumulator for sampling)
      rms_norm.wgsl
      add_norm.wgsl            Fused residual add + RMSNorm
      rope.wgsl                Splits QKV, applies rotary
      kv_append.wgsl           Writes into paged KV cache
      attention.wgsl           Paged attention (vLLM-style page table)
      fused_ffn.wgsl           Gate + up + SiLU, fused
      embedding.wgsl
      argmax.wgsl

  tvm-shaders/          THE EVIDENCE — all 85 TVM-emitted WGSL kernels,
                        captured by dump-tvm.ts. Keep this next to
                        compiler/shaders/ and the replacement is auditable.
```

`RESEARCH.md` is the writeup of how the shader capture worked and what we learned reading TVM's output. `SHADER-ANALYSIS.md` has per-shader notes. Read those before the code if you want the "why" before the "how."

## Known caveats

- **Phi-3-mini-4k-instruct Q4 only.** The shaders hardcode head dim 96, 32 layers, 32 heads, FFN 8192, vocab 32064. Porting to another architecture means re-reading the config and editing `PHI3` in `src/compiler/compiler.ts`.
- **Greedy decoding (argmax).** No temperature, top-k, or top-p. Adding them is a new shader or a CPU-side sampler; deliberately left out to keep the stack minimal.
- **Sequential prefill.** Each prompt token is run through the decode path. Fine for chat-length prompts; not optimized for long-context ingest.
- **Requires `shader-f16`.** Matmuls are f16. The LM head uses f32 accumulation because argmax over 32064 logits is sensitive to precision drift.
- **Weights are downloaded from HuggingFace on first run.** The loader expects the [`mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC`](https://huggingface.co/mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC) repo layout.

## License

MIT. See [LICENSE](LICENSE).

## Citation

If this repo is useful to your research or writing, cite it as:

```
Gunaydin, A. B. (2026). Zero-TVM: Phi-3 in a browser on 10 WGSL shaders.
https://github.com/abgnydn/zero-tvm
```
