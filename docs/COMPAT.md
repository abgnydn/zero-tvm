# Model compatibility

<!-- GENERATED from src/compiler/constraints.ts (SUPPORT_MATRIX) by
     scripts/add-model.mjs --compat. Edit the matrix there, not this file. -->

`scripts/add-model.mjs <hf-repo>` checks a checkpoint against this matrix and
either generates a registered ModelSpec (green) or names the missing kernel per
failure (red). The checks themselves live in `src/compiler/constraints.ts` —
same file as this table's source, so they cannot drift far.

| Area | Supported | Not supported | Lifting it needs |
|---|---|---|---|
| Weight format | MLC q4f16_1 shards (group 32, symmetric); MLX safetensors (group 64, affine, 4-bit; 8-bit router; 3-bit expert stacks via convert-q3-experts) | f16/bf16 unquantised, GPTQ/AWQ, other group sizes | new dequant paths in int4_matmul.gen.ts |
| Attention | MHA and GQA, headDim %32 ≤256, full RoPE or partial (rotary fraction), qk-norm optional, gated attention (Qwen3.5/3.6), paged f16 KV; int8 KV for headDim ≤128 | sliding window, MLA, ALiBi, softcap, attention biases | windowed/MLA variants of attention.wgsl; bias epilogue in matmuls |
| RoPE | plain theta (any base), partial rotary factor, llama3 and yarn frequency scaling (precomputed inv_freq table; yarn also scales attention logits by mscale^2) | longrope frequency scaling | a frequency formula in model-spec.ts ropeInvFreqTable() — rope.wgsl already reads the table |
| FFN | SwiGLU dense (fused kernel for MLC, matmul+silu_mul chain for MLX-affine) | GeGLU / ReLU / non-gated FFN, FFN biases | activation-parameterised fused_ffn.wgsl + silu_mul.wgsl |
| Norm | RMSNorm with plain gamma | LayerNorm (beta), Gemma-style (1+gamma), post-norm sandwiches | variants of rms_norm/add_norm.wgsl |
| MoE | ≤256 routed experts, top-K ≤32, with OR without a shared expert (when present it stacks as index E and must equal moe_intermediate in width), router at 4-bit, 8-bit or unquantized f16, norm_topk_prob either way; subgroups required | grouped/expert-parallel routing, a stack mixing dense and MoE layers | a routing layout where experts are sharded across dispatches rather than stacked; a per-layer block kind for mixed stacks |
| Linear attention | GatedDeltaNet (Qwen3.5/3.6): GVA, headV %32 ≤256, conv width 4 | Mamba/S4, RWKV, conv hybrids (LFM2 layer_types 'conv'), other conv widths | new recurrence kernels |
| Embedding / head | quantised embedding (symmetric or affine), tied or untied lm_head, vocab %4 | unquantised embedding tables | f16 gather path |
| Tokenizer | SentencePiece (Phi-3), byte-level BPE (Qwen/Llama-style tokenizer.json) | tekken, WordPiece, custom pipelines | new pipeline beside tokenizer-bpe.ts |
| Chat template | Phi-3, ChatML (non-thinking), Llama-3 header template, DeepSeek prose turns | Gemma turns, Mistral [INST], thinking-mode rendering | renderer branch in model-select.ts — the single highest-leverage gap in the survey (docs/PORTING.md): it blocks 29 of 51 refused repos and is the SOLE blocker on 5 |
| Decoding | greedy argmax (default), seeded temperature / top-p / min-p sampling, streaming, cross-turn prefix reuse | top-k, repetition/presence penalties, beam search, batch > 1 | a rank selection pass beside sampler.wgsl's mass threshold; a per-sequence token-count buffer for penalties |

Dimension rules (all enforced by the checker): heads divisible by kvHeads;
headDim %32 and ≤256; d, qkvDim, kvDim %256; every matmul K (d, qDim, ffn,
gdnVDim) %64 — the scale-group width, since the general int4 matmul strides K
and stops on a bound rather than a trip count; vocab %4.
