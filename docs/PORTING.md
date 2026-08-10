# Porting speed — what this repository can actually claim

Measured 2026-08-10. Every input is a `git log` entry, a HuggingFace API
response, or a `scripts/add-model.mjs --check-only` run reproducible from the
commands in this file. Numbers that came out worse than hoped are kept, in the
style BENCH.md set.

## The claim under test

> This project's differentiator is not tokens/sec — it is **how fast a newly
> released architecture becomes runnable in a browser**, and it is the only
> project that can put a number on that.

**The mechanism is real. The headline is not.**

"Days from release to running" is, for every model here, dominated by when the
author started looking at it rather than by how long the port took. On the one
model where a head-to-head is possible — Qwen3.5-4B — the WebLLM/TVM stack
shipped browser support **62 days before** this engine did.

What survives is narrower and still worth saying:

1. For a genuinely new block layout, this engine has a path and the TVM stack
   currently does not. Qwen3.6-35B-A3B has run here since 2026-08-05; as of
   2026-08-10 there is no MLC conversion of it at any size.
2. Once an architecture is *covered*, adding a checkpoint is minutes. The
   Llama-3.2-1B port is 24 minutes of commits and 129 lines, most of them
   machine-written.
3. The RED report's granularity — a named rule and a named kernel file per
   failure — is unusual. The existence of a refusal is not: llama.cpp's
   `convert_hf_to_gguf.py` logs `Model {arch} is not supported`, and MLC keeps
   an explicit model registry. What is different here is that the refusal names
   the shader to change, not the architecture that was missing.

---

## 1. What actually happened, per model

Two clocks matter and they disagree.

- **release → runs** — calendar days from the upstream model's public release
  to the commit where it generated correct text here.
- **work span** — from the first commit that mentions the model to the commit
  that says it runs. Work done before the first commit is invisible to git, so
  this is a **lower bound**, not a measurement of effort.

The repository's first commit is **2026-04-01**. Any model older than that was
added late by choice, and its release→runs figure says nothing about porting
speed. Those rows are marked *(started late)* and must not be quoted as
porting-speed evidence.

| model | released | checkpoint consumed (HF `createdAt`) | first commit | "it runs" commit | release → runs | work span |
|---|---|---|---|---|---|---|
| Phi-3-mini-4k-instruct | 2024-04-23 | `mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC` 2024-05-08 | `9ae59ce` 2026-04-01 18:21 | `12818ec` 2026-04-03 22:26 | 710 d *(started late — this was the founding project)* | 2 d 4 h |
| Llama-3.2-1B-Instruct | 2024-09-25 | `mlx-community/Llama-3.2-1B-Instruct-4bit` 2024-09-25 | `cd4bf95` 2026-08-06 15:52 | `b1f75a5` 2026-08-06 16:16 | 680 d *(started late)* | **24 m** |
| Qwen3-4B | 2025-04-29 | `mlc-ai/Qwen3-4B-q4f16_1-MLC` 2025-05-01 | `ab13fc3` 2026-07-28 14:31 | `d3ed34b` 2026-07-28 15:54 | 455 d *(started late)* | 1 h 23 m |
| Qwen3-30B-A3B | 2025-04-29 | `mlx-community/Qwen3-30B-A3B-4bit` 2025-04-28 | — (single commit) | `6fcda67` 2026-08-08 07:12 | 466 d *(started late)* | not measurable |
| Qwen3.5-4B (GDN hybrid) | 2026-03-02 | `mlc-ai/Qwen3.5-4B-q4f16_1-MLC` 2026-04-22 | `de0cfc3` 2026-07-28 19:41 | `1adcc56` 2026-07-28 20:13 | **148 d** | 32 m |
| Qwen3.6-35B-A3B (MoE + GDN) | 2026-04-16 | `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` 2026-04-16 | `24eea55` 2026-08-04 20:40 | `65f5d48` 2026-08-05 16:41 | **111 d** | ~9 h 40 m over two sessions |
| DeepSeek-V2-Lite (MLA) | 2024-05-16 | `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx` | `3029dbe` 2026-08-09 16:46 | **not running** | — | ~17 h of commits so far |

Notes that keep these numbers honest:

- **Only two rows can carry the claim.** Qwen3.5-4B and Qwen3.6-35B-A3B are the
  only models released after this project began. Both took three to four
  months, not days.
- **Three models arrived via someone else's conversion.** Phi-3, Qwen3-4B and
  Qwen3.5-4B are loaded from `mlc-ai/*-q4f16_1-MLC` repos. For Qwen3.5-4B that
  conversion did not exist until **2026-04-22**, so nothing before that date
  was even possible on the shipped spec.
- **`Qwen3-30B-A3B`'s work span is not measurable.** It landed as one commit at
  07:12 on 2026-08-08, but its generated spec block is stamped 2026-08-07, so
  the pipeline ran the previous day and the engine work (a 4-bit router
  entry point in `moe_router_logits.wgsl`, a no-shared-expert path) followed.
  414 insertions across 12 files — this was *not* a mechanical add.
- **`Llama-3.2-1B`'s 24 minutes are real but not free.** The two commits
  immediately before it (`cd4bf95` llama3 RoPE frequency table, `ded5eca`
  Llama-3 header template + tokenizer pipeline) are prerequisites built in the
  same hour. The generated part — `LLAMA_3_2_1B_INSTRUCT_4BIT` in
  `model-spec.ts`, the registry rows — is genuinely machine-written and carries
  the `generated by scripts/add-model.mjs` stamp.
- **DeepSeek MLA is in progress and is counted as such.** Three MLA kernels
  exist (`mla_proj`, `mla_scores`, `mla_combine`) and are checked against real
  DeepSeek weights, but `checkModel()` still refuses MLA unconditionally and no
  MLA spec is registered. It does not run.

## 2. The comparison the claim implies

Reproduce with `https://huggingface.co/api/models/<repo>` (`createdAt`) and
`https://registry.npmjs.org/@mlc-ai/web-llm` (`time`).

| model | released | MLC conversion exists | days | this engine | days |
|---|---|---|---|---|---|
| Phi-3-mini | 2024-04-23 | 2024-05-08 | 15 | 2026-04-03 | 710 |
| Llama-3.2-1B | 2024-09-25 | 2024-09-25 | **0** | 2026-08-06 | 680 |
| Qwen3-4B / 30B-A3B | 2025-04-29 | 2025-05-01 | **2** | 2026-07-28 / 08-08 | 455 / 466 |
| Qwen3.5-0.8B / 2B | 2026-03-02 | 2026-03-29 | 27 | — | — |
| Qwen3.5-4B / 9B | 2026-03-02 | 2026-04-22 | 51 | 2026-07-28 | 148 |
| Qwen3.6 (any size) | 2026-04-16 | **none** | 116+ | 2026-08-05 | **111** |

And the browser runtime, not just the weights:

- `@mlc-ai/web-llm` **0.2.84** — the first release carrying Qwen3.5 prebuilt
  libs, per this repo's own CLAUDE.md — was published **2026-05-27**, 86 days
  after the Qwen3.5-4B weights. This engine landed Qwen3.5 on 2026-07-28, day
  148. **WebLLM was 62 days faster.** That is the honest headline for Qwen3.5
  and it goes the other way.
- 0.2.84 is still the latest release. A HuggingFace search for `author:mlc-ai`
  and for the `mlc-llm` tag returns **zero** Qwen3.6 repos. BENCH.md states the
  same independently ("WebLLM ships zero Qwen3.6 builds"). So for Qwen3.6 the
  claim is not "faster" — it is **"at all"**, and that is a real difference.

Not checked, and therefore not claimed: whether llama.cpp/GGUF or
`@wllama/wllama` (latest 3.5.1, 2026-06-15) can run Qwen3.6. "No browser stack
runs it" is a stronger statement than the evidence gathered here supports; what
was verified is only that the MLC/WebLLM path does not.

## 3. Method for the scoreboard

```bash
node scripts/add-model.mjs <hf-repo> --check-only
```

The checker fetches `config.json`, `tokenizer_config.json`, `tokenizer.json`,
`model.safetensors.index.json` and each shard's header, normalises them into
`DetectedModel`, and runs `checkModel()` from `src/compiler/constraints.ts`. No
checkpoint is downloaded and no GPU is touched (`--check-only` exits before the
compile gate, which needs a device).

Three outcomes; only two are evidence.

- **GREEN** — every rule passes. Running `add-model` without `--check-only`
  would then generate a registered spec and gate it.
- **RED** — at least one rule fired; each is recorded by rule name.
- **probe failed / did not complete** — transport failure or timeout. **Not a
  result about the model.** The script enforces this itself: only a genuine 404
  is interpreted, everything else aborts rather than becoming a false negative.
  Two probes landed here on the first pass (`gemma-3-1b-it-4bit`,
  `gemma-3-4b-it-qat-4bit`, killed mid-write when the batch was restarted) and
  were re-run to completion; the final table has no unknowns.

**GREEN is the weakest of three evidence levels**, and the scoreboard below
only establishes the first:

1. `--check-only` GREEN — the constraint matrix accepts the checkpoint.
2. full `add-model` — plus every WGSL kernel compiles under the new dims
   (`tests/kernels/compile-spec.mjs`, needs a GPU).
3. `scripts/validate-model.mjs` — logits and greedy decode diffed against
   `mlx_lm` on the same checkpoint (needs the weights and a GPU).

Of the shipped models only `qwen3mlx` (cosine 0.999879), `qwen30b`
(0.999985 in f32) and `llama32` carry level-3 evidence in the registry.

**Selection bias, stated up front.** Repositories were chosen for family
breadth and download rank within `mlx-community`, preferring 4-bit MLX repacks
because that is the format the loader reads. This is a survey of "what someone
browsing mlx-community would try", not an unbiased sample of open-weight
models.

## 4. The scoreboard

**62 repositories probed on 2026-08-10 · 11 GREEN · 51 RED · 0 unknown.**

### GREEN fraction: 11/62 = **18%**

Every GREEN falls into exactly two architecture families — `llama` and
`qwen3`/`qwen3_moe`:

| repo | family |
|---|---|
| `mlx-community/Llama-3.2-1B-Instruct-4bit` | llama |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | llama |
| `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` | llama |
| `mlx-community/MiniCPM5-1B-4bit` | llama (MiniCPM5 ships a Llama-shaped config) |
| `mlx-community/Qwen3-0.6B-4bit` | qwen3 |
| `mlx-community/Qwen3-4B-4bit` | qwen3 |
| `mlx-community/Qwen3-4B-Instruct-2507-4bit` | qwen3 |
| `mlx-community/Qwen3-8B-4bit` | qwen3 |
| `mlx-community/Qwen3-14B-4bit` | qwen3 |
| `mlx-community/Qwen3-30B-A3B-4bit` | qwen3_moe |
| `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` | qwen3_moe |

That is the honest shape of the result: **the engine generalises across sizes
within a family it has already ported, and across essentially nothing else.**
Three of the eleven are Llama-3.x, seven are Qwen3 or Qwen3-MoE, and the
eleventh is MiniCPM5-1B — a Llama config wearing a different name. Nothing from
Mistral, Gemma, Phi, SmolLM, InternLM, GLM, Granite, OLMo, DeepSeek, Falcon,
EXAONE, Nemotron, LFM2, Kimi, Hunyuan, Ling, ERNIE or Seed passed.

### Full results

**Qwen**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` | qwen3_5_moe_text * | **RED** | bias: linear-layer biases; config: unread field(s) |
| `Qwen2.5-7B-Instruct-4bit` | qwen2 | **RED** | bias: linear-layer biases |
| `Qwen2.5-Coder-7B-Instruct-4bit` | qwen2 | **RED** | bias: linear-layer biases |
| `Qwen3-0.6B-4bit` | qwen3 | **GREEN** | — |
| `Qwen3-14B-4bit` | qwen3 | **GREEN** | — |
| `Qwen3-30B-A3B-4bit` | qwen3_moe | **GREEN** | — |
| `Qwen3-30B-A3B-Instruct-2507-4bit` | qwen3_moe | **GREEN** | — |
| `Qwen3-4B-4bit` | qwen3 | **GREEN** | — |
| `Qwen3-4B-Instruct-2507-4bit` | qwen3 | **GREEN** | — |
| `Qwen3-8B-4bit` | qwen3 | **GREEN** | — |
| `Qwen3-Coder-30B-A3B-Instruct-4bit` | qwen3_moe | **RED** | config: unread field(s) |
| `Qwen3.5-4B-MLX-4bit` | qwen3_5_text * | **RED** | bias: linear-layer biases; config: unread field(s) |
| `Qwen3.5-9B-MLX-4bit` | qwen3_5_text * | **RED** | bias: linear-layer biases; config: unread field(s) |
| `Qwen3.6-27B-4bit` | qwen3_5_text * | **RED** | bias: linear-layer biases; config: unread field(s) |
| `Qwen3.6-35B-A3B-4bit` | qwen3_5_moe_text * | **RED** | bias: linear-layer biases; config: unread field(s) |

**Llama**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `Falcon3-7B-Instruct-4bit` | llama | **RED** | template: not phi3/chatml/llama3/deepseek |
| `Llama-3.2-1B-Instruct-4bit` | llama | **GREEN** | — |
| `Llama-3.2-3B-Instruct-4bit` | llama | **GREEN** | — |
| `Meta-Llama-3.1-8B-Instruct-4bit` | llama | **GREEN** | — |
| `MiniCPM5-1B-4bit` | llama | **GREEN** | — |
| `SmolLM-135M-Instruct-4bit` | llama | **RED** | dims: d 576 % 256 != 0 (×3) |
| `SmolLM2-1.7B-Instruct` | llama | **RED** | quant: unquantized (bf16); config: unread field(s) |
| `SmolLM3-3B-4bit` | smollm3 | **RED** | config: unread field(s) |

**Mistral**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `Ministral-8B-Instruct-2410-4bit` | mistral | **RED** | template: not phi3/chatml/llama3/deepseek |
| `Mistral-7B-Instruct-v0.3-4bit` | mistral | **RED** | template: not phi3/chatml/llama3/deepseek |
| `Mistral-Nemo-Instruct-2407-4bit` | mistral | **RED** | template: not phi3/chatml/llama3/deepseek |
| `Mistral-Small-3.1-24B-Instruct-2503-4bit` | mistral * | **RED** | index: no safetensors index; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `Mistral-Small-4-119B-2603-4bit` | mistral4 * | **RED** | config: unread field(s); attention: MLA; template: not phi3/chatml/llama3/deepseek |

**Gemma**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `gemma-3-1b-it-4bit` | gemma3_text | **RED** | activation: not SiLU; attention: sliding window; dims: d 1152 % 256 != 0; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `gemma-3-4b-it-4bit` | gemma3_text * | **RED** | index: no safetensors index; rope: linear; attention: sliding window; config: unread field(s) (×2); template: not phi3/chatml/llama3/deepseek |
| `gemma-3-4b-it-qat-4bit` | gemma3_text * | **RED** | index: no safetensors index; activation: not SiLU; rope: linear; attention: sliding window; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `gemma-4-31b-it-4bit` | gemma4_text * | **RED** | activation: not SiLU; attention: sliding window; blocks: sliding_attention; config: unread field(s); template: not phi3/chatml/llama3/deepseek |

**Phi**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `Phi-3-mini-4k-instruct-4bit` | phi3 | **RED** | attention: sliding window; config: unread field(s) |
| `Phi-3.5-mini-instruct-4bit` | phi3 | **RED** | rope: longrope; attention: sliding window; config: unread field(s) |
| `phi-4-4bit` | phi3 | **RED** | config: unread field(s) |
| `Phi-4-mini-instruct-4bit` | phi3 | **RED** | rope: longrope; attention: sliding window; config: unread field(s); template: not phi3/chatml/llama3/deepseek |

**DeepSeek / MLA**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx` | deepseek_v2 | **RED** | attention: MLA; moe: mixed dense/MoE stack |
| `DeepSeek-R1-Distill-Qwen-7B-4bit` | qwen2 | **RED** | bias: linear-layer biases; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `DeepSeek-V2-Lite-Chat-4bit-mlx` | deepseek_v2 | **RED** | attention: MLA; moe: mixed dense/MoE stack |
| `DeepSeek-V4-Flash-4bit` | deepseek_v4 | **RED** | quant: 4-bit group 32 (×129); attention: sliding window; dims: headDim 512 > 256; config: unread field(s); moe: non-softmax routing; template: not phi3/chatml/llama3/deepseek |

**GLM**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `GLM-4.5-Air-4bit` | glm4_moe | **RED** | bias: linear-layer biases; config: unread field(s); moe: mixed dense/MoE stack; template: not phi3/chatml/llama3/deepseek |
| `GLM-4.7-Flash-4bit` | glm4_moe_lite | **RED** | dims: headDim 102.4 % 32 != 0; config: unread field(s); attention: MLA; moe: mixed dense/MoE stack; template: not phi3/chatml/llama3/deepseek |
| `GLM-5.2-4bit` | glm_moe_dsa | **RED** | bias: linear-layer biases; config: unread field(s); attention: MLA; moe: mixed dense/MoE stack (×2); template: not phi3/chatml/llama3/deepseek |

**Granite**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `granite-3.3-2b-instruct-4bit` | granite | **RED** | dims: vocab 49159 % 4 != 0; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `granite-4.0-h-tiny-4bit` | granitemoehybrid | **RED** | quant: 8-bit group 64 (×40); bias: linear-layer biases; blocks: mamba; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `granite-4.1-30b-4bit` | granite | **RED** | quant: 4-bit group 32; config: unread field(s); template: not phi3/chatml/llama3/deepseek |

**OLMo**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `OLMo-2-1124-7B-Instruct-4bit` | olmo2 | **RED** | template: not phi3/chatml/llama3/deepseek |
| `Olmo-3-7B-Instruct-4bit` | olmo3 | **RED** | attention: sliding window; blocks: sliding_attention; dims: vocab 100278 % 4 != 0 |
| `OLMoE-1B-7B-0125-Instruct-4bit` | olmoe | **RED** | config: unread field(s) (×2); template: not phi3/chatml/llama3/deepseek |

**Other**

| repo (`mlx-community/` unless noted) | family | verdict | rules that fired |
|---|---|---|---|
| `ERNIE-4.5-21B-A3B-PT-4bit` | ernie4_5_moe | **RED** | config: unread field(s); tokenizer; template: not phi3/chatml/llama3/deepseek |
| `exaone-4.0-1.2b-4bit` | exaone4 | **RED** | config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `EXAONE-4.0-32B-4bit` | exaone4 | **RED** | attention: sliding window; config: unread field(s); template: not phi3/chatml/llama3/deepseek |
| `Falcon-H1-1.5B-Instruct-4bit` | falcon_h1 | **RED** | bias: linear-layer biases; dims: vocab 65537 % 4 != 0; config: unread field(s) |
| `Hunyuan-A13B-Instruct-4bit` | hunyuan | **RED** | quant: 16-bit group 0 (×193); rope: dynamic; config: unread field(s) (×2); tokenizer; template: not phi3/chatml/llama3/deepseek |
| `internlm3-8b-instruct-4bit` | internlm3 | **RED** | rope: dynamic; config: unread field(s); tokenizer |
| `Kimi-Linear-48B-A3B-Instruct-4bit` | kimi_linear | **RED** | dims: headDim 72 % 32 != 0; config: unread field(s); attention: MLA; moe: mixed dense/MoE stack; tokenizer; template: not phi3/chatml/llama3/deepseek |
| `LFM2.5-2.6B-4bit` | lfm2 * | **RED** | blocks: conv; config: unread field(s) |
| `Ling-mini-2.0-4bit` | bailing_moe | **RED** | config: unread field(s); moe: mixed dense/MoE stack; template: not phi3/chatml/llama3/deepseek |
| `MiniCPM4-8B-4bit` | minicpm | **RED** | rope: longrope; config: unread field(s) |
| `Moonlight-16B-A3B-Instruct-4-bit` | deepseek_v3 | **RED** | config: unread field(s); attention: MLA; moe: mixed dense/MoE stack (×2); tokenizer; template: not phi3/chatml/llama3/deepseek |
| `NVIDIA-Nemotron-3-Nano-4B-4bit` | nemotron_h | **RED** | bias: linear-layer biases; dims: d 3136 % 256 != 0; config: unread field(s) |
| `Seed-OSS-36B-Instruct-4bit` | seed_oss | **RED** | rope: default; bias: linear-layer biases; config: unread field(s); template: not phi3/chatml/llama3/deepseek |

`*` = dims read from a multimodal `text_config` / `language_model` root.

## 5. Which single missing piece unlocks the most

Two counts, because they answer different questions.

| rule | REDs it appears in | REDs where it is the ONLY blocker |
|---|---:|---:|
| `config` (fields the checker does not read) | 40 | 3 |
| `template` (chat template renderer) | 29 | **5** |
| `attention` (sliding window, MLA) | 17 | 0 |
| `bias` (additive bias epilogue in the matmuls) | 14 | 2 |
| `moe` (mixed dense/MoE stack, non-softmax routing) | 9 | 0 |
| `dims` (%256 / %32 / %4 divisibility) | 9 | 1 |
| `rope` (longrope, dynamic, linear) | 8 | 0 |
| `quant` (group 32, 8-bit non-router, unquantized) | 5 | 0 |
| `tokenizer` (neither SPM nor byte-level BPE) | 5 | 0 |
| `blocks` (`mamba`, `conv`, `sliding_attention`) | 4 | 0 |
| `index` (no safetensors index in the repo) | 3 | 0 |
| `activation` (GeGLU) | 3 | 0 |

**The single highest-leverage item is not a kernel.** It is the chat-template
renderer branch in `model-select.ts`: 29 of 51 REDs, and 5 of them turn GREEN
on that alone (Mistral ×3, OLMo-2, Falcon3 — all of which are otherwise
Llama-shaped models that the engine could already run). It is a string
formatter, not a shader.

**Among actual kernels, the additive-bias epilogue in the matmuls wins.** It
fires on 14 REDs, is the sole blocker on 2 (Qwen2.5-7B, Qwen2.5-Coder-7B), and
is one of exactly two blockers on the entire Qwen3.5 / Qwen3.6 MLX line — the
family this engine already runs. `config` is a review gate rather than
engineering work (each key is either dismissed after reading or turns out to be
real architecture); treating it as free, `bias` and `template` each unlock 7
more models, and no other rule unlocks more than 1.

> ### CORRECTION, 2026-08-10 — the `bias` row above is wrong
>
> **`bias` fires on 14 REDs. Five of them are false, and four more are
> mislabelled. The real number is 5, and the paragraph above should not be
> quoted.**
>
> The rule counted **every** `*.bias` record in the checkpoint, including
> towers this engine never loads. The tell was sitting in the table the whole
> time: `Qwen3.6-35B-A3B-MLX-4bit` is listed RED on `bias`, and that model
> **ships here and runs**. Its language model has *zero* bias records; all 166
> live under `vision_tower.*`.
>
> Probed directly from each repo's `model.safetensors.index.json`:
>
> | repo | `.bias` records | in the text tower |
> |---|---:|---:|
> | `Qwen3.5-4B-MLX-4bit` | 148 | **0** (all `vision_tower.*`) |
> | `Qwen3.5-9B-MLX-4bit` | 166 | **0** |
> | `Qwen3.6-27B-4bit` | 166 | **0** |
> | `Qwen3.6-35B-A3B-4bit` | 166 | **0** |
> | `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` | 166 | **0** |
> | `Qwen2.5-7B-Instruct-4bit` | 84 | 84 — `self_attn.{q,k,v}_proj.bias` |
> | `Qwen2.5-Coder-7B-Instruct-4bit` | 84 | 84 — same |
> | `DeepSeek-R1-Distill-Qwen-7B-4bit` | 84 | 84 — same (Qwen2 base) |
> | `GLM-4.5-Air-4bit` | 138 | 138 — `self_attn.k_proj.bias` |
> | `Seed-OSS-36B-Instruct-4bit` | 192 | 192 — `self_attn.k_proj.bias` |
> | `granite-4.0-h-tiny-4bit` | 36 | 36 — `mamba.conv1d.bias` |
> | `Falcon-H1-1.5B-Instruct-4bit` | 24 | 24 — `mamba.conv1d.bias` |
> | `NVIDIA-Nemotron-3-Nano-4B-4bit` | 21 | 21 — `mixer.conv1d.bias` |
> | `GLM-5.2-4bit` | 21 | 21 — `self_attn.indexer.k_norm.bias` |
>
> So: **5 false positives** (multimodal, marked `*` in the tables above — the
> asterisk was the clue and I did not follow it), **5 genuine** matmul biases,
> and **4 that are real records but not this kernel** — three mamba `conv1d`
> biases and one norm bias, in models already RED on `blocks: mamba` and on
> MLA respectively. Building the epilogue does not move those four.
>
> **The corrected ranking puts `bias` below `attention` rather than at the top
> of the kernels**, and the `bias + config + template` pair table above is
> overstated by roughly the same five. `template` — a string formatter, not a
> shader — was already the highest-leverage item and is now clear of the field.
>
> Both defects are fixed: `add-model.mjs` counts only text-tower records, and
> the failure now **names** the records that triggered it, so `conv1d.bias`
> can no longer read as "linear-layer biases". The rest of this document was
> generated by the uncorrected checker; treat every `bias` cell in the tables
> above as suspect and the other rules as unaffected.

Pairs, for planning:

| lift | REDs that go fully GREEN |
|---|---:|
| `config` + `template` | 10 |
| `bias` + `config` | 10 |
| `bias` + `config` + `template` | **18** |

**MLA is the strategic one, not the numerous one.** It blocks 7 repos here —
DeepSeek-V2-Lite, DeepSeek-Coder-V2-Lite, Moonlight-16B-A3B, GLM-4.7-Flash,
GLM-5.2, Mistral-Small-4-119B, Kimi-Linear — and every one of them is a 2026-era
frontier open-weight release. But **MLA alone unlocks nothing**: six of the
seven also need a mixed dense/MoE stack and five need a template branch. The
in-progress MLA work is a bet on where the field is going, not a way to move
this scoreboard.

The other 10 `attention` failures are sliding window — the whole Gemma line,
`Phi-3-mini-4k`, `Phi-3.5-mini`, `Phi-4-mini`, `Olmo-3-7B`, `EXAONE-4.0-32B`,
`DeepSeek-V4-Flash`. That is a different kernel from MLA, and it is the one
standing between the engine and the MLX repack of its own default model.

## 6. Honest reading

**What the evidence supports.**

- The pipeline is real and mechanical for covered architectures. Llama-3.2-1B
  went from prerequisite to registered spec in 24 minutes and 129 lines.
- The RED reports are specific and actionable: rule, detail, and the shader
  file to change. No comparable per-rule report was found in the two other
  stacks checked (llama.cpp's converter, MLC's registry), though this was not
  an exhaustive survey.
- On a genuinely new block layout (GDN + 256-expert MoE), this engine reached
  runnable while the TVM stack has not, 116 days on.

**What it does not support.**

- "Fastest from release to running." The two measurable ports took 111 and 148
  days; MLC converts a covered architecture in 0–2 days and beat this engine to
  Qwen3.5 by 62 days.
- "Broad architecture coverage." 18% of a survey deliberately weighted toward
  families the engine already runs. Outside Llama-3.x and Qwen3, the GREEN
  count is zero.
- "The only project that can put a number on it." Both llama.cpp's converter
  and MLC's model registry refuse unknown architectures by name. The
  *granularity* here is better; the *capability* is not unique.

**The claim that does survive, stated so it can be checked:** for an
architecture whose blocks the kernel set already covers, this repository turns
an MLX checkpoint into a registered, compile-gated spec in minutes; for one it
does not cover, it produces a per-rule report naming the missing kernel instead
of a silent failure. Both halves are demonstrated above. Neither is a speed
record.

## 7. Where the evidence is thinner than the claim

1. **The engine's own flagship checkpoints are RED.**
   `lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit` — the exact repo behind
   `?model=qwen36` — fails on `bias` and `config`. So does
   `mlx-community/Phi-3-mini-4k-instruct-4bit`, the MLX repack of the default
   model, on `attention` (sliding window) and `config`. Both models demonstrably
   run, via hand-written specs that predate those rules. **The checker's GREEN
   set and the engine's actual reach are different sets, in both directions**,
   so the 18% is a property of the checker, not a measurement of the engine.

2. **The verdict is a property of the repository, not the architecture.**
   `gemma-3-4b-it-4bit` and `gemma-3-4b-it-qat-4bit` are the same model and
   produce *different* rule sets — the QAT repack additionally trips
   `activation`. `SmolLM2-1.7B-Instruct` is RED only because that particular
   repo ships bf16; a 4-bit repack would be judged on other grounds entirely.
   Reading the scoreboard as "architecture X is/isn't supported" over-reads it.

3. **The probe is 50–100× larger than documented.** README and CLAUDE.md say "a
   few hundred KB of ranged reads". Measured from the HF file tree:
   Llama-3.2-1B-4bit **16.9 MB**, Qwen3-4B-4bit **11.2 MB**,
   Qwen3.6-35B-A3B-MLX-4bit **19.8 MB** — almost all of it `tokenizer.json`,
   which `add-model.mjs` fetches whole. Still ~0.1% of a checkpoint, but the
   stated figure is wrong.

4. **`SUPPORT_MATRIX` has drifted from `checkModel()`**, and `docs/COMPAT.md` is
   generated from the stale half. Three rows:
   - RoPE lists **yarn** under "Not supported"; `checkModel` has accepted yarn
     since `b6d7da4` (2026-08-09).
   - Chat template omits **DeepSeek**; accepted since `4a064c5` (2026-08-09).
   - MoE says "8-bit router"; 4-bit and unquantized f16 routers are accepted
     since `6fcda67` / `f8e6a4d`.
   `constraints.ts` says the only way the docs can drift is within its own ~300
   lines. They drifted there.

5. **The RED report degrades badly on unfamiliar names.** `DeepSeek-V4-Flash`
   emits **129** near-identical `[quant]` lines, `Hunyuan-A13B` **193**, and
   `granite-4.0-h-tiny` **40** — the last because Granite spells its router
   `block_sparse_moe.router.layer`, which `ROUTER_PATH` does not match, so a
   legitimately-8-bit router is reported forty times as a violation. "Names the
   exact missing kernel" holds for the first line, not for the report.

6. **One reported reason is an artifact.** `GLM-4.7-Flash` is refused partly for
   `headDim 102.4 % 32 != 0` — a fractional head dim from the `d / heads`
   fallback on an MLA model that has no `head_dim` field. The verdict is right
   (MLA is refused two lines later); that line is not a real finding.

7. **Work spans are lower bounds.** They measure the interval between commits,
   not effort. The first commit of each port is large and contains work of
   unknown duration. `Qwen3-30B-A3B` has no measurable span at all.

8. **Release dates are secondary sources.** Phi-3 2024-04-23, Llama-3.2
   2024-09-25, Qwen3 2025-04-29, Qwen3.5 series 2026-02-16 (4B weights
   2026-03-02), Qwen3.6-35B-A3B 2026-04-16 — from vendor blogs and press via
   web search, cross-checked against HF `createdAt` where the two are within a
   few days. Only `createdAt` was verified directly.

9. **No claim is made about non-MLC browser stacks.** Whether wllama or
   transformers.js can run Qwen3.6 was not tested.

## Reproducing

```bash
# any single verdict (network only, no GPU, no checkpoint)
node scripts/add-model.mjs mlx-community/Qwen3-8B-4bit --check-only

# the support matrix the checks are documented from
npm run add-model -- --compat        # regenerates docs/COMPAT.md

# the timeline
git log --format='%h|%ad|%s' --date=format:'%Y-%m-%d %H:%M' --reverse

# the comparison
curl -s https://huggingface.co/api/models/mlc-ai/Qwen3.5-4B-q4f16_1-MLC | jq .createdAt
curl -s https://registry.npmjs.org/@mlc-ai/web-llm | jq '.time["0.2.84"]'
```
