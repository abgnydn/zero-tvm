# Zero-TVM decode benchmarks (Phi-3 headline + Qwen3-4B v1)

All numbers are decode-only (prefill excluded via warmup runs), measured with
`npm run bench` (128 decode tokens × 5 runs, median) or `bench(128, 3)` in
devtools. Newest measurements first; prior-generation numbers are preserved
under "Prior measurements" — nothing is retconned out.

## Headline — head-to-head vs WebLLM (2026-07-25, Apple M2 Max)

Same machine, same run, same Phi-3-mini-4k-instruct-q4f16_1 weight bytes
(served from `/local-weights/*`), same prompt, greedy decoding. Google Chrome
150.0.7871.182. Protocol: `npm run bench` — 128 decode tokens × 5 runs,
median. Artifact of the final run: `bench/results.json`. Only the inference
engine differs.

| Engine                         | decode tok/s (median) |
|-------------------------------:|----------------------:|
| **Zero-TVM (this repo, defaults)** | **<!--bench:zt-->62.90<!--/bench:zt-->** |
| WebLLM 0.2.80 (`tensor-cache`) |   **<!--bench:webllm-->47.9<!--/bench:webllm-->** |

**Zero-TVM is ~28% faster than WebLLM on decode** on this machine with our
own weights (auto-synced gap — Zero-TVM's deficit as a share of WebLLM's
throughput, negative = ahead: <!--bench:gap-->-31<!--/bench:gap-->%).
Zero-TVM runs (latest artifact, 2026-07-27): 62.79, 63.11, 63.14, 62.90, 62.88 tok/s.

Absolute numbers are machine-state-dependent — across three same-run pairs
taken over two days the medians were 66.33/51.98 (−28%), 62.90/47.92 (−31%),
and one clearly-degraded pair (62.63/41.51, taken immediately after ~10
back-to-back GPU bench sessions) that depressed both engines and is not used
for any claim. **The ratio is the stable quantity: −28% to −31% in healthy
sessions.** Every quoted pair is same-run, same-weights.

Scope caveats, stated up front: one machine (M2 Max), one model
(Phi-3-mini q4f16_1), one browser (Chrome 150), decode-only, short-context
bench. The defaults now include the vec4-load kernels measured below and,
since 2026-07-27, split-K attention (long-context A/B below).

**This supersedes the old "22% behind" story** (42.14 vs ~51.5 tok/s on an
M2 Pro, 2026-06). That comparison was a different machine AND an older
engine — three correctness bugs have been fixed since (fused_ffn f32
accumulation, an attention workgroup-barrier bug, a decode off-by-one), plus
the unified engine and the vec4 defaults. The +24 tok/s delta over the old
number is NOT all optimization; the valid comparator for each Zero-TVM
median is the WebLLM figure from the same run (66.33 ↔ 51.98; 62.90 ↔ 47.92).
The old numbers are preserved in "Prior measurements" below.

Entry point: `webllm-bench.html` + `src/webllm-bench/main.ts` — wires a custom
`AppConfig` with `model: ${origin}/local-weights/resolve/main/` and the
published Phi-3-mini WASM `model_lib`. The vite middleware aliases
`tensor-cache.json` ← `ndarray-cache.json` (WebLLM renamed it in v0.2.80) and
strips the HF-style `resolve/main/` prefix.

## Qwen3-4B (2026-07-28, Apple M2 Max) — v1 port, same-weights A/B vs WebLLM

First cross-engine measurement of the Qwen3-4B port (`?model=qwen3`). Same
protocol shape as the Phi-3 headline: same machine, same session (halves run
back-to-back), same Qwen3-4B-q4f16_1 weight bytes served from the local
mirror (`/local-weights/Qwen3-4B-q4f16_1-MLC/`), greedy decoding, Chrome
150.0.7871.187. Zero-TVM half: `BENCH_QUERY="?model=qwen3" npm run bench`,
128 decode tokens × 5 runs, median. WebLLM half: `webllm-bench.html?model=qwen3`
— WebLLM v0.2.80's own prebuilt `Qwen3-4B-q4f16_1-ctx4k_cs1k-webgpu.wasm`
model_lib, its fixed protocol (3 × 120-token completions + warmup, median,
wall-clock — identical accounting to the Phi-3 headline's WebLLM half).
`bench/results.json` is untouched — it remains the Phi-3 headline artifact.

| Engine (Qwen3-4B q4f16_1)                | decode tok/s (median) |
|-----------------------------------------:|----------------------:|
| **Zero-TVM (`?model=qwen3`, defaults)**   | **25.43** |
| WebLLM 0.2.80 (prebuilt Qwen3-4B lib)     | **14.15** |

Raw runs — Zero-TVM: 25.53, 25.43, 25.40, 33.12, 17.18 (two outliers in
both directions; the median is the stable statistic — a separate session
earlier the same day measured 25.34). WebLLM: 14.62, 12.81, 14.15; WebLLM's
own reported `decode_tokens_per_s` was 15.57 (the wall-clock figure includes
its ~0.5 s prefill, same as every number in this file).

Zero-TVM measures **+79.8% over WebLLM on this pair** — but do NOT read this
as "the Qwen port is tuned". The caveats, up front:

- **The Qwen path is v1-unfused.** QK-norm must run between the QKV matmul
  and RoPE, which is incompatible with the fused QKV+RoPE+KV-append kernel —
  so Qwen runs the unfused reference composition, 10 dispatches/layer vs the
  Phi-3 chat path's 7. The vec4-load matmuls engage only where K % 1024 == 0
  (o_proj's K=4096 yes; d=2560 and ffn=9728 no). No int8-KV. The gap vs
  WebLLM is measured on an untuned port.
- **Both engines run Qwen3-4B far below their Phi-3 rates on this machine**
  (Zero-TVM 62.9 → 25.4; WebLLM 47.9 → 14.2). WebLLM's prebuilt Qwen3 lib
  appears at least as untuned for this GPU as our v1 path, so the +80% says
  as much about that lib as about our kernels. The Phi-3 headline above is
  the tuned-vs-tuned comparison; this one is untuned-vs-untuned.
- **One machine, one pair, short-context, decode-only.** Same scope limits
  as everything else in this file, with even less replication (a single
  same-session pair plus one corroborating Zero-TVM-only session).

This is the recorded baseline for the Qwen tuning phase (fused-path work,
vec4 builds for K≡512 (mod 1024) shapes, int8-KV), not a headline claim.

## Measured 2026-07-25 (M2 Max) — flag A/B series

Each cell is the median of 5 × 128-token runs, versus the same-day
pre-vec4-default baseline of **60.96 tok/s** (old defaults). Run-to-run
medians varied 60.5–61.8 across boots, so treat **±1 tok/s as noise**.

| Config                                | tok/s | Δ vs 60.96 | Verdict |
|---------------------------------------|------:|-----------:|---------|
| old defaults (baseline)                | 60.96 |          — | — |
| `?vec4=1` (oProj/ffnDown/LM-head)      | 63.73 |      +4.5% | win |
| `?vec4qkv=1` (qkv_fused vec4 sibling)  | 63.55 |      +4.2% | win |
| both vec4 flags                        | 65.27 |      +7.1% | **promoted to default** |
| `?splitk=4`                            | 62.70 |      ~+3%  | **promoted to default** (long-context A/B below) |
| `?splitk=8`                            | 62.82 |      ~+3%  | **promoted to default** (long-context A/B below) |
| `?fuseprologue=1`                      | 52.62 |     −13.7% | **falsified on Apple** |
| `?vec4=1&vec4qkv=1&splitk=8`           | 68.36 |       +12% | best measured config |

Outcomes:

1. **Vectorized loads — promoted to default.** Both vec4 experiments won
   individually and compose (+7.1% together). They are now the default path
   where the sg32 builds exist; opt OUT with `?vec4=0` / `?vec4qkv=0` for
   A/B. The headline pairs run with these defaults on (plus split-K after
   its 2026-07-27 promotion).

2. **Split-K attention — promoted to default (N=8) after the long-context
   A/B.** The short-context result was ~+3%; the promotion gate was "does
   the win grow with KV depth, as the occupancy hypothesis predicts?" It
   does. 1024-token decode (KV growing to ~1k), median of 3 × 1024 on the
   same M2 Max, 2026-07-27:

   | Config (1024-token decode) | tok/s | Δ |
   |---|---:|---:|
   | `?splitk=0` (off)          | 60.21 |   — |
   | `?splitk=4`                | 62.27 | +3.4% |
   | `?splitk=8`                | 62.62 | **+4.0%** |

   +3.1% at 128 tokens → +4.0% at 1024, tight variance, no measured regime
   where it loses. Default is now `splitk=8` where the sg32 path exists;
   `?splitk=0` opts out, `?splitk=N` re-tunes. The best measured short-run
   config stacks all promoted flags: 68.36 tok/s (+12% over the old
   defaults, ~+32% vs the same-session WebLLM 51.98).

3. **FFN prologue fusion — falsified on Apple.** −13.7%. The hypothesis was
   that removing 32 dispatch bubbles (addNorm1 folded into the FFN's
   shared-memory phase) would beat the cost of every FFN workgroup
   redundantly recomputing the residual add + RMSNorm. Measurement says the
   opposite: the redundant per-WG RMSNorm recompute costs far more than the
   saved dispatch bubbles. Same treatment as the tiling and spec-decode
   negatives: flag + shaders kept compiled for A/B on other GPUs
   (`?fuseprologue=1`), documented as a negative result, not shipped.

## Prior measurements (M2 Pro, 19-core — historical)

Everything below this line was measured on an **M2 Pro with the pre-2026-07
engine** (`bench(128, 3)` in devtools, 120–128 tokens). Three correctness
bugs have been fixed since — fused_ffn f32 accumulation, an attention
workgroup-barrier bug, and a decode off-by-one — so the old Zero-TVM numbers
**understate the current engine** even on that hardware. The per-kernel
percentages and the negative results remain valid as relative findings on
Apple GPUs; the absolute tok/s figures are of historical interest only.

### FFN tiling milestone (M2 Pro)

| Config                                   | tok/s (median) | ms/token | GPU compute ms |
|-----------------------------------------:|---------------:|---------:|---------------:|
| scalar FFN (baseline)                    |          22.01 |     45.5 |           37.7 |
| **tiled+subgroup FFN**                   |      **42.14** | **23.8** |       **20.2** |

**End-to-end: ~1.91×** from porting the 4-row tiled subgroup strategy to
`fused_ffn.wgsl` (the kernel that was 67.7% of GPU time).

Correctness: A/B tested with `?sgffn=0` URL toggle. Same prompt produces
**bit-identical greedy output** between tiled and scalar FFN paths on a 28-token
completion.

### Head-to-head vs WebLLM (M2 Pro, 2026-06 — superseded)

| Engine                        | decode tok/s | end-to-end tok/s |
|------------------------------:|-------------:|-----------------:|
| WebLLM 0.2.80 (`tensor-cache`) |    **~51.5** |            48.22 |
| Zero-TVM (pre-fix engine)     |    **42.14** |                — |

This was the origin of the "22% behind WebLLM" headline. Superseded by the
2026-07-25 M2 Max head-to-head above (different machine AND fixed engine —
do not read the two Zero-TVM numbers as a pure optimization delta).

### Profile — tiled+subgroup FFN (M2 Pro)

```
bench(128, 3)
  runs:   42.63, 42.07, 42.14 tok/s
  median: 42.14
  mean:   42.28
  min:    42.07
  max:    42.63
```

#### Per-kernel GPU profile (timestamp-query, single instrumented step)

Total GPU compute: **20.19 ms/token**

| Kernel        | calls | ms/token |   %   |
|---------------|------:|---------:|------:|
| fusedFfn      |    32 |     8.91 | 44.2% |
| qkvFused      |    32 |     3.93 | 19.5% |
| ffnDown       |    32 |     3.28 | 16.2% |
| oproj         |    32 |     1.57 |  7.8% |
| addNorm1      |    32 |     1.05 |  5.2% |
| addNorm2      |    32 |     0.79 |  3.9% |
| attention     |    32 |     0.33 |  1.6% |
| lmHead        |     1 |     0.33 |  1.6% |
| embedding     |     1 |     0.00 |   ~0% |
| rmsNorm_init  |     1 |     0.00 |   ~0% |
| argmax        |     1 |     0.00 |   ~0% |

CPU/readback gap: 23.8 − 20.2 = **3.6 ms/token** (~15%). Pipelining is hiding
essentially all submit/readback cost — further end-to-end wins must come from
reducing GPU compute.

### Profile — scalar FFN (M2 Pro)

Total GPU compute: **37.68 ms/token**

| Kernel        | calls | ms/token |   %   |
|---------------|------:|---------:|------:|
| fusedFfn      |    32 |    25.49 | 67.7% |
| qkvFused      |    32 |     4.46 | 11.8% |
| ffnDown       |    32 |     4.00 | 10.6% |
| oproj         |    32 |     1.31 |  3.5% |
| addNorm1      |    32 |     0.79 |  2.1% |
| attention     |    32 |     0.66 |  1.7% |
| addNorm2      |    32 |     0.33 |  0.9% |
| lmHead        |     1 |     0.33 |  0.9% |
| embedding     |     1 |     0.26 |  0.7% |
| argmax        |     1 |     0.07 |  0.2% |

### What changed (FFN tiling)

`src/compiler/shaders/fused_ffn_tiled_sg.wgsl` — new kernel:
- 4 output rows per workgroup (was 1)
- 32 threads = one subgroup; `subgroupAdd` for the 32→1 reduction
- f32 accumulation (scalar kernel used f16 with f32 upcast only before SiLU)
- Input vector (3072 f16) loaded into workgroup shared mem once; all 8 weight
  rows (4 gate + 4 up) reuse the cached input
- 2048 WGs instead of 8192 → 4× less WG-launch overhead

Bind-group layout and uniform are unchanged, so `chat.ts` swaps the pipeline
reference without rebuilding bind groups. Feature-gated behind the `subgroups`
WebGPU feature + a runtime probe that checks `sg_size == 32`; falls back to the
scalar kernel otherwise.

### Where the remaining compute went (M2 Pro era)

After the FFN win, **qkvFused + ffnDown = 7.2 ms = 35.7%** of GPU time and were
the dominant matmuls. Both were still using the scalar reduction form.

### Negative result: 8-row matmul tile (tiled8)

Ported the same tiled+sg idea to the int4 matmul with `ROWS_PER_WG = 8` in
the `int4_matmul_tiled8` and `int4_matmul_f32_tiled8` variants (now emitted by
`src/compiler/shaders/int4_matmul.gen.ts`). Math says it
should halve input-vector DRAM traffic for ffnDown and lmHead. Measurement
disagrees:

| Kernel        | tiled4 ms | tiled8 ms |
|---------------|----------:|----------:|
| qkvFused      |      3.93 |      4.13 |
| ffnDown       |      3.28 |      3.47 |
| oproj         |      1.57 |      1.57 |

End-to-end: **42.14 → 40.25 tok/s** (~5% regression). 8 f32 accumulators ×
register use likely dropped occupancy enough to outweigh the saved DRAM reads.

Shaders kept compiled; default reverted to `tiled` (4 rows). `?matmul=tiled8`
re-enables for comparison on other GPUs.

### Negative result: tiled qkv_fused (input cache + multi-pair tile)

Ported the FFN winning pattern (shared-mem input cache + multi-row tile +
32-thread single-subgroup reduction) to `qkv_fused_sg` in
`qkv_fused_tiled_sg.wgsl`. Tried two tile sizes; both regressed:

| qkv variant            | WGs   | acc/thread | qkvFused ms | tok/s |
|------------------------|------:|-----------:|------------:|------:|
| `qkv_fused_sg` (baseline) | 4608 |          2 |        3.93 | 42.14 |
| tiled_sg, 4 pairs/WG   | 1152  |          8 |        6.16 | 35.82 |
| tiled_sg, 2 pairs/WG   | 2304  |          4 |        7.01 | 33.90 |

Both tile sizes are worse than `_sg`; 2 pairs/WG was counter-intuitively the
worst. The likely reasons the FFN template doesn't transfer:

- FFN scalar baseline fired 8192 WGs; tiled dropped to 2048, still far above
  the occupancy sweet spot. qkv `_sg` already fires 4608 WGs with 64 threads
  each — halving to 2304 with 32 threads drops total active warps below what
  Apple needs to hide int4 dequant+fma latency.
- `_sg` uses 2 subgroups per WG (64 threads) which provides better ILP inside
  a single WG than the 32-thread tiled version.
- qkv is more bandwidth-bound than FFN per row (2 rows per RoPE pair vs 2
  rows for gate+up), so saving input reads matters less proportionally.

Shader kept compiled; off by default (opt-in via `?qkvtile=1`). Default
remains `qkv=_sg`.

### Negative result: tiled2sg qkv (64 threads, 2 pairs, 2304 WGs)

Follow-up to the 32-thread tiled attempts: preserved 64 threads/WG (2 subgroups,
each subgroup owning one pair of rows), shared-mem input cache, 2 pairs/WG.
Hypothesis: the earlier regressions were driven by thread-count collapse (64→32)
rather than WG-count collapse (4608→2304). Falsified:

| qkv variant                 | WGs   | threads | end-to-end tok/s | Δ vs `_sg` |
|-----------------------------|------:|--------:|-----------------:|-----------:|
| `qkv_fused_sg` (baseline)   |  4608 |      64 |            40.23 |          — |
| `qkv_fused_tiled_sg` 4-pair |  1152 |      32 |            35.82 |       −11% |
| `qkv_fused_tiled_sg` 2-pair |  2304 |      32 |            33.90 |       −16% |
| `qkv_fused_tiled2sg` 2-pair |  2304 |      64 |            36.18 |       −10% |

Keeping 64 threads helps vs the 32-thread 2-pair variant (36.18 vs 33.90), but
halving WG count (4608 → 2304) still costs ~10%. The occupancy floor for this
dispatch on M2 Pro sits at 4608 WGs. Input caching can't recover the loss
because each hidden element is already read exactly once per WG in the `_sg`
variant (no per-WG redundancy to amortize).

Shader kept compiled; off by default (opt-in via `?qkvtile2=1`). The three
qkv-tile negative results together close off hand-tuned qkv-matmul tiling as a
lever on this hardware. (The `?vec4qkv=1` win above is a different lever —
load width, not tile shape.)

### Falsifiability result: batched M-dim matmul caps at ~2× on Apple

Before porting `fused_ffn`, `qkv_fused`, `attention`, `argmax` to M-dim batched
variants (to enable fast prefill + prompt-lookup spec decoding), we tested the
weight-reuse claim in isolation. The `int4_matmul_batched_m4` variant (now
emitted by `int4_matmul.gen.ts`) computes
`[M=4, K] × [K, N] → [M=4, N]` with TILE_M=4 × ROWS_PER_WG=4 — loads each
weight row once and reuses it across 4 batch rows.

Bench (`await benchBatched(4, 500, target)`, 300-500 iters, warmup included,
output readback + checksum to confirm real work):

| target  | K    | batched ms/pack | tiled×4 ms | speedup |
|---------|-----:|----------------:|-----------:|--------:|
| ffnDown | 8192 |          0.180  |     0.363  |  2.00×  |
| o_proj  | 3072 |          0.070  |     0.154  |  2.19×  |

**Apple ceiling is ~2.0-2.2×, not the 4× hoped for.** Speedup is essentially
K-independent: ffnDown (K=8192) and o_proj (K=3072) land within 0.2× of each
other, which rules out "it's just bandwidth" as the only bottleneck.

Why not 4×: at M=1 the per-dispatch work is already bounded by weight DRAM
traffic (~12 MB weight bytes vs ~16 KB input vector). For 4 sequential tiled
calls, the repeated input reads coalesce in Apple's L1/L2, so tiled×4 isn't
paying 4× the input bandwidth. The batched shader's win is mainly from
amortizing WG-launch overhead + some L2-evicted weight re-reads.

### Attempted shared-mem variant (K=3072, M=4) — hit WebGPU 16 KB cap

Tried `int4_matmul_batched_sm_m4.wgsl` — M=4 × K=3072 × 2 = 24 KB shared mem
— to see if input-cache amortization could push the K=3072 case closer to 4×.
Initial bench reported 22× speedup, which was obviously impossible.

Added output readback + checksum to the bench: **SM variant wrote all zeros**.
Apple reports `maxComputeWorkgroupStorageSize = 16384` bytes via WebGPU
(half the 32 KB Metal native limit). The shader silently returned no-op work
and the bench reported fantasy numbers. Shader deleted.

Lesson: every micro-bench for a new shader must verify non-zero output against
a known-good baseline before reporting numbers.

### Implications for the RESEARCH.md batched-forward plan

- Prefill: realistic **~2×** end-to-end, not 14×. On a 100-token prompt,
  ~2.4 s drops to ~1.2 s.
- Decode via prompt-lookup spec: **~1.3-1.7×** (factors in 2× kernel win ×
  40-60% accept on repeat-heavy prompts). 42 → ~55-70 tok/s realistic.

The 2× compute win is real and stable. The batched-forward plan still has
payoff, but 4 more shader ports + engine plumbing for a 2× ceiling is a
smaller bet than the plan originally promised. Worth pursuing if the decode
tok/s or prefill latency numbers become a product constraint; otherwise
prompt-lookup spec decoding on the current forward gets ~60% of the upside
for ~20% of the engineering cost.

Shader kept compiled as a primitive. Not dispatched in the default forward.

### Negative result: prompt-lookup speculative decoding

**Ruled out** by CPU-only acceptance simulation on three prompt types
(prose, code, list summary). Measured with `specSim(160, N, K)` over actual
Phi-3 greedy generations:

| prompt  | hit@N=3 | α@N=3,K=3 | speedup | hit@N=2 | α@N=2,K=2 | speedup |
|---------|--------:|----------:|--------:|--------:|----------:|--------:|
| prose   |    2.5% |        0% |   0.50× |    8.5% |      1.3% |   0.68× |
| code    |   11.6% |      3.2% |   0.55× |   27.3% |      7.6% |   0.77× |
| summary |    1.3% |      0.2% |   0.50× |    7.6% |      1.0% |   0.68× |

Acceptance stays under 8% in the best case — nowhere near the 50–67% floor
needed to overcome batched-forward's 2× cost. With speedup = 2·(1+α·K)/(K+1),
α=0 bottoms out at 2/(K+1), so PLD is a **guaranteed net regression** on
these workloads regardless of K tuning. Phi-3 greedy prose doesn't repeat
n-grams from prompt or itself often enough for prompt-lookup to work.

Simulator: `src/zero-tvm/spec-sim.ts`. Window helper: `await specSim(160, N, K)`.

### Remaining levers (updated after the 2026-07-25 measurements)

1. **Split-K attention** — ~~long-context A/B~~ done 2026-07-27: +4.0% at
   1024-token decode, promoted to default (N=8). Remaining: an int8-KV
   split-K variant (`?kv8=1` currently ignores splitk) and per-GPU N tuning.
2. **Fuse addNorm into the matmul that feeds it** (addNorm1 ← oproj output,
   addNorm2 ← ffnDown output) — still open, but the prologue-fusion negative
   above says the fold must not duplicate the norm work per workgroup;
   a matmul-tail reduction is the only shape left with a chance.
3. **Batched-forward for prefill** — uses the already-landed M=4 primitive
   at measured 2× ceiling. Drops 100-token prompt from ~2.4 s to ~1.2 s.
   Independent of decode tok/s.
4. ~~Prompt-lookup speculative decoding~~ — ruled out (see above).
5. ~~Subgroup-tile qkvFused~~ — ruled out by 3 negative results above.
6. ~~FFN prologue fusion~~ — ruled out on Apple (2026-07-25, −13.7%).

## How to reproduce

```bash
npm run bench   # both engines, identical local weights, 128 tok × 5 runs;
                # writes bench/results.json and syncs the marked numbers
```

```js
// Or in devtools on zero-tvm.html after the badge flips to Ready:
await bench(128, 3)            // end-to-end tok/s, 3 runs + 1 warmup
await bench(0, 0, true)        // per-kernel profile (requires timestamp-query)
```

URL toggles for A/B bisection:

- `?sg=0`        — disable all subgroup shaders
- `?sgqkv=0`     — disable _sg qkv only
- `?sgattn=0`    — disable _sg attention only
- `?sgargmax=0`  — disable _sg argmax only
- `?sgffn=0`     — disable tiled_sg FFN (use scalar fused_ffn)
- `?matmul=scalar|sg|tiled` — force matmul variant
- `?vec4=0` / `?vec4qkv=0` — disable the vec4-load defaults (measured +7.1%
  together on M2 Max; default ON since 2026-07-25)

Opt-in experiments (measured 2026-07-25 on M2 Max — see the A/B table above):

- `?splitk=N`       — split-K attention + combine (f16 KV only): +3.1% at
  128-token / +4.0% at 1024-token decode — **default N=8 since 2026-07-27**;
  `?splitk=0` to disable
- `?fuseprologue=1` — add_norm folded into the FFN prologue: −13.7% on Apple
  (falsified there; kept for A/B on other GPUs)
