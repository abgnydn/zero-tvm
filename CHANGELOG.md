# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/) starting
from `0.1.0`.

## [Unreleased]

### 2026-08-20 — qwen38's chunked prefill quarantined at 256; the chunking gate ran at six tokens

On qwen38, chunked prefill and the per-token path disagree at long context. The
contract chunked prefill is held to is empirical TOKEN identity — it has never
been bit-equal to per-token — and at ~16k that fails: correct per-token and at
cap 256, invents tool names at the shipped cap of 1024, generic greeting at
4096. Cleared at a depth that fails: the model (`mlx_lm` correct at every depth
tried), the prompt (byte-identical to the vendor jinja), int8 KV, cross-turn
reuse, retrieval. `gdn_recur`'s workgroup barriers are correct, including the
one before the staging buffers are reused. Those runs were made by hand against
a loaded station and their OUTPUTS are not committed. The harness that re-runs
the identity check at depth is `scripts/chunk-prefill-test.mjs` with an explicit
PROMPT and CAP (the invocation is recorded on the `maxChunkCap` field);
`chunk-cap-sweep.mjs` sweeps the same caps but measures tok/s and compares no
output, so it does not reproduce the correctness column.

**The mechanism is not found.** `ModelSpec.maxChunkCap` bounds the damage where
the bisection put the threshold — 256 for qwen38, the largest cap TESTED
correct, not a proven ceiling (512 was never swept for correctness at this
depth — BENCH.md has it at 931.3 tok/s in the throughput sweep — so the
threshold is somewhere in (256, 1024]; 24k is untested under the quarantine). It
clamps the DEFAULT only: an explicit cap is honoured, because the sweep that
found the threshold has to be able to cross it. `resolveChunkCap` derives that
default — a pooled MoE build lowers it again downstream — and was extracted so a
test can assert the real function rather than a copy; the mutation gate
reinstates the pre-quarantine expression.

Not free, though qwen38's own cost is UNMEASURED: the closest comparable sweep
is llama32 at 4k (844.7 vs 972.1 tok/s, ~13% between those caps) and llama32 is
not clamped. The GDN rewind ring's lookback drops from ~4k tokens to ~1k; what
that ring buys was measured on an unquarantined qwen36q3 session, so that cost
is unmeasured too. E5 survives the clamp (256 satisfies `CHUNK_CAP % 64 === 0`,
and qwen38's d/qDim/ffn/gdnVDim are all multiples of 64).

Why nothing saw it: `gdn_chunk_chain` asserted chunk-vs-stepwise bit identity at
SIX tokens against a shipped cap of 1024 and stayed green throughout. It is now
parameterized by scale with a 1024 arm — which still does NOT reach the failing
configuration: it runs qwen35 dims through `int4_matmul_batched_dyn`, while cap
1024 is reached by default only where the subgroup-matrix feature exists, which
is exactly when the engine picks E5. `skip()` also returns `pass: true`, so a
machine without 32-lane subgroups reports PASS while running nothing.

Recorded, not fixed: the chunk-GEMM `dimsOK` guard omits `gdnProjRows`, and
among the shipped hybrids qwen38 is the only one whose value is not a multiple
of 64 (16480 = 64x257 + 32; the other four are 12352 = 64x193). E5 tiles N by 32
and guards the ragged column, so this is a lead, not a diagnosis.

Also: `chunkCap`'s documented default disagreed with the code in SIX places
across four files — three inside `engine-core.ts`, plus `src/lib/index.ts`,
`CLAUDE.md` and `tests/kernels/compile-qwen35.mjs`; all six now agree.
`scripts/chunk-prefill-test.mjs` (which contradicts itself),
`scripts/gemm-bench.mjs`, `docs/PAGING_PLAN.md`, `docs.html` and
`hf-space/README.md` still say 64, and `scripts/chunk-cap-sweep.mjs` still
frames 256 as the shipped cap; none are touched. BENCH.md's cap-1024 token
identity result and its "pinned bit-exactly" note are scoped rather than
retracted, and THREE stale qwen35 suite counts are deleted rather than
incremented (`CLAUDE.md`'s and both of `BENCH.md`'s round-A ones);
`docs/PAGING_PLAN.md` still carries two and is left for follow-up.
`scripts/mutation-gate.mjs` is NOT concurrency-safe — it mutates
`src/` in place, so two runs in one worktree each see a false red baseline.

### 2026-08-19 — the agentic loop at depth: three prompt bugs, none of them a kernel

qwen36q3 through the station at ~24k tokens of history read all three files it
needed, computed the right answer, and then replied in prose instead of calling
the tool. Perfect at ~500 tokens. int8 KV was cleared first (`KV8=0` failed
identically), then cross-turn reuse and the rewind ring. The fault was in the
prompt we build, above the engine; no WGSL was involved.

Settled by asking mlx_lm the same question with the same conversation through
the checkpoint's own jinja (`scripts/toolcall-depth-ref.py`): it emits the call
at 0 / 8k / 24k where we emitted prose.

- **Tool results were sent unwrapped.** Both hosts mapped `role: 'tool'` to
  `'user'` and passed the tool's output through as the turn's whole content;
  the template wants `<tool_response>…</tool_response>`. `renderToolResults`
  had done this correctly, with tests, since it was written — and had **zero
  callers**. The model read every tool result as something the user had typed.
- **An empty `<think>` block on every past assistant turn.** No Qwen template
  does that. At 24k it was 286 spurious blocks and 5,434 characters. The real
  rule is three generations, now three `chatTemplateId` values: `chatml`
  (Qwen3 — only a trailing turn), `chatml-q35` (Qwen3.5/3.6 — the current tool
  round), `chatml-q38` (Qwen3.8 — every turn, because its condition opens
  `preserve_thinking is undefined` and nothing defines it).
- **The tool dialect was wrong for both Qwen3.5 builds**, which were served the
  Qwen3-era JSON dialect against templates that ship the XML form. It was a
  prefix match on the model's NAME, hand-written in both hosts; it is derived
  from the chat-template id now, in one place.
- **`|trim` and the `</think>` split**, found by the same method after the
  first two: Qwen3 trims nothing, 3.5+ trims every message; 3.5/3.6 split a
  `</think>` back out of assistant text and 3.8 does not. Reachable from a
  remote guest's history in a room.
- **The rewind ring was only filled by chunked prefill.** Invisible until now —
  the spurious `<think>` block had made every turn an exact token extension, so
  reuse never needed a rewind. Rendering correctly ended that, and any build
  that cannot chunk (a pooled MoE, a hybrid without subgroups, `?chunk=0`)
  would have gone from full reuse to none. Snapshots now come from the
  per-token path too.
- `detectChatTemplate` learned the three generations, so `add-model` stops
  reintroducing this — Qwen3.8 shipped with the wrong id on 2026-08-17 and was
  hand-corrected two days later.

All six ChatML specs now render byte-identical to their own templates, verified
by `scripts/render-diff.py` (which diffs our prompt against the checkpoint's
jinja) at depth 0, 8000 and 24000 and across a battery of awkward shapes. The
24k eval goes from prose to SOLVED: 3/3 files, no wasted reads.

Published copy corrected in the same pass — the int8 default, the KV figures,
split-K's availability, and the prefix-reuse claims this batch made false.

### 2026-08-17/18 — agent serving: the cache stops re-reading, int8 KV goes wide

Driven by a real Cline session on qwen36q3 that cost ~5.5 minutes per turn.
Four independent faults, none of which was "the cache doesn't work".

- **GDN rewind ring.** Hybrid prefix reuse was all-or-nothing: the recurrence
  cannot be rewound a token at a time, so ONE changed token late in a prompt
  re-read the whole conversation. Agent clients regenerate a trailing metadata
  block every turn, so this fired constantly — 16 changed tokens discarded
  43,693 good ones. Four snapshots of the recurrent state are now kept at chunk
  boundaries and a divergence replays from the nearest one: **392.50s → 12.76s**
  on the session that prompted it. Lookback is 4 × CHUNK_CAP. Allocated lazily,
  so engines that never chunk do not pay the 0.19–0.57 GB.
- **The KV disk pool had never worked.** The native OPFS shim wrote strings as
  a LENGTH, so every `meta.json` — the commit record — was zero bytes beside a
  payload of hundreds of megabytes. Restores reported "no entry" forever.
- **Cancel did nothing**, and prefill progress was measuring submissions rather
  than work: `record()` only submits, so the CPU walked 43 chunks in
  milliseconds and then blocked on one un-cancellable readback.
- **int8 KV left the fused path.** It required fused QKV, which meant Phi-3
  alone — a 4k window, the one model where the cache never binds. It now runs
  unfused, on hybrids, and through chunked prefill (new
  `attention_prefill_int8.wgsl`), **1.98× smaller cache** for a measured
  −0.09%/+0.10% perplexity, i.e. nothing. 4-bit was measured too and REJECTED:
  +0.77% at 4k windows, growing with context.
- **`kv_quantize_int8` packed only the first 32 words of a row**, so every
  head-dim-256 spec left dims 128..255 holding stale bytes — fluent wrong text,
  not noise. Latent since the kernel was written; only reachable once int8 left
  Phi-3.
- **Tool calls**: unparseable ones are no longer forwarded (they made clients
  fail the tool and retry an identical prompt — six errors then a loop),
  truncation reports `finish_reason: length`, and schema mismatches are named.
- **Station** (`npm run station`): owns model lifecycle on :8017, proxying to
  the engine on :8019, so a model swap never changes a client's URL.

TurboQuant was investigated and **killed**: its premise is a few dominant
channels for a rotation to spread, and our K/V measure 3.1× max-over-median,
not the 20× assumed. docs/TURBOQUANT_PLAN.md carries the measurements.

### Added — Qwen3.6-35B-A3B: the first MoE, the first MLX checkpoint, and a 3-bit build

- **`?model=qwen36`** — Qwen3.6-35B-A3B (30 GDN + 10 attention layers, 256
  experts top-8 + shared) on the same hand-written engine. Every layer type is
  validated against **mlx_lm's own modules** on the real checkpoint (GDN
  2.26e-3, attention 7.59e-4, MoE block 1.57e-4, whole decoder layer 5.16e-4,
  greedy argmax matches mlx). New kernels: `moe_router_logits`,
  `moe_router_topk` (32 lanes × 8 experts in registers), `moe_combine`, and
  `affine`/`moe` variants of the int4 matmul (MLX `w = s·q + b`, group 64,
  expert-as-grid-z). 19.7 GB resident — needs ~24 GB free RAM.
- **`?model=qwen36q3`** — the 32 GB-Mac build: expert stacks requantised to
  3-bit (`scripts/convert-q3-experts.py`, 16 s), resident 15.7 GB. The MLX
  bits=3 layout is a continuous LSB-first bitstream straddling u32 words; the
  `q3` kernel variants walk 8-value/24-bit windows. Chosen over 2-bit by
  measurement (block cosine 0.936 vs ~0.79 for 2-bit). ~55 t/s on a quiet 32 GB
  M2 Max. NOTE 2026-08-10: that block cosine is a FIDELITY number and cannot
  see quality — see docs/QUALITY.md. The 3-bit-vs-4-bit perplexity that would
  settle it is unrun.
- **MLX safetensors loader** — byte-range reads (a 5.3 GB shard is never one
  ArrayBuffer), per-BufferPlan OPFS caching of BUILT buffers, bf16→f16/f32
  conversion with subnormal-exact rounding and hard-error on overflow; 268 KB
  of headers maps all 19.5 GB. Repacking is byte-verified against the
  checkpoint (`npm run test:kernels:mlx`).


Correctness fixes, the first measured optimization promoted to default, and a
new headline number. The engine now measures **faster than WebLLM** on the
one machine benchmarked — and, since the 2026-07-30 corrected protocol, every
pair is quoted with **both** total wall-clock and decode-only throughput. It
is now measured against **two** baselines: WebLLM/TVM on byte-identical
q4f16_1 weights (the same-bytes A/B), and llama.cpp via wllama on GGUF (a
runtime **and** quantization comparison, never presented as same-bytes).

### Fixed

- **Benchmark protocol defect — two published A/B pairs withdrawn
  (2026-07-30).** `bench()` in `src/zero-tvm/bench-console.ts` looped its runs
  against the same prompt and never called `engine.resetKVTracking()`, while
  `benchPrefill()`, `specSim()` and `validate.ts` all did. Harmless until
  cross-turn prefix reuse shipped on 2026-07-29 (PR #24) — after that, runs
  2..N of every bench found the whole prompt already absorbed and prefilled
  exactly **one** token, while the WebLLM half (a fresh chat completion per
  run) kept paying a full prefill inside its wall clock. **Our half of every
  A/B was measuring decode-only against WebLLM's prefill + decode.**
  - *Withdrawn as not like-for-like:* Qwen3-4B **75.74 / 43.75 ("+73.1%")**
    and Qwen3.5-4B **65.67 / 34.04 ("+92.9%")**, both 2026-07-29. Everything
    earlier — the Phi-3 headline, the 2026-07-28 v1 pairs, and the Qwen3.5
    perf round's 53.07 / 32.36 ("+64.0%") — predates prefix reuse, so both
    halves genuinely paid prefill: superseded, not defective.
    Zero-TVM-vs-Zero-TVM flag ladders are unaffected (same accounting on both
    sides), so the per-item deltas (fused-qk +2.3%, vec4h +5.7% / +2.0%)
    stand.
  - *Fix:* `bench()` calls `resetKVTracking()` before **every** run; both
    halves now split TTFT from decode instead of reporting one blended rate;
    WebLLM's own `decode_tokens_per_s` / `prefill_tokens_per_s` are captured
    on every run and reduced to a median instead of being logged once on run
    1 and discarded.
  - *Prior published numbers were not overwritten* — they stay in BENCH.md as
    dated history with the defect explained in place.

### Changed

- **All three pairs re-measured under the corrected protocol (2026-07-30,
  Apple M2 Max, Chrome 150, identical local q4f16_1 weights, same-session
  interleaved pairs, 128-token target × 5 runs vs WebLLM's 3 × 120, medians).
  Both metrics are now reported everywhere — leading with total wall-clock,
  decode alongside; quoting one alone is cherry-picking.**

  | Model | ZT total | ZT TTFT | ZT decode | WebLLM total | WebLLM decode | Δ total | Δ decode |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Phi-3-mini | 69.55 | 291 ms | 83.10 | 59.95 | 63.23 | **+16.0%** | **+31.4%** |
  | Qwen3-4B | 59.85 | 453 ms | 75.49 | 45.46 | 47.77 | **+31.7%** | **+58.0%** |
  | Qwen3.5-4B | 65.28 | 171 ms | 73.30 | 32.56 | 34.32 | **+100.5%** | **+113.6%** |

  - **The advantage grows with architecture recency** (2024 Phi-3 +16/+31%,
    2025 Qwen3 +32/+58%, 2026 Qwen3.5 +100/+114%) — monotonic on both
    metrics, and the headline insight of the round. Consistent with compiler
    stacks having had less time to tune newer architectures. An observation
    across three points on one machine, **not** a proven law.
  - **Honest negative, published not buried: WebLLM has the better
    time-to-first-token on short prompts.** Its prefill runs 251 tok/s on
    Phi-3 and 263–271 on Qwen3-4B — an implied TTFT around 150 ms on Phi-3
    against our measured 291 ms, and 453 ms on Qwen3-4B. Only Qwen3.5 is a
    wash. We win sustained decode decisively and lose the first-token sprint
    on short inputs; chunked prefill is strong on long ones (202 tok/s at 816
    tokens) and prefix reuse makes follow-up turns free. Added as the top item
    on BENCH.md's levers list.
  - **Absolutes moved for both engines** since 2026-07-29 with no code change
    on the Phi-3 path (ours 62.90 → 69.55, WebLLM's 47.92 → 59.95) — the same
    machine-state variance already documented in BENCH.md's session note.
    Only same-session pairs are meaningful. The old "−28…−31% stable across
    sessions" band is **retired**: it was a total-wall-clock ratio labelled
    "decode", and this session's total ratio is −16.0%.
  - New durable artifacts: `bench/results/phi3-mini.json`,
    `bench/results/qwen3-4b.json`, `bench/results/qwen35-4b.json` — the two
    Qwen pairs had no machine-readable home because cross-engine A/B mode
    deliberately never writes `bench/results.json`. Each carries total, TTFT,
    decode, WebLLM's self-reported rates, raw runs, and a `supersedes` note.
  - Gate-dialog rate labels (`src/zero-tvm/model-select.ts`) now quote the
    corrected **totals**: Phi-3 ~70 t/s, Qwen3-4B ~60 t/s, Qwen3.5-4B ~65 t/s.
  - Documentation synced across BENCH.md, README.md, index.html, docs.html,
    `hf-space/README.md`, `bench/README.md` and `sites.json`.

### Added

- **Third benchmark baseline: llama.cpp via wllama (WebGPU) (2026-07-30).**
  `@wllama/wllama@3.5.1` (libllama b9640-dd4623a) as a second opponent
  alongside WebLLM, driven from a new `wllama-bench.html` /
  `src/wllama-bench/main.ts` page and `BENCH_BASELINE=wllama npm run bench`.
  - **NOT a same-bytes comparison, and labelled as such everywhere it
    appears.** wllama reads GGUF (`Q4_K_M` on the Qwen models, `Q4` on Phi-3)
    while Zero-TVM and WebLLM read MLC `q4f16_1`, so these pairs compare a
    **runtime and a quantization together** — unlike the WebLLM baseline,
    where both engines load byte-identical weight files. The two baselines are
    reported in separate tables and must not be merged. A true same-bytes race
    needs GGUF support in Zero-TVM (container parser + a `Q4_K_M` super-block
    dequant kernel + tensor-name mapping); **that is not done.**
  - **Measured (M2 Max, Chrome 150, same-session pairs, medians, total
    wall-clock | decode):** Phi-3 68.15 | 81.61 vs wllama 24.62 | 26.16
    (**+176.8% / +212.0%**) · Qwen3-4B 53.25 | 67.10 vs 19.22 | 21.59
    (**+177.0% / +210.8%**; +213.7% against Zero-TVM's clean 60.30 half) ·
    Qwen3.5-4B 63.59 | 71.07 vs 16.48 | 19.18 (**+285.8% / +270.5%**).
  - **The LlamaWeb result does not reproduce here, in the opposite
    direction.** arXiv 2605.20706 reports llama.cpp-WebGPU decoding 45–69%
    faster than WebLLM across 16 devices; on this machine **WebLLM is 2.1–2.4×
    faster than wllama**. Prefill is the smoking gun — llama.cpp self-reports
    79 / 62 / 42 tok/s on a 24–31-token prompt against WebLLM's 249 / 264 /
    176, which points at an immature batched-matmul path on Dawn/Metal.
  - **Published with the suspicion attached.** A ~3× win in our own favour
    against a well-regarded engine is when to be most skeptical: one device,
    one libllama build, an unresolved quantization confound, no mechanism
    isolated. **The WebLLM same-bytes baseline is not retired in favour of
    this one** — it stays primary because the weights are identical there.
  - **Unflattering finding, published: Zero-TVM is the less stable engine.**
    Running our half twice per model exposed an 11.7% disagreement between the
    two Qwen3-4B medians (60.30 vs a thermally disturbed 53.25) and downward
    drift inside single Qwen3.5 invocations. wllama held ≤2% spread — it never
    loads the GPU hard enough to throttle.
  - **A fake number caught before publication.** The first sweep read Qwen3.5
    at **0.00 tok/s ("+Infinity%")** and Qwen3 at 0 tokens on 2 of 4 runs: the
    Qwen GGUF chat templates enable thinking by default, so llama.cpp streamed
    every token as `delta.reasoning_content` while the page counted only
    `delta.content`. Also a genuine like-for-like defect — the Zero-TVM half
    runs Qwen's non-thinking template. Fixed with
    `chat_template_kwargs: { enable_thinking: false }` (inert on Phi-3), a
    `reasoningChunks` counter, a zero-token guard, and a `contentAccountingOk`
    flag surfaced through `bench/run.mjs`. All published numbers are post-fix.
    Two further traps recorded in BENCH.md: llama.cpp's `cache_prompt`
    defaults ON (set to `false`, or runs 2..N skip prefill), and Vite's dev
    server ships no COOP/COEP so wllama silently drops to single-thread WASM
    (replicated production's `same-origin` + `credentialless`, scoped to
    `/wllama-bench*` only so the other two pages keep byte-identical headers).
  - **WebGPU proven, not assumed.** `isSupportWebGPU()` is just
    `!!navigator.gpu`, so the page greps llama.cpp's own native stdout via
    `WllamaConfig.logger`. All six halves: `backend=webgpu`,
    `adapter_info: vendor: apple | architecture: metal-3`, 33/37/33 layers
    offloaded, `crossOriginIsolated: true`, 6 threads. Falsification control
    `?ngl=0` on Phi-3 drops to 8.29 total / 9.71 decode on `wasm-cpu` — 3.0×
    slower, so the GPU is doing the work.
  - New artifacts `bench/results/{phi3-mini,qwen3-4b,qwen35-4b}-wllama.json`,
    each carrying `"sameBytes": false` and a `quantizationCaveat` string.
    `bench/results.json` is **deliberately not written** under
    `BENCH_BASELINE=wllama` and `sync-docs.mjs` is skipped — that file's
    `webllmDecode` field feeds the published headline, and a GGUF number
    landing there unlabelled is exactly the confusion this baseline is
    supposed to avoid.
  - Supporting changes: `vite.config.ts` gains a `/local-gguf/*` dev mirror
    (HEAD + Range + 416) over `.weights-local/gguf/` so the run is fully
    offline, plus a `wllama-bench` build input; `bench/run.mjs` gains
    `BENCH_BASELINE` and a `runBaseline()` helper. Site pages (`index.html`,
    `docs.html`) were left alone — no claim on them is falsified by this
    round, and a site rebuild is planned separately.

- **Qwen3-4B tuning round (2026-07-29)** — the v1-unfused Qwen3 decode path
  gets its first dedicated round. **The cross-engine pair published here —
  75.74 vs WebLLM 0.2.84's 43.75 tok/s, "+73.1%" — is WITHDRAWN** (bench
  harness defect, see "Fixed" above; the corrected pair is 59.85 / 45.46
  total, +31.7% total / +58.0% decode). The per-item deltas below are
  Zero-TVM-vs-Zero-TVM against the same-day flags-off half and are unaffected
  (the 2026-07-28 25.43/14.15 pair did not reproduce — both engines moved ~3×
  together; session note in BENCH.md):
  - *Fused qk_norm+RoPE+KV-append* (`qk_norm_rope_append.wgsl`,
    `?fuseqk=0` opts out) — one 32-thread WG per (token, head) over Q, K
    and V heads: per-head RMSNorm reduction, normalized head staged in
    workgroup memory for the RoPE pair exchange, K/V written straight into
    the paged cache. 4 post-norm dispatches → 2; **10 → 8
    dispatches/layer** (364 → 292/token). +2.3% alone; ~+0.1% on top of
    vec4h (documented as non-composing).
  - *K%512 vec4 matmul variants* (`_vec4h` generator siblings, `?vec4h=0`
    opts out) — vec2<u32> weight loads halve the per-thread unroll,
    relaxing the vec4 K-divisibility gate from 1024 to 512 so Qwen3's
    d=2560 (qkv/gate_up/LM-head) and ffn=9728 (down_proj) instances get
    wide loads; per-instance resolution in `resolveMatmul`. +5.7% alone on
    Qwen3; also engages on Qwen3.5's K=2560 projections (see BENCH.md).
    4 new generated pipelines (f16/f32 × sg/tiled); generator now emits 18
    variants (55 WGSL kernels total — shader-count copy updated across the
    site/docs).
  - Kernel suite grows to **26/26** (fused-kernel test with V-region
    negative controls; _vec4h correctness at K=2560/9728 incl. f32; ±vec4h
    resolution cases). Skipped, documented: the whole-head qkv_fused
    restructure (occupancy-falsified by the existing qkv-tiling negatives)
    and the unfused int8-KV composition (memory win only; conflicts with
    the fused qk kernel).
- **Prefill round A (2026-07-29)** — chunked GDN prefill + cross-turn
  prefix reuse, measured on the same M2 Max (BENCH.md "Prefill round A"):
  - *Chunked GDN prefill (Qwen3.5, `?chunk=0` opts out)* — prompt tokens run
    in chunks of ≤64: every projection is ONE `int4_matmul_batched_dyn`
    dispatch (new generator variant: the m=4 register block looping over
    runtime-M row blocks), the conv/gates/norm run batched
    (`gdn_conv_seq` + `gdn_conv_commit`, seq+stride uniforms on
    `gdn_gates`/`gdn_norm_out`), the recurrence is ONE `gdn_recur` dispatch
    per layer per chunk, and the 8 attention layers batch through the new
    causal `attention_prefill` (bit-exact vs the decode kernel) with a
    batched FFN (`silu_mul`). 816-token prompt: **67.9 → 202.1 prefill
    tok/s** (TTFT 12.0 → 4.0 s). Chunk-vs-per-token equality is pinned
    BIT-EXACTLY (f32 recurrent state included) in the kernel suite, now
    **19/19** (6 new tests).
  - *Cross-turn prefix reuse (all models, `?reuse=0` opts out)* — the engine
    records the exact (position, token) of every submitted forward pass
    (including the pipelined stop-token overrun, reconstructed from the
    readback chain) and prefills only the delta on the next turn. Hybrid
    reuse is all-or-nothing (the GDN recurrence can't rewind); the ChatML
    non-thinking template now re-renders past assistant turns WITH the
    empty `<think>` block so each turn extends the absorbed sequence
    exactly. Turn-3 first-token latency (~950–1105-token conversations):
    Phi-3 15,405 → **269 ms**, Qwen3-4B 14,554 → **438 ms**, Qwen3.5
    14,340 → **194 ms**. Reused-prefix logits verified **bit-identical**
    to a fresh prefill on all three models (`checkReuse()`, opt-in).
  - New devtools harnesses `benchPrefill` / `benchTurns` / `checkReuse`
    (bench-console), engine API `debugCompareReuse` / `getLastPrefill`,
    and 2 new multi-turn e2e tests (13/13).
- **Qwen3.5 hybrid perf round (2026-07-29)** — two engine changes, measured
  **53.07** vs WebLLM 0.2.84's **32.36** tok/s on the same-session M2 Max
  pair (**+64.0%**, up from the v1 floor's 47.99 vs 31.99 / +50.0%). This
  round landed *before* cross-turn prefix reuse, so both halves paid a full
  prefill and the pair was like-for-like — **superseded, not defective**, by
  the 2026-07-30 corrected pair (65.28 / 32.56 total, +100.5% total /
  +113.6% decode). The same-day "65.67 vs 34.04, +92.9%" cross-check
  published in the Qwen3 tuning round *is* withdrawn; see "Fixed" above.
  - *Fused GDN input projection* — the four per-DeltaNet-layer in_proj
    matmuls (qkv 8192 + z 4096 + a 32 + b 32 rows, all K=d) pack into ONE
    12352-row int4 dispatch. The loader concatenates the q4f16_1 records at
    upload; downstream kernels read the packed output in place (`gdn_conv`
    at offset 0, `gdn_norm_out`'s z region and `gdn_gates`' [a|b] pair via
    256-aligned bind-group offsets). 412 → **340 dispatches/token**.
  - *Incremental blocking decode* — the blocking `generate()` no longer
    replays the whole prompt: the engine tracks the GDN state position
    (`gdnStatePos`) and reuses the non-idempotent recurrent state whenever
    it provably matches `startPos` (falling back to a full replay — which
    re-zeroes state — otherwise). After `forwardLogits`, the validate
    battery's decode now runs zero prompt passes; its reported rate went
    ~17 → ~40 tok/s. `generatePipelined` (chat/bench) was already
    incremental; its ≤1-token overrun past a stop is documented as safe.
  All suites re-verified: kernels 28/28 + 21/21 + 13/13 (gdn_gates/gdn_block
  updated to the packed layout, CPU reference `ref-gdn.mjs` unchanged),
  287 unit, e2e 11/11, validate battery lexically correct.
- **Qwen3.5-4B hybrid port (v1, `?model=qwen35`)** — the first *hybrid*
  architecture on the engine: 24 gated-DeltaNet (linear-attention) layers +
  8 gated full-attention layers (GQA 16/4, head_dim 256, partial RoPE 64 of
  256, sigmoid attention gate), 248k vocab with renumbered specials, tied
  lm_head. To our knowledge the first hand-written-kernel int4 gated-DeltaNet
  hybrid in a browser. The path is **v1-scalar-GDN** (scalar DeltaNet
  kernels, no chunked prefill — prompts replay token-by-token, unfused GDN
  projections). Suites: `npm run test:kernels:qwen35` (13/13 GDN kernel
  family vs CPU reference) + mirror-gated `tests/e2e/qwen35.test.ts`.
- **Qwen3.5 same-weights A/B vs WebLLM** — `BENCH_QUERY="?model=qwen35"
  npm run bench` runs both engines back-to-back in one session on the same
  local weight bytes. Measured pair (2026-07-28, M2 Max, Chrome 150):
  Zero-TVM **47.99** vs WebLLM **31.99** tok/s (+50.0%), WebLLM via its own
  prebuilt Qwen3.5-4B lib. Recorded in BENCH.md with the v1-scalar-GDN
  floor caveat; `bench/results.json` untouched. A `?splitk=0` vs default
  spot-check on the hybrid's 8 attention layers is recorded there too.
- **Qwen3-4B port (v1, `?model=qwen3`)** — the spec-parameterized engine now
  runs a second architecture end-to-end in the browser: GQA 32/8 with
  qDim ≠ d, per-head QK-norm, byte-level BPE tokenizer, tied lm_head, ChatML
  (non-thinking) template. `zero-tvm.html?model=qwen3` and
  `validate.html?model=qwen3`; Phi-3 stays the default and all existing URLs
  keep their exact behavior. The Qwen path is **v1-unfused** (QK-norm rules
  out the fused QKV kernel → 10 dispatches/layer; vec4 loads only where
  K % 1024 == 0; no int8-KV). Suites: `npm run test:kernels:qwen` (21/21
  compile-and-shape gate) + a mirror-gated e2e file (`tests/e2e/qwen.test.ts`).
- **Qwen3 same-weights A/B vs WebLLM** — `BENCH_QUERY="?model=qwen3"
  npm run bench` now runs BOTH engines back-to-back in one session against
  the same local weight bytes (WebLLM via its own prebuilt Qwen3-4B wasm and
  a per-model mirror route) and prints both medians + the gap, without
  touching `bench/results.json` (that stays the Phi-3 headline artifact).
  First measured pair (2026-07-28, M2 Max, Chrome 150): Zero-TVM **25.43**
  vs WebLLM **14.15** tok/s (+79.8%) — but both engines run Qwen3-4B far
  below their Phi-3 rates on this machine, so the gap reflects WebLLM's
  prebuilt Qwen3 lib as much as our v1 port. Recorded in BENCH.md as the
  baseline for the Qwen tuning phase, not promoted to any headline.

### Fixed

- **fused_ffn f32 accumulation** — the fused FFN now accumulates in f32
  throughout instead of f16.
- **Attention workgroup-barrier bug** — a missing barrier in the attention
  kernel.
- **Decode off-by-one** — the decode loop was off by one token position.

### Changed

- **`@mlc-ai/web-llm` dev-dependency bumped `^0.2.80` → `0.2.84`** — the
  Qwen3.5 hybrid model libs first ship in WebLLM's v0_2_84 prebuilt set.
  `webllm-bench` now uses the v0_2_84 lib names for all three models
  (upstream dropped the `ctx4k_` segment from the wasm names and renamed
  the `useIndexedDBCache` AppConfig flag to `cacheBackend`). The recorded
  Phi-3 / Qwen3-4B pair numbers in BENCH.md were measured against the
  v0_2_80-era libs and stand as dated history.
- **vec4 loads are now the default** (`?vec4=0` / `?vec4qkv=0` to opt out).
  Measured 2026-07-25 on Apple M2 Max vs the same-day pre-vec4 baseline of
  60.96 tok/s: `?vec4=1` +4.5%, `?vec4qkv=1` +4.2%, both together **+7.1%**
  (65.27 tok/s). Full A/B table in BENCH.md.
- **New headline head-to-head** (2026-07-25, Apple M2 Max, Chrome
  150.0.7871.182, identical local Phi-3-mini q4f16_1 weights, 128 tokens ×
  5 runs, median): Zero-TVM **66.33 tok/s** vs WebLLM v0.2.80 **51.98 tok/s**
  — **~28% faster**. Supersedes the "22% behind" M2 Pro numbers (different
  machine AND a since-fixed engine — the delta is not all optimization; the
  same-run WebLLM figure is the valid comparator). Best measured opt-in
  config `?vec4=1&vec4qkv=1&splitk=8`: 68.36 tok/s. Old M2 Pro numbers kept
  in BENCH.md "Prior measurements".
- README / BENCH.md / site copy rewritten around the measured result; the
  split-K attention flag (`?splitk=N`, ~+3% at short context) stays opt-in
  pending a long-context A/B.

### Falsified (measured and not shipped)

- **FFN prologue fusion** (`?fuseprologue=1`) — folding the FFN-entry
  add_norm into the FFN kernel's shared-memory phase measured **−13.7%**
  (52.62 tok/s) on M2 Max: the redundant per-workgroup RMSNorm recompute
  costs far more than the −32 dispatch bubbles save. Flag + shaders kept
  for A/B on other GPUs, documented as a negative result in BENCH.md —
  same treatment as the tiling and spec-decode negatives.

## [0.2.0] — 2026-06-25

Tooling, tests, and docs pass. Engine behavior is unchanged; this release adds
what makes it auditable, reproducible, and citable.

### Added

- **Headless WebGPU kernel-correctness suite** (`npm run test:kernels`) — runs
  the real WGSL kernels against independent CPU references, 8 of the 10 roles,
  via the Dawn-native binding on Mesa lavapipe (no GPU; CI-ready).
- **Auto-bench + doc-sync** (`npm run bench`) — drives both engines on a WebGPU
  GPU, writes `bench/results.json`, propagates numbers into the docs. Ships a
  Docker image + a Colab notebook for cloud GPUs.
- **Citable DOI scaffolding** — `CITATION.cff` + `.zenodo.json`.

### Changed

- README leads with the comparison table and the "~80% of WebLLM's decode
  speed" framing; refreshed the hero screenshot to the current site.
- Engine requests its WebGPU adapter with `powerPreference: 'high-performance'`.
- Migrated hosting from Vercel to Cloudflare Pages (`public/_headers` carries
  the COOP/COEP cross-origin-isolation headers).

### Fixed

- CI was red — the workflow called the renamed-away `lint` script; restored as
  an alias to `typecheck`.
- Corrected CLAUDE.md's architecture map (kernels live in
  `src/compiler/shaders/`, not `src/shaders/`).

### Removed

- 1,078 lines of provably-dead code (5 superseded prototype modules).

## [0.1.0] — 2026-05-04

First public release. **Phi-3-mini running in a browser on hand-written
WGSL shaders. No TVM. No WebLLM runtime. No compiler.**

### Headline numbers (vs WebLLM/TVM, identical model + weights)

|                                                        | WebLLM (TVM)        | Zero-TVM            |
| ------------------------------------------------------ | ------------------- | ------------------- |
| Unique WGSL kernels                                    | **85**              | **10 roles / 27 files** |
| Total WGSL lines                                       | **12,962** (gen)    | **3,078** (hand)    |
| Dispatches per decode step                             | **342**             | **228** (f16 KV)    |
| Runtime                                                | TVM → WASM scheduler | Plain TypeScript    |
| JS bundle (chat page, excl. weights)                   | 5.9 MB / 2.1 MB gz  | **157 kB / 33 kB gz** |

Zero-TVM issues **fewer** dispatches than TVM by fusing operations TVM's
default pipeline doesn't (qkv-fused incl. RoPE + paged-KV append; paged
attention + page-table read combined).

### Added

- **10 kernel roles** (27 WGSL files, counting subgroup / tiled / int8
  variants): qkv-fused, paged attention, FFN-fused, RMSNorm, RoPE,
  int4-dequant matmul, argmax sampling, and supporting kernels.
- **~2,000 lines of TypeScript** for the engine, tokenizer (BPE from
  scratch), and weight loader (direct HuggingFace fetch + OPFS cache).
- **Live demo** at https://zerotvm.com — chat UI with live throughput,
  zero-runtime architecture inspectable end-to-end.
- **Bench harness** ([BENCH.md](./BENCH.md)) — head-to-head vs WebLLM on
  identical hardware / weights.

### Companion projects

- [neuropulse.live](https://neuropulse.live) — same Phi-3-mini weights,
  every intermediate tensor rendered live as the model thinks.
- [webgpu-fusion-max](https://github.com/abgnydn/webgpu-fusion-max) — the
  "how far does it scale?" sister project pushing single-dispatch fusion
  to the same model size.
- [kernelfusion.dev](https://kernelfusion.dev) — research umbrella.

[0.2.0]: https://github.com/abgnydn/zero-tvm/releases/tag/v0.2.0
[0.1.0]: https://github.com/abgnydn/zero-tvm/releases/tag/v0.1.0
