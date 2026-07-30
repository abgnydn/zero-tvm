# Zero-TVM benchmarks (Phi-3 headline + Qwen3-4B / Qwen3.5-4B)

Since the 2026-07-30 corrected protocol every pair reports **two** numbers:
**total** wall-clock throughput (prefill + decode — the conservative figure,
with both engines doing identical work) and **decode-only** (the kernel-level
figure). Both are always quoted together; quoting one alone is cherry-picking.
Measured with `npm run bench` (128-token target × 5 runs, median) or
`bench(128, 3)` in devtools. Newest measurements first; earlier numbers stay
exactly as they were recorded, marked superseded with the reason — nothing is
retconned out.

Sections dated before 2026-07-30 label their figures "decode". Read those as
**total wall-clock** — except the Zero-TVM halves of the two withdrawn
2026-07-29 pairs, which really were decode-only, which is exactly the defect.
The corrected-protocol section immediately below has the details.

## Corrected protocol (2026-07-30, Apple M2 Max) — all three pairs re-measured

### The defect: our half of the A/B stopped paying prefill

`bench()` in `src/zero-tvm/bench-console.ts` looped its runs against the same
prompt and never called `engine.resetKVTracking()` — while `benchPrefill()`,
`specSim()` and `validate.ts` all did. That was harmless until **cross-turn
prefix reuse shipped on 2026-07-29 (PR #24)**. After it, runs 2..N of every
bench found the whole prompt already absorbed and prefilled exactly **one**
token. The WebLLM half never changed: it issues a fresh chat completion per
run, so it kept paying a full prefill inside its wall clock.

From that commit onward the two halves were therefore measuring different
work — ours decode-only, WebLLM's prefill + decode. Exactly two published
pairs were measured after it, and **both are withdrawn as comparisons**:

- **Qwen3-4B 75.74 / 43.75, "+73.1%"** (tuning round, 2026-07-29)
- **Qwen3.5-4B 65.67 / 34.04, "+92.9%"** (same-day cross-check, 2026-07-29)

Everything earlier — the Phi-3 headline pair, the 2026-07-28 Qwen3 (25.43 /
14.15) and Qwen3.5 (47.99 / 31.99) v1 pairs, and the Qwen3.5 hybrid perf
round's 53.07 / 32.36 ("+64.0%") — predates prefix reuse. Both halves
genuinely paid a full prefill in those, so they were like-for-like; they are
superseded by the re-measurement below, not defective. The
Zero-TVM-vs-Zero-TVM flag ladders from 2026-07-29 also stand: both sides of
each of those A/Bs share the same accounting, so the *deltas* are unaffected
even though the absolute values are decode-only.

Worth recording as a near-miss: the tuning-round session note below already
reconciled 71.57 (post-reuse, flags-off) against 56.02 (pre-reuse control)
*by pointing out that the old code re-prefilled inside every run's wall clock
and the new code did not*. The prefill-accounting change was spotted; what
was missed is that it applied to only one of the two engines in the A/B.

### The fix (this branch)

- `bench()` calls `engine.resetKVTracking()` before **every** run, so both
  halves do a full prefill on every run.
- Both halves now split **TTFT** from **decode** instead of reporting one
  blended rate. Ours: `ttftMs` is prefill plus the first decode step,
  `decodeTokPerS` is the remaining `n−1` tokens.
- WebLLM's own `decode_tokens_per_s` / `prefill_tokens_per_s` are captured on
  **every** run and reduced to a median, instead of being logged once on run 1
  and thrown away.

### Protocol

Apple M2 Max, Chrome 150, identical local q4f16_1 weight bytes served from
`/local-weights/*`, greedy decoding, same-session interleaved pairs (the two
halves run back-to-back in one session, never across sessions). Zero-TVM half:
128-token target × 5 runs + warmup, median. WebLLM half: its fixed protocol,
3 × 120-token completions + warmup, median. `bench/results.json` is the
artifact for the Phi-3 pair; the two Qwen pairs are recorded in
`bench/results/` (see "Where the Qwen pairs live" below) because cross-engine
A/B mode deliberately never writes `results.json`.

### Corrected pairs

| Model (q4f16_1) | ZT total | ZT TTFT | ZT decode | WebLLM total | WebLLM decode (self-reported) | WebLLM prefill (self-reported) | **Δ total** | **Δ decode** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phi-3-mini    | **69.55** | 291 ms | 83.10 | **59.95** | 63.23 | 251 tok/s | **+16.0%** | **+31.4%** |
| Qwen3-4B      | **59.85** | 453 ms | 75.49 | **45.46** | 47.77 | 263–271 tok/s | **+31.7%** | **+58.0%** |
| Qwen3.5-4B    | **65.28** | 171 ms | 73.30 | **32.56** | 34.32 | 175–177 tok/s | **+100.5%** | **+113.6%** |

Raw runs (tok/s, total wall-clock):

- **Phi-3-mini** — Zero-TVM: 69.55, 69.67, 69.44, 69.29, 69.71 (the
  `bench/results.json` artifact). WebLLM: median 59.95; the harness records
  the median for this half, not the individual completions.
- **Qwen3-4B** — Zero-TVM: 60.23, 59.85, 60.07, 59.75, 59.43. WebLLM: 45.46,
  45.87, 44.67.
- **Qwen3.5-4B** — Zero-TVM: 65.32, 65.28, 65.25, 65.41, 65.19. WebLLM: 32.56,
  32.52, 33.72.

Auto-synced headline artifact (the Phi-3 pair, **total wall-clock**, from
`bench/results.json`):

| Engine                             | total tok/s (median) |
|-----------------------------------:|---------------------:|
| **Zero-TVM (this repo, defaults)** | **<!--bench:zt-->69.55<!--/bench:zt-->** |
| WebLLM 0.2.80 (`tensor-cache`)     | **<!--bench:webllm-->60.0<!--/bench:webllm-->** |

Auto-synced gap — Zero-TVM's deficit as a share of WebLLM's throughput,
negative = ahead: <!--bench:gap-->-16<!--/bench:gap-->% (i.e. +16.0% ahead on
total; +31.4% ahead on decode-only). The marker rounds WebLLM to 1 dp; the
exact median is 59.95.

### The trend: the advantage grows with architecture recency

This is the headline insight of the corrected round, and it is monotonic on
**both** metrics:

| Model | arch. released | Δ total | Δ decode |
|---|---|---:|---:|
| Phi-3-mini   | 2024 | +16.0% | +31.4% |
| Qwen3-4B     | 2025 | +31.7% | +58.0% |
| Qwen3.5-4B   | 2026 | +100.5% | +113.6% |

The reading that fits: compiler stacks have had less time to tune newer
architectures, so the margin a hand-written kernel set can take is larger the
newer the model. That is an **observation across three points**, not a proven
law — three models on one machine in one browser on one day, with no
mechanism isolated and no attempt to control for how differently each model
stresses the two engines. It is a hypothesis worth testing on a fourth model,
not a scaling result.

### Honest negative: WebLLM has the better first token on short prompts

We lose the first-token sprint. WebLLM's self-reported prefill runs at
**251 tok/s** on Phi-3 and **263–271 tok/s** on Qwen3-4B; against the
~35-token bench prompt that implies a TTFT of roughly **150 ms** on Phi-3,
where we measure **291 ms**. On Qwen3-4B our **453 ms** is worse still. Only
on Qwen3.5-4B is it roughly a wash (our 171 ms against an implied ~0.2 s from
its 175–177 tok/s prefill). Stated plainly: **we win sustained decode
decisively and lose time-to-first-token on short inputs.**

Two accounting caveats, both of which cut against us rather than for us:
WebLLM's implied TTFT is derived from its self-reported prefill rate, so it is
an estimate, not a measurement; and our `ttftMs` includes the first decode
step on top of prefill, which WebLLM's prefill-rate figure does not. The gap
is real either way, but the exact millisecond delta should not be read too
precisely.

The weakness is specifically *short* prompts. On long ones the chunked prefill
path is strong — **202 tok/s on an 816-token prompt**, measured 2026-07-29
(see "Prefill round A" below) — and cross-turn prefix reuse removes prefill
entirely on follow-up turns. Note what that means for the two models that lose
worst here: chunked prefill currently exists only on the Qwen3.5 hybrid path,
so Phi-3 and Qwen3 still run every prompt token through the full per-token
dispatch chain. That is the mechanism behind the loss, and porting the chunked
composition to the attention-only specs is the fix. Added to the levers list
below as the top open item.

### Absolute numbers moved for both engines; ratios within a session did not

Between 2026-07-29 and this round the Phi-3 absolutes moved a long way on
**both** sides — ours 62.90 → 69.55, WebLLM's 47.92 → 59.95 — with no engine
change between them on the Phi-3 path. This is the same machine-state
variance already documented in the tuning-round session note below (where a
control re-bench of a pre-round commit measured ~2.2× its recorded numbers on
a later day). **Only same-session pairs are meaningful; cross-session absolute
tok/s is not.**

One consequence that must be said out loud: the ratio is not as stable as this
file used to claim. The retired "−28…−31% across sessions" range was a
*total-wall-clock* ratio labelled "decode", and this session's total ratio is
−16.0%. The corrected decode-only ratio (+31.4%) happens to land inside that
old range, but that is a coincidence of accounting, not confirmation. Quote
per-pair, same-session, both metrics — not a band.

### Where the Qwen pairs live

`npm run bench` writes `bench/results.json` only in the plain (Phi-3) flow;
cross-engine A/B mode (`BENCH_QUERY="?model=qwen3"`) deliberately never
touches it. So that the Qwen pairs are not prose-only, this round also writes
two static artifacts alongside it:

- `bench/results/qwen3-4b.json`
- `bench/results/qwen35-4b.json`

Same shape as `results.json` plus the TTFT / decode / WebLLM-self-reported
fields, and both carry a `supersedes` note pointing at the withdrawn
2026-07-29 pair. They are not read by `sync-docs.mjs` — the table above is
their rendering.

## Headline — head-to-head vs WebLLM (2026-07-25/27, Apple M2 Max) — SUPERSEDED

> **Superseded by the 2026-07-30 corrected protocol above.** Not defective —
> this pair predates cross-turn prefix reuse, so both halves paid a full
> prefill and the comparison was like-for-like. But it is a *total
> wall-clock* number labelled "decode", it splits out no TTFT, and the
> 2026-07-30 re-measurement of the same pair on the same machine reads
> 69.55 / 59.95. Kept verbatim as recorded.

Same machine, same run, same Phi-3-mini-4k-instruct-q4f16_1 weight bytes
(served from `/local-weights/*`), same prompt, greedy decoding. Google Chrome
150.0.7871.182. Protocol: `npm run bench` — 128 decode tokens × 5 runs,
median. Only the inference engine differs.

| Engine                         | tok/s (median, as recorded) |
|-------------------------------:|----------------------:|
| **Zero-TVM (this repo, defaults)** | **62.90** |
| WebLLM 0.2.80 (`tensor-cache`) |   **47.92** |

**Zero-TVM measured ~31% faster than WebLLM** on this machine with our own
weights. Zero-TVM runs (2026-07-27 artifact): 62.79, 63.11, 63.14, 62.90,
62.88 tok/s.

Absolute numbers are machine-state-dependent — across three same-run pairs
taken over two days the medians were 66.33/51.98 (−28%), 62.90/47.92 (−31%),
and one clearly-degraded pair (62.63/41.51, taken immediately after ~10
back-to-back GPU bench sessions) that depressed both engines and is not used
for any claim. This is where the old "−28% to −31% in healthy sessions" band
came from; the corrected round retires it (see above). Every quoted pair is
same-run, same-weights.

Scope caveats, stated up front: one machine (M2 Max), one model
(Phi-3-mini q4f16_1), one browser (Chrome 150), short-context bench. The
defaults now include the vec4-load kernels measured below and, since
2026-07-27, split-K attention (long-context A/B below).

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

## Qwen3-4B (2026-07-28, Apple M2 Max) — v1 port, same-weights A/B vs WebLLM — SUPERSEDED

> **Superseded by the 2026-07-30 corrected protocol.** Not defective — this
> pair predates cross-turn prefix reuse, so both halves paid a full prefill.
> It is superseded twice over: the session was a degraded machine state (both
> engines ~3× low, see the tuning-round session note below), and the port has
> been tuned since. Kept verbatim as recorded.

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

## Qwen3-4B tuning round (2026-07-29, Apple M2 Max) — fused qk_norm+RoPE+append + K%512 vec4

> **The cross-engine pairs in this section are WITHDRAWN** — 75.74 / 43.75
> ("+73.1%") for Qwen3-4B and 65.67 / 34.04 ("+92.9%") for the Qwen3.5
> cross-check. Both were measured after cross-turn prefix reuse landed
> (PR #24) and before the bench harness was fixed to reset it, so the
> Zero-TVM half prefilled one token per run while the WebLLM half paid a full
> prefill: not like-for-like. See "Corrected protocol (2026-07-30)" at the top
> for the defect and the replacement pairs (Qwen3-4B 59.85 / 45.46 total,
> +31.7%; Qwen3.5-4B 65.28 / 32.56 total, +100.5%).
>
> **The Zero-TVM-vs-Zero-TVM flag ladder below is NOT withdrawn.** Both sides
> of each of those A/Bs share the same accounting, so the per-item deltas
> (fused-qk +2.3%, vec4h +5.7%, combined +5.8%) stand — only their absolute
> tok/s values are decode-only rather than total. Kept verbatim as recorded.

Two engine changes to the Qwen3 decode path, each behind its own URL flag so
the A/B halves run in the same build:

1. **Fused qk_norm+RoPE+KV-append (`?fuseqk=0` opts out).** The qkNorm
   blocker was never the whole fusion — only folding into `qkv_fused`, whose
   one-RoPE-pair-per-WG shape has nowhere to run the per-head norm reduction.
   Keeping the QKV matmul separate and fusing everything AFTER it works: the
   new `qk_norm_rope_append` kernel runs one 32-thread WG per (token, head)
   over Q, K **and** V heads — per-head RMSNorm reduction in f32, normalized
   head staged in workgroup memory so RoPE reads its ±HALF_ROTARY partner
   from shared, K/V written straight into the paged cache. 4 dispatches
   (matmul → qk_norm → rope → kv_append) become 2; **10 → 8
   dispatches/layer** (364 → 292/token). Pinned against a composed CPU
   reference in the Qwen suite with negative controls (V region must be raw
   and bit-exact; no stray page writes).
2. **K%512 vec4 matmul variants (`?vec4h=0` opts out).** The vec4 loads
   required K % 1024 == 0 (32 threads × 32 K-elements per iteration),
   excluding Qwen3's two hottest instances (d=2560: qkv/gate_up/LM-head;
   ffn=9728: down_proj). The `_vec4h` generator siblings halve the
   per-thread unroll — vec2<u32> weight loads (16 nibbles = half a scale
   group), activations still vec4<u32> — relaxing the constraint to
   K % 512 == 0, which both shapes satisfy. `resolveMatmul` now resolves
   per instance: K%1024 → `_vec4`, else K%512 → `_vec4h`, else scalar-load.
   4 new pipelines (f16/f32 × sg/tiled), correctness-tested at K=2560 and
   K=9728.

Also considered, not shipped: (b) restructuring `qkv_fused` to compute a
whole head per WG with an in-WG norm reduction (4 dispatches → 1). Skipped
on prior evidence rather than re-measured: it would dispatch HEADS+2·KV_HEADS
= 48 WGs where the tiled matmul dispatches 1536, and the three qkv-tiling
negatives above showed that even halving 4608 → 2304 WGs costs ~10% on
Apple. The optional int8-KV unfused composition was also skipped this round:
its value is KV memory (not decode speed — it costs an extra dispatch per
layer) and it cannot ride the new fused kernel (which writes f16 pages).

**Session note — the 2026-07-28 baseline did not reproduce.** Same machine
(M2 Max, 32 GB), same Chrome 150.0.7871.187, same local weight bytes, and
the flags-off configuration (dispatch-for-dispatch the 2026-07-28 decode
chain) measured **71.57 tok/s where 25.43 was recorded the day before —
and WebLLM's half moved with it (14.15 → 45–46 tok/s, on v0_2_84 libs vs
0.2.80 then)**. To rule out a repo-side cause, the pre-round commit
(`7a66144`, before the hybrid-perf and prefill rounds) was re-benched from
a clean worktree the same day: **56.02 zt / 46.13 WebLLM** — i.e. the exact
07-28 code also runs ~2.2×/3.3× its recorded numbers today. The two
Zero-TVM figures even reconcile arithmetically: the old code re-prefills
the ~35-token bench prompt inside every run's wall clock (no prefix reuse
yet), and 128/(163/71.57) = 56.2 ≈ 56.02 — so the old commit and today's
flags-off half measure the SAME underlying decode rate, differing only in
prefill accounting. Conclusion: the 07-28 session (both engines, both
recorded sessions that day) was in a degraded machine state — the same
phenomenon as the "clearly-degraded pair" note in the Phi-3 headline.
Consequence: the per-item deltas below are computed against the SAME-DAY
flags-off half, not against 25.43; the 07-28 numbers stay in the section
above as recorded.

Ladder, same day, same protocol (`BENCH_QUERY="?model=qwen3&…" npm run
bench`, 128 decode tokens, greedy, local mirror; 3 runs for the A/B halves,
5 for the headline pair; each config is a separate same-day session):

| Config (Qwen3-4B q4f16_1)              | tok/s (median) | Δ vs flags-off |
|----------------------------------------|---------------:|---------------:|
| flags-off (`?fuseqk=0&vec4h=0` — the 07-28 composition) | 71.57 | — |
| item 1 only (`?vec4h=0`)               | 73.22 | +2.3% |
| item 2 only (`?fuseqk=0`)              | 75.63 | +5.7% |
| **both (defaults)**                    | **75.74** | **+5.8%** |

Raw runs — flags-off: 71.42/71.57/71.71; item 1: 73.24/72.95/73.22; item 2:
74.75/75.63/75.64; defaults: 74.29/75.74/75.91/76.16/75.49. Honest
composition note: the two wins do NOT stack — with the vec4h matmuls in
place the fused-qk win shrinks from +2.3% to ~+0.1% (75.63 → 75.74, inside
the ±0.5 tok/s run spread). Both stay default-on: each is a clean standalone
win, `?fuseqk` also drops 72 dispatches/token, and neither regresses — but
the round's throughput is essentially the vec4h number.

Headline same-session pair (defaults, 5 × 128 tokens vs WebLLM 0.2.84's
prebuilt `Qwen3-4B-q4f16_1_cs1k-webgpu.wasm`, its usual 3 × 120-token
protocol; `bench/results.json` untouched):

| Engine (Qwen3-4B q4f16_1) — **WITHDRAWN, see banner** | decode tok/s (median) |
|-----------------------------------------:|----------------------:|
| **Zero-TVM (`?model=qwen3`, defaults)**   | **75.74** |
| WebLLM 0.2.84 (prebuilt Qwen3-4B lib)     | **43.75** |

**"+73.1% over WebLLM on this pair" — withdrawn.** The Zero-TVM half
prefilled one token per run here and the WebLLM half prefilled the whole
prompt; the corrected same-pair measurement (2026-07-30) is 59.85 / 45.46
total, **+31.7% total / +58.0% decode**. WebLLM halves across the day's four
sessions: 45.26/46.24/45.51/43.75 (±3%). Scope: one machine, one day,
short-context — same limits as every section in this file.
The Qwen3 path remains unfused at the QKV matmul itself (8 dispatches/layer
vs Phi-3's 7); the remaining gap to a Phi-3-style path is the qkv_fused
fold ruled out under (b) above.

**Qwen3.5 cross-check (same day).** The `_vec4h` variants also engage on the
hybrid's K=2560 instances (fused GDN in_proj, c_attn, gate_up); the fused-qk
kernel does not (the hybrid attention chain keeps the reference composition).
Same-day pair: **65.67** vs WebLLM 0.2.84's **34.04** tok/s (**+92.9%** —
**withdrawn**, same defect as the Qwen3 pair above; the corrected 2026-07-30
pair is 65.28 / 32.56 total, **+100.5% total / +113.6% decode**);
`?vec4h=0` half: 64.38 (raw runs 65.73/65.29/65.67 vs 64.63/64.32/64.38) —
**vec4h is +2.0% on the hybrid** (a Zero-TVM-vs-Zero-TVM A/B, unaffected),
no regression. Note on 53.07 → 64.38
(the 2026-07-29 perf-round number vs today's flags-off half): that jump is
NOT this round's work — the prefill round's cross-turn prefix reuse
(landed between the two measurements) removes the ~35-token re-prefill
from every bench run's wall-clock window, which alone accounts for the
bulk of the difference (the two figures reconcile to the same underlying
decode rate if the old runs prefilled the short shallow-KV prompt at
~110 tok/s — plausible for the readback-free per-token prefill, though
not directly measured at that length). Accounting change plus ordinary
session variance, documented here so nobody reads it as a kernel win
(WebLLM's half moved only 32.36 → 33.2–34.0 across the same sessions).

## Qwen3.5-4B hybrid (2026-07-28, Apple M2 Max) — v1 scalar-GDN, same-weights A/B vs WebLLM — SUPERSEDED

> **Superseded by the 2026-07-30 corrected protocol.** Not defective — this
> pair predates cross-turn prefix reuse, so both halves paid a full prefill.
> Superseded by the fused-GDN work that followed and by the corrected
> re-measurement (65.28 / 32.56 total, +100.5% total / +113.6% decode). Kept
> verbatim as recorded.

First cross-engine measurement of the Qwen3.5-4B hybrid port
(`?model=qwen35`: 24 gated-DeltaNet layers + 8 gated-attention layers).
Same protocol as the other pairs: same machine, same session (halves
back-to-back), same Qwen3.5-4B-q4f16_1 weight bytes served from the local
mirror (`/local-weights/Qwen3.5-4B-q4f16_1-MLC/`, `tensor-cache.json`
manifest), greedy decoding, Chrome 150.0.7871.187. Zero-TVM half:
`BENCH_QUERY="?model=qwen35" npm run bench`, 128-token target × 5 runs,
median (the model emits its stop token at 94 tokens on the bench prompt, so
each run measures 94 decode tokens). WebLLM half: `webllm-bench.html?model=qwen35`
— **WebLLM 0.2.84** (dep bumped from the 0.2.80-era lib set: the Qwen3.5
hybrid first ships in WebLLM's v0_2_84 prebuilt libs) running its own
prebuilt `Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm` model_lib, its fixed protocol
(3 × 120-token completions + warmup, median, wall-clock — identical
accounting to the other WebLLM halves). `bench/results.json` untouched — it
remains the Phi-3 headline artifact.

| Engine (Qwen3.5-4B q4f16_1)                 | decode tok/s (median) |
|--------------------------------------------:|----------------------:|
| **Zero-TVM (`?model=qwen35`, defaults)**     | **47.99** |
| WebLLM 0.2.84 (prebuilt Qwen3.5-4B lib)      | **31.99** |

Raw runs — Zero-TVM: 47.85, 47.99, 48.02, 48.16, 47.97 (unusually tight —
±0.3 tok/s). WebLLM: 32.24, 31.99, 31.74; WebLLM's own reported
`decode_tokens_per_s` was 34.03 (the wall-clock figure includes its ~0.2 s
prefill, same as every number in this file). One protocol wart on the WebLLM
half, stated for completeness: the upstream repo's `mlc-chat-config.json`
ships stale Qwen3 stop ids (151643/151645 — ordinary BPE tokens in the 248k
vocab), so WebLLM generates to its 120-token budget instead of stopping at
`<|im_end|>`; that does not affect per-token decode-rate accounting.

Zero-TVM measures **+50.0% over WebLLM on this pair**. The caveats, up
front:

- **The Zero-TVM path is v1-scalar-GDN — this number is a floor.** The 24
  DeltaNet layers run scalar (non-subgroup) kernels; there is no chunked
  prefill (the recurrent state means prompts replay token-by-token through
  the sequential `generate` path); and the GDN projections (qkv/z/a/b) are
  separate unfused dispatches. None of the Phi-3 fusion story has been
  applied to the GDN half yet.
- **One machine, one pair, short-context, decode-only** — same scope limits
  as every other section, with single-pair replication.
- Notable but unweighted observation: the hybrid runs much closer to the
  Phi-3 rate than Qwen3-4B does on both engines (Zero-TVM 62.9 → 48.0 vs
  → 25.4; WebLLM 47.9 → 32.0 vs → 14.2). The 24 recurrent layers avoid both
  engines' attention/KV costs; draw no stronger conclusion from one pair.

Split-K spot check on the hybrid (single run each, same session —
`?splitk=0` 47.72 vs default `splitk=8` 48.17 tok/s, +0.9%): split-K
compiles and runs correctly at head_dim 256, and its effect is within noise
here — expected, since only 8 of 32 layers run attention at all. Kept at
the default (`splitk=8` where sg32 exists), same as the other models.

## Qwen3.5-4B hybrid perf round (2026-07-29, Apple M2 Max) — fused GDN in_proj + incremental blocking decode — SUPERSEDED

> **Superseded by the 2026-07-30 corrected protocol.** Not defective — this
> round landed (PR #23) *before* cross-turn prefix reuse (PR #24), so both
> halves of the 53.07 / 32.36 pair paid a full prefill. Superseded by the
> corrected re-measurement (65.28 / 32.56 total, +100.5% total / +113.6%
> decode). Kept verbatim as recorded.

Two changes since the 2026-07-28 v1 floor above (kept as history):

1. **Fused GDN input projection (4 dispatches → 1).** The four separate
   in_proj matmuls per DeltaNet layer (`in_proj_qkv` 8192 rows + `in_proj_z`
   4096 + `in_proj_a` 32 + `in_proj_b` 32, all K=d) are now ONE 12352-row
   int4 matmul: the loader concatenates the four q4f16_1 weight/scale
   records at upload (row-major, so row concat is byte concat) and the
   engine runs a single dispatch into a packed output buffer. Downstream
   kernels read the regions in place — `gdn_conv` reads the qkv region at
   offset 0, `gdn_norm_out` binds the z region and `gdn_gates` the [a|b]
   pair via 256-aligned bind-group offsets. 412 → **340 dispatches/token**
   (24 GDN layers × 3 fewer). The two tiny 32-row matmuls were pure
   dispatch overhead; the z rows now ride the same weight-scan as qkv.
2. **Incremental blocking decode (no within-call prompt replay).** The
   blocking `generate()` used to replay the whole prompt from position 0 on
   every call because the GDN state (conv ring + recurrent S) is
   non-idempotent and the engine couldn't prove it matched `startPos`. The
   engine now tracks the state position (`gdnStatePos`, advanced by every
   submitted forward pass) and skips the replay when it provably sits at the
   requested boundary — the validate battery's `generate(prompt, len, 24)`
   after `forwardLogits` now runs **zero** prompt passes (it reads back the
   argmax the final prefill pass already produced) + one pass per generated
   token. This does not touch the chat/bench path (`generatePipelined` was
   already incremental — one chained pass per token); it fixes the harness
   path, whose reported decode rate jumped from ~17 to ~40 tok/s (the old
   figure amortized a full hidden replay). Cross-turn chat still re-prefills
   from 0 (unchanged, same as Qwen3/Qwen3.5 chat behavior).

Same protocol as above (same machine, same session, halves back-to-back,
local-mirror weight bytes, greedy, Chrome 150; `BENCH_QUERY="?model=qwen35"
npm run bench`, 128-token target × 5 runs — the model stops at 94 tokens on
the bench prompt; WebLLM 0.2.84 on its prebuilt
`Qwen3.5-4B-q4f16_1_cs1k-webgpu.wasm`, 3 × 120-token completions).
`bench/results.json` untouched.

| Engine (Qwen3.5-4B q4f16_1)                 | decode tok/s (median) |
|--------------------------------------------:|----------------------:|
| **Zero-TVM (`?model=qwen35`, defaults)**     | **53.07** |
| WebLLM 0.2.84 (prebuilt Qwen3.5-4B lib)      | **32.36** |

Raw runs — Zero-TVM: 53.32, 53.07, 52.87, 53.54, 52.81. WebLLM: 32.46,
32.15, 32.36 (WebLLM's own reported `decode_tokens_per_s` 34.20 —
wall-clock accounting as everywhere in this file). **+10.6% over the
2026-07-28 Zero-TVM floor (47.99 → 53.07); +64.0% over WebLLM on this
pair** (was +50.0%). The chat-path gain is entirely change 1 — change 2
moves the validation/blocking path only. Still on the table for the GDN
half: subgroup/vec4 variants for the GDN kernels and a fused conv+gates
step (chunked prefill landed in the 2026-07-29 prefill round below).

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

## Prefill round A (2026-07-29, Apple M2 Max) — chunked GDN prefill + cross-turn prefix reuse

Decode was never the whole story: every prompt token used to run the FULL
per-token dispatch chain (Qwen3.5: 340 dispatches/token), and every chat
turn re-prefilled the WHOLE conversation. Two changes, measured with the
new devtools harnesses (`benchPrefill(800, 3)` / `benchTurns()` /
`checkReuse()` on the chat page; A/B via `?chunk=0` / `?reuse=0` in the
same build — the flags-off halves ARE the pre-change behavior). Decode
path untouched (`recordForward` unchanged; same 340 dispatches/token).
`bench/results.json` untouched.

1. **Chunked GDN prefill (Qwen3.5, `?chunk=0` opts out).** Prompt tokens
   before the last run in chunks of ≤64: every projection (fused GDN
   in_proj, out_proj, c_attn, o_proj, gate_up, down) becomes ONE
   `int4_matmul_batched_dyn` dispatch — the m=4 register block looping over
   runtime-M row blocks, unpacking each weight word once per 4 batch rows
   (4× weight-traffic amortization, tile L2-resident across blocks);
   `gdn_conv_seq`+`gdn_conv_commit` batch the causal conv (ring read/rotate
   split so the chunk is race-free), `gdn_gates`/`gdn_norm_out` grew
   seq+stride uniforms, and the recurrence — already seq-capable — runs as
   ONE `gdn_recur` dispatch per layer per chunk. The 8 attention layers
   batch too: rope/kv_append were already seq-capable and the new
   `attention_prefill` walks per-token causal kv_len (bit-exact vs the
   decode kernel, pinned in tests). FFN runs batched gate_up → `silu_mul` →
   batched down. ~394 dispatches per 64-token chunk ≈ **6.2/token vs 340**.
   Chunked-vs-per-token equality is pinned bit-exactly (f32 recurrent state
   included) by `gdn_chunk_chain` in `tests/kernels/compile-qwen35.mjs`
   (suite now 19/19). Requires sg32; falls back per-token otherwise.

   | Qwen3.5 prefill, 816-token prompt (median of 3) | prefill tok/s | TTFT |
   |---|---:|---:|
   | before (`?chunk=0&reuse=0`)                     |  67.9 | 12.0 s |
   | after (chunked, defaults)                       | **202.1** | **4.0 s** |

   Raw runs — before: 69.4 / 67.9 / 67.7; after: 213.8 / 197.4 / 202.1
   (13 chunks/run). **2.98× prefill throughput.**

2. **Cross-turn prefix reuse (all three models, `?reuse=0` opts out).** The
   engine records the exact (position, token) of every submitted forward
   pass — including the pipelined path's ≤1-step overrun past a stop token,
   reconstructed from the readback chain — and a new turn prefills only the
   delta past the longest reusable prefix. Trust rules: pure-attention
   specs reuse `min(LCP, len-1)` (KV rewrite is idempotent; stale slots are
   overwritten in order). Hybrid GDN is all-or-nothing: the recurrence
   can't rewind, so reuse requires the new prompt to extend EVERY absorbed
   token with the state boundary exactly there, else full re-prefill (which
   re-zeroes GDN state at position 0). Template mechanics measured:
   - *ChatML (Qwen3/Qwen3.5, non-thinking)*: past assistant turns are now
     re-rendered WITH the empty `<think>\n\n</think>\n\n` block (it was
     genuinely part of the generation prompt the model continued), making
     each turn an exact token-level extension of the absorbed sequence —
     the ≤1-token overrun is the `<|im_end|>` stop id, which the next
     prompt also contains. Turn 3 below reuses 922 of 950 tokens.
   - *Phi-3 (SPM)*: the previous prompt reuses in full; the generated text
     re-encodes with a boundary merge at `<|assistant|>\n` ↔ first response
     token, so reuse conservatively restarts there when the merge differs
     (LCP handles it — no correctness impact). Turn 3 reused 1086 of 1105.

   3-turn conversation (`benchTurns()`: ~120 generated tokens/turn,
   turn-3 prompt ≈ 950–1105 tokens absorbed), TTFT of turn 3:

   | Model | turn-3 TTFT before | after | reused | speedup |
   |---|---:|---:|---:|---:|
   | Phi-3    | 15,405 ms | **269 ms** | 1086/1105 | 57× |
   | Qwen3-4B | 14,554 ms | **438 ms** |  922/950  | 33× |
   | Qwen3.5  | 14,340 ms | **194 ms** |  922/950  | 74× |

   (Qwen3.5 turn-2, where the reusable prefix is ~58%: 12,083 → 1,790 ms —
   reuse + chunking compose.)

**Correctness gates.** `checkReuse()` (opt-in debug assertion,
`engine.debugCompareReuse`) prefills a turn-2 prompt via the reused prefix,
reads the final-position logits, then does a fresh full prefill of the same
prompt and diffs: **max|Δ| = 0 (bit-exact) on all three models** (Qwen3.5
asserted with `?chunk=0`; with chunking on, the absorbed prefix was built
by the batched-matmul kernels whose reduction order differs from the
per-token GEMV, and the diff measures that variant numerics instead —
max|Δ| 0.019 / mean 0.0023 on the 248k-logit vector, the same class as the
sg-vs-scalar differences). Suites: kernels 28/28 + 21/21 + **19/19** (6 new
chunked-prefill tests, incl. bit-exact chunk-vs-stepwise with f32 state),
unit 287, e2e **13/13** (2 new multi-turn tests asserting turn-2 coherence
AND that reuse actually engaged, per model family). Decode sanity after
the round: 51.2 tok/s on a quick 2×64-token check (headline protocol
unchanged — decode dispatches untouched).

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

### Remaining levers (updated after the 2026-07-30 corrected round)

0. **Short-prompt TTFT — the one place we lose.** WebLLM's prefill is
   251–271 tok/s on Phi-3/Qwen3 (implied TTFT ~150 ms on Phi-3) against our
   measured 291 ms / 453 ms; only Qwen3.5 is a wash. Long prompts are already
   fine (202 prefill tok/s at 816 tokens) and repeat turns are free (prefix
   reuse), so the target is specifically the fixed cost of a short, cold
   prefill — the obvious candidate is extending the Qwen3.5 chunked-prefill
   composition to the attention-only specs so a 35-token prompt is one batched
   pass instead of 35 per-token passes. Top open item.
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
await bench(128, 3)            // total tok/s + TTFT + decode tok/s, 3 runs + 1 warmup
await bench(0, 0, true)        // per-kernel profile (requires timestamp-query)
```

`bench()` resets the engine's absorbed-token record before every run, so each
run pays a full prefill — the same work the WebLLM half does. It returns
`{ median, medianDecode, medianTtftMs, runs }`; `webllm-bench.html` returns
`{ median, medianSelfDecode, runs }` where each run also carries WebLLM's own
`selfDecode`. Always report the total and the decode number together.

URL toggles for A/B bisection:

- `?sg=0`        — disable all subgroup shaders
- `?sgqkv=0`     — disable _sg qkv only
- `?sgattn=0`    — disable _sg attention only
- `?sgargmax=0`  — disable _sg argmax only
- `?sgffn=0`     — disable tiled_sg FFN (use scalar fused_ffn)
- `?matmul=scalar|sg|tiled` — force matmul variant
- `?vec4=0` / `?vec4qkv=0` — disable the vec4-load defaults (measured +7.1%
  together on M2 Max; default ON since 2026-07-25)
- `?vec4h=0` — disable just the K%512 `_vec4h` half-unroll matmul siblings
  (+5.7% on Qwen3, +2.0% on Qwen3.5; default ON since 2026-07-29)
- `?fuseqk=0` — disable the fused qk_norm+RoPE+KV-append kernel on the
  Qwen3 unfused path (+2.3% alone; default ON since 2026-07-29)

Opt-in experiments (measured 2026-07-25 on M2 Max — see the A/B table above):

- `?splitk=N`       — split-K attention + combine (f16 KV only): +3.1% at
  128-token / +4.0% at 1024-token decode — **default N=8 since 2026-07-27**;
  `?splitk=0` to disable
- `?fuseprologue=1` — add_norm folded into the FFN prologue: −13.7% on Apple
  (falsified there; kept for A/B on other GPUs)
