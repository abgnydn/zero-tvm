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
The corrected-protocol section below has the details.

Two baselines now. **WebLLM/MLC** is the same-bytes A/B: identical q4f16_1
weight files, so it isolates the runtime. **llama.cpp via wllama** is *not*
same-bytes — it reads GGUF — so it measures runtime and quantization together.
The two are reported separately and must not be merged into one table.

## Qwen3.6-35B-A3B MoE (2026-08-05) — no baseline exists

**This model has no A/B column.** WebLLM ships zero Qwen3.6 builds, so the
same-bytes protocol that governs every other section cannot be run. What can
be stated is what was measured, on the usual machine (Apple M2 Max 32 GB,
Chrome, dev mirror weights):

| build | resident | condition | decode |
|---|---|---|---|
| `?model=qwen36q3` (3-bit experts) | 15.7 GB | quiet machine (owner-run) | **~50–60 tok/s** |
| `?model=qwen36q3` | 15.7 GB | heavily loaded machine (0.2 GB free RAM) | 11.4 tok/s (23-token sample) |
| `?model=qwen36` (full 4-bit) | 19.7 GB | loaded machine | unusable — GPU process killed mid-prefill |

Correctness is anchored elsewhere, not by eyeballing chat: every layer type is
validated against mlx_lm's own modules on the real checkpoint
(`npm run test:kernels:real` — MoE block 3.35e-4, whole decoder layer 5.16e-4,
greedy argmax identical to mlx_lm on the tested prompt). The quiet-machine
number is a single owner-run session, not a median-of-N protocol run; treat it
as indicative until a proper protocol round is recorded. The memory-pressure
row is the honest caveat: this model's floor is RAM, not kernels — the same
kernels do 16 ms/token (62 tok/s) in the isolated 40-layer benchmark with a
480 MB working set.

## Third baseline: llama.cpp via wllama (WebGPU) (2026-07-30, Apple M2 Max)

**This is not a same-bytes comparison.** wllama reads GGUF — `Q4_K_M` for the
two Qwen models, `Q4` for Phi-3 — while Zero-TVM and WebLLM read MLC
`q4f16_1`, so every number in this section compares a **runtime *and* a
quantization** together, unlike the WebLLM pairs below where both engines load
byte-identical weight files.

Keep that distinction attached to these figures wherever they are quoted. A
different quantization changes the bytes moved per token, the dequant kernel,
and the arithmetic — Q4_K_M's super-block structure is not q4f16_1's, and this
bench is memory-bound, so weight-format differences land directly on tok/s.
Nothing here separates "llama.cpp's kernels are slower" from "Q4_K_M is more
expensive to move and unpack than q4f16_1". Both are folded into one number.

### Why a third baseline

Every Zero-TVM number published before today was measured against WebLLM
alone. That is one baseline, and it is a compiler stack — which makes it the
*easier* opponent for the thesis this repo is testing. llama.cpp's WebGPU
backend (landed in wllama v3.1, upstream PR #215; here `@wllama/wllama@3.5.1`,
libllama **b9640-dd4623a**) is the hand-written-C++ opponent, and published
work says it is the stronger one: **arXiv 2605.20706 (LlamaWeb) reports
llama.cpp-WebGPU decoding 45–69% faster than WebLLM across 16 devices.** If
that held on this machine, WebLLM would be the wrong baseline to be publishing
against and several of the numbers above would need re-framing. So it was
worth measuring rather than assuming.

### Result — it does not reproduce, and the gap runs the other way

| Model | engine | total tok/s | TTFT | decode tok/s | gen tok |
|---|---|---:|---:|---:|---:|
| **Phi-3-mini** | Zero-TVM q4f16_1 (paired w/ WebLLM) | **69.70** | 290 ms | **83.20** | 120 |
| | Zero-TVM q4f16_1 (paired w/ wllama) | 68.15 | 303 ms | 81.61 | 120 |
| | WebLLM 0.2.80, q4f16_1 | 56.98 | ~113 ms † | 60.07 ‡ | 119 |
| | **wllama, GGUF Q4** | **24.62** | 321 ms | **26.16** | 119 |
| **Qwen3-4B** | Zero-TVM q4f16_1 (paired w/ WebLLM) | **60.30** | 446 ms | **75.94** | 127 |
| | Zero-TVM q4f16_1 (paired w/ wllama) | 53.25 ⚠ | 506 ms | 67.10 ⚠ | 127 |
| | WebLLM 0.2.84, q4f16_1 | 46.95 | ~138 ms † | 49.44 ‡ | 119 |
| | **wllama, GGUF Q4_K_M** | **19.22** | 626 ms | **21.59** | 101 |
| **Qwen3.5-4B** | Zero-TVM q4f16_1 (paired w/ WebLLM) | **65.17** | 171 ms | **73.19** | 94 |
| | Zero-TVM q4f16_1 (paired w/ wllama) | 63.59 | 170 ms | 71.07 | 94 |
| | WebLLM 0.2.84, q4f16_1 | 34.54 | ~204 ms † | 36.54 ‡ | 119 |
| | **wllama, GGUF Q4_K_M** | **16.48** | 853 ms | **19.18** | 95 |

† Derived from WebLLM's self-reported prefill rate against the prompt length,
not measured — its page reports no TTFT. ‡ WebLLM's own
`decode_tokens_per_s`. ⚠ Thermally disturbed Zero-TVM run; see "Our own half
was the noisy one" below.

Same-session deltas, Zero-TVM against wllama (each pair measured back-to-back
in one browser session, so these are the only deltas that mean anything):

| Model | Δ total | Δ decode |
|---|---:|---:|
| Phi-3-mini | **+176.8%** | **+212.0%** |
| Qwen3-4B (disturbed ZT half) | **+177.0%** | **+210.8%** |
| Qwen3-4B (clean ZT half, cross-session) | +213.7% | +251.7% |
| Qwen3.5-4B | **+285.8%** | **+270.5%** |

And the number that actually falsifies the LlamaWeb result on this device —
**WebLLM beats wllama**, measured the same day on the same machine, in the
opposite direction and by roughly 3–3.6× the reported magnitude:

| Model | WebLLM total | wllama total | WebLLM is |
|---|---:|---:|---:|
| Phi-3-mini | 56.98 | 24.62 | **2.31× faster** |
| Qwen3-4B | 46.95 | 19.22 | **2.44× faster** |
| Qwen3.5-4B | 34.54 | 16.48 | **2.10× faster** |

The two engines' medians were taken from separate sessions on the same day, so
that last table is weaker evidence than the same-session pairs above. It is
still a 2× effect, well outside the ~10 tok/s session-to-session drift this
file already documents.

### Raw runs (tok/s, total wall-clock)

- **Phi-3** — ZT/webllm `69.61 69.81 69.54 69.76 69.70` · ZT/wllama
  `69.84 68.99 63.49 67.25 68.15` · WebLLM `56.98 56.98 57.68` · wllama
  `24.46 24.63 24.62`
- **Qwen3-4B** — ZT/webllm `60.17 60.14 60.30 60.71 60.61` · ZT/wllama
  `48.84 48.31 54.03 53.50 53.25` · WebLLM `46.95 47.56 45.95` · wllama
  `19.22 19.23 18.85`
- **Qwen3.5-4B** — ZT/webllm `65.62 65.50 65.17 64.90 56.31` · ZT/wllama
  `66.25 66.33 63.59 59.61 57.66` · WebLLM `34.79 34.45 34.54` · wllama
  `16.36 16.48 16.68`

### Prefill is the smoking gun

llama.cpp's WebGPU backend self-reports **79 / 62 / 42 prefill tok/s** (Phi-3 /
Qwen3 / Qwen3.5) on a 24–31-token prompt, against WebLLM's **249 / 264 / 176**
on the same prompts. That is a 3–4× prefill deficit, materially worse than its
~2.3× decode deficit, and 42 tok/s of prefill on a 4B model is not a plausible
steady-state figure for any competent batched matmul. The most likely reading
is an immature batched-matmul path on Dawn/Metal rather than a decode-loop
deficit — which is exactly the kind of thing that would differ on a newer
libllama build or a non-Apple GPU. It is also the reason not to treat this as
a verdict on llama.cpp's WebGPU backend in general.

### Our own half was the noisy one — publish that too

Zero-TVM ran twice per model (once against each baseline), which makes its own
drift visible, and it is not flattering:

- **Qwen3-4B spread 11.7% between the two invocations** — 60.30 clean against
  53.25 in the wllama session, whose per-run ramp `48.84 → 48.31 → 54.03 →
  53.50 → 53.25` is a thermally disturbed session, not a measurement. Treat
  60.30 as Qwen3's value and 53.25 as a degraded outlier.
- **Qwen3.5 drifted downward inside a single invocation** in both sessions
  (`65.62 … 56.31` and `66.25 … 57.66`). The median is doing real work there.
- **wllama was the more stable engine throughout** (≤2% spread on every run
  set), because at ~20 tok/s it never loads the GPU hard enough to throttle.
  Our engine's numbers are the ones that need a median and a warm machine.

So the honest form of the headline is a range with a reason, not a point:
Zero-TVM is roughly **2.8–3.9× wllama's wall-clock throughput on this machine
at this quantization pairing**, and the top of that range is partly our own
best-case session.

### Why we distrust our own result

This is a very large win in our own favour against a well-regarded engine,
which is exactly the situation this file's falsified-experiment entries exist
to guard against. Recorded reasons to hold it loosely:

1. **One device, one browser, one day** — Apple M2 Max, Chrome 150.0.7871.187.
   LlamaWeb's claim spans 16 devices; a single Apple GPU cannot refute it, only
   fail to reproduce it here.
2. **One libllama build** — b9640-dd4623a, as vendored by
   `@wllama/wllama@3.5.1`. The WebGPU backend is young and moving fast; a
   newer build could close much of the prefill gap.
3. **The quantization confound is unresolved** — see the top of this section.
   Some unknown share of the gap is Q4_K_M-vs-q4f16_1, not runtime.
4. **No mechanism was isolated.** The prefill diagnostic points at batched
   matmul, but nothing here profiles llama.cpp's kernels to prove it.

Conclusion recorded as such: **the WebLLM same-bytes baseline is not retired
in favour of this one.** It remains the primary comparison because it is the
one where the weights are identical. This section is a second data point with
a confound, published because omitting a baseline we actually measured would
be the worse dishonesty.

### A bug that produced a fake number first (found and fixed pre-publication)

The first sweep reported **Qwen3.5 at 0.00 tok/s ("+Infinity%")** and Qwen3 at
0 tokens on 2 of 4 runs, while llama.cpp simultaneously self-reported 19–22
tok/s. The cause: the Qwen3/Qwen3.5 **GGUF chat templates enable thinking by
default**, so llama.cpp streamed every token as `delta.reasoning_content` and
the page — counting only `delta.content` — divided 0 tokens by 5.8 s. A
plausible-looking zero that silently poisoned the median.

It was also a genuine like-for-like defect independent of the counting bug:
the Zero-TVM half runs Qwen's **non**-thinking template (`model-select.ts`
passes `{thinking: false}`), so the two halves were not decoding the same
task. Fixed in `src/wllama-bench/main.ts` with
`chat_template_kwargs: { enable_thinking: false }` (verified inert on Phi-3 —
24.85 vs 24.49, noise), plus a `reasoningChunks` counter, a loud zero-token
guard, and a `contentAccountingOk` flag surfaced through `bench/run.mjs`.
Every number in this section is post-fix.

Two more protocol traps worth recording, both of which would have inflated
wllama's number in *its* favour or deflated it unfairly:

- **`cache_prompt` defaults to ON in llama.cpp.** Left alone, runs 2..N skip
  prefill entirely — the same class of defect as the withdrawn 2026-07-29
  pairs, pointed the other way. Set to `false`, and the stable 319–443 ms TTFT
  across runs confirms a full prefill every time.
- **Cross-origin isolation.** Vite's dev server sets no COOP/COEP, so
  `SharedArrayBuffer` is absent and wllama silently drops to single-thread
  WASM. Production already sets `same-origin` + `credentialless` site-wide in
  `public/_headers`; that is replicated in dev but **scoped to
  `/wllama-bench*` only**, so the Zero-TVM and WebLLM pages keep serving
  byte-identical headers and their recorded numbers stay comparable. Result:
  `crossOriginIsolated: true`, 6 threads.

### WebGPU was proven active, not assumed

`wllama.isSupportWebGPU()` is worthless as proof — it is literally
`!!navigator.gpu` (`node_modules/@wllama/wllama/src/utils.ts:267`). Instead the
page captures llama.cpp's own native stdout through `WllamaConfig.logger` and
greps it. All six wllama halves reported `backend=webgpu`,
`adapter_info: vendor: apple | architecture: metal-3`,
`crossOriginIsolated: true`, 6 threads, `contentAccountingOk: true`, 0
reasoning chunks. Layers offloaded: Phi-3 **33/33** (2228.82 MiB WebGPU model
buffer), Qwen3 **37/37** (2375.91 MiB), Qwen3.5 **33/33** (2603.51 MiB).
`adapter_info` is only emitted by `ggml-webgpu.cpp` after both a `GPUAdapter`
and a `GPUDevice` were acquired.

Falsification control (`?ngl=0`, wired in for exactly this), same session,
Phi-3:

| | total | decode | TTFT | detected backend | layers |
|---|---:|---:|---:|---|---|
| default (`n_gpu_layers=99999`) | **24.62** | 26.16 | 321 ms | `webgpu` | 33/33 |
| `?ngl=0` | **8.29** | 9.71 | 2199 ms | `wasm-cpu` | 0/33 |

3.0× wall-clock / 2.7× decode. The GPU is doing the work; the log line is not
cosmetic. `window.wllamaResult.backend` / `.webgpuProven` / `.adapterInfo` /
`.layersOnGpu` carry this to the harness, and `bench/run.mjs` prints
`*** WEBGPU NOT PROVEN ***` if it ever flips.

### Sanity checks

- **Output is coherent on all three models**, not garbage — e.g. Qwen3.5:
  *"Photosynthesis is the process by which green plants, algae, and some
  bacteria convert light energy into chemical energy…"*
- **Token counts vs the 120 target:** Phi-3 119, Qwen3 101, Qwen3.5 95 — all
  `finish_reason: stop`, none truncated.
- **No wall-clock/self-report disagreement remains.** wllama's own
  `predicted_per_second` sits 3–17% above the page's decode figure (26.61 vs
  26.16; 22.23 vs 21.59; 19.59 vs 19.18), as expected: llama.cpp times decode
  only, the page's window includes per-token JS callback overhead.
- **Independent replication** in a fresh session with a separate driver script
  agreed within 2.5%: 24.00 / 19.13 / 16.43.
- **Weights verified byte-exact** against HuggingFace `content-length` with
  `GGUF` magic present — `phi3-q4.gguf` 2,393,231,072 B; `qwen3-4b-q4km.gguf`
  2,497,280,256 B; `qwen35-4b-q4km.gguf` 2,740,937,888 B.

### What a true same-bytes race would require — not done

To turn this into the same kind of A/B the WebLLM baseline is, **Zero-TVM
would have to read GGUF directly**: a GGUF container parser (header, KV
metadata, tensor directory), a `Q4_K_M` dequant kernel matching llama.cpp's
super-block layout (256-element blocks, 6-bit sub-scales/mins, per-block f16
scale — structurally unlike q4f16_1's flat group-wise scheme), and a name
mapping from GGUF tensor names to this engine's parameter scheme. That is a
separate project and **it has not been done.** Until it is, no number in this
section is a clean runtime comparison, and this section must not be quoted as
one.

The cheaper half-step, also not done: run WebLLM and Zero-TVM against a
q4f16_1 build while running wllama against a GGUF quantization chosen to match
bits-per-weight as closely as possible, and report the bpw of each. That would
bound the confound without removing it.

### Reproducing

```bash
export PUPPETEER_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BENCH_BASELINE=wllama npm run bench                              # Phi-3 pair
BENCH_BASELINE=wllama BENCH_QUERY="?model=qwen3"  npm run bench  # Qwen3-4B pair
BENCH_BASELINE=wllama BENCH_QUERY="?model=qwen35" npm run bench  # Qwen3.5-4B pair
```

GGUF files are served from `.weights-local/gguf/` by the `/local-gguf/*` dev
middleware in `vite.config.ts` — fully offline, no HuggingFace fetch. The page
runs directly at `/wllama-bench.html?model=phi3|qwen3|qwen35`, with `&ngl=0`
for the CPU falsification control.

`BENCH_BASELINE=wllama` **deliberately does not write `bench/results.json`**
and skips `sync-docs.mjs`. That file's schema field is `webllmDecode`, which
`sync-docs.mjs` propagates into the published headline numbers; a GGUF figure
landing there unlabelled is precisely the confusion this section exists to
prevent. The pairs are printed to stdout and recorded in
`bench/results/*-wllama.json` instead.

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

Acceptance stays under 8% in the best case — nowhere near the 25–33% floor
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
npm run bench   # Zero-TVM vs WebLLM, identical local q4f16_1 weights,
                # 128 tok × 5 runs; writes bench/results.json and syncs
                # the marked numbers

BENCH_BASELINE=wllama npm run bench   # Zero-TVM vs llama.cpp/wllama (GGUF).
                # NOT same-bytes — see "Third baseline" above. Prints the pair,
                # never writes results.json, never runs sync-docs.
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

---

## KV paging feasibility (2026-08-10, Apple M2 Max, 32 GB) — §0.3 of `docs/PAGING_PLAN.md`

Not a throughput benchmark. Four numbers the paging plan requires **before** any
threshold is written into code, so that "cold restore takes seconds, not
minutes" stops being a guess. Reproduce with:

```bash
node scripts/paging-measure.mjs --full     # 625 MiB, 3 runs each
node scripts/paging-measure.mjs --size 128 # smaller working set
```

625 MiB is Qwen3.5's KV for a 20k-token prefix (32 KiB/token, per
`tests/unit/kv-budget.test.ts`).

### (a) Adapter limits — previously unrecorded

| limit | value |
|---|---|
| `maxStorageBufferBindingSize` | **4096 MiB** |
| `maxBufferSize` | **4096 MiB** |
| adapter | `apple / metal-3`, subgroups available |

Consequence for Phase 3: a 20k Qwen3.5 prefix is ~78 MiB per attention layer
(625 MiB over 8 attention layers). Two resident sequences per layer fit inside
the existing per-layer buffer with three orders of magnitude to spare. The
"does Phase 3 need split bindings" question is answered: no.

### (b)–(d) The four transfer rates, 625 MiB, three runs each

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| OPFS write (`createSyncAccessHandle`, worker) | 7692 | 7821 | 7710 |
| OPFS read | 8223 | 7576 | 8669 |
| GPU→CPU readback, 16 MiB chunks | 4360 | 5800 | 5764 |
| GPU→CPU readback, 64 MiB chunks | 6805 | 8141 | 8275 |
| GPU→CPU readback, one 625 MiB chunk | 3166 | 7550 | 8856 |
| CPU→GPU `writeBuffer` | 2416 | 4228 | 6660 |

All MB/s. **64 MiB is the readback sweet spot** — 16 MiB chunks cost ~30%, and
one whole-buffer copy is the most variable. `peer-weights`' 240 KB piece size is
tuned for a DataChannel and is the wrong unit here.

**`writeBuffer` is the slowest link**, not the disk — which inverts the
assumption the plan was written under.

### What it means, from the SLOWEST run of each

| operation | cost for 625 MiB |
|---|---|
| save = GPU readback + OPFS write | **0.29 s** |
| restore = OPFS read + `writeBuffer` | **0.36 s** |
| the threshold it had to beat (1/5 of a ~99 s cold re-prefill) | 20 s |

**56× under the threshold.** The conclusion survives these numbers being an
order of magnitude optimistic — and they may well be.

### Caveats, and they are the point

1. **The OPFS numbers are measured through the page cache.** 7.7 GB/s exceeds
   what this SSD sustains; `h.flush()` does not guarantee a write to media on
   macOS, and 625 MiB fits trivially in 32 GB of RAM. Treat them as an upper
   bound. A real restore in a fresh tab is a *cold* read; this is warm.
2. **Apple Silicon is unified memory.** Readback and `writeBuffer` are memcpy,
   with no PCIe hop. Neither number transfers to a discrete GPU, where restore
   would be bounded by the bus.
3. **`maxContext` correction.** An earlier draft of this section said
   Qwen3.5's `maxContext` is 7168. It is **32768** (`maxPages: 2048`,
   `pageSize: 16`); 7168 belongs to `QWEN3_4B`. The 625 MiB reference workload
   is a 20k prefix, which Qwen3.5 *can* hold — its full cache at 32k is
   1.00 GiB. Nothing else in this section changes.
4. **Rates climb across repetitions** (`writeBuffer`: 2416 → 4228 → 6660).
   That is warming, not the device getting faster. Every figure above is
   reported per-run for that reason, and the extrapolation deliberately uses
   the slowest.

This is why the margin matters more than the number: at 56×, none of the three
caveats changes the decision. They would each have to be wrong by more than an
order of magnitude, together, to make cold restore not worth shipping.

### One claim checked and withdrawn

An earlier note in `scripts/paging-measure.mjs` said the engine's KV buffers do
not carry `COPY_SRC` and that Phase 1 would need an allocation change. False:
`engine-core.ts:36` defines `STORAGE = STORAGE | COPY_SRC | COPY_DST`, and every
KV buffer is allocated through it. Phase 1 needs no allocation change at all.

---

## 3-bit experts: the number the decision was made on (2026-08-10)

`index.html` promises that *"every claim — including the withdrawn ones — is in
BENCH.md."* The 3-bit-expert justification was not. It is now.

**The claim.** `qwen36q3` ships 3-bit expert stacks instead of 4-bit, chosen on
a **block-output cosine of 0.936** against the 4-bit block. 2-bit scored ~0.79
and was not offered.

**Three defects, all confirmed 2026-08-10:**

1. **It is a fidelity number, not a quality one.** It measures how closely a
   3-bit block reproduces the *4-bit block's* output on one input. That cannot
   see whether the resulting model is any good. This repo already has the
   counterexample in its own history: `model-spec.ts` records that CLS pooling
   on the embedding model has the **highest** gold cosine of any variant
   (0.9379) and gets **0 of 6** retrieval queries right.
2. **It is not reproducible.** No committed script computes it. `git show
   --stat a69d749` lists ten files and no harness that would produce it.
3. **It disagrees with itself.** The 2-bit figure is `0.79` in
   `model-spec.ts` and `0.785` in `index.html` and `CHANGELOG.md`.

**What would settle it** is `scripts/quality-ab.py`: perplexity of the 3-bit
build against the 4-bit build over identical windows, with error bars and a z
score. Both checkpoints are on disk (19.7 GB and 15.7 GB), it needs no
download, and it takes ~20 minutes on a quiet machine.

**It has not been run.** An attempt hit 23.3 GB of 23.5 GB swap in
uninterruptible wait. `docs/colab/quality-ab.ipynb` is the way around that —
MLX's CUDA backend runs `mlx_lm` on an A100, the 4-bit build is public so
Colab downloads it directly, and the 3-bit build is regenerated there by the
repo's own converter, so nothing is uploaded. **Treat 3-bit-vs-4-bit quality on this model as
unknown**, not as measured — the 0.936 does not carry the weight the surrounding
prose gave it.

The harness is validated on four builds, 24 independent windows of 512 tokens,
bf16, ~10 s per arm:

| baseline | candidate | ppl | vs base | paired z | B worse on |
|---|---|---:|---:|---:|---:|
| Llama-3.2-1B-4bit | 4-bit (itself) | 109.6 | — | — | — |
| Llama-3.2-1B-4bit | MLP-only 3-bit | 193.6 | +75.8% | 35.5 | 24/24 |
| Llama-3.2-1B-4bit | all 3-bit | 322.9 | +193.7% | 31.8 | 24/24 |
| **OLMoE-1B-7B-4bit** | **experts-only 3-bit** | **52.3** | **+8.4%** | **14.7** | **24/24** |

**The OLMoE row is the closest evidence that exists for `qwen36q3`.** 64
experts, top-8, only the expert stacks requantized — the same intervention, on
a 3.6 GB model. Expert-only 3-bit costs **+8.4%**, about an order of magnitude
less than dense 3-bit on a comparable model, which is the sparsity working as
intended: 8 experts of 64 fire per token, and attention, router and embeddings
stay at 4 bits.

**It is an analogue, not a prediction.** OLMoE is 64 experts / 1B active;
Qwen3.6-35B-A3B is 256 experts / 3B active. This says expert-only 3-bit is a
mild intervention on *a* real MoE and that the harness resolves an 8% effect.
It does not give `qwen36q3` a number.

---

## Chunked prefill for plain-attention + MLX-affine specs (2026-08-11, M2 Max)

Until now chunked prefill was hybrid-and-MLC-only (`qwen35`), and the gate's
`!AFFINE` arm was not a design decision but a missing kernel. Landed:
`int4_matmul_batched_dyn_affine` (w = s·q + b, group 64, pinned vs CPU at
4.70e-4 at engine scale N=256/M=31/CAP=64), the plain-attention chunk branch
(no gatedQkvSplit/attnGate; `cAttnDim` collapses to `qkvDim`), and the affine
embedding in the chunk path.

Same-page A/B (`scripts/chunk-prefill-test.mjs`, 800-token prompt + 2 decode,
one run per arm — **not a protocol round**): tokens are identical between the
arms in every configuration, 26 chunks per run.

| model | config | per-token | chunked | speedup |
|---|---|---:|---:|---:|
| qwen3mlx (Qwen3-4B MLX) | shipped (sg+tiled+splitK) | 11.48 s | **3.53 s** | **3.26×** |
| qwen3mlx | scalar | 24.24 s | 3.60 s | 6.73× |
| llama32 (Llama-3.2-1B) | shipped | 3.61 s | **1.01 s** | **3.57×** |
| llama32 | scalar | 9.25 s | 1.06 s | 8.77× |

MoE stays per-token (`ids[]` has no token dimension — one token's expert choice
would apply to the whole chunk).

**The bug this shipped over.** With the gate first opened, qwen3mlx diverged
from per-token at the FIRST generated token: the chunk path bound `P.embedding`
unconditionally, dequantizing MLX-affine embeddings with the symmetric formula
(no bias). One line. The equivalence test now also refuses to pass a run in
which zero chunks executed — its first version reported PASS at short prompt
lengths where `CHUNK_MIN` kept chunking off in both arms and it compared the
per-token path to itself.

---

### MoE: the experts are the MINORITY of the work (2026-08-13)

`scripts/decode-profile-native.mjs qwen30b`, LM Studio idle (a first run with
it generating read 12.12 ms vs 12.06 idle, so contention was not material).
MoE prefills per token today, so a decode profile IS a prefill profile here.

| kernel | share | expert-bound? |
|---|---:|---|
| moeUp | 13.6% | yes |
| moeGate | 13.0% | yes |
| qkvMatmul | 12.5% | no |
| moeDown | 11.4% | yes |
| oproj | 10.3% | no |

The three expert matmuls total **~38%**. The other ~62% — projections,
attention, norms, the router's own matmul, the combine — is the same dense
chain every other spec already chunks.

This inverts the plan. The stated blocker ("ids[] has no token dimension") is
real and does prevent batching the expert matmuls, but it does not touch the
rest of the layer, and the rest of the layer is the majority. Chunking
everything EXCEPT the experts needs no new kernel, no permutation and no
grouped GEMM — only a token dimension on the router's two outputs. Plan in
docs/MOE_CHUNK_PLAN.md.

### Where a decode token's time goes (2026-08-13)

`scripts/decode-profile-native.mjs qwen3mlx` — one step, 36 layers, 14 distinct
kernels. Decode is the one axis still behind LM Studio, and prefill only moved
because it was measured before it was touched; this is the same first step.

| kernel | share | cum |
|---|---:|---:|
| ffnGateUp | 27.7% | 27.7% |
| ffnDown | 26.2% | 53.9% |
| attention | 7.3% | 61.3% |
| addNorm2 | 7.3% | 68.6% |
| addNorm1 | 6.8% | 75.4% |
| qkvMatmul | 6.3% | 81.7% |
| oproj | 6.3% | 88.0% |
| lmHead | 5.2% | 93.2% |

**The FFN is the decode budget** — two kernels, 54%. Nothing else is close, and
the tail below lmHead is rounding.

Two caveats, both load-bearing:

- `profileStep()` requires `timestamp-query`, and requesting that feature costs
  ~3x decode on its own (see above). Everything here is SERIALIZED, so the
  totals are inflated and are not a decode rate, and kernels that normally
  overlap are over-counted. The SHARES are the signal.
- Serialization is also why the two norms read 14% combined: 36 dispatches of
  an elementwise pass over d=2560 is ~25 us each, which is mostly launch
  latency that would otherwise hide. Read it as "dispatch count costs
  something", not "the norms are slow".

This model runs the MLX-affine path, which is **unfused** — `fused_ffn` and
`qkv_fused` are symmetric-only, so an affine spec pays gate_up + silu_mul +
down as three dispatches where a symmetric one pays fewer. Two levers follow,
in order: the weight-read efficiency of the two FFN matmuls (decode is
memory-bound, so this is bytes/second against the M2 Max's ceiling), and an
affine `fused_ffn` to cut the dispatch count.

## Head-to-head vs LM Studio (2026-08-13) — SAME CHECKPOINT BYTES, native host

The 08-12 comparison below ran different bytes on each side (ours MLC q4f16_1,
theirs MLX 4-bit) and drove our BROWSER. This one removes both differences.

LM Studio has no `Qwen3-4B-4bit`, and it reads its own store — but symlinking
`.weights-local/Qwen3-4B-4bit` into `~/.lmstudio/models/mlx-community/` makes it
load the files we already have. Verified, not assumed: **2,278,972,183 bytes on
both sides.** The only difference left is the runtime.

`MODEL=qwen3mlx GEMM=e5 LMS_URL=http://<addr>:1234 node --experimental-strip-types
scripts/lmstudio-ab.mjs --native` — ~980-token prompt, 128 decode tokens, 3
interleaved rounds, medians; three separate processes.

| | zero-tvm (dawn.node, E5) | LM Studio (MLX) | ratio |
|---|---:|---:|---:|
| prefill tok/s | 497 / 467 / 458 | 427 / 423 / 411 | **1.10-1.16x us** |
| decode tok/s | 71.2 / 65.4 / 62.9 | 78.7 / 76.2 / 74.0 | 0.85-0.90x |

**Prefill is won.** The 0.17x -> 0.35x arc in this file was measured against
LM Studio's 1,385 tok/s on Qwen3.5-4B; on the same bytes and the same model,
our chunked prefill with E5 is ahead. Decode remains the gap, unchanged in
character at ~0.85x.

Three defects had to be fixed before any of this meant anything, and each one
printed a plausible-looking number first:

- **Our prefix cache served rounds 2..N**, reporting 16,443 tok/s "prefill".
  Fixed with a distinct prompt per round; the harness now throws when a round
  reuses more than 5% of its ids.
- **THEIR cache served the whole second run** — 3,751 tok/s against our cold
  491 — because LM Studio's server outlives our process and had seen those
  prompts already. Our guard could not see it: it only watched our side. The
  prompt now carries a per-process nonce, so neither engine can be warm.
- **LM Studio ignored both thinking switches.** `reasoning_effort: none` and
  `chat_template_kwargs.enable_thinking` were accepted and had no effect — all
  128 tokens went to reasoning, content was empty, and the ttft arithmetic
  produced a NEGATIVE prefill rate that printed as a result. `/no_think` in the
  message body is what works on this checkpoint. The harness now refuses a
  response with no content deltas.

Caveats that remain: Qwen3-4B dense, not the Qwen3.5-4B hybrid of the run
below; LM Studio held its own 9 GB model resident alongside, and our decode
(63-73) sits under the ~84 tok/s measured solo, so contention is likely on both
sides; LM Studio ran ctx 8192, parallel 4. Ratios are same-run and interleaved,
which is what they are for.

Also worth knowing: LM Studio may bind ONE non-loopback interface (here a
Tailscale address) with "serve on local network" on, and `127.0.0.1:1234` is
then refused outright while `lms server status` still says running. Find it with
`lsof -nP -iTCP:1234 -sTCP:LISTEN`.

## Head-to-head vs LM Studio (2026-08-12, M2 Max 32 GB) — same machine, same session, interleaved

The first cross-runtime A/B in this file that is not against another browser
stack. Qwen3.5-4B both sides — **different checkpoints** (ours MLC q4f16_1,
theirs MLX 4-bit), same architecture and size, identical ~1k-token prompt of
this repo's own docs, thinking off, temp 0, 3 interleaved runs after a warm-up,
`scripts/lmstudio-ab.mjs`.

| | zero-tvm (browser WebGPU) | LM Studio (MLX native) | ratio |
|---|---:|---:|---:|
| **context (tokens)** | **262,144** (`?ctx=`, native max) | 198,400 (fitted) | **1.32× us** |
| decode tok/s | 72.4 | 89.0 | 0.81× |
| prefill tok/s | 231 | 1,385 | 0.17× |

Raw runs — ours decode: 72.4 / 72.4 / 72.4; theirs: 89.0 / 85.0 / 102.0†.
Ours prefill: 212–234; theirs: 1,381–1,433.
† the 102.0 was the first measured round; a later round settled at 85–89.

**Context is won, and not narrowly.** Our KV cost/token on this model is ~3×
lower than LM Studio's per-token working set (32 KiB vs ~101 KiB — KV on only
8 of 32 layers), so the model's own 262,144 native window fits in 8 GiB of KV
where their formula caps at 198,400. Verified live: boot at `?ctx=262144`,
generate, tokens identical to the default engine, one step at position
262,143 (the last slot) with zero GPU errors — an addressing check, not a
long-range-quality claim. The old 32,768 ceiling was a flat ~1 GiB budget
constant from one commit in July, not a limit.

**Decode is close (0.81×)** for hand-written WGSL in a browser against
native MLX — and the first version of this A/B read 30.5 tok/s (0.30×)
because the smoke page's deliberately-plain engine was timed instead of the
chat composition. The variant flags are 2.4× of decode; always time the
config users run.

**Prefill is decisively lost (0.17×), and the sweep says why.** CHUNK_CAP
64→512 moves qwen3mlx only 3.4% (3.52→3.40 s) and makes qwen35 *worse*
(3.32→3.93 s), so per-chunk overhead is not the cost — the batched GEMM
itself is. `int4_matmul_batched_dyn` is a 4-row × 4-batch subgroup matvec,
not a tiled GEMM; at M=64 it re-reads activations per row-block and cannot
touch MLX's Metal GEMMs. Closing this needs a real tiled (workgroup-memory)
batched GEMM — the single highest-value kernel left in the repo.

A note my own records force: an earlier measurement had LM Studio prefill at
465 tok/s — that number came from agentic-loop server logs (uncached tokens ÷
prefill seconds under real load), not a clean benchmark, and today's clean
1,385 supersedes it for hardware-capability claims.

### Tiled batched GEMM (2026-08-12): correct, NOT yet default — no clean measurement exists

`int4_matmul_tiled_m(_affine)` landed: 32×32 output tile, activations staged in
workgroup memory, no subgroups required, same bindings/uniforms as batched_dyn.
Correct at every ragged edge (M=47/CAP=64 → two y-tiles, N=200 → ragged x-tile;
4.84e-4 / 8.42e-4 vs CPU, M-invariant, no writes past M) and token-identical
in-engine on llama32 and qwen35.

**It is opt-in (`chunkTiled` / `?tiled=1`), because no honest speed number
exists.** Every timing on landing day was noise-dominated — the machine had
just recovered from a memory freeze (a 19.5 GB LM Studio model loaded during
GPU tests) and the per-token baseline drifted 11.5 s → 38.7 s within hours.
The clean-ish runs read parity with the matvec, not a win: at M ≤ 64, L2
evidently covers most of the matvec's activation re-reads. A quiet-machine
same-session A/B (`CAP=64` tiled vs `CAP=68` matvec) decides the default;
until then the claim is exactly "correct and available".

### GEMM microbench (2026-08-12, `scripts/gemm-bench.mjs`) — the scoreboard that replaced guessing

Isolated kernel timing on qwen3mlx's real chunk shapes, one Chrome boot, no
model load, correctness-gated (a kernel that fails a from-the-formula CPU
check gets its timings marked VOID — the gate fired twice today and both
catches were real). Median-of-20, warm.

| M=256 | matvec | tiled-v1 | **tiled-v2** | sgmat (experimental) |
|---|---:|---:|---:|---:|
| gate_up 2560→19456 | 1901 GF | 1689 | **2209** | 1851 |
| ffn_down 9728→2560 | 1658 | 1697 | **2189** | 1864 |
| o_proj 4096→2560 | 1610 | 1802 | **2256** | 1959 |

At M=64: v2 ≈ matvec (the L2 covers the matvec's re-reads at small M).

**tiled-v2** (dequantized weights staged in shared f32, bias folded into the
stage, pure-FMA inner loop) is the winner and now backs the `chunkTiled`
pipelines: +15–35% over the matvec at M=256, correct at 4.7e-4. **tiled-v1
lost to the matvec it was meant to replace** — both amortize dequant over 4
rows, so the roofline said parity and parity is what measured; it stays in
gen.ts for the bench only.

**`chromium-experimental-subgroup-matrix` EXISTS on this machine and the
kernel runs** — Metal's simdgroup_float8x8 from WGSL, the primitive MLX's
GEMMs are built on. First cut: compiles, executes, 1851–1959 GF (below v2 —
naive fragment scheduling), and 3.4e-2 vs the CPU gate because f16 fragments
round the staged weights, which is MLX-class arithmetic but fails this
repo's strict gate. It is THE path to LM Studio-class prefill (their ~7–9 TF
effective); needs fragment-layout tuning plus an explicit precision policy
before it can be more than an experiment. Recorded, not shipped.

Context for the ceiling: hand-written WGSL FMA tops out ~2.2–2.4 TF here;
the remaining gap to MLX is the matrix unit, reachable only through the
experimental feature above.

### Subgroup-matrix GEMM graduated (2026-08-12): `int4_matmul_sgmat(_affine)`

The winning variant from the bench iterations — one subgroup per workgroup
(the validator requires WORKGROUP-uniform fragment offsets, so multi-subgroup
tiles are impossible), f16 fragments with f32 accumulate, A loaded straight
from storage (measured faster than staging it), W dequantized into shared
with the bias folded. Same run, same throttled thermal state, ratios only:

| vs (M=256) | tiled-v2 | shipped matvec (M=64) |
|---|---:|---:|
| gate_up | 1.15× | — |
| ffn_down | 1.29× | 1.5× |
| o_proj | 1.38× | 1.37× |
| gate_up (M=64) | 1.37× | 2.2× |

**f32 fragments were tried and are the wrong answer:** correct (4.7e-4) but
~5× slower (~270 GF) — the matrix unit's fast path is f16, so the precision
question is unavoidable: f16 fragments round the staged weights (3.4e-2 vs a
from-the-formula reference under cancellation). That is the same arithmetic
MLX runs on the same unit, but it is NOT bit-comparable to the per-token
path, so the kernel is graduated into gen.ts and the bench — **not wired
into the engine** — until the chunked-vs-per-token equivalence policy for
f16-fragment arithmetic is decided and a quiet-machine run measures it
in-engine. The feature itself is experimental Chrome
(`chromium-experimental-subgroup-matrix`), so the engine must feature-detect
regardless.

### sgmat wired and DEFAULT where the hardware allows (2026-08-12, later)

The precision policy resolved itself on precedent: chunked prefill was never
bit-equal to per-token (`checkReuse` needs `?chunk=0` for exactly this
reason) — every chunk kernel holds an EMPIRICAL token-identity bar, and sgmat
now passes it on every chunking spec:

```
llama32   tokens identical   gemm sgmat   chunked 1.02s
qwen3mlx  tokens identical   gemm sgmat   chunked 3.24s   (matvec, same
qwen35    tokens identical   gemm sgmat   chunked 3.10s    sequence: 4.95s)
```

GEMM ladder in the engine: **sgmat** (when the device was created with
`chromium-experimental-subgroup-matrix`) → tiled-v2 (`chunkGemm:'tiled'`) →
matvec. chat and agent-host request the feature when the adapter offers it;
on any other browser the ladder degrades silently. The in-sequence matvec
pair (3.24 vs 4.95 s) is thermally soft — the microbench's same-run 2.2×
GEMM ratio is the number to trust; 3.24 s is also the best qwen3mlx chunked
prefill ever recorded here, on a warm machine.

### CAP sweep under sgmat (2026-08-12): M-scaling is real once the kernel can use it

The 2026-08-11 sweep said cap didn't matter — measured with the matvec, which
cannot exploit M beyond its 4-row block. Under the matrix-unit GEMM
(qwen3mlx, 800-token prompt, tokens identical at every point):

| cap | 64 | 256 | 512 |
|---|---:|---:|---:|
| chunked prefill | 3.26 s | **2.80 s** | 2.70 s |

Default is now **256 when sgmat is active** (512's extra 3.6% is inside
thermal noise and doubles the activation buffers), 64 for the FMA kernels.
Re-gated at the new default: llama32 **0.84 s**, qwen35 **2.79 s** — best
prefill numbers ever recorded in this repo, tokens identical.

Running ratio vs LM Studio's 1,385 tok/s clean prefill: ours is now ~287
tok/s on the same-family model — **0.17× → ~0.21×**. Still ~5× behind;
the next levers are sgmat tile tuning and larger caps on long prompts.

### E1 lands (2026-08-12, evening): the research plan's first experiment, delivered

`docs/PREFILL_RESEARCH.md` (4-angle survey of ORT/llama.cpp/MLX/the field)
overturned two "facts": the workgroup-uniform-offset wall was Tint's
conservative diagnostic (`diagnostic(off, chromium.subgroup_matrix_uniformity)`
— ORT and llama.cpp both disable it), and multi-subgroup workgroups are THE
convergent shape. E1 = 128 threads / 4 subgroups, 32x64 tile, TILE_K 32,
stride-40 staging, 8 named accumulators, vectorized nibble dequant.

Microbench, same run, correctness-gated: **2.8–3.8 TF** (previous best
2.3–2.6) — past the 3.5 TF plain-WGSL bar. llama.cpp's 256-thread/8-subgroup
config was tried verbatim and lost on this hardware (2.1–3.0 TF).

In-engine, clean thermal window, identity-gated on all three chunking specs:

| | prefill (800 tok) | tok/s | this morning |
|---|---:|---:|---:|
| llama32 | **0.66 s** | 1,212 | 3.61 s (per-token) |
| qwen3mlx | **2.03 s** | 394 | 3.53 s |
| qwen35 | **2.13 s** | 376 | 3.32 s |

Running ratio vs LM Studio's clean 1,385 tok/s: **0.17× → ~0.27×** in one
day. Remaining arms: direct store, TILE_K/tile sweep, swizzle (E3/E5), the
robustness-tax measurement (E2), split-K for small dispatches (E4). The
research's ceiling estimate stands: 0.70–0.75× native ≈ 620–810 tok/s.

### E2: the Chrome tax, measured and (locally) reclaimed (2026-08-12, late)

`--enable-dawn-features=disable_robustness` — Chrome's mandatory GPU
bounds-checking, off. Kernel-level (same run): sg-e1 3,832 → 4,839 GF; and
llama.cpp's 8-subgroup config FLIPS to the winner at **5,095 GF = 5.1 TF**,
inside the research's 5.0–6.5 TF native-ceiling band. End-to-end qwen3mlx
prefill: 2.03 → **1.66 s** (394 → **482 tok/s**), tokens identical.

Not web-shippable — but `ztvm` launches its own Chrome, so the LOCAL agent
surface claims it by default (separate profile that only ever loads our
localhost page; `--safe` opts out). zerotvm.com stays on stock Chrome rules.

Running prefill ratio vs LM Studio's 1,385: 0.17× (morning) → 0.27× → **0.35×
(local agent)**. Next known steps: graduate the 8-subgroup config as the
unsafe-mode kernel (5.1 TF vs e1's 4.8), then E3/E5 sweep arms.

### Dawn-native probe: GO (2026-08-12)

The `webgpu` npm package (official dawn.node prebuilt, dawn-gpu/node-webgpu)
passes every gate the two earlier non-browser attempts failed:

| | @kmamal/gpu (old) | Deno/wgpu | **dawn.node prebuilt** |
|---|---|---|---|
| empty submit-and-wait | ~100 ms | 13 ms (pipelines away) | **0.14 ms** |
| subgroups / f16 | yes | no subgroups | **yes** |
| subgroup-matrix | no | no | **YES** |
| disable_robustness | — | — | **accepted** |
| sg-e1 gate_up M=256 | — | — | **4,833 GF = unsafe-Chrome parity** |

`scripts/dawn-probe.mjs` reproduces this in ~30 s. Consequence: the native
host (PREFILL_RESEARCH L2) is unblocked — same WGSL kernels, no browser
process, no tab throttling, robustness off by default. Remaining work is the
engine shims (navigator.gpu global, OPFS→fs weights, the `build:lib` bundle)
— the 1-2 day estimate stands, with the riskiest unknown now retired.

### The native host ships (2026-08-12): `ztvm native` — no browser at all

`scripts/agent-native.mjs`: the OpenAI server with the engine IN-PROCESS on
dawn.node (the `webgpu` prebuilt), robustness off, same wire contract as the
browser host. Three shims made the whole browser engine run unmodified
(`scripts/native/shims.mjs`): WebGPU globals before any src import,
fs-backed OPFS under `~/.zerotvm/opfs` (weight cache AND the KV pool
persist), and a fetch rewrite that sends HF resolve URLs to the local vite
mirror — added after the first boot silently pulled from huggingface.co at
0.9 MB/s for ten minutes (Node has no `import.meta.env.DEV`, so
weight-loader skipped its mirror).

| | native (dawn.node) | flagged Chrome | note |
|---|---:|---:|---|
| boot to ready | 2.6–6 s | ~8 s + a window | |
| llama32 prefill | **1,523 tok/s** | 1,212 | +26% |
| qwen3-4B prefill | 467 tok/s | 482 | parity, first-request |
| tool round trip | ✓ | ✓ | same contract |

The lib grew `createEngineRaw` (+ ctx override, + subgroup-matrix feature)
and a LAZY `hostSurface()` — lazy because a static re-export would evaluate
weight-loader's module-scope `GPUBufferUsage` and crash every non-GPU import
of the library, the exact failure its lazy-import design exists to prevent.

### The native decode mystery (2026-08-13): three suspects, one guilty — timestamp-query

The first native DECODE measurement read qwen35 at **19.5 tok/s** against
72.4 in the browser. Over one morning that number was blamed on three
different causes; two were retracted the same day. The record, in order:

1. **RETRACTED: "dawn.node backs off completion delivery when the loop
   idles."** A hot `setImmediate` chain appeared to lift 19.5 → 69.7 tok/s,
   and a hot-loop fix shipped on that basis. The intervention was confounded:
   the hot run also happened to be the first after a lib rebuild that added
   timestamp-query (see 3), on a machine whose battery was dying (see 2).
   The clean A/B — healthy power, no timestamp-query, idle/hot phases
   alternating IN ONE PROCESS (`scripts/decode-bench-native.mjs`) — shows
   **no idle/hot gap at all** (qwen3mlx 75-85 both ways; qwen35 65-77 both
   ways), and on llama32 the spinner is ~8% SLOWER (277-278 idle vs 250-259
   hot — the busy core steals from the submit loop). The hot-loop code is
   reverted everywhere.
2. **REAL but environmental: the battery.** The machine spent the morning at
   5% on an adapter that could not charge it ("AC attached; not charging"),
   and macOS hard-throttles the SoC in that state — decode fell to 7 tok/s
   with a cool, idle GPU. Every absolute number taken in that state was
   VOID. Check `pmset -g batt` before blaming thermals or code.
3. **REAL and in the code: requesting `timestamp-query` at device creation
   costs ~3× decode** — with no query ever taken. qwen3mlx: a uniform
   ~19 tok/s with the feature on the device, 54-84 without, A/B'd minutes
   apart on the same machine. Dawn's Metal timestamp path serializes
   command execution device-wide. The lib now requests it only under
   `ZTVM_PROFILE=1`; `engine.profileStep()` still works when asked for.

**The clean decode table** (healthy power, no timestamp-query, dawn.node,
robustness off; browser + LM Studio columns from the 08-12 A/B):

| model | native decode | browser (flagged) | LM Studio |
|---|---:|---:|---:|
| llama32 | **277 tok/s** | — | — |
| qwen3mlx | **~84 tok/s** | — | — |
| qwen35 | **~76 tok/s** | 72.4 | 89.0 |

Native now BEATS the browser on the A/B model (76 vs 72.4) and stands at
**0.85× LM Studio** decode — with context (1.32×) and cold-restore (~35×)
already won. Numbers taken at 20% charge on a working adapter; treat as
same-day-comparable, not protocol-final.

### Sweep round 2 (2026-08-13): WITHDRAWN — both winners were ranked against a handicapped baseline

Round 2 reported `sg4 32x64 k32 pad sk4` +8% at M=64 and `sg8 64x64 k32 swz`
+7% at M=256, both "over E1". Neither claim survives.

The sweep never timed the shipped E1. It timed its own parameterized
reconstruction of E1's config and quoted the real kernel from a **previous
run's header line** — a cross-run ratio, which this file rejects everywhere
else. Running the shipped generator as arm zero of the same process:
shipped E1 4.36 TF at M=256, the builder's supposedly-identical arm 3.46 TF.
**26% apart.** Against the real kernel, round 2's two winners are −7% and
−16%. They lost.

The gap was the builder's, not the kernel's: every staging loop carried a
range guard (`if (idx < TM*TK)`) that is provably always true for all but
one config, since the thread count divides the work evenly. An always-true
branch per staged element, in the two loops that feed the matrix unit.

It also **skewed the ranking**, not just the level: the tax scales with
elements staged per thread, so `64x32` tiles (16 A-elements/thread) paid
twice what `32x64` tiles (8) did. Emitting each guard only where the
division is ragged — one config, `sg8 128x32` — moved `sg4 64x32 k32 swz`
from near-last to first.

### Sweep round 3 (2026-08-13, 3 runs, AC power): the swizzle at a 64x32 tile beats E1

`sg4 64x32 k32 swz` — 4 subgroups, 64(M)x32(N) tile, TILE_K 32, llama.cpp
8x8-block dense B layout. E1's tile shape transposed, plus the swizzle.

| | shipped E1 | sg4 64x32 k32 swz | |
| --- | --- | --- | --- |
| M=256 mean | 4.37 / 4.26 / 4.37 | 5.15 / 5.19 / 5.16 | **+18 to +22%** |
| M=64 mean | 3.30 / 3.25 / 3.00 | 3.83 / 4.25 / 4.12 | **+16 to +37%** |

Three independent processes, same machine, AC power, no other GPU work.
M=256 is the trustworthy row — both kernels reproduce inside ±1.5% there,
so the ratio is solid. M=64 swings ±5% per kernel and the ratio with it;
read it as "wins, magnitude uncertain".

It wins every shape at M=256, not just the mean (gate_up 5.78 vs 4.83,
ffn_down 5.09 vs 4.26, o_proj 4.62 vs 4.00), so there is nothing to
per-shape switch between — one config replaces E1 outright.

Gate 4.6e-4, same band as every other arm.

Two harness bugs fixed alongside: the sweep never exited (dawn.node holds
the loop open after the last submit — one such process was found still
running overnight at 35% CPU, holding the GPU beside a new run), and its
stdout buffered invisibly when piped.

### E5 in the engine (2026-08-13): +13-15% prefill, opt-in

`sg4 64x32 k32 swz` is now generated by `int4MatmulSgE5WGSL` and selected
with `chunkGemm: 'e5'` (`?gemm=e5` on model-smoke). The hand-written
generator reproduces the sweep arm it came from: 5.21 vs 5.20 TF at M=256,
same run.

**The isolated win is not the engine win.** 20% on the kernel is 13-15% on
prefill, because a chunk dispatches this GEMM among attention, norms and
bind-group churn that did not get faster:

| | sgmat (E1) | e5 | |
| --- | --- | --- | --- |
| llama32 | 1566 tok/s | 1772 | **+13.1%** |
| qwen3mlx | 479 tok/s | 549 | **+14.7%** |

`scripts/prefill-gemm-ab.mjs` — two engines over one weight load, arms
alternated A,B,B,A within one process, 1024-token prompt, fresh ids per
round so no round can be served from another's prefix. Medians of 4;
llama32's spreads do not overlap (1525-1566 vs 1719-1773).

That script's first version measured **nothing**: it drove `forwardLogits`,
which prefills token by token and never enters the chunk path, and duly
reported 205 tok/s for both arms. It now refuses to report a round that did
not chunk. The same run also exposed a real bug — asking for `'e5'` dropped
CHUNK_CAP to 64, because the cap default tested `chunkGemm === 'sgmat'`.

Verified before shipping:

- Numerics, both quant flavors, ragged M/N/K (13x96x128): 4.7e-4 and 4.8e-4.
  The **symmetric** halves of E1 and E5 had never been graded anywhere — the
  sweep only ever built the MLX-affine flavor — so `gateSym` was added; no
  MLC checkpoint is on this disk, and that is the only place they can be
  checked. The gate also now allocates 64 activation rows, since a tm=64 tile
  stages 64 rows whatever M is and was previously reading past its buffer.
- Token identity vs per-token prefill: llama32 (150 and 600-token prompts,
  caps 64 and 256) and qwen3mlx (cap 256), 0 GPU errors.

**DEFAULT as of 2026-08-13.** The bar was token identity on every chunking
spec family, and qwen35 closed both open ones at once — it is MLC-symmetric
*and* hybrid GDN. Full set:

| spec | family | token-identical | in-engine prefill |
|---|---|---|---|
| llama32 | dense, MLX affine | yes (caps 64 and 256) | +13.1% |
| qwen3mlx | dense, MLX affine | yes | +14.7% |
| qwen35 | MLC symmetric + hybrid GDN | yes | +39.7% |

qwen35's arm is noisy — E1 145-219, e5 183-275 across rounds — but e5 wins
every PAIRED round, and pairing is what the interleaving is for. The GDN
recurrence adds variance the dense specs do not have.

E5 falls back to sgmat by itself when the chunk cap does not tile by 64, so
the ladder still holds on any spec that cannot take it.

### `ztvm native big` gate (2026-08-13): qwen35 @ 65,536 PASS

Native host boots qwen35 at ctx 65,536 in 6.2 s (warm cache), answers a
completion correctly (finish_reason stop, 42 tokens, 736 ms wall including
prefill), pool fingerprint-miss then saves 66 tokens on idle. The `ztvm big`
preset is safe to hand to pi/Cline against the native host.
