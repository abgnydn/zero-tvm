# zerotvm.com

## Goal

Browser-native LLM inference on hand-written WGSL kernel roles and
TypeScript — replacing the TVM-autotuned shaders WebLLM ships. The
pedagogical thesis: the entire LLM forward pass is readable end-to-end
in a single sitting.

Do NOT publish file, kernel or line counts. They have been wrong on every
surface that carried them, in both directions, and they rot on every commit.
"10 kernel roles" is the claim that does not.

Head-to-head vs WebLLM numbers live in `BENCH.md`.

## Architecture

Vite-built multi-page static site. Each HTML file is a standalone demo:

- `index.html` — the ENTRANCE: a full-screen character-select screen
  (`src/landing.ts` renders splash, stage, roster, sheet, deeds into
  `#model-browser`), nav above and footer below, nothing else. ENTER mounts
  the chat IN PLACE (`landing-chat.ts`) rather than navigating; the ⟁ Room
  tool hosts from there (`landing-room.ts` over `room-host.ts`). The hero,
  kernel window and carousel left with the 2026-08-15 redesign; docs.html
  carries the why.
- `zero-tvm.html` — the hand-written chat surface
- `docs.html`, `validate.html` — docs and the validation harness
- `share.html` — host a model for other devices; `agent-host.html` — the
  local agent-server surface (noindex, needs `npm run agent`)

DELETED 2026-08-14 (pre-publish review): `architecture.html`, `demo.html`,
`compiler-chat.html`, `dump.html`, `shaders.html`, `webllm-bench.html`. All
were orphaned yet in the sitemap; `dump`/`shaders`/`webllm-bench` started
multi-GB downloads on page load. Their src/ modules remain, unbuilt.
- `src/zero-tvm/` — the hand-written engine. There is ONE decode loop:
  `engine-core.ts`'s `buildDecodeEngine(device, weights, kv, opts)`,
  driven by BOTH pages. `validate.html` (via `loading-ui.ts`'s
  `bootEngine`) runs the default unfused/scalar config with blocking
  `generate`/`forwardLogits`; `zero-tvm.html`'s `chat.ts` is a thin
  page module that boots the fused config with URL-flag shader
  variants (`variants.ts`, e.g. `?sg=0`, `?matmul=`, `?kv8=1`) and
  streams via `generatePipelined`. Support modules: `tokenizer.ts`,
  `weight-loader.ts`, `spec-sim.ts`, `markdown.ts` (chat's renderer),
  `bench-console.ts` (window.bench/benchBatched/specSim harnesses).
- `src/compiler/` — despite the name, this is where OUR kernels live.
  `compiler.ts` is the hand-written pipeline builder: `compile()`
  creates every GPU compute pipeline (no codegen, no TVM); activation
  buffers and bind groups are owned by `engine-core.ts`.
  The engine is **spec-parameterized**: `model-spec.ts` defines
  `ModelSpec` (base dims + derived address math + weight naming +
  HF repo) with `PHI3` as the default everywhere and `QWEN3_4B` as a
  second spec for the GQA port. `shaderPrelude(spec)` injects the WGSL
  const block; `compile()`, `buildDecodeEngine()`, and `loadWeights()`
  all take an optional spec (default PHI3 — Phi-3 behavior unchanged).
  `npm run test:kernels:qwen` runs under QWEN3_4B dims — it compiles every
  shader AND asserts numerics (maxRel < 0.02), so it needs a GPU.
  `src/compiler/shaders/` holds the 37 WGSL files
  (10 roles + tiled/subgroup/int8/f32 variants; the int4_matmul generator
  emits 18 more). The WebLLM/TVM
  reference path is `chat-v2.ts` (the `compiler-chat` page), which is
  the only file here that imports `@mlc-ai/web-llm`.
- `src/shaders/` — 3 *unwired* "max-fusion" experiments
  (`fused-norm-matmul`, `fused-ffn`, `argmax`). Documented in
  `architecture.html` but NOT imported by the engine, which still uses
  the `src/compiler/shaders/` versions.
- `src/tvm-shaders/` — 85 dumped TVM shaders, for apples-to-apples
  inspection against `src/compiler/shaders/`.
- `src/webllm-bench/` — the head-to-head benchmarker (imports
  `@mlc-ai/web-llm`).
- `src/{engine,capture,ui}.ts` — an older engine generation +
  TVM-capture tooling. Not in the Vite build graph and not imported by
  the shipped pages (`capture.ts`/`dump-tvm.ts` back the RESEARCH.md
  interception work).
- `sites.json` — synced from `~/sites-shared/sites.ts` (consumed by the
  sibling-link renderer in the HTML pages).

`index.html` embeds the JSON-LD `Person` + `sameAs` block and the
"Related work" grid; the "More by Ahmet" footer is repeated (non-identical)
across ~5 pages. Update by hand until sites-shared HTML partials land.

## Commands

```bash
npm install
npm run dev          # Vite dev server
npm run build        # tsc && vite build → dist/
npm run preview      # preview built dist/
npm run typecheck    # tsc --noEmit
npm run test         # vitest run
npm run test:e2e     # puppeteer end-to-end
npm run check        # typecheck + test
```

Deploy: `npx wrangler pages deploy dist --project-name=zerotvm --branch=main` (CF Pages, project
`zerotvm`).

## Qwen3-4B (`?model=qwen3`)

Second architecture on the same engine (GQA 32/8, QK-norm, byte-level BPE,
tied lm_head, ChatML non-thinking). Phi-3 stays the default everywhere;
`model-select.ts` maps `?model=qwen3` → `QWEN3_4B`. The Qwen chat path
keeps the QKV matmul unfused (QK-norm is incompatible with the fused QKV
kernel) but since the 2026-07-29 tuning round runs the fused
`qk_norm_rope_append` kernel after it (**8 dispatches/layer**; `?fuseqk=0`
restores the 10-dispatch reference chain) and the `_vec4h` K%512 matmul
variants on d=2560 / ffn=9728 (`?vec4h=0` opts out). Still no int8-KV —
see BENCH.md's Qwen sections for the measured WebLLM gap.

```bash
node scripts/download-weights.mjs --model qwen3   # ~2.3 GB → .weights-local/
npm run dev                                       # then zero-tvm.html?model=qwen3
                                                  #  or validate.html?model=qwen3
npm run test:kernels:qwen                         # 21/21 compile + numerics (needs a GPU)
npm run test:e2e                                  # includes tests/e2e/qwen.test.ts —
                                                  # skips loudly if the mirror isn't primed
BENCH_QUERY="?model=qwen3" npm run bench          # same-session A/B vs WebLLM's
                                                  # prebuilt Qwen3-4B; never writes results.json
```

## Qwen3.5-4B hybrid (`?model=qwen35`)

Third model, first hybrid: 24 gated-DeltaNet layers + 8 gated-attention
layers (GQA 16/4, head_dim 256, partial RoPE 64/256, sigmoid attention
gate), 248k vocab, tied lm_head; `model-select.ts` maps `?model=qwen35` →
`QWEN35_4B`. The GDN decode kernels are **scalar** (no subgroup variants
yet). Since 2026-07-29 the four GDN input projections are **fused into one
12352-row int4 matmul** (loader packs qkv‖z‖a‖b at upload; downstream
kernels read regions via 256-aligned bind offsets — 340 dispatches/token),
and the blocking `generate()` is **incremental** (a `gdnStatePos` tracker
reuses the non-idempotent recurrent state when it matches `startPos`
instead of replaying the prompt). Prompt PREFILL on the chat page is
**chunked** (perf round A): tokens run in chunks of ≤64 — every projection
one `int4_matmul_batched_dyn` dispatch (runtime M), rope/kv_append/conv/
gates/norm batched, ONE `gdn_recur` dispatch per layer per chunk, causal
`attention_prefill` for the 8 attention layers; `?chunk=0` opts out
(engines without 32-lane subgroups fall back per-token automatically).
All engines also do **cross-turn prefix reuse** (`?reuse=0` opts out): the
engine tracks the exact absorbed (position, token) record and prefills only
the delta on the next turn — hybrid reuse requires the new prompt to extend
EVERY absorbed token (the ChatML non-thinking template re-renders past
assistant turns WITH the empty `<think>` block so it does). When that fails,
a ring of **four GDN state snapshots taken at chunk boundaries** lets the turn
replay from the nearest one at or below the divergence instead of prefilling
from zero (~0.19-0.24 GB of VRAM). Lookback is 4 x CHUNK_CAP, so ~4k tokens
where the matrix unit gives cap 1024 and only ~256 on a browser without it. Agent clients that
rewrite a trailing metadata block every turn hit this constantly: measured
392.50s → 12.76s. WebLLM A/B
needs `@mlc-ai/web-llm` ≥ 0.2.84 (Qwen3.5 first ships in the v0_2_84
prebuilt libs).

```bash
node scripts/download-weights.mjs --model qwen35  # ~2.6 GB → .weights-local/
npm run dev                                       # then zero-tvm.html?model=qwen35
                                                  #  or validate.html?model=qwen35
npm run test:kernels:qwen35                       # 19/19 GDN + chunked-prefill kernels vs CPU reference
npm run test:e2e                                  # includes tests/e2e/qwen35.test.ts —
                                                  # skips loudly if the mirror isn't primed
BENCH_QUERY="?model=qwen35" npm run bench         # same-session A/B vs WebLLM's
                                                  # prebuilt Qwen3.5-4B; never writes results.json
```

## Qwen3.6-35B-A3B (`?model=qwen36`, `?model=qwen36q3`)

Fourth model, first MoE and first MLX-format checkpoint: 40 layers (30 GDN +
10 gated-attention, GQA 16/2), 256 experts top-8 plus a shared expert, untied
lm_head, MLX-affine quantization (`w = s·q + b`, group 64, per-tensor biases).
`model-select.ts` maps `?model=qwen36` → `QWEN36_35B_A3B` (4-bit, 19.7 GB
resident, needs ~24 GB free RAM) and `?model=qwen36q3` → `QWEN36_35B_A3B_Q3`
(3-bit expert stacks, 15.7 GB resident, ~66 t/s on a quiet 32 GB M2 Max (65.56, protocol round 2026-08-13; supersedes the earlier ~55 single run)).

Engine notes: the MoE block is 7 dispatches (router_logits → router_topk →
gate/up/silu/down → combine) with the expert index in grid `z` and the shared
expert stacked as index E of every expert tensor (its gate is router row E) —
no special cases anywhere. Since 2026-08-13 MoE CHUNKS its prefill like every
other spec: `moeIds`/`moeScores` are `[chunk, slots]`, the router's two kernels
and `moe_combine` take the token in grid `y`, and the expert matmul takes it in
grid `y` with the slot still in `z`. Experts never mix tokens, so this is a
re-indexing, not a grouped GEMM — see `docs/MOE_CHUNK_PLAN.md` for what Phase B
would add. MLA is now the only spec shape that cannot chunk (no chunked
attention path). MoE specs require subgroups — there is no scalar router/expert
path and `buildDecodeEngine` throws rather than running the dense FFN.

```bash
node scripts/chunk-prefill-test.mjs qwen30b      # MoE, pure attention, no shared expert
PROMPT=400 node scripts/chunk-prefill-test.mjs qwen36q3   # MoE + GDN + shared + 3-bit
```

Weights load via `weight-loader-mlx.ts`: byte-range fetches (never a whole
5.3 GB shard), OPFS keyed by BUILT buffer, bf16 conversion at load. The dev
mirror serves both checkpoints; `?model=qwen36q3` needs the locally-converted
`.weights-local/Qwen3.6-35B-A3B-MLX-q3exp` (16.36 GB) until it is uploaded to
`abgunaydin/Qwen3.6-35B-A3B-MLX-q3exp`.

```bash
# 4-bit: huggingface-cli download lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit \
#          --local-dir .weights-local/Qwen3.6-35B-A3B-MLX-4bit
# 3-bit experts (from the 4-bit, ~16 s):
#   cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/convert-q3-experts.py
npm run test:kernels          # includes the synthetic affine/q3 matmul tests
npm run test:kernels:mlx      # byte-exact repack + loader replay + model budget
npm run test:kernels:real     # kernels vs mlx_lm's own modules on real weights
```

## Chunk-prefill GEMM (`chunkGemm`)

The chunk path picks a GEMM: `sgmat` (E1 on Metal's matrix unit, the default
where the device has `chromium-experimental-subgroup-matrix`) → `tiled` →
`matvec`. `e5` is the same unit at a 64(M)x32(N) tile with a swizzled B —
18-22% over E1 in isolation, **13-15% on real prefill** (BENCH.md, "Sweep
round 3" and "E5 in the engine"). Opt-in until it passes token identity on an
MLC-symmetric and a hybrid spec; neither checkpoint is on this disk.

An EXPLICIT `chunkGemm` that cannot run now **throws** instead of falling back.
Silent substitution is how a kernel A/B measures the same code twice.

```bash
npm run test:kernels:sgmat    # numerics for both matrix-unit kernels, both
                              # quant flavors, ragged M/N/K — exits non-zero
                              # on failure (needs a GPU + `npm i --no-save webgpu`)
node --experimental-strip-types scripts/gemm-sweep-native.mjs   # the tile sweep,
                              # ranked against the SHIPPED kernel in the same run
node --experimental-strip-types scripts/prefill-gemm-ab.mjs llama32
                              # in-engine A/B: two engines, one weight load,
                              # arms alternated; throws if a round did not chunk
GEMM=e5 node scripts/chunk-prefill-test.mjs llama32   # token identity
```

Read the sweep's header before trusting any number it prints: rounds 1 and 2
ranked configs against the harness's own reconstruction of E1 rather than the
shipped kernel, and were wrong by 26%.

## Adding models (`scripts/add-model.mjs`)

Covered-architecture MLX checkpoints are added by pipeline, not by hand:

```bash
npm run add-model -- <hf-repo> [--param <url>] [--check-only]
# green: spec generated into model-spec.ts (ADD-MODEL:SPECS marker) + registry
#        rows (model-registry.ts markers) + compile gate under the new dims
# red:   per-failure {rule, detail, needs-this-kernel} report, exit 1
npm run add-model -- --compat        # regenerate docs/COMPAT.md from constraints.ts
node tests/kernels/compile-spec.mjs <SPEC_EXPORT>   # generic compile gate

# numerical trust (weights required locally):
hf download <repo> --local-dir .weights-local/<repo-tail>
cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-ref.py \
    --model ~/dev/zero-tvm/.weights-local/<tail> --out /tmp/ref-<param>
node scripts/validate-model.mjs <param> --ref /tmp/ref-<param>
```

Constraint matrix lives in `src/compiler/constraints.ts` (checks + the
SUPPORT_MATRIX docs/COMPAT.md is generated from). The registry
(`model-registry.ts`) is the single source for `?model=` params, landing
cards, switcher and branding — `specFromSearch` derives from it. Proof run
2026-08-06: `mlx-community/Qwen3-4B-4bit` → `?model=qwen3mlx`, logits cosine
0.999879 vs mlx_lm, greedy token-exact ("The capital of France is
**Paris**."); dense MLX models run the affine FFN chain (gate_up matmul →
silu_mul → down; fused_ffn is symmetric-only).

## Sharing + peer weights (`share.html`, `workers/share-signal/`)

`share.html?model=<param>` hosts the model running in that tab; `share.html#<room>`
is the guest. Prompts and tokens ride a DTLS-encrypted WebRTC DataChannel; the
Cloudflare Durable Object at `workers/share-signal/` relays SDP/ICE only. The
room id is 128 random bits in the link fragment (never sent to the static host).
The guest renders with `chat-ui.ts` + `public/chat-ui.css` — the same surface as
the chat page, which is why the two cannot drift.

A SECOND channel (`weights`) replicates the host's OPFS weight cache to the
guest's device, so a second machine never re-downloads gigabytes the first one
has: `peer-weights.ts` streams the model's OPFS **directory** (format-agnostic —
MLC shards and MLX built buffers are both flat files there) in 240 KB pieces,
each with a SHA-256 the receiver checks. The guest then runs the model locally.
The hash catches corruption, NOT a dishonest host — in a room you already trust
the host to run the model.

A room holds MANY hosts. `?model=X#<room>` serves an EXISTING room from this
device — the URL a guest is offered after it copies the weights, which is how a
room grows into a swarm. The DO assigns each guest to the least-loaded host and
REASSIGNS its guests when a host's tab closes; the guest rebuilds its peer
connection and keeps the conversation (history lives on the guest, and the host
is stateless between requests).

```bash
npx wrangler deploy                      # in workers/share-signal (prod relay)
node scripts/room-routing-test.mjs       # multi-host routing, no browser, ~10s
node scripts/share-e2e.mjs               # vite + wrangler dev + 2 tabs, real RTC
node scripts/peer-weights-e2e.mjs        # TWO browser profiles; replicates 2.26 GB
MODEL=qwen35 node scripts/peer-weights-e2e.mjs
```

Room routing (assignment, relaying, takeover, departure) is tested with plain
WebSocket clients against `wrangler dev` — seconds, deterministic. Do NOT test
it by booting two engines in one browser: two full models on one GPU hangs long
before it proves anything about routing (tried; the tab never answered).

## Pipeline stages (`layerRange`)

`buildDecodeEngine(..., { layerRange: { start, end } })` builds ONE stage of a
split model: no embedding when `start > 0`, no final norm / LM head / argmax
when `end < layers`, and bind groups only for the layers in range (so a stage
can run without ever loading the others' weights). Drive a stage with
`pipelineStep(input, position)` — token id in / residual out on the first,
residual in / token out on the last. The hand-off is the bare RESIDUAL (d f16
= 4-5 KB per token): every stage re-normalises with its own first layer's
gamma, so no stage needs a weight from its neighbour. Each stage keeps the KV
cache and GDN state of its own layers.

```bash
node scripts/pipeline-split-test.mjs --ref /tmp/ref-llama32 llama32
node scripts/pipeline-split-test.mjs qwen35     # hybrid GDN path
```

`loadWeights(..., spec, { start, end })` loads ONE stage's weights — its
layers, the embedding only if it starts the model, the lm_head / final norm
only if it ends it. MLX checkpoints only: the MLC path fetches whole shards, so
skipping layers would save no download. A tied lm_head IS the embedding table,
so a tied model carries that table on BOTH ends (Qwen3.6-35B is untied and does
not).

`share.html?model=X&layers=0-k` hosts a room holding the first layers;
`share.html?model=X&layers=k-N#<room>` joins it holding the rest (the helper
signs in as a GUEST and offers compute — the relay never learns a third role).
A normal guest chats with the pair; a third DataChannel, `pipeline`, carries
one residual per token.

```bash
node scripts/split-serve-e2e.mjs          # two Chrome profiles, real WebRTC
MODEL=qwen3mlx SPLIT=18 node scripts/split-serve-e2e.mjs
```

Measured 2026-08-07: llama32 split 0-8 / 8-16 answers "The capital of France
is Paris." at 25.3 ms per token across the two stages.

Verified 2026-08-07: splits at layers 1 / mid / N-1 reproduce the whole model
token-for-token on llama32 (real prompt; the whole model's answer matches
mlx_lm) and on qwen35 (32 layers, GDN state per stage). With each stage loading
only its own layers — the real arrangement — llama32 is 0.42 + 0.42 GB against
0.70 GB whole, still token-identical.

Gotchas: `?sig=<port|url>` overrides the signaling relay in DEV ONLY (two test
drivers run their own wrangler concurrently; in prod a link could otherwise
point a guest's signaling at a third party). The host tab needs the "keep this
tab awake" checkbox when backgrounded — Chrome throttles it to ~1 MB/s serving
and ~23 tok/s generating otherwise. DataChannel backpressure must re-CHECK
`bufferedAmount` in a loop; waiting on a single `bufferedamountlow` event
deadlocks when the queue drains between the check and the listener.

## Agent server (`scripts/agent-server.mjs`, `agent-host.html`)

An OpenAI-shaped front door for the model running in a browser tab, so pi and
Cline can drive the engine. The engine cannot leave the browser (weight-loader
reads `GPUBufferUsage` at module scope; weights live in OPFS), so the server
comes to the model: SSE jobs down to the tab, POSTed token batches back, no
WebSocket dependency.

```bash
npm run agent -- qwen3mlx   # ONE command: vite + server + Chrome tab + pi
                            # config patched (providers.zerotvm -> id "ztvm",
                            # contextWindow matched to the real engine ctx),
                            # waits for hosting, prints the pi/Cline/curl lines.
                            # --ctx 65536 raises the window; --pool 0 opts out.
npm run test:agent-server   # e2e: real Chrome, real weights, tool round trip
```

`pi --model ztvm` always means "whatever the launcher started" — relaunching
with a different model rewrites the entry, and the server 400s any OTHER model
name instead of silently serving the resident one (the LM Studio failure mode,
deliberately closed).

Native tool calling is mandatory — Cline removed its XML fallback in v4.0.0
and pi has no text fallback at all. tool-calls.ts renders/parses three
dialects (pinned byte-exact against vendor jinja). Verified end to end:
streaming with finish_reason on the last chunk (pi hard-fails without it),
multi-part content arrays, `developer`/`tool` role folding, a full
get_weather round trip, and 4xx (never 5xx) for unrecoverable errors — pi
retries 5xx with backoff and hangs ~2 min. One request at a time: a single KV
cache; concurrent generations would interleave. Keep the host tab visible —
Chrome throttles backgrounded tabs to ~1/3 throughput.

## Quality vs fidelity (`docs/QUALITY.md`)

Everything the repo verified before 2026-08-10 was **fidelity** — does the
engine compute what the reference computes? `scripts/mlx-ref.py:29` is
`mlx_lm.load(args.model)`, the SAME quantized checkpoint, so a model quantized
into gibberish passes every gate (verified by requantizing Llama-1B to 2-bit:
the reference itself emits `-a-a-m-mowcarecare…` and `validate-model.mjs`
prints "model validates against mlx_lm"). The one pre-existing quality gate is
the five-prompt lexical battery in `tests/e2e/*.test.ts` — 4 of 9 models,
neither MoE build, and CI cannot run it.

```bash
# compare two CHECKPOINTS — the tool to reach for. Runs on the reference.
cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/quality-ab.py \
    --a <baseline> --b <candidate> --windows 24 --window 512
# build a known-degraded build to prove the harness can see damage
uv run python ~/dev/zero-tvm/scripts/requantize.py --src <4bit> --dst <out> \
    --bits 3 --scope experts|mlp|all
# the ENGINE's own perplexity (browser), plus a fidelity check vs mlx_lm
npm run quality -- llama32 --tokens 256 --dump-ids /tmp/ids.json   # no GPU
npm run quality -- llama32 --tokens 256 --ref /tmp/ppl.json
```

`quality-ab.py` is **paired** — both arms score identical windows, so
differencing per window cancels between-window difficulty. On the OLMoE run
the same data gave unpaired z = 0.8 ("not distinguishable") and paired
z = 14.7. Validated on four builds; expert-only 3-bit on a real 64-expert MoE
costs **+8.4%**, dense 3-bit costs +76% to +194%.

`requantize.py` and `convert-q3-experts.py` both REFUSE a zero-conversion run.
The expert markers are layout-specific (`.mlp.switch_mlp.` fused vs
`.mlp.experts.<N>.` per-expert, both live in one model family), and a
converter that matches nothing writes a valid byte-identical copy that the A/B
then reports as "no significant difference".

`?model=` fixtures for this live in `.weights-local/`; `OLMoE-1B-7B-0125-4bit`
(3.6 GB) is the MoE fixture, `Llama-3.2-1B-Instruct-4bit` the dense one.

## KV paging + prefix pool (`docs/PAGING_PLAN.md`)

The page table has been the IDENTITY since it was written — `attention*.wgsl`
resolve every page through `page_table_values`, but the writers (`kv_append`,
`qkv_fused`, `qk_norm_rope_append`, `kv_quantize_int8`, `mla_kv_write`) do not
take it at all and compute `position / PAGE_SIZE` directly.

```bash
node scripts/paging-test.mjs llama32     # falsifier; --expect-fail is the DEFAULT
node scripts/paging-measure.mjs --full   # the four transfer rates
node scripts/prefix-stability-test.mjs --list
```

Phase 0 is complete and its four results are load-bearing:
- **K is RoPE'd BEFORE the cache write on every path**, so a cached page is
  valid only at the positions it was written at. This is a *prefix* pool and
  can never be a vLLM-style relocatable block pool.
- Agent prompts ARE append-only in token ids — nine real transcripts, ~340
  transitions, 0 breaks, tool traffic included. Phi-3 (SPM) untested and is
  the one template BENCH.md already records breaking.
- Save 0.29 s / restore 0.36 s for 625 MiB, 56x under the threshold.
  `writeBuffer` is the slowest link, not the disk.
- A permuted table changes the logits on HEAD, as it must. `paging-test.mjs`
  asserts that today and flips to `--expect-pass` when Phase 2 lands.

## Cross-site context

`sites.json` is synced from `~/sites-shared/sites.ts`. Edit URLs,
taglines, and the `sameAs` identity list there.

## Pre-push gates (`lefthook.yml`)

`lefthook install` once per clone. pre-commit is the house template (biome +
gitleaks on staged files). pre-push is where this repo differs, because its
bugs differ: **every defect that reached a user here was SILENT** — fluent
wrong output, never a crash.

- `npm run typecheck`
- `npm run test` (~3 s headless)
- `node scripts/regression-test-gate.mjs` — **a commit whose SUBJECT claims a
  fix and which touches `src/`, `scripts/` or `workers/` must also touch a
  test.** Escape when the fix genuinely needs hardware this machine cannot
  give a hook: `SKIP_TEST_GATE=1 git push`, and say why in the message.

Why the third one exists, from this repo's own history: the Phi-3 `ropeFreqs`
P0 ran broken for ten days through two deploys; `kv_quantize_int8` wrote half
of every head-dim-256 row and produced *fluent wrong text* rather than noise,
while the suite covering it tested head-dim 128, so re-introducing it would
still have passed; a GDN rewind restored a dead conversation's state in 2,929
of 20,000 simulated conversations while six live turns looked perfect. None of
those were caught by reading a diff.

**The test that matters is the one that FAILS on the old code.** Write it, run
it against the bug, watch it fail, then fix. A test written after the fix and
never seen red proves nothing.

The GPU suites are NOT in the hook — they need hardware CI and the agent
sandbox do not have. They are the gate for anything touching WGSL or the
engine, and they run in a human's shell:

```bash
npm run test:kernels:qwen35   # hybrid + the int8 pack at head-dim 256
npm run test:kernels:mlx      # loader replay, byte-exact
node scripts/chunk-prefill-test.mjs <spec>   # token identity
```

A hook that skipped those silently would be worse than one that names them,
because it would read as coverage.

## Known gaps

- **No ESLint.** Fixed 2026-08-10: ci.yml calls `npm run typecheck` directly
  and the misleading `lint` alias is gone. An actual ESLint config is still a
  future addition — `biome` is the house choice per the global setup.
- The "More by Ahmet" footer is repeated (and has drifted out of sync)
  across ~5 HTML pages. The JSON-LD `Person` block and "Related work" grid
  live only in `index.html`. Will migrate to sites-shared HTML partials.
- **Dead top-level modules.** `src/{engine,capture,ui}.ts` are
  unreferenced by the build; some are research/capture tooling. Prune
  or relocate under a `tools/` dir.
