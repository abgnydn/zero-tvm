# Benchmarks — measure once, propagate everywhere

The engine already measures itself: `chat.ts` exposes `window.bench(nTokens,
nRuns)` (decode tok/s, with warmup) and a per-kernel profiler
(`window.bench(128, 3, true)` via GPU timestamp queries), and
`webllm-bench.html` runs WebLLM on the same weights. What was missing was
driving both on real hardware and getting the numbers into the docs without
hand-editing six files. That's this.

## `npm run bench` (needs a real GPU)

```bash
BENCH_HW="M2 Pro, 19-core" npm run bench
```

Drives the actual browser engine — so it needs WebGPU (a dev Mac) and ~2 GB of
Phi-3 weights (first run downloads them; cached after). It **cannot** run in a
GPU-less CI sandbox. It:

1. boots `zero-tvm.html` and calls `window.bench()` → Zero-TVM total median,
   plus median TTFT and median decode-only rate (every run resets the
   engine's absorbed-token record, so every run pays a full prefill),
2. boots `webllm-bench.html` → WebLLM total median, plus its self-reported
   decode/prefill rates captured per run,
3. writes `bench/results.json` (the single source of truth — commit it after
   each bench run so the repo records the numbers the docs were synced from),
4. runs `sync-docs.mjs --write` to update the docs.

Env knobs: `BENCH_TOKENS` (default 128), `BENCH_RUNS` (default 5), `BENCH_HW`.

### A/B modes (`BENCH_QUERY`) — never touch results.json

`BENCH_QUERY` appends a query string to `/zero-tvm.html` and switches the
script into A/B mode. A/B runs **never** write `bench/results.json` — that
file is the Phi-3 headline artifact and only the plain `npm run bench` flow
updates it.

- **Flag A/B** — `BENCH_QUERY="?vec4=0" npm run bench` runs only the
  Zero-TVM half with that flag. For comparing shader variants against the
  defaults.
- **Cross-engine model A/B** — `BENCH_QUERY="?model=qwen3" npm run bench`
  runs BOTH halves on that model: `zero-tvm.html?model=qwen3` and
  `webllm-bench.html?model=qwen3` (WebLLM's own Qwen3-4B-q4f16_1 prebuilt
  wasm against the same local weight mirror), back-to-back in the same
  session, and prints both medians + the gap. Same-session pairing matters:
  absolute tok/s drifts with machine state, so only the pair is meaningful.
  Note the WebLLM page always runs its fixed protocol (3 × 120-token runs +
  warmup, median) regardless of `BENCH_TOKENS`/`BENCH_RUNS` — identical to
  how the Phi-3 headline's WebLLM half is measured.

## `npm run bench:sync` (no GPU needed)

Propagates `bench/results.json` into the docs. Dry-run by default; `--write`
to apply. Updates:

- **BENCH.md, README.md, index.html, docs.html** — every number wrapped in
  `<!--bench:KEY-->…<!--/bench:KEY-->` markers (`zt`, `webllm`, `gap`) in
  tables and stat tiles. The markers are invisible on GitHub and in the
  rendered pages.
- **`src/webllm-bench/main.ts`** — the `// bench:zt` constant.

Prose mentions in README / docs.html / demo.html / index.html are **reported,
not rewritten** — auto-editing prose is how benchmark numbers get mangled.
Prose should cite a dated, same-session pair with **both** metrics ("+16.0%
total / +31.4% decode, Phi-3, 2026-07-30"), not exact medians and not a
cross-session band. The old "28–31% faster across sessions" phrasing was
retired on 2026-07-30: it was a total-wall-clock ratio labelled "decode", and
the corrected round measured −16.0% total on the same pair. Exact medians
belong only in marker-wrapped table/stat-tile positions. Anything the script
prints in its prose checklist is either intentional history (the old M2 Pro
numbers, the withdrawn 2026-07-29 pairs, falsified experiments) or needs
rewording.

`results.json` schema:

```json
{ "ztDecode": 69.55, "webllmDecode": 59.95, "hardware": "Apple M2 Max", "date": "2026-07-30" }
```

**Naming caveat:** `ztDecode` / `webllmDecode` hold **total wall-clock**
medians (prefill + decode), not decode-only rates. The names predate the
2026-07-30 metric split and are the wire format `run.mjs` writes and
`sync-docs.mjs` reads. For the unambiguous per-model records — total, TTFT,
decode, and WebLLM's self-reported prefill/decode — see
[`bench/results/`](results/), which also holds the two Qwen pairs that A/B
mode never writes here.

## Run it on a cloud GPU (no Mac)

You don't need a local machine — any NVIDIA GPU instance works (RunPod, Lambda,
Modal, a GPU CI runner, …). `bench/Dockerfile` boots headless Chrome on the
container's GPU, primes the weights, runs both engines, and prints
`results.json`:

```bash
docker build -f bench/Dockerfile -t zerotvm-bench .
docker run --gpus all --rm zerotvm-bench
```

Needs the NVIDIA Container Toolkit (so the container sees the driver's Vulkan
ICD) and outbound `huggingface.co`. Copy the printed JSON into
`bench/results.json` locally and run `npm run bench:sync -- --write` (no GPU) to
update the docs. Caveat: a cloud NVIDIA GPU is a different baseline than the
M2 Pro in BENCH.md — absolute tok/s will differ; the Zero-TVM-vs-WebLLM **gap**
is the portable number. `cloud-bench.sh` auto-labels the run with the GPU name.

### As a GPU CI job

On a GPU-enabled runner (needs `workflow` scope to add):

```yaml
  bench:
    runs-on: [self-hosted, gpu]   # or a GPU cloud runner
    steps:
      - uses: actions/checkout@v4
      - run: docker build -f bench/Dockerfile -t zerotvm-bench .
      - run: docker run --gpus all --rm -v ${{ github.workspace }}/out:/app/bench zerotvm-bench
      # then commit out/results.json and run `npm run bench:sync -- --write`
```

> The kernel-correctness suite (`npm run test:kernels`) needs **no** GPU — it
> runs on Mesa lavapipe in any container. Only this throughput bench needs a
> real adapter.
