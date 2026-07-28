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

1. boots `zero-tvm.html` and calls `window.bench()` → Zero-TVM decode median,
2. boots `webllm-bench.html` → WebLLM decode median,
3. writes `bench/results.json` (the single source of truth — commit it after
   each bench run so the repo records the numbers the docs were synced from),
4. runs `sync-docs.mjs --write` to update the docs.

Env knobs: `BENCH_TOKENS` (default 128), `BENCH_RUNS` (default 5), `BENCH_HW`.

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
Prose should cite the stable ratio range ("28–31% faster across sessions"),
not exact medians; exact medians belong only in marker-wrapped table/stat-tile
positions. Anything the script prints in its prose checklist is either
intentional history (the old M2 Pro numbers, falsified experiments) or needs
rewording.

`results.json` schema:

```json
{ "ztDecode": 66.33, "webllmDecode": 51.98, "hardware": "Apple M2 Max", "date": "2026-07-25" }
```

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
