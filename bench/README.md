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
3. writes `bench/results.json` (the single source of truth, git-ignored),
4. runs `sync-docs.mjs --write` to update the docs.

Env knobs: `BENCH_TOKENS` (default 128), `BENCH_RUNS` (default 5), `BENCH_HW`.

## `npm run bench:sync` (no GPU needed)

Propagates `bench/results.json` into the docs. Dry-run by default; `--write`
to apply. Updates:

- **BENCH.md** — numbers wrapped in `<!--bench:KEY-->…<!--/bench:KEY-->`
  markers (`zt`, `webllm`, `gap`). The markers are invisible on GitHub.
- **`src/webllm-bench/main.ts`** — the `// bench:zt` constant.

Prose mentions in README / docs.html / demo.html are **reported, not
rewritten** — auto-editing prose is how benchmark numbers get mangled. Update
those by hand from the printed checklist.

`results.json` schema:

```json
{ "ztDecode": 42.14, "webllmDecode": 51.5, "hardware": "M2 Pro, 19-core", "date": "2026-06-23" }
```
