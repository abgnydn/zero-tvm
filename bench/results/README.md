# `bench/results/` — per-model corrected pairs

`bench/results.json` (one level up) is the auto-written artifact of the plain
`npm run bench` flow, and it covers the **Phi-3 pair only**. Cross-engine A/B
mode (`BENCH_QUERY="?model=qwen3" npm run bench`) deliberately never writes it,
so the Qwen pairs had no machine-readable home. These files are that home.

One file per model, hand-written from the run logs of the 2026-07-30 corrected
protocol (BENCH.md § "Corrected protocol (2026-07-30)"). `sync-docs.mjs` does
**not** read them — the BENCH.md table is their rendering, and these are the
falsifiable artifact behind it.

Every pair carries both metrics, because reporting one alone is
cherry-picking:

- `totalTokPerS` — wall-clock throughput, prefill + decode. Both engines do
  identical work. This is the conservative number and the one to lead with.
- `decodeTokPerS` — steady-state decode only, excluding prefill and the first
  token. This is the kernel-level number.
- `ttftMs` (ours) / `prefillTokPerS` (WebLLM's self-report) — the first-token
  cost that `totalTokPerS` folds in.

Field-naming caveat on the parent `results.json`: its `ztDecode` /
`webllmDecode` keys hold **total wall-clock** medians, not decode-only rates.
The names predate the split and are the wire format `bench/run.mjs` writes and
`bench/sync-docs.mjs` reads; they were left alone rather than renamed. Use
`phi3-mini.json` here for the unambiguous version of the same pair.
