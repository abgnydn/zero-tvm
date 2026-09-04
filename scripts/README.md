# scripts/ — what each script is and how to run it

One line per script. Category first:

- **active-infra** — gates, harnesses, servers. Keep green; ask before retiring.
- **bench-harness** — measures something on hardware. Numbers go to BENCH.md or nowhere.
- **on-demand-fixture** — generates test fixtures or reference bundles. Run when fixtures rot.
- **reference-builder** — produces a reference output another check compares against.
- **historical-probe** — answered a question already; provenance for a doc figure. Do not extend.
- **site-ops** — deploy, cleanup, art.

## Active infra (gates, servers, harnesses)

- `add-model.mjs` — active-infra. `npm run add-model -- <hf-repo> [--check-only]`; `--compat` regenerates docs/COMPAT.md.
- `agent.mjs` — active-infra. `npm run agent -- <model>`; launcher: vite + server + tab + pi config.
- `agent-native.mjs` — active-infra. `node scripts/agent-native.mjs <model> [--ctx N] [--pool 0] [--kv8 0]`; in-process /v1 on dawn.node.
- `agent-server.mjs` — active-infra. `node scripts/agent-server.mjs`; OpenAI-shaped front door for the tab engine.
- `agent-server-test.mjs` — active-infra. `npm run test:agent-server`; e2e incl. tool round trip.
- `agentic-eval.mjs` — bench-harness. `node scripts/agentic-eval.mjs qwen36q3`; short agentic task score.
- `check-numbers.mjs` — active-infra. `node scripts/check-numbers.mjs [--hardware m4max]`; rejects physically impossible throughputs.
- `check-vitest-skips.mjs` — active-infra. `node scripts/check-vitest-skips.mjs <report.json>`; fails on any skip not named in ci-skip-allowlist.json with a reason.
- `chunk-cap-sweep.mjs` — bench-harness. `node --experimental-strip-types scripts/chunk-cap-sweep.mjs [spec] [tokens]`; correctness vs chunk cap.
- `chunk-prefill-test.mjs` — active-infra. `node scripts/chunk-prefill-test.mjs <spec>`; chunked vs per-token token identity.
- `clean-deployments.mjs` — site-ops. `node scripts/clean-deployments.mjs [--apply]`; prune CF preview deploys.
- `ctx-test.mjs` — active-infra. `node scripts/ctx-test.mjs [spec]`; does a big ?ctx= actually boot and generate.
- `deploy-space.sh` — site-ops. Deploys the HuggingFace Space demo.
- `download-weights.mjs` — active-infra. `node scripts/download-weights.mjs --model <param>`; prime .weights-local/.
- `eval-registry.mjs` — active-infra. Check registry read by the eval CLI and UI alike.
- `evals.mjs` / `evals.html` — active-infra. Unified eval runner + page (replaces scattered npm scripts).
- `kv-pool-test.mjs` — active-infra. `node scripts/kv-pool-test.mjs [spec]`; cold restore token-identical to prefill.
- `mla-engine-test.mjs` — active-infra. `npm run test:mla-engine`; MLA engine test.
- `moe-pool-test.mjs` — active-infra. `node --experimental-strip-types scripts/moe-pool-test.mjs qwen30b`; pooled vs unpooled.
- `mutation-gate.mjs` — active-infra. `node scripts/mutation-gate.mjs [--list]`; reinstates shipped bugs, suite must go red.
- `needle-test.mjs` — active-infra. `node scripts/needle-test.mjs [spec]`; retrieval sweep at depth.
- `nightly-fidelity.sh` — active-infra. Unattended fidelity-at-depth run.
- `paging-measure.mjs` — bench-harness. `node scripts/paging-measure.mjs [--full]`; transfer-rate legs for the paging plan.
- `paging-test.mjs` — active-infra. `node scripts/paging-test.mjs [spec] [--expect-fail]`; page-table falsifier.
- `peer-weights-e2e.mjs` — active-infra. `node scripts/peer-weights-e2e.mjs`; two profiles, real OPFS replication.
- `pipeline-split-test.mjs` — active-infra. `node scripts/pipeline-split-test.mjs [--ref dir] [spec]`; split token-identity.
- `prefix-stability-test.mjs` — active-infra. `node scripts/prefix-stability-test.mjs --list`; transcript append-only check.
- `quality-eval.mjs` — active-infra. `npm run quality -- <spec> --tokens 256`; engine perplexity + fidelity.
- `regression-test-gate.mjs` — active-infra. Pre-push: a fix-claim commit must touch a test.
- `release-check.mjs` — active-infra. `node scripts/release-check.mjs [--list]`; pre-deploy checklist in one place.
- `room-routing-test.mjs` — active-infra. `node scripts/room-routing-test.mjs`; relay routing without a browser (runs in CI).
- `share-e2e.mjs` — active-infra. `node scripts/share-e2e.mjs`; vite + wrangler dev + 2 tabs, real RTC.
- `split-cost-bench.mjs` — bench-harness. `node scripts/split-cost-bench.mjs`; whole vs split A/B.
- `split-serve-e2e.mjs` — active-infra. `node scripts/split-serve-e2e.mjs`; two Chrome profiles, real WebRTC pipeline.
- `station.mjs` — active-infra. `npm run station`; model-lifecycle desktop surface on :8017.
- `validate-model.mjs` — active-infra. `node scripts/validate-model.mjs <param> --ref <dir>`; engine vs mlx_lm fidelity.

## Bench harnesses (hardware numbers)

- `decode-bench-native.mjs` — bench-harness. `node --experimental-strip-types scripts/decode-bench-native.mjs qwen35`; per-token decode rate.
- `decode-kernel-ab.mjs` — bench-harness (historical — BENCH.md provenance). `node --experimental-strip-types scripts/decode-kernel-ab.mjs qwen3mlx`.
- `decode-profile-native.mjs` — bench-harness. `node --experimental-strip-types scripts/decode-profile-native.mjs qwen3mlx`; where decode GPU time goes.
- `gemm-bench.mjs` — bench-harness. `node scripts/gemm-bench.mjs`; chunk GEMM kernels in isolation.
- `gemm-sweep-native.mjs` — bench-harness. `node --experimental-strip-types scripts/gemm-sweep-native.mjs [--full]`; tile sweep (+ `--gate-only` numerics gate).
- `kernel-ab.mjs` — bench-harness (historical — BENCH.md provenance). `node --experimental-strip-types scripts/kernel-ab.mjs`.
- `kernel-ab-ours.mjs` — bench-harness (historical). `node --experimental-strip-types scripts/kernel-ab-ours.mjs --json`.
- `lmstudio-ab.mjs` — bench-harness (historical — BENCH.md provenance). `node scripts/lmstudio-ab.mjs [--native]`.
- `moe-group-probe.mjs` — historical-probe. `node --experimental-strip-types scripts/moe-group-probe.mjs`.
- `moe-optimistic-probe.mjs` — historical-probe. `node --experimental-strip-types scripts/moe-optimistic-probe.mjs`.
- `moe-pool-control.mjs` — bench-harness. `node --experimental-strip-types scripts/moe-pool-control.mjs qwen30b`; control arm for moe-pool-test.
- `moe-replay.mjs` — bench-harness. `node scripts/moe-replay.mjs bench/quality/moe-trace-qwen30b.json`; replay a captured MoE trace.
- `moe-slab-bytes.mjs` — bench-harness. `node --experimental-strip-types scripts/moe-slab-bytes.mjs [dir] [layer]`.
- `moe-slab-probe.mjs` — historical-probe. `node --experimental-strip-types scripts/moe-slab-probe.mjs`.
- `moe-stream-probe.mjs` — historical-probe. `node --experimental-strip-types scripts/moe-stream-probe.mjs`.
- `moe-trace.mjs` — bench-harness. `node --experimental-strip-types scripts/moe-trace.mjs qwen30b --tokens 400`; capture an MoE trace.
- `multi-turn-ab.mjs` — bench-harness. `node scripts/multi-turn-ab.mjs`; rendering vs reuse on a wrong turn-2.
- `prefill-gemm-ab.mjs` — bench-harness. `node --experimental-strip-types scripts/prefill-gemm-ab.mjs llama32`; in-engine GEMM A/B.
- `dawn-probe.mjs` — historical-probe. Native-host go/no-go (docs/PREFILL_RESEARCH.md).
- `deno-gpu-probe.ts` — historical-probe. `deno run --unstable-webgpu --allow-all scripts/deno-gpu-probe.ts`; browser-tab cost question.
- `depth-bisect.mjs` — bench-harness. `node scripts/depth-bisect.mjs qwen38`; which subsystem breaks the loop at depth.
- `task-eval.sh` — active-infra. `bash scripts/task-eval.sh <checkpoint-dir> <tag>`; benchmarks with a right answer.

## Fixtures and references (on demand)

- `convert-q3-experts.py` — on-demand-fixture. 4-bit → 3-bit expert stacks (refuses zero-conversion runs).
- `embed-ref.py` — reference-builder. Reference sentence embeddings from mlx_lm.
- `gen-tokenizer-fixtures.mjs` / `gen-tokenizer-fixtures-qwen.mjs` / `gen-tokenizer-fixtures-llama.py` — on-demand-fixture. `npm run gen:tokenizer-fixtures*`.
- `gen-toolcall-fixtures.py` — on-demand-fixture. Tool-call fixture generation.
- `kv-bits-gate.py` — reference-builder. What per-row int-b KV does to attention output.
- `kv-quality-ab.py` — bench-harness. Paired perplexity: does int8 KV cost quality.
- `make-dsv2-layer-ref.py` / `make-kernel-ref.py` / `make-mla-ref.py` — reference-builder. Real-weight kernel-validation bundles.
- `mlx-kernel-ab.py` — bench-harness (historical — BENCH.md provenance). Times mx.quantized_matmul at our shapes.
- `mlx-perplexity.py` — reference-builder. Per-token NLL from mlx_lm over an exact sequence.
- `mlx-ref.py` — reference-builder. Logits bundle for validate-model.mjs.
- `pull-tensors.mjs` — on-demand-fixture. `node scripts/pull-tensors.mjs <hf-repo> --match <regex> --out <dir>`.
- `quality-ab.py` — bench-harness. Paired checkpoint perplexity A/B (uv run under ml-research).
- `quality-sweep.sh` — bench-harness. Window-length sweep for quality-eval.
- `render-diff.py` — reference-builder. Diff our rendered prompt vs the checkpoint jinja.
- `render-dump.mjs` — historical-probe. Prompt-render dump.
- `requantize.py` — on-demand-fixture. Build known-degraded checkpoints to prove the harness sees damage.
- `toolcall-depth-ref.py` — reference-builder. Does tool-call emission survive depth on the reference.
- `toolcall_case.py` — reference-builder. The failing agentic conversation, in one place.
- `turboquant-real.py` / `turboquant-ref.py` — reference-builder. TurboQuant plan phases 0/0b gates.
- `yarn-ref.py` — reference-builder. YaRN inv_freq table from DeepSeek's code.

## Site ops and art

- `capture-mascot.mjs` — site-ops. `node scripts/capture-mascot.mjs`; render a model's mascot palette.
- `gen-og.html` / `og-card.html` — site-ops. OG image generation templates.
