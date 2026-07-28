# zerotvm.com

## Goal

Full browser-native Phi-3-mini inference on 10 hand-written kernel roles
(27 WGSL files counting subgroup/tiled/int8 variants) + ~2k lines of
TypeScript — replacing the 85 TVM-autotuned shaders WebLLM ships. The
pedagogical thesis: the entire LLM forward pass is readable end-to-end
in a single sitting.

Head-to-head vs WebLLM numbers live in `BENCH.md`.

## Architecture

Vite-built multi-page static site. Each HTML file is a standalone demo:

- `index.html` — marketing essay / hub
- `zero-tvm.html` — the hand-written Phi-3 chat (157 kB JS bundle)
- `compiler-chat.html` — WebLLM/TVM reference chat (5.9 MB bundle)
- `docs.html`, `architecture.html`, `demo.html`, `dump.html`,
  `shaders.html`, `validate.html`, `webllm-bench.html` — docs, shader
  viewers, benchmark runner.
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
  `npm run test:kernels:qwen` is a compile-only gate under QWEN3_4B dims.
  `src/compiler/shaders/` holds the 27 WGSL files
  (10 roles + tiled/subgroup/int8/f32 variants). The WebLLM/TVM
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
`model-select.ts` maps `?model=qwen3` → `QWEN3_4B`. The Qwen chat path is
**v1-unfused** (QK-norm is incompatible with the fused QKV kernel → 10
dispatches/layer; vec4 loads only where K % 1024 == 0; no int8-KV) — see
BENCH.md's Qwen section for the measured WebLLM gap.

```bash
node scripts/download-weights.mjs --model qwen3   # ~2.3 GB → .weights-local/
npm run dev                                       # then zero-tvm.html?model=qwen3
                                                  #  or validate.html?model=qwen3
npm run test:kernels:qwen                         # 21/21 compile+shape gate (no GPU)
npm run test:e2e                                  # includes tests/e2e/qwen.test.ts —
                                                  # skips loudly if the mirror isn't primed
BENCH_QUERY="?model=qwen3" npm run bench          # same-session A/B vs WebLLM's
                                                  # prebuilt Qwen3-4B; never writes results.json
```

## Qwen3.5-4B hybrid (`?model=qwen35`)

Third model, first hybrid: 24 gated-DeltaNet layers + 8 gated-attention
layers (GQA 16/4, head_dim 256, partial RoPE 64/256, sigmoid attention
gate), 248k vocab, tied lm_head; `model-select.ts` maps `?model=qwen35` →
`QWEN35_4B`. The path is **v1-scalar-GDN** (scalar DeltaNet kernels, no
chunked prefill — prompts replay token-by-token, unfused GDN projections),
so its BENCH.md number is a floor. WebLLM A/B needs `@mlc-ai/web-llm`
≥ 0.2.84 (Qwen3.5 first ships in the v0_2_84 prebuilt libs).

```bash
node scripts/download-weights.mjs --model qwen35  # ~2.6 GB → .weights-local/
npm run dev                                       # then zero-tvm.html?model=qwen35
                                                  #  or validate.html?model=qwen35
npm run test:kernels:qwen35                       # 13/13 GDN kernel family vs CPU reference
npm run test:e2e                                  # includes tests/e2e/qwen35.test.ts —
                                                  # skips loudly if the mirror isn't primed
BENCH_QUERY="?model=qwen35" npm run bench         # same-session A/B vs WebLLM's
                                                  # prebuilt Qwen3.5-4B; never writes results.json
```

## Cross-site context

`sites.json` is synced from `~/sites-shared/sites.ts`. Edit URLs,
taglines, and the `sameAs` identity list there.

## Known gaps

- **No ESLint.** `.github/workflows/ci.yml` calls `npm run lint`, which
  `package.json` aliases to `typecheck` (`tsc --noEmit`) — that alias shipped
  and CI is green. Cosmetic follow-up: point ci.yml at `npm run typecheck`
  directly and drop the alias. ESLint config is still a future addition.
- The "More by Ahmet" footer is repeated (and has drifted out of sync)
  across ~5 HTML pages. The JSON-LD `Person` block and "Related work" grid
  live only in `index.html`. Will migrate to sites-shared HTML partials.
- **Dead top-level modules.** `src/{engine,capture,ui}.ts` are
  unreferenced by the build; some are research/capture tooling. Prune
  or relocate under a `tools/` dir.
