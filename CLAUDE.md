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
- `src/zero-tvm/` — the hand-written engine. `zero-tvm.html` loads
  `chat.ts`, which pulls in `tokenizer.ts`, `weight-loader.ts`,
  `spec-sim.ts`, and the pipeline builder `compile()` from
  `src/compiler/compiler.ts`. NOTE: the decode loop is forked —
  `chat.ts` carries its own `buildDecodeEngine`, and a second copy
  lives in `engine-core.ts` (used by `validate.html` + the dump/dev
  tools). Keep edits to the inner loop in sync across both.
- `src/compiler/` — despite the name, this is where OUR kernels live.
  `compiler.ts` is the hand-written pipeline builder: `compile()`
  creates every GPU pipeline, buffer, and bind group itself (no
  codegen, no TVM). `src/compiler/shaders/` holds the 27 WGSL files
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
- `src/{engine,phi3v2,standalone,capture,ui,main}.ts` — an older engine
  generation + TVM-capture tooling. Not in the Vite build graph and not
  imported by the shipped pages (`capture.ts`/`dump-tvm.ts` back the
  RESEARCH.md interception work).
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

Deploy: `node ~/sites-shared/deploy.mjs zerotvm` (CF Pages, project
`zerotvm`).

## Cross-site context

`sites.json` is synced from `~/sites-shared/sites.ts`. Edit URLs,
taglines, and the `sameAs` identity list there.

## Known gaps

- **No ESLint.** Previously `package.json` had `lint: tsc --noEmit` which
  was a landmine (running "lint" actually typechecked). Renamed to
  `typecheck` in the scripts-normalization pass — `.github/workflows/ci.yml`
  was left calling the deleted `npm run lint` (CI red until fixed) and now
  calls `npm run typecheck`. ESLint config is still a future addition.
- The "More by Ahmet" footer is repeated (and has drifted out of sync)
  across ~5 HTML pages. The JSON-LD `Person` block and "Related work" grid
  live only in `index.html`. Will migrate to sites-shared HTML partials.
- **Forked decode loop.** `buildDecodeEngine` exists in both `chat.ts`
  (shipped) and `engine-core.ts` (validate + dev tools). De-dupe pending.
- **Dead top-level modules.** `src/{engine,phi3v2,standalone,capture,ui,
  main}.ts` are unreferenced by the build; some are research/capture
  tooling. Prune or relocate under a `tools/` dir.
