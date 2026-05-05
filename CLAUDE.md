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
- `src/zero-tvm/` — the hand-written engine (tokenizer, weight loader,
  inference loop).
- `src/shaders/` — the 27 WGSL files, grouped by role (int4 matmul,
  RoPE, RMSNorm, attention, fused FFN, etc.).
- `src/compiler/` — WebLLM/TVM comparison harness — NOT our kernels;
  it imports `@mlc-ai/web-llm` for the head-to-head.
- `src/tvm-shaders/` — dumped TVM output for apples-to-apples
  inspection against `src/shaders/`.
- `src/webllm-bench/` — the head-to-head benchmarker.
- `sites.json` — synced from `~/sites-shared/sites.ts` (consumed by the
  sibling-link renderer in the HTML pages).

Most pages embed JSON-LD with the `Person` + `sameAs` block. All eight
pages share the "Related work" grid and the "More by Ahmet" footer —
update these by hand until sites-shared HTML partials land.

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
  `typecheck` in the scripts-normalization pass. ESLint config is a
  future addition.
- The JSON-LD + "Related work" grid + footer are duplicated across 8+
  HTML files. Will migrate to sites-shared HTML partials.
