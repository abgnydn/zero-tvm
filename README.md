<div align="center">

<a href="https://zerotvm.com"><img src="docs/banner.svg" alt="Zero-TVM — LLM inference in the browser, on hand-written WGSL" width="100%" /></a>

<p align="center">
  <a href="https://zerotvm.com"><img src="https://img.shields.io/badge/TRY_IT-zerotvm.com-E8955A?style=for-the-badge&labelColor=0B1020" alt="Live: zerotvm.com" /></a>
</p>

<p align="center">
  <a href="https://github.com/abgnydn/zero-tvm/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/abgnydn/zero-tvm/ci.yml?branch=main&style=for-the-badge&label=CI&labelColor=0B1020" alt="CI status" /></a>
  <a href="./BENCH.md"><img src="https://img.shields.io/badge/benchmarks-vs_WebLLM_%2B_LM_Studio-1E293B?style=for-the-badge&labelColor=0B1020" alt="Benchmarks" /></a>
  <a href="./docs/COMPAT.md"><img src="https://img.shields.io/badge/compatibility-matrix-1E293B?style=for-the-badge&labelColor=0B1020" alt="Compatibility matrix" /></a>
  <a href="https://doi.org/10.5281/zenodo.20838918"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20838918-059669?style=for-the-badge&labelColor=0B1020" alt="DOI" /></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-059669?style=for-the-badge&labelColor=0B1020" alt="License: MIT" /></a>
</p>

<p align="center">
  <a href="https://zerotvm.com"><strong>Try it →</strong></a> ·
  <a href="https://zerotvm.com/docs"><strong>Docs</strong></a> ·
  <a href="./BENCH.md"><strong>Benchmarks</strong></a> ·
  <a href="./CHANGELOG.md"><strong>Changelog</strong></a>
</p>

</div>

# Zero-TVM

**[zerotvm.com](https://zerotvm.com)** — pick a model, it runs in your browser
tab.

Zero-TVM is an LLM engine written by hand in TypeScript and WGSL, the WebGPU
shading language. It uses no ML framework, no compiler and no autotuner —
every GPU shader is a file you can open and read. That is the point of the
project: the complete forward pass of a modern LLM, readable end to end in a
single sitting.

Two things here are unusual:

- **Every shader is hand-written** — [the kernels](#the-kernels). The models
  range from a small 1B up to Qwen3.6-35B-A3B, a sparse mixture-of-experts.
- **A model too large for one machine can be split across several** —
  [the swarm](#the-swarm). Each machine holds a slice of the layers, and a
  guest with the room link just chats.

## Try it

Open **[zerotvm.com](https://zerotvm.com)** and pick a model. The weights
download once from Hugging Face, then stay cached in your browser, so the next
visit starts instantly. In a single tab nothing you type leaves your machine —
no prompt is sent to any server.

You need Chrome or Edge with WebGPU (on by default in recent versions) and the
free RAM the model's card asks for. Every shipped model — its size, RAM need
and `?model=` flag — is defined in one table,
[`src/zero-tvm/model-registry.ts`](src/zero-tvm/model-registry.ts); the site's
cards render from it, so the card and the engine cannot disagree.

## Quick start (developers)

```bash
npm install
npm run dev            # dev server — serves every page
npm run build          # production build → dist/
npm run check          # typecheck + unit tests
```

Weights for local development mirror into `.weights-local/` — per-model
download commands are in [CLAUDE.md](CLAUDE.md).

## The kernels

All shaders live in [`src/compiler/shaders/`](src/compiler/shaders/), plus one
small readable generator for the int4/int3 matmul family. The engine is
spec-parameterized: layer count, head shape and block kinds come from
[`src/compiler/model-spec.ts`](src/compiler/model-spec.ts), so one decode loop
([`src/zero-tvm/engine-core.ts`](src/zero-tvm/engine-core.ts)) serves a small
dense model, an attention/DeltaNet hybrid and a 256-expert mixture-of-experts.

For comparison, [`src/tvm-shaders/`](src/tvm-shaders/) holds the
machine-generated WGSL that WebLLM ships, captured from a running session.
Keeping the two side by side is what makes the replacement auditable.

## The swarm

A model that does not fit on one machine is cut into layer ranges, one machine
per range. Each machine downloads and holds only its own layers; a token's
intermediate state hops from machine to machine over WebRTC; a guest with the
room link chats without downloading anything — it does not even need WebGPU.
Conversations ride an encrypted peer-to-peer channel; the only server involved
relays connection setup, and the room id lives in the part of the link
browsers never send to a server.

The entrance builds the links for you: on
[zerotvm.com/#swarm](https://zerotvm.com/#swarm), pick a model that can be
split and press *Too big for one machine? Split it* under the enter buttons.

```
share.html?model=X                   HOST    opens a room serving the whole model
share.html?model=X&layers=0-k        HOST    opens a room, holds the first layers
share.html?model=X&layers=k-N#<room> HELPER  joins that room, holds the rest
share.html#<room>                    GUEST   chats; runs nothing locally
```

**The query must come before the `#`.** Pasting a room link and typing
`?model=…` after it silently opens a brand-new room instead of joining — two
machines then wait for each other in different rooms. One parser,
[`src/zero-tvm/room-url.ts`](src/zero-tvm/room-url.ts), decides every role,
and its test feeds each generated link back through the parser, including that
case.

The honest limits:

- **Same network, or an ordinary home router.** There is no TURN relay, so
  corporate and hotel networks usually will not connect.
- **Splitting needs an MLX checkpoint** (the site only offers links for models
  that can split).
- **The hop is serial with decoding**, so this is for machines you own on one
  network — over the open internet the round trip dominates.
- **A backgrounded tab is throttled hard.** Serving roles carry a "keep this
  tab awake" toggle; the browser shows its audio indicator while it is on.

Weights also replicate machine-to-machine, so a second machine never
re-downloads what the first already has
([`src/zero-tvm/peer-weights.ts`](src/zero-tvm/peer-weights.ts)). Each piece
carries a hash — that catches corruption, not a dishonest host; a room already
trusts its host to run the model.

## How fast is it

Measured two ways, protocol and every number in [BENCH.md](BENCH.md):

- **Against WebLLM** (the standard browser LLM stack), on identical weights in
  the same session: faster on every model measured.
- **Against LM Studio** (a native runtime, no browser in the way), on the same
  checkpoint bytes: ahead on prompt processing, close behind on decode — while
  the individual kernels measure slower than MLX's. The advantage is
  structure, not shader speed.

BENCH.md also keeps every withdrawn claim, dated, with the reason — nothing is
retconned.

## How it's checked

Three layers of tests, each catching what the previous cannot:

```bash
npm run test:kernels        # every kernel vs a JS reference (synthetic)
npm run test:kernels:real   # layers vs mlx_lm's OWN modules on real weights
npm run test:kernels:mlx    # checkpoint repacking, byte-for-byte
```

Every defect that ever reached a user here was silent — fluent wrong output,
never a crash. So `node scripts/mutation-gate.mjs` re-introduces bugs that
actually shipped and checks the suite still fails on them. The doctrine is
[docs/VERIFICATION.md](docs/VERIFICATION.md); its one rule: **something that
looks like a check is not a check.**

## Adding a model

A model whose building blocks the kernels already cover is added with one
command:

```bash
npm run add-model -- mlx-community/Qwen3-4B-4bit --param qwen3mlx
```

It probes the checkpoint, generates the spec, registers it on every surface
and compiles every kernel under the new dimensions. If a needed kernel does
not exist, it names it — [docs/COMPAT.md](docs/COMPAT.md) is the support
matrix. Then verify numerically:

```bash
node scripts/validate-model.mjs qwen3mlx --ref /tmp/ref-qwen3mlx
```

That checks the engine computes what `mlx_lm` computes on the same checkpoint.
It is not a quality claim — [docs/QUALITY.md](docs/QUALITY.md) explains the
difference and holds the tools that do measure quality.

## Where things live

| page | what it is |
|---|---|
| `index.html` | The entrance: pick a character, chat in place. Also the swarm link builder. |
| `zero-tvm.html` | The same chat as a direct link (`?model=…`). |
| `share.html` | Host, help or join a room — [the swarm](#the-swarm). |
| `validate.html` | Multi-prompt smoke test against local weights. |
| `agent-host.html` | An OpenAI-shaped front door so local agent tools can drive the engine (`npm run agent`). |
| `docs.html` | The annotated reference, including the kernel walkthrough. |

| what | where |
|---|---|
| Every measured number + protocol | [BENCH.md](BENCH.md) |
| Per-module detail, per-model commands, known gaps | [CLAUDE.md](CLAUDE.md) |
| Shipped model list (the source of truth) | [`src/zero-tvm/model-registry.ts`](src/zero-tvm/model-registry.ts) |
| Reference docs + diagrams | [zerotvm.com/docs](https://zerotvm.com/docs) |
| How this project verifies itself | [docs/VERIFICATION.md](docs/VERIFICATION.md) |
| What a new model needs | [docs/COMPAT.md](docs/COMPAT.md) |
| Quality vs fidelity | [docs/QUALITY.md](docs/QUALITY.md) |
| How the TVM shader capture worked | [RESEARCH.md](RESEARCH.md) |
| Release history | [CHANGELOG.md](CHANGELOG.md) |

## License

MIT. See [LICENSE](LICENSE).

## Citation

This repo ships a [`CITATION.cff`](CITATION.cff), so GitHub's "Cite this
repository" button renders APA / BibTeX automatically. Each release is
archived to Zenodo — cite the concept DOI
[10.5281/zenodo.20838918](https://doi.org/10.5281/zenodo.20838918) for all
versions.

```
Gunaydin, A. B. (2026). Zero-TVM: browser LLM inference on hand-written
WGSL kernels. https://zerotvm.com | https://github.com/abgnydn/zero-tvm
```
