# Site redesign — decided 2026-08-13

The design direction is settled. This file is the spec; the build is not started.
Full mockups: `claude.ai/code/artifact/9bab524c-5d9a-4764-866e-c151485f077c`.

## Why

Three separate problems, all shipping today.

1. **The site disagrees with the family.** Nine sibling sites share a house
   style — blue-black grounds, off-white ink, a serif/sans/mono triple, one
   accent per site, an identity footer. zerotvm.com uses none of it, so it
   reads as unrelated work.
2. **The story is a year stale.** The landing still leads on Phi-3 and a
   WebLLM ratio. The engine now runs nine models, splits across devices,
   serves an OpenAI endpoint, and persists a KV pool.
3. **The July IA rot still ships.** 17 HTML files, 8 in nav. `share.html` and
   `agent-host.html` — two of the four things worth showing — are in neither.
   `dump.html` and `shaders.html` start ~1.8 GB downloads with no gate.

## Direction: Foundry, evergreen

The family chassis with the one accent no sibling claimed: molten amber.

The first draft opened on a stat band. That is rejected. This repo's numbers
change daily and the model list grows by pipeline, so a hardcoded figure is a
stale figure waiting to happen — the stale `sites.json` tagline is the proof.

### Numbers are data, not copy

Three tiers, and they are load-bearing:

- **Landing: zero hand-typed figures.** Anything numeric renders from
  `model-registry.ts` or bench JSON at runtime, or it does not appear.
  Adding a model updates the landing by itself.
- **Chat: live-measured telemetry only.** Context gauge, pool-restore system
  chips, per-turn tok/s meta. Every number on screen was measured in that tab,
  in that session.
- **`/proof`: the only home for recorded numbers.** A dated lab notebook —
  every entry stamped with date + commit, including the falsified-experiments
  ledger. Stale by honest design rather than by accident.

The hero earns credibility from the kernel window and the capability band
instead, both of which are permanent. Cost of this choice: the kernel window
has to be genuinely well-made, and the `measured →` link to `/proof` has to be
one click away.

### Copy register

Declarative subject-verb sentences that state what the thing is. The model is
wgpu's own copy: "wgpu is a safe and portable graphics library for Rust based
on the WebGPU API."

Banned constructions, because they read as generated:

- the negation triad ("no X, no Y, no Z")
- the em-dash pivot
- the punchy fragment
- the contrast punch ("X, not Y")
- puffery adjectives ("real", "blazing")
- persona labels ("the daily driver")
- coined verbs ("prime")

The test: every sentence should be defensible in a bug report.

Settled hero copy:

> **LLM inference in the browser, _on hand-written WGSL._**
>
> zero-tvm is an LLM inference engine written by hand in WGSL and TypeScript.
> It runs entirely in the browser: weights download once, cache locally, and
> inference runs on your own GPU.

## Tokens

| Token | Value | Notes |
| --- | --- | --- |
| `--bg` / `--surface` | `#05070d` / `#0b0f1a` | family lane; panels one step up |
| `--text` / `--muted` / `--dim` | `#edf0f6` / `#a9b3c6` / `#5e6a7e` | 3-step ramp, off-white ink |
| `--accent` / `--accent-hi` | `#f0a860` / `#ffc98f` | molten amber; CTAs, stats, em, gauge |
| `--ok` / `--warn` / `--err` | `#34e3c0` / `#ffc98f` / `#ff7a6b` | semantic, separate from accent |

Type: Fraunces (display) · Spline Sans (body) · Spline Sans Mono (structure) —
the triple already proven on webgpu-fly.

The alias layer `--surface --border --text --muted --accent` is the shared API
webgpu-dna and neuropulse already declare, so a future
`sites-shared/tokens.css` can drive all of them.

## Components

| Component | Used on | Source of truth |
| --- | --- | --- |
| Eyebrow + pulse dot | all pages | fly's kicker |
| Stat cell (serif number, mono caption with reference) | landing, proof | fly's numbers band + dna's cited captions |
| Kernel window (tabs, syntax color, window chrome) | landing hero, chat code blocks, docs | TypeGPU's polish over zerotvm's existing terminal |
| Kernel wall (10 role tiles, mono labels → GitHub) | landing signature section | compute.toys' gallery |
| Comparison table (wins teal, losses amber) | landing, /proof | — |
| Terminal/log pane (ok/warn/err) | gate, rooms host, /agent | dna's harness log |
| Context gauge | chat header | new — the 262k win, always visible |
| System chip (in-stream events) | chat, rooms | new — pool restore, room joins |
| Model card (name, size, honest rate label) | landing, gate | registry-driven, exists — restyled |
| Identity footer + sibling grid + JSON-LD `isPartOf` | all pages | fly/dna's research-line block |

## Information architecture

17 files → 5 public doors.

| Door | Holds |
| --- | --- |
| `/` | story + registry-driven models + kernel wall + proof strip |
| `/chat` | the instrument |
| `/rooms` | share, swarm, split — promoted out of obscurity |
| `/agent` | `ztvm` quickstart + endpoint |
| `/proof` | A/B tables + protocol + falsified ledger |

Unlisted and linked only from `/proof`: `validate`, the benches, `model-smoke`.

Kill list: `architecture.html` (footer still says webgpu-fusion-webllm),
`demo.html` (salvage the content), `compiler-chat.html`. `dump.html` and
`shaders.html` leave nav until they gain a download gate.

## Phases

1. **Tokens + landing. — DONE 2026-08-14.** `public/tokens.css` (family
   chassis + molten amber), `public/landing.css`, `public/fonts.css`.
   `index.html` rebuilt: evergreen hero with a three-tab kernel window over
   real shader source, capability band, registry-driven cards, kernel wall,
   proof strip, identity footer + JSON-LD `isPartOf`. Nav is the five doors.

   Fonts are SELF-HOSTED (292 KB, `public/fonts/`). The page claims nothing
   leaves your machine; a webfont request to Google would leak every visitor's
   IP and make that claim false on the landing page itself.

   Two numbers were removed rather than styled: the meta description's model
   count, and an "Ten roles" heading over ten tiles — both go stale the day a
   model or a role is added, which is the rule this phase exists to enforce.
   The only figures on the page now come from `model-branding` at runtime, and
   a model with no measured rate renders no rate chip (qwen36 is the live
   example).
2. **The chat as instrument.** Restyle `chat-ui.css` to the token set — which
   moves the chat page and the rooms guest together, since they share the
   surface — then add the context gauge, pool system chips, per-turn meta
   lines, instrument strip, and the priming gate.
3. **The other three doors.** `/rooms` host panel, `/agent` quickstart, and
   `/proof` as the dated lab notebook.
4. **Plumbing.** Footer and sibling-grid sync across pages, fix the stale
   zerotvm tagline in `sites.json`, redeploy the HF Space mirror.

## Survey behind the decision

Ten sibling repos read for the house style; webgpu-fly, neuropulse and
webgpu-dna are the strongest. Seven popular WebGPU sites rendered: WebLLM is
visually weak, nobody shows numbers above the fold, TypeGPU proves code works
as the hero, compute.toys proves the gallery pattern that becomes the kernel
wall.
