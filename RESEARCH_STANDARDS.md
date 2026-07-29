# Research-grade engineering standards

**Canonical document. Mirrored across four sibling WebGPU/WGSL research
projects:**

- [`webgpu-q`](https://github.com/abgnydn/webgpu-q) — quantum chemistry
- [`webgpu-dna`](https://github.com/abgnydn/webgpu-dna) — radiation track-structure / radiobiology
- [`zero-tvm`](https://github.com/abgnydn/zero-tvm) — Phi-3 LLM inference (hand-written WGSL, head-to-head vs WebLLM)
- [`neuropulse`](https://github.com/abgnydn/neuropulse) — live 1:1 LLM forward-pass visualization (Phi-3, 3.8B params)

Edit any one and propagate. Project-specific examples in §§ 1, 6, 7, 8, 10
diverge per repo; sections 2–5, 9, 11–15 are universal.

This is the discipline that makes the work publishable in JOSS, citable
years later, and reproducible by reviewers on different hardware. The
patterns matured in different repos and back-port / forward-port between
them (research-grade artifact discipline first in `webgpu-dna`, the
**"falsify before shipping" CPU pre-screen first in `zero-tvm`** —
which validates every kernel against a CPU reference before the WGSL
even reaches the GPU — automated doc-vs-code drift detection in
`neuropulse`, full porting framework in `webgpu-q`). Future siblings
inherit the union.

**Umbrella thesis**: every advanced physics simulation in the world
should ship as a URL. The browser/WebGPU layer is what's novel; the
chemistry/physics/model architecture is textbook. **Hand-write only the
novel layer; port everything with a peer-reviewed reference.**

For `zero-tvm` specifically, the novel contribution is the
**hand-written WGSL stack** — proving you can match TVM's autotuned
kernels in 37 files of WGSL and 2k lines of TypeScript, by
out-fusing TVM's default pipeline. The Phi-3 architecture, MLC weight
format, and BPE tokenizer are textbook and ported.

---

## 1. Single source of truth for quantitative claims

All measured numbers for `zero-tvm` live in **one** canonical place:

- `BENCH.md` — every head-to-head vs WebLLM (dispatches/token,
  tokens/s, JS bundle, total kernels, WGSL LOC, KV-cache mode,
  identical hardware/weights). Every cell links to the underlying
  artifact in `tests/results/`.
- `README.md` § What's actually in the box — derived from `BENCH.md`,
  cross-linked back to source files in the repo.

Anywhere else (`CLAUDE.md`, `RESEARCH.md`, `index.html`, badges,
landing page) may *summarize* numbers but never *introduce* new ones.

**If a number isn't in `BENCH.md`, it isn't measured.**

Before stating a measurement anywhere:

  protocol → run head-to-head against WebLLM on identical hardware → commit JSON artifact → add `BENCH.md` row → quote

Not the other way around. Every claim is meant to be **head-to-head
falsifiable** by anyone who runs the same hardware against
`zerotvm.com` vs `webllm.mlc.ai`.

---

## 2. Falsifiable JSON artifacts back every claim

Path: `tests/results/YYYY-MM-DD/<id>.json`.

Shape (locked; don't add top-level keys without updating the harness):

```json
{
  "meta":     { "protocol": "...", "hypothesis": "...", "passBar": "...",
                "seed": "named-seed-id", "warmup": 5, "trials": 20 },
  "env":      { "gitSha": "...", "userAgent": "...", "adapter": {...},
                "limits": {...}, "timestamp": "2026-05-14T...",
                "shaderHashes": {"matmul_wgsl": "...", "qkv_fused_wgsl": "...",
                                 "attention_wgsl": "...", "fused_ffn_wgsl": "...",
                                 "add_norm_wgsl": "..."} },
  "rows":     [ { /* per-cell measurements, including head-to-head WebLLM cells */ } ],
  "status":   "pass" | "fail" | "noisy" | "partial",
  "diagnosis": "first-failing-cell + smoking-gun explanation"
}
```

Re-runnable deterministically given fixed seed + identical GPU + same
shader hash. fp16 GEMM and fp16 reductions are NOT order-deterministic
across GPU vendors — same WGSL on different hardware (Apple Metal vs
Nvidia Vulkan vs Intel iGPU) yields statistically equivalent logits
(top-k mass within ε of HF reference) but not bit-exact;
`shaderHashes` lets reviewers group rows correctly.

---

## 3. Status labels are first-class

- **`pass`** — meets the protocol's pass bar (top-k match vs HF
  reference logits, throughput within band of WebLLM, etc.).
- **`fail`** — doesn't. Commit anyway with a `diagnosis` field naming
  the first failing cell and the smoking gun. **Never silently rerun
  until pass.**
- **`noisy`** — `std/median > 0.1` on any cell. Informational, not
  pass/fail.
- **`partial`** — some cells pass, others don't; explicit `N of M`
  count in the diagnosis.
- **`honest negative`** — failures that are evidence. `CLAUDE.md §
  Known gaps` cites the artifact and the rejected hypothesis. The
  2026-06 "22% behind WebLLM" headline was an honest negative encoded
  as a pass-bar miss with documented root cause; it stood until the
  2026-07-25 M2 Max re-measurement (correctness fixes + vec4-load
  defaults) flipped it to ~28% ahead — the negative's full arc is
  preserved in BENCH.md, not retconned out.

Honest negatives become the project's evidence base. They are not
bugs to fix; they are findings.

---

## 4. Reproducibility (no randomness left to chance)

- `Math.random()` is **banned** in any test/experiment path. Sampling
  uses argmax (deterministic) or seeded top-p with a named seed.
  WGSL random draws (none in the current forward pass) would use a
  uniform-routed seed channel.
- Every JSON artifact records: git SHA (when available), full
  `navigator.userAgent`, `adapter.info`, WebGPU `limits`, UTC ISO8601
  timestamp, **shader-file SHA-256 / git-rev-parse hashes** for each
  of the 10 kernel roles (37 WGSL files counting A/B/int8 variants).
- 5 warmup samples are discarded; 20 trials retained.
- Report **median + p10/p90/p99 + std + IQR** for throughput
  measurements — never single-shot.
- If `std/median > 0.1` on any cell → label the artifact `"noisy"`.

---

## 5. GPU timing requires a forced sync

`performance.now()` deltas around `queue.submit` alone are fiction —
WebGPU is asynchronous. **Mandatory pattern**: a mapped readback of a
tiny buffer (a single `f32`) before AND after the work. The throughput
counter shown live in the chat UI uses this pattern; the offline
benchmark harness in `tests/perf/` likewise. WebLLM's reported
throughput uses the same pattern, so head-to-head comparisons are
apples-to-apples.

---

## 6. Multi-level correctness verification

Match against more than one reference frame. Listed in increasing
sophistication / decreasing strength:

1. **Closed-form invariants**: norm preservation across RMSNorm,
   softmax mass = 1 across attention, KV-cache append idempotence,
   tokenizer round-trip identity on UTF-8 corpora.
2. **CPU pre-screen**: every WGSL kernel has a TypeScript CPU
   reference (`src/cpu/*.ts`) that runs on the same inputs. The CPU
   path catches algorithmic bugs *before* the WGSL even reaches the
   GPU — this is `zero-tvm`'s back-port-worthy contribution to the
   sibling discipline.
3. **Peer-reviewed reference packages**:
   - HuggingFace `microsoft/Phi-3-mini-4k-instruct` in PyTorch fp16
     as the bit-comparable reference for *logits* (top-k mass match
     within ε of fp16 noise floor).
   - MLC q4f16_1 quantized weights as the bit-comparable reference
     for *weights*. Both `zero-tvm` and WebLLM read these from the
     same OPFS cache.
   - **WebLLM (`mlc-ai/web-llm`) as the head-to-head reference** —
     same hardware, same weights, same prompt. The head-to-head
     headline (~28% ahead as of the 2026-07-25 M2 Max run; 22%
     behind before it) is the falsifiable claim.
4. **Experiment**: human-readable output equivalence on a fixed
   prompt corpus, scored against WebLLM's output for "did this
   answer the question equally well."

Multiple independent reference frames > one. Each artifact should
state which it's checking against in `meta.hypothesis`.

---

## 7. Port from references; hand-write only the novel layer

This is the architectural rule. The differentiator of `zero-tvm` is
the **hand-written, fused WGSL kernel stack** — proving you can match
WebLLM's TVM-autotuned 85 kernels with 10 hand-written kernel roles
by out-fusing the default emission pipeline. So:

- **Hand-written and owned** (the contribution):
  - All 10 WGSL kernel roles / 37 files: `qkv_fused.wgsl` (Q/K/V proj
    + RoPE + paged-KV append in one dispatch), `attention.wgsl`
    (paged attention + page-table read), `fused_ffn.wgsl` (gate + up
    + SiLU), `add_norm.wgsl` (residual + RMSNorm), `matmul.wgsl`
    (int4-dequant GEMM + subgroup/tiled variants), `rmsnorm.wgsl`,
    `embed.wgsl`, `sample.wgsl`, etc.
  - The 228-dispatch (f16 KV) / 260-dispatch (int8 KV) per-token
    schedule that beats TVM's 342.
  - All ~2k lines of TypeScript engine, tokenizer, and weight loader.
  - The OPFS cache layer over HuggingFace's CDN.
- **Ported from peer-reviewed source with attribution**:
  - **Phi-3 architecture spec** (32 layers / 32 attention heads /
    32 KV heads / 96 head dim / SwiGLU / RoPE / RMSNorm /
    grouped-query attention) from Microsoft's released
    `Phi-3-mini-4k-instruct` model card and `config.json`.
  - **MLC q4f16_1 quantization scheme** and `ndarray-cache.json`
    weight format from `mlc-ai/mlc-llm` / `mlc-ai/web-llm`, so
    `zero-tvm` and WebLLM read the same on-disk weights.
  - **BPE tokenizer** patterns from `huggingface/tokenizers` /
    `sentencepiece`.
  - **RoPE / GQA / SwiGLU** reference numerics from the original
    papers (Su et al. 2021 RoFormer, Ainslie et al. 2023 GQA,
    Shazeer 2020 GLU).
  - **Apache-TVM kernel patterns** referenced (not copied) as the
    head-to-head baseline; the comparison is precisely *against*
    that pipeline's emission.

**Per-file header** for ported code:

```
// Ported from <upstream> (<upstream-url>), <license> license.
// Source: <relative-path> at commit <SHA>
// Original authors: <upstream/AUTHORS>
// Adaptations for zero-tvm:
//   - <substantive change 1>
//   - ...
// See LICENSE-<UPSTREAM> at repo root for the <license> notice.
```

**Repo-level**: `LICENSE-MLC` and `LICENSE-PHI3` at root (verbatim
from upstream). Per-module status table belongs in a `MIGRATION.md`
table:

| module | reference | license | status |
|---|---|---|---|
| `tokenizer.ts` BPE | `huggingface/tokenizers` | Apache 2.0 | 🟢 |
| weight loader | MLC `ndarray-cache.json` format | Apache 2.0 | 🟢 |
| Phi-3 architecture | Microsoft Phi-3 model card / config | MIT | 🟢 |
| WGSL kernels | hand-written (this repo) | MIT | n/a |

License compatibility: MIT + Apache 2.0 work together — the ported
portion keeps its upstream license obligations (notice + state
changes); the rest of the repo (including all hand-written WGSL)
stays MIT.

---

## 8. No fudge factors without a citation

Any tunable scalar in production code that isn't backed by a
peer-reviewed source is:

1. **Labeled empirical** in the code comment at point of use.
2. **Documented in `CLAUDE.md` § Known gaps** with the magnitude of
   the empirical correction and what observable it was tuned against.
3. **Queued for removal** once the structural fix lands.
4. **Tracked in `CHANGELOG.md` / commit messages** when added and
   when removed.

`zero-tvm` aims for **zero fudge factors in the forward pass** —
every numeric scale (head-dim scale, RMS epsilon, RoPE base, softmax
temperature, int4 dequant zero-point convention) is sourced from
Phi-3's released config and the MLC quantization spec. The
empirical knobs are kernel-tile sizes (subgroup vs scalar, A/B
variants of `matmul.wgsl`), which are auto-selected at init based
on `adapter.info` — not hidden physics constants.

Tested-and-rejected hypotheses (e.g., "fusing all of `qkv_fused`
will exceed register pressure on Apple M2" — falsified by the
228-dispatch result) go into the same documents so future sessions
don't re-test them.

---

## 9. Shader byte-hashing for reproducibility

Every artifact records the SHA-256 (or `git rev-parse <gitSha>:<path>`
short hash) of each of the 37 WGSL shader files the experiment
depended on. This lets reviewers group rows by shader version when a
kernel implementation changes (subgroup tile size, int4 dequant
strategy, fused-vs-unfused FFN, A/B variant selection, …).

The `env` block carries `shaderHashes: { matmul_wgsl: "...",
qkv_fused_wgsl: "...", attention_wgsl: "...", fused_ffn_wgsl: "...",
add_norm_wgsl: "...", ... }`.

---

## 10. Living open-gaps document

`CLAUDE.md § Known gaps` at repo root lists each open issue as:

```
## N. The <observable> deficit vs <reference> (<artifact>, <date>)

Observed.  <quantitative gap with σ-significance>

Hypothesis A — <candidate root cause>
Hypothesis B — <alternative>

Falsification experiment: <what would distinguish them>
```

Standing entry (**closed 2026-07-25**): the 22% throughput gap
behind WebLLM on M2 Pro, identical-weights identical-prompt. Closed
by correctness fixes (fused_ffn f32 accumulation, attention barrier,
decode off-by-one) plus promoting the measured vec4-load kernels to
default; the M2 Max head-to-head now reads ~28% *ahead* (BENCH.md).
Hardware changed too (M2 Pro → M2 Max), so per-hypothesis attribution
is not fully resolved; the surviving open sub-question — split-K
attention at long context — is queued in BENCH.md.

Entries are removed when the underlying gap closes; the artifact
references stay in `CHANGELOG.md`. Tested-and-rejected hypotheses
get a strikethrough entry with the refutation artifact link, so
the same hypothesis isn't tried twice.

---

## 11. Honest self-corrections

When a prior claim turns out wrong, revise it **in the same commit
that surfaces the data**, with the full arc preserved. Examples:

- "Zero-TVM matches WebLLM throughput" — false on initial measurement.
  README and BENCH.md were updated in the same commit as the
  head-to-head artifact to read "22% behind WebLLM" instead. The
  earlier overclaim is preserved in `CHANGELOG.md`. (The arc
  continued: the 2026-07-25 M2 Max re-measurement flipped the
  headline to ~28% ahead, recorded the same way — measurement first,
  claim second.)
- "228 dispatches/token" was the result after fusing QKV+RoPE+KV-append
  into one dispatch; the earlier 3-dispatch-per-layer count is
  archived rather than retconned out.
- "Hand-written kernels can't beat TVM on dispatch count" — refuted
  by 228 vs 342. The fusion contribution is the answer; documented
  in `README.md` § "What's actually in the box".

This is publication-grade transparency. **Wrong hypotheses become
part of the public scientific record, not an embarrassment to
hide.**

---

## 12. Citation infrastructure per release

Each minor release ships:

1. Git tag (`v0.X.Y`)
2. GitHub Release with notes drawn from `CHANGELOG.md`
3. **Zenodo DOI** minted via the GitHub-Zenodo integration
4. `CITATION.cff` `preferred-citation` block updated with the real
   DOI

Patch releases (doc-only, refactor, etc.) skip the Zenodo step.

---

## 13. WebGPU gotchas (carry forward across all projects)

- `initGPU()` MUST pass `requiredLimits` for
  `maxStorageBufferBindingSize` and `maxBufferSize`. The default
  128 MiB cap silently truncates large dispatches; Phi-3-mini's
  full weight set exceeds this without explicit limits.
- `atomicAdd` works only on `u32` — not f32. The forward pass
  avoids atomic reductions entirely (tree reductions in shared
  memory instead).
- No recursion in WGSL. All shaders are single-pass.
- Uniform buffers must be 16-byte aligned.
- No subgroup intrinsics in WebGPU 1.0 spec — `zero-tvm` ships A/B
  variants of `matmul.wgsl` (subgroup + scalar) and selects at init
  based on `adapter.info`. Once WebGPU 1.1 ships subgroup
  intrinsics, the A/B split collapses.

---

## 14. Test discipline (non-negotiable)

- TypeScript `strict` + `noUncheckedIndexedAccess`. No exceptions.
- ESLint clean — 0 errors. Warnings tracked, ideally 0.
- CI green. Every PR runs unit + Playwright e2e + typecheck + lint.
- Each kernel has paired test coverage by **intent**, not by metric:
  - **Closed-form invariant** (norm preservation, softmax mass = 1,
    tokenizer round-trip) where it exists.
  - **CPU pre-screen** (TypeScript reference under `src/cpu/`) on
    every kernel — falsify on CPU before shipping the WGSL.
  - **Peer-package** (HuggingFace Phi-3-mini fp16 logits, MLC
    quantized weights, WebLLM throughput) on a fixed prompt.
- Honest negatives (status: "fail" tests) live alongside passes; they
  don't break CI but they're surfaced in the suite output.

---

## 15. Release cadence

- **Minor releases** (`v0.X.0`) for substantive features or
  scientific findings. Tag + GitHub Release + Zenodo DOI.
- **Patch releases** (`v0.X.Y`) for doc-only, refactor, SVG refresh,
  narrative updates. Tag + GitHub Release, no DOI.
- **CHANGELOG** follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  format: `### Added / Changed / Fixed / Documented / Honest negatives`.
- **CITATION.cff version** matches `package.json` version matches
  Git tag matches GitHub Release tag, all pinned per release.

---

## On adding a new sibling project

Inherit these 15 principles from day one. Copy this file verbatim into
the new repo. Replace project-specific references in sections 1, 6, 7,
8, 10 with the new project's analogs. Cross-link sibling projects in
the header.

The discipline is the product.

---

*Last revised: 2026-05-14. Canonical mirror of
[`webgpu-q/RESEARCH_STANDARDS.md`](https://github.com/abgnydn/webgpu-q/blob/main/RESEARCH_STANDARDS.md).
Edit either and propagate.*
