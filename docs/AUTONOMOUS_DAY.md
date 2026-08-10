# A day of work, unattended

A queue that can be run start to finish without a human in the loop, ordered so
that every item is **verifiable before the next one depends on it**. Written
2026-08-10, after a session whose recurring failure was not bad code but
plausible assumptions nobody checked — so every entry below names its gate, and
an item whose gate cannot run is **blocked, not skipped**.

Read `CLAUDE.md` first. Nothing here overrides it.

---

## The goal

**Run DeepSeek-V2-Lite in a browser.** It would be the first MLA model to do so.
Everything else in the queue is either a prerequisite, or something worth doing
while a long verification runs.

State at the time of writing: the MLA kernels are verified against real weights
(seven stages, f16 floor), yarn is in, the unquantized router is in, the chat
template is in, mixed dense/MoE stacks are expressible in spec and loader. What
is missing is **engine integration**, in two pieces.

---

## Guardrails

1. **Do not push or deploy without explicit approval.** Committing locally is
   fine and expected. `git push` is not.
2. **Never claim a green that was not run.** If a gate could not execute, say
   which and why, in the commit message.
3. **One GPU.** Kernel tests, real-weight bundles and browser e2e all contend.
   Run them serially. Two engines on one GPU has hung a test outright.
4. **Watch machine health.** This Mac degraded once from repeated browser e2e
   runs: free RAM hit ~4%, then `dscl`/`getpwuid` began returning
   `eServerError`, which broke git SSH signing *and* Chrome launching. Symptom
   to watch: `id -un` returning a numeric uid. If it appears, stop browser work,
   report, and do not try to fix it — the fix is a reboot, which is the user's.
   Check with `memory_pressure | tail -2` between e2e suites.
5. **Bandwidth is scarce.** Never `hf download` a whole checkpoint. Use
   `scripts/pull-tensors.mjs --dry-run` first; it is resumable and chunked.

---

## The queue

### 1. MLA into the engine — attention path

**Do:** wire `mla_scores` / `mla_combine` / `mla_proj` into `recordForward`
beside the attention and GDN paths, with a latent KV layout (576 values/token:
512 latent + 64 shared RoPE key) instead of per-head K/V. Add the MLA dims to
`ModelSpec`. Do the load-time prep in the loader: `kv_b_proj` dequantized to f16
with its K half transposed, and the pe rows of `q_proj`/`kv_a_proj` permuted.

**Gate:** a new `real-weights.mjs` handler that drives the **engine's own** layer
against `.weights-local/kernel-refs/dsv2layer0`, the way `qwen36Layer` does —
not the hand-driven dispatches `dsv2Layer` uses today. It must reach the same
~3e-4 per stage.

**If blocked:** if the latent layout cannot reuse the page machinery without
disturbing existing models, put it *beside* them rather than generalising the
pages — a regression in Phi-3's cache costs more than a second code path.

**Watch for:** `layout:'auto'` drops bindings an entry point never reads. That
has bitten this repo twice; a six-buffer bind group against a four-binding
layout is rejected outright.

**Follow `docs/MLA_ENGINE_PLAN.md`** — eight ordered steps with their own gates,
and a Part 0 that resolves four contradictions between plausible designs. Two of
its claims were re-verified by hand: `headDim` is 128 (192 makes the engine read
a 384-word o_proj row against a 256-word one), and widening `layerKinds` to
carry `'mla'` compiles with zero errors while silently zeroing the KV budget.

### 2. Mixed dense/MoE stack — engine half

**Do:** `spec.ffnKinds` / `ffnWidthAt` already exist and the loader already
branches on them. `engine-core` still sizes FFN buffers and dispatch shapes from
a single `S.ffn`. Build both and choose per layer.

**Gate:** DeepSeek layer 0 is dense at 10944 while every other layer is MoE at
1408, so the layer-0 bundle exercises the dense half directly. The MoE half
needs the layer-1 bundle (item 5).

**Do not** let `denseLayers` default its width. It throws today, on purpose:
silently reusing `moe_intermediate_size` builds the dense layer's buffers eight
times too small.

### 3. BENCH protocol round

**Do:** `npm run bench` for the models shipping blank rate labels — `llama32`,
`qwen3mlx`, `qwen30b`, and a protocol round for `qwen36q3` (its ~55 t/s is a
single owner run, not a protocol number). Then `npm run bench:sync`.

**Gate:** BENCH.md's own protocol — same session, interleaved, and a drift
control. Discard any pair whose control moved more than the effect.

**Why here:** it is the only item that wants a *quiet* machine, so it goes
before the browser-heavy work rather than beside it.

### 4. Sampler throughput

**Do:** measure what the top-p bisection costs. It re-reads the vocabulary 31
times inside one workgroup, and nothing was written into BENCH.md because
nothing was measured.

**Gate:** decode rate at `?temp=0` versus `?temp=0.8&topp=0.95` versus
`?temp=0.8` (top-p off, which skips the search entirely), same model, same
session. If the search costs more than ~10%, note it and consider a
count-based cutoff instead of a threshold search — do not optimise blind.

### 5. The MoE layer bundle — needs a Colab session

**Do:** run the remaining cells of `docs/colab/build-bundle.ipynb` — the 329 MB
pull, the reference, the zip. Bring back ~50 MB.

**Gate:** the notebook prints the routing line; the reference cross-checks
latent-vs-naive and refuses to write a bundle if they disagree.

**Blocked without:** a browser session on Colab. If unavailable, items 1-4 and 6
proceed; item 2's MoE half and any whole-model DeepSeek run wait.

### 6. Whole-model DeepSeek — the finish line

**Do:** register the spec, run `?model=deepseekv2` in the browser.

**Gate:** `scripts/validate-model.mjs` against an mlx_lm reference — cosine
≥0.999 and greedy token-exact, the same bar every other model passed. **Build
the reference in f32** (`scripts/mlx-ref.py` does this now); a bf16 reference is
noisier than the engine it grades.

**Blocked without:** the full 9 GB checkpoint locally, or a Colab-produced
whole-model reference. This is the one item that cannot be faked with a layer
bundle, and it is the one that decides whether the headline is true.

---

## While a long run is going

Verification runs are minutes. Useful things that do not touch the GPU:

- **The `rope_parameters` rename.** Newer MLX configs spell rope scaling
  `rope_parameters`, not `rope_scaling`. The detector would miss it entirely and
  generate a spec with no scaling — the same shape as the yarn-codegen bug found
  today. Add it to the reader and to `CONFIG_KEYS_READ`.
- **`createEngine` has no `destroy()`.** GPU buffers live as long as the page,
  which the library's own report flagged.
- **`gate.test.ts` is flaky**, and has been for weeks. Its own comment admits
  the `wipeOpfs`-vs-boot-probe race. Fix the race or quarantine it explicitly.
- **`qwen36 --check-only` is red on vision-tower `.bias` records** — detection
  over-reach, open since the Llama sprint.

---

## What needs a human, and should be asked for once, early

- **A push or a deploy.** Everything else here is local.
- **A Colab session** (item 5).
- **A second machine** for the split test. Every distributed number in this repo
  is loopback; a `cloudflared` tunnel plus a laptop on a phone hotspot converts
  "six browsers on one Mac" into a defensible claim, and costs an afternoon.

---

## How to stop

Stop and report — do not improvise past — when:

- a gate fails twice for different reasons (the second failure means the model
  of the problem is wrong, not the fix)
- `id -un` returns a number (see guardrail 4)
- an item needs a decision whose wrong answer would be expensive to unwind,
  such as changing the KV page layout for every existing model
