# Verification doctrine

Why this file exists: on 2026-08-19 twelve defects were found in one day, and
in almost every case **the check already existed and was correct.** It was
applied at the wrong scale, at the wrong level, or was printed and never
asserted. That is a property of how the gates were written, not of how carefully
anyone was working, so it will recur unless the gates change shape.

This is not a style guide. Every rule below is derived from a specific defect
that shipped here, or from one the assistant made while fixing them.

---

## The failure this codebase actually has

**Nothing crashes.** A wrong kernel, a wrong prompt, a wrong scale factor and a
wrong cache all produce fluent, plausible text. The Phi-3 `ropeFreqs` P0 ran
broken for ten days through two deploys. `kv_quantize_int8` wrote half of every
head-dim-256 row and produced *fluent wrong text* rather than noise.

Everything below follows from that: a test that asserts "it ran" or "it returned
something" asserts nothing here.

---

## Rule 0 — something that looks like a check is not a check

**Every defect found here has been something occupying the position of a
verification without performing one.**

That is the whole diagnosis. The five rules below are its prescriptions, and
they are worth less than this sentence, because the sentence generalises to
gates nobody has written yet.

The evidence is unusually good, because it comes from two independent sources
that produced the *same* taxonomy: the twelve defects that shipped in this
codebase, and roughly twenty mistakes the assistant made while fixing them in a
single day. Two different authors, two different kinds of work, one shape.

Four families. Every instance from both sources is one of them.

### 1. Right assertion, wrong inputs

The check is correct and runs where the defect cannot appear.

| | instance |
|---|---|
| out-of-domain scale | `chunk-prefill-test` asserts chunked ≡ per-token — at 150 tokens, which is ONE chunk. The defect needs sixteen. |
| non-discriminating fixture | every ChatML fixture ended with a real user query, the one shape where the right and wrong `<think>` rules render identically |
| unreached subject | `renderToolResults` was correct and tested; nothing called it |

**A test asserts P(x) for the x it happened to use, and nothing states the
domain.** Green means "not here", not "not anywhere".

### 2. Right inputs, wrong subject

Something adjacent to the claim gets asserted instead of the claim.

| | instance |
|---|---|
| restated copy | a test that re-implemented the formula it was checking, and stayed green when the real one was mutated — then the same mistake again, on a different function, within the hour |
| proxy metric | the e2e gate asserted `reused > 0`, which a snapshot mislabelled by one token also satisfies while corrupting a recurrence |
| lying instrument | a refused HTTP request scored as "the model answered with prose"; a facts checker reporting "0 errors" from a mode that executes nothing |

**A restatement, a downstream number, or the tool itself — none of them is the
subject.**

### 3. Produced but never compared

The value exists, is displayed, and nothing checks it against what it claims.

| | instance |
|---|---|
| unasserted value | the eval PRINTED the engine's flags in its banner and never compared them to the flags it had requested |
| silent fallback | `hosting.includes('')` is true for everything, so every model was bounded by Phi-3's ceiling; `MODEL=qwen30b` rendered the default character |
| duplicated rule | `int8KV` expressed in variants AND options — four surfaces ran f16 while their own flags said int8 |
| authored derivation | the tool dialect prefix-matched on the model's NAME, wrong for both Qwen3.5 builds |

**Producing a value and validating it are different acts, and the first
reliably masquerades as the second.** A fallback is the sharpest case: it
converts a question into a confident wrong answer.

### 4. The conclusion outran the evidence

| | instance |
|---|---|
| undated claim | throughput measured before int8 became the default; a planning doc still calling the old figure "today" |
| hypothesis as evidence | a rewind-ring fix wrong on all three of its premises; M-RoPE, attention-over-distance and int8 quantisation each proposed as the root cause and each wrong, at one GPU run apiece |

**A measurement is a function of a configuration, and publishing the value
discards the domain.** A plausible mechanism is not evidence. Bisection is.

### Using it

Four questions, asked of any gate before trusting it:

1. **Domain** — at what size and shape does this run, and at what size and shape
   does the thing ship? If they differ, say by how much.
2. **Subject** — does it assert the thing, or a copy of it, a number downstream
   of it, or the tool that measures it?
3. **Comparison** — is the value compared against something, or only produced?
   Anything printed and not compared is decoration.
4. **Conditions** — under what configuration was this true, and is that still
   the configuration?

A gate that cannot answer all four is a gate whose green is uninformative.

---

## The five rules — the prescriptions

### 1. Scale is part of the claim

A gate that asserts an equivalence must run at a size where the two sides could
actually diverge, and must **refuse to pass** at a size where they cannot.

| gate | asserted | ran at | shipped at |
|---|---|---|---|
| `chunk-prefill-test` | chunked ≡ per-token | 150 tokens = **1 chunk** | 16+ chunks |
| `gdn_chunk_chain` | chunk ≡ stepwise, bit-exact | **6 tokens** | `CHUNK_CAP` 1024 |
| `attention_prefill` | chunk ≡ per-token, bit-exact | **5 tokens** | 1024 |
| `validate-model` | logits ≡ mlx_lm | **~20 tokens** | 32k context |

All four were green while chunked prefill was corrupting output at 16k. The
defect degrades *monotonically with chunk size* — exactly the axis none of them
varied.

`chunk-prefill-test` now fails on a single chunk. Do the same for any new
equivalence gate: state the dimension it is blind along, and make the blind
configuration a failure rather than a pass.

### 2. Assert the output, not the metric

The e2e multi-turn gate asserted `reused > 0`. A snapshot mislabelled by one
token also satisfies `reused > 0`, while replaying a token into a non-idempotent
recurrence. The metric was green; the output was corrupt.

Prefer, in order: **token identity** > logits cosine > a scalar the engine
reports about itself. The last one is a proxy for a proxy.

### 3. Test at the definition, not next to it

`renderToolResults` wrapped tool results correctly, had unit tests, and had
**zero callers** for weeks. The hosts each hand-rolled the wrong thing.

Two consequences:

- Test the **caller's output**, not the helper. `host-normalize.test.ts` asserts
  what the host produces; `tool-calls.test.ts` asserts what the renderer
  produces. Only the first would have caught it.
- If a function cannot be reached by a unit test, **move it** until it can.
  `normalize()` left `agent-native.mjs` (which boots a GPU on import), the argv
  builder left `station.mjs` (which binds a port on import), and
  `kvBytesPerTokenShown` left `landing.ts` (which touches the DOM). Each move
  was made *because* the test could not otherwise exist.

A test that **re-implements** its subject to check it is not a test. The first
version of `kv-figure.test.ts` copied the formula out of `landing.ts`; mutating
the real one left the copy green. The mutation gate caught it in one run.

### 4. Derive, never author

Every hand-typed value that could have been derived from a checkpoint or a spec
has drifted here at least once: the tool dialect (prefix-matched on model *name*,
wrong for both Qwen3.5 builds), the quantisation label, the KV figure, the
template id, the RAM warning.

If a checkpoint knows it, read it from the checkpoint. If the spec knows it,
compute it. If neither does, the value is a **measurement** and belongs in
`facts.json` with a command that reproduces it.

### 5. Fixtures must be able to tell the answers apart

Every ChatML fixture in the suite ended with a real user query — the one shape
where the correct and incorrect `<think>` rules render **identically**. The
fixtures could not discriminate, so they passed under both.

Before trusting a fixture set, ask: *what wrong implementation would also pass
this?* If the answer is "the one we had", the fixtures are decorative.
`scripts/gen-toolcall-fixtures.py` generates the discriminating case from the
real templates.

---

## Mechanisms

| mechanism | what it enforces |
|---|---|
| `scripts/mutation-gate.mjs` | reinstates 8 shipped bugs, asserts the suite goes red. A green suite looks the same whether it is checking something or not. |
| `scripts/render-diff.py` | our rendered prompt vs the checkpoint's own jinja — `--shapes` for awkward inputs, `--plain` for chat, `--depth` for length |
| `scripts/depth-bisect.mjs` | turns off one length-dependent subsystem at a time |
| `scripts/needle-test.mjs` | can the model still see the start of its context from the end |
| `scripts/agentic-eval.mjs` | does the tool loop survive depth |
| `scripts/release-check.mjs` | runs what it can, and reports what it cannot as UNRUN — never as a pass |

**UNRUN is not a pass.** Anything needing a GPU is reported, with the command,
and never silently skipped.

---

## How to debug a silent failure here

Derived from the qwen38 investigation, which took a day and where **every
hypothesis was wrong and the bisection was right.**

1. **Establish it is ours.** Run the same conversation through the reference
   (`mlx_lm`) with the checkpoint's own template. If the reference is correct and
   we are not, it is ours. If both fail, stop — it is the model.
2. **Clear the prompt.** `render-diff.py` against the vendor jinja, at the depth
   that fails. Byte-identical or it is the prompt.
3. **Bisect the subsystems.** One flag at a time (`--kv8 0`, `--reuse 0`,
   `--chunk 0`), at a depth that *fails*. Never at a depth that passes: an arm
   that also passes there has shown nothing.
4. **Bisect the parameter.** Once a subsystem is named, sweep its size
   (`--cap 256/1024/4096`). A monotonic gradient localises the mechanism;
   a threshold gives you a workaround the same afternoon.
5. **Only then form a hypothesis.**

Hypotheses formed before step 4 during that investigation: M-RoPE layout,
attention over distance, int8 KV quantisation noise. All three were wrong, and
each cost a GPU run. The CAP sweep found the answer in three minutes.

**Prefer the cheap arm.** The per-token arm at 16k costs 46 minutes because
prefill runs at ~6 tok/s; the cap sweep costs one minute per arm because both
arms stay on the fast path. When an arm is slow, cap the *work* (`STEPS=1`), not
the *depth* — changing the depth changes what the comparison means.

---

## Harnesses are product code

Three separate times in one day a harness reported something confidently wrong:

- a refused HTTP request (409, engine busy) was scored as *"the model answered
  with prose"*
- the reference was run on a conversation the failing model never reaches
- `"did it call a tool?"` scored a **hallucinated** tool name as success

Every eval harness must: assert its preconditions (engine ready, and the flags it
requested are the flags it got), fail loudly on a malformed response rather than
coercing it into a datum, and name a control that must pass — if the control
fails, the run says nothing.

---

## Numbers

A published number is a claim, and an unfalsifiable claim is not a measurement.

`/health` reported `3803 tok/s` prefill for a 27.8B model — **211 TFLOP/s** on a
machine that peaks near 13. Nothing bounded it, and it was one step from
`BENCH.md`. It was wrong in the flattering direction, which is the direction
nobody double-checks.

Before publishing any throughput figure, check it against both rooflines:

```
decode  ≤ memory bandwidth / bytes read per token
prefill ≤ peak FLOP/s / (2 × params)
```

And check *when* it was measured: int8 KV became the default 2026-08-18 and
disables split-K; the chat template changed 2026-08-19. Any number older than a
default it depends on is stale, not wrong — and must say so.
