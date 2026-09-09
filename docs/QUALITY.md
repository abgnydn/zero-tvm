# Quality vs fidelity

Almost everything this repo verified before 2026-08-10 was **fidelity**: does
the engine compute the same numbers as the reference? Very little of it was
**quality**: is the model any good?

They are not the same question, and the gap has a precise mechanism.

> **Correction.** I first told the user "nothing in this repo measures quality."
> That was wrong, and an audit caught it. `tests/e2e/zero-tvm.test.ts:160-167`
> (mirrored in `qwen`, `qwen35`, `llama32`, `multi-turn`) asserts on generated
> text with **no reference in the loop** — Paris, 42, `len`, "yes", and 2 of 8
> colour words. That is a real quality gate and it would go red on gibberish.
> Its limits are what matter: five lexical checks, **4 of 9 shipped models**,
> **neither MoE build has an e2e file at all**, and `.github/workflows/ci.yml`
> cannot run any of it (*"e2e tests use Puppeteer + WebGPU which Ubuntu runners
> don't expose"*). So no automated gate has ever asserted output quality on any
> model, and nothing at all has asserted it on `qwen36q3`.

## The hole

`scripts/validate-model.mjs` is the strongest existing gate. It checks argmax,
cosine ≥ 0.999, top-5 overlap, and greedy token-exactness for 8 tokens against
`scripts/mlx-ref.py`. And `mlx-ref.py:29` is:

```python
model, tokenizer = load(args.model)
```

`mlx_lm.load` on **the same quantized checkpoint the engine is running**. The
comment two lines down confirms it: only the non-quantized tensors upcast to
f32, the quantized weights stay packed.

So quantize a checkpoint until it emits gibberish, and MLX emits the same
gibberish:

| check | result on a model quantized into nonsense |
|---|---|
| `argmax` matches | ✅ passes |
| `cosine ≥ 0.999` | ✅ passes (~0.99998) |
| `top-5 overlap ≥ 4` | ✅ passes |
| `greedy token-exact, 8 tokens` | ✅ passes — same gibberish, token for token |

Every gate green. Not a defect in those gates — it is exactly what they are
built to measure, and `validate-model.mjs:16` states its scope honestly:
*"Exit 0 = the generated spec computes what mlx_lm computes."* The problem is
that the prose around it ("Numerical trust", "worth believing") carries more
weight than that line admits.

**This was verified by doing it**, not reasoned about. Requantizing
`Llama-3.2-1B-Instruct-4bit` to 2-bit and running `mlx-ref.py` against it:

| checkpoint | `argmax` | `greedy_text` |
|---|---|---|
| shipped 4-bit | 791 | `The capital of France is Paris.<\|eot_id\|>` |
| 2-bit requant | 7561 | `-a-a-m-mowcarecarecarecarecarecarecare` |

The second row **is the reference**. An engine faithfully reproducing that
gibberish passes all four checks and the script prints `model validates against
mlx_lm`.

## What closes it

Perplexity, which is **absolute** — it needs no reference model, only held-out
text.

### For comparing two checkpoints — `scripts/quality-ab.py`

**This is the one to reach for.** The quality question is about the *weights*,
not the engine, and the engine's fidelity to whatever weights it is given is
already pinned to ~1e-4. So measure on the reference, where it costs seconds.

```bash
cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/quality-ab.py \
    --a ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-4bit \
    --b ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp
```

Independent windows, identical for both sides, with error bars and a z score.
Verdicts: within +10% with separated bars → ship; +10–25% → a task benchmark
decides, not this; >+25% → don't.

**Validated by making it fail, at four severities — including a real MoE.**
`scripts/requantize.py` builds a deliberately-degraded copy of a checkpoint
already on disk; the A/B then has to separate it. 24 independent windows of 512
tokens, bf16, ~10 s per arm.

| baseline | candidate | ppl | vs base | paired z | B worse on | verdict |
|---|---|---:|---:|---:|---:|---|
| Llama-3.2-1B-4bit | `--scope mlp --bits 3` | 193.6 | +75.8% | 35.5 | 24/24 | DO NOT SHIP |
| Llama-3.2-1B-4bit | `--scope all --bits 3` | 322.9 | +193.7% | 31.8 | 24/24 | DO NOT SHIP |
| **OLMoE-1B-7B-4bit** | **`--scope experts --bits 3`** | **52.3** | **+8.4%** | **14.7** | **24/24** | **SHIP** |
| any | a no-op requant | — | — | — | — | *refused* |

The OLMoE row is the one that matters: **64 experts, top-8, and only the expert
stacks requantized — exactly what `qwen36q3` does**, on a 3.6 GB model that runs
in seconds instead of a 19.7 GB one that does not fit.

And it is informative. Expert-only 3-bit costs **+8.4%**, against +75.8% for
dense-MLP-only 3-bit on a comparable-sized model. Roughly an order of magnitude
gentler — which is the mechanism working as designed: experts are sparsely
activated (8 of 64), so each one's error is diluted, and attention, the router
and the embeddings all stay at 4 bits.

**This is an analogue for `qwen36q3`, not a prediction.** OLMoE is 64 experts
with 1B active; Qwen3.6-35B-A3B is 256 experts with 3B active. It establishes
that expert-only 3-bit is a mild intervention on *a* real MoE and that the
harness can resolve an 8% effect. It does not license a number for the 35B.

### The paired test, which is what makes an 8% effect visible

Both arms score the **identical** windows, so the comparison must be paired.
Between-window variance dominates — some passages are simply harder, and that
difficulty is common to both checkpoints — so an unpaired test spends nearly
all its power measuring the corpus rather than the quantization. On the OLMoE
run, the same 24 windows give:

```
unpaired z = 0.8    NOT DISTINGUISHABLE at this window count
paired   z = 14.7   SHIP: within +10% with separated error bars
```

Same data, same +8.4%. The tool reported the first version and correctly
refused to conclude; differencing per window resolved it. Both are printed now,
and the verdict uses the paired one. The `B worse on N/N windows` count is a
distribution-free cross-check — 24/24 is p ≈ 6e-8 on a sign test alone.

A harness that has only ever passed proves nothing, so this one was pointed at
known-broken weights first.

### The no-op trap, and the guard for it

`scripts/convert-q3-experts.py` selects experts by the literal strings
`.mlp.switch_mlp.` and `.mlp.shared_expert.` — MLX's **fused** expert layout.
Checkpoints converted another way name them `.mlp.experts.<N>.` instead, and on
one of those the script matched **zero** tensors, copied every weight through,
wrote a valid checkpoint, stamped it `requantized: experts->3bit from 4bit`,
and exited 0.

The A/B would then report "NOT DISTINGUISHABLE" — true, and read as *"3-bit is
fine"* rather than *"nothing was quantized"*. That is the whole failure mode
this document is about, one level down.

Both layouts are live **in the same model family**: `mlx-community/
OLMoE-1B-7B-0125-4bit` is `switch_mlp` (144 tensors), its Instruct sibling is
`.mlp.experts.` (9216). Now both scripts count what they convert and a zero is
a hard failure that deletes its own output.

### For the engine — `scripts/quality-eval.mjs`

```bash
# 1. pick the tokens (no GPU, no browser)
node scripts/quality-eval.mjs llama32 --tokens 256 --dump-ids /tmp/ids.json

# 2. reference, over exactly those ids
cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-perplexity.py \
    --model ~/dev/zero-tvm/.weights-local/Llama-3.2-1B-Instruct-4bit \
    --ids /tmp/ids.json --out /tmp/ppl.json

# 3. the engine, same ids, with the comparison
node scripts/quality-eval.mjs llama32 --tokens 256 --ref /tmp/ppl.json
```

Step 3 alone gives the quality number. Step 2 adds a fidelity check that is
much stronger than the existing one — hundreds of scored positions instead of
one prompt's final logits.

### Why not `python -m mlx_lm.perplexity`

It exists, it works, and **for comparing two checkpoints it is the right tool
and needs none of this**:

```bash
uv run python -m mlx_lm.perplexity --model <path> --num-samples 32
```

What it cannot do is score the *same tokens the engine scored* — it samples its
own windows from a HuggingFace dataset. Two perplexity numbers computed over
different text are not comparable, and the comparison is the entire point.
`scripts/mlx-perplexity.py` exists only to pin the reference to a given id
sequence. `mlx_lm.evaluate` (the lm-eval API) is the other half of the ready-made
tooling and is worth using directly for task benchmarks.

## Task benchmarks — `scripts/task-eval.sh`

Perplexity has no right answer in it. Published work puts the accuracy drop at
3-bit near **3x** the perplexity drop, so the +10.4% this repo measured on the
3-bit expert build is consistent with anything from unnoticeable to unusable.
These tasks resolve that.

```bash
bash scripts/task-eval.sh .weights-local/Qwen3.6-35B-A3B-MLX-4bit  qwen36-4bit
bash scripts/task-eval.sh .weights-local/Qwen3.6-35B-A3B-MLX-q3exp qwen36-q3exp
```

| task | limit | what it needs |
|---|---:|---|
| arc_challenge | 400 | grade-school science reasoning |
| winogrande | 400 | pronoun resolution, real-world modelling |
| gsm8k | 100 | multi-step arithmetic — where 3-bit is claimed worst |
| mmlu | 10 x57 | broad knowledge |

Runs on mlx_lm, not our engine, because the question is about the CHECKPOINT —
and the engine is already pinned to mlx_lm at ratio 1.000-1.005 on identical
tokens, so a checkpoint result transfers.

`lm-eval` is layered in with `uv run --with` rather than added to
`~/dev/ml-research`, which is a shared env and not this project's.

**`python -m mlx_lm.evaluate` silently does nothing.** `evaluate.py` has no
`if __name__ == "__main__"` guard, so `-m` imports the module, calls nothing,
and exits 0 with no output at all — including for `--help`. The console script
`mlx_lm.evaluate` is the entry point.

Quote the stderr with any number here. The limits are small on purpose: the
job is to separate two builds, not to publish an absolute score, and these are
not comparable to a leaderboard (different shots, limits, harness version).

## The two corpora

| `--corpus` | what it is | what it answers |
|---|---|---|
| `prose` (default) | `tests/fixtures/quality-corpus.txt` | is this model producing language at all |
| `code` | this repo's own `src/**/*.ts`, sorted | can it write code — the agentic target |

The prose fixture is **written for this repo, not lifted from wikitext or
Gutenberg**. Those sit in the training set of every model here, which deflates
perplexity by an unknown amount. Original text is not truly out of distribution
— nothing in English is — but it removes the worst contamination.

## Reading a result

**Absolute perplexity is only meaningful against another build on the same
tokens.** Compare 4-bit vs 3-bit. Do not compare this number to a published
one: different tokenizer, different corpus, different window.

**And `quality-eval.mjs`'s number is not comparable across `--tokens`.** It
scores one contiguous prefix, so early positions — which have nearly no context
and enormous NLL — dominate, and the mean falls steeply as the window grows.
Measured on Llama-3.2-1B over identical text:

| positions | ppl | ±1σ |
|---:|---:|---:|
| 128 | 151.15 | 31.7% |
| 256 (default) | 65.57 | 20.8% |
| 512 | 41.31 | 13.4% |
| 1198 (whole prose fixture) | 32.18 | 8.3% |

**5× between the ends.** At the 256 default you need a ~59% change to clear 2σ
— fine for gibberish, useless for the 10–20% regression a real quantization
change produces. The script now prints its block standard error and refuses to
let the number be quoted without its `--tokens`. For checkpoint comparisons use
`quality-ab.py`, which windows independently and does not have this problem.

The one absolute statement available is a floor. Uniform over the vocabulary is
perplexity = vocab size (128,256 for Llama-3.2). A model near that is not
producing language. The harness prints the bound and shouts if you are within
2× of it. Being far above the floor is not evidence of being good.

## Measured 2026-08-10 — Llama-3.2-1B-Instruct-4bit, 256 positions

| corpus | engine ppl | mlx_lm f32 ppl | ratio | max ΔNLL |
|---|---:|---:|---:|---:|
| prose | 65.579 | 65.574 | 1.0001 | 3.3e-2 |
| code | 37.032 | 37.020 | 1.0003 | 1.4e-2 |

40.5 positions/s, 0 GPU errors. Both numbers are baselines for comparison, not
grades.

Two things worth noting. The engine tracks mlx_lm to 1 part in 10,000 across
512 scored positions — a stronger fidelity result than anything previously in
the repo. And a 1B model at 4 bits scores materially *better* on this codebase
than on English prose, which is a property of the corpus (repetitive, highly
structured, long identifier runs), not a claim about the model.

## What is still missing

- **No task benchmark.** Perplexity is the cheapest signal, not the best one.
  Published work finds reasoning degrades faster than perplexity under
  quantization — the math-accuracy drop at 3-bit is roughly 3× the perplexity
  drop. So a 3-bit build can look fine here and still be materially worse at
  the thing you want. `mlx_lm.evaluate` is the ready-made path to closing this.
- **No 4-bit vs 3-bit comparison for the 35B**, which is the question that
  prompted all of this. `scripts/quality-ab.py` is written and validated for
  it and both checkpoints are on disk — but it needs a quiet machine. An
  earlier attempt hit 23.3 GB of 23.5 GB swap in uninterruptible wait.
  **Or run it on Colab: `docs/colab/quality-ab.ipynb`.** MLX ships a CUDA
  backend now (`pip install "mlx[cuda]"`), the 4-bit build is public on HF so
  Colab pulls it at datacenter speed, and the 3-bit build is *regenerated
  there* by `scripts/convert-q3-experts.py` — nothing is uploaded and the
  local connection is never in the path. Needs an A100/high-RAM runtime, and
  the notebook probes `gather_qmm` on CUDA in seconds before committing to a
  19.7 GB download. **Assume no number until it is run**;
  the Llama-1B result above is a sensitivity check on the harness, not a
  prediction for a 256-expert MoE where only the experts are 3-bit.
- **The prose fixture is only 1,199 tokens**, which caps `quality-eval.mjs`
  precision at ±8.3% however large `--tokens` gets. Grow it before quoting
  tight margins.
- **No golden-output regression** — fixed prompts whose continuations are
  recorded and diffed.
