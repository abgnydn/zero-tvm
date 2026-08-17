# TurboQuant KV — plan

Status: **planned, nothing built.** No claim in this file has been measured
here; every number below is either arithmetic on our own specs (marked as
such) or the paper's, attributed. It exists so the build can be judged before
it is started, and killed early if the gates fail.

## What the method is

TurboQuant (Google Research, ICLR 2026, arXiv:2504.19874). From the abstract,
which is precise about the shape even though the constants need the PDF:

1. **Randomly rotate** the input vector. The rotation is data-oblivious —
   fixed at build time, not learned — and it "induc[es] a concentrated Beta
   distribution on coordinates", which is what makes a *scalar* quantizer
   near-optimal on each coordinate independently. Outliers, which are the
   whole reason naive low-bit KV fails, get spread across coordinates.
2. **Quantize each coordinate** with an MSE-optimal scalar quantizer.
3. **Correct the bias.** An MSE-optimal quantizer is biased *for inner
   products*, which is the only thing attention actually computes. So a
   **1-bit QJL sketch of the residual** is stored alongside, giving an
   unbiased inner-product estimator.

Reported: "absolute quality neutrality with 3.5 bits per channel" and
marginal degradation at 2.5 — against our 16.

The property that makes it implementable for us: a rotation preserves inner
products, so `<Rq, Rk> = <q, k>`. Rotate the query the same way at attention
time and the score is unchanged; nothing downstream of attention learns that
the cache is quantized.

## Why this engine, specifically

KV is our ceiling, and the context picker (2026-08-16) put a price tag on it
in the UI, which makes the cost impossible to ignore. Arithmetic on our own
specs, at the paper's two operating points:

| model | KV/token today | at 3.5 b | at 2.5 b | KV at 262k: today → 3.5 b |
|---|---:|---:|---:|---|
| Qwen3.6-35B-A3B (both builds) | 20 KB | 4.4 KB | 3.1 KB | 5.00 GB → **1.09 GB** |
| Qwen3.8-27B | 64 KB | 14.0 KB | 10.0 KB | 16.00 GB → 3.50 GB |
| Qwen3.5-9B | 32 KB | 7.0 KB | 5.0 KB | 8.00 GB → 1.75 GB |
| Qwen3-30B-A3B | 96 KB | 21.0 KB | 15.0 KB | 24.00 GB → 5.25 GB |

The headline that decides whether this is worth building: **the 4-bit
Qwen3.6-35B — the build with no quantization quality cost at all — would hold
its full 262k window in ~1.1 GB of KV**, putting the whole thing near 21 GB
on a 32 GB machine. Today that same window costs 5 GB of KV and the 3-bit
expert build is the only one that fits, at a measured +10% perplexity. This
is the arc that removes a compromise rather than adding a feature.

## What exists to reuse — and the mistake not to repeat

The int8-KV path is the structural precedent and it is already the right
shape: quantize-on-append (`kv_quantize_int8.wgsl`) writing packed codes plus
a per-(page, slot, head, side) f16 scale, and an attention variant
(`attention_int8.wgsl`) that reads codes + scale instead of f16 with
"identical online-softmax math". `allocKVPagesInt8` sizes the buffers. That
is the same three-piece skeleton this needs.

**But int8-KV is nearly unreachable today, and that is the lesson.** It is
gated to the fused-QKV path, which means Phi-3 alone — none of the MLX
checkpoints, neither MoE, no hybrid. And it was excluded from chunked prefill
on 2026-08-17 because the chunk path binds the f16 append/attention kernels
and would corrupt half-size int8 pages. So the one quantized-KV feature we
have does not run on a single model where context is the constraint.

**TurboQuant must land on the path the big models actually use** — unfused
QKV, MLX-affine, chunked prefill, GQA, and the hybrid layouts where only some
layers hold KV — or it will be another correct feature nobody can switch on.
That is a gate below, not an aspiration.

## Shape constraints, checked

A fast Walsh-Hadamard transform needs a power-of-two length. Head dims, from
the registry:

| head dim | models | WHT |
|---|---|---|
| 256 | Qwen3.5-4B/9B, Qwen3.6 (both), Qwen3.8-27B | yes |
| 128 | Qwen3-4B (both builds), Qwen3-30B-A3B | yes |
| 64 | Llama-3.2-1B | yes |
| **96** | **Phi-3-mini** | **no** |

Nine of ten. Phi-3 is the exception and also the model where this matters
least (4k window, 384 KB/token but only ~1.5 GB of KV total). It is excluded
in phase 1 rather than padded — a 96→128 pad would cost 33% of the saving and
add a shape no other model exercises.

Also load-bearing, from `docs/PAGING_PLAN.md`: **K is RoPE'd before the cache
write on every path.** So the rotation is applied to the *post-RoPE* K, and
the query must be rotated post-RoPE too. Values carry no RoPE.

## Design sketch

Per (page, slot, kv_head, side), stored:

- **codes** — one 3-bit (K) or 2-bit (V) code per coordinate, packed into u32
  words. HEAD_DIM 256 at 3 bits = 96 bytes; the packing must not straddle a
  u32 in a way that costs a second load per coordinate group.
- **scale** — f16 per (slot, head, side), exactly as the int8 path does.
- **QJL sign bits** — 1 bit per coordinate of the residual sketch, 32 bytes
  at HEAD_DIM 256. (Whether the paper's "3.5 bits per channel" *includes*
  this bit is the first thing to settle from the PDF; if it does not, our
  real rate is 4.5 and the table above is optimistic by ~25%. **Do not
  publish the table's numbers until this is settled.**)

Kernels, mirroring the int8 trio:

1. `kv_rotate_quantize.wgsl` — WHT over HEAD_DIM in workgroup memory, then
   per-coordinate quantize + QJL sign extraction. One workgroup per
   (kv_head, side), like `kv_quantize_int8`.
2. `attention_tq.wgsl` — rotate Q (same WHT, once per head per token), score
   against codes, add the QJL correction term, unchanged online softmax.
3. Prefill variants: the chunked path needs the same pair, or TurboQuant
   inherits int8's fate.

The rotation is fixed and shared: a Hadamard matrix with a per-(layer, head)
random sign vector, generated from a seed in the spec so it is reproducible
and needs no storage.

## Phase 0 RESULT (2026-08-17) — measured, and it changes the plan

`scripts/turboquant-ref.py`: Algorithms 1 and 2 transcribed from the PDF, then
the estimator implemented a SECOND time by the route a kernel would take.

**Settled by reading the paper:**

- **The stated bits/channel INCLUDE the QJL bit.** Algorithm 2 line 2
  instantiates the MSE quantizer at `b-1`. So "3.5 bits" is 2.5 bits of codes
  plus the sketch, not 3.5 + 1. The table above stands.
- **Their fractional bit-widths are an outlier split, not fractional coding**:
  "in our 2.5-bit setup, 32 outlier channels are quantized at 3 bits, while
  the remaining 96 channels use 2 bits … (32x3 + 96x2)/128 = 2.5". Two
  independent TurboQuant instances over two channel groups.
- **Stored per vector**: `(idx, qjl, ||r||)` — plus the vector norm, since
  Algorithm 1 works on the unit sphere.
- **The kernel identity holds**: `<q, S^T qjl> = <S q, qjl>`, so neither the
  rotation nor the sketch ever touches a cached vector at read time. The query
  is projected once per head; each cached token costs one sign-dot. The two
  implementations agree to 1e-14, so the algebra a shader would rely on is
  right.

**Measured (inner-product error, relative, vs max-scaled int-b at EQUAL bits):**

| data | 3 bits | 4 bits | 5 bits |
|---|---|---|---|
| isotropic Gaussian, d=256 | int 1.5x better | int 1.9x better | int 2.1x better |
| outlier-heavy (4 ch x20), d=128 | int 1.1x better | **TQ 1.12x** | **TQ 1.33x** |
| outlier-heavy (4 ch x20), d=256 | int 1.02x better | **TQ 1.68x** | **TQ 1.93x** |

Three things follow, and two of them were not visible from the paper:

1. **The method's value is entirely conditional on outliers.** On isotropic
   data a rotation has nothing to fix and the QJL bit is simply spent, so
   plain max-scaling wins by ~2x. This is not a criticism of the paper — it
   is the regime it was designed for — but it means our gate can only be
   settled on REAL key/value vectors, not synthetic ones. The outlier proxy
   here (4 channels x20) is a guess at the shape.
2. **It needs 4+ bits to win at all.** At 3 bits plain scaling is still ahead
   even with outliers. So the paper's 2.5-bit operating point is not
   reachable by this construction on our shapes, and the honest target is
   4 bits — a **4x** KV reduction, not 6.4x.
3. **The win grows with head dim** (1.68x at 256 vs 1.12x at 128), which is
   the right direction: our long-context models — Qwen3.5/3.6/3.8 — are all
   head-dim 256.

**And the uncomfortable consequence, which the plan has to face:** at 4 bits,
plain int4-per-row KV gets the SAME 4x memory reduction with none of the
machinery — no rotation, no sketch, no codebook, no second scalar. TurboQuant
buys ~1.7x lower inner-product error at that size, not extra compression.

So the question the next phase must answer is no longer "does TurboQuant
work" but **"is int4-per-row KV good enough end to end, and if not, does
TurboQuant's error advantage rescue it?"** Building the complicated thing
first would answer the wrong question.

**Revised phase order:**

- **Phase 0b — real vectors.** Dump K/V from a real forward pass (qwen36q3,
  a long prompt) and re-run the gate on them. This decides whether the
  outlier premise holds for OUR models, and settles the codebook shape.
- **Phase 1 — int4-per-row KV first.** It is a small change to the shipped
  int8 path (kv_quantize_int8 generalized, attention_int8 generalized), it
  delivers the same 4x, and it is the baseline every later claim needs. If
  paired perplexity at 32k is neutral, TurboQuant may not be needed at all.
- **Phase 2+ — TurboQuant only if phase 1's quality is short**, in which case
  the reference here is ready and the kernel work is the rotation, the
  codebook lookup, and the sign-dot.

Nothing about the memory ceiling changes: 4 bits still puts the 4-bit
Qwen3.6-35B's full 262k window near 1.25 GB of KV. The route there just got
cheaper and the risk lower.

## Phases and gates (original — superseded above where they conflict)

**Phase 0 — settle the algorithm (no code).** Read the PDF and the QJL paper;
write `scripts/turboquant-ref.py` implementing rotate → quantize → QJL
inner-product estimation in NumPy, and a second, independent implementation
of the estimator. *Two implementations in Python before any shader* is the
rule that paid for MLA: it separates "we misunderstood the algorithm" from
"the shader is wrong", which a WGSL test cannot distinguish.
Gate: the two agree, and the estimator is unbiased on random vectors at our
actual head dims (64/128/256).

**Phase 1 — kernels, synthetic.** The three kernels above against the Python
reference on random data, in `tests/kernels/`. Gate: inner-product error
within the paper's predicted band at 3.5 bits, and *no worse than int8's
measured error* at equal bits — if a rotation plus a sketch cannot beat plain
int8 per bit, the method is not worth its complexity here.

**Phase 2 — real weights, one model.** qwen36q3 (256 head dim, GQA 16/2, the
model this is for). Gate: `validate-model` cosine ≥ 0.999 and greedy
token-exact against mlx_lm on a short prompt, exactly as every model port has
had to pass.

**Phase 3 — long context, the actual claim.** Gate: `quality-ab.py` paired
perplexity, f16-KV vs TurboQuant-KV, at 32k and 128k. The claim under test is
"quality neutral at 3.5 bits". Paired, because unpaired comparison on this
harness has already produced a wrong answer once (z = 0.8 vs 14.7 on OLMoE).

**Phase 4 — speed.** Gate: decode within 5% of f16 KV on the same machine,
AC power, same-session A/B. Attention gets *more* arithmetic (rotate Q, QJL
correction) against *less* memory traffic; which wins is an empirical
question and the answer may differ between decode (memory-bound) and prefill.

**Phase 5 — reach.** Chunked prefill, GQA, hybrid layouts, the pooled MoE
path. Gate: `chunk-prefill-test` token-identical with TurboQuant on, on every
chunking spec. Without this it is int8 again.

## Predictions, so the measurement can refute them

1. Decode throughput **improves** on the big-context models: attention is
   memory-bound at long context and this cuts KV traffic ~4.5x. If decode
   gets *slower* at 32k+, the QJL correction is costing more than the traffic
   saved and the design is wrong.
2. The 4-bit 35B at 262k fits under 22 GB resident.
3. Quality at 3.5 bits is within noise of f16 KV on paired ppl — and if it is
   not, the honest outcome is publishing the cost the way the 3-bit expert
   build's +10.4% was published, not quietly shipping it.

## Kill criteria

- Phase 1 fails to beat int8 per bit → stop; the complexity buys nothing.
- Phase 3 shows a perplexity cost comparable to the 3-bit expert build → the
  memory would be better spent going *back* to 4-bit experts with f16 KV.
- Phase 4 shows a decode regression that does not close → ship it opt-in as a
  memory feature with the speed cost published, or not at all.

## Open questions

- Does "3.5 bits per channel" include the QJL bit? (Decides the table above.)
- Per-coordinate quantizer: does the Beta concentration let one shared
  codebook serve all coordinates, or is it per-coordinate?
- Is one rotation per (layer, head) enough, or does the sign vector need to
  vary per token? (Storage-free either way, but it changes the Q-side cost.)
- Prefill: rotating K for a whole chunk is a batched WHT — does it fold into
  the existing chunk kernels or need its own dispatch?
