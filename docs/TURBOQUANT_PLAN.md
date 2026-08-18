# TurboQuant KV — plan

Status: **TurboQuant KILLED, int8 KV BUILT AND SHIPPED** (2026-08-17/18). Read
the phase sections in order — this file was written as a plan and then used as
the log of the measurements that refuted it, so LATER SECTIONS SUPERSEDE
EARLIER ONES wherever they disagree, and several do:

- TurboQuant is dead for our checkpoints: its premise (a few dominant channels
  for a rotation to spread) is false here — max ÷ median per-channel magnitude
  is 3.1x, not the 20x the proxy assumed. See "Phase 0b RESULT".
- Plain int8 KV won instead, was measured free end to end (paired perplexity
  -0.09% / +0.10%, within noise), and now runs on the unfused path, on hybrids
  and through chunked prefill. See "Phase 1 BUILT".
- Everything above "Phase 0 RESULT" is the ORIGINAL PITCH, kept for the record.
  Its memory table is TurboQuant-at-3.5-bits and does not describe anything
  that exists.

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

## Why this engine, specifically — SUPERSEDED (the original pitch)

> The table below prices TurboQuant at 3.5 bits, which phase 0b killed. It is
> kept because the memory ceiling it describes is real; the route to it is not.


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

**But int8-KV was nearly unreachable when this was written, and that was the lesson** (fixed 2026-08-17/18 — see "Phase 1 BUILT"): It is
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

## Phase 0b RESULT (2026-08-17) — real vectors say NO. TurboQuant is killed.

Phase 0 ended conditional: the method wins only where a handful of channels
carry outsized magnitude, and that could not be settled by inventing vectors.
So this ran the same gate on real ones.

**Where the vectors came from, and why no GPU was needed.** The prefix pool
writes the KV cache to disk (`kv-pool.ts`), so `entry.bin` IS a dump of a real
forward pass in the engine's own paged layout. `scripts/turboquant-real.py`
reads it directly. Keys there are POST-RoPE, which is what attention scores
against and therefore what a quantizer must compress. Only the cache tensor
and meta.json's shape fields are read — never `meta.ids`, so no prompt content
is involved.

Layout confirmed before trusting a single number, because a misread would
manufacture exactly this kind of result: computed bytes/layer match
`meta.layerBytes` exactly; all 335 token slots are distinct; and adjacent
tokens sit at **0.63** cosine against **0.39** for shuffled pairs, while a
deliberately scrambled read (head and slot transposed) shows no structure at
all (−0.003 vs 0.16). Real sequence structure only survives the correct
mapping.

**The premise is FALSE for our models** (qwen36q3, 10 attention layers, 2 KV
heads, dim 256, 6,700 vectors of each kind):

| | K (post-RoPE) | V | the proxy phase 0 assumed |
|---|---|---|---|
| per-channel mean\|x\|, max ÷ median | **3.1x** | **2.5x** | 20x |
| mass in a vector's largest 4 of 256 coords | 8.5% | 8.2% | — |
| per-vector kurtosis | 7.7 | 11.8 | Gaussian = 3 |

Heavier-tailed than Gaussian, yes — but nowhere near the massive-activation
regime the method is built for. There is no small set of dominant channels for
a rotation to spread out.

**And the gate agrees** (relative inner-product error, equal bits, queries
drawn from the key distribution rather than isotropic):

| bits | K: TurboQuant | K: int-b row | V: TurboQuant | V: int-b row |
|---|---|---|---|---|
| 3 | 0.194 | 0.210 (TQ 1.08x) | 0.240 | 0.241 (TQ 1.00x) |
| 4 | 0.109 | **0.103 (int 1.06x)** | 0.129 | 0.132 (TQ 1.02x) |
| 5 | 0.057 | **0.047 (int 1.19x)** | 0.076 | **0.071 (int 1.08x)** |

TurboQuant's only edge is at 3 bits, where the absolute error (~0.2 relative)
is far too large to ship, and it is a rounding error even there. At the 4 and
5 bits that matter, plain per-row max-scaling — the scheme
`kv_quantize_int8.wgsl` already implements — is BETTER, while costing no
rotation, no sketch, no codebook and no second scalar.

**Decision: kill TurboQuant for these models.** Phase 0's synthetic outlier
proxy (4 channels x20) overstated the regime by ~7x, and the honest conclusion
is that the method is aimed at a distribution our checkpoints do not have.
Phase 1 — int4-per-row KV — stands on its own and is now the whole plan: same
4x reduction, a small generalization of the shipped int8 path, and better
inner-product error than the complicated alternative.

**What would reopen it:** these are 335 tokens of one prompt on one model at
short context. If outlier channels emerge at 32k+, or a future checkpoint
shows a max÷median ratio in double digits, the reference in
`scripts/turboquant-ref.py` is ready and the algebra is verified. Also worth
noting the gate used keys as a stand-in for queries — real Q vectors are a
different projection, and a sharper test would dump those too.

## Phase 1 GATE (2026-08-17) — int4-per-row does NOT get 4x, and the error is large

Phase 0b concluded "int4-per-row KV gets the same 4x with better error and none
of the machinery". Measuring it properly refutes the first half of that
sentence and puts the second in question. `scripts/kv-bits-gate.py` runs the
same real cached vectors through a full attention block — `softmax(QK/sqrt d) @ V`
— because that is what leaves the layer, and it is the only number that decides
whether to ship. Inner-product error ranks quantizers; attention output tells
you if it matters. Softmax forgives a uniform score shift, and the weighted sum
can concentrate error, so neither direction is predictable from phase 0b.

**Per-row int-b, the scheme the shipped int8 path implements:**

| bits | bytes/token/layer | vs f16 | attn output rel err |
|---|---|---|---|
| 4 | 520 | 3.9x | **0.207** |
| 5 | 648 | 3.2x | 0.101 |
| 8 | 1032 | 2.0x | **0.012** |

Two corrections to the plan fall straight out. The "4x" was never 4x — one f16
scale per (token, head, side) is real memory, so 4 bits buys 3.9x and 8 bits
buys 2.0x, not 2x/4x of the raw payload. And 20% error on a block's output is
not a small quality risk next to int8's 1.2%.

**Finer groups and a zero point help, but not enough, and not the way expected:**

| config | bytes/token/layer | attn output rel err |
|---|---|---|
| 4-bit, group 256 (per-row), symmetric | 520 | 0.207 |
| 4-bit, group 64, asymmetric | 576 | 0.108 |
| 4-bit, group 32, asymmetric | 640 | 0.088 |
| **5-bit, group 256, asymmetric** | **656** | **0.077** |
| 8-bit, group 256, symmetric | 1032 | 0.012 |

**The counterintuitive part, and the one worth remembering: at equal memory,
MORE BITS WITH COARSE GROUPS beats fewer bits with fine groups.** 4-bit at
group 32 asymmetric costs 640 bytes for 0.088; 5-bit per-row asymmetric costs
656 for 0.077 — 2.5% more memory, 12% less error. Shrinking the group to
rescue a lower bit-width spends the saving on scales and loses. So the
grouping knob, which is where this kind of work usually goes next, is a dead
end here; asymmetry is the cheap win (a zero point costs one extra f16 per
group and cuts error ~20% at every width).

**What this means for what to build.** The genuinely safe win is not 4 bits at
all — it is that the EXISTING int8 path (1.2% error, 2.0x) runs on no model
where context matters: it demands `fused` mode, so Phi-3 alone, and it is
excluded from chunked prefill. Making int8 work on the hybrid / MLX-affine /
chunked models IS the memory win, at a quality cost already known to be small,
with no new quantizer to justify. That is a pure engineering task with a
measurable gate, and it should come before anything below 8 bits.

Anything under 8 bits needs a perplexity gate first — `quality-ab.py` is
paired and has already shown it can see damage. Until that runs, treat 4-bit
KV as unproven, not as a plan.

**Caveats, same as phase 0b:** 335 tokens of one prompt on one model, and the
queries are cached keys standing in for real Q vectors, which are a different
projection. The attention-output metric is much closer to what matters than
inner-product error, but it is still not perplexity.

## Phase 1 QUALITY GATE (2026-08-17) — int8 KV is free on qwen36q3

Run on the model that would use it, not by analogy to Phi-3 (which is the only
spec the current int8 path serves, is architecturally unlike it, and is not
even on this disk). `scripts/kv-quality-ab.py`, 12 windows x 1024 tokens:

    f16 cache     ppl 21.8273   nll 3.08316
    8-bit cache   ppl 21.8088   nll 3.08231
    paired dNLL -0.000851 +/- 0.001648   z = -0.5   worse on 5/12 windows
    perplexity cost: -0.09%

Within noise, and the 8-bit arm was fractionally BETTER — a coin flip on 5 of
12 windows is what no effect looks like. The offline attention-output number
(1.2%) predicted this, so the two metrics agree at 8 bits.

**The caveat that limits it: 1024-token windows.** Cache quantization error
accumulates with context — every cached token is another rounded value the
softmax sums over — and real prompts here are 14k-65k. Confirm at
`--window 4096` before treating 8 bits as settled for long context.

**Also worth running: `--kv-bits 4`.** The attention-output gate put 4-bit at
0.207, seventeen times the 8-bit figure. If perplexity says otherwise, that
proxy overstates damage and should stop being trusted for go/no-go.

So the quality objection to int8 is answered, and the remaining question is
priority, not safety: at the contexts actually in use it saves ~0.57 GB, and
the work is three new WGSL kernels (int8 attention_prefill, int8 versions of
the two subgroup decode variants, a quantizing writer for the unfused path)
that this sandbox cannot GPU-test.

## Phase 1 QUALITY GATE, part 2 (2026-08-17) — and my attention-output metric was WRONG

Two more runs, and the second one refutes a measurement earlier in this file.

**8 bits holds at realistic context** (6 windows x 4096, the length actually
served rather than the 1k first tried):

    f16 cache     ppl 13.1855
    8-bit cache   ppl 13.1990
    paired dNLL +0.001023 +/- 0.000865   z = 1.2   worse on 3/6   +0.10%

**4 bits costs far less than predicted** (12 windows x 1024, same windows as
the 8-bit run at that size, so directly comparable):

    f16 cache     ppl 21.8273
    4-bit cache   ppl 21.9467
    paired dNLL +0.005456 +/- 0.002731   z = 2.0   worse on 8/12   +0.55%

**THE CORRECTION: the attention-output gate above overstates the damage by
more than an order of magnitude.** It put 4-bit at 0.207 relative error
against 8-bit's 0.012 — 17x — and concluded 4-bit was "not a trade you make
quietly". End to end the same comparison is +0.55% perplexity against ~0%.
A 20% error in what leaves an attention block turns into half a percent of
perplexity, because the residual stream and the layers above absorb it. The
proxy ranked the options correctly and was worthless for deciding whether
either was acceptable — the exact mistake phase 0b made one level down, where
inner-product error was used to judge what only attention output could answer.
**Stop using offline error metrics for go/no-go. They rank; they do not
decide.**

For scale: this repo shipped the 3-bit expert build at +8.4% perplexity. A
KV quantizer at +0.55% is an order of magnitude inside that.

The verdict banding in `kv-quality-ab.py` was fixed as a result: 4-bit landed
at z = 2.0, worse on 8 of 12 windows, and the old binary cut printed "WITHIN
NOISE" because it fell a hair under the threshold. That is a borderline result
being reported as a null one, so the script now bands it and never prints a
verdict without the effect size.

**Where this leaves the plan.** 8 bits is free and confirmed at 4k context.
4 bits looks cheap but is measured only at 1k and only at borderline
significance; the missing run is `--kv-bits 4 --window 4096 --windows 6`,
because cache error accumulates with context and that is the regime that
matters. If 4-bit holds there, the target is 4 bits (3.9x) rather than 8
(2.0x) and the kernel work is the same shape either way.

## SETTLED (2026-08-17) — 8 bits is free, 4 bits is real and grows. Build 8.

The full paired grid on qwen36q3, both arms scoring identical windows:

| bits | window | paired dNLL | z | windows worse | ppl cost |
|---|---|---|---|---|---|
| 8 | 1024 | -0.000851 +/- 0.001648 | -0.5 | 5/12 | **-0.09%** |
| 8 | 4096 | +0.001023 +/- 0.000865 | 1.2 | 3/6 | **+0.10%** |
| 4 | 1024 | +0.005456 +/- 0.002731 | 2.0 | 8/12 | +0.55% |
| 4 | 4096 | +0.007668 +/- 0.002001 | **3.8** | **6/6** | **+0.77%** |

**8 bits is free and stable.** Two window sizes, both within noise, sign
flipping between them — the signature of no effect.

**4 bits is a real cost that GROWS WITH CONTEXT.** 0.55% at 1k becomes 0.77% at
4k, z climbs 2.0 → 3.8, and every window is worse rather than 8 of 12. That is
the accumulation the earlier caveat predicted: each cached token is another
rounded value the softmax sums over, so the longer the context the more of the
answer rests on rounded numbers. It does not bite at 8 bits; it plainly does at
4. Note the regime that matters most — 32k-64k, the reason to compress a cache
at all — is still unmeasured and the trend points the wrong way.

**DECISION: build 8-bit, everywhere.** It is proven free on the model that
would use it, at the context length actually served. It is the existing
kernel generalized rather than a new packing. And the layout is already
parameterized by bit-width, so if 4 bits is ever wanted, it is a small step
from a working 8-bit path rather than a fresh one — with the 32k measurement
as its gate.

For scale, the trade being declined: +0.77% buys 3.9x instead of 2.0x. Not
absurd — this repo shipped the 3-bit expert build at +8.4% — but there is no
reason to spend quality on a first implementation when the free option
delivers the same kernels.

## Phase 1 BUILT (2026-08-17/18) — int8 KV runs where context is the cost

Shipped and GPU-verified. int8 KV had been gated to the fused QKV path, which
is Phi-3 and nothing else: a 4k window where the cache never binds. It now runs
on the unfused path, on hybrids, and through chunked prefill.

| model | result |
|---|---|
| llama32 (dense MLX, head-dim 64) | 5/6 greedy answers token-identical; 6,211-token chunked prompt token-identical |
| qwen35 (hybrid, head-dim 256) | **6/6** token-identical; 7,037-token chunked prompt token-identical |
| qwen36q3 (hybrid MoE, the one that wanted it) | 5/6 token-identical, the sixth a correct paraphrase |

Cost: ~5-8% of prefill throughput (llama32 681→625 tok/s, qwen35 443→419) for
the quantize dispatch and the unpack in the score loop. Memory on qwen36q3 is
**1.98x** — 2.50 GB → 1.26 GB at 131k, 5.00 GB → 2.52 GB at 262k.

**Two bugs found by building it, both silent:**

1. `allocKVPagesInt8` counted `spec.layers` where `allocKVPages` counts
   ATTENTION layers. Equal while int8 was fused-only; on Qwen3.6 that is 40
   buffers for 10 KV layers, quadrupling the memory the feature exists to
   halve.
2. **The pack phase covered only the first 32 words of a row.** Each of 32
   threads wrote one u32 and returned, which is correct only while
   `KV_I8_ROW_WORDS` (HEAD_DIM/4) <= 32 — true for Phi-3 (24) and Qwen3 (32),
   false at HEAD_DIM 256, where 64 words meant **dims 128..255 were never
   written** and each row's top half kept whatever the buffer held. Every
   head-dim-256 model we ship is affected: Qwen3.5, Qwen3.6, Qwen3.8.
   It produced fluent, plausible, WRONG text — "12" for 17+25 where f16 said
   "42" — not noise. That is the kind of latent bug that ships, and it was
   only ever reachable once int8 left Phi-3.

Still excluded, by a throw that names the reason: MLA (caches a latent, not
per-head K/V).

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
