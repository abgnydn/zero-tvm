# Chunked prefill for MoE — plan

MoE is the last spec family that prefills one token at a time. `qwen36` (35B)
and `qwen30b` are the two largest models this engine ships, so it is also the
family where slow prefill is most visible.

The stated blocker, in `engine-core.ts`:

> `ids[]` is indexed by SLOT with no token dimension, so batching a chunk would
> apply one token's expert choice to every token in it.

That is true, and it is why the expert matmuls cannot simply take a runtime M.
It is **not** a reason the rest of the layer cannot chunk.

## The measurement that sets the plan

MoE prefill is per-token today, so a decode profile *is* a prefill profile.
`scripts/decode-profile-native.mjs qwen30b`, LM Studio idle, 48 layers:

| kernel | share | expert-bound? |
|---|---:|---|
| moeUp | 13.6% | yes |
| moeGate | 13.0% | yes |
| qkvMatmul | 12.5% | no |
| moeDown | 11.4% | yes |
| oproj | 10.3% | no |
| moeRouterTopk | ~4.3% | no |
| lmHead | ~4.3% | no |
| qkNormRopeAppend, attention, addNorm1, addNorm2 | ~3.8% each | no |
| moeRouterLogits, moeCombine | ~3.2% each | no |
| attnCombine | ~2.7% | no |
| moeSilu | ~1.1% | no |

**The expert matmuls are only ~38% of the work.** The other ~62% is the same
dense chain every other spec already chunks — projections, attention, norms,
the router's own matmul, the combine.

The hard part of MoE chunking is therefore the *minority* of the win. That
inverts the natural plan.

## Phase A — LANDED 2026-08-13, and it batched the experts too

Written as "loop over m for the expert pass", implemented as a grid dimension:
`workgroup_id.y` is the token in `int4_matmul`'s moe variant, in both router
kernels and in `moe_combine`, so the loop is the GPU's, not the encoder's. The
whole layer is seven dispatches per CHUNK where it was seven per TOKEN.

The expert matmuls read the same weight bytes per (token, slot) pair either
way — batching them here removes dispatches, not memory traffic. That is Phase
B's job, and the profile above says it is the smaller half.

Gates, both MoE families, token-identical against per-token prefill:

| spec | covers | result |
|---|---|---|
| qwen30b | pure attention, no shared expert, 4-bit router | PASS, 0 GPU errors |
| qwen36q3 | hybrid GDN, shared expert, 8-bit router, 3-bit experts | PASS, 0 GPU errors |

## Phase A as planned — chunk everything except the expert matmuls

Batch the whole layer for M tokens exactly as the dense path does, and inside
the MoE block run the expert pass M times sequentially:

```
for each chunk of M tokens:
  batched: norm -> qkv -> rope/append -> attention -> o_proj -> norm
  batched: router logits  ([M,d] x [E+1,d]^T -> [M,E+1])
  batched: router topk    (grid over tokens -> ids[M][slots], scores[M][slots])
  for m in 0..M:                       <-- the only per-token part left
      gate/up/silu/down for token m's slots
  batched: combine, residual
```

Correct by construction: experts never mix tokens, so a loop over m computes
exactly what the per-token path computes. No permutation, no grouped GEMM, no
new kernel — the batched variants of every dense kernel already exist and are
already token-identity-gated by `chunk-prefill-test.mjs`.

Requires only that `moeIds`/`moeScores` grow a token dimension (`[M][slots]`)
and that the router's two kernels take a token grid. Both are shape changes,
not algorithm changes.

Expected: ~62% of the work moves from M dispatches to 1.

## Phase B — group tokens by expert: MEASURED, and NOT worth building

The plan was to sort the (token, slot) pairs by expert so each expert's weights
are read once for all the rows that chose it. `scripts/moe-group-probe.mjs`
measures that ceiling on synthetic buffers at real shapes before anyone builds
it: arm A is the SHIPPED moe kernel, arm B is the SHIPPED E5 chunk GEMM
dispatched once per expert with that expert's rows.

| chunk x slots / experts | rows per expert | A (shipped) | B (grouped) | |
|---|---:|---:|---:|---|
| 256 x 8 / 128 — **qwen30b as shipped** | 16 | **5.30 ms** | 13.59 ms | grouping **0.39x** |
| 1024 x 8 / 128 | 64 | 20.97 ms | **14.88 ms** | grouping 1.41x |
| 256 x 8 / 32 | 64 | 5.40 ms | **3.68 ms** | grouping 1.47x |

**At the cap we actually ship, grouping is 2.6x SLOWER** — while reading 16x
FEWER weight bytes. That is the whole finding: the traffic argument this plan
rested on is real and it does not matter, because the caches already serve the
re-reads. Arm A moves 1728 MiB of logical weight reads in 5.30 ms; sustaining
that from DRAM would need 326 GB/s on a 400 GB/s machine, so most of it never
reaches DRAM.

What decides it is **rows per expert**, which is `chunk x slots / experts`:

- qwen30b at cap 256: 256x8/128 = **16** — grouping loses
- qwen36 at cap 256: 256x9/257 ≈ **9** — worse
- break-even is around 64, E5's tile height

So Phase B needs `CHUNK_CAP` at 1024 to pay at all, and there it returns 1.41x
on the expert matmuls — which are 38% of a MoE step, so ~1.15x overall, bought
with a permutation kernel, per-expert offsets, a grouped GEMM, and 4x the
chunk activation buffers.

**Not building it.** Revisit only if the cap rises or a model ships with few
enough experts to put rows-per-expert near 64.

One caveat kept honest: arm B uses E5, whose tile is 64 rows, so at 16 rows per
expert it wastes 75% of every tile. A GEMM with a 16-row tile would close some
of that — and building one is part of Phase B's cost, not an argument against
this measurement. It would have to be ~2.6x better than E5-at-M=16 merely to
draw.

The probe binds the whole expert stack as one storage buffer, so E=256 exceeds
the default 128 MiB binding limit and it fails loudly rather than timing a
skipped dispatch — which is how the first run of it reported arm A at 0.013 ms.

## Gates

Both phases hold the bar every chunk kernel has held: token identity vs
per-token prefill via `chunk-prefill-test.mjs`, which already refuses a run in
which no chunk executed. MoE specs need subgroups, so there is no scalar arm to
compare against — the per-token path is the reference.

`paging-test.mjs` and the kernel suites must stay green; the router change
touches `moe_router_logits`/`moe_router_topk`, which `real-weights.mjs` covers
against a JS reference at both router widths.


## Expert residency — measured 2026-08-14, before building anything

93% of the 35B is expert weights and only top-8 of 257 run per token, so the
idle 249 are the obvious thing to stop holding. `scripts/moe-trace.mjs` records
what the router really chose (via the engine's own `traceMoe` flag, copying
`moeIds` straight after each layer's top-k), `scripts/moe-stream-probe.mjs`
measures the two costs, and `scripts/moe-replay.mjs` scores policies offline.

### Skew is weak; locality is what works

qwen30b, 1827 decode steps: **all 128 experts get used**, and the top 10% take
13.7% of requests against 10% for uniform routing. This does not work because a
few experts dominate. It works because of temporal locality inside a
generation.

| pool | LRU | LFU | LFRU | Belady |
|---|---:|---:|---:|---:|
| 13/128 | 50.0% | 16.2% | 49.0% | 68.9% |
| 32/128 | 77.8% | 36.1% | 76.5% | 87.6% |
| 64/128 | **94.5%** | 57.3% | 92.9% | 95.3% |

**LFU loses badly**, which contradicts the paper claiming activation is skewed
rather than time-local — on these models it is the other way round. LRU, the
simple thing llama.cpp shipped, is right. And LRU sits within 0.8 points of
Belady at the half pool, so better prediction has almost nothing left to buy.

Longer generations barely move it: 243 steps and 1827 steps agree within a
point everywhere except the half pool, which improves from 90.2% to 94.5%.

### The binding cost is the readback, not bandwidth

Both measured on this machine, not carried over:

- `writeBuffer` at the real slab size: **8.19 GB/s** at 2.53 MiB, 9.10 at 1.31.
  An earlier projection used 2.4 GB/s from a range measured at a different
  size and was 3.5x too pessimistic.
- router-id readback: **0.199 ms** per round trip, x48 layers = **9.6 ms of
  every token**, against 11.3 ms of compute.

WebGPU has no GPU-waits-on-host primitive — verified against the API surface,
which contains no fence, semaphore or event — so each layer's ids must reach JS
before its experts can be fetched. That is exactly what llama.cpp's Metal build
avoids with `MTLSharedEvent`.

| pool | resident | miss | serial | overlapped | pipelined |
|---|---:|---:|---:|---:|---:|
| 13/128 | 2.5 GB | 50% | 12.0 t/s | 13.9 t/s | 16.1 t/s |
| 32/128 | 4.8 GB | 22% | 20.6 t/s | 26.9 t/s | 36.2 t/s |
| 64/128 | 8.6 GB | 6% | 36.0 t/s | 47.9 t/s | 88.4 t/s |

Serial is the naive build. Overlapped hides the fetch behind compute. Pipelined
additionally hides the READBACK, which is possible only because per-layer
compute (0.236 ms) just exceeds a round trip (0.199 ms) — it needs next-layer
speculation to be right, and a wrong guess is a stall. Treat it as a ceiling.

### What was tried and does not work

Prefetching from the PREVIOUS TOKEN's ids would need one readback per token
instead of 48. Token-to-token overlap is **43.1%** (per layer 11-62%), and that
is not the useful 43%: an expert the previous token used is recent, so LRU
already holds it. The misses are precisely the experts the previous token did
NOT choose, which is what this cannot predict.

### Verdict

Worth building. 8.6 GB at 36-48 t/s against 16 GB at 88 t/s, and 4.6 GB for the
35B. The order is: slot pool + OPFS streaming first (LRU, nothing cleverer),
then next-layer speculation, which is worth more here than any policy change
because it attacks the readback rather than the hit rate.


## Repriced 2026-08-14 after measuring a miss end to end

The projection above priced a miss as a `writeBuffer`. That is the GPU half.
`scripts/moe-slab-probe.mjs` measures both halves against a real checkpoint:

| stage | ms | GB/s |
|---|---:|---:|
| disk read (2.53 MiB, random offset) | 0.174 | 15.26 |
| writeBuffer upload | 0.318 | 8.35 |
| **chained — what a miss is** | **1.007** | 2.63 |

Chained costs MORE than read plus upload because they do not pipeline:
`writeBuffer` snapshots its source, so the next read cannot begin until the copy
finishes. A miss is 1.007 ms, 3.4x the 0.292 ms the earlier projection used.

Two methodology notes, both of which cost a wrong number first:

- The first version reused one offset list across arms, so the read-only pass
  warmed the page cache for the chained pass and "chained" came out FASTER than
  "read alone" — impossible, and exactly how a benchmark reports a speed the
  machine cannot deliver. Arms use disjoint offsets now.
- The read arm ran at **15 GB/s, which is RAM, not disk**. A 16 GiB checkpoint
  on a 32 GiB machine is largely page-cached. On the machine this feature
  exists for — one too small to hold the checkpoint — reads are cold and
  slower. Treat 1.007 ms as a floor.

### The pool has to be big

| pool | resident | miss | serial | pipelined |
|---|---:|---:|---:|---:|
| 13/128 | 2.5 GB | 50% | 8.2 t/s | 9.7 t/s |
| 32/128 | 4.8 GB | 22% | 15.3 t/s | 21.4 t/s |
| 64/128 | 8.6 GB | 6% | 31.9 t/s | **78.1 t/s** |

The small pools are gone. At 50% miss a token pays 102 ms of transfer and no
amount of pipelining rescues it. The half pool survives because its miss rate is
6%, which is few enough that the transfer hides under compute.

**So the honest offer is 8.6 GB at 31-78 t/s, against 16 GB at 88 t/s** — half
the memory, and the spread depends entirely on whether the engine submits the
next layer before awaiting the previous readback. The 2.5 GB configuration that
looked attractive is not usable.
