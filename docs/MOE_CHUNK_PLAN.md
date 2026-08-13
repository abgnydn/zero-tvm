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
