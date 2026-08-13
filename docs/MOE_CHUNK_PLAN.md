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

## Phase A — chunk everything except the expert matmuls

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

## Phase B — group tokens by expert

The remaining ~38%. Sort the (token, slot) pairs by expert, build per-expert
offsets, and run one grouped GEMM where expert `e` covers its contiguous block
of token rows; scatter back through the permutation.

At a 256-token chunk with top-8 that is 2048 pairs over up to 256 experts —
about 8 rows per expert on average, so the GEMM stays narrow and the win comes
from dispatch count, not tile efficiency. Worth doing after Phase A is
measured, and worth re-measuring before assuming the shape.

## Gates

Both phases hold the bar every chunk kernel has held: token identity vs
per-token prefill via `chunk-prefill-test.mjs`, which already refuses a run in
which no chunk executed. MoE specs need subgroups, so there is no scalar arm to
compare against — the per-token path is the reference.

`paging-test.mjs` and the kernel suites must stay green; the router change
touches `moe_router_logits`/`moe_router_topk`, which `real-weights.mjs` covers
against a JS reference at both router widths.
