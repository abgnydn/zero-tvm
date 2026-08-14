# DRAFT — comment for huggingface/transformers.js#1599
# NOT POSTED. Review, edit, post manually when you're happy with it.
#
# !!! 2026-08-04 — COMPLETE REFRAME. The bug is ALREADY FIXED upstream.
# The original model card links `new_version: onnx-community/Qwen3.5-4B-ONNX-OPT`,
# and that re-export replaces all 24 Scans with ORT's fused LinearAttention op —
# which WebGPU *does* implement. Measured same machine / prompt / runtime:
# 13.0 s -> 0.9 s prefill (~14x). It also adds num_logits_to_keep, so prefill
# returns [1,1,248320] instead of [1,252,248320], skipping a 125 MB tensor.
#
# So the old draft (reporting the Scan as an open bug) would have been reporting
# a solved problem. This version instead points the thread at the fix, and keeps
# the diagnosis only as the "why" — which the thread never got to.
#
# All the earlier analysis still holds and is still in docs/: the Scan really was
# ~92% of prefill in the old export. It's just no longer news.
#
# 2026-08-04 REWRITE: cut roughly in half and de-formalised. The previous version
# had a bold header on every paragraph, three tables, and a caveat attached to
# every claim — it read like a defence brief. Same findings, fewer scaffolds:
# one code block, numbers inline, and @youqibing addressed directly since the
# unanswered question in the thread is his.
#
# REVISION (2026-08-03): restructured so that every claim in the comment is
# reproducible by the reader, and the parts that weren't have been removed.
#
#  - Leads with the graph structure and links a Colab notebook that re-derives
#    it in ~15 s from a 1.4 MB download. No GPU, no auth, no 2.5 GB of weights.
#    A maintainer can verify the whole claim before deciding whether to care.
#  - The absolute TTFT table (4.3 / 18.6 / 79-110 / 164 s) is REMOVED. Those
#    numbers came off a machine running your own CPU sweeps at ~195%, and the
#    two engines were timed in separate sessions, which breaks this repo's own
#    same-session pairing rule. They were the one thing in the draft nobody else
#    could check — and the Scan finding makes the same point structurally,
#    without asking anyone to trust a benchmark.
#  - The "my own engine does it in 0.36/2.1/8.0/20.5 s" paragraph is removed for
#    the same reason. What remains is a one-line offer of the chunking scheme,
#    which is useful to a maintainer and claims nothing unverifiable.
#  - Kept: the sliding-window correction, the ORT #27780 explanation, and the
#    chunkability question — these are the parts that redirect the investigation.
#
# UPDATE 2026-08-04: there IS a numbers claim now, and it is properly measured —
# in-place ablation of the Scan (not extrapolation), same session, with a control
# re-run, reproduced twice: 91.3% and 93.1% of prefill. The earlier isolated-Scan
# method was discarded after it returned >100%, which is impossible; it overcounts
# by ~25%. Both the method and the negative result are in docs/webgpu-placement/.
#
# THREAD STATE (re-checked 2026-08-14): issue is OPEN, still 5 comments, NO new
# ones — nobody has posted this diagnosis. @xenova replied twice suggesting ORT
# #27780; @youqibing pushed back ("I'm not entirely sure whether these two issues
# are related") and that doubt was never resolved; @kokroo added "+1" 2026-04-05.
# CORRECTION: real last activity is 2026-04-05, not 04-12 — the 04-12 timestamp is
# `comment_deleted` by huggingface (spam removal), not a reply.
#
# The Colab link is no longer in the posted text (removed in the 08-04 rewrite), so
# the branch-pinning warning is moot. For the record: `tjs-1599-repro` is already
# merged into main and the notebook resolves on both refs.
#
# 2026-08-14 PRE-POST VERIFICATION — five defects found and fixed in the body below.
# (1) NO VERSION FLOOR: `-OPT` needs `LinearAttention` AND `CausalConvWithState`,
#     both added by onnxruntime#27996 (merged 2026-04-09). @youqibing is on
#     transformers.js 4.0.0-next.7 → ORT 1.25.0-dev.20260307, which predates them, so
#     the recommendation as written sends him to a model that FAILS TO LOAD. 4.1.0 is
#     the first release that can run it. Version floor now stated explicitly.
# (2) #27780 MISCHARACTERISED: it is not "tunes FlashAttention for MHA/GQA" — it is
#     "Optimize FlashAttention for M4 Max (20x speedup)", Apple-vendor-gated, merged
#     2026-05-14, first shipped in ORT 1.27 (NOT the 1.26.0-dev that tjs 4.2.0 pins).
#     It is also xenova's own PR.
# (3) "FIX CAME FROM THE EXPORT SIDE RATHER THAN THE KERNEL SIDE" was WRONG: it took
#     both. #27996 added the kernels; the re-export emits them. Export alone = nothing.
# (4) "~92%" DID NOT FOLLOW from 13.99 -> 1.24 (that is 91.1%; the repo's 91.3% uses a
#     mean-of-two-runs denominator the comment never showed). Now quotes both runs and
#     the 91-93% range, which is what the artifacts support.
# (5) FRAMING: xenova authored `-OPT`, added the `new_version` banner himself, AND
#     wrote #27780. The old opening presented all three to him as discoveries. The
#     opening now credits him and states what is actually unpublished: the mechanism.
#
# STILL TRUE after re-verification: issue open, no new comments, `new_version` field
# present verbatim on the old card, `-OPT` public and not itself superseded, WebGPU EP
# still has no `Scan` kernel (only `If` in controlflow/), and the python snippet was
# RUN against both real graphs — prints exactly `24 8 0` and `0 8 24`.
#
# KNOWN WEAKNESS, accepted: the headline 13.0 / 0.9 s table has no committed harness.
# Its only record is commit a5c22c8's message (13.00/12.99 and 0.85/0.93). Both runs
# are now quoted inline so the reader sees the spread rather than a bare median.

---

This looks like it's already fixed, just not in the repo this issue points at.
`onnx-community/Qwen3.5-4B-ONNX-OPT` — the re-export @xenova pushed on 2026-04-22 and linked as
`new_version` on the original card — doesn't have the problem. Posting the measurement and the
mechanism because the issue was never updated and the `-OPT` card has no description, so nobody
landing here learns that the swap is the fix or why.

Same machine, same prompt, same runtime — M2 Max, Chrome, onnxruntime-web 1.26.0-dev, 252-token
prefill; two runs, each a median of 3: 13.00 / 12.99 s and 0.85 / 0.93 s.

| | prefill |
|---|---:|
| `onnx-community/Qwen3.5-4B-ONNX` | 13.0 s |
| `onnx-community/Qwen3.5-4B-ONNX-OPT` | **0.9 s** |

Since the thread never got to a diagnosis, the reason the old export is slow: its 24
gated-DeltaNet layers are each an ONNX `Scan` — a sequential loop over the prompt, one position
per iteration. You can see it without downloading any weights, since the graph is only 1.4 MB:

```python
# onnx/decoder_model_merged_q4f16.onnx from either repo — 1.4 MB, the weights
# live in separate .onnx_data files and are not needed here.
import onnx, collections
m = onnx.load('decoder_model_merged_q4f16.onnx', load_external_data=False)
ops = collections.Counter()
def walk(g):
    for n in g.node:
        ops[n.op_type] += 1
        for a in n.attribute:
            if a.g and a.g.node: walk(a.g)
walk(m.graph)
print(ops['Scan'], ops['GroupQueryAttention'], ops['LinearAttention'])
# old: 24 8 0      OPT: 0 8 24
```

All 24 sit in the `else_branch` of a per-layer `If` on `.../gdn/is_decode`, so only prefill
pays for them — which is why decode looked much less affected than TTFT. And onnxruntime-web
has no WebGPU kernel for `Scan`, so it runs on CPU while its body dispatches to the GPU, once
per position, 24 layers deep. I measured the cost by taking the old decoder and replacing every
`Scan` with shape-preserving `Identity` nodes, then re-running in the same session: prefill went
13.99 s -> 1.24 s in one run and 13.71 s -> 0.95 s in another, so the recurrence is 91-93% of it.

The OPT export replaces each of those with a single fused `LinearAttention` node, and WebGPU
*does* have a kernel for that one, so the layer stays on the GPU. It also adds
`num_logits_to_keep`, so prefill returns `[1, 1, 248320]` instead of `[1, 252, 248320]` and
skips building a 125 MB logits tensor.

@youqibing — on your question about whether microsoft/onnxruntime#27780 was related: partly.
It's a real speedup, but it's a vendor-gated FlashAttention prefill path (merged 2026-05-14,
first shipped in ORT 1.27, so not in the build transformers.js pins today), and only 8 of these
32 layers are attention at all. The other 24 are gated-DeltaNet, which is where the prefill time
went. What fixed those was microsoft/onnxruntime#27996 (merged 2026-04-09), which added WebGPU
kernels for `LinearAttention` and `CausalConvWithState`; the `-OPT` re-export is what emits them.
So it took both a kernel change and an export change, just not that kernel.

One catch if anyone swaps the model id: `-OPT` needs an onnxruntime-web that has those kernels.
4.0.0-next.7, the version in this issue, pins `1.25.0-dev.20260307`, which predates
onnxruntime#27996 — `LinearAttention` has no kernel there at all. 4.1.0 pins
`1.26.0-dev.20260410` and is the first release that can load it; 4.2.0 is current. Not the `next`
tag, which is still 4.0.0-next.11.

The Hub already shows the `new_version` banner on the old card, but this issue doesn't, and
`-OPT` has no README, so anyone arriving here still lands on the slow repo. Might be worth a note
on the issue (or closing it) with the minimum version attached.
