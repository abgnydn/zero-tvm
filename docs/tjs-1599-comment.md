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
# THREAD STATE (checked 2026-08-04): issue is OPEN, 5 comments, last activity
# 2026-04-12. @xenova replied twice suggesting ORT #27780; @youqibing pushed back
# ("I'm not entirely sure whether these two issues are related") and that doubt was
# never resolved; @kokroo added "+1 Experiencing the same issue" on 2026-04-05.
# So this is a stalled thread with an unanswered question, not a closed one — and
# the comment below answers exactly the question that was left hanging.
#
# The Colab link below is live and was checked anonymously (14 cells render,
# no sign-in needed to read). It points at the `tjs-1599-repro` branch — if you
# ever delete or rename that branch, the link dies. Merge the notebook to main
# before posting if you want it to outlive the branch.

---

I think this is already fixed, just not in the repo the issue points at.
`onnx-community/Qwen3.5-4B-ONNX-OPT` is a re-export of the same model (it's linked as
`new_version` on the original model card) and it doesn't have the problem.

Same machine, same prompt, same runtime — M2 Max, Chrome, onnxruntime-web 1.26.0-dev, 252-token
prefill, median of 3:

| | prefill |
|---|---:|
| `onnx-community/Qwen3.5-4B-ONNX` | 13.0 s |
| `onnx-community/Qwen3.5-4B-ONNX-OPT` | **0.9 s** |

~14x, and it reproduced across two runs.

Since the thread never got to a diagnosis, the reason the old export is slow: its 24
gated-DeltaNet layers are each an ONNX `Scan` — a sequential loop over the prompt, one position
per iteration. You can see it without downloading any weights, since the graph is only 1.4 MB:

```python
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
`Scan` with a shape-preserving `Identity`: prefill went 13.99 s -> 1.24 s, so the recurrence
was ~92% of it.

The OPT export replaces each of those with a single fused `LinearAttention` node, and WebGPU
*does* have a kernel for that one, so the layer stays on the GPU. It also adds
`num_logits_to_keep`, so prefill returns `[1, 1, 248320]` instead of `[1, 252, 248320]` and
skips building a 125 MB logits tensor.

@youqibing — on your question about whether #27780 was related: I think both things were true.
That PR tunes FlashAttention for MHA/GQA, which is real, but only 8 of these 32 layers are
attention; the other 24 are gated-DeltaNet and were where the prefill time actually went. The
fix ended up coming from the export side rather than the kernel side.

Might be worth pointing the issue at `-OPT` (or closing it) so the next person doesn't land on
the old repo.
