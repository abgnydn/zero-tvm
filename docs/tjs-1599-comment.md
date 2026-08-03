# DRAFT — comment for huggingface/transformers.js#1599
# NOT POSTED. Review, edit, post manually when you're happy with it.
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
# If you want a numbers claim in here later, re-measure interleaved on an idle
# machine and add it as a separate follow-up comment.
#
# Before posting, replace COLAB_LINK_HERE with the real notebook URL.

---

I went looking in the exported graph for where the prefill time goes, and I think the
cause is an ONNX `Scan` on the prefill path rather than anything in the attention kernels.

The ONNX weights live in sibling `.onnx_data` files, so the *graph* is only 1.4 MB and its
structure is cheap to check. Here's a Colab that re-derives everything below in about 15
seconds — no GPU, no auth, and it downloads only that 1.4 MB:

**COLAB_LINK_HERE**

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
# -> 24 8 0
```

- **24 `Scan`** — one per linear-attention layer
- 8 `GroupQueryAttention` — the full-attention layers
- **0 `LinearAttention`** — ORT's fused contrib op isn't used by this graph

All 24 `Scan`s sit in the **`else_branch`** of a per-layer `If` switching on
`/model/layers.N/gdn/is_decode` — that is, specifically the **prefill** path. Each body
takes `state_in, q_in, k_in, v_in, beta_in, g_in` and is 16 nodes
(`Unsqueeze`×6, `Mul`×5, `ReduceSum`×2, `Exp`, `Sub`, `Add`) — the gated delta rule — with
`scan_input_axes: [2,2,2,2,2]`, so it walks the sequence one position at a time.

So a 1024-token prompt runs on the order of 1024 × 24 × 16 ≈ 390k sequential node
executions before the first token can be produced, and that count grows linearly with
prompt length with nothing batched across positions.

**One thing that may have sent the earlier investigation sideways:** Qwen3.5's hybrid is
3:1 **gated-DeltaNet (linear attention)** + full attention, not sliding-window attention.
`config.json` says so explicitly under `text_config`: `full_attention_interval: 4` and a
`layer_types` array of exactly 24 `"linear_attention"` and 8 `"full_attention"` entries
(the notebook cross-checks that against the op counts). That would explain why
[microsoft/onnxruntime#27780](https://github.com/microsoft/onnxruntime/pull/27780) didn't
move this number — it targets the attention path, and 24 of the 32 layers here aren't
attention at all. For the same reason, changes to ORT's `LinearAttention` kernel can't
affect this model either: that op never appears in the graph.

**The question:** is the sequential `Scan` intentional for this export, or is there a
chunked form that could be emitted instead? The gated delta rule is chunkable — the
reference implementations evaluate a block of positions in parallel (intra-chunk terms
computed together, state carried across chunks), which is what makes prefill practical
elsewhere. If the export currently only emits the sequential form, this would be a
**prefill** problem originating in the export rather than in the runtime's shaders — which
would fit a large TTFT regression sitting next to a much smaller decode one.

I've implemented the chunked form of this recurrence in WGSL for this model, so I'm happy
to share the chunking scheme, or to test any export variant that would be useful.
