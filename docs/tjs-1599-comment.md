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
# The Colab link below is live and was checked anonymously (14 cells render,
# no sign-in needed to read). It points at the `tjs-1599-repro` branch — if you
# ever delete or rename that branch, the link dies. Merge the notebook to main
# before posting if you want it to outlive the branch.

---

I went looking in the exported graph for where the prefill time goes, and I think the
cause is an ONNX `Scan` on the prefill path rather than anything in the attention kernels.

The ONNX weights live in sibling `.onnx_data` files, so the *graph* is only 1.4 MB and its
structure is cheap to check. Here's a Colab that re-derives everything below in about 15
seconds — no GPU, no auth, and it downloads only that 1.4 MB. Two optional appendices go
further against the real weights: one counts what the runtime actually executes, the other
reports which execution provider each op is assigned to and what share of prefill it costs:

**[Open in Colab](https://colab.research.google.com/github/abgnydn/zero-tvm/blob/tjs-1599-repro/docs/tjs-1599-repro.ipynb)**

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

**Does the runtime actually execute it that way?** A fair objection is that the shipped graph
says nothing about execution — an optimizer might fold the `Scan` away. It doesn't. Running
prefill under ONNX Runtime's profiler with `ORT_ENABLE_ALL` (CPU EP; the notebook's appendix
reproduces this):

| prompt tokens | `Scan` nodes | delta-rule body-op invocations |
|---:|---:|---:|
| 16 | 24 | 6,785 |
| 32 | 24 | 12,929 |
| 64 | 24 | 25,217 |
| 128 | 24 | 49,793 |

That is exactly `384 × prompt_tokens + 641`, where 384 = 24 `Scan` layers × 16 body nodes.
The residual is *identical* at every length, so this is a counting identity rather than a
fitted trend — every op that grows is inside a `Scan` body. Extrapolated, a 1024-token prompt
is ~394k body-op invocations before the first token.

Those are CPU timings and I'm not offering them as a WebGPU measurement. But the *count* is a
property of graph execution rather than of the backend.

**And the WebGPU backend can't run the `Scan` at all.** onnxruntime-web has no WebGPU kernel
for it, so it falls back to CPU. ORT reports this itself with `logSeverityLevel: 0` — this is
on `onnxruntime-web@1.26.0-dev.20260416`, the build `@huggingface/transformers@4.2.0` pins:

```
[GetCapability] webgpu kernel not found in registries for Op type: Scan
Node placements
  Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 16
  Node(s) placed on [CPUExecutionProvider].    Number of nodes: 1
```

Those 16 are the delta-rule body; the 1 is the `Scan` itself. So the loop control sits on CPU
while its body is dispatched to the GPU, and a `MemcpyFromHost` gets inserted at the boundary.
`Scan`, `If` and `Loop` are also absent from the shipped op registry. I get the identical split
on the older `1.22.0-dev` (where the provider was still named `JsExecutionProvider`), so this
isn't a regression in one build.

Because it's a property of the op registry rather than of any particular model, none of this
needs the 2.5 GB weights — a small model with the same shape reproduces it.

**How much of the time is that?** The `Scan` body is pure elementwise/reduce arithmetic with
no weights, so it can be lifted out of the export into a standalone ~2.5 KB model (again from
just the 1.4 MB graph) and timed on WebGPU on its own, at real dimensions — state
`[1,32,128,128]`, one layer. Apple M2 Max, Chrome, three runs across both runtime builds:

| | 1.22 run 1 | 1.22 run 2 | **1.26** |
|---|---:|---:|---:|
| ms per position, one layer | 2.83 | 2.54 | **2.19** |
| one layer, 252 positions | 714 ms | 640 ms | **552 ms** |
| **× 24 GDN layers** | **17.1 s** | **15.4 s** | **13.2 s** |

The number that matters isn't the absolute: it's that cost per position **does not amortize as
the prompt grows**. From 8 positions to 252 — a 30× range — each position still costs about the
same. That's the signature of a loop that batches nothing, and it's what makes the total scale
linearly with prompt length.

I want to be careful about what I'm comparing to. 13–17 s is the recurrence *alone*, measured
in isolation on my hardware; the ~16 s TTFT in this issue was measured on someone else's, at a
prompt length I don't know. So I'm not claiming a precise percentage. What I'd say is that the
recurrence lands in the same order as the entire reported TTFT, which makes it look like the
dominant term rather than one contributor among several — and the spread across my own runs is
the honest error bar on that.

So: a per-position recurrence, 24 layers deep, whose loop is driven from the CPU side while
its body dispatches to the GPU — a mechanism that wouldn't show up by looking at the attention
kernels at all.

**One thing that may have sent the earlier investigation sideways:** Qwen3.5's hybrid is
3:1 **gated-DeltaNet (linear attention)** + full attention, not sliding-window attention.
`config.json` says so explicitly under `text_config`: `full_attention_interval: 4` and a
`layer_types` array of exactly 24 `"linear_attention"` and 8 `"full_attention"` entries
(the notebook cross-checks that against the op counts). That would explain why
[microsoft/onnxruntime#27780](https://github.com/microsoft/onnxruntime/pull/27780) didn't move
this number — it tunes FlashAttention (`flash_attention.wgsl.template`) for
MultiHeadAttention/GroupQueryAttention, so it should speed up the 8 attention layers here, but
the other 24 aren't attention at all. For the same reason, changes to ORT's `LinearAttention`
kernel can't affect this model either: that op never appears in the graph.

None of this is specific to the quantized build — `decoder_model_merged.onnx`,
`_fp16` and `_q4f16` all give the same 24 / 8 / 0, so it comes from the export rather than
from quantization.

**The question:** is the sequential `Scan` intentional for this export, or is there a
chunked form that could be emitted instead? The gated delta rule is chunkable — the
reference implementations evaluate a block of positions in parallel (intra-chunk terms
computed together, state carried across chunks), which is what makes prefill practical
elsewhere. If the export currently only emits the sequential form, this would be a
**prefill** problem originating in the export rather than in the runtime's shaders — which
would fit a large TTFT regression sitting next to a much smaller decode one.

I've implemented the chunked form of this recurrence in WGSL for this model, so I'm happy
to share the chunking scheme, or to test any export variant that would be useful.
