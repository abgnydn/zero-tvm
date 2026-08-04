# DRAFT — comment for huggingface/transformers.js#1599
# NOT POSTED. Review, edit, post manually when you're happy with it.
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

I had a dig through the exported graph and I think the TTFT here is a graph problem rather
than a kernel one.

The 24 gated-DeltaNet layers are exported as ONNX `Scan` — a sequential loop over the
sequence axis. The weights sit in the `.onnx_data` siblings so the graph itself is only
1.4 MB, and you can check it in a few lines:

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
# 24 8 0
```

All 24 are in the `else_branch` of a per-layer `If` on `/model/layers.N/gdn/is_decode` — so
the prefill side — and each body is the gated delta rule (16 nodes) with
`scan_input_axes: [2,2,2,2,2]`, one position per iteration. Counts are the same in
`decoder_model_merged.onnx` and `_fp16`, so it's coming from the export rather than from
quantization.

Two things make that expensive. First, onnxruntime-web has no WebGPU kernel for `Scan`, so it
falls back to CPU while the body ops go to the GPU. With `logSeverityLevel: 0` on
`1.26.0-dev` (what 4.2.0 pins):

```
[GetCapability] webgpu kernel not found in registries for Op type: Scan
Node placements
  Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 16
  Node(s) placed on [CPUExecutionProvider].    Number of nodes: 1
```

which is a CPU-driven loop dispatching GPU kernels once per position, 24 layers deep. `If`
and `Loop` aren't in the registry either.

Second, it's most of the prefill. I took a copy of the decoder with every `Scan` swapped for
shape-preserving `Identity` — everything else in the GDN layers left alone — and timed both in
one session on a 252-token prompt (M2 Max, Chrome): full 13.99 s vs ablated 1.24 s, and
13.71 s vs 0.95 s on a second run. So ~92%. Everything that isn't the recurrence, including
all the quantized matmuls and the 8 attention layers, comes to about a second.

@youqibing — on whether #27780 is related, I think you were right to wonder. It tunes
FlashAttention for MHA/GQA, so it should help the 8 attention layers here, but the other 24
aren't attention at all: Qwen3.5's hybrid is 3:1 gated-DeltaNet + full attention
(`text_config.layer_types` is exactly 24 `linear_attention` / 8 `full_attention`), not
sliding-window. Same reason ORT's `LinearAttention` kernel wouldn't move it — that op doesn't
appear in this graph.

Main question then is about the export: is the sequential `Scan` intentional here, or could a
chunked form be emitted instead? The gated delta rule chunks fine — you evaluate a block of
positions in parallel and carry the state across blocks, which is what makes prefill tractable
elsewhere. If the export only has the serial form that would fit a large TTFT regression
sitting next to a much smaller decode one.

Colab that re-derives all of the above, 1.4 MB download and no GPU needed:
https://colab.research.google.com/github/abgnydn/zero-tvm/blob/tjs-1599-repro/docs/tjs-1599-repro.ipynb

I've implemented the chunked form of this recurrence in WGSL for the same model, so happy to
share the chunking scheme or to test any export variant that would help.
