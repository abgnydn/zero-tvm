# DRAFT — comment for huggingface/transformers.js#1599
# NOT POSTED. Review, edit, post manually when you're happy with it.
#
# What changed from the first draft: the original guess (ORT's
# linear_attention.wgsl shader) was WRONG — that op does not appear in this
# model's graph at all. The real cause is an ONNX `Scan` in the prefill branch,
# verified by inspecting the exported graph. Everything below is checkable by
# anyone who downloads a 1.4 MB file.
#
# Tone notes:
#  - Evidence first, and the evidence is reproducible in ~5 lines of Python.
#  - The sliding-window correction and the Scan finding both redirect the
#    investigation away from dead ends, without telling anyone they were wrong.
#  - Own engine appears once, near the end, caveated. Delete that paragraph if
#    you'd rather not mention it — the comment stands without it.
#  - Decode numbers omitted: the 8-token sample proved unreliable (my own
#    figure moved 2x on re-measurement). Only TTFT is claimed.

---

I ran a prompt-length sweep to check whether the TTFT cost here is a fixed overhead or
per-token, then went looking in the exported graph. I think the cause is an ONNX `Scan`
in the prefill path.

**Measurements** — Qwen3.5-4B q4f16 (`onnx-community/Qwen3.5-4B-ONNX`),
@huggingface/transformers 4.2.0, `device: 'webgpu'`, Chrome 150.0.7871.187, Apple M2 Max.
3 runs per point plus one unmeasured warm-up at each length, median:

| prompt tokens | TTFT | ms per prompt token |
|---:|---:|---:|
| 60 | 4.3 s | 72 |
| 252 | 18.6 s | 74 |
| 1024 | 79–110 s | 78–107 |
| 2044 | 164 s | 80 |

Cost per prompt token stays roughly flat across a 34× change in prompt length. A fixed
startup overhead would shrink per-token as the prompt grows, so this looks like every
prompt token is paying its own pass. The 252-token point (18.6 s) is close to
@youqibing's original 16.1 s, so I believe it's the same problem.

**What's in the graph.** The ONNX file is small (1.4 MB — the weights are in the
`.onnx_data` siblings), so the structure is easy to check:

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

Each `Scan` sits in the `else_branch` of a per-layer `If` that switches on
`/model/layers.N/gdn/is_decode`, i.e. it is specifically the **prefill** path. Its body
takes `state_in, q_in, k_in, v_in, beta_in, g_in` and is 16 nodes
(`Mul`×5, `ReduceSum`×2, `Exp`, `Sub`, `Add`, …) — the gated delta rule — with
`scan_input_axes: [2,2,2,2,2]`, so it iterates the sequence axis one position at a time.

For a 1024-token prompt that's roughly 1024 × 24 × 16 ≈ 390k sequential node executions
before the first token can be produced, which would explain the flat per-token cost far
better than anything in the attention kernels.

**One thing that may have sent the earlier investigation sideways:** Qwen3.5's hybrid is
3:1 **gated-DeltaNet (linear attention)** + full attention, not sliding-window attention.
`config.json` says so explicitly under `text_config`: `full_attention_interval: 4` and a
`layer_types` array of exactly 24 `"linear_attention"` and 8 `"full_attention"` entries.
That's probably why [microsoft/onnxruntime#27780](https://github.com/microsoft/onnxruntime/pull/27780)
didn't move this number — it targets the attention path, and 24 of the 32 layers here
aren't attention at all.

**The question:** is the `Scan` prefill path intentional for this export, or is there a
chunked form that could be emitted instead? The gated delta rule is chunkable — the
reference implementations evaluate a block of tokens in parallel (intra-chunk terms
computed together, state carried across chunks), which is what makes prefill practical
elsewhere. If the export currently only has the sequential form, this would be a prefill
problem specifically, largely independent of decode — consistent with a ~20× TTFT
regression sitting next to a much smaller decode regression.

For reference on what the same hardware can do: I maintain a separate hand-written WGSL
engine for this model, and running the same prompts through a chunked prefill (64 tokens
per pass) gives 0.36 s / 2.1 s / 8.0 s / 20.5 s at those four lengths. Different
quantization (MLC q4f16_1, not ONNX q4f16), so treat it as an order-of-magnitude
reference rather than a like-for-like benchmark — the point is just that the recurrence
doesn't have to be serial.

Happy to share the sweep harness, or to test any export variant that would be useful.
