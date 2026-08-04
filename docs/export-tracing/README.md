# Can the chunked delta rule be traced to ONNX?

Supporting the transformers.js#1599 analysis. The obvious suggestion — "export
`torch_chunk_gated_delta_rule` instead of the per-position one" — does not work,
and this is the check that shows why.

`modeling_qwen3_5.py` ships both pure-PyTorch fallbacks and dispatches by length:

```python
if use_precomputed_states and seq_len == 1:
    self.recurrent_gated_delta_rule(...)   # per position
else:
    self.chunk_gated_delta_rule(...)       # torch_chunk_gated_delta_rule, chunk_size=64
```

## Result

Exported at S=128 with the sequence axis marked dynamic, then run at other lengths
and compared against PyTorch (max relative error):

| | nodes | control flow | MatMul | S=128 | S=192 | S=252 |
|---|---:|---|---:|---|---|---|
| `torch_chunk_gated_delta_rule` (dynamo) | 1,436 | none | 13 | 1.4e-06 | **1.0** | **1.0** |
| `torch_chunk_gated_delta_rule` (legacy) | 8,711 | none | 13 | 3.8e-06 | **1.0** | **1.0** |
| `torch_recurrent_gated_delta_rule` (legacy) | 11,564 | none | 0 | 4.1e-06 | **1.0** | **1.0** |
| `torch_recurrent_gated_delta_rule` (dynamo) | — | export fails | — | — | — | — |

Both unroll their Python loop, so neither survives a change of sequence length —
and they fail *silently*: the graph runs, returns the correct shape, and is
numerically wrong. That is a good reason for the shipped export to use a `Scan`,
which is correct at any length.

So the open question is not "why not the chunked function" but **"why does the
`Scan` iterate over positions rather than over chunks"** — same recurrence, body
takes a block of 64, state carried block to block, ~4 iterations instead of 252.
Note the chunked export is 8x smaller and does its work in 13 `MatMul`s, while the
recurrent one has none — per-position elementwise all the way down.

## Reproduce

```bash
pip install torch transformers onnx onnxscript onnxruntime
python export_both.py       # exports 4 variants, prints node census
python verify_numerics.py   # runs each at 128/192/252 vs PyTorch
```

transformers 5.12.1, torch 2.12.1. Small dims (4 heads, 32 dim) — the structure is
what matters, not the size.
