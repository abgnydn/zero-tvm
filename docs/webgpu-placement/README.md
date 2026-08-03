# Where does onnxruntime-web put an ONNX `Scan`?

Supporting evidence for the transformers.js#1599 analysis (see
`../tjs-1599-comment.md` and `../tjs-1599-repro.ipynb`).

The notebook establishes that the Qwen3.5-4B ONNX export evaluates its 24
gated-DeltaNet layers with a sequential `Scan` on the prefill path. This
directory answers the follow-up: **what does the WebGPU backend do with it?**

## Result

`Scan` has no WebGPU kernel. ORT says so itself and falls back to CPU:

```
[GetCapability] webgpu kernel not found in registries for Op type: Scan
[transformer_memcpy] Add MemcpyFromHost after y for JsExecutionProvider
Node placements
  Node(s) placed on [JsExecutionProvider]. Number of nodes: 9
    Exp, Mul, Mul, ReduceSum, Sub, Add, Mul, MatMul, MemcpyFromHost
  Node(s) placed on [CPUExecutionProvider]. Number of nodes: 1
    Scan ()
```

So the loop control runs on CPU while the body ops are assigned to WebGPU,
with a host copy inserted at the boundary. Independently, `Scan`, `If` and
`Loop` appear zero times in onnxruntime-web's shipped JSEP op registry
(178 entries), which is consistent with the log above.

This is a property of the runtime's op registry rather than of any one model,
so it does not depend on the 2.5 GB Qwen3.5 weights — a ~1 KB model with the
same shape reproduces it.

## Reproduce

```bash
python3 build-toy.py                                  # writes toy_scan.onnx (~1 KB)
cp -r ../../node_modules/onnxruntime-web/dist ./ort-dist
node run.mjs
```

`toy_scan.onnx` is one `Scan` whose body is a miniature gated delta rule
(`Exp`/`Mul`/`ReduceSum`/`Sub`/`Add`) plus a `MatMul` outside it as a control —
the `MatMul` is a op WebGPU definitely supports, so it shows the partitioning
is real and not a wholesale CPU fallback.

## How much of the prefill time is this?

`build-real-scan.py` lifts the export's **own** delta-rule `Scan` out into a
standalone ~2.5 KB model (the body is pure elementwise/reduce arithmetic with no
weights, so only the 1.4 MB graph is needed), and `scan-bench.html` times it on
WebGPU at real dimensions — state `[1,32,128,128]`, one layer.

```bash
python3 build-real-scan.py         # 1.4 MB download -> real_scan.onnx (2.5 KB)
node run.mjs scan-bench.html
```

On an M2 Max, Chrome, cost per position is flat across a 252x range of sequence
length — the signature of a loop that batches nothing:

| | run 1 | run 2 |
|---|---:|---:|
| ms per position (one layer) | 2.83 | 2.54 |
| one layer, 252 positions | 714 ms | 640 ms |
| **x24 GDN layers** | **17.1 s** | **15.4 s** |

Per prompt token that is ~61–68 ms of `Scan` alone. The issue reports ~16 s TTFT
at a comparable prompt length, and an end-to-end sweep on this same machine gave
72–74 ms per prompt token. So the recurrence is not one contributor among many —
it accounts for most of the prefill.

Treat the spread between runs as the real error bar; absolute numbers move with
machine load, which is why the range is quoted rather than a single figure.

## Scope

Measured: `Scan` is rejected by the WebGPU EP and placed on CPU; body ops go to
WebGPU; a memcpy is inserted between them; and the isolated real `Scan` costs
~15–17 s for 24 layers at a 252-token prompt. Chrome, onnxruntime-web
1.22.0-dev, Apple M2 Max. Not measured: the full model end-to-end in one
session, or any other backend.
