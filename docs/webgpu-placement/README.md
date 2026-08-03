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

## Scope

Measured: `Scan` is rejected by the WebGPU EP and placed on CPU; body ops are
placed on WebGPU; a memcpy is inserted between them. Observed on this toy model
in Chrome with onnxruntime-web 1.22.0-dev. Not measured here: the placement of
the real model's fp16 bodies, or any wall-clock attribution.
