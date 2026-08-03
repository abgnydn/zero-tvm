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
Node placements
  Node(s) placed on [WebGpuExecutionProvider]. Number of nodes: 16
  Node(s) placed on [CPUExecutionProvider].    Number of nodes: 1
```

(on `onnxruntime-web@1.26.0-dev.20260416`, the build `@huggingface/transformers@4.2.0`
pins — the 16 are the delta-rule body, the 1 is the `Scan`. The identical split appears
on `1.22.0-dev`, where the provider was still called `JsExecutionProvider`.)

So the loop control runs on CPU while the body ops are assigned to WebGPU,
with a host copy inserted at the boundary. Independently, `Scan`, `If` and
`Loop` appear zero times in onnxruntime-web's shipped op registry
(178 entries in both 1.22 and 1.26), which is consistent with the log above.

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

On an M2 Max, Chrome — note the per-position cost barely moves as the sequence
grows:

| | 1.22 run 1 | 1.22 run 2 | 1.26 |
|---|---:|---:|---:|
| ms per position (one layer) | 2.83 | 2.54 | 2.19 |
| one layer, 252 positions | 714 ms | 640 ms | 552 ms |
| **x24 GDN layers** | **17.1 s** | **15.4 s** | **13.2 s** |

Cost per position does not amortize: from 8 to 252 positions — a 30x range — each
position still costs about the same. That is the signature of a loop that batches
nothing, and it makes the total scale linearly with prompt length.

13-17 s is the recurrence *alone*, in isolation, on this machine. The ~16 s TTFT in
the issue was measured on different hardware at an unknown prompt length, so this is
not a percentage — it is "the same order as the entire reported TTFT", i.e. the
dominant term rather than one contributor among several. Treat the spread across runs
as the error bar.

**Test against the ORT-web build transformers.js pins**, not whatever is hoisted in
node_modules: 4.2.0 -> `onnxruntime-web@1.26.0-dev.20260416`, 3.8.1 -> `1.22.0-dev`.

## Scope

Measured: `Scan` is rejected by the WebGPU EP and placed on CPU; body ops go to
WebGPU; a memcpy is inserted between them; and the isolated real `Scan` costs
~13-17 s for 24 layers at a 252-token prompt. Chrome, Apple M2 Max,
onnxruntime-web 1.22.0-dev and 1.26.0-dev. Not measured: the full model
end-to-end in one session, or any backend other than WebGPU.
