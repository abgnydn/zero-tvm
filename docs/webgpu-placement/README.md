# Where does onnxruntime-web put an ONNX `Scan`?

Supporting evidence for the transformers.js#1599 analysis (see
`../tjs-1599-repro.ipynb`).

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
with a host copy inserted at the boundary. Independently, the WebGPU EP's
`core/providers/webgpu/controlflow/` directory contains only `if.cc` and `if.h`
at HEAD, so `Scan` and `Loop` have no WebGPU kernel to be assigned to.

(An earlier version of this file cited a count of 178 entries in the shipped JS
op registry as the evidence here. That method was retracted in `a5c22c8`: from
1.26 the WebGPU EP is native WASM rather than JSEP, so the JS registry is no
longer authoritative. The provider source is. The conclusion is unchanged.)

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
WebGPU; a memcpy is inserted between them; and the recurrence accounts for
**91-93% of prefill** at a 252-token prompt, measured by in-place ablation with a
control re-run, reproduced twice. Chrome, Apple M2 Max, onnxruntime-web 1.22.0-dev
and 1.26.0-dev. Not measured: any backend other than WebGPU, or any hardware other
than this one.

## Full model vs isolated Scan (same session)

`full-model-bench.html` loads the real 2.5 GB decoder in the browser and times
prefill, alongside the isolated `Scan`, in one session — so the ratio is not
affected by machine load the way two separate absolute numbers would be.
Serve `.weights-local/onnx/Qwen3.5-4B-ONNX/onnx` at `/weights/` (stream it; the
data file is 2.1 GB) and open with `?s=252`.

### The right way: ablate the `Scan` in place

Extrapolating from an isolated `Scan` is biased — it charges per-run host/device
transfers 24 times over, and returns >100% in the fast regime (see below), which is
impossible. The fix is to not extrapolate at all.

`build-ablated.py` rewrites the real decoder with every `Scan` replaced by
shape-preserving `Identity` nodes (`state_fin <- state_f`, `y <- vf`). Numerically
meaningless, structurally valid, ~free to execute, and it shares the same 2.1 GB
`.onnx_data` so it costs no extra disk. Everything else in the model is untouched —
including the other 78 nodes of each GDN layer (projections, conv, gates, norms), so
what is removed is the recurrence and nothing else.

```bash
python3 build-ablated.py     # 1.4 MB graph in, 24 Scans out
node run.mjs ablation.html   # full vs ablated, same session, with a control re-run
```

| | full (with `Scan`) | ablated (no `Scan`) | control drift | **`Scan` share** |
|---|---:|---:|---:|---:|
| run 1 | 13.99 s | 1.24 s | 4% | **91.3%** |
| run 2 | 13.71 s | 0.95 s | 2% | **93.1%** |

252-token prompt, Apple M2 Max, Chrome, onnxruntime-web 1.26.0-dev. The full model is
re-measured after the ablated one as a control; a run whose two full-model timings
disagree is discarded. Everything that is *not* the recurrence — every quantized
matmul, all 8 attention layers, the norms, and a 125 MB logits tensor — accounts for
the remaining ~1 second.

### What the isolated method got wrong

Kept here because it is a useful negative result. Timing one layer's `Scan` standalone
and multiplying by 24 gives, across three runs whose before/after controls agreed:

| run | control drift | isolated x24 | full prefill | ratio |
|---|---|---:|---:|---:|
| 1 | 8% | 27.8 s | 31.06 s | 90% |
| 4 | 14% | 16.0 s | 13.90 s | 115% |
| 5 | 1% | 16.5 s | 13.59 s | 122% |

A `Scan` inside the model cannot cost more than the model, so this is an upper bound
with systematic upward bias — about 25% high against the ablation figure. The 90% in
run 1 was luck, not measurement.

Two of the five runs were discarded before they produced a number, because their
before/after controls disagreed by 29% and 110%. That discard rule is the only
reason the remaining three were trustworthy enough to refute the method with.

Note the two machine regimes: runs 1-3 caught the machine about 2x slower
(isolated 1.1-1.3 s, full 31 s), runs 4-5 the fast regime (isolated ~0.69 s, full
~13.6 s). Absolute times move by 2x with load, which is why only same-session
ratios are worth reporting.

The loop control runs on the CPU, so CPU contention hits this workload directly.
Under a single busy core the isolated `Scan` slowed 4.4x (546 ms -> 1229 ms at the
same size) and session creation went 4.6 s -> 37.6 s. A GPU-bound stage would not
lose 4.4x to one busy core.
