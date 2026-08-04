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
~13-17 s for 24 layers at a 252-token prompt, which is the same order as the
full prefill measured in the same session. Chrome, Apple M2 Max,
onnxruntime-web 1.22.0-dev and 1.26.0-dev. NOT established: a precise share —
the isolated x24 method is an upper bound and returns >100% in the fast regime.
Also not measured: any backend other than WebGPU.

## Full model vs isolated Scan (same session)

`full-model-bench.html` loads the real 2.5 GB decoder in the browser and times
prefill, alongside the isolated `Scan`, in one session — so the ratio is not
affected by machine load the way two separate absolute numbers would be.
Serve `.weights-local/onnx/Qwen3.5-4B-ONNX/onnx` at `/weights/` (stream it; the
data file is 2.1 GB) and open with `?s=252`.

Five controlled runs (2026-08-04, machine load ~3.3). Each run times the isolated
`Scan` **before and after** the full-model prefill; if those two controls disagree
the run is discarded, so a drifting machine eliminates itself rather than
producing a plausible-looking number.

| run | control drift | isolated x24 | full prefill | ratio |
|---|---|---:|---:|---:|
| 1 | 8% ok | 27.8 s | 31.06 s | 90% |
| 2 | 29% — discarded | | | |
| 3 | 110% — discarded | | | |
| 4 | 14% ok | 16.0 s | 13.90 s | 115% |
| 5 | **1% ok** | 16.5 s | 13.59 s | 122% |

**Read this as a refutation of the method, not as a 122% share.** A `Scan` that lives
inside the model cannot cost more than the whole model, so `isolated x 24` is an
**upper bound with a systematic upward bias**, not an estimate. Across valid runs it
lands at 90-122%, which is too loose to quote a percentage from.

What survives: at a 252-token prompt the recurrence alone costs on the order of the
entire prefill (16.5 s against 13.6 s measured in the same session). That supports
"the recurrence dominates prefill" and does not support any specific number.

Note the two regimes: runs 1-3 caught the machine at ~2x slower (isolated 1.1-1.3 s,
full 31 s), runs 4-5 in the fast regime (isolated ~0.69 s, full ~13.6 s) matching the
first clean run ever taken (546 ms / 13.11 s). Absolute times move by 2x with load;
that is why the same-session ratio is the only thing worth reporting — and even that
is bounded, not pinned.

## Scope

Measured: `Scan` is rejected by the WebGPU EP and placed on CPU; body ops go to
WebGPU; a memcpy is inserted between them; and the isolated real `Scan` costs
~13-17 s for 24 layers at a 252-token prompt, which is the same order as the
full prefill measured in the same session. Chrome, Apple M2 Max,
onnxruntime-web 1.22.0-dev and 1.26.0-dev. NOT established: a precise share —
the isolated x24 method is an upper bound and returns >100% in the fast regime.
Also not measured: any backend other than WebGPU.

## Full model vs isolated Scan (same session)

`full-model-bench.html` loads the real 2.5 GB decoder in the browser and times
prefill, alongside the isolated `Scan`, in one session — so the ratio is not
affected by machine load the way two separate absolute numbers would be.
Serve `.weights-local/onnx/Qwen3.5-4B-ONNX/onnx` at `/weights/` (stream it; the
data file is 2.1 GB) and open with `?s=252`.

One clean run (session created in 4.6 s, three repeats within 0.5%):

| | |
|---|---:|
| isolated `Scan` x24, 252 positions | 13.10 s |
| **full decoder prefill, 252 tokens** | **13.11 s** (13.05 / 13.11 / 13.12) |

i.e. the recurrence accounted for essentially the whole prefill. **Treat this as
one data point, not a settled number.** Repeat runs afterwards were unusable: a
leaked browser process was pegging a core, the isolated `Scan` slowed 4.4x
(546 ms -> 1229 ms at the same size), session creation went 4.6 s -> 37.6 s, and
the ratio moved to ~73%. Confirm on a quiet machine before quoting a percentage.

Worth noting *why* it degrades so hard under CPU load: the loop control runs on
the CPU, so CPU contention hits this workload directly. A GPU-bound stage would
not lose 4.4x to a single busy core.
