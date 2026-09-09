#!/usr/bin/env python3
"""MLX-KERNEL-AB — time mx.quantized_matmul alone, at the shapes ours runs.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-kernel-ab.py --json

The LM Studio comparison in BENCH.md is a RUNTIME result — two whole pipelines,
neither isolating a kernel. This is the other half of the kernel measurement:
`scripts/kernel-ab-ours.mjs` times our WGSL int4 matmul, this times MLX's own
Metal one, and `scripts/kernel-ab.mjs` alternates the two processes.

Nothing ready-made existed to reuse. mlx ships no benchmarks in the wheel;
mlx_lm.benchmark is model-level (tok/s) and takes no shapes; and upstream's
own benchmarks/python/comparative/bench_mlx.py `quant_matmul_*` entries build
operands with mx.random.normal and die on
`[quantized_matmul] The weight matrix should be uint32 but received float32` —
they cannot have been run. So: written here, deliberately small.

THE THINGS THAT MAKE IT WRONG IF YOU SKIP THEM
  - mx arrays are LAZY. Timing without mx.eval measures graph construction:
    0.0014 ms/iter against a true 2.1 ms, i.e. 1500x too fast.
  - mx.eval is itself blocking, so mx.synchronize after it is redundant (it is
    only needed after mx.async_eval).
  - The first call builds and caches a Metal pipeline. Warm up.
  - The inputs must be eval'd BEFORE the timed region or their construction
    lands in the measurement.

AND THE ONE THAT WOULD HAVE FLATTERED MLX: weights are ROTATED over distinct
sets, exactly as the WGSL side does. Looping one matrix measures cache, and our
side pays cold reads because real decode walks a different layer's weights
every dispatch. Timing MLX warm against ours cold would not be a comparison.
"""

import argparse, json, time
import mlx.core as mx

# Qwen3.5-9B-MLX-4bit — the model the LM Studio A/B runs.
SHAPES = [
    ("ffn_gate_up", 4096, 24576),
    ("ffn_down", 12288, 4096),
    ("o_proj", 4096, 4096),
    ("c_attn", 4096, 10240),
]
GROUP, BITS = 64, 4

p = argparse.ArgumentParser()
p.add_argument("--json", action="store_true")
p.add_argument("--iters", type=int, default=200)
p.add_argument("--ms", default="1,256", help="comma-separated M values")
a = p.parse_args()
MS = [int(x) for x in a.ms.split(",")]

out = []
for name, K, N in SHAPES:
    # uint32 (N, K/8) + f16 scales/biases (N, K/64) — the same bytes our kernel
    # reads, so GB/s means the same thing on both sides.
    w_bytes = N * (K // 8) * 4 + N * (K // GROUP) * 4
    copies = max(2, min(24, -(-(256 * 2**20) // w_bytes)))

    sets = []
    for _ in range(copies):
        wf = mx.random.normal((N, K)).astype(mx.float16)
        sets.append(mx.quantize(wf, group_size=GROUP, bits=BITS))
    mx.eval([t for s in sets for t in s])

    for M in MS:
        x = mx.random.normal((M, K)).astype(mx.float16)
        mx.eval(x)

        def batch(n):
            # One list of n calls then a single barrier — the analogue of
            # encoding n dispatches into one compute pass and waiting once.
            ys = []
            for i in range(n):
                w, s, b = sets[i % len(sets)]
                ys.append(mx.quantized_matmul(
                    x, w, s, b, transpose=True, group_size=GROUP, bits=BITS))
            mx.eval(ys)

        batch(10)                      # warm: pipeline build + cache
        t0 = time.perf_counter()
        batch(a.iters)
        ms = (time.perf_counter() - t0) * 1000 / a.iters

        out.append({
            "name": name, "M": M, "K": K, "N": N, "ms": ms, "copies": copies,
            "gbPerS": w_bytes / (ms / 1000) / 1e9,
            "gflops": (2 * M * K * N) / (ms / 1000) / 1e9,
            "kernel": f"mx.quantized_matmul g{GROUP} b{BITS}",
        })
        del x
    del sets
    mx.clear_cache()   # MLX's buffer cache otherwise perturbs the next shape

if a.json:
    print(json.dumps({"engine": "mlx", "iters": a.iters, "results": out}))
else:
    print(f"\nMLX mx.quantized_matmul (4-bit, group 64), {a.iters} calls over rotating weight sets\n")
    print(f"  {'shape':<13} {'M':>4} {'K':>6} {'N':>6} {'ms':>8} {'GB/s':>7} {'GFLOP/s':>9}")
    for r in out:
        print(f"  {r['name']:<13} {r['M']:>4} {r['K']:>6} {r['N']:>6} "
              f"{r['ms']:>8.3f} {r['gbPerS']:>7.0f} {r['gflops']:>9.0f}")
