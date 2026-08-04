#!/usr/bin/env python3
"""
Produce a real-weight kernel-validation bundle for tests/kernels/real-weights.mjs.

The synthetic suite (tests/kernels/run.mjs) proves a kernel matches our own
reference on our own random data. It cannot prove we read a VENDOR's on-disk
quantization correctly — nibble order, group size, symmetric vs affine. So this
pulls a real tensor out of a real checkpoint and computes the reference with the
vendor's OWN library, which is the only thing that settles those questions.

The reference is deliberately computed at the precision the GPU will see
(f16 scales/activations, f32 accumulation); otherwise the comparison is unfair
to the kernel and the tolerance has to be loosened until it proves nothing.

Adding a format: write a new producer function and register it in PRODUCERS.
The bundle shape (meta.json + raw .bin) never changes, so the .mjs side is untouched.

  python3 scripts/make-kernel-ref.py --model qwen36moe
  node tests/kernels/real-weights.mjs
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, ".weights-local/kernel-refs")


def qwen36moe(seed: int):
    """MLX affine int4 (group 64) — one expert slice of a Qwen3.6 MoE FFN projection."""
    import mlx.core as mx

    src = os.path.join(ROOT, ".weights-local/Qwen3.6-35B-A3B-MLX-4bit/model-00001-of-00004.safetensors")
    if not os.path.exists(src):
        sys.exit(f"missing checkpoint: {src}\n"
                 "  huggingface-cli download lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit "
                 f"--local-dir {os.path.dirname(src)}")
    key = "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
    expert = 0
    w = mx.load(src)
    Wq = np.array(w[key + ".weight"][expert].astype(mx.uint32))     # [N, K/8] u32
    S = np.array(w[key + ".scales"][expert].astype(mx.float32))     # [N, K/64]
    B = np.array(w[key + ".biases"][expert].astype(mx.float32))

    N, KP = Wq.shape
    K, G = KP * 8, 64
    NG = K // G
    assert S.shape == (N, NG), f"scales {S.shape} != {(N, NG)}"

    f16 = lambda a: a.astype(np.float16)
    # The kernel consumes bias2 = 7*scale + bias, because it reuses the existing
    # (nibble-7) dot:  w = s*(q-7) + (7s+b).
    scales_f16, bias2_f16 = f16(S), f16(7.0 * S + B)
    x_f16 = f16(np.random.default_rng(seed).standard_normal(K) * 0.05)

    nib = np.zeros((N, K), dtype=np.float32)
    for i in range(8):
        nib[:, i::8] = (Wq >> (4 * i)) & 0xF
    xg = x_f16.astype(np.float32).reshape(NG, G)
    dot = ((nib.reshape(N, NG, G) - 7.0) * xg[None]).sum(axis=2)
    y = (scales_f16.astype(np.float32) * dot).sum(1) + (bias2_f16.astype(np.float32) * xg.sum(1)[None]).sum(1)

    # Independent cross-check against the vendor's own dequantiser. If this
    # disagrees, our understanding of the format is wrong and the bundle is void.
    W_true = np.array(mx.dequantize(w[key + ".weight"][expert], w[key + ".scales"][expert],
                                    w[key + ".biases"][expert], group_size=G, bits=4).astype(mx.float32))
    y_true = W_true @ x_f16.astype(np.float32)
    rel = float(np.abs(y - y_true).max() / (np.abs(y_true).max() + 1e-9))
    if rel > 5e-3:
        sys.exit(f"reference disagrees with mx.dequantize (rel {rel:.2e}) — format misread, refusing to write")

    return {
        "arrays": {"weights_u32": Wq.astype(np.uint32), "scales_f16": scales_f16,
                   "bias2_f16": bias2_f16, "x_f16": x_f16, "y_ref_f32": y.astype(np.float32)},
        "meta": {"kernel": "affine_matmul",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": f"{key} [expert {expert}]",
                 "reference": "mlx.core.dequantize",
                 "N": int(N), "K": int(K), "K_PACKED": int(KP), "GROUP": G,
                 "GROUPS_PER_ROW": int(NG),
                 "note": "y = sum_g( s_g*dot_g + bias2_g*xsum_g ), bias2 = 7*s + b",
                 "cross_check_rel_err": rel},
    }


PRODUCERS = {"qwen36moe": qwen36moe}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRODUCERS))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = os.path.join(OUT_ROOT, a.model)
    os.makedirs(out, exist_ok=True)
    b = PRODUCERS[a.model](a.seed)
    for name, arr in b["arrays"].items():
        arr.tofile(os.path.join(out, f"{name}.bin"))
    json.dump(b["meta"], open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"wrote {out}")
    print(f"  {b['meta']['tensor']}  N={b['meta']['N']} K={b['meta']['K']}")
    print(f"  cross-check vs {b['meta']['reference']}: {b['meta']['cross_check_rel_err']:.2e}")
