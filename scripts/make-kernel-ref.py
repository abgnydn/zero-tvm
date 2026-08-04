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


def qwen36moe_block(seed: int):
    """Whole SparseMoeBlock of layer 0 — reference computed by MLX's OWN block.

    Staged deliberately: the routed-expert path is the part with no precedent in
    this engine, so the bundle also carries the top-k indices/scores the
    reference chose. A GPU test can validate the expert FFN + combine against
    them before the router (which is 8-bit quantised, unlike everything else)
    is implemented.
    """
    import mlx.core as mx, mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModelArgs, SparseMoeBlock

    W = os.path.join(ROOT, ".weights-local/Qwen3.6-35B-A3B-MLX-4bit")
    if not os.path.exists(W):
        sys.exit(f"missing checkpoint: {W}")
    cfg = json.load(open(os.path.join(W, "config.json")))
    t = cfg.get("text_config", cfg)
    args = TextModelArgs(**{k: v for k, v in t.items() if k in TextModelArgs.__dataclass_fields__})

    blk = SparseMoeBlock(args)
    qz = cfg["quantization"]
    nn.quantize(blk, group_size=qz["group_size"], bits=qz["bits"])
    PRE = "language_model.model.layers.0.mlp."
    g8 = qz.get(PRE + "gate")          # the router ships at 8 bits, not 4
    if g8:
        blk.gate = nn.QuantizedLinear(args.hidden_size, args.num_experts, bias=False,
                                      group_size=g8["group_size"], bits=g8["bits"])
        blk.shared_expert_gate = nn.QuantizedLinear(args.hidden_size, 1, bias=False,
                                                    group_size=g8["group_size"], bits=g8["bits"])
    w = mx.load(os.path.join(W, "model-00001-of-00004.safetensors"))
    blk.load_weights([(k[len(PRE):], v) for k, v in w.items() if k.startswith(PRE)])
    mx.eval(blk.parameters())

    # Realistic scale: the MLP sees post_attention_layernorm(h), i.e. RMS ≈ 1.
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, args.hidden_size)).astype(np.float32)).astype(mx.bfloat16)
    y = blk(x)
    # Split the reference so a partial GPU implementation can be validated
    # against exactly the part it implements, instead of a loosened tolerance.
    g = mx.softmax(blk.gate(x).astype(mx.float32), axis=-1, precise=True)
    k = args.num_experts_per_tok
    inds = mx.argpartition(g, kth=-k, axis=-1)[..., -k:]
    sc = mx.take_along_axis(g, inds, axis=-1)
    if args.norm_topk_prob:
        sc = sc / sc.sum(axis=-1, keepdims=True)
    y_routed = (blk.switch_mlp(x, inds) * sc[..., None]).sum(axis=-2)
    y_shared = mx.sigmoid(blk.shared_expert_gate(x)) * blk.shared_expert(x)
    mx.eval(y, inds, sc, y_routed, y_shared)

    arrays = {
        "x_f16": np.array(x.astype(mx.float32)).astype(np.float16).ravel(),
        "y_ref_f32": np.array(y.astype(mx.float32)).astype(np.float32).ravel(),
        "y_routed_f32": np.array(y_routed.astype(mx.float32)).astype(np.float32).ravel(),
        "y_shared_f32": np.array(y_shared.astype(mx.float32)).astype(np.float32).ravel(),
        "topk_idx_u32": np.array(inds).astype(np.uint32).ravel(),
        "topk_score_f32": np.array(sc).astype(np.float32).ravel(),
    }
    # Expert + shared weights, exactly as stored (nibbles untouched); bias2 = 7s+b.
    for proj in ("gate_proj", "up_proj", "down_proj"):
        for who, key in (("exp", f"switch_mlp.{proj}"), ("shd", f"shared_expert.{proj}")):
            Wq = np.array(w[PRE + key + ".weight"].astype(mx.uint32))
            S = np.array(w[PRE + key + ".scales"].astype(mx.float32))
            B = np.array(w[PRE + key + ".biases"].astype(mx.float32))
            arrays[f"{who}_{proj}_w_u32"] = Wq.ravel()
            arrays[f"{who}_{proj}_s_f16"] = S.astype(np.float16).ravel()
            arrays[f"{who}_{proj}_b2_f16"] = (7.0 * S + B).astype(np.float16).ravel()

    # Router + shared gate ship at 8 bits, group 64 — a different kernel path.
    for who, key in (("router", "gate"), ("shdgate", "shared_expert_gate")):
        arrays[f"{who}_w_u32"] = np.array(w[PRE + key + ".weight"].astype(mx.uint32)).ravel()
        arrays[f"{who}_s_f16"] = np.array(w[PRE + key + ".scales"].astype(mx.float32)).astype(np.float16).ravel()
        arrays[f"{who}_b_f16"] = np.array(w[PRE + key + ".biases"].astype(mx.float32)).astype(np.float16).ravel()

    shp = lambda k_: list(np.array(w[PRE + k_ + ".weight"]).shape)
    return {
        "arrays": arrays,
        "meta": {"kernel": "moe_block",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": "layers.0.mlp (SparseMoeBlock)",
                 "reference": "mlx_lm qwen3_next.Qwen3NextSparseMoeBlock",
                 "hidden": int(args.hidden_size), "num_experts": int(args.num_experts),
                 "top_k": int(k), "moe_intermediate": int(args.moe_intermediate_size),
                 "shared_intermediate": int(args.shared_expert_intermediate_size),
                 "norm_topk_prob": bool(args.norm_topk_prob),
                 "group": qz["group_size"],
                 "exp_shapes": {p: shp(f"switch_mlp.{p}") for p in ("gate_proj", "up_proj", "down_proj")},
                 "shd_shapes": {p: shp(f"shared_expert.{p}") for p in ("gate_proj", "up_proj", "down_proj")},
                 "router_bits": int(g8["bits"]) if g8 else int(qz["bits"]),
                 "note": "routed y = sum_k score_k * down(silu(gate)*up); shared added with sigmoid gate"},
    }


PRODUCERS = {"qwen36moe": qwen36moe, "qwen36moe_block": qwen36moe_block}

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
    m = b["meta"]
    print(f"  {m['tensor']}  (kernel: {m['kernel']}, reference: {m['reference']})")
    if "cross_check_rel_err" in m:
        print(f"  cross-check vs {m['reference']}: {m['cross_check_rel_err']:.2e}")
    print(f"  arrays: {', '.join(sorted(b['arrays']))[:200]}")
