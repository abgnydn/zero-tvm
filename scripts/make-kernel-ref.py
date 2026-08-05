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
    # The affine kernel consumes MLX's bias verbatim: y = s*sum(x*q) + b*sum(x).
    scales_f16, bias_f16 = f16(S), f16(B)
    x_f16 = f16(np.random.default_rng(seed).standard_normal(K) * 0.05)

    nib = np.zeros((N, K), dtype=np.float32)
    for i in range(8):
        nib[:, i::8] = (Wq >> (4 * i)) & 0xF
    xg = x_f16.astype(np.float32).reshape(NG, G)
    dot = (nib.reshape(N, NG, G) * xg[None]).sum(axis=2)
    y = (scales_f16.astype(np.float32) * dot).sum(1) + (bias_f16.astype(np.float32) * xg.sum(1)[None]).sum(1)

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
                   "bias_f16": bias_f16, "x_f16": x_f16, "y_ref_f32": y.astype(np.float32)},
        "meta": {"kernel": "affine_matmul",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": f"{key} [expert {expert}]",
                 "reference": "mlx.core.dequantize",
                 "N": int(N), "K": int(K), "K_PACKED": int(KP), "GROUP": G,
                 "GROUPS_PER_ROW": int(NG),
                 "note": "y = sum_g( s_g*dot_g + b_g*xsum_g ), nibbles read raw",
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
    # f32, not the checkpoint's bfloat16 — a bf16 reference is ~1.7e-2 off f32
    # truth on a projection this size, several times our own error, so it would
    # set the tolerance instead of the kernel setting it (see qwen36gdn).
    blk.set_dtype(mx.float32)
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, args.hidden_size)).astype(np.float32))
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
    # LAYOUT: the shared expert is appended to the stacked expert tensors as
    # index E, and its gate as row E of the router. It is then not a special
    # case at all — one kernel, one dispatch, K+1 slots, and `moe_combine`
    # becomes a plain weighted sum. Costs one expert's worth of memory (0.4%)
    # and removes an entire second matmul path. This is the layout the engine's
    # loader should build, so the harness validates the kernels as they'll run.
    # Nibbles and biases are stored verbatim (see the note in qwen36moe).
    def stack(key_e, key_s):
        """[E, ...] ++ shared -> [E+1, ...], for weight/scales/biases alike."""
        out = []
        for part in ("weight", "scales", "biases"):
            dt = mx.uint32 if part == "weight" else mx.float32
            e = np.array(w[f"{PRE}{key_e}.{part}"].astype(dt))
            s = np.array(w[f"{PRE}{key_s}.{part}"].astype(dt))
            if s.ndim == e.ndim - 1:
                s = s[None]                   # switch_mlp is [E, N, ..]; shared_expert is [N, ..]
            assert e.shape[1:] == s.shape[1:], f"{key_s}.{part} {s.shape} cannot append to {e.shape}"
            out.append(np.concatenate([e, s]))
        return out

    for proj in ("gate_proj", "up_proj", "down_proj"):
        Wq, S, B = stack(f"switch_mlp.{proj}", f"shared_expert.{proj}")
        arrays[f"exp_{proj}_w_u32"] = Wq.ravel()
        arrays[f"exp_{proj}_s_f16"] = S.astype(np.float16).ravel()
        arrays[f"exp_{proj}_b_f16"] = B.astype(np.float16).ravel()

    # Router ships at 8 bits, group 64 — a different kernel path. Row E is the
    # shared-expert gate, so one dispatch produces every logit the block needs.
    Wq, S, B = stack("gate", "shared_expert_gate")
    arrays["router_w_u32"] = Wq.ravel()
    arrays["router_s_f16"] = S.astype(np.float16).ravel()
    arrays["router_b_f16"] = B.astype(np.float16).ravel()

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
                 "shared_expert_index": int(args.num_experts),
                 "exp_shapes": {p: shp(f"switch_mlp.{p}") for p in ("gate_proj", "up_proj", "down_proj")},
                 "shd_shapes": {p: shp(f"shared_expert.{p}") for p in ("gate_proj", "up_proj", "down_proj")},
                 "router_bits": int(g8["bits"]) if g8 else int(qz["bits"]),
                 "note": "every stacked tensor carries the shared expert at index shared_expert_index; "
                         "y = sum over K+1 slots of score_slot * down(silu(gate)*up), "
                         "with score_K = sigmoid(router logit E)"},
    }


def _qwen36_args():
    """TextModelArgs + the checkpoint's own quantization config."""
    import json as _json
    W = os.path.join(ROOT, ".weights-local/Qwen3.6-35B-A3B-MLX-4bit")
    if not os.path.exists(W):
        sys.exit(f"missing checkpoint: {W}")
    cfg = _json.load(open(os.path.join(W, "config.json")))
    t = cfg.get("text_config", cfg)
    from mlx_lm.models.qwen3_5 import TextModelArgs
    args = TextModelArgs(**{k: v for k, v in t.items() if k in TextModelArgs.__dataclass_fields__})
    return W, cfg, args


def _affine(w, key, out, prefix):
    """Emit one MLX affine int4 tensor (weight/scales/biases) verbatim."""
    import mlx.core as mx
    import numpy as _np
    out[f"{prefix}_w_u32"] = _np.array(w[key + ".weight"].astype(mx.uint32)).ravel()
    out[f"{prefix}_s_f16"] = _np.array(w[key + ".scales"].astype(mx.float32)).astype(_np.float16).ravel()
    out[f"{prefix}_b_f16"] = _np.array(w[key + ".biases"].astype(mx.float32)).astype(_np.float16).ravel()


def _f16(a):
    import mlx.core as mx
    import numpy as _np
    return _np.array(a.astype(mx.float32)).astype(_np.float16).ravel()


def _f32(a):
    import mlx.core as mx
    import numpy as _np
    return _np.array(a.astype(mx.float32)).astype(_np.float32).ravel()


def qwen36gdn(seed: int):
    """Layer 0's gated-DeltaNet sub-block: input_layernorm -> linear_attn -> r.

    The engine already runs GDN for Qwen3.5 and its kernels are checked against a
    JS reference in compile-qwen35.mjs. What is NOT checked there is this model:
    Qwen3.6 quantises every weight MLX-affine (group 64) rather than MLC-symmetric
    (group 32), and a shared reference can share a misreading with the kernel. So
    this compares against mlx_lm's OWN module, on the checkpoint's own weights.

    A DECODE step with a SEEDED cache, not a first token: three tokens are
    prefilled first so the depthwise conv sees four real taps and the recurrent
    state is non-zero. A first-token bundle would leave three of the four conv
    taps at zero and let a tap-ordering bug pass.
    """
    import mlx.core as mx, mlx.nn as nn
    from mlx_lm.models.qwen3_5 import GatedDeltaNet
    from mlx_lm.models.cache import ArraysCache

    W, cfg, args = _qwen36_args()
    qz = cfg["quantization"]
    gdn = GatedDeltaNet(args)
    norm1 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    nn.quantize(gdn, group_size=qz["group_size"], bits=qz["bits"])
    _f32_params = True

    PRE = "language_model.model.layers.0."
    w = mx.load(os.path.join(W, "model-00001-of-00004.safetensors"))
    gdn.load_weights([(k[len(PRE + "linear_attn."):], v) for k, v in w.items()
                      if k.startswith(PRE + "linear_attn.")])
    norm1.load_weights([("weight", w[PRE + "input_layernorm.weight"])])
    gdn.set_dtype(mx.float32)
    norm1.set_dtype(mx.float32)
    mx.eval(gdn.parameters(), norm1.parameters())

    # f32, not the checkpoint's bfloat16. bf16 carries 8 mantissa bits, so a
    # bf16 reference is itself ~1.7e-2 wrong against f32 truth on this
    # projection — measured — which is SEVEN TIMES our own f16-activation error
    # of 2.4e-3. A test whose noise floor sits above the error it is looking for
    # cannot fail for the right reason. The quantized weights are untouched; only
    # the activations and accumulation move to f32.
    rng = np.random.default_rng(seed)
    D, PREFILL = args.hidden_size, 3
    hs = mx.array(rng.standard_normal((1, PREFILL + 1, D)).astype(np.float32))

    cache = ArraysCache(2)
    gdn(norm1(hs[:, :PREFILL]), cache=cache)          # seed conv + recurrent state
    mx.eval(cache[0], cache[1])
    conv_state, recur_state = cache[0], cache[1]
    state_shape = list(recur_state.shape)

    x = hs[:, PREFILL:]                                # the token the GPU will step
    xn = norm1(x)

    # Re-run GatedDeltaNet.__call__ stage by stage so a GPU mismatch localises to
    # one kernel instead of "the block is wrong". The module's own output is
    # computed too and asserted against this transcription — if they disagree the
    # bundle is void, exactly like the mx.dequantize cross-check in qwen36moe.
    from mlx_lm.models.gated_delta import gated_delta_update
    qkv = gdn.in_proj_qkv(xn)
    z_p = gdn.in_proj_z(xn)
    a_p = gdn.in_proj_a(xn)
    b_p = gdn.in_proj_b(xn)
    conv_in = mx.concatenate([conv_state, qkv], axis=1)
    conv_out = nn.silu(gdn.conv1d(conv_in))
    kd, vd = gdn.key_dim, gdn.value_dim
    q_, k_, v_ = [t.reshape(1, 1, h, d) for t, h, d in zip(
        mx.split(conv_out, [kd, 2 * kd], -1),
        [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
        [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim])]
    inv = gdn.head_k_dim ** -0.5
    qn = (inv ** 2) * mx.fast.rms_norm(q_, None, 1e-6)
    kn = inv * mx.fast.rms_norm(k_, None, 1e-6)
    rec, state_out = gated_delta_update(qn, kn, v_, a_p, b_p, gdn.A_log, gdn.dt_bias,
                                        recur_state, None, use_kernel=True)
    gn = gdn.norm(rec, z_p.reshape(1, 1, gdn.num_v_heads, gdn.head_v_dim))
    y_mine = gdn.out_proj(gn.reshape(1, 1, -1))

    r = gdn(xn, cache=cache)
    mx.eval(xn, r, cache[1], qkv, z_p, a_p, b_p, conv_out, rec, gn, y_mine, state_out)
    rel = float(mx.abs(y_mine - r).max() / (mx.abs(r).max() + 1e-9))
    if rel > 1e-5:
        sys.exit(f"stage-by-stage transcription disagrees with the module (rel {rel:.2e}) "
                 "— refusing to write")

    arrays = {
        "x_f16": _f16(x), "xnorm_ref_f32": _f32(xn), "y_ref_f32": _f32(r),
        "proj_ref_f32": np.concatenate([_f32(qkv), _f32(z_p), _f32(a_p), _f32(b_p)]),
        "conv_ref_f32": _f32(conv_out),
        "recur_ref_f32": _f32(rec),
        "gnorm_ref_f32": _f32(gn),
        "conv_state_f16": _f16(conv_state),
        # TRANSPOSED. mlx allocates the recurrent state as (B, Hv, Dv, Dk) —
        # gated_delta.gated_delta_update — i.e. S[h][dv][dk]; gdn_recur.wgsl
        # indexes S[h][dk][dv] (dk stride GDN_HEAD_V) so each thread owns one
        # dv column. head_k_dim == head_v_dim == 128 here, so feeding mlx's
        # layout straight through is silently wrong rather than a shape error.
        "recur_state_f32": _f32(mx.swapaxes(recur_state, -1, -2)),
        "recur_state_out_f32": _f32(mx.swapaxes(cache[1], -1, -2)),
        "norm1_gamma_f16": _f16(w[PRE + "input_layernorm.weight"]),
        "conv1d_f16": _f16(w[PRE + "linear_attn.conv1d.weight"]),
        "A_log_f32": _f32(w[PRE + "linear_attn.A_log"]),
        "dt_bias_f32": _f32(w[PRE + "linear_attn.dt_bias"]),
        "gnorm_gamma_f16": _f16(w[PRE + "linear_attn.norm.weight"]),
    }
    for name, key in (("qkv", "in_proj_qkv"), ("z", "in_proj_z"),
                      ("a", "in_proj_a"), ("b", "in_proj_b"), ("out", "out_proj")):
        _affine(w, PRE + "linear_attn." + key, arrays, name)

    return {
        "arrays": arrays,
        "meta": {"kernel": "qwen36_gdn",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": "layers.0 (input_layernorm + linear_attn), decode step 3",
                 "reference": "mlx_lm qwen3_5.GatedDeltaNet",
                 "hidden": int(args.hidden_size),
                 "kHeads": int(args.linear_num_key_heads), "vHeads": int(args.linear_num_value_heads),
                 "headK": int(args.linear_key_head_dim), "headV": int(args.linear_value_head_dim),
                 "convK": int(args.linear_conv_kernel_dim),
                 "convDim": int(args.linear_key_head_dim * args.linear_num_key_heads * 2
                                + args.linear_value_head_dim * args.linear_num_value_heads),
                 "group": qz["group_size"], "rmsEps": float(args.rms_norm_eps),
                 "prefill": PREFILL, "state_shape": state_shape, "state_layout": "h,dk,dv (transposed from mlx h,dv,dk)",
                 "note": "r = linear_attn(input_layernorm(x)) at position 3, conv+recurrent "
                         "state seeded by 3 prefill tokens; residual/MoE are NOT included"},
    }


def qwen36attn(seed: int):
    """Layer 3's gated-attention sub-block: input_layernorm -> self_attn -> r.

    Layer 3 because is_linear = (i+1) % full_attention_interval != 0, so layers
    3, 7, ... 39 are the ten full-attention layers.

    q_proj emits 8192 rows for 16 heads x 256: the second 4096 are the SIGMOID
    GATE, not query rows. Same decode-with-seeded-cache shape as qwen36gdn.
    """
    import mlx.core as mx, mlx.nn as nn
    from mlx_lm.models.qwen3_5 import Attention
    from mlx_lm.models.cache import KVCache

    W, cfg, args = _qwen36_args()
    qz = cfg["quantization"]
    attn = Attention(args)
    norm1 = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    nn.quantize(attn, group_size=qz["group_size"], bits=qz["bits"])

    PRE = "language_model.model.layers.3."
    w = mx.load(os.path.join(W, "model-00001-of-00004.safetensors"))
    attn.load_weights([(k[len(PRE + "self_attn."):], v) for k, v in w.items()
                       if k.startswith(PRE + "self_attn.")])
    norm1.load_weights([("weight", w[PRE + "input_layernorm.weight"])])
    mx.eval(attn.parameters(), norm1.parameters())

    # f32 for the same reason as qwen36gdn: a bfloat16 reference is further from
    # f32 truth than our f16 path is, so it would set the tolerance itself.
    attn.set_dtype(mx.float32)
    norm1.set_dtype(mx.float32)
    mx.eval(attn.parameters(), norm1.parameters())
    rng = np.random.default_rng(seed)
    D, PREFILL = args.hidden_size, 3
    hs = mx.array(rng.standard_normal((1, PREFILL + 1, D)).astype(np.float32))

    cache = KVCache()
    attn(norm1(hs[:, :PREFILL]), mask=None, cache=cache)
    mx.eval(cache.keys, cache.values)
    k_seed, v_seed = cache.state[0][:, :, :PREFILL], cache.state[1][:, :, :PREFILL]
    mx.eval(k_seed, v_seed)

    x = hs[:, PREFILL:]
    xn = norm1(x)

    # Stage-by-stage transcription of Qwen3NextAttention.__call__, asserted
    # against the module's own output below.
    from mlx_lm.models.base import scaled_dot_product_attention
    H, KVH, HD = args.num_attention_heads, args.num_key_value_heads, args.head_dim
    qp = attn.q_proj(xn)
    q_, gate = mx.split(qp.reshape(1, 1, H, -1), 2, axis=-1)
    gate = gate.reshape(1, 1, -1)
    k_, v_ = attn.k_proj(xn), attn.v_proj(xn)
    q_ = attn.q_norm(q_).transpose(0, 2, 1, 3)
    kh = attn.k_norm(k_.reshape(1, 1, KVH, -1)).transpose(0, 2, 1, 3)
    vh = v_.reshape(1, 1, KVH, -1).transpose(0, 2, 1, 3)
    q_r = attn.rope(q_, offset=PREFILL)
    k_r = attn.rope(kh, offset=PREFILL)
    kk = mx.concatenate([k_seed, k_r], axis=2)
    vv = mx.concatenate([v_seed, vh], axis=2)
    o = scaled_dot_product_attention(q_r, kk, vv, cache=None, scale=attn.scale, mask=None)
    o = o.transpose(0, 2, 1, 3).reshape(1, 1, -1)
    gated = o * mx.sigmoid(gate)
    y_mine = attn.o_proj(gated)

    r = attn(xn, mask=None, cache=cache)
    mx.eval(xn, r, qp, k_, v_, q_r, k_r, o, gated, y_mine)
    rel = float(mx.abs(y_mine - r).max() / (mx.abs(r).max() + 1e-9))
    if rel > 1e-5:
        sys.exit(f"stage-by-stage transcription disagrees with the module (rel {rel:.2e}) "
                 "— refusing to write")

    arrays = {
        "x_f16": _f16(x), "xnorm_ref_f32": _f32(xn), "y_ref_f32": _f32(r),
        "k_cache_f16": _f16(k_seed), "v_cache_f16": _f16(v_seed),
        # c_attn as the engine lays it out: per-head [Q|gate] (already how
        # q_proj emits it), then K, then V.
        "cattn_ref_f32": np.concatenate([_f32(qp), _f32(k_), _f32(v_)]),
        "q_rope_ref_f32": _f32(q_r.transpose(0, 2, 1, 3)),
        "k_rope_ref_f32": _f32(k_r.transpose(0, 2, 1, 3)),
        "attn_ref_f32": _f32(o), "gated_ref_f32": _f32(gated),
        "norm1_gamma_f16": _f16(w[PRE + "input_layernorm.weight"]),
        "q_norm_f16": _f16(w[PRE + "self_attn.q_norm.weight"]),
        "k_norm_f16": _f16(w[PRE + "self_attn.k_norm.weight"]),
    }
    for name, key in (("q", "q_proj"), ("k", "k_proj"), ("v", "v_proj"), ("o", "o_proj")):
        _affine(w, PRE + "self_attn." + key, arrays, name)

    return {
        "arrays": arrays,
        "meta": {"kernel": "qwen36_attn",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": "layers.3 (input_layernorm + self_attn), decode step 3",
                 "reference": "mlx_lm qwen3_next.Qwen3NextAttention",
                 "hidden": int(args.hidden_size), "heads": int(args.num_attention_heads),
                 "kvHeads": int(args.num_key_value_heads), "headDim": int(args.head_dim),
                 "rotaryDim": int(args.head_dim * args.partial_rotary_factor),
                 "ropeTheta": float(args.rope_theta),
                 "group": qz["group_size"], "rmsEps": float(args.rms_norm_eps),
                 "prefill": PREFILL, "pos": PREFILL,
                 "note": "q_proj rows [0,4096) are queries and [4096,8192) the sigmoid gate; "
                         "residual/MoE are NOT included"},
    }


def qwen36layer(seed: int):
    """A WHOLE decoder layer: norm1 -> linear_attn -> +x -> norm2 -> MoE -> +h.

    The three sub-block bundles each prove one piece against mlx_lm. This proves
    they COMPOSE, which is what engine-core will do: the residual adds, the
    post-attention norm, and — the part worth checking — that the GDN block's
    f16 output is the right dtype and layout for the MoE block's input.

    Carries ONLY reference vectors and the seeded state. The weights are layer
    0's, which qwen36gdn and qwen36moe_block already hold, so duplicating them
    would cost 450 MB to say nothing new; the test reads them from there and
    this bundle defines the x they must all be run on.
    """
    import mlx.core as mx, mlx.nn as nn
    from mlx_lm.models.qwen3_5 import DecoderLayer
    from mlx_lm.models.cache import ArraysCache

    W, cfg, args = _qwen36_args()
    qz = cfg["quantization"]
    layer = DecoderLayer(args, 0)
    nn.quantize(layer, group_size=qz["group_size"], bits=qz["bits"])
    PRE = "language_model.model.layers.0."
    g8 = qz.get(PRE + "mlp.gate")            # router + shared gate ship at 8 bits
    if g8:
        layer.mlp.gate = nn.QuantizedLinear(args.hidden_size, args.num_experts, bias=False,
                                            group_size=g8["group_size"], bits=g8["bits"])
        layer.mlp.shared_expert_gate = nn.QuantizedLinear(args.hidden_size, 1, bias=False,
                                                          group_size=g8["group_size"], bits=g8["bits"])
    w = mx.load(os.path.join(W, "model-00001-of-00004.safetensors"))
    layer.load_weights([(k[len(PRE):], v) for k, v in w.items() if k.startswith(PRE)])
    layer.set_dtype(mx.float32)              # f32 reference — see qwen36gdn
    mx.eval(layer.parameters())

    rng = np.random.default_rng(seed)
    D, PREFILL = args.hidden_size, 3
    hs = mx.array(rng.standard_normal((1, PREFILL + 1, D)).astype(np.float32))

    cache = ArraysCache(2)
    layer(hs[:, :PREFILL], cache=cache)
    mx.eval(cache[0], cache[1])
    conv_state, recur_state = cache[0], cache[1]

    x = hs[:, PREFILL:]
    y = layer(x, cache=cache)                # the module's own answer

    # Same thing stage by stage, from a cache restored to the snapshot, so a GPU
    # mismatch localises instead of just saying "the layer is wrong".
    c2 = ArraysCache(2)
    c2[0], c2[1] = conv_state, recur_state
    r = layer.linear_attn(layer.input_layernorm(x), None, c2)
    h = x + r
    n2 = layer.post_attention_layernorm(h)
    m = layer.mlp(n2)
    y_mine = h + m
    mx.eval(y, r, h, n2, m, y_mine)
    rel = float(mx.abs(y_mine - y).max() / (mx.abs(y).max() + 1e-9))
    if rel > 1e-5:
        sys.exit(f"stage-by-stage transcription disagrees with the module (rel {rel:.2e}) "
                 "— refusing to write")

    return {
        "arrays": {
            "x_f16": _f16(x),
            "r_ref_f32": _f32(r), "h_ref_f32": _f32(h), "norm2_ref_f32": _f32(n2),
            "moe_ref_f32": _f32(m), "y_ref_f32": _f32(y),
            "conv_state_f16": _f16(conv_state),
            # transposed for gdn_recur's S[h][dk][dv] — see qwen36gdn
            "recur_state_f32": _f32(mx.swapaxes(recur_state, -1, -2)),
            # the one weight the other two bundles do not carry
            "norm2_gamma_f16": _f16(w[PRE + "post_attention_layernorm.weight"]),
        },
        "meta": {"kernel": "qwen36_layer",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": "layers.0 (whole DecoderLayer), decode step 3",
                 "reference": "mlx_lm qwen3_5.DecoderLayer",
                 "hidden": int(args.hidden_size), "num_experts": int(args.num_experts),
                 "top_k": int(args.num_experts_per_tok),
                 "moe_intermediate": int(args.moe_intermediate_size),
                 "norm_topk_prob": bool(args.norm_topk_prob),
                 "prefill": PREFILL, "group": qz["group_size"],
                 "weights_from": ["qwen36gdn", "qwen36moe_block"],
                 "note": "y = h + mlp(post_attention_layernorm(h)) where h = x + "
                         "linear_attn(input_layernorm(x)); weights live in the bundles "
                         "named by weights_from — this one carries references plus "
                         "post_attention_layernorm, which neither of those has"},
    }


def qwen36embed(seed: int):
    """Embedding rows for a handful of token ids, dequantized by MLX itself.

    embedding.wgsl is hard-symmetric — `(nibble - 7) * scale` over groups of 32,
    no bias binding. The MLX table is affine over groups of 64. Binding one to
    the other is in bounds and error-free; it just returns wrong token vectors,
    at the very top of the forward pass, where nothing downstream points back.
    So the affine sibling gets its own real-weight check.
    """
    import mlx.core as mx

    W, cfg, args = _qwen36_args()
    key = "language_model.model.embed_tokens"
    w = mx.load(os.path.join(W, "model-00001-of-00004.safetensors"))
    if key + ".weight" not in w:
        # embed_tokens lives in whichever shard the index says; find it.
        import json as _json
        idx = _json.load(open(os.path.join(W, "model.safetensors.index.json")))["weight_map"]
        w = mx.load(os.path.join(W, idx[key + ".weight"]))

    qz = cfg["quantization"]
    G = qz["group_size"]
    # Real ids plus the two ends of the table — an off-by-one in the row stride
    # shows up at the edges first.
    ids = np.array([0, 1, 1000, 100000, args.vocab_size - 1], dtype=np.uint32)
    rows = mx.array(ids.astype(np.int32))
    ref = mx.dequantize(mx.take(w[key + ".weight"], rows, axis=0),
                        mx.take(w[key + ".scales"], rows, axis=0),
                        mx.take(w[key + ".biases"], rows, axis=0),
                        group_size=G, bits=qz["bits"])
    mx.eval(ref)

    return {
        "arrays": {
            "ids_u32": ids,
            "weights_u32": np.array(w[key + ".weight"].astype(mx.uint32)).ravel(),
            "scales_f16": np.array(w[key + ".scales"].astype(mx.float32)).astype(np.float16).ravel(),
            "bias_f16": np.array(w[key + ".biases"].astype(mx.float32)).astype(np.float16).ravel(),
            "y_ref_f32": _f32(ref),
        },
        "meta": {"kernel": "affine_embedding",
                 "model": "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
                 "tensor": key,
                 "reference": "mlx.core.dequantize",
                 "hidden": int(args.hidden_size), "vocab": int(args.vocab_size),
                 "group": G, "n_ids": int(ids.size),
                 "note": "rows for ids [0, 1, 1000, 100000, vocab-1]; affine w = s*q + b, group 64"},
    }


PRODUCERS = {"qwen36moe": qwen36moe, "qwen36moe_block": qwen36moe_block,
             "qwen36gdn": qwen36gdn, "qwen36attn": qwen36attn,
             "qwen36layer": qwen36layer, "qwen36embed": qwen36embed}

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
