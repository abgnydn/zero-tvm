"""MAKE-DSV2-LAYER-REF — a whole DeepSeek-V2 layer, stage by stage.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/make-dsv2-layer-ref.py \
        --bundle ~/dev/zero-tvm/.weights-local/kernel-refs/dsv2layer0

    # the MoE half of the same stack (layer 1..26), from a 'layers\\.1\\.' bundle:
    python scripts/make-dsv2-layer-ref.py \
        --bundle .weights-local/kernel-refs/dsv2moe --layer model.layers.1

make-mla-ref.py covers attention in isolation. This covers the layer around it:
norms, residuals, RoPE at real positions, and the FFN — which for layer 0 of
DeepSeek-V2 is an ORDINARY mlp at intermediate_size 10944 while every other
layer is MoE at 1408 (first_k_dense_replace=1). So the two bundles between them
exercise all of the remaining blockers: MLA inside a real layer, and BOTH halves
of a mixed stack.

Which FFN runs is read off the bundle — `mlp.gate` exists only in a MoE layer —
rather than from a flag, because a flag is a second place for the two to
disagree. Everything before the FFN is shared, including the interleaved-RoPE
permutation proof below, which is the part of this script most expensive to get
wrong twice.

NUMPY ONLY, DELIBERATELY. The bandwidth to pull a 329 MB MoE layer is not on the
machine with Apple silicon, so this must run on a Colab box. dequant() below is
the same arithmetic as make-mla-ref.py's dequant_numpy, which `--backend both`
proved bit-identical to mx.dequantize.

TWO THINGS THIS PINS THAT NOTHING ELSE DOES.

1. RoPE is INTERLEAVED. modeling_deepseek.py's apply_rotary_pos_emb is labelled
   "Copied from transformers…llama" but is not: it adds

       q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

   before the usual rotate_half. That de-interleaves, which means q_pe and k_pe
   are STORED interleaved ([a0,b0,a1,b1,…]) where our rope.wgsl pairs
   (j, j+half). Applying ours directly rotates the wrong pairs — no error,
   fluent output, a quietly broken model. Reproduced here so the engine has
   something to be wrong against.

2. yarn stretches the frequencies AND scales the logits by mscale^2. The second
   part is not RoPE and is easy to leave out; at factor 40 it is 1.59x, which
   is far too large to be lost in noise but far too small to look broken.

3. (MoE layers only) DeepSeek's MoE block is NOT the Qwen3.6 block this engine
   already runs, and none of the four differences raise an error — see the long
   comment on the MoE branch. Briefly: the router is UNQUANTIZED f16, the shared
   expert is twice a routed expert's width and so cannot be stacked as slot K,
   it carries no gate, and norm_topk_prob is false so the routed weights sum to
   well under 1.
"""

import argparse
import json
import math
import pathlib

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--layer", default="model.layers.0")
# 20, not 6. Position 0 is RoPE-identity (cos 1, sin 0), so a short cache
# barely exercises the rotation, and a position-vs-slot bug in the KV append
# shows up as error GROWING with t — which six positions cannot distinguish
# from noise. Costs nothing: the script generates x from a seed and re-reads
# tensors already on disk, so a longer bundle needs no re-pull.
p.add_argument("--tokens", type=int, default=20)
p.add_argument("--seed", type=int, default=11)
# DeepSeek-V2-Lite's rope_scaling, from config.json.
p.add_argument("--rope-theta", type=float, default=10000.0)
p.add_argument("--factor", type=float, default=40.0)
p.add_argument("--beta-fast", type=float, default=32.0)
p.add_argument("--beta-slow", type=float, default=1.0)
p.add_argument("--original-max", type=int, default=4096)
p.add_argument("--mscale-all-dim", type=float, default=0.707)
# MoE knobs, from config.json. n_routed_experts and moe_intermediate_size are
# NOT flags — they are solved from the tensor shapes, the same way the head
# split is, so a bundle from a different DeepSeek cannot be described wrongly.
p.add_argument("--top-k", type=int, default=6)              # num_experts_per_tok
p.add_argument("--routed-scaling", type=float, default=1.0)  # routed_scaling_factor
p.add_argument("--norm-topk-prob", action="store_true",
               help="DeepSeek-V2-Lite says FALSE; the flag exists so that is a stated choice")
args = p.parse_args()

bundle = pathlib.Path(args.bundle)
meta = json.loads((bundle / "meta.json").read_text())
TEN = meta["tensors"]
DT = {"F16": np.float16, "U32": np.uint32, "F32": np.float32}


def raw(name):
    # memmap rather than read_bytes: a routed-expert stack is 92 MB and this
    # script uses SIX of its 64 experts. Reading the file once per expert would
    # be half a gigabyte of I/O to look at 26 MB of it, and holding all three
    # stacks resident is 280 MB the Colab box has no reason to spend.
    rec = TEN[f"{args.layer}.{name}"]
    a = np.memmap(bundle / rec["file"], dtype=DT[rec["dtype"]], mode="r")
    return a.reshape(rec["shape"])


def dequant_from(q, scales, biases, group=64, bits=4):
    """MLX affine, in numpy — bit-identical to mx.dequantize (verified by
    make-mla-ref.py --backend both), so this script needs no Apple silicon.

    Takes the three arrays rather than a name because the routed experts arrive
    as slices of a [E, N, ...] stack, not as tensors of their own.
    """
    q = np.asarray(q, dtype=np.uint32)
    per_word, rows = 32 // bits, q.shape[0]
    k = q.shape[1] * per_word
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    vals = ((q[:, :, None] >> shifts[None, None, :]) & ((1 << bits) - 1)).reshape(rows, k)
    s = np.asarray(scales, dtype=np.float32).repeat(group, axis=1)[:, :k]
    b = np.asarray(biases, dtype=np.float32).repeat(group, axis=1)[:, :k]
    return (vals.astype(np.float32) * s + b).astype(np.float16).astype(np.float32)


def dequant(name, group=64, bits=4):
    return dequant_from(raw(f"{name}.weight"), raw(f"{name}.scales"), raw(f"{name}.biases"),
                        group, bits)


def dequant_expert(name, e, group=64, bits=4):
    """Expert `e` out of a stacked [E, N, ...] tensor.

    The stack is C-contiguous so this is a plain leading-axis slice — but
    weight is [E, N, K/8] while scales and biases are [E, N, K/group], and
    slicing one of them on the wrong axis (or off by an expert) still yields a
    well-formed matrix that computes a DIFFERENT expert. Nothing downstream can
    tell; the layer just answers wrongly.
    """
    return dequant_from(raw(f"{name}.weight")[e], raw(f"{name}.scales")[e],
                        raw(f"{name}.biases")[e], group, bits)


def rms(x, gamma, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * np.asarray(gamma, np.float32)


def yarn_inv_freq(dim):
    """DeepSeek's table. Kept here rather than imported so this script and
    ropeInvFreqTable() are independent implementations of the same thing."""
    i = np.arange(0, dim, 2, dtype=np.float32)
    extra = 1.0 / (args.rope_theta ** (i / dim))
    inter = extra / args.factor

    def corr(rot):
        return dim * math.log(args.original_max / (rot * 2 * math.pi)) / (2 * math.log(args.rope_theta))

    low, high = max(math.floor(corr(args.beta_fast)), 0), min(math.ceil(corr(args.beta_slow)), dim - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    mask = 1.0 - ramp
    return inter * (1 - mask) + extra * mask


def rope_interleaved(x, pos, inv_freq):
    """x is [..., dim] stored INTERLEAVED. De-interleave, then rotate_half —
    exactly the two steps modeling_deepseek.py performs."""
    dim = x.shape[-1]
    x = x.reshape(*x.shape[:-1], dim // 2, 2).swapaxes(-1, -2).reshape(*x.shape[:-1], dim)
    ang = pos * np.concatenate([inv_freq, inv_freq])
    cos, sin = np.cos(ang, dtype=np.float32), np.sin(ang, dtype=np.float32)
    half = dim // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


# ── weights ────────────────────────────────────────────────────────────────
Wq, Wkva, Wkvb, Wo = (dequant(f"self_attn.{n}") for n in
                      ("q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"))
g_in, g_post, g_kva = (np.asarray(raw(n), np.float32) for n in
                       ("input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn.kv_a_layernorm.weight"))
# Which FFN this layer has, read off the bundle: mlp.gate is the router and
# exists only in a MoE layer. A --moe flag would be a second place for the
# script and the weights to disagree about what they are.
MOE = f"{args.layer}.mlp.gate.weight" in TEN

D, KV_LORA = Wq.shape[1], g_kva.shape[0]
ROPE = Wkva.shape[0] - KV_LORA
HEADS = None
for h in range(1, 129):
    if Wq.shape[0] % h or Wkvb.shape[0] % h:
        continue
    nope = Wq.shape[0] // h - ROPE
    if nope > 0 and Wkvb.shape[0] // h == 2 * nope:
        HEADS, NOPE, VDIM = h, nope, nope
        break
print(f"d={D} heads={HEADS} nope={NOPE} rope={ROPE} kv_lora={KV_LORA} "
      f"ffn={'moe' if MOE else 'dense'}")

rng = np.random.default_rng(args.seed)
T = args.tokens
x = (rng.standard_normal((T, D)) * 0.5).astype(np.float32)

# ── attention ──────────────────────────────────────────────────────────────
h1 = rms(x, g_in)
q = (h1 @ Wq.T).reshape(T, HEADS, NOPE + ROPE)
q_nope, q_pe = q[..., :NOPE], q[..., NOPE:]
kva = h1 @ Wkva.T
c = rms(kva[:, :KV_LORA], g_kva)
k_pe = kva[:, KV_LORA:]

# ── the de-interleave, moved off the runtime path ──────────────────────────
# DeepSeek de-interleaves q_pe/k_pe before rotating. That is a permutation of a
# PROJECTION'S OUTPUT, and a permutation of the output commutes with permuting
# the matrix's ROWS — so doing it once to the weights leaves the runtime with
# ordinary half-split RoPE and no new kernel. Asserted below rather than
# assumed, because "obviously equivalent" is how the interleave got missed in
# the first place.
DEINT = np.arange(ROPE).reshape(ROPE // 2, 2).T.reshape(ROPE)   # [0,2,4,…,1,3,5,…]

Wq_perm = Wq.reshape(HEADS, NOPE + ROPE, D).copy()
Wq_perm[:, NOPE:, :] = Wq_perm[:, NOPE:, :][:, DEINT, :]
Wq_perm = Wq_perm.reshape(HEADS * (NOPE + ROPE), D)
Wkva_perm = Wkva.copy()
Wkva_perm[KV_LORA:, :] = Wkva_perm[KV_LORA:, :][DEINT, :]

inv = yarn_inv_freq(ROPE)
pos = np.arange(T, dtype=np.float32)[:, None]
q_pe_r = np.stack([rope_interleaved(q_pe[t], pos[t], inv) for t in range(T)])
k_pe_r = np.stack([rope_interleaved(k_pe[t], pos[t], inv) for t in range(T)])

# Same rotation, half-split, on the permuted projections' output.
def rope_halfsplit(x, pos, inv_freq):
    dim = x.shape[-1]
    ang = pos * np.concatenate([inv_freq, inv_freq])
    cos, sin = np.cos(ang, dtype=np.float32), np.sin(ang, dtype=np.float32)
    half = dim // 2
    rot = np.concatenate([-x[..., half:], x[..., :half]], axis=-1)
    return x * cos + rot * sin


q_perm = (h1 @ Wq_perm.T).reshape(T, HEADS, NOPE + ROPE)
kva_perm = h1 @ Wkva_perm.T
q_pe_alt = np.stack([rope_halfsplit(q_perm[t, :, NOPE:], pos[t], inv) for t in range(T)])
k_pe_alt = np.stack([rope_halfsplit(kva_perm[t, KV_LORA:], pos[t], inv) for t in range(T)])
for label, a, b in (("q_pe", q_pe_alt, q_pe_r), ("k_pe", k_pe_alt, k_pe_r)):
    err = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9)
    if err > 1e-6:
        raise SystemExit(f"permuting {label} rows does NOT reproduce the interleaved rotation ({err:.2e})")
print(f"row permutation reproduces DeepSeek's interleaved RoPE exactly "
      f"(q_pe {np.max(np.abs(q_pe_alt - q_pe_r)):.2e}, k_pe {np.max(np.abs(k_pe_alt - k_pe_r)):.2e})")

mscale = 0.1 * args.mscale_all_dim * math.log(args.factor) + 1.0
scale = (NOPE + ROPE) ** -0.5 * mscale * mscale
qi = T - 1

Wkvb_h = Wkvb.reshape(HEADS, NOPE + VDIM, KV_LORA)
Wk, Wv = Wkvb_h[:, :NOPE, :], Wkvb_h[:, NOPE:, :]
q_lat = np.einsum("hj,hjc->hc", q_nope[qi], Wk)
scores = (q_lat @ c.T + q_pe_r[qi] @ k_pe_r.T) * scale
prob = np.exp(scores - scores.max(-1, keepdims=True))
prob /= prob.sum(-1, keepdims=True)
o_lat = prob @ c
o_heads = np.einsum("hc,hvc->hv", o_lat, Wv)
attn_out = o_heads.reshape(-1) @ Wo.T

# Cross-check against the naive form, the same guard make-mla-ref.py uses.
kv = (c @ Wkvb.T).reshape(T, HEADS, NOPE + VDIM)
k_nope, v = kv[..., :NOPE], kv[..., NOPE:]
s_naive = ((q_nope[qi][None] * k_nope).sum(-1).T + q_pe_r[qi] @ k_pe_r.T) * scale
if np.max(np.abs(s_naive - scores)) / (np.max(np.abs(s_naive)) + 1e-9) > 1e-4:
    raise SystemExit("latent and naive scores disagree — the latent algebra is wrong")

resid1 = x[qi] + attn_out


def dump(name, a):
    np.ascontiguousarray(np.asarray(a, np.float32)).tofile(bundle / f"{name}.bin")


def dump_raw(name, a):
    """Verbatim, in the array's own dtype — for the u32/f16 checkpoint bytes a
    kernel has to unpack. dump() would silently widen them to f32 and the test
    would then be validating a float matrix that is not in the model."""
    np.ascontiguousarray(a).tofile(bundle / f"{name}.bin")


def silu_mul(gate, up):
    return gate / (1.0 + np.exp(-gate)) * up


h2 = rms(resid1, g_post)
extra, ffn_meta = [], {}

if not MOE:
    # ── dense FFN (layer 0 only: first_k_dense_replace = 1) ────────────────
    Wg, Wu, Wd = (dequant(f"mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj"))
    FFN = Wg.shape[0]
    ffn_out = silu_mul(h2 @ Wg.T, h2 @ Wu.T) @ Wd.T
    print(f"dense ffn={FFN}")
    ffn_meta = {"dense_ffn": FFN}
else:
    # ── MoE FFN (layers 1..26) ─────────────────────────────────────────────
    # This is NOT the Qwen3.6 MoE block the engine already runs, and not one of
    # the four differences raises an error — each produces a fluent wrong model:
    #
    #  1. THE ROUTER IS UNQUANTIZED. mlp.gate.weight is a plain f16 [E, d]; the
    #     bundle has no mlp.gate.scales at all. moe_router_logits (8-bit) reads
    #     D/4 words per row and its _q4 twin D/8, so pointing either at this
    #     tensor mixes neighbouring experts' bytes into every logit. DeepSeek
    #     needs an f16 matvec the block does not currently have.
    #  2. THE SHARED EXPERT IS NOT AN EXPERT-SHAPED THING. Its width is
    #     n_shared_experts * moe_intermediate (2816 here, 2x a routed expert's
    #     1408), so it cannot be appended to the [E, N, ...] stacks as index E
    #     the way Qwen's is, and moe_router_topk's `hasShared` slot cannot carry
    #     it. It is a separate dense chain, summed after the combine —
    #     DeepSeek's MoE block is 2 matmul chains wide, not 1.
    #  3. IT HAS NO GATE. There is no shared_expert_gate row, no sigmoid: the
    #     shared branch enters at weight exactly 1.
    #  4. norm_topk_prob IS FALSE. The K routed weights are raw softmax
    #     probabilities and sum to well under 1 (printed below). Renormalising
    #     them — what a Mixtral-shaped implementation does by default, and what
    #     moe_router_topk does when normTopk=1 — rescales the entire routed
    #     branch by 1/mass. The run prints that factor so the error has a size.
    #
    # topk_method is "greedy" with n_group = topk_group = 1, which is a plain
    # top-k over all E scores; the group-limited path never runs for this model.
    Wrouter = np.asarray(raw("mlp.gate.weight"), np.float32)   # [E, d], f16 on disk
    E = Wrouter.shape[0]
    K = args.top_k

    router_logits = h2 @ Wrouter.T
    probs = np.exp(router_logits - router_logits.max())
    probs /= probs.sum()

    # DESCENDING score, ties to the lower expert index — the order
    # moe_router_topk.wgsl emits, so a GPU test can compare slot for slot. The
    # combine is a sum, so the block's OUTPUT does not depend on this; the
    # per-slot dumps below do.
    ids = np.argsort(-probs, kind="stable")[:K].astype(np.uint32)
    # modeling_deepseek.py's MoEGate is an if/else: renormalise, ELSE multiply by
    # routed_scaling_factor. Scaling then normalising is the same thing for both
    # branches (the factor cancels) and does not need the branch. mlx_lm's
    # deepseek_v2 drops the norm case entirely and always scales — it agrees with
    # HF only because this config says norm_topk_prob false, so anyone flipping
    # the flag should check HF, not mlx, for what the model means.
    topk_w = probs[ids] * args.routed_scaling
    if args.norm_topk_prob:
        topk_w = topk_w / topk_w.sum()

    # A top-k that is not the top k is the one routing bug with no numerical
    # signature: every downstream stage stays self-consistent.
    rest = np.setdiff1d(np.arange(E), ids)
    if probs[ids].min() < probs[rest].max():
        raise SystemExit("selected experts are not the top-k of the router distribution")
    if abs(probs.sum() - 1.0) > 1e-5:
        raise SystemExit(f"router softmax does not sum to 1 ({probs.sum():.6f})")

    # The experts that will ship. Sliced from the stacks as VERBATIM bytes so
    # the numbers below are computed from exactly what a test will read; six of
    # sixty-four experts is 29 MB where the stacks are 311 MB, and what has to
    # come back over the slow link is the whole point of the exercise.
    PROJS = ("gate_proj", "up_proj", "down_proj")
    SEL = {proj: tuple(np.ascontiguousarray(raw(f"mlp.switch_mlp.{proj}.{part}")[ids])
                       for part in ("weight", "scales", "biases"))
           for proj in PROJS}

    MOE_INT = SEL["gate_proj"][0].shape[1]
    expert_h = np.empty((K, MOE_INT), np.float32)   # silu(gate)*up, per slot
    expert_y = np.empty((K, D), np.float32)         # down(...), per slot, UNWEIGHTED
    for j, e in enumerate(ids):
        W = {}
        for proj in PROJS:
            w, s, b = SEL[proj]
            W[proj] = dequant_from(w[j], s[j], b[j])
            # The slice is the only thing that ships. If row j of it is not
            # expert e, everything downstream is self-consistent and wrong, so
            # this is checked against an independent index into the full stack
            # rather than assumed from the slicing syntax.
            if not np.array_equal(W[proj], dequant_expert(f"mlp.switch_mlp.{proj}", int(e))):
                raise SystemExit(f"slice row {j} of {proj} is not expert {e}")
        expert_h[j] = silu_mul(h2 @ W["gate_proj"].T, h2 @ W["up_proj"].T)
        expert_y[j] = expert_h[j] @ W["down_proj"].T
    expert_out = (topk_w[:, None] * expert_y).sum(0)

    Wgs, Wus, Wds = (dequant(f"mlp.shared_experts.{n}")
                     for n in ("gate_proj", "up_proj", "down_proj"))
    shared_h = silu_mul(h2 @ Wgs.T, h2 @ Wus.T)
    shared_out = shared_h @ Wds.T
    ffn_out = expert_out + shared_out

    mass = float(probs[ids].sum())
    print(f"moe experts={E} top_k={K} moe_int={MOE_INT} shared_int={Wgs.shape[0]}")
    print(f"  routed to {list(map(int, ids))}")
    print(f"  top-{K} probability mass {mass:.4f} — weights sum to {float(topk_w.sum()):.4f} "
          f"(norm_topk_prob={args.norm_topk_prob}); renormalising would scale the "
          f"routed branch by {1.0 / mass:.3f}x")
    print(f"  routed {np.abs(expert_out).max():.4f} / shared {np.abs(shared_out).max():.4f} "
          "(max abs) — the shared branch is ungated and enters at weight 1")

    for proj in PROJS:
        w, s, b = SEL[proj]
        dump_raw(f"exp_{proj}_w_u32", w)
        dump_raw(f"exp_{proj}_s_f16", s)
        dump_raw(f"exp_{proj}_b_f16", b)
    dump_raw("ref_topk_ids", ids)                 # u32, NOT f32 like the ref_* floats
    extra = [("ref_router_logits", router_logits), ("ref_router_probs", probs),
             ("ref_topk_weights", topk_w), ("ref_expert_h", expert_h),
             ("ref_expert_y", expert_y), ("ref_expert_out", expert_out),
             ("ref_shared_h", shared_h), ("ref_shared_out", shared_out)]
    ffn_meta = {
        "num_experts": int(E), "top_k": int(K), "moe_intermediate": int(MOE_INT),
        "shared_intermediate": int(Wgs.shape[0]),
        "n_shared_experts": int(Wgs.shape[0] // MOE_INT),
        "norm_topk_prob": bool(args.norm_topk_prob),
        "routed_scaling_factor": float(args.routed_scaling),
        "scoring_func": "softmax", "topk_method": "greedy",
        "router_quantized": False, "router_dtype": TEN[f"{args.layer}.mlp.gate.weight"]["dtype"],
        "group": 64, "bits": 4,
        "selected_experts": [int(i) for i in ids],
        "topk_probability_mass": mass,
        "topk_ids_dtype": "u32",
        "expert_stack": "exp_*_{w_u32,s_f16,b_f16} hold the SELECTED experts only, "
                        "row j = selected_experts[j]; the shared expert is NOT stacked",
        "note": "ffn_out = sum_j topk_weights[j] * down_j(silu(gate_j(h2))*up_j(h2)) "
                "+ down_s(silu(gate_s(h2))*up_s(h2)); the shared term is ungated "
                "(no sigmoid, no router row) and the routed weights are NOT renormalised",
    }

out = resid1 + ffn_out

# Pre-RoPE outputs of the PERMUTED projections. The permutation is applied to
# quantized rows at load, so a test needs to see that the int4 matmul over the
# shuffled rows lands where the reference says before any rotation happens.
dumps = [("wk_t", np.transpose(Wk, (0, 2, 1))), ("wv", Wv),
         ("ref_qperm", q_perm[qi]), ("ref_kvaperm", kva_perm[qi]),
         ("ref_qnope", q_nope[qi]),
         ("ref_x", x), ("ref_h1", h1), ("ref_c", c), ("ref_kpe", k_pe_r),
         ("ref_qlat", q_lat), ("ref_qpe", q_pe_r[qi]), ("ref_scores", scores),
         ("ref_olat", o_lat), ("ref_oheads", o_heads), ("ref_attn_out", attn_out),
         ("ref_resid1", resid1), ("ref_h2", h2), *extra,
         ("ref_ffn_out", ffn_out), ("ref_out", out)]
for n, a in dumps:
    dump(n, a)

meta.update({
    # A DISTINCT kernel name for the MoE layer. real-weights.mjs dispatches on
    # this key and its dsv2_layer handler knows nothing about a router, so
    # reusing the name would run the wrong check; an unknown one SKIPs loudly,
    # which is the correct state until a MoE handler exists.
    "kernel": "dsv2_moe_layer" if MOE else "dsv2_layer", "model": meta["repo"],
    "tensor": f"{args.layer} (MLA + {'MoE' if MOE else 'dense'} FFN)", "layer": args.layer,
    "reference": "numpy, MLX-affine dequant verified bit-identical to mlx",
    "d": D, "heads": HEADS, "nope": NOPE, "rope": ROPE, "v": VDIM,
    "kv_lora": KV_LORA, **ffn_meta, "tokens": T, "query_at": qi, "seed": args.seed,
    "softmax_scale": float(scale), "yarn_mscale_sq": float(mscale * mscale),
    # NOT "rope" — that key already holds the rope head dim a few lines up, and
    # a string would silently replace the number the test reads.
    "rope_convention": "interleaved (deepseek), yarn frequencies",
    "cache_values_per_token": KV_LORA + ROPE,
    "mha_equivalent_per_token": HEADS * (NOPE + VDIM),
})
(bundle / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"scale {scale:.6f} (includes yarn mscale^2 {mscale*mscale:.4f})")
print(f"wrote {bundle}/meta.json + {len(dumps)} reference .bin"
      + (" + 9 expert-slice .bin + ref_topk_ids.bin (u32)" if MOE else ""))
