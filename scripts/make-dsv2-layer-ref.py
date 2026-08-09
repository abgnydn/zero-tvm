"""MAKE-DSV2-LAYER-REF — a whole DeepSeek-V2 layer, stage by stage.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/make-dsv2-layer-ref.py \
        --bundle ~/dev/zero-tvm/.weights-local/kernel-refs/dsv2layer0

make-mla-ref.py covers attention in isolation. This covers the layer around it:
norms, residuals, RoPE at real positions, and the dense FFN — which for layer 0
of DeepSeek-V2 is an ORDINARY mlp at intermediate_size 10944 while every other
layer is MoE at 1408. So this one bundle exercises both of the remaining
blockers: MLA inside a real layer, and the dense half of a mixed stack.

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
"""

import argparse
import json
import math
import pathlib

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--layer", default="model.layers.0")
p.add_argument("--tokens", type=int, default=6)
p.add_argument("--seed", type=int, default=11)
# DeepSeek-V2-Lite's rope_scaling, from config.json.
p.add_argument("--rope-theta", type=float, default=10000.0)
p.add_argument("--factor", type=float, default=40.0)
p.add_argument("--beta-fast", type=float, default=32.0)
p.add_argument("--beta-slow", type=float, default=1.0)
p.add_argument("--original-max", type=int, default=4096)
p.add_argument("--mscale-all-dim", type=float, default=0.707)
args = p.parse_args()

bundle = pathlib.Path(args.bundle)
meta = json.loads((bundle / "meta.json").read_text())
TEN = meta["tensors"]
DT = {"F16": np.float16, "U32": np.uint32, "F32": np.float32}


def raw(name):
    rec = TEN[f"{args.layer}.{name}"]
    a = np.frombuffer((bundle / rec["file"]).read_bytes(), dtype=DT[rec["dtype"]])
    return a.reshape(rec["shape"])


def dequant(name, group=64, bits=4):
    """MLX affine, in numpy — bit-identical to mx.dequantize (verified by
    make-mla-ref.py --backend both), so this script needs no Apple silicon."""
    q = np.asarray(raw(f"{name}.weight"), dtype=np.uint32)
    per_word, rows = 32 // bits, q.shape[0]
    k = q.shape[1] * per_word
    shifts = np.arange(per_word, dtype=np.uint32) * bits
    vals = ((q[:, :, None] >> shifts[None, None, :]) & ((1 << bits) - 1)).reshape(rows, k)
    s = np.asarray(raw(f"{name}.scales"), dtype=np.float32).repeat(group, axis=1)[:, :k]
    b = np.asarray(raw(f"{name}.biases"), dtype=np.float32).repeat(group, axis=1)[:, :k]
    return (vals.astype(np.float32) * s + b).astype(np.float16).astype(np.float32)


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
Wg, Wu, Wd = (dequant(f"mlp.{n}") for n in ("gate_proj", "up_proj", "down_proj"))
g_in, g_post, g_kva = (np.asarray(raw(n), np.float32) for n in
                       ("input_layernorm.weight", "post_attention_layernorm.weight",
                        "self_attn.kv_a_layernorm.weight"))

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
FFN = Wg.shape[0]
print(f"d={D} heads={HEADS} nope={NOPE} rope={ROPE} kv_lora={KV_LORA} dense_ffn={FFN}")

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

# ── dense FFN (this is the DENSE layer of a MoE model) ─────────────────────
h2 = rms(resid1, g_post)
gate = h2 @ Wg.T
up = h2 @ Wu.T
act = gate / (1.0 + np.exp(-gate)) * up          # SiLU * up
ffn_out = act @ Wd.T
out = resid1 + ffn_out

def dump(name, a):
    np.ascontiguousarray(np.asarray(a, np.float32)).tofile(bundle / f"{name}.bin")

# Pre-RoPE outputs of the PERMUTED projections. The permutation is applied to
# quantized rows at load, so a test needs to see that the int4 matmul over the
# shuffled rows lands where the reference says before any rotation happens.
for n, a in [("wk_t", np.transpose(Wk, (0, 2, 1))), ("wv", Wv),
             ("ref_qperm", q_perm[qi]), ("ref_kvaperm", kva_perm[qi]),
             ("ref_qnope", q_nope[qi]),
             ("ref_x", x), ("ref_h1", h1), ("ref_c", c), ("ref_kpe", k_pe_r),
             ("ref_qlat", q_lat), ("ref_qpe", q_pe_r[qi]), ("ref_scores", scores),
             ("ref_olat", o_lat), ("ref_oheads", o_heads), ("ref_attn_out", attn_out),
             ("ref_resid1", resid1), ("ref_h2", h2), ("ref_ffn_out", ffn_out), ("ref_out", out)]:
    dump(n, a)

meta.update({
    "kernel": "dsv2_layer", "model": meta["repo"],
    "tensor": f"{args.layer} (MLA + dense FFN)", "layer": args.layer,
    "reference": "numpy, MLX-affine dequant verified bit-identical to mlx",
    "d": D, "heads": HEADS, "nope": NOPE, "rope": ROPE, "v": VDIM,
    "kv_lora": KV_LORA, "dense_ffn": FFN, "tokens": T, "query_at": qi,
    "softmax_scale": float(scale), "yarn_mscale_sq": float(mscale * mscale),
    # NOT "rope" — that key already holds the rope head dim a few lines up, and
    # a string would silently replace the number the test reads.
    "rope_convention": "interleaved (deepseek), yarn frequencies",
    "cache_values_per_token": KV_LORA + ROPE,
    "mha_equivalent_per_token": HEADS * (NOPE + VDIM),
})
(bundle / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"scale {scale:.6f} (includes yarn mscale^2 {mscale*mscale:.4f})")
print(f"wrote {bundle}/meta.json + 14 ref_*.bin")
