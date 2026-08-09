"""MAKE-MLA-REF — a real-weights reference for multi-head latent attention.

Consumes a bundle from scripts/pull-tensors.mjs (one layer, ~8 MB of Range
requests) and produces the numbers a WGSL kernel has to reproduce.

Run from an env with mlx:

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/make-mla-ref.py \
        --bundle ~/dev/zero-tvm/.weights-local/kernel-refs/dsv2mla

WHY TWO IMPLEMENTATIONS. MLA can be evaluated two ways, and the whole reason to
want it is that they are equal:

  NAIVE   up-project the latent into per-head K_nope and V, then ordinary
          attention. This is what modeling_deepseek.py does, and it caches
          nothing smaller than MHA unless you throw the up-projection away.
  LATENT  push q_nope THROUGH kv_b_proj's K half into the 512-wide latent
          space, score directly against the cached latent, and bring the
          weighted sum back out through kv_b_proj's V half. The cache is then
          one 512 latent plus one shared 64 RoPE key per token — 576 values
          where MHA stores 4096.

The second is what we would implement. Checking it against the first, here in
Python where both are cheap, separates "we misunderstood MLA" from "the shader
is wrong" — two failures that look identical from a WGSL test.

RoPE IS OFF in this reference (--rope none). DeepSeek-V2 uses yarn, which is a
separate refused feature, and rope.wgsl is already covered by its own kernel
test; leaving it out keeps this bundle about the part that is actually new. The
q_pe/k_pe path still runs, unrotated, so the shared-key contribution is
exercised — what is deferred is only which angle it is rotated by.
"""

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--bundle", required=True)
p.add_argument("--layer", default="model.layers.1")
p.add_argument("--tokens", type=int, default=6, help="cache length to score against")
p.add_argument("--rope", default="none", choices=["none"])
p.add_argument("--seed", type=int, default=7)
p.add_argument("--backend", default="mlx", choices=["mlx", "numpy", "both"],
               help="both = compute with mlx and assert the portable numpy path agrees")
args = p.parse_args()
BACKEND = args.backend

bundle = pathlib.Path(args.bundle)
meta = json.loads((bundle / "meta.json").read_text())
T = meta["tensors"]

DT = {"F16": np.float16, "BF16": None, "U32": np.uint32, "F32": np.float32}


def load(name):
    """A tensor exactly as it sits in the shard — shape and dtype from the
    safetensors header, no reinterpretation."""
    rec = T[f"{args.layer}.{name}"]
    dt = DT[rec["dtype"]]
    if dt is None:
        raise SystemExit(f"{name}: dtype {rec['dtype']} not handled here")
    a = np.frombuffer((bundle / rec["file"]).read_bytes(), dtype=dt)
    return mx.array(np.ascontiguousarray(a.reshape(rec["shape"])))


def dequant_numpy(q, scales, biases, group=64, bits=4):
    """MLX affine unpacking, without mlx.

    Colab has no Apple silicon, so a reference that can only be produced on this
    Mac cannot be produced where the bandwidth is. This is the same scheme the
    WGSL kernels read: `bits` per value packed little-end-first into u32 words,
    and one (scale, bias) per `group` values along the INPUT axis, with
    w = scale*q + bias and q unsigned (no -7 offset — that is MLC's symmetric
    layout, not this one).

    Checked against mx.dequantize whenever mlx is present; see `--backend both`.
    """
    q = np.asarray(q, dtype=np.uint32)
    per_word = 32 // bits
    rows, words = q.shape
    k = words * per_word
    shifts = (np.arange(per_word, dtype=np.uint32) * bits)
    vals = ((q[:, :, None] >> shifts[None, None, :]) & ((1 << bits) - 1)).reshape(rows, k)
    s = np.asarray(scales, dtype=np.float32).repeat(group, axis=1)[:, :k]
    b = np.asarray(biases, dtype=np.float32).repeat(group, axis=1)[:, :k]
    # Narrowed to f16 because mx.dequantize does: its scales and biases are f16
    # and so is its output. Returning f32 was closer to the true value and
    # therefore WRONG as a stand-in — it disagreed with mlx by 6.1e-5, f16's
    # ULP at that magnitude. A portable path that is quietly MORE precise than
    # the one it replaces is not portable, it is a second reference.
    return (vals.astype(np.float32) * s + b).astype(np.float16)


def dequant(name, group=64, bits=4):
    """MLX affine: w = scale*q + bias, group 64 along the input axis."""
    q, sc, bi = load(f"{name}.weight"), load(f"{name}.scales"), load(f"{name}.biases")
    if BACKEND == "numpy":
        return mx.array(dequant_numpy(np.array(q), np.array(sc), np.array(bi), group, bits))
    out = mx.dequantize(q, sc, bi, group_size=group, bits=bits)
    if BACKEND == "both":
        ours = dequant_numpy(np.array(q), np.array(sc), np.array(bi), group, bits)
        err = float(np.max(np.abs(ours.astype(np.float32) - np.array(out, dtype=np.float32))))
        if err > 0:
            raise SystemExit(f"{name}: numpy dequant differs from mlx by {err:.3e} — "
                             "the portable path would produce a different reference")
        print(f"  {name}: numpy dequant is bit-identical to mlx")
    return out


def rms(x, gamma, eps=1e-6):
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)) * gamma


# ── the layer, dequantized ──────────────────────────────────────────────────
Wq = dequant("self_attn.q_proj")                    # [heads*(nope+rope), d]
Wkva = dequant("self_attn.kv_a_proj_with_mqa")      # [kv_lora + rope, d]
Wkvb = dequant("self_attn.kv_b_proj")               # [heads*(nope+v), kv_lora]
Wo = dequant("self_attn.o_proj")                    # [d, heads*v]
g_in = load("input_layernorm.weight")
g_kva = load("self_attn.kv_a_layernorm.weight")

D = Wq.shape[1]
KV_LORA = g_kva.shape[0]
ROPE = Wkva.shape[0] - KV_LORA
HEADS = Wo.shape[1] // ((Wkvb.shape[0] // (Wq.shape[0] // 1)) or 1) if False else None
# Solve the head split from shapes rather than trusting config.json: q rows are
# heads*(nope+rope) and kv_b rows are heads*(nope+v), with v == nope here.
# heads = kv_b_rows / (2*nope) and q_rows = heads*(nope+rope) pin both.
for heads in range(1, 129):
    if Wq.shape[0] % heads or Wkvb.shape[0] % heads:
        continue
    nope = Wq.shape[0] // heads - ROPE
    if nope > 0 and Wkvb.shape[0] // heads == 2 * nope:
        HEADS, NOPE, VDIM = heads, nope, nope
        break
if HEADS is None:
    raise SystemExit("could not solve the head split from the tensor shapes")
print(f"d={D} heads={HEADS} nope={NOPE} rope={ROPE} v={VDIM} kv_lora={KV_LORA}")

rng = np.random.default_rng(args.seed)
xs = mx.array(rng.standard_normal((args.tokens, D), dtype=np.float32) * 0.5)

# ── shared front half: what every position contributes to the cache ─────────
h = rms(xs, g_in)                                   # [T, d]
q_all = (h @ Wq.T).reshape(args.tokens, HEADS, NOPE + ROPE)
q_nope, q_pe = q_all[..., :NOPE], q_all[..., NOPE:]

kva = h @ Wkva.T                                    # [T, kv_lora + rope]
c = rms(kva[:, :KV_LORA], g_kva)                    # [T, kv_lora]  ← THE CACHE
k_pe = kva[:, KV_LORA:]                             # [T, rope]     ← shared key

scale = (NOPE + ROPE) ** -0.5
qi = args.tokens - 1                                # score the last position

# ── NAIVE: up-project the whole cache, then ordinary attention ─────────────
kv = (c @ Wkvb.T).reshape(args.tokens, HEADS, NOPE + VDIM)
k_nope, v = kv[..., :NOPE], kv[..., NOPE:]
scores_naive = (mx.sum(q_nope[qi][None] * k_nope, axis=-1).T
                + (q_pe[qi] @ k_pe.T)) * scale      # [heads, T]
p_naive = mx.softmax(scores_naive, axis=-1)
o_naive = mx.sum(p_naive[..., None] * v.transpose(1, 0, 2), axis=1)   # [heads, v]
out_naive = o_naive.reshape(-1) @ Wo.T

# ── LATENT: never materialise K or V ───────────────────────────────────────
# kv_b_proj's rows are [head][nope|v]; the K half turns a head's q_nope into a
# latent query, the V half turns a latent context back into that head's output.
Wkvb_h = Wkvb.reshape(HEADS, NOPE + VDIM, KV_LORA)
Wk, Wv = Wkvb_h[:, :NOPE, :], Wkvb_h[:, NOPE:, :]

q_lat = mx.sum(q_nope[qi][:, :, None] * Wk, axis=1)                   # [heads, kv_lora]
scores_lat = (q_lat @ c.T + q_pe[qi] @ k_pe.T) * scale                # [heads, T]
p_lat = mx.softmax(scores_lat, axis=-1)
o_lat = p_lat @ c                                                      # [heads, kv_lora]
o_heads = mx.sum(o_lat[:, None, :] * Wv, axis=2)                       # [heads, v]
out_lat = o_heads.reshape(-1) @ Wo.T

mx.eval(out_naive, out_lat, scores_naive, scores_lat)


def relerr(a, b):
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-9))


err_s = relerr(scores_lat, scores_naive)
err_o = relerr(out_lat, out_naive)
print(f"latent vs naive — scores {err_s:.2e}, output {err_o:.2e}")
if max(err_s, err_o) > 1e-4:
    raise SystemExit("the two formulations DISAGREE — the latent algebra is wrong, "
                     "and no shader written against it would be right")
print("the two formulations agree: caching the 512 latent loses nothing")
print(f"cache per token: {KV_LORA + ROPE} values, vs {HEADS * (NOPE + VDIM)} for the MHA equivalent")

# Raw f32 .bin, not .npy — tests/kernels/real-weights.mjs reads bundles that
# way and a second format would be a second thing to get wrong.
out = bundle


def dump(name, arr):
    np.ascontiguousarray(np.array(arr, dtype=np.float32)).tofile(out / f"{name}.bin")


# kv_b_proj, dequantized and SPLIT into the two roles it plays. Wk is stored
# TRANSPOSED: q_lat = Wk^T q_nope contracts over kv_b's stored ROW axis, which
# a quantized kernel cannot do (the scale groups run the other way). Flipping it
# once at load turns both uses into ordinary row-major matvecs over the last
# axis, and costs 4 MB a layer in f16 — 1% of the model, against a KV cache that
# just got 7x smaller. Quantized-and-transposed would save that 1% and buy a
# kernel with strided scale lookups; not worth it.
dump("wk_t", mx.transpose(Wk, (0, 2, 1)))   # [heads, kv_lora, nope]
dump("wv", Wv)                              # [heads, v, kv_lora]
dump("ref_qnope", q_nope[qi])               # [heads, nope]   before the latent hop
dump("ref_oheads", o_heads)                 # [heads, v]      after coming back out
dump("ref_c", c)                    # [T, kv_lora]   the cache
dump("ref_kpe", k_pe)               # [T, rope]      the shared key
dump("ref_qlat", q_lat)             # [heads, kv_lora]  q_nope already in latent space
dump("ref_qpe", q_pe[qi])           # [heads, rope]
dump("ref_scores", scores_naive)    # [heads, T]     pre-softmax, scaled
dump("ref_olat", o_lat)             # [heads, kv_lora]  the kernel's output
dump("ref_out", out_naive)          # [d]            after Wv and o_proj
# Merged INTO meta.json rather than written beside it: real-weights.mjs walks
# bundle dirs and dispatches on meta.kernel, so a bundle that keeps its identity
# in a second file is a bundle that suite silently skips.
meta.update({
    "kernel": "mla_attention",
    "model": meta["repo"],
    "tensor": f"{args.layer}.self_attn (q/kv_a/kv_b/o)",
    "layer": args.layer,
    "reference": "mlx.core.dequantize + MLA computed two ways",
    "d": D, "heads": HEADS, "nope": NOPE, "rope": ROPE, "v": VDIM,
    "kv_lora": KV_LORA, "tokens": args.tokens, "query_at": qi,
    "softmax_scale": scale, "rope_applied": args.rope,
    "latent_vs_naive_rel_err": {"scores": err_s, "output": err_o},
    "cache_values_per_token": KV_LORA + ROPE,
    "mha_equivalent_per_token": HEADS * (NOPE + VDIM),
})
(out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
print(f"wrote {out}/meta.json + ref_*.bin")
