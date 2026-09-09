"""
Phase 1 gate — what does per-row int-b KV do to ATTENTION OUTPUT?

Phase 0b compared quantizers on inner-product error, which is the right metric
for ranking them but the wrong one for deciding whether to ship. What leaves an
attention block is softmax(QK^T/sqrt(d)) @ V, and both halves are quantized: an
error in K moves the weights, an error in V moves what they average. Softmax
also forgives a lot — a uniform shift in scores cancels — so score error
overstates the damage, while the weighted sum can concentrate it.

So this measures the block's OUTPUT on real cached vectors, against exact f16,
for the bit-widths worth considering. Same source as phase 0b: the prefix
pool's on-disk KV (shape fields only, never meta.ids).

    uv run python scripts/kv-bits-gate.py --entry <dir> [--spec qwen36]
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from importlib import import_module

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
_real = import_module("turboquant-real")
load_kv = _real.load_kv


def quant_row(x: np.ndarray, bits: int, group: int = 0, asym: bool = False) -> np.ndarray:
    """Per-GROUP max-scaling — kv_quantize_int8.wgsl's scheme, generalized.

    `group` is how many values share one scale; 0 means the whole row, which is
    what the shipped int8 path does (HEAD_DIM = 256 values per scale). That is
    fine at 8 bits and ruinous at 4: one large coordinate sets the scale and the
    other 255 collapse into a couple of levels. Smaller groups cost more scales
    and are the obvious knob to measure before writing any kernel.

    `asym` stores a zero point as well as a scale, which matters for values that
    are not centred on zero (V often is not).
    """
    g = group or x.shape[-1]
    n, d = x.shape
    assert d % g == 0, f"group {g} does not divide head dim {d}"
    y = x.reshape(n, d // g, g)
    if asym:
        lo = y.min(axis=-1, keepdims=True)
        hi = y.max(axis=-1, keepdims=True)
        levels = (1 << bits) - 1
        s = np.maximum((hi - lo) / levels, 1e-8)
        q = np.rint((y - lo) / s).clip(0, levels)
        out = q * s + lo
    else:
        lim = (1 << (bits - 1)) - 1
        s = np.maximum(np.abs(y).max(axis=-1, keepdims=True) / lim, 1e-8)
        q = np.rint(y / s).clip(-lim, lim)
        out = q * s
    return out.reshape(n, d).astype(np.float32)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Causal attention, one head. Q,K,V are [tokens, dim]."""
    d = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d)
    n = scores.shape[0]
    scores = np.where(np.tril(np.ones((n, n), dtype=bool)), scores, -np.inf)
    scores -= scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w /= w.sum(axis=-1, keepdims=True)
    return w @ V, w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True)
    ap.add_argument("--spec", default="qwen36")
    ap.add_argument("--bits", default="3,4,5,8")
    a = ap.parse_args()

    K, V = load_kv(pathlib.Path(a.entry), a.spec)
    layers, heads, tokens, dim = K.shape
    print(f"real KV — {layers} attention layers x {heads} kv heads x {tokens} tokens, dim {dim}")
    print("queries are the cached keys (a stand-in: real Q is a different projection)\n")

    exact_per_token = 2 * heads * dim * 2  # K+V, f16
    print(f"  {'bits':>4} {'group':>6} {'mode':>6} {'bytes/tok':>10} {'vs f16':>7}"
          f" {'attn out rel err':>17} {'max wt shift':>13} {'top-1':>7}")
    for bits in [int(b) for b in a.bits.split(",")]:
        for group in [g for g in (0, 128, 64, 32) if g == 0 or dim % g == 0]:
            for asym in (False, True):
                out_err, w_shift, top1 = [], [], []
                for L in range(layers):
                    for h in range(heads):
                        k, v = K[L, h], V[L, h]
                        ref, wref = attention(k, k, v)
                        got, wq = attention(k, quant_row(k, bits, group, asym),
                                            quant_row(v, bits, group, asym))
                        out_err.append(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-9))
                        w_shift.append(np.abs(wq - wref).max())
                        top1.append(float((wq.argmax(-1) == wref.argmax(-1)).mean()))
                g = group or dim
                # one f16 scale per group per (head, side), plus a zero point if asymmetric
                scal = (2 if asym else 1) * 2 * (dim // g) * 2 * heads
                per_token = 2 * heads * dim * bits / 8 + scal
                print(f"  {bits:>4} {g:>6} {'asym' if asym else 'sym':>6} {per_token:>10.0f}"
                      f" {exact_per_token / per_token:>6.1f}x {float(np.mean(out_err)):>17.5f}"
                      f" {float(np.mean(w_shift)):>13.5f} {float(np.mean(top1)) * 100:>6.1f}%")

    print(f"\n  f16 baseline: {exact_per_token} bytes/token/layer"
          f"  →  {exact_per_token * layers / 1024:.1f} KB per token across {layers} layers")


if __name__ == "__main__":
    main()
