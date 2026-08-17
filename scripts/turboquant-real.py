"""
TurboQuant phase 0b — run the gate on REAL key/value vectors.

Phase 0 settled the algebra and measured the method on synthetic data, and the
result was conditional: on isotropic Gaussians a rotation has nothing to fix
and plain max-scaling wins by ~2x, while on an outlier proxy (4 channels x20)
TurboQuant wins from 4 bits up. Which of those regimes our models actually
live in cannot be decided by inventing vectors, so this reads real ones.

Where they come from: the engine's KV prefix pool writes its cache to disk
(src/zero-tvm/kv-pool.ts), so `entry.bin` IS a dump of a real forward pass in
the engine's own paged layout — no GPU and no re-run needed. Keys there are
POST-RoPE, which is what attention actually scores against and therefore what
a KV quantizer has to compress.

This reads only the cache tensor and the shape fields of meta.json. It never
touches meta.ids, so nothing about the prompt's content is required or read.

    uv run python scripts/turboquant-real.py --entry <dir> [--spec qwen36]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from importlib import import_module

_ref = import_module("turboquant-ref")
TurboQuant = _ref.TurboQuant
intb_row = _ref.intb_row
RNG = _ref.RNG

# Paged-cache geometry, from ModelSpec. See kv_append.wgsl's header:
#   pages[page * KV_PAGE_STRIDE + head * HEAD_PAGE_STRIDE + slot * HEAD_DIM + dim]
#   K at offset 0, V at V_PAGE_OFFSET = KV_HEADS * HEAD_PAGE_STRIDE
SPECS = {
    # id: (pageSize, kvHeads, headDim, attnLayerCount)
    "qwen36": (16, 2, 256, 10),
    "qwen38": (16, 4, 256, 16),
    "qwen35": (16, 4, 256, 8),
}


def load_kv(entry: pathlib.Path, spec: str):
    """-> (K, V), each [layer, head, token, headDim] float32."""
    meta = json.loads((entry / "meta.json").read_text())
    page_size, kv_heads, head_dim, layers = SPECS[spec]
    tokens = int(meta["tokens"])
    if int(meta["layerCount"]) != layers:
        raise SystemExit(f"entry has {meta['layerCount']} layers, spec {spec} expects {layers}")

    head_page = page_size * head_dim
    kv_page = 2 * kv_heads * head_page
    v_off = kv_heads * head_page
    pages = -(-tokens // page_size)
    per_layer = pages * kv_page
    if per_layer * 2 != int(meta["layerBytes"]):
        raise SystemExit(
            f"layout mismatch: computed {per_layer * 2} bytes/layer, meta says {meta['layerBytes']}"
        )

    raw = np.fromfile(entry / "entry.bin", dtype=np.float16, count=per_layer * layers)
    raw = raw.reshape(layers, pages, 2, kv_heads, page_size, head_dim)
    # -> [layer, side, head, page, slot, dim] then fold page/slot into token
    both = raw.transpose(0, 2, 3, 1, 4, 5).reshape(layers, 2, kv_heads, pages * page_size, head_dim)
    both = both[:, :, :, :tokens, :].astype(np.float32)
    del raw
    assert v_off == kv_heads * head_page  # documented, unused beyond the reshape
    return both[:, 0], both[:, 1]


def outlier_report(name: str, X: np.ndarray) -> None:
    """Is the premise true for OUR vectors? Per-channel magnitude spread."""
    flat = X.reshape(-1, X.shape[-1])
    per_ch = np.abs(flat).mean(axis=0)
    med = float(np.median(per_ch))
    order = np.argsort(per_ch)[::-1]
    top = per_ch[order[:8]]
    # Per-vector: how much of the mass sits in the largest few coordinates?
    mag = np.abs(flat)
    top4 = np.sort(mag, axis=1)[:, -4:].sum(axis=1)
    share = float(np.mean(top4 / np.maximum(mag.sum(axis=1), 1e-9)))
    print(f"  {name}: {flat.shape[0]} vectors of dim {flat.shape[-1]}")
    print(f"    per-channel mean|x|: median {med:.4f}, max {per_ch.max():.4f} "
          f"→ ratio {per_ch.max() / max(med, 1e-9):.1f}x")
    print(f"    top-8 channels: {' '.join(f'{v / max(med, 1e-9):.0f}x' for v in top)}")
    print(f"    mass in a vector's largest 4 of {flat.shape[-1]} coords: {share * 100:.1f}%")
    print(f"    kurtosis (per-vector, mean): {float(np.mean(_kurt(flat))):.1f}  (Gaussian = 3)")


def _kurt(X: np.ndarray) -> np.ndarray:
    m = X.mean(axis=1, keepdims=True)
    c = X - m
    v = (c ** 2).mean(axis=1)
    return (c ** 4).mean(axis=1) / np.maximum(v ** 2, 1e-12)


def gate(name: str, X: np.ndarray, bits_list, n_vec: int, n_query: int) -> None:
    """Relative inner-product error, TurboQuant vs max-scaled int-b, EQUAL bits."""
    flat = X.reshape(-1, X.shape[-1])
    d = flat.shape[-1]
    take = RNG.choice(flat.shape[0], size=min(n_vec, flat.shape[0]), replace=False)
    vecs = flat[take]
    # Queries drawn from the SAME distribution as the keys, not isotropic:
    # attention scores a query against keys it co-evolved with, and an
    # isotropic query would average away exactly the structure being tested.
    qtake = RNG.choice(flat.shape[0], size=n_query, replace=False)
    queries = flat[qtake]
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    print(f"\n  {name} — relative inner-product error (lower is better), {len(vecs)} vectors "
          f"x {n_query} queries")
    print(f"    {'bits':>5} {'TurboQuant':>12} {'int-b row':>12}   verdict")
    for bits in bits_list:
        tq = TurboQuant(d, bits)
        exact, e_tq, e_int = [], [], []
        for x in vecs:
            c = tq.quant(x)
            xi = intb_row(x, bits)
            for q in queries:
                t = float(q @ x)
                exact.append(t)
                e_tq.append(tq.ip_kernel(q, c) - t)
                e_int.append(float(q @ xi) - t)
        scale = np.sqrt(np.mean(np.square(exact)))
        r_tq = float(np.sqrt(np.mean(np.square(e_tq))) / scale)
        r_int = float(np.sqrt(np.mean(np.square(e_int))) / scale)
        better = "TQ" if r_tq < r_int else "int"
        ratio = max(r_tq, r_int) / max(min(r_tq, r_int), 1e-12)
        print(f"    {bits:>5} {r_tq:>12.5f} {r_int:>12.5f}   {better} {ratio:.2f}x better")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", required=True, help="dir holding meta.json + entry.bin")
    ap.add_argument("--spec", default="qwen36", choices=sorted(SPECS))
    ap.add_argument("--vectors", type=int, default=120)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--bits", default="3,4,5")
    a = ap.parse_args()

    K, V = load_kv(pathlib.Path(a.entry), a.spec)
    print(f"real KV from a live forward pass — spec {a.spec}, "
          f"{K.shape[0]} attention layers, {K.shape[1]} kv heads, {K.shape[2]} tokens, dim {K.shape[3]}")
    print("\nIS THE OUTLIER PREMISE TRUE HERE?")
    outlier_report("K (post-RoPE)", K)
    outlier_report("V", V)

    bits_list = [int(b) for b in a.bits.split(",")]
    print("\nTHE GATE — does rotation+sketch beat plain max-scaling at the SAME bit-width?")
    gate("K (post-RoPE)", K, bits_list, a.vectors, a.queries)
    gate("V", V, bits_list, a.vectors, a.queries)


if __name__ == "__main__":
    main()
