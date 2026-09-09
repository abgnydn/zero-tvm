#!/usr/bin/env python3
"""
TURBOQUANT REFERENCE — phase 0 of docs/TURBOQUANT_PLAN.md.

Transcribed from arXiv:2504.19874, Algorithms 1 and 2, and then implemented a
SECOND time by a different route so the two can disagree. That is the rule
that paid for MLA: a WGSL test cannot tell "we misunderstood the algorithm"
from "the shader is wrong", so the algorithm gets settled in Python first.

Algorithm 1 (Quant_mse), verbatim from the paper:
    y     <- Pi . x                                  (random rotation)
    idx_j <- argmin_k |y_j - c_k|                    (b-bit scalar codebook)
    x~    <- Pi^T . c[idx]                           (dequantize)

Algorithm 2 (Quant_prod), which is what attention needs:
    2: instantiate Quant_mse at bit-width b-1        <-- the stated bits/channel
                                                         INCLUDE the QJL bit
    idx  <- Quant_mse(x)
    r    <- x - DeQuant_mse(idx)                     (residual)
    qjl  <- sign(S . r)                              (1-bit sketch, S Gaussian)
    store (idx, qjl, ||r||)
    x~   <- DeQuant_mse(idx) + sqrt(pi/2)/d . ||r|| . S^T . qjl

The identity this whole thing rests on, and the reason it is cheap in an
attention kernel: a rotation preserves inner products, and

    <q, S^T qjl> = <S q, qjl>

so S never touches a cached vector at read time — the QUERY is projected once
per head, and each cached token costs one sign-dot.

WHAT THIS SCRIPT DECIDES (none of it is settled by reading):
  1. Is our fast Hadamard (with random signs) an acceptable stand-in for the
     paper's general random rotation? The paper's theory wants a Haar
     rotation; a d^2 matmul per cached token is not affordable in a decode
     kernel, so this measures the substitution rather than assuming it.
  2. Does the estimator beat our shipped int8-per-row KV at equal bits? That
     is the plan's phase-1 gate — if it does not, the complexity buys nothing.
  3. Is the estimator unbiased in practice at our real head dims (64/128/256)?

    uv run python scripts/turboquant-ref.py
"""

import numpy as np

RNG = np.random.default_rng(0)


# ── codebook: Lloyd–Max for the post-rotation coordinate distribution ────────
# After a random rotation of a unit vector, coordinates are Beta-distributed
# and, for the dimensions we care about, close to N(0, 1/d). The paper solves
# Eq. (4) numerically once per bit-width; Lloyd's algorithm on samples of the
# actual distribution is the same answer by a different route.
def lloyd_max(samples: np.ndarray, levels: int, iters: int = 60) -> np.ndarray:
    lo, hi = np.quantile(samples, [0.001, 0.999])
    c = np.linspace(lo, hi, levels)
    for _ in range(iters):
        edges = (c[:-1] + c[1:]) / 2
        idx = np.searchsorted(edges, samples)
        for k in range(levels):
            sel = samples[idx == k]
            if sel.size:
                c[k] = sel.mean()
        c.sort()
    return c


def codebook_for(d: int, bits: int, n: int = 400_000) -> np.ndarray:
    """Codebook for one coordinate of a rotated unit vector in d dims."""
    x = RNG.standard_normal((n // d + 1, d))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return lloyd_max(x.ravel(), 1 << bits)


# ── rotations ───────────────────────────────────────────────────────────────
def haar_rotation(d: int) -> np.ndarray:
    """A true random rotation — the paper's Pi. O(d^2) to apply."""
    q, r = np.linalg.qr(RNG.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def hadamard(d: int) -> np.ndarray:
    """Normalized Walsh–Hadamard. Requires d a power of two; O(d log d) to
    apply as a butterfly, which is why we want it instead of Pi."""
    assert d & (d - 1) == 0, "Hadamard needs a power-of-two dimension"
    h = np.ones((1, 1))
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


def rht(d: int) -> np.ndarray:
    """Randomized Hadamard: H . diag(random signs). The standard fast
    stand-in for a Haar rotation; whether it is good ENOUGH here is measured
    below rather than assumed."""
    return hadamard(d) * RNG.choice([-1.0, 1.0], size=d)


# ── the algorithms ──────────────────────────────────────────────────────────
class TurboQuant:
    """Algorithm 2 (which contains Algorithm 1 at bit-width b-1)."""

    def __init__(self, d: int, bits: int, rotation: str = "rht"):
        assert bits >= 2, "one bit goes to the QJL sketch"
        self.d, self.bits = d, bits
        self.mse_bits = bits - 1                      # Algorithm 2, line 2
        self.Pi = rht(d) if rotation == "rht" else haar_rotation(d)
        self.S = RNG.standard_normal((d, d))          # Algorithm 2, line 3
        self.cb = codebook_for(d, self.mse_bits)

    def quant(self, x: np.ndarray) -> dict:
        norm = np.linalg.norm(x)
        u = x / norm                                  # Alg 1 assumes the sphere
        y = self.Pi @ u
        idx = np.abs(y[:, None] - self.cb[None, :]).argmin(axis=1)
        r = u - self.Pi.T @ self.cb[idx]              # residual, Alg 2 line 6
        qjl = np.sign(self.S @ r)
        qjl[qjl == 0] = 1.0
        return {"idx": idx, "qjl": qjl, "rnorm": np.linalg.norm(r), "norm": norm}

    # -- estimator A: reconstruct the vector, then dot (literal Algorithm 2) --
    def dequant(self, c: dict) -> np.ndarray:
        x_mse = self.Pi.T @ self.cb[c["idx"]]
        x_qjl = np.sqrt(np.pi / 2) / self.d * c["rnorm"] * (self.S.T @ c["qjl"])
        return (x_mse + x_qjl) * c["norm"]

    def ip_reconstruct(self, q: np.ndarray, c: dict) -> float:
        return float(q @ self.dequant(c))

    # -- estimator B: the kernel's route — never materialize the vector -------
    # <q, Pi^T c> = <Pi q, c>  and  <q, S^T qjl> = <S q, qjl>, so both the
    # rotation and the sketch touch only the QUERY. This is the arithmetic an
    # attention kernel would actually run; if A and B disagree, the algebra a
    # shader would rely on is wrong.
    def ip_kernel(self, q: np.ndarray, c: dict) -> float:
        Piq = self.Pi @ q
        Sq = self.S @ q
        mse_part = float(Piq @ self.cb[c["idx"]])
        qjl_part = np.sqrt(np.pi / 2) / self.d * c["rnorm"] * float(Sq @ c["qjl"])
        return (mse_part + qjl_part) * c["norm"]

    def bytes_per_vector(self) -> float:
        """codes + sketch + two f16 scalars (vector norm, residual norm)."""
        return self.d * self.mse_bits / 8 + self.d / 8 + 4


# ── the baseline we must beat, AT EQUAL BITS ────────────────────────────────
# kv_quantize_int8.wgsl's scheme generalized to b bits: symmetric, one f16
# scale per (slot, head, side), HEAD_DIM values sharing it. Comparing
# TurboQuant against int8 is meaningless — int8 spends twice the bits. The
# question is whether rotation+sketch beats plain max-scaling AT THE SAME
# BIT-WIDTH, which is the only reason to pay for the complexity.
def intb_row(x: np.ndarray, bits: int) -> np.ndarray:
    lim = (1 << (bits - 1)) - 1
    if lim < 1:
        # 1 bit: sign times the mean magnitude is the MSE-optimal 2-level code
        return np.sign(x) * np.abs(x).mean()
    s = np.abs(x).max() / lim
    return np.rint(x / s).clip(-lim, lim) * s


# ── the distribution that actually matters ──────────────────────────────────
# Isotropic Gaussian vectors are the one case where a rotation has NOTHING to
# fix, so measuring only on them understates the method to the point of
# meaninglessness. Real attention keys are the opposite: a handful of channels
# carry far larger magnitude than the rest (the "massive activation" /
# outlier-channel finding that every KV-quantization paper builds on), and
# per-row max-scaling is exactly what those destroy — one big channel sets the
# scale and the other 250 coordinates land in a couple of levels.
#
# This is a PROXY, not real weights: phase 2 of the plan replays real K/V.
def outlier_vector(d: int, n_out: int = 4, ratio: float = 20.0) -> np.ndarray:
    x = RNG.standard_normal(d)
    ch = RNG.choice(d, size=n_out, replace=False)
    x[ch] *= ratio
    return x


def main() -> None:
    print("TurboQuant reference — phase 0\n")
    q_trials, k_trials = 200, 200

    print("A vs B (the algebra a kernel would rely on)")
    for d in (64, 128, 256):
        tq = TurboQuant(d, 4)
        worst = 0.0
        for _ in range(50):
            x = RNG.standard_normal(d)
            q = RNG.standard_normal(d)
            c = tq.quant(x)
            worst = max(worst, abs(tq.ip_reconstruct(q, c) - tq.ip_kernel(q, c)))
        print(f"  d={d:<4} max |A-B| = {worst:.3e}   {'agree' if worst < 1e-9 else 'DISAGREE'}")

    print("\nUnbiasedness of the inner-product estimate (mean error / |true|)")
    for d in (64, 128, 256):
        for bits in (3, 4):
            tq = TurboQuant(d, bits)
            errs, mags = [], []
            for _ in range(k_trials):
                x = RNG.standard_normal(d)
                c = tq.quant(x)
                for _ in range(q_trials // 20):
                    q = RNG.standard_normal(d)
                    errs.append(tq.ip_kernel(q, c) - float(q @ x))
                    mags.append(abs(float(q @ x)))
            bias = np.mean(errs) / np.mean(mags)
            rel = np.std(errs) / np.mean(mags)
            print(f"  d={d:<4} b={bits}  bias={bias:+.4f}  rel.err={rel:.4f}")

    print("\nRotation: does the fast Hadamard stand in for a Haar rotation?")
    for d in (128, 256):
        for rot in ("haar", "rht"):
            tq = TurboQuant(d, 4, rotation=rot)
            errs, mags = [], []
            for _ in range(k_trials):
                x = RNG.standard_normal(d)
                c = tq.quant(x)
                q = RNG.standard_normal(d)
                errs.append(tq.ip_kernel(q, c) - float(q @ x))
                mags.append(abs(float(q @ x)))
            print(f"  d={d:<4} {rot:<5} rel.err={np.std(errs)/np.mean(mags):.4f}")

    print("\nGATE(b) — the same, on OUTLIER-HEAVY vectors (4 channels x20)")
    print("  the regime the rotation exists for; real keys look like this")
    for d in (128, 256):
        for bits in (3, 4, 5):
            eb, mb = [], []
            for _ in range(600):
                x = outlier_vector(d)
                q = RNG.standard_normal(d)
                eb.append(float(q @ intb_row(x, bits)) - float(q @ x))
                mb.append(abs(float(q @ x)))
            base = np.std(eb) / np.mean(mb)
            tq = TurboQuant(d, bits)
            et, mt = [], []
            for _ in range(600):
                x = outlier_vector(d)
                c = tq.quant(x)
                q = RNG.standard_normal(d)
                et.append(tq.ip_kernel(q, c) - float(q @ x))
                mt.append(abs(float(q @ x)))
            tqe = np.std(et) / np.mean(mt)
            print(f"  d={d:<4} {bits}b  int={base:.4f}  TQ={tqe:.4f}  "
                  f"ratio={base/tqe:.2f}x  {'TQ wins' if tqe < base else 'int wins'}")

    print("\nGATE — TurboQuant vs max-scaled int-b, AT EQUAL BITS PER CHANNEL")
    print("  (the plan's phase-1 gate: rotation+sketch must beat plain scaling)")
    trials = 600
    for d in (128, 256):
        for bits in (2, 3, 4, 5, 8):
            eb, mb = [], []
            for _ in range(trials):
                x = RNG.standard_normal(d)
                q = RNG.standard_normal(d)
                eb.append(float(q @ intb_row(x, bits)) - float(q @ x))
                mb.append(abs(float(q @ x)))
            base = np.std(eb) / np.mean(mb)
            if bits >= 2:
                tq = TurboQuant(d, bits)
                et, mt = [], []
                for _ in range(trials):
                    x = RNG.standard_normal(d)
                    c = tq.quant(x)
                    q = RNG.standard_normal(d)
                    et.append(tq.ip_kernel(q, c) - float(q @ x))
                    mt.append(abs(float(q @ x)))
                tqe = np.std(et) / np.mean(mt)
                verdict = "TQ wins" if tqe < base else "int wins"
                print(f"  d={d:<4} {bits}b  int={base:.4f}  TQ={tqe:.4f}  "
                      f"ratio={base/tqe:.2f}x  {verdict}")
            else:
                print(f"  d={d:<4} {bits}b  int={base:.4f}")


if __name__ == "__main__":
    main()
