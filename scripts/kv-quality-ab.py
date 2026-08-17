"""
Does quantizing the KV CACHE cost quality? Paired perplexity, same windows.

WHY THIS IS NOT quality-ab.py. That script scores "teacher-forced, no cache,
one forward per window" — which is right for comparing two checkpoints and
structurally blind here: with no cache, a KV quantizer never runs. Measuring
it requires feeding each window through the model INCREMENTALLY so the cache
is written, quantized, and read back exactly as it would be in a chat.

Everything else follows quality-ab.py deliberately: the same corpus, the same
independent windows scored by BOTH arms, and a PAIRED difference. Pairing is
what makes a modest regression visible — on the OLMoE run the same data gave
unpaired z = 0.8 ("not distinguishable") and paired z = 14.7.

Needs Metal, so it runs in your shell, not the agent's sandbox:

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/kv-quality-ab.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp --kv-bits 8

Read the result as a gate, not a score: if 8-bit is within noise, the engine's
int8 KV work is justified on THIS model rather than by analogy to Phi-3.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--kv-bits", type=int, default=8, help="cache bit-width for arm B")
p.add_argument("--kv-group-size", type=int, default=64)
p.add_argument("--windows", type=int, default=12)
p.add_argument("--window", type=int, default=1024, help="tokens per window")
p.add_argument("--chunk", type=int, default=128, help="tokens per forward while filling the cache")
p.add_argument("--text", default="", help="glob for corpus files; defaults to this repo's docs+src")
args = p.parse_args()


def corpus_text() -> str:
    """Same corpus as quality-ab.py — ordinary English and code, no cherry-picking."""
    if args.text:
        files = sorted(glob.glob(args.text)) or [args.text]
    else:
        files = sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
        files += sorted(glob.glob(os.path.join(REPO, "src", "zero-tvm", "*.ts")))
        files += [os.path.join(REPO, "BENCH.md"), os.path.join(REPO, "CLAUDE.md")]
    out = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                out.append(fh.read())
        except OSError:
            pass
    return "\n\n".join(out)


def quantize_cache(cache, bits: int, group_size: int) -> list:
    """Convert the ATTENTION caches to quantized, leaving others alone.

    Qwen3.5/3.6/3.8 are hybrids: most layers are gated-DeltaNet and carry a
    recurrent state, not a KV cache. Quantizing those would be meaningless at
    best; mlx_lm's own helper converts only what is a KVCache, which is exactly
    the set the engine's int8 path would touch.
    """
    try:
        from mlx_lm.models.cache import maybe_quantize_kv_cache
        maybe_quantize_kv_cache(cache, 0, group_size, bits)
        return cache
    except ImportError:
        pass
    converted = []
    for c in cache:
        to_q = getattr(c, "to_quantized", None)
        converted.append(to_q(group_size=group_size, bits=bits) if to_q else c)
    if not any(getattr(c, "to_quantized", None) for c in cache):
        raise SystemExit("this mlx_lm has no quantized KV cache — upgrade it before trusting a result")
    return converted


def score(model, windows: list[list[int]], bits: int | None) -> dict:
    """Mean NLL per window, filled through a real cache chunk by chunk."""
    per_window = []
    t0 = time.time()
    for i, w in enumerate(windows):
        cache = make_prompt_cache(model)
        if bits:
            cache = quantize_cache(cache, bits, args.kv_group_size)
        lps = []
        ids = mx.array(w)[None]
        for s in range(0, len(w) - 1, args.chunk):
            piece = ids[:, s:min(s + args.chunk, len(w) - 1)]
            logits = model(piece, cache=cache)
            lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
            tgt = mx.array(w[s + 1:s + 1 + piece.shape[1]])
            lps.append(mx.take_along_axis(lp[0], tgt[:, None], axis=-1)[:, 0])
        picked = mx.concatenate(lps)
        mx.eval(picked)
        per_window.append(float(-picked.mean()))
        del cache
        mx.clear_cache()
        if (i + 1) % 4 == 0:
            print(f"    {i + 1}/{len(windows)} windows ({time.time() - t0:.0f}s)", flush=True)
    n = len(per_window)
    mean = sum(per_window) / n
    var = sum((x - mean) ** 2 for x in per_window) / max(1, n - 1)
    return {"per_window": per_window, "mean_nll": mean, "perplexity": math.exp(mean),
            "sem_nll": math.sqrt(var / n), "seconds": time.time() - t0}


model, tokenizer = load(args.model)
ids = tokenizer.encode(corpus_text())
need = args.windows * args.window
if len(ids) < need:
    raise SystemExit(f"corpus has {len(ids)} tokens; {args.windows} x {args.window} needs {need}")
windows = [ids[i * args.window:(i + 1) * args.window] for i in range(args.windows)]
print(f"{args.windows} windows x {args.window} tokens, cache filled {args.chunk} at a time\n")

print("  arm A — f16 cache")
a = score(model, windows, None)
print(f"  arm B — {args.kv_bits}-bit cache, group {args.kv_group_size}")
b = score(model, windows, args.kv_bits)

# PAIRED: difference per window, so between-window difficulty cancels.
d = [y - x for x, y in zip(a["per_window"], b["per_window"])]
n = len(d)
md = sum(d) / n
vd = sum((x - md) ** 2 for x in d) / max(1, n - 1)
sd = math.sqrt(vd / n)
z = md / sd if sd > 0 else float("inf")
worse = sum(1 for x in d if x > 0)

print(f"\n  f16 cache      ppl {a['perplexity']:.4f}   nll {a['mean_nll']:.5f}")
print(f"  {args.kv_bits}-bit cache    ppl {b['perplexity']:.4f}   nll {b['mean_nll']:.5f}")
print(f"  paired dNLL {md:+.6f} +/- {sd:.6f}   z = {z:.1f}   worse on {worse}/{n} windows")
print(f"  perplexity cost: {(b['perplexity'] / a['perplexity'] - 1) * 100:+.2f}%")
print("\n  " + ("WITHIN NOISE — the cache quantizer is not measurably hurting this model"
                if abs(z) < 2 else
                "REAL DIFFERENCE — quantizing the cache costs quality on this model"))
print("  (paired z, so |z| < 2 means the per-window differences do not separate from zero)")
