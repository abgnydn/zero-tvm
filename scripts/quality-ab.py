#!/usr/bin/env python3
"""Perplexity A/B between two MLX checkpoints, with error bars.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/quality-ab.py \
        --a ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-4bit \
        --b ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp

THE QUESTION THIS ANSWERS: did quantizing further cost anything?

It is a question about the WEIGHTS, not about the engine. The engine's
fidelity to whatever weights it is given is already pinned to ~1e-4 by
tests/kernels/real-weights.mjs and, over 512 scored positions, by
scripts/quality-eval.mjs. So measure quality on the reference, where it costs
seconds instead of minutes, and let the existing fidelity apparatus carry the
result across to the browser.

WHY NOT scripts/quality-eval.mjs FOR THIS. That harness scores ONE contiguous
prefix, so early positions — which have almost no context and therefore huge
NLL — dominate. Measured on Llama-3.2-1B over the same text:

      128 positions   ppl 151.15  (+/- 31.7%)
      256 positions   ppl  65.57  (+/- 20.8%)
      512 positions   ppl  41.31  (+/- 13.4%)
     1198 positions   ppl  32.18  (+/-  8.3%)

The number moves 5x with the window length. Two runs at different --tokens are
not comparable at all, and at the 256 default you need a ~59% change to clear
2 sigma — fine for a model quantized into gibberish, useless for the 10-20%
regression an expert-quantization change would actually produce.

This scores many INDEPENDENT windows and reports the standard error, which is
what makes a modest regression visible. Same windows for both checkpoints.

Corpus is local text, never a download: this repo's own markdown and source by
default. --text points it anywhere.
"""

import argparse
import glob
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

p = argparse.ArgumentParser(description="perplexity A/B between two checkpoints")
p.add_argument("--a", required=True, help="baseline checkpoint (the parent build)")
p.add_argument("--b", help="candidate checkpoint; omit to just measure --a")
p.add_argument("--text", default=None, help="a file, or a glob; default = this repo's docs+src")
p.add_argument("--window", type=int, default=512, help="tokens per independent window")
p.add_argument("--windows", type=int, default=24, help="how many windows to score")
p.add_argument("--out", default=None, help="write JSON here")
p.add_argument(
    "--compare",
    nargs=2,
    metavar=("A.json", "B.json"),
    help=(
        "combine two single-arm runs instead of scoring. For a model too large "
        "to load twice safely: run --a X --out a.json, then --a Y --out b.json, "
        "then --compare a.json b.json. Each arm peaks at ONE checkpoint in its "
        "own process, so memory is released by the OS rather than by "
        "mx.clear_cache(). The windows are identical across runs because they "
        "are the first N of a deterministic tokenization of the same corpus — "
        "which this re-checks rather than assumes."
    ),
)
p.add_argument(
    "--dtype",
    default="bfloat16",
    choices=["bfloat16", "float32"],
    help=(
        "activation dtype. bfloat16 (default) because BOTH sides get the same "
        "treatment and only their difference is read — f32 buys nothing for a "
        "comparison and doubles resident memory, which is what made the 35B "
        "pair hit 23.3 of 23.5 GB swap. Use float32 when grading the ENGINE "
        "against a reference (scripts/mlx-ref.py), where activation error is "
        "the thing being measured."
    ),
)
args = p.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def report(ra: dict, rb: dict, window: int) -> dict:
    """The paired comparison and the verdict. Shared by the one-process path
    and --compare, so both cannot drift apart."""
    if len(ra["per_window"]) != len(rb["per_window"]):
        raise SystemExit(
            f"arms scored different window counts ({len(ra['per_window'])} vs "
            f"{len(rb['per_window'])}) — not comparable")
    delta = (rb["perplexity"] / ra["perplexity"] - 1) * 100
    d = [b - a for a, b in zip(ra["per_window"], rb["per_window"])]
    n = len(d)
    dm = sum(d) / n
    dvar = sum((x - dm) ** 2 for x in d) / max(1, n - 1)
    dsem = math.sqrt(dvar / n)
    z = abs(dm) / dsem if dsem > 0 else float("inf")
    z_unpaired = abs(rb["mean_nll"] - ra["mean_nll"]) / math.sqrt(ra["sem_nll"] ** 2 + rb["sem_nll"] ** 2)
    worse = sum(1 for x in d if x > 0)
    print(f"  A perplexity {ra['perplexity']:.3f}  [{ra['ppl_lo']:.3f}, {ra['ppl_hi']:.3f}]")
    print(f"  B perplexity {rb['perplexity']:.3f}  [{rb['ppl_lo']:.3f}, {rb['ppl_hi']:.3f}]")
    print(f"\n  B is {delta:+.1f}% perplexity vs A, paired z = {z:.1f}"
          f"  (unpaired {z_unpaired:.1f}; B worse on {worse}/{n} windows)")
    if z < 2:
        print("  NOT DISTINGUISHABLE at this window count — raise --windows before concluding")
    elif delta <= 10:
        print("  SHIP: within +10% with separated error bars")
    elif delta <= 25:
        print("  MARGINAL: +10-25%. Real, and a task benchmark should decide, not this")
    else:
        print("  DO NOT SHIP on this evidence: >+25%")
    print("\n  Perplexity understates reasoning damage — published work puts the")
    print("  math-accuracy drop at 3-bit around 3x the perplexity drop. A pass")
    print("  here is necessary, not sufficient.")
    return {"delta_pct": delta, "z": z, "z_unpaired": z_unpaired,
            "windows_b_worse": worse, "window": window}


if args.compare:
    a = json.load(open(args.compare[0]))
    b = json.load(open(args.compare[1]))
    ra, rb = a["a"], b["a"]
    if a.get("window") != b.get("window") or a.get("windows") != b.get("windows"):
        raise SystemExit("the two runs used different windowing — not comparable")
    # The windows must be the SAME TEXT, not merely the same count. Both runs
    # record the tokenization's fingerprint for exactly this check.
    if a.get("ids_digest") != b.get("ids_digest"):
        raise SystemExit(
            f"the two runs scored DIFFERENT tokens "
            f"({a.get('ids_digest')} vs {b.get('ids_digest')}) — not comparable")
    print(f"A  {ra['path']}\nB  {rb['path']}\n")
    out = report(ra, rb, a.get("window"))
    if args.out:
        json.dump({"a": ra, "b": rb, **out}, open(args.out, "w"), indent=1)
        print(f"\n  -> {args.out}")
    raise SystemExit(0)


def corpus_text() -> str:
    if args.text:
        files = sorted(glob.glob(args.text)) or [args.text]
    else:
        files = sorted(glob.glob(os.path.join(REPO, "docs", "*.md")))
        files += sorted(glob.glob(os.path.join(REPO, "src", "zero-tvm", "*.ts")))
        files += [os.path.join(REPO, "BENCH.md"), os.path.join(REPO, "CLAUDE.md")]
    out = []
    for f in files:
        try:
            out.append(open(f, encoding="utf-8").read())
        except OSError:
            pass
    return "\n\n".join(out)


def score(model_path: str, windows: list[list[int]]) -> dict:
    """Mean NLL per window. Teacher-forced, no cache, one forward per window."""
    model, _ = load(model_path)
    if args.dtype == "float32":
        model.set_dtype(mx.float32)
    per_window = []
    t0 = time.time()
    for i, w in enumerate(windows):
        arr = mx.array(w)[None]
        logits = model(arr[:, :-1])
        lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
        tgt = mx.array(w[1:])
        picked = mx.take_along_axis(lp[0], tgt[:, None], axis=-1)[:, 0]
        mx.eval(picked)
        per_window.append(float(-picked.mean()))
        if (i + 1) % 8 == 0:
            print(f"    {i + 1}/{len(windows)} windows ({time.time() - t0:.0f}s)", flush=True)
    del model
    mx.clear_cache()
    n = len(per_window)
    mean = sum(per_window) / n
    var = sum((x - mean) ** 2 for x in per_window) / max(1, n - 1)
    sem = math.sqrt(var / n)
    return {
        "mean_nll": mean,
        "perplexity": math.exp(mean),
        # Error bars on perplexity are asymmetric because exp() is convex; both
        # ends are reported rather than a single +/- that would be wrong on one
        # side. This is the number that decides whether a difference is real.
        "ppl_lo": math.exp(mean - sem),
        "ppl_hi": math.exp(mean + sem),
        "sem_nll": sem,
        "windows": n,
        "per_window": per_window,
        "seconds": time.time() - t0,
    }


# Tokenize once, with A's tokenizer, and score BOTH on the identical windows.
# Two checkpoints of the same family share a tokenizer; if they did not, the
# comparison would be meaningless no matter how the windows were chosen.
_, tokenizer = load(args.a)
text = corpus_text()
ids = tokenizer.encode(text)
need = args.windows * args.window
if len(ids) < need:
    avail = len(ids) // args.window
    print(f"corpus has {len(ids)} tokens; {args.windows} x {args.window} needs {need}")
    if avail < 2:
        raise SystemExit("need at least 2 full windows — pass a bigger --text")
    print(f"scoring {avail} windows instead")
    args.windows = avail
windows = [ids[i * args.window : (i + 1) * args.window] for i in range(args.windows)]
print(f"corpus {len(ids)} tokens -> {args.windows} independent windows of {args.window} ({args.dtype})\n")

print(f"A  {args.a}")
ra = score(args.a, windows)
print(f"   perplexity {ra['perplexity']:.3f}  [{ra['ppl_lo']:.3f}, {ra['ppl_hi']:.3f}]"
      f"  over {ra['windows']} windows, {ra['seconds']:.0f}s\n")

result = {"a": {"path": args.a, **ra}, "window": args.window, "windows": args.windows}

if args.b:
    print(f"B  {args.b}")
    rb = score(args.b, windows)
    print(f"   perplexity {rb['perplexity']:.3f}  [{rb['ppl_lo']:.3f}, {rb['ppl_hi']:.3f}]"
          f"  over {rb['windows']} windows, {rb['seconds']:.0f}s\n")
    result["b"] = {"path": args.b, **rb}

    delta = (rb["perplexity"] / ra["perplexity"] - 1) * 100

    # PAIRED, because both arms scored the IDENTICAL windows. Between-window
    # variance is the dominant term — some passages are simply harder than
    # others, and that difficulty is common to both checkpoints — so an
    # unpaired test spends nearly all its power measuring the corpus instead
    # of the quantization. Differencing per window cancels it.
    # Measured on OLMoE expert-3-bit: unpaired z = 0.8 ("not distinguishable"),
    # paired z on the same 24 windows resolves the same +8.4% cleanly.
    d = [b - a for a, b in zip(ra["per_window"], rb["per_window"])]
    n = len(d)
    dm = sum(d) / n
    dvar = sum((x - dm) ** 2 for x in d) / max(1, n - 1)
    dsem = math.sqrt(dvar / n)
    z = abs(dm) / dsem if dsem > 0 else float("inf")
    # Kept for reference: the unpaired form, which is what this used to report.
    z_unpaired = abs(rb["mean_nll"] - ra["mean_nll"]) / math.sqrt(ra["sem_nll"] ** 2 + rb["sem_nll"] ** 2)
    result["delta_pct"] = delta
    result["z"] = z
    result["z_unpaired"] = z_unpaired
    result["windows_b_worse"] = sum(1 for x in d if x > 0)

    print(f"  B is {delta:+.1f}% perplexity vs A, paired z = {z:.1f}"
          f"  (unpaired {z_unpaired:.1f}; B worse on {result['windows_b_worse']}/{n} windows)")
    if z < 2:
        print("  NOT DISTINGUISHABLE at this window count — raise --windows before concluding")
    elif delta <= 10:
        print("  SHIP: within +10% with separated error bars")
    elif delta <= 25:
        print("  MARGINAL: +10-25%. Real, and a task benchmark should decide, not this")
    else:
        print("  DO NOT SHIP on this evidence: >+25%")
    print("\n  Perplexity understates reasoning damage — published work puts the")
    print("  math-accuracy drop at 3-bit around 3x the perplexity drop. A pass")
    print("  here is necessary, not sufficient.")

if args.out:
    json.dump(result, open(args.out, "w"), indent=1)
    print(f"\n  -> {args.out}")
