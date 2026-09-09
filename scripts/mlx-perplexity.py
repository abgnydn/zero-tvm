#!/usr/bin/env python3
"""Per-token NLL from mlx_lm over an EXACT token sequence.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-perplexity.py \
        --model ~/dev/zero-tvm/.weights-local/Llama-3.2-1B-Instruct-4bit \
        --ids /tmp/corpus-ids.json --out /tmp/ppl-llama32.json

WHY NOT `python -m mlx_lm.perplexity`. That exists, it works, and for the
question "did 3-bit experts cost this checkpoint anything" it is the right
tool and needs none of this file. Use it:

    uv run python -m mlx_lm.perplexity --model <path> --num-samples 32

What it cannot do is score the SAME tokens our engine scored. It samples its
own windows from a HuggingFace dataset, so its number is not comparable to
ours token for token — and the comparison is the whole point. Two numbers that
disagree tell you nothing unless they were computed over identical input.

So: this reads the token ids from a JSON file that scripts/quality-eval.mjs
also feeds to the browser, and returns per-position NLL for exactly those.

The reference runs in FLOAT32 for the same reason scripts/mlx-ref.py does — a
bf16 forward contributed twelve times more error than the engine it was
grading (Qwen3-30B-A3B: engine cosine 0.999985, mlx-in-bf16 0.997726). The
quantized weights stay packed either way, so this is cheap.
"""

import argparse
import json
import math
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

p = argparse.ArgumentParser(description="per-token NLL over an exact id sequence")
p.add_argument("--model", required=True, help="path to an MLX checkpoint")
p.add_argument("--ids", required=True, help="JSON file: {\"ids\": [...]} or a bare [...]")
p.add_argument("--out", required=True, help="where to write the result JSON")
p.add_argument("--chunk", type=int, default=512, help="tokens per forward pass")
p.add_argument(
    "--dtype",
    default="float32",
    choices=["float32", "bfloat16"],
    help="activation dtype; float32 is the default for the reason in the docstring",
)
args = p.parse_args()

raw = json.load(open(args.ids))
ids = raw["ids"] if isinstance(raw, dict) else raw
if len(ids) < 2:
    raise SystemExit("need at least 2 ids")

model, tokenizer = load(args.model)
if args.dtype == "float32":
    model.set_dtype(mx.float32)

t0 = time.time()
nll: list[float] = []

# One pass over the whole sequence, chunked. Each chunk carries the previous
# chunk's last token as context so position p is always scored with the full
# prefix — the same conditioning the engine's sequential loop gives it.
# Deliberately NOT using a KV cache here: the reference should take the
# simplest correct path, since it is what the engine is graded against.
full = mx.array(ids)[None]
start = 0
while start < len(ids) - 1:
    end = min(start + args.chunk, len(ids) - 1)
    # Score positions [start, end): each needs logits conditioned on ids[0..p].
    logits = model(full[:, : end])  # [1, end, vocab]
    lp = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    tgt = mx.array(ids[start + 1 : end + 1])
    got = lp[0, start:end, :]
    picked = mx.take_along_axis(got, tgt[:, None], axis=-1)[:, 0]
    mx.eval(picked)
    nll.extend((-picked).tolist())
    start = end
    print(f"  {start}/{len(ids) - 1} scored ({time.time() - t0:.0f}s)", flush=True)

mean = sum(nll) / len(nll)
out = {
    "model": args.model,
    "dtype": args.dtype,
    "tokens_scored": len(nll),
    "mean_nll": mean,
    "perplexity": math.exp(mean),
    "nll": nll,
}
json.dump(out, open(args.out, "w"))
print(f"\n  tokens {len(nll)}  mean NLL {mean:.4f}  perplexity {math.exp(mean):.3f}")
print(f"  -> {args.out}")
