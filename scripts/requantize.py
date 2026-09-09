#!/usr/bin/env python3
"""Requantize an MLX checkpoint — the general form of convert-q3-experts.py.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/requantize.py \
        --src <4-bit checkpoint> --dst <out> --bits 3 --scope experts

    --scope experts   only the MoE expert stacks (what qwen36q3 ships)
    --scope all       every quantized tensor except embeddings and lm_head
    --scope mlp       dense FFN projections only

Exists for two reasons.

ONE: the quality suite needs a KNOWN-DEGRADED build to prove it can see damage.
A harness that has only ever passed proves nothing, and the cheapest degraded
build is a requantized copy of something already on disk.

TWO: convert-q3-experts.py can silently do nothing. It selects experts by the
literal strings ".mlp.switch_mlp." and ".mlp.shared_expert.", which is MLX's
FUSED expert layout. Checkpoints converted a different way name their experts
".mlp.experts.<N>." instead, and on one of those the script matches zero
tensors, copies every weight through unchanged, writes a valid checkpoint,
stamps it `requantized: experts->3bit from 4bit`, and exits 0. The A/B would
then report "NOT DISTINGUISHABLE" — true, and completely misleading about why.
Both layouts are live in the same model family: mlx-community's
OLMoE-1B-7B-0125-4bit uses switch_mlp (144 tensors) and its Instruct sibling
uses .mlp.experts. (9216).

So this handles both layouts and it COUNTS. Zero converted tensors is a hard
failure, never a quiet success.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import mlx.core as mx

p = argparse.ArgumentParser(description="requantize an MLX checkpoint")
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--bits", type=int, default=3, choices=[2, 3, 4, 6, 8])
p.add_argument("--group", type=int, default=64)
p.add_argument("--scope", default="experts", choices=["experts", "mlp", "all"])
args = p.parse_args()

# Both MLX expert layouts. `switch_mlp` is the fused stack (one tensor holding
# every expert); `.mlp.experts.<N>.` is one tensor per expert. A checkpoint has
# one or the other, never both, and which one you get depends on the converter
# version rather than on the model.
EXPERT_MARKS = (".mlp.switch_mlp.", ".mlp.shared_expert.", ".mlp.experts.")
MLP_MARKS = (".mlp.gate_proj.", ".mlp.up_proj.", ".mlp.down_proj.")
# Never touched: the output head and the embedding table are the two places
# where a quantization error turns into visible garbage fastest, and neither is
# what a MoE expert-precision experiment is about.
NEVER = ("embed_tokens", "lm_head")


def in_scope(name: str) -> bool:
    if any(n in name for n in NEVER):
        return False
    if args.scope == "all":
        return True
    if args.scope == "experts":
        return any(m in name for m in EXPERT_MARKS)
    return any(m in name for m in MLP_MARKS)


os.makedirs(args.dst, exist_ok=True)
shards = sorted(os.path.basename(f) for f in glob.glob(os.path.join(args.src, "*.safetensors")))
if not shards:
    sys.exit(f"no .safetensors in {args.src}")

overrides: dict[str, dict] = {}
converted = 0
copied = 0

for shard in shards:
    tensors = mx.load(os.path.join(args.src, shard))
    out = {}
    for name, t in tensors.items():
        base = name[: -len(".weight")] if name.endswith(".weight") else None
        is_quantized = base is not None and (base + ".scales") in tensors
        if is_quantized and in_scope(name):
            src_bits = 4  # every checkpoint this is pointed at is 4-bit
            deq = mx.dequantize(t, tensors[base + ".scales"], tensors[base + ".biases"],
                                group_size=args.group, bits=src_bits)
            w, s, b = mx.quantize(deq, group_size=args.group, bits=args.bits)
            out[name] = w
            out[base + ".scales"] = s.astype(tensors[base + ".scales"].dtype)
            out[base + ".biases"] = b.astype(tensors[base + ".biases"].dtype)
            overrides[base] = {"group_size": args.group, "bits": args.bits}
            converted += 1
            mx.eval(w, s, b)
            del deq, w, s, b
        elif name.endswith((".scales", ".biases")) and name.rsplit(".", 1)[0] in overrides:
            continue  # already written beside its .weight
        else:
            out[name] = t
            copied += 1
    mx.save_safetensors(
        os.path.join(args.dst, shard), out,
        metadata={"format": "mlx",
                  "requantized": f"{args.scope}->{args.bits}bit group{args.group}"})
    print(f"  {shard}: {converted} converted so far", flush=True)
    del tensors, out
    mx.clear_cache()

# THE GUARD. Everything above succeeds on a checkpoint whose experts are named
# differently — it just copies. A degraded build that is byte-identical to its
# parent makes the A/B report "no significant difference", which reads as
# "3-bit is fine" rather than "nothing was quantized".
if converted == 0:
    shutil.rmtree(args.dst, ignore_errors=True)
    sys.exit(
        f"REFUSING: --scope {args.scope} matched ZERO tensors in {args.src}.\n"
        f"  Nothing would have been requantized and the output would be a copy.\n"
        f"  Expert layouts known here: {', '.join(EXPERT_MARKS)}\n"
        f"  Inspect the real names with:\n"
        f"    python -c \"import json;print(list(json.load(open('{args.src}/"
        f"model.safetensors.index.json'))['weight_map'])[:20])\""
    )

new_map = {}
for shard in shards:
    st = mx.load(os.path.join(args.dst, shard))
    for name in st:
        new_map[name] = shard
    del st
    mx.clear_cache()
total = sum(os.path.getsize(os.path.join(args.dst, s)) for s in shards)
json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
          open(os.path.join(args.dst, "model.safetensors.index.json"), "w"), indent=1)

cfg = json.load(open(os.path.join(args.src, "config.json")))
q = dict(cfg.get("quantization", {}))
q.update(overrides)
cfg["quantization"] = q
json.dump(cfg, open(os.path.join(args.dst, "config.json"), "w"), indent=1)

for f in glob.glob(os.path.join(args.src, "*.json")) + glob.glob(os.path.join(args.src, "*.txt")):
    if os.path.basename(f) not in ("config.json", "model.safetensors.index.json"):
        shutil.copy(f, args.dst)

src_gb = sum(os.path.getsize(os.path.join(args.src, s)) for s in shards) / 2**30
print(f"\n  {converted} tensors requantized to {args.bits}-bit, {copied} copied")
print(f"  {src_gb:.2f} GB -> {total / 2**30:.2f} GB")
print(f"  -> {args.dst}")
