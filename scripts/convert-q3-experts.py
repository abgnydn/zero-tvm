#!/usr/bin/env python3
"""
Requantize the EXPERT tensors of the Qwen3.6-35B-A3B MLX checkpoint to 3-bit,
leaving everything else at its shipped precision.

Why only experts: they are 16.2 GB of the 19.7 GB resident set, and decode on a
32 GB machine dies the moment the working set exceeds physical RAM (measured:
GPU process killed mid-prefill). 3-bit experts bring resident to ~15.7 GB —
comfortable headroom — while attention/GDN/lm_head keep their 4-bit quality.
The shared_expert is included because the engine stacks it as expert index E:
one stacked tensor must be one format.

Requantizing from the 4-bit checkpoint (not the bf16 original) adds only a
small extra error on top of 3-bit's own: snapping an already-quantized value to
a coarser grid mostly lands where the original would have. The alternative is a
~70 GB download.

Writes a sibling directory with the same file names and structure, so the same
loader/mirror machinery serves it; config.json gains per-tensor quantization
overrides in the exact per-path style the checkpoint already uses for its
8-bit router, so mlx_lm remains able to load the result as a REFERENCE.

  uv run python scripts/convert-q3-experts.py          # from ~/dev/ml-research
"""
import json, os, shutil, sys, time

import mlx.core as mx

SRC = "/Users/ahmetbarisgunaydin/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-4bit"
DST = "/Users/ahmetbarisgunaydin/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp"
G = 64
# A tensor is requantized iff its path contains one of these.
EXPERT_MARKS = (".mlp.switch_mlp.", ".mlp.shared_expert.")


def is_expert_weight(name: str) -> bool:
    return any(m in name for m in EXPERT_MARKS) and name.endswith(".weight")


def main() -> None:
    os.makedirs(DST, exist_ok=True)
    index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = sorted(set(weight_map.values()))

    new_sizes: dict[str, int] = {}
    overrides: dict[str, dict] = {}
    t0 = time.time()
    for shard in shards:
        print(f"[{time.time()-t0:6.0f}s] {shard} …", flush=True)
        tensors = mx.load(os.path.join(SRC, shard))
        out = {}
        for name, t in tensors.items():
            if is_expert_weight(name):
                base = name[: -len(".weight")]
                s = tensors[base + ".scales"]
                b = tensors[base + ".biases"]
                # dequantize at the checkpoint's 4-bit, requantize at 3
                deq = mx.dequantize(t, s, b, group_size=G, bits=4)
                w3, s3, b3 = mx.quantize(deq, group_size=G, bits=3)
                out[name] = w3
                out[base + ".scales"] = s3.astype(s.dtype)
                out[base + ".biases"] = b3.astype(b.dtype)
                overrides[base] = {"group_size": G, "bits": 3}
                mx.eval(w3, s3, b3)
                del deq, w3, s3, b3
            elif any(m in name for m in EXPERT_MARKS) and (
                name.endswith(".scales") or name.endswith(".biases")
            ):
                continue  # written alongside their .weight above
            else:
                out[name] = t
        mx.save_safetensors(os.path.join(DST, shard), out,
                            metadata={"format": "mlx", "requantized": "experts->3bit from 4bit"})
        for name in out:
            pass
        # sizes for the new index
        del tensors, out
        mx.clear_cache()

    # GUARD. EXPERT_MARKS are the FUSED expert layout (".mlp.switch_mlp.").
    # A checkpoint whose experts are named ".mlp.experts.<N>." matches none of
    # them, and everything above still succeeds: every tensor is copied
    # through, a valid checkpoint is written, and the metadata claims
    # "requantized: experts->3bit from 4bit". The result is a byte-identical
    # copy that a perplexity A/B reports as "no significant difference" —
    # which reads as "3-bit is fine" rather than "nothing happened". Both
    # layouts are live in one family: mlx-community's OLMoE-1B-7B-0125-4bit is
    # switch_mlp, its Instruct sibling is .mlp.experts.
    # scripts/requantize.py handles both; this script stays 35B-specific.
    if not overrides:
        raise SystemExit(
            f"REFUSING: no tensor in {SRC} matched {EXPERT_MARKS}.\n"
            "  Nothing was requantized. Use scripts/requantize.py, which knows\n"
            "  the .mlp.experts.<N>. layout too and counts what it converts.")

    # Rebuild the index from the written shards (offsets changed).
    new_map = {}
    for shard in shards:
        st = mx.load(os.path.join(DST, shard))
        for name in st:
            new_map[name] = shard
        del st
        mx.clear_cache()
    total = sum(os.path.getsize(os.path.join(DST, s)) for s in shards)
    json.dump({"metadata": {"total_size": total}, "weight_map": new_map},
              open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)

    # config.json: keep everything, add the per-path overrides (same mechanism
    # the checkpoint already uses for its 8-bit router rows).
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    qz = cfg.get("quantization", {})
    qz.update(overrides)
    cfg["quantization"] = qz
    if "quantization_config" in cfg:
        cfg["quantization_config"] = qz
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)

    # Tokenizer + template files: copy verbatim.
    for f in os.listdir(SRC):
        if f.endswith((".json", ".jinja", ".txt")) and f not in (
            "config.json", "model.safetensors.index.json",
        ) and not f.startswith("model-"):
            shutil.copy2(os.path.join(SRC, f), os.path.join(DST, f))

    print(f"done in {time.time()-t0:.0f}s -> {DST}  ({total/1e9:.2f} GB, "
          f"{len(overrides)} tensors requantized)")


if __name__ == "__main__":
    sys.exit(main())
