"""KEV-MERGE — turn a kev checkpoint (LoRA adapter + pointer head on a Qwen3
base) into what the engine loads: a merged, MLX affine-quantized (group 64)
checkpoint plus a small sidecar with the pointer head.

    cd ~/dev/ml-research && uv run --with peft python ~/dev/zero-tvm/scripts/kev-merge.py \
        --out ~/dev/zero-tvm/.weights-local/kev-0.6b-mlx-4bit
    uv run --with peft python ~/dev/zero-tvm/scripts/kev-merge.py \
        --adapter jaredpalmer/kev-4b --revision qwen3 --out ~/dev/zero-tvm/.weights-local/kev-4b-mlx-4bit

Also writes <out>/../<name>-merged-bf16 (the un-quantized merge) so the torch
reference (scripts/kev-ref.py) scores the SAME merged weights.

Which bit width a checkpoint survives is MEASURED, not assumed:
scripts/kev-quant-sweep.py. kev-0.6b keeps its decisions at 8 bits and loses
8 of 21 argmaxes at 4 (2026-09-22).
"""

import argparse, json, pathlib, subprocess, sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--adapter", default="jaredpalmer/kev-0.6b")
p.add_argument("--revision", default=None, help="adapter revision, e.g. qwen3 for kev-4b's Qwen3 generation")
p.add_argument("--out", required=True)
p.add_argument("--bits", type=int, default=4)
args = p.parse_args()

out = pathlib.Path(args.out).expanduser()
merged_dir = out.parent / (out.name.split("-mlx-")[0] + "-merged-bf16")
adapter_dir = pathlib.Path(snapshot_download(args.adapter, revision=args.revision))
train_cfg = json.load(open(adapter_dir / "training_config.json"))
head = torch.load(adapter_dir / "head.pt", map_location="cpu", weights_only=False)
base, rev = head["base"], head["base_revision"]
assert head["head_dim"] == 256 and not head["option_isolation"], head
print("base", base, "@", rev, "| lora r", head["lora"])

if not (merged_dir / "config.json").exists():
    import warnings
    tok = AutoTokenizer.from_pretrained(base, revision=rev)
    lm = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.bfloat16)
    # kev trains the LoRA on the BARE backbone (kev/model.py wraps `self.lm`),
    # so the adapter's keys are `base_model.model.layers.N…`. Wrapping the
    # CausalLM instead puts peft's keys at `base_model.model.model.layers.N…`,
    # peft matches nothing, WARNS, and hands back the base model unchanged —
    # which is exactly what happened on 2026-09-22: every downstream number
    # (parity, the quantization sweep) was measured on base Qwen3 + kev's head.
    # Wrap the inner model, and make an unmatched key a hard failure.
    # The sentinel is the first Linear the adapter targets — layer 0 of a
    # Qwen3.5 hybrid is a DeltaNet layer with no self_attn at all.
    targets = json.load(open(adapter_dir / "adapter_config.json"))["target_modules"]
    sentinel = next(n for n, m in lm.model.named_modules()
                    if isinstance(m, torch.nn.Linear) and n.split(".")[-1] in targets)
    before = dict(lm.model.named_modules())[sentinel].weight.detach().float().clone()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pm = PeftModel.from_pretrained(lm.model, str(adapter_dir))
    bad = [str(w.message)[:200] for w in caught if "adapter keys" in str(w.message)]
    if bad:
        raise SystemExit("peft did not load every adapter tensor:\n  " + "\n  ".join(bad))
    lm.model = pm.merge_and_unload()
    after = dict(lm.model.named_modules())[sentinel].weight.detach().float()
    rel = ((after - before).norm() / before.norm()).item()
    if rel == 0.0:
        raise SystemExit(f"merge changed nothing: {sentinel} is byte-identical to the base")
    print(f"merged ({sentinel} moved by {rel:.2e} relative) ->", merged_dir)
    merged_dir.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(merged_dir, safe_serialization=True)
    tok.save_pretrained(merged_dir)

# transformers >= 5 writes rope_theta under `rope_parameters`; mlx_lm's qwen3
# ModelArgs still wants the flat key. Same value, both spellings.
cfg_path = merged_dir / "config.json"
cfg = json.load(open(cfg_path))
if "rope_theta" not in cfg:
    cfg["rope_theta"] = cfg["rope_parameters"]["rope_theta"]
    cfg.setdefault("rope_scaling", None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)

# transformers saves a text-only Qwen3.5 as model_type `qwen3_5_text`, which
# mlx_lm does not know; its `qwen3_5` loader takes the same flat config. torch
# must keep the original (the multimodal class would look for language_model.*
# keys), so mlx_lm converts from a sibling dir: symlinked weights + tokenizer,
# one patched config.json.
mlx_src = merged_dir
if cfg["model_type"].endswith("_text"):
    mlx_src = merged_dir.parent / (merged_dir.name + "-mlxsrc")
    mlx_src.mkdir(exist_ok=True)
    for f in merged_dir.iterdir():
        if f.name != "config.json" and not (mlx_src / f.name).exists():
            (mlx_src / f.name).symlink_to(f)
    json.dump({**cfg, "model_type": cfg["model_type"][: -len("_text")]}, open(mlx_src / "config.json", "w"), indent=2)

# Quantize FIRST: mlx_lm.convert refuses an existing --mlx-path, so the head
# sidecar (below) goes in only once the directory is mlx_lm's.
if not (out / "model.safetensors.index.json").exists() and not (out / "model.safetensors").exists():
    subprocess.check_call([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", str(mlx_src),
                           "--mlx-path", str(out), "-q", "--q-bits", str(args.bits), "--q-group-size", "64"])
    print("quantized ->", out)

# Pointer head: q/k Linear(d -> 256) with bias, fp32. Stored next to the
# quantized checkpoint so the engine fetches it from the same directory.
hw = {k: v.float().contiguous() for k, v in head["head"].items()}
save_file(hw, str(out / "kev_head.safetensors"), metadata={
    "scale": str(1 / 16), "temperature": str(head.get("temperature", 1.0)),
    "adapter": args.adapter + (f"@{args.revision}" if args.revision else ""), "base": base, "base_revision": rev,
})
print("head ->", out / "kev_head.safetensors", {k: tuple(v.shape) for k, v in hw.items()})
print(sorted(p.name for p in out.iterdir()))
