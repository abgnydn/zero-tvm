"""KEV-QUANT-SWEEP — which quantization of the kev backbone still decides like
kev? In-memory: load the UNQUANTIZED MLX conversion once, quantize it several
ways with mlx.nn.quantize, run kev's rows through each, score the pointer head
and compare every arm with the f32 torch reference (kev-ref.py's meta.json).

Reported per arm: argmax agreement over all questions, mean and max |Δp|.
The engine can load affine 4-bit / group 64 (and 3-bit expert stacks) today;
every other arm here is a measurement of what a loader change would buy.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/kev-quant-sweep.py \
        --mlx-bf16 ~/dev/zero-tvm/.weights-local/kev-0.6b-mlx-bf16 \
        --ref /tmp/ref-kev --records ~/dev/zero-tvm/tests/fixtures/kev-records.json --kev-src /tmp/kev
"""

import argparse, copy, json, pathlib, sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from huggingface_hub import snapshot_download
from mlx_lm import load
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--mlx-bf16", required=True)
p.add_argument("--ref", required=True)
p.add_argument("--records", required=True)
p.add_argument("--kev-src", required=True)
p.add_argument("--adapter", default="jaredpalmer/kev-0.6b")
p.add_argument("--revision", default=None)
p.add_argument("--quick", action="store_true", help="only the pipeline check, 8-bit and 4-bit g64")
args = p.parse_args()

sys.path.insert(0, args.kev_src)
from kev.model import PointerHead, encode, rows_of  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.mlx_bf16)
ref = json.load(open(pathlib.Path(args.ref) / "meta.json"))["records"]
records = json.load(open(args.records))
head_pt = torch.load(pathlib.Path(snapshot_download(args.adapter, revision=args.revision)) / "head.pt", map_location="cpu", weights_only=False)
head = PointerHead(head_pt["head"]["q.weight"].shape[1], dp=head_pt["head_dim"])
head.load_state_dict(head_pt["head"]); head.temperature = float(head_pt.get("temperature", 1.0)); head.eval()

# Rows once; every arm sees identical ids.
rows_all = []
for rec in records:
    enc = encode(tok, {"state": rec["state"], "questions": [{**q, "label": 0} for q in rec["questions"]]})
    S, _, rows = rows_of(enc)
    rows_all.append([(S + r["ids"], len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for r in rows])


def score(model):
    hidden_model = getattr(model, "model", None) or model.language_model.model
    agree = n = 0; dps = []
    with torch.no_grad():
        for ri, rows in enumerate(rows_all):
            for qi, (ids, d, opts) in enumerate(rows):
                h = torch.from_numpy(np.array(hidden_model(mx.array([ids]))[0].astype(mx.float32)))
                pr = torch.softmax(head(h[d], h[torch.tensor(opts)]), -1).tolist()
                want = ref[ri]["probs"][qi]
                agree += int(max(range(len(pr)), key=pr.__getitem__) == max(range(len(want)), key=want.__getitem__)); n += 1
                dps.append(max(abs(a - b) for a, b in zip(pr, want)))
    return agree, n, float(np.mean(dps)), float(np.max(dps))


def fresh():
    m, _ = load(args.mlx_bf16)
    m.set_dtype(mx.float32)
    return m


def arm(name, bits=None, group=64, skip=lambda path, mod: False):
    m = fresh()
    if bits is not None:
        nn.quantize(m, group_size=group, bits=bits,
                    class_predicate=lambda path, mod: isinstance(mod, (nn.Linear, nn.Embedding)) and not skip(path, mod))
        mx.eval(m.parameters())
    a, n, mean_dp, max_dp = score(m)
    print(f"{name:<44} argmax {a:>2}/{n}  mean|Δp| {mean_dp:.3f}  max|Δp| {max_dp:.3f}", flush=True)


arm("bf16 conversion, f32 compute (pipeline check)")
arm("8-bit / g64", bits=8)
if args.quick:
    arm("4-bit / g64 (what ships)", bits=4)
    sys.exit(0)
arm("6-bit / g64", bits=6)
arm("4-bit / g32", bits=4, group=32)
arm("4-bit / g64 (what ships)", bits=4)
arm("4-bit / g64, embedding kept", bits=4, skip=lambda p, m: isinstance(m, nn.Embedding))
arm("4-bit / g64, embedding + lm_head kept", bits=4, skip=lambda p, m: isinstance(m, nn.Embedding) or "lm_head" in p)
arm("4-bit / g64, embedding + layers 0-1 kept", bits=4,
    skip=lambda p, m: isinstance(m, nn.Embedding) or any(f"layers.{i}." in p for i in (0, 1)))
arm("4-bit / g64, embedding + last 2 layers kept", bits=4,
    skip=lambda p, m: isinstance(m, nn.Embedding) or any(f"layers.{i}." in p for i in (26, 27)))
arm("4-bit / g64, attention kept (MLP only)", bits=4, skip=lambda p, m: "self_attn" in p or isinstance(m, nn.Embedding))
arm("4-bit / g64, MLP kept (attention only)", bits=4, skip=lambda p, m: "mlp" in p or isinstance(m, nn.Embedding))
