"""KEV-REF-MLX — the same reference as kev-ref.py, but the backbone is mlx_lm's
forward over the QUANTIZED checkpoint the engine loads. kev-ref.py grades
against unquantized f32; this arm grades against the exact 4-bit weights, so
the two together split "engine bug" from "4-bit loss".

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/kev-ref-mlx.py \
        --mlx ~/dev/zero-tvm/.weights-local/kev-0.6b-mlx-4bit \
        --records ~/dev/zero-tvm/tests/fixtures/kev-records.json \
        --kev-src /tmp/kev --out /tmp/ref-kev-mlx
"""

import argparse, json, pathlib, subprocess, sys

import mlx.core as mx
import numpy as np
import torch
from huggingface_hub import snapshot_download
from mlx_lm import load
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--mlx", required=True)
p.add_argument("--records", required=True)
p.add_argument("--kev-src", required=True)
p.add_argument("--adapter", default="jaredpalmer/kev-0.6b")
p.add_argument("--revision", default=None, help="adapter revision — must match the merge's (see kev-ref.py)")
p.add_argument("--out", required=True)
args = p.parse_args()

sys.path.insert(0, args.kev_src)
from kev.model import PointerHead, encode, rows_of  # noqa: E402

kev_commit = subprocess.run(["git", "-C", args.kev_src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
tok = AutoTokenizer.from_pretrained(args.mlx)
model, _ = load(args.mlx)
model.set_dtype(mx.float32)   # non-quantized tensors in f32; packed weights stay 4-bit
# The hidden model (self.norm(h) == HF last_hidden_state): `model.model` on a
# plain text checkpoint, one level down on mlx_lm's multimodal-shaped qwen3_5.
hidden_model = getattr(model, "model", None) or model.language_model.model

head_pt = torch.load(pathlib.Path(snapshot_download(args.adapter, revision=args.revision)) / "head.pt", map_location="cpu", weights_only=False)
head = PointerHead(head_pt["head"]["q.weight"].shape[1], dp=head_pt["head_dim"])
head.load_state_dict(head_pt["head"]); head.temperature = float(head_pt.get("temperature", 1.0)); head.eval()

records = json.load(open(args.records))
out = []
with torch.no_grad():
    for rec in records:
        enc = encode(tok, {"state": rec["state"], "questions": [{**q, "label": 0} for q in rec["questions"]]})
        S, Sp, rows = rows_of(enc)
        rec_out = {"state": S, "rows": [], "probs": []}
        for r in rows:
            ids = S + r["ids"]
            # mlx_lm: model.model(ids) is self.norm(h) — HF's last_hidden_state.
            h = torch.from_numpy(np.array(hidden_model(mx.array([ids]))[0].astype(mx.float32)))
            d = len(S) + r["decide"]; opts = [len(S) + o for o in r["opts"]]
            z = head(h[d], h[torch.tensor(opts)])
            rec_out["rows"].append({"ids": ids, "decide": d, "opts": opts})
            rec_out["probs"].append(torch.softmax(z, -1).tolist())
        out.append(rec_out)
        print(f"record {len(out)}:", [[round(x, 3) for x in pr] for pr in rec_out["probs"]])

outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
json.dump({"merged": args.mlx, "adapter": args.adapter + (f"@{args.revision}" if args.revision else ""), "kev_commit": kev_commit,
           "temperature": head.temperature, "records": out}, open(outdir / "meta.json", "w"))
print("wrote", outdir / "meta.json")
