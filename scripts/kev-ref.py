"""KEV-REF — reference probabilities for the kev decision model, from kev's own
encoder and head over the MERGED backbone in torch.

The kev sibling of mlx-ref.py. For every record in the fixture file it runs
kev/model.py's `encode` + `rows_of` (so the token ids and readout positions
are kev's, not a re-derivation), the merged Qwen3 backbone in FLOAT32 on each
`state + branch` causal row, and PointerHead from head.pt. Dumps:

    <out>/meta.json      per record: rows (ids, decide, opts) and probs per question

kev's model.py is imported from a checkout of github.com/jaredpalmer/kev
(--kev-src); the commit is recorded in meta.json.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/kev-ref.py \
        --merged ~/dev/zero-tvm/.weights-local/kev-0.6b-merged-bf16 \
        --records ~/dev/zero-tvm/tests/fixtures/kev-records.json \
        --kev-src /tmp/kev --out /tmp/ref-kev
"""

import argparse, json, pathlib, subprocess, sys

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--merged", required=True)
p.add_argument("--records", required=True)
p.add_argument("--kev-src", required=True)
p.add_argument("--adapter", default="jaredpalmer/kev-0.6b", help="where head.pt comes from")
p.add_argument("--revision", default=None, help="adapter revision — MUST match the merge's (kev-4b@qwen3 vs its main branch "
               "ship heads of the same shape for different backbones, and the wrong one loads silently)")
p.add_argument("--out", required=True)
args = p.parse_args()

sys.path.insert(0, args.kev_src)
from kev.model import PointerHead, encode, rows_of  # noqa: E402

kev_commit = subprocess.run(["git", "-C", args.kev_src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

tok = AutoTokenizer.from_pretrained(args.merged)
# f32 for the same reason mlx-ref.py gives: a bf16 reference contributes more
# error than the engine it is grading.
lm = AutoModel.from_pretrained(args.merged, torch_dtype=torch.float32).eval()

head_pt = torch.load(pathlib.Path(snapshot_download(args.adapter, revision=args.revision)) / "head.pt", map_location="cpu", weights_only=False)
head = PointerHead(lm.config.hidden_size, dp=head_pt["head_dim"])
head.load_state_dict(head_pt["head"])
head.temperature = float(head_pt.get("temperature", 1.0))
head.eval()

records = json.load(open(args.records))
out = []
with torch.no_grad():
    for rec in records:
        enc = encode(tok, {"state": rec["state"], "questions": [{**q, "label": 0} for q in rec["questions"]]})
        S, Sp, rows = rows_of(enc)
        rec_out = {"state": S, "rows": [], "probs": []}
        for r in rows:
            ids = S + r["ids"]; pos = Sp + r["pos"]
            assert pos == list(range(len(ids))), "rows form must be plain sequential positions (option_isolation=False)"
            # Positions are asserted sequential above, so they are left implicit
            # (checked equivalent to passing them on the 4B).
            h = lm(input_ids=torch.tensor([ids])).last_hidden_state[0].float()
            d = len(S) + r["decide"]; opts = [len(S) + o for o in r["opts"]]
            z = head(h[d], h[torch.tensor(opts)])
            rec_out["rows"].append({"ids": ids, "decide": d, "opts": opts})
            rec_out["probs"].append(torch.softmax(z, -1).tolist())
        out.append(rec_out)
        print(f"record {len(out)}: state {len(S)} tok, {len(rows)} q ->", [[round(x, 3) for x in pr] for pr in rec_out["probs"]])

outdir = pathlib.Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
json.dump({"merged": args.merged, "adapter": args.adapter + (f"@{args.revision}" if args.revision else ""), "kev_commit": kev_commit,
           "temperature": head.temperature, "records": out}, open(outdir / "meta.json", "w"))
print("wrote", outdir / "meta.json")
