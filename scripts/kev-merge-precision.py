"""KEV-MERGE-PRECISION — does merging the LoRA into bf16 weights change the
decisions? kev serves the adapter UNMERGED (base(x) + lora(x), both computed);
scripts/kev-merge.py folds W + BA into one bf16 tensor, and any delta below the
base weight's bf16 ulp is rounded away there. Three arms over the fixture
records, all computed in f32:

    unmerged   base f32 + adapter at runtime (peft)       <- the served model
    merged16   W_bf16 + BA rounded to bf16, then upcast   <- what kev-merge.py wrote
    merged32   W_f32 + BA in f32                          <- the merge done right

    cd ~/dev/ml-research && uv run --with peft python ~/dev/zero-tvm/scripts/kev-merge-precision.py \
        --adapter jaredpalmer/kev-0.6b --records ~/dev/zero-tvm/tests/fixtures/kev-records.json --kev-src /tmp/kev
"""

import argparse, json, pathlib, sys

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--adapter", required=True)
p.add_argument("--revision", default=None)
p.add_argument("--records", required=True)
p.add_argument("--kev-src", required=True)
args = p.parse_args()
sys.path.insert(0, args.kev_src)
from kev.model import PointerHead, encode, rows_of  # noqa: E402

adir = pathlib.Path(snapshot_download(args.adapter, revision=args.revision))
hp = torch.load(adir / "head.pt", map_location="cpu", weights_only=False)
base, rev = hp["base"], hp["base_revision"]
tok = AutoTokenizer.from_pretrained(base, revision=rev)
head = PointerHead(hp["head"]["q.weight"].shape[1], dp=hp["head_dim"]); head.load_state_dict(hp["head"]); head.eval()
records = json.load(open(args.records))
rows_all = []
for rec in records:
    enc = encode(tok, {"state": rec["state"], "questions": [{**q, "label": 0} for q in rec["questions"]]})
    S, _, rows = rows_of(enc)
    rows_all.append([(S + r["ids"], len(S) + r["decide"], [len(S) + o for o in r["opts"]]) for r in rows])


def probs_of(lm):
    out = []
    with torch.no_grad():
        for rows in rows_all:
            out.append([])
            for ids, d, opts in rows:
                h = lm(input_ids=torch.tensor([ids])).last_hidden_state[0].float()
                out[-1].append(torch.softmax(head(h[d], h[torch.tensor(opts)]), -1).tolist())
    return out


print("unmerged (peft, f32)…", flush=True)
lm = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.float32)
# The adapter wraps the BARE backbone (see kev-merge.py): peft over lm.model,
# not over lm, or nothing matches and every arm below is the base model.
pm = PeftModel.from_pretrained(lm.model, str(adir)).eval()
assert sum(1 for k in pm.state_dict() if "lora_A" in k) > 0
ref = probs_of(pm)
print("merged32…", flush=True)
m32 = pm.merge_and_unload().eval()
p32 = probs_of(m32)
print("merged16…", flush=True)
m16 = m32.to(torch.bfloat16).to(torch.float32).eval()   # the bf16 round trip kev-merge.py's tensor took
p16 = probs_of(m16)


def report(name, got):
    agree = n = 0; dps = []
    for r, w in zip(got, ref):
        for pr, wr in zip(r, w):
            n += 1; agree += int(max(range(len(pr)), key=pr.__getitem__) == max(range(len(wr)), key=wr.__getitem__))
            dps.append(max(abs(a - b) for a, b in zip(pr, wr)))
    print(f"{name:<10} vs unmerged: argmax {agree}/{n}  mean|Δp| {sum(dps)/n:.4f}  max|Δp| {max(dps):.4f}")


report("merged32", p32)
report("merged16", p16)
for i in (1, 6, 7):
    print(f"rec{i+1} unmerged {[round(x,3) for x in ref[i][0]]} | merged16 {[round(x,3) for x in p16[i][0]]}")
