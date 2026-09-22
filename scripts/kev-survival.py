"""KEV-SURVIVAL — what the SHIPPED 4-bit file keeps of the checkpoint's own
decisions: kev's f32 torch forward on the bf16 merge (scripts/kev-ref.py)
against mlx_lm's forward on the converted 4-bit file (scripts/kev-ref-mlx.py),
argmax agreement and max |Δp| over the fixture questions. Both dumps are
committed under docs/kev-parity/refs/, so this needs no GPU and no model.

    python3 scripts/kev-survival.py docs/kev-parity/refs/kev-f32.json docs/kev-parity/refs/kev-mlx4.json
    python3 scripts/kev-survival.py <f32.json> <mlx4.json> --kept      # just the integer
"""

import json, sys

a = json.load(open(sys.argv[1]))["records"]
b = json.load(open(sys.argv[2]))["records"]
kept = n = 0; worst = 0.0; flips = []
for i, (x, y) in enumerate(zip(a, b)):
    for k, (p, q) in enumerate(zip(x["probs"], y["probs"])):
        n += 1
        if max(range(len(p)), key=p.__getitem__) == max(range(len(q)), key=q.__getitem__): kept += 1
        else: flips.append(f"rec{i + 1} q{k + 1}")
        worst = max(worst, max(abs(u - v) for u, v in zip(p, q)))
if "--kept" in sys.argv: print(kept)
else: print(f"argmax kept {kept}/{n}, max |Δp| {worst:.3f}, flips: {', '.join(flips) or 'none'}")
