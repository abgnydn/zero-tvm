"""YARN-REF — the inv_freq table yarn produces, from DeepSeek's own code.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/yarn-ref.py \
        --modeling /tmp/modeling_deepseek.py --out tests/fixtures/yarn-ref.json

Reimplementing yarn from the paper and then testing our implementation against
our own reading of it proves only that we are consistent. So the four helpers
(correction dim, correction range, ramp mask, mscale) are EXECUTED out of the
checkpoint's shipped modeling_deepseek.py — 80 KB fetched once, and the part
that decides the answer is the vendor's.

The assembly below is transcribed from _set_cos_sin_cache in the same file. It
is five lines and visible in the source; the fiddly parts are the helpers.

Emits a fixture small enough to commit (a few hundred floats), so the test does
not need the model, the internet, or torch.
"""

import argparse
import json
import math
import pathlib
import re

import torch  # only because the vendor helpers use it

p = argparse.ArgumentParser()
p.add_argument("--modeling", required=True, help="path to modeling_deepseek.py")
p.add_argument("--out", required=True)
p.add_argument("--dim", type=int, default=64, help="qk_rope_head_dim")
p.add_argument("--base", type=float, default=10000.0, help="rope_theta")
p.add_argument("--factor", type=float, default=40.0)
p.add_argument("--beta-fast", type=float, default=32.0)
p.add_argument("--beta-slow", type=float, default=1.0)
p.add_argument("--original-max", type=int, default=4096)
p.add_argument("--mscale", type=float, default=0.707)
p.add_argument("--mscale-all-dim", type=float, default=0.707)
args = p.parse_args()

src = pathlib.Path(args.modeling).read_text()
start = src.index("def yarn_find_correction_dim")
end = src.index("class DeepseekV2YarnRotaryEmbedding")
ns = {"math": math, "torch": torch}
exec(compile(src[start:end], args.modeling, "exec"), ns)  # noqa: S102 — vendor helpers, read above
for fn in ("yarn_find_correction_dim", "yarn_find_correction_range",
           "yarn_get_mscale", "yarn_linear_ramp_mask"):
    if fn not in ns:
        raise SystemExit(f"{fn} not found — did modeling_deepseek.py change shape?")
print(f"executed {len(re.findall(r'^def ', src[start:end], re.M))} vendor helpers")

dim, base = args.dim, args.base
freq_extra = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
freq_inter = 1.0 / (args.factor * base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
low, high = ns["yarn_find_correction_range"](args.beta_fast, args.beta_slow, dim, base, args.original_max)
inv_freq_mask = 1.0 - ns["yarn_linear_ramp_mask"](low, high, dim // 2).to(torch.float32)
inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

# The logit scale, applied squared — see DeepseekV2Attention.__init__.
mscale = ns["yarn_get_mscale"](args.factor, args.mscale_all_dim) if args.mscale_all_dim else 1.0
attn_scale = mscale * mscale

out = pathlib.Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "source": "modeling_deepseek.py (DeepSeek-V2-Lite), helpers executed verbatim",
    "dim": dim, "base": base, "factor": args.factor,
    "betaFast": args.beta_fast, "betaSlow": args.beta_slow,
    "originalMaxPositionEmbeddings": args.original_max,
    "mscale": args.mscale, "mscaleAllDim": args.mscale_all_dim,
    "correctionRange": [low, high],
    "attnScale": attn_scale,
    "invFreq": [float(x) for x in inv_freq],
}, indent=1) + "\n")
print(f"low={low} high={high} attn_scale={attn_scale:.6f}")
print(f"wrote {out} — {len(inv_freq)} frequencies")
