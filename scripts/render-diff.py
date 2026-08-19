"""Diff the prompt OUR host renders against the one the checkpoint's jinja does.

The reference test settled that the fault is on our side: mlx_lm emits the tool
call at 24k on the same checkpoint where we answer in prose. This finds WHERE,
by comparing the two prompts directly instead of reasoning about them. One
divergence has already been found and fixed this way round (tool results were
sent unwrapped, with no <tool_response> markers) and it changed the failure
without curing it, so there is at least one more.

Runs the node half in a subprocess: jinja needs transformers and our renderer is
TypeScript, so they cannot share a process.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/render-diff.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp

Loads the tokenizer only — no weights, no GPU, seconds not minutes.
"""
from __future__ import annotations

import argparse
import difflib
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import toolcall_case as case  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--param", default="qwen36q3", help="our ?model= param for the same checkpoint")
p.add_argument("--depth", type=int, default=0,
               help="tokens of padding. The divergence is structural, so 0 shows it "
                    "in a readable diff; raise it only to check length changes nothing.")
p.add_argument("--context", type=int, default=3)
# Several MLX checkpoints ship no tokenizer.json, so AutoTokenizer cannot load
# them at all — but the comparison only needs their TEMPLATE. Borrow a tokenizer
# that loads and hand it the other checkpoint's template text.
p.add_argument("--template", default=None,
               help="checkpoint dir to take chat_template from (default: --model's own)")
a = p.parse_args()

from transformers import AutoTokenizer  # noqa: E402

# --model always supplies the tokenizer; --template only overrides the template.
tok = AutoTokenizer.from_pretrained(a.model)


def chat_template(d: pathlib.Path):
    j = d / "chat_template.jinja"
    if j.exists():
        return j.read_text()
    cfg = json.loads((d / "tokenizer_config.json").read_text())
    t = cfg.get("chat_template")
    return t[0]["template"] if isinstance(t, list) else t


extra = {}
if a.template:
    extra["chat_template"] = chat_template(pathlib.Path(a.template))

theirs = tok.apply_chat_template(
    case.conversation(a.depth, wire=False), tools=case.TOOLS,
    add_generation_prompt=True, enable_thinking=False, tokenize=False, **extra)

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"messages": case.conversation(a.depth, wire=True),
               "tools": case.TOOLS, "param": a.param}, f)
    case_path = f.name

r = subprocess.run(["node", "scripts/render-dump.mjs", case_path],
                   cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr.strip() or "node render-dump failed")
    sys.exit(1)
ours = r.stdout

if ours == theirs:
    print(f"IDENTICAL — {len(ours)} chars, depth {a.depth}.\n"
          "The prompt is not where the difference is. Look at the engine next.")
    sys.exit(0)

# Line diff, since the divergences so far have been whole missing/extra lines.
d = list(difflib.unified_diff(
    theirs.splitlines(keepends=True), ours.splitlines(keepends=True),
    fromfile="jinja (the checkpoint's own template)", tofile="ours (the host)",
    n=a.context))
sys.stdout.writelines(d)

# Character counts make "one whitespace moved" vs "a block is missing" obvious
# at a glance, and the first divergence point localises it.
i = next((i for i, (x, y) in enumerate(zip(theirs, ours)) if x != y), min(len(theirs), len(ours)))
print(f"\n  jinja {len(theirs)} chars · ours {len(ours)} chars · first differ at char {i}")
print(f"  jinja: {theirs[i:i + 90]!r}")
print(f"  ours:  {ours[i:i + 90]!r}")
