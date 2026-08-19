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
p.add_argument("--shapes", action="store_true",
               help="a battery of awkward conversations — untrimmed content, an assistant "
                    "turn containing </think>, a tool result that is not wholly a "
                    "<tool_response>. All reachable: the chat page stores the model's raw "
                    "output as history, and a room host renders a remote guest's history.")
p.add_argument("--plain", action="store_true",
               help="a plain multi-turn chat with NO tools — what the public chat page "
                    "sends. The tool case exercises none of that path, and the chat "
                    "page shares the same builder.")
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

# --plain: user/assistant alternation and no tools. The tool case can be
# byte-perfect while this is not — a finished round renders differently from an
# open one, and the chat page only ever sends finished rounds.
PLAIN = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
         {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
         {"role": "user", "content": "q3"}]

SHAPES = {
    "clean": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
              {"role": "user", "content": "again"}],
    "user trailing newline": [{"role": "user", "content": "hi\n"},
                              {"role": "assistant", "content": "hello"},
                              {"role": "user", "content": "again"}],
    "assistant leading space": [{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": " hello"},
                                {"role": "user", "content": "again"}],
    "assistant has </think>": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "<think>\nmusing\n</think>\n\nhello"},
                               {"role": "user", "content": "again"}],
    "bare </think> in reply": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "print </think> here"},
                               {"role": "user", "content": "again"}],
    "system trailing newline": [{"role": "system", "content": "sys\n"},
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"},
                                {"role": "user", "content": "again"}],
    "tool result, untrimmed": [{"role": "user", "content": "go"},
                               {"role": "assistant", "content": "", "tool_calls": [
                                   {"type": "function", "function": {"name": "read_file",
                                                                     "arguments": {"path": "a.ts"}}}]},
                               {"role": "tool", "content": "  padded  \n"}],
    "assistant is last turn": [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "partial"}],
}

msgs = PLAIN if a.plain else case.conversation(a.depth, wire=False)
tools = [] if a.plain else case.TOOLS

def wire(ms):
    """The OpenAI shape our host parses: `arguments` as a JSON STRING."""
    out = []
    for m in ms:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [{**c, "function": {**c["function"],
                                                  "arguments": json.dumps(c["function"]["arguments"])}}
                               for c in m["tool_calls"]]
        out.append(m)
    return out


def ours(ms, has_tools):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"messages": wire(ms), "tools": case.TOOLS if has_tools else [],
                   "param": a.param}, f)
        path = f.name
    r = subprocess.run(["node", "scripts/render-dump.mjs", path],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr.strip() or "node render-dump failed")
    return r.stdout


def theirs_of(ms, has_tools):
    return tok.apply_chat_template(
        ms, tools=case.TOOLS if has_tools else None, add_generation_prompt=True,
        enable_thinking=False, tokenize=False, **extra)


if a.shapes:
    bad = 0
    print(f"  {a.param}\n")
    for name, ms in SHAPES.items():
        has_tools = any(m["role"] == "tool" for m in ms)
        try:
            t = theirs_of(ms, has_tools)
        except Exception as e:                      # jinja raises on some shapes
            print(f"  {name:26} jinja RAISED: {type(e).__name__} — ours renders anyway")
            continue
        o = ours(ms, has_tools)
        ok = o == t
        bad += 0 if ok else 1
        print(f"  {name:26} {'same' if ok else 'DIFFERS'}")
        if not ok:
            i = next((i for i, (x, y) in enumerate(zip(t, o)) if x != y), min(len(t), len(o)))
            print(f"      jinja {t[i:i + 70]!r}\n      ours  {o[i:i + 70]!r}")
    raise SystemExit(1 if bad else 0)

theirs = tok.apply_chat_template(
    msgs, tools=tools or None,
    add_generation_prompt=True, enable_thinking=False, tokenize=False, **extra)

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump({"messages": PLAIN if a.plain else case.conversation(a.depth, wire=True),
               "tools": tools, "param": a.param}, f)
    case_path = f.name

r = subprocess.run(["node", "scripts/render-dump.mjs", case_path],
                   cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr.strip() or "node render-dump failed")
    sys.exit(1)
ours = r.stdout

if ours == theirs:
    print(f"IDENTICAL — {len(ours)} chars, {'plain chat' if a.plain else f'depth {a.depth}'}.\n"
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
