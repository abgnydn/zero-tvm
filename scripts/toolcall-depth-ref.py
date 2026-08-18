"""
Does tool-call emission survive depth on the REFERENCE, or only on ours?

The finding this exists to test: at ~24k tokens of history our engine's
qwen36q3 reads the right files, computes the right answer, and then answers in
PROSE instead of emitting the tool call — "7 * 12 = 84", correct, unusable to a
client. At ~500 tokens the same model emits the call every time. int8 KV is not
the cause (f16 fails identically).

That leaves two possibilities and only a reference can separate them:
  A. the model loses the tool-call format at depth — nothing to fix in the engine
  B. OUR prompt rendering decays with depth — a bug, and ours

So this asks mlx_lm the same question with the SAME conversation, built by the
checkpoint's OWN chat template via apply_chat_template(tools=...). If mlx_lm
emits a <tool_call> block where we emit prose, the fault is on our side.

Needs Metal, so it runs in your shell:

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/toolcall-depth-ref.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp
"""
from __future__ import annotations

import argparse
import json

from mlx_lm import load, generate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--depths", default="0,8000,24000", help="approx tokens of prior conversation")
p.add_argument("--max-tokens", type=int, default=120)
a = p.parse_args()

TOOLS = [
    {"type": "function", "function": {
        "name": "attempt_completion",
        "description": "Report the final answer and finish the task",
        "parameters": {"type": "object", "properties": {"result": {"type": "string"}},
                       "required": ["result"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read one file by path",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
]

SYSTEM = ("You are a coding agent. Call exactly one tool per step. Use what each tool "
          "returns. When you know the answer, call attempt_completion. "
          "Never read a file you have already read.")


def padding(target: int):
    """Inert prior conversation — must not contain or hint at the answer, or this
    measures retrieval instead of whether the tool format survives length."""
    msgs, approx, i = [], 0, 0
    while approx < target:
        q = f"Step {i}: summarise what changed in release 2.{i % 9}.{i % 5}."
        ans = (f"Release 2.{i % 9}.{i % 5} adjusted logging thresholds, renamed two internal "
               "helpers, and left public behaviour unchanged. No configuration keys moved, and "
               "the deployment procedure is identical to the previous release. Nothing here "
               "affects pipeline capacity or the dimension constants used elsewhere.")
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
        approx += (len(q) + len(ans)) // 4
        i += 1
    return msgs


# The state our engine reaches right before it fails: three files read, the
# arithmetic available. The only correct next move is attempt_completion("84").
TAIL = [
    {"role": "user", "content": "What number does capacity() return? Read only the files you "
                                "need, then call attempt_completion with just the number."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "read_file", "arguments": json.dumps({"path": "src/pipeline.ts"})}}]},
    {"role": "tool", "tool_call_id": "c1",
     "content": 'import { WIDTH } from "./dims.ts"\nimport { scale } from "./scale.ts"\n'
                "export function capacity() { return scale(WIDTH) }"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function",
         "function": {"name": "read_file", "arguments": json.dumps({"path": "src/dims.ts"})}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "export const WIDTH = 12"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c3", "type": "function",
         "function": {"name": "read_file", "arguments": json.dumps({"path": "src/scale.ts"})}}]},
    {"role": "tool", "tool_call_id": "c3",
     "content": "export const FACTOR = 7\nexport function scale(n: number) { return n * FACTOR }"},
]

model, tokenizer = load(a.model)
print(f"{a.model}\n")
print(f"  {'depth':>8} {'prompt tok':>11}  emitted a tool call?   what it produced")

for d in [int(x) for x in a.depths.split(",")]:
    msgs = [{"role": "system", "content": SYSTEM}] + padding(d) + TAIL
    prompt = tokenizer.apply_chat_template(msgs, tools=TOOLS, add_generation_prompt=True)
    ntok = len(prompt)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=a.max_tokens, verbose=False)
    called = "<tool_call>" in out or "attempt_completion" in out
    first = out.strip().replace("\n", " ")[:90]
    print(f"  {d:>8} {ntok:>11}  {'YES' if called else 'NO ':<21}  {first!r}")

print("""
  Reading this:
    reference emits the call at depth, ours does not  -> OUR rendering decays; a bug
    reference ALSO stops emitting at depth            -> the model; nothing to fix here
    both emit it                                      -> the difference is our tool
                                                         DIALECT or where the tools
                                                         block sits, worth diffing next
""")
