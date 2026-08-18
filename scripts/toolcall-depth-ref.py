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

DEPTH 0 IS THE CONTROL. Our engine emits the call there every time, so if the
reference does not, this script is measuring itself and the deeper rows say
nothing. The first version failed exactly that way: Qwen3.6's template invites
reasoning BEFORE the call, --max-tokens was 120, and the run ended mid-reasoning
at every depth including 0.

Needs Metal, so it runs in your shell:

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/toolcall-depth-ref.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3.6-35B-A3B-MLX-q3exp
"""
from __future__ import annotations

import argparse

from mlx_lm import load, generate

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--depths", default="0,8000,24000", help="approx tokens of prior conversation")
p.add_argument("--max-tokens", type=int, default=700)
# Our engine renders the NON-thinking form: the generation prompt ends with an
# empty <think>\n\n</think> block, pinned byte-exact against this vendor
# template in tests/unit/tool-calls.test.ts. apply_chat_template DEFAULTS to
# thinking on, so the first two runs of this script compared a thinking model
# against a non-thinking engine and the rows were meaningless. --thinking runs
# the other arm deliberately.
p.add_argument("--thinking", action="store_true",
               help="render WITH the thinking block (default: match our engine, which does not)")
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


# NB: `arguments` is a DICT, not a JSON string. The Qwen template does
# `tool_call.arguments | items`, which raises "Can only get item pairs from a
# mapping" on a string — the OpenAI wire format uses a string here and the
# jinja does not.
# The state our engine reaches right before it fails: three files read, the
# arithmetic available. The only correct next move is attempt_completion("84").
TAIL = [
    {"role": "user", "content": "What number does capacity() return? Read only the files you "
                                "need, then call attempt_completion with just the number."},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "read_file", "arguments": {"path": "src/pipeline.ts"}}}]},
    {"role": "tool", "tool_call_id": "c1",
     "content": 'import { WIDTH } from "./dims.ts"\nimport { scale } from "./scale.ts"\n'
                "export function capacity() { return scale(WIDTH) }"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c2", "type": "function",
         "function": {"name": "read_file", "arguments": {"path": "src/dims.ts"}}}]},
    {"role": "tool", "tool_call_id": "c2", "content": "export const WIDTH = 12"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c3", "type": "function",
         "function": {"name": "read_file", "arguments": {"path": "src/scale.ts"}}}]},
    {"role": "tool", "tool_call_id": "c3",
     "content": "export const FACTOR = 7\nexport function scale(n: number) { return n * FACTOR }"},
]

model, tokenizer = load(a.model)
print(f"{a.model}\n")

# Print the end of the rendered prompt before measuring anything. It is the one
# place the engine and the reference must agree, it is three lines long, and
# reading it would have caught the thinking mismatch on the first run instead of
# the third. Ours ends: '<|im_start|>assistant\n<think>\n\n</think>\n\n'
_probe = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM}] + TAIL, tools=TOOLS,
    add_generation_prompt=True, enable_thinking=a.thinking)
print(f"  thinking={a.thinking} · rendered prompt ends: "
      f"{tokenizer.decode(_probe[-14:])!r}\n")
print(f"  {'depth':>8} {'prompt tok':>11} {'gen':>6}     called?     how it ENDED")

for d in [int(x) for x in a.depths.split(",")]:
    msgs = [{"role": "system", "content": SYSTEM}] + padding(d) + TAIL
    prompt = tokenizer.apply_chat_template(
        msgs, tools=TOOLS, add_generation_prompt=True, enable_thinking=a.thinking)
    ntok = len(prompt)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=a.max_tokens, verbose=False)
    called = "<tool_call>" in out or "attempt_completion" in out
    # Qwen3.6's own template invites reasoning BEFORE the call ("you may provide
    # optional reasoning ... BEFORE the function call, but NOT after"), so the
    # call is the LAST thing generated, not the first. Two consequences the
    # first version of this script got wrong: 120 tokens truncated the reasoning
    # before the model ever reached the call, and printing the first 90
    # characters showed the reasoning instead of the answer.
    gen = len(tokenizer.encode(out))
    cap = "CUT" if gen >= a.max_tokens - 1 else "   "
    tail = out.strip().replace("\n", " ")[-90:]
    print(f"  {d:>8} {ntok:>11} {gen:>6}{cap}  {'YES' if called else 'NO ':<10}  …{tail!r}")

print("""
  Reading this — depth 0 is the CONTROL and must be YES. Our engine emits the
  call there every time, so a NO at depth 0 means this harness is broken (too
  few max tokens, wrong tools, wrong template) and the deeper rows mean nothing.
  A "CUT" in the gen column is that failure: it never reached the call.

    depth 0 YES, deeper NO   -> the MODEL loses the format with length; not ours
    every depth YES          -> ours: same conversation, same checkpoint, and we
                                emit prose where mlx_lm emits the call
    depth 0 NO               -> fix this harness before reading anything else

  And check the rendered tail above against what our engine sends. A run whose
  prompt ends in a THINKING generation prompt is not comparable to our output
  at all, whatever the rows say — the model answers instead of calling, at
  every depth, and that is the template's behaviour rather than a finding.
""")
