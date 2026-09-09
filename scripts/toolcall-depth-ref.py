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
import pathlib
import sys

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
p.add_argument("--case", choices=("tail", "eval-turn1"), default="tail",
               help="tail: three files already read, the state our engine gave prose on. "
                    "eval-turn1: the agentic eval's FIRST call — six tools, no history — "
                    "which is where qwen38 actually fails, inventing a tool name.")
p.add_argument("--thinking", action="store_true",
               help="render WITH the thinking block (default: match our engine, which does not)")
a = p.parse_args()

# The conversation lives in toolcall_case.py — scripts/render-diff.py needs the
# same one, and a second copy would drift.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import toolcall_case as case  # noqa: E402

TOOLS, SYSTEM = case.TOOLS, case.SYSTEM


model, tokenizer = load(a.model)
print(f"{a.model}\n")

# Print the end of the rendered prompt before measuring anything. It is the one
# place the engine and the reference must agree, it is three lines long, and
# reading it would have caught the thinking mismatch on the first run instead of
# the third. Ours ends: '<|im_start|>assistant\n<think>\n\n</think>\n\n'
# The SELECTED case, not always the tail — this line printed a </tool_response>
# ending for --case eval-turn1, which has no tool history at all.
_probe = tokenizer.apply_chat_template(
    case.eval_turn1(0) if a.case == "eval-turn1" else case.conversation(0),
    tools=case.EVAL_TOOLS if a.case == "eval-turn1" else TOOLS,
    add_generation_prompt=True, enable_thinking=a.thinking)
print(f"  thinking={a.thinking} · rendered prompt ends: "
      f"{tokenizer.decode(_probe[-14:])!r}\n")
print(f"  case: {a.case}")
print(f"  {'depth':>8} {'prompt tok':>11} {'gen':>6}     called?     how it ENDED")

for d in [int(x) for x in a.depths.split(",")]:
    msgs = case.eval_turn1(d) if a.case == "eval-turn1" else case.conversation(d)
    tools = case.EVAL_TOOLS if a.case == "eval-turn1" else TOOLS
    prompt = tokenizer.apply_chat_template(
        msgs, tools=tools, add_generation_prompt=True, enable_thinking=a.thinking)
    ntok = len(prompt)
    out = generate(model, tokenizer, prompt=prompt, max_tokens=a.max_tokens, verbose=False)
    called = "<tool_call>" in out or "attempt_completion" in out
    # A tool name that is not on offer is its own outcome, not a pass: qwen38
    # emits well-formed <tool_call> blocks naming mcp__tools__* functions that
    # appear nowhere in the prompt, which "did it call something?" scores as YES.
    names = {t["function"]["name"] for t in tools}
    invented = "mcp__" in out or ("<function=" in out and not any(f"<function={n}>" in out for n in names))
    # Qwen3.6's own template invites reasoning BEFORE the call ("you may provide
    # optional reasoning ... BEFORE the function call, but NOT after"), so the
    # call is the LAST thing generated, not the first. Two consequences the
    # first version of this script got wrong: 120 tokens truncated the reasoning
    # before the model ever reached the call, and printing the first 90
    # characters showed the reasoning instead of the answer.
    gen = len(tokenizer.encode(out))
    cap = "CUT" if gen >= a.max_tokens - 1 else "   "
    tail = out.strip().replace("\n", " ")[-90:]
    verdict = "INVENTED" if invented else ("YES" if called else "NO")
    print(f"  {d:>8} {ntok:>11} {gen:>6}{cap}  {verdict:<10}  …{tail!r}")

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
