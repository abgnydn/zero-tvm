"""The conversation the agentic loop fails on, in one place.

Both the reference test (toolcall-depth-ref.py) and the render diff
(render-diff.py) need this exact conversation, and a second copy of it would be
free to drift — which is the failure mode this whole investigation kept hitting.

Two wire shapes, because the two consumers disagree and the disagreement is
load-bearing:

  wire=False  what jinja wants: `arguments` as a MAPPING. The Qwen template does
              `tool_call.arguments | items`, which raises on a string.
  wire=True   what a client actually sends, and therefore what our host parses:
              `arguments` as a JSON STRING, per the OpenAI shape.

Passing the wrong one is not a subtle failure in either direction — jinja
raises, and the host's JSON.parse falls back to {} — but they are different
objects and only one of them is what the engine sees in production.
"""
from __future__ import annotations

import json

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

# The state our engine reaches right before it fails: three files read, the
# arithmetic available. The only correct next move is attempt_completion("84").
_READS = [
    ("c1", "src/pipeline.ts",
     'import { WIDTH } from "./dims.ts"\nimport { scale } from "./scale.ts"\n'
     "export function capacity() { return scale(WIDTH) }"),
    ("c2", "src/dims.ts", "export const WIDTH = 12"),
    ("c3", "src/scale.ts",
     "export const FACTOR = 7\nexport function scale(n: number) { return n * FACTOR }"),
]


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


def tail(wire: bool = False):
    out = [{"role": "user",
            "content": "What number does capacity() return? Read only the files you "
                       "need, then call attempt_completion with just the number."}]
    for cid, path, body in _READS:
        args = {"path": path}
        out.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {
                "name": "read_file",
                "arguments": json.dumps(args) if wire else args}}]})
        out.append({"role": "tool", "tool_call_id": cid, "content": body})
    return out


# ── the agentic eval's TURN ONE ──────────────────────────────────────────────
# A different shape from the tail above and the one that actually fails: six
# tools instead of two, and NO tool history — the model has taken no action yet.
# qwen38 fails on this first call, inventing `mcp__tools__repo_search`, so the
# 3-reads-done tail was never the failing state for it. Verified byte-identical
# to the checkpoint's own template at depth 0 and 24000.
EVAL_TOOLS = [
    {"type": "function", "function": {"name": "list_files",
     "description": "List every file in the workspace",
     "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {"name": "read_file",
     "description": "Read one file by path",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "search",
     "description": "Search file contents for a string",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "write_file",
     "description": "Write content to a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command",
     "description": "Run a shell command in the workspace",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "attempt_completion",
     "description": "Report the final answer and finish the task",
     "parameters": {"type": "object", "properties": {"result": {"type": "string"}}, "required": ["result"]}}},
]

EVAL_TASK = ("What number does capacity() return? Read only the files you need, "
             "then call attempt_completion with just the number. Do not guess.")


def eval_padding(target: int):
    """The eval's own filler — a different divisor and different prose from
    padding() above, so it must not be shared with it."""
    msgs, approx, i = [], 0, 0
    import math
    while approx < target:
        q = f"Step {i}: summarise what changed in release 2.{i % 9}.{i % 5}."
        a = (f"Release 2.{i % 9}.{i % 5} adjusted logging thresholds, renamed two internal "
             "helpers, and left public behaviour unchanged. No configuration keys moved, and the "
             "deployment procedure is identical to the previous release. Nothing here affects "
             "pipeline capacity or the dimension constants used elsewhere in the service.")
        msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        approx += math.ceil((len(q) + len(a)) / 3.6)
        i += 1
    return msgs


def eval_turn1(depth: int):
    return [{"role": "system", "content": SYSTEM}] + eval_padding(depth) + \
           [{"role": "user", "content": EVAL_TASK}]


def conversation(depth: int, wire: bool = False):
    return [{"role": "system", "content": SYSTEM}] + padding(depth) + tail(wire)
