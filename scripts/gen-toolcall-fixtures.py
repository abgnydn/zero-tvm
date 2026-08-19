"""Render the multi-round tool conversation through the vendor templates.

The fixtures already in tests/unit/tool-calls.test.ts all END with a real user
query, so every assistant turn sits before it and NEITHER Qwen generation emits
a past-turn <think> block. They cannot tell the two rules apart, which is how an
unconditional block survived in buildChatPrompt.

This renders a conversation that does distinguish them: a completed round, then
a new query, then a tool call. Output is pasted into the test as a fixture.

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/gen-toolcall-fixtures.py
"""
from __future__ import annotations

import pathlib

from transformers import AutoTokenizer

W = pathlib.Path(__file__).resolve().parent.parent / '.weights-local'

# tools omitted, like the neighbouring fixtures — the tools block is pinned
# separately and would otherwise swamp this.
MESSAGES = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "weather in Istanbul?"},
    {"role": "assistant", "content": "", "tool_calls": [
        {"type": "function", "function": {"name": "get_weather",
                                          "arguments": {"city": "İstanbul"}}}]},
    {"role": "tool", "content": "18C, clear"},
]

for label, repo in [("QWEN3", "Qwen3-4B-4bit"), ("QWEN36", "Qwen3.6-35B-A3B-MLX-4bit")]:
    tok = AutoTokenizer.from_pretrained(W / repo)
    out = tok.apply_chat_template(MESSAGES, add_generation_prompt=True,
                                  enable_thinking=False, tokenize=False)
    print(f"const {label}_MULTI_ROUND = {out!r}\n")
