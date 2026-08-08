"""MLX-REF — reference logits + greedy continuation from mlx_lm itself.

The decisive half of validating a generated ModelSpec: whatever the browser
engine computes for the SAME checkpoint must agree with mlx_lm's own forward
pass. Dumps, for a fixed ChatML prompt (or --prompt-ids):

    <out>/meta.json     prompt ids, argmax, top-16 (id, logit), greedy ids + text
    <out>/logits.bin    full final-position logits, f32 little-endian

Run from an env with mlx_lm (the shared ML env):

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-ref.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3-4B-4bit --out /tmp/ref
"""

import argparse, json, pathlib

import mlx.core as mx
from mlx_lm import load

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--prompt", default="What is the capital of France?")
p.add_argument("--prompt-ids", default=None, help="comma-separated ids; overrides --prompt")
p.add_argument("--tokens", type=int, default=24)
args = p.parse_args()

model, tokenizer = load(args.model)

# Run the reference in FLOAT32. mlx_lm loads a checkpoint at its published
# activation dtype, which for these repos is bf16 — and a bf16 forward pass is
# NOT precise enough to grade the engine against. Measured on Qwen3-30B-A3B
# (48 layers), against this same model in f32:
#
#     engine        cosine 0.999985   rms 0.017
#     mlx in bf16   cosine 0.997726   rms 0.216
#
# so the bf16 reference "failed" the 0.999 gate by contributing twelve times
# more error than the thing it was measuring. The error is broadband and grows
# with depth, which is why a 36-layer dense model still scored 0.9999 and this
# one did not. Only the non-quantized tensors upcast (the quantized weights stay
# packed), so the cost is small even on the 35B.
model.set_dtype(mx.float32)

if args.prompt_ids:
    ids = [int(t) for t in args.prompt_ids.split(",")]
else:
    # The engine renders ChatML NON-thinking (empty <think> block) — match it
    # exactly or the token streams are incomparable.
    msgs = [{"role": "user", "content": args.prompt}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)

x = mx.array([ids])
logits = model(x)[0, -1, :].astype(mx.float32)
mx.eval(logits)

# Greedy continuation, KV-less re-forward each step (slow, exact, simple).
greedy = []
cur = list(ids)
for _ in range(args.tokens):
    out = model(mx.array([cur]))[0, -1, :]
    nxt = int(mx.argmax(out).item())
    greedy.append(nxt)
    cur.append(nxt)
    if nxt in (getattr(tokenizer, "eos_token_id", None),):
        break

lf = logits.tolist()
order = sorted(range(len(lf)), key=lambda i: -lf[i])
out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "logits.bin").write_bytes(memoryview(mx.array(lf, dtype=mx.float32)).tobytes())
(out / "meta.json").write_text(json.dumps({
    "model": args.model,
    "prompt_ids": ids,
    "argmax": order[0],
    "top16": [[i, lf[i]] for i in order[:16]],
    "greedy": greedy,
    "greedy_text": tokenizer.decode(greedy),
}, indent=1))
print(f"prompt {len(ids)} ids; argmax {order[0]} ({tokenizer.decode([order[0]])!r})")
print(f"greedy: {tokenizer.decode(greedy)!r}")
print(f"wrote {out}/meta.json + logits.bin ({len(lf)} f32)")
