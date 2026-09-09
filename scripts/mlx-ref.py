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
p.add_argument("--depth", type=int, default=0,
               help="pad the prompt to roughly this many tokens before the question. "
                    "The whole roster is verified at ~20 tokens and NONE at depth, which "
                    "is how a model that is correct short and broken past 8k passes its "
                    "gate. Forwards in chunks, because materialising logits for every "
                    "position at 16k is ~24 GB.")
p.add_argument("--chunk", type=int, default=512,
               help="tokens per reference forward pass when --depth is set")
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

def forward_final(seq):
    """Final-position logits for a sequence of any length.

    model(x) returns logits for EVERY position — at 16k tokens over a 248k
    vocabulary that is ~24 GB and the process dies. So a long sequence is fed
    through a prompt cache in chunks and only the last chunk's last position is
    kept, which is the same arithmetic the engine does and the only part being
    compared.
    """
    if len(seq) <= args.chunk:
        return model(mx.array([seq]))[0, -1, :].astype(mx.float32)
    try:
        from mlx_lm.models.cache import make_prompt_cache
    except ImportError as e:      # the module moved between mlx_lm versions
        raise SystemExit(
            "--depth needs mlx_lm's prompt cache and this build does not expose it "
            f"at mlx_lm.models.cache ({e}). Without a cache the forward is quadratic "
            "in memory and will OOM; refusing rather than pretending."
        )
    cache = make_prompt_cache(model)
    out = None
    for i in range(0, len(seq), args.chunk):
        out = model(mx.array([seq[i:i + args.chunk]]), cache=cache)
        mx.eval(out)
    return out[0, -1, :].astype(mx.float32)


if args.depth:
    # Inert filler: it must not contain or hint at the answer, or this measures
    # retrieval instead of fidelity. Same shape the agentic eval uses.
    pad = []
    approx, i = 0, 0
    while approx < args.depth:
        q = f"Step {i}: summarise what changed in release 2.{i % 9}.{i % 5}."
        a = (f"Release 2.{i % 9}.{i % 5} adjusted logging thresholds, renamed two internal "
             "helpers, and left public behaviour unchanged. No configuration keys moved.")
        pad += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        approx += (len(q) + len(a)) // 4
        i += 1
    msgs = pad + [{"role": "user", "content": args.prompt}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    print(f"depth {args.depth}: {len(ids)} prompt ids, forwarding in chunks of {args.chunk}")

logits = forward_final(ids)
mx.eval(logits)

# Greedy continuation, KV-less re-forward each step (slow, exact, simple).
greedy = []
# Greedy continuation. The short path re-forwards the whole sequence per token —
# slow, exact, simple, and it was fine while prompts were one sentence. At depth
# it is 24 full forwards of a 16k sequence, so a cache is built ONCE from the
# prompt and each new token is fed alone. Same arithmetic, linear instead of
# quadratic.
if args.depth:
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(model)
    out = None
    for i in range(0, len(ids), args.chunk):
        out = model(mx.array([ids[i:i + args.chunk]]), cache=cache)
        mx.eval(out)
    nxt = int(mx.argmax(out[0, -1, :]).item())
    for _ in range(args.tokens):
        greedy.append(nxt)
        if nxt == getattr(tokenizer, "eos_token_id", None):
            break
        out = model(mx.array([[nxt]]), cache=cache)
        mx.eval(out)
        nxt = int(mx.argmax(out[0, -1, :]).item())
else:
    cur = list(ids)
    for _ in range(args.tokens):
        out = forward_final(cur)
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
    "depth": args.depth,
}, indent=1))
print(f"prompt {len(ids)} ids; argmax {order[0]} ({tokenizer.decode([order[0]])!r})")
print(f"greedy: {tokenizer.decode(greedy)!r}")
print(f"wrote {out}/meta.json + logits.bin ({len(lf)} f32)")
