"""EMBED-REF — reference sentence embeddings from mlx_lm itself.

The embedding sibling of mlx-ref.py. A Qwen3-Embedding checkpoint is an
ordinary CAUSAL DECODER (config.json says Qwen3ForCausalLM); what differs is
only the tail. mlx_lm's `model.model(ids)` returns `self.norm(h)` — the final
RMSNorm'd hidden state, i.e. HF's `last_hidden_state` — which is exactly the
buffer the engine's forwardEmbedding() reads back (B.hidden1, what the LM head
would have consumed). So this dumps the reference the engine must reproduce.

THE POOLING RECIPE, and where each part comes from — get any of it wrong and
the embeddings stay plausible (unit norm, sane-looking cosines) while ranking
the wrong document first:

  1. instruct prefix  config_sentence_transformers.json "prompts.query":
                      "Instruct: <task>\nQuery:" on QUERIES only, "" on
                      documents. Asymmetric on purpose.
  2. EOS marker       tokenizer.json's post_processor is a TemplateProcessing
                      that appends <|endoftext|> (151643) to EVERY sequence.
                      The pooled token IS that marker. HF's fast tokenizer
                      applies it inside encode(); a hand-written tokenizer that
                      only reads vocab+merges will silently pool the last WORD
                      instead.
  3. pooling          1_Pooling/config.json: pooling_mode_lasttoken = true
                      (every other mode false). Causal decoder + unpadded
                      sequence => the final position sees everything.
  4. normalise        modules.json ends with sentence_transformers.models
                      .Normalize, so embeddings are L2-normalised and cosine
                      similarity is a plain dot product.

Run from an env with mlx_lm (the shared ML env):

    cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/embed-ref.py \
        --model ~/dev/zero-tvm/.weights-local/Qwen3-Embedding-0.6B-4bit-DWQ \
        --out /tmp/ref-qwen3emb

With no --texts it embeds the four strings the Qwen3-Embedding-0.6B model card
uses, whose 2x2 query-document cosine matrix the card publishes — so a run with
no local ground truth still has an external number to check against.
"""

import argparse, json, pathlib

import mlx.core as mx
from mlx_lm import load

# The model card's own example (queries first, then documents) and the scores
# it publishes for them. Reproducing these is the check that the recipe above
# is the vendor's recipe and not merely a self-consistent one.
CARD_TASK = "Given a web search query, retrieve relevant passages that answer the query"
CARD_QUERIES = ["What is the capital of China?", "Explain gravity"]
CARD_DOCUMENTS = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other."
    " It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]
CARD_SCORES = [[0.7645568251609802, 0.14142508804798126],
               [0.13549736142158508, 0.5999549627304077]]

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--out", required=True)
p.add_argument("--texts", default=None,
               help='JSON file {"task": str, "queries": [...], "documents": [...]}; '
                    "default is the model card's own example")
args = p.parse_args()

if args.texts:
    spec = json.loads(pathlib.Path(args.texts).read_text())
    task = spec.get("task", CARD_TASK)
    queries, documents = spec["queries"], spec["documents"]
else:
    task, queries, documents = CARD_TASK, CARD_QUERIES, CARD_DOCUMENTS

model, tokenizer = load(args.model)

# Same reason as mlx-ref.py: a bf16 reference contributed more error than the
# engine it was grading. Only the unquantized tensors upcast.
model.set_dtype(mx.float32)


def instruct(q):
    return f"Instruct: {task}\nQuery:{q}"


def embed(text):
    """Tokenize (the post-processor appends <|endoftext|>), run the decoder,
    take the LAST position of the final-normed hidden state, L2-normalise."""
    ids = tokenizer.encode(text)
    h = model.model(mx.array([ids]))[0, -1, :].astype(mx.float32)
    mx.eval(h)
    return ids, h / mx.linalg.norm(h)


texts = [instruct(q) for q in queries] + list(documents)
rows = [embed(t) for t in texts]
embs = mx.stack([e for _, e in rows])
mx.eval(embs)

nq = len(queries)
scores = (embs[:nq] @ embs[nq:].T).tolist()

out = pathlib.Path(args.out)
out.mkdir(parents=True, exist_ok=True)
(out / "emb.bin").write_bytes(memoryview(embs.astype(mx.float32)).tobytes())
meta = {
    "model": args.model,
    "task": task,
    "queries": queries,
    "documents": documents,
    "texts": texts,
    "dim": int(embs.shape[1]),
    "eos_id": tokenizer.encode("x")[-1],
    "prompt_ids": [ids for ids, _ in rows],
    "scores": scores,
}
(out / "meta.json").write_text(json.dumps(meta, indent=1))

print(f"{len(texts)} texts, dim {meta['dim']}; last id of a bare 'x' = {meta['eos_id']}"
      f" (expect the EOS marker, not a word piece)")
for i, row in enumerate(scores):
    print(f"  q{i}: " + "  ".join(f"{v:.4f}" for v in row))
if not args.texts:
    err = max(abs(a - b) for ra, rb in zip(scores, CARD_SCORES) for a, b in zip(ra, rb))
    print(f"model card publishes {CARD_SCORES}")
    print(f"max abs deviation from the card: {err:.4f}")
print(f"wrote {out}/meta.json + emb.bin ({len(texts)}x{meta['dim']} f32)")
