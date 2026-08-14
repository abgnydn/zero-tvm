#!/usr/bin/env bash
# QUALITY-SWEEP — the number this repo has never had.
#
#   bash scripts/quality-sweep.sh 2>&1 | tee /tmp/quality-sweep.log
#
# Everything else here is FIDELITY: validate-model.mjs asks whether the engine
# computes what mlx_lm computes, and mlx-ref.py loads the SAME quantized
# checkpoint — so a model quantized into gibberish passes every gate (verified:
# a 2-bit Llama-1B does). Nothing has measured whether what we ship is GOOD.
#
# Two different questions, and this runs both:
#
#   1. Does OUR ENGINE cost quality? Engine perplexity against mlx_lm's, over
#      the IDENTICAL token ids. Hundreds of scored positions instead of one
#      prompt's final logits — a much stronger fidelity check than we had, and
#      it is the engine's own output being scored, not a reference's.
#
#   2. Is a build any good in absolute terms? Perplexity on held-out text.
#      Only comparable against another build on the SAME tokens, never against
#      a published figure (different tokenizer, corpus, window).
#
# BOTH CORPORA. `prose` is written for this repo rather than lifted from
# wikitext, which sits in every one of these models' training sets; `code` is
# this repo's own TypeScript, which is the agentic target.
#
# Serial by construction: one GPU. The models run smallest first so a failure
# on the big ones still leaves the small ones measured.
#
# NOT INCLUDED, and deliberately: the qwen36 4-bit vs 3-bit checkpoint A/B.
# That is the highest-value quality question we have (we SHIP the 3-bit build
# and nothing has ever checked it) but it loads a 19.7 GB model in mlx while
# LM Studio holds ~6 GB on a 32 GB machine, which is the configuration that
# froze this Mac once. Run it deliberately, with the owner present:
#   uv run python scripts/quality-ab.py --a <4bit> --out /tmp/a.json
#   uv run python scripts/quality-ab.py --a <q3exp> --out /tmp/b.json
#   uv run python scripts/quality-ab.py --compare /tmp/a.json /tmp/b.json
# (--compare exists for exactly this: never hold both in memory.)

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML="${ML_RESEARCH:-$HOME/dev/ml-research}"
OUT="${OUT_DIR:-$ROOT/bench/quality}"
TOKENS="${TOKENS:-512}"
mkdir -p "$OUT"

# param : local checkpoint directory under .weights-local
MODELS=(
  "llama32:Llama-3.2-1B-Instruct-4bit"
  "qwen3mlx:Qwen3-4B-4bit"
  "qwen35mlx:Qwen3.5-9B-MLX-4bit"
  "qwen30b:Qwen3-30B-A3B-4bit"
)

echo "=== quality sweep: $TOKENS tokens, corpora prose+code ==="
pmset -g batt | tail -1

for entry in "${MODELS[@]}"; do
  param="${entry%%:*}"; dir="${entry##*:}"
  for corpus in prose code; do
    tag="$param-$corpus"
    ids="$OUT/ids-$tag.json"
    ref="$OUT/ref-$tag.json"
    echo ""
    echo "--- $tag ---"

    # 1. Pick the ids. No GPU, no browser — and doing it FIRST is what makes
    #    the reference score the same text the engine scores. Two perplexities
    #    over different windows are not comparable, which is the whole point.
    node "$ROOT/scripts/quality-eval.mjs" "$param" --corpus "$corpus" \
      --tokens "$TOKENS" --dump-ids "$ids" || { echo "SKIP $tag: id dump failed"; continue; }

    # 2. The reference, pinned to those exact ids.
    ( cd "$ML" && uv run python "$ROOT/scripts/mlx-perplexity.py" \
        --model "$ROOT/.weights-local/$dir" --ids "$ids" --out "$ref" ) \
      || { echo "SKIP $tag: reference failed"; continue; }

    # 3. The engine, same ids, with the comparison.
    node "$ROOT/scripts/quality-eval.mjs" "$param" --corpus "$corpus" \
      --tokens "$TOKENS" --ref "$ref" 2>&1 | tee "$OUT/engine-$tag.txt" \
      || echo "FAILED $tag: engine run"
  done
done

echo ""
echo "=== done. artifacts in $OUT ==="
