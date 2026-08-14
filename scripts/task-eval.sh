#!/usr/bin/env bash
# TASK-EVAL — benchmarks with a RIGHT ANSWER, which perplexity is not.
#
#   bash scripts/task-eval.sh <checkpoint-dir> <tag>
#   bash scripts/task-eval.sh .weights-local/Qwen3.6-35B-A3B-MLX-4bit qwen36-4bit
#
# Perplexity said the 3-bit expert build costs +10.4% (BENCH.md). That is the
# wrong instrument to decide whether to ship it: published work puts the
# accuracy drop at 3-bit near 3x the perplexity drop, so a 10% perplexity cost
# is consistent with anything from "unnoticeable" to "cannot do arithmetic".
# These tasks have a right answer.
#
# Runs on the REFERENCE (mlx_lm), not our engine, because the question is about
# the CHECKPOINT. Our engine is already pinned to mlx_lm at ratio 1.000-1.005
# on the same tokens, so a checkpoint result transfers; an engine result would
# only re-measure that agreement more expensively.
#
# lm-eval is layered in with `uv run --with` so ~/dev/ml-research's own
# pyproject is untouched — it is a shared env, not this project's.
#
# NOTE `python -m mlx_lm.evaluate` SILENTLY DOES NOTHING: evaluate.py has no
# `if __name__ == "__main__"` guard, so -m imports it, calls nothing, and exits
# 0 with no output. The console script `mlx_lm.evaluate` is the entry point.
# That cost twenty minutes of looking for a broken pipe.
#
# Limits are per task and they are small. The point is to separate two builds,
# not to publish an absolute score — quote the stderr alongside any number, and
# do not compare these to a leaderboard (different shots, different limits,
# different harness version).

set -uo pipefail
CKPT="${1:?usage: task-eval.sh <checkpoint-dir> <tag>}"
TAG="${2:?usage: task-eval.sh <checkpoint-dir> <tag>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ML="${ML_RESEARCH:-$HOME/dev/ml-research}"
OUT="$ROOT/bench/quality/tasks-$TAG"
mkdir -p "$OUT"

# Cheapest downloads first, so a slow link fails on the big one having already
# produced results. gsm8k is GENERATIVE (one full completion per item) and by
# far the slowest per sample, hence the smallest limit.
#   task            limit  why
#   arc_challenge   400    grade-school science reasoning, multiple choice
#   winogrande      400    pronoun resolution, needs real world modelling
#   gsm8k           100    multi-step arithmetic — where 3-bit is claimed worst
#   mmlu             10    x57 subjects = 570 items, broad knowledge
TASKS=("arc_challenge:400" "winogrande:400" "gsm8k:100" "mmlu:10")

cd "$ML" || exit 1
echo "=== task eval: $TAG ($CKPT) ==="
date

for entry in "${TASKS[@]}"; do
  task="${entry%%:*}"; limit="${entry##*:}"
  echo ""
  echo "--- $task (limit $limit) ---"
  uv run --with lm-eval mlx_lm.evaluate \
    --model "$CKPT" --tasks "$task" --limit "$limit" \
    --output-dir "$OUT" 2>&1 | rg -v "it/s\]|examples/s\]" | tail -20 \
    || echo "FAILED $task"
done

echo ""
echo "=== $TAG done -> $OUT ==="
date
