#!/bin/bash
# FIDELITY AT DEPTH, unattended.
#
# This check is the one docs/VERIFICATION.md calls the gap that let qwen38 ship
# broken, and it is also the most expensive thing in the repo: 20,489 tokens at
# qwen38's quarantined cap of 256 is 80 sequential chunks, and an interactive
# attempt ran past 60 minutes without finishing. So it runs overnight instead of
# in front of someone, with a protocol timeout that will not cut it off.
#
# The mlx_lm reference is NOT regenerated — .evals/ref-qwen38-16k was produced
# in 24 minutes and is depth-specific, so it is reused. Regenerating it here
# would double the runtime and prove nothing new.
#
# Installed by scripts/install-nightly.sh as a launchd agent; it unloads itself
# after one run so a machine left alone does not repeat an hour of GPU work
# every night.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

STAMP=$(date +%Y%m%d-%H%M)
OUT=".evals/nightly-fidelity-$STAMP.log"
REF=".evals/ref-qwen38-16k"

{
  echo "=== fidelity at depth · engine ==="
  echo "started:  $(date -Iseconds)"
  echo "ref:      $REF"
  echo "host:     $(sw_vers -productVersion) $(uname -m)"
  echo "free:     $(vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} END{printf "%.1f GB", (f+i)*ps/1073741824}')"
  echo

  if [ ! -f "$REF/meta.json" ]; then
    echo "ABORT: no reference at $REF. Generate it first:"
    echo "  cd ~/dev/ml-research && uv run python ~/dev/zero-tvm/scripts/mlx-ref.py \\"
    echo "      --model ~/dev/zero-tvm/.weights-local/Qwen3.8-27B-4bit \\"
    echo "      --depth 16000 --out ~/dev/zero-tvm/.evals/ref-qwen38-16k"
    exit 2
  fi

  # caffeinate: a display that sleeps mid-run takes the GPU with it, and the
  # failure looks like a hang rather than a sleep.
  ZTVM_PROTOCOL_MIN=240 ZTVM_PAGE_LOG=1 \
    caffeinate -dimsu node scripts/validate-model.mjs qwen38 --ref "$REF"
  CODE=$?

  echo
  echo "exit:     $CODE"
  echo "finished: $(date -Iseconds)"
} >> "$OUT" 2>&1

# One run, not a nightly habit. Remove the trigger; the log stays.
launchctl bootout "gui/$(id -u)/com.zerotvm.fidelity" 2>/dev/null
rm -f "$HOME/Library/LaunchAgents/com.zerotvm.fidelity.plist"
exit 0
