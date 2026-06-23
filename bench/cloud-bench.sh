#!/usr/bin/env bash
# Entrypoint for bench/Dockerfile. Primes the Phi-3 weights, runs the bench on
# the container's GPU, and prints bench/results.json for you to copy back.
set -euo pipefail

echo "== GPU / Vulkan =="
if ! vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion"; then
  echo "!! No Vulkan device visible. Did you run with '--gpus all' and the NVIDIA"
  echo "!! Container Toolkit installed? WebGPU needs a real adapter here."
  exit 1
fi

# Label the run with the actual GPU unless the caller set BENCH_HW.
export BENCH_HW="${BENCH_HW:-$(vulkaninfo --summary 2>/dev/null | awk -F'= ' '/deviceName/{print $2; exit}')}"
echo "== Bench hardware: ${BENCH_HW:-unknown} =="

# Prime the MLC weight mirror Vite serves at /local-weights (both Zero-TVM and
# WebLLM load from there). Skips the download if it's already cached.
echo "== Fetching Phi-3-mini q4f16_1 weights (~2 GB, first run only) =="
huggingface-cli download mlc-ai/Phi-3-mini-4k-instruct-q4f16_1-MLC >/dev/null

echo "== Running npm run bench =="
npm run bench

echo
echo "==================== bench/results.json ===================="
cat bench/results.json
echo "============================================================"
echo "Copy that JSON into your repo at bench/results.json, then run:"
echo "    npm run bench:sync -- --write"
echo "to update BENCH.md + the bench page (no GPU needed for that step)."
