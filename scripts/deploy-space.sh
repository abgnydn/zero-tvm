#!/usr/bin/env bash
# Deploy the built site to the Hugging Face Space mirror
# (https://huggingface.co/spaces/abgunaydin/zero-tvm).
#
#   scripts/deploy-space.sh
#
# Builds dist/ (npm run build), then force-pushes dist/* plus
# hf-space/README.md (as the Space's README.md card) as a single commit —
# the Space mirrors the built site, it has no useful history of its own.
#
# Auth:
#   - locally: uses your ambient git credentials for huggingface.co
#   - in CI:   set HF_TOKEN (a HF write token) — used via the push URL,
#              never printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPACE_PATH="huggingface.co/spaces/abgunaydin/zero-tvm"
if [[ -n "${HF_TOKEN:-}" ]]; then
  SPACE_URL="https://abgunaydin:${HF_TOKEN}@${SPACE_PATH}"
else
  SPACE_URL="https://${SPACE_PATH}"
fi

SHORT_SHA="$(git -C "$ROOT" rev-parse --short HEAD)"

echo "==> Building dist/"
(cd "$ROOT" && npm run build)

[[ -f "$ROOT/hf-space/README.md" ]] || { echo "Missing hf-space/README.md" >&2; exit 1; }

# Throwaway work dir (kept under .tests-cache/ so it's already gitignored).
WORK_DIR="$(mktemp -d "$ROOT/.tests-cache/space-deploy.XXXXXX" 2>/dev/null || mktemp -d "${TMPDIR:-/tmp}/space-deploy.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

cp -R "$ROOT/dist/." "$WORK_DIR/"
cp "$ROOT/hf-space/README.md" "$WORK_DIR/README.md"

echo "==> Pushing to $SPACE_PATH as github@${SHORT_SHA}"
cd "$WORK_DIR"
git init -q -b main
git add -A
git -c user.name="zero-tvm deploy" \
    -c user.email="abgunaydin94@gmail.com" \
    -c commit.gpgsign=false \
    commit -q -m "sync from github@${SHORT_SHA}"
# Single-commit force push: the Space is a mirror of dist/, not a history.
git remote add space "$SPACE_URL"
git push -q --force space main
echo "==> Deployed https://${SPACE_PATH}"
