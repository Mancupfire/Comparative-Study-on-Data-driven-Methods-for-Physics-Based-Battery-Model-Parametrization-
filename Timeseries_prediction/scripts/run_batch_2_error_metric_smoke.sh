#!/usr/bin/env bash
# Explicit Batch 2 wrapper (requested canonical name). Delegates to the existing
# Batch 2 launcher so the current Batch 2 setup is reproduced EXACTLY.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO_ROOT"
exec bash scripts/run_error_metric_smoke.sh "$@"
