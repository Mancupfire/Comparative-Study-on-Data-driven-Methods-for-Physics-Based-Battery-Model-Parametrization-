#!/usr/bin/env bash
# Batch 3 — run BOTH task smoke tests with a shared run id.
# Usage: bash scripts/run_batch_3_all_smoke.sh [run_id]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${1:-batch3_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

echo "===== BATCH 3 ALL SMOKE (run_id=$RUN_ID, GPU=$CUDA_VISIBLE_DEVICES) ====="
bash scripts/run_batch_3_time_series_smoke.sh  "${RUN_ID}_ts"
bash scripts/run_batch_3_error_metric_smoke.sh "${RUN_ID}_em"
echo "===== BATCH 3 ALL SMOKE DONE ====="
