#!/usr/bin/env bash
# Batch 4 — run BOTH task smoke tests with a shared run id.
# Usage: bash scripts/run_batch_4_all_smoke.sh [run_id]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RUN_ID="${1:-batch4_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

echo "===== BATCH 4 ALL SMOKE (run_id=$RUN_ID, GPU=$CUDA_VISIBLE_DEVICES) ====="
bash scripts/run_batch_4_time_series_smoke.sh  "${RUN_ID}_ts"
bash scripts/run_batch_4_error_metric_smoke.sh "${RUN_ID}_em"
echo "===== BATCH 4 ALL SMOKE DONE ====="
