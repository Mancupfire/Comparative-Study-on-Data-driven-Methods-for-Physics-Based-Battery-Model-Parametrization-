#!/usr/bin/env bash
# Batch 4 — run BOTH full tasks with one shared timestamped run id, on one GPU.
# Time-series (84 runs) then error-metric (2 runs), sequentially on the same GPU.
# Writes launch info + per-task logs under the Batch 4 namespaces.
#
# Usage:
#   bash scripts/run_batch_4_all_full.sh [run_id]
#   CUDA_VISIBLE_DEVICES=5 nohup bash scripts/run_batch_4_all_full.sh batch4_full_001 \
#       > logs/Data_Batch_4/batch4_full_001/console.log 2>&1 &
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch4_full_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"   # single free GPU

LAUNCH_DIR="logs/Data_Batch_4/${RUN_ID}"
mkdir -p "$LAUNCH_DIR"
{
  echo "run_id=$RUN_ID"
  echo "pid=$$"
  echo "gpu=$CUDA_VISIBLE_DEVICES"
  echo "host=$(hostname)"
  echo "started_at=$(date -Iseconds)"
  echo "ts_out=outputs/Data_Batch_4/time_series_downsampled_160/${RUN_ID}"
  echo "em_out=outputs/Data_Batch_4/error_metric/${RUN_ID}"
} > "$LAUNCH_DIR/launch_info.txt"
cat "$LAUNCH_DIR/launch_info.txt"

echo "===== [1/2] Batch 4 TIME-SERIES full ====="
bash scripts/run_batch_4_time_series_full.sh "$RUN_ID"

echo "===== [2/2] Batch 4 ERROR-METRIC full ====="
bash scripts/run_batch_4_error_metric_full.sh "$RUN_ID"

echo "completed_at=$(date -Iseconds)" >> "$LAUNCH_DIR/launch_info.txt"
echo "===== BATCH 4 ALL FULL DONE (run_id=$RUN_ID) ====="
