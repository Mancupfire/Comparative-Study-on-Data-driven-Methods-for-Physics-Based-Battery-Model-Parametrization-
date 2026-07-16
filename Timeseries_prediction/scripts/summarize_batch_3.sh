#!/usr/bin/env bash
# Aggregate Batch 3 time-series metrics for a given run id.
# Usage: bash scripts/summarize_batch_3.sh <run_id>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?usage: summarize_batch_3.sh <run_id>}"
OUT_ROOT="outputs/Data_Batch_3/time_series_downsampled_160/${RUN_ID}"
LOG_DIR="logs/Data_Batch_3/time_series_downsampled_160/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "[summarize] run_id=$RUN_ID  out_root=$OUT_ROOT"
python scripts/summarize_batch2.py --run-dir "$OUT_ROOT" \
  --comparison-root "outputs/comparisons/Batch_2_vs_Batch_3" \
  2>&1 | tee "$LOG_DIR/summarize.log"
echo "[summarize] done. See $OUT_ROOT/experiment_summary.md"
