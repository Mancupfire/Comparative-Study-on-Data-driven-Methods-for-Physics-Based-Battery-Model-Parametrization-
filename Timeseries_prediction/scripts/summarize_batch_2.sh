#!/usr/bin/env bash
# Aggregate Batch 2 time-series metrics + write Batch1-vs-Batch2 comparison.
#
# Usage:
#   bash scripts/summarize_batch_2.sh <run_id>
#   bash scripts/summarize_batch_2.sh batch2_full_001
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?usage: summarize_batch_2.sh <run_id>}"
OUT_ROOT="outputs/Data_Batch_2/time_series/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/time_series/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "[summarize] run_id=$RUN_ID  out_root=$OUT_ROOT"
python scripts/summarize_batch2.py --run-dir "$OUT_ROOT" \
  --batch1-outputs outputs 2>&1 | tee "$LOG_DIR/summarize.log"
echo "[summarize] done. See $OUT_ROOT/experiment_summary.md"
