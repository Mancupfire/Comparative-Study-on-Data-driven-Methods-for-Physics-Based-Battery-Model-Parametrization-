#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
LATEST="$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_PROGRESS_RUN.txt"

if [[ -f "$LATEST" ]]; then
  OUT="$(cat "$LATEST")"
else
  OUT="$(find "$PROJECT_ROOT/outputs/lhs_1000_seed42/time_series" -maxdepth 1 -type d -name 'lhs_progress_*' 2>/dev/null | sort | tail -1)"
fi

[[ -n "${OUT:-}" && -d "$OUT" ]] || { echo "No progress run found."; exit 1; }

echo "RUN: $OUT"
echo "=== PROCESS ==="
ps -ef | grep '[e]mergency_lhs_train.py' || true
echo "=== LAST LOG LINES ==="
tail -40 "$OUT/train.log" 2>/dev/null || true
echo "=== METRICS ==="
cat "$OUT/metrics/model_metrics.csv" 2>/dev/null || true
echo "=== SUMMARY ==="
sed -n '1,100p' "$OUT/SUMMARY.md" 2>/dev/null || true
