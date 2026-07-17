#!/usr/bin/env bash
# LHS retrain SMOKE: exercise BOTH official pipelines end-to-end on 30 unique
# sample_ids using the same group-aware split logic, every model family, neural
# models at 2 epochs and reduced tree/boosting settings. Then verify saves,
# reloads, finite inference, timing fields, figures and CSVs, and build a smoke
# bundle. This never starts full training.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"
export GPU_ID="${GPU_ID:-7}"
export DEVICE="${DEVICE:-cuda}"
export PYTHON

cd "$PROJECT_ROOT"

echo "===================================================================="
echo "[retrain-smoke 1/4] Time-series official smoke (30 sample_ids, 2 epochs)"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/run_lhs_official_smoke.sh"

echo "===================================================================="
echo "[retrain-smoke 2/4] Error-metric official smoke (30 sample_ids, 2 epochs)"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/run_lhs_error_metrics_smoke.sh"

TS_RUN="$(cat "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_OFFICIAL_SMOKE_RUN.txt")"
EM_RUN="$(cat "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_ERROR_METRICS_SMOKE_RUN.txt")"

BUNDLE="$PROJECT_ROOT/outputs/lhs_1000_seed42/retrain_bundles/retrain_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BUNDLE"

echo "===================================================================="
echo "[retrain-smoke 3/4] Building smoke bundle"
echo "===================================================================="
"$PYTHON" "$PROJECT_ROOT/scripts/summarize_lhs_retrain.py" \
  --time-series-run "$TS_RUN" \
  --error-metrics-run "$EM_RUN" \
  --bundle-dir "$BUNDLE"
echo "$BUNDLE" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_RETRAIN_SMOKE_RUN.txt"

echo "===================================================================="
echo "[retrain-smoke 4/4] Verifying both runs"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/check_lhs_retrain_status.sh" "$TS_RUN" "$EM_RUN"

echo "Finished retrain smoke."
echo "  time-series : $TS_RUN"
echo "  error-metric: $EM_RUN"
echo "  bundle      : $BUNDLE"
