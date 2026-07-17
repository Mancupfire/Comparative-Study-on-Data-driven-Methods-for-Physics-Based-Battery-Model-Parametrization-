#!/usr/bin/env bash
# LHS retrain FULL orchestration. Runs BOTH official full pipelines against the
# regenerated data/lhs_1000_seed42 dataset, preserves every previous and new
# timestamped run (nothing is overwritten), then assembles a combined bundle.
#
# Steps:
#   1. official time-series full pipeline (all 1000 sample_ids)
#   2. official error-metric full pipeline (reuses the stored time-series split)
#   3. build a timestamped combined bundle directory
#   4. write outputs/lhs_1000_seed42/LATEST_RETRAIN_RUN.txt
#
# The error-metric step reads the stored split written by the time-series step,
# so run order matters and is enforced here.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"
export GPU_ID="${GPU_ID:-7}"
export DEVICE="${DEVICE:-cuda}"
export PYTHON

cd "$PROJECT_ROOT"

echo "===================================================================="
echo "[retrain-full 1/4] Time-series official FULL pipeline"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/run_lhs_official_full.sh"

echo "===================================================================="
echo "[retrain-full 2/4] Error-metric official FULL pipeline (stored split)"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/run_lhs_error_metrics_full.sh"

TS_RUN="$(cat "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_OFFICIAL_RUN.txt")"
EM_RUN="$(cat "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_ERROR_METRICS_RUN.txt")"

BUNDLE="$PROJECT_ROOT/outputs/lhs_1000_seed42/retrain_bundles/retrain_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BUNDLE"

echo "===================================================================="
echo "[retrain-full 3/4] Building combined bundle"
echo "===================================================================="
"$PYTHON" "$PROJECT_ROOT/scripts/summarize_lhs_retrain.py" \
  --time-series-run "$TS_RUN" \
  --error-metrics-run "$EM_RUN" \
  --bundle-dir "$BUNDLE"

echo "===================================================================="
echo "[retrain-full 4/4] Verifying both runs and recording LATEST pointer"
echo "===================================================================="
bash "$PROJECT_ROOT/scripts/check_lhs_retrain_status.sh" "$TS_RUN" "$EM_RUN"
echo "$BUNDLE" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_RETRAIN_RUN.txt"

echo "Finished retrain full run."
echo "  time-series : $TS_RUN"
echo "  error-metric: $EM_RUN"
echo "  bundle      : $BUNDLE"
