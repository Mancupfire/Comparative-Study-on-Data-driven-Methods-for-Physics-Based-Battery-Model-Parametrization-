#!/usr/bin/env bash
# Error-metric full run: all available sample_ids, the official stored split and
# hyperparameters, all 9 model families. Writes the run path to
# outputs/lhs_1000_seed42/LATEST_ERROR_METRICS_RUN.txt.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/lhs_error_metrics_train.py"
RUN_ID="lhs_error_metrics_full_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/outputs/lhs_1000_seed42/error_metrics/$RUN_ID"

PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p "$OUT"
[[ -f "$DATA_DIR/error_metrics_by_case.csv" ]] || { echo "Missing dataset" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

# Uses the stored time-series split (LATEST_OFFICIAL_RUN.txt) automatically;
# override with --split-json if needed.
"$PYTHON" "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --models ann mlp wide_deep_mlp gated_mlp deep_ensemble_mlp \
           random_forest extra_trees xgboost catboost \
  --max-sample-ids 0 \
  --epochs 200 \
  --patience 20 \
  --ensemble-size 5 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --hidden-size 256 \
  --dropout 0.10 \
  --seed 42 \
  --device "$DEVICE" \
  --inference-repeats 5 \
  2>&1 | tee "$OUT/train.log"

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_ERROR_METRICS_RUN.txt"
echo "Finished error-metrics full run: $OUT"
