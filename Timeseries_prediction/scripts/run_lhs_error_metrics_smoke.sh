#!/usr/bin/env bash
# Error-metric smoke run: 30 unique sample_ids, all 9 model families,
# neural models 2 epochs, small tree settings. Verifies the full pipeline
# (train, save, metrics, figures). Does NOT start full training.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/lhs_error_metrics_train.py"
RUN_ID="lhs_error_metrics_smoke_$(date +%Y%m%d_%H%M%S)"
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

"$PYTHON" "$SCRIPT" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --models ann mlp wide_deep_mlp gated_mlp deep_ensemble_mlp \
           random_forest extra_trees xgboost catboost \
  --max-sample-ids 30 \
  --epochs 2 \
  --patience 2 \
  --ensemble-size 3 \
  --batch-size 64 \
  --hidden-size 256 \
  --seed 42 \
  --device "$DEVICE" \
  --inference-repeats 2 \
  --smoke \
  2>&1 | tee "$OUT/train.log"

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_ERROR_METRICS_SMOKE_RUN.txt"
echo "Finished smoke: $OUT"

# Post-run verification (save/reload/inference/metrics/figures).
bash "$PROJECT_ROOT/scripts/check_lhs_error_metrics.sh" "$OUT"
