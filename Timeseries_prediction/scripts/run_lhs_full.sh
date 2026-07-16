#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/emergency_lhs_train.py"
RUN_ID="lhs_full_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/outputs/lhs_1000_seed42/time_series/$RUN_ID"
GPU_ID="${GPU_ID:-7}"

cd "$PROJECT_ROOT"
[[ -f "$DATA_DIR/generated_dataset.h5" ]] || {
  echo "Dataset is not extracted at $DATA_DIR. Run the progress setup first." >&2
  exit 1
}
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 1; }
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

python "$SCRIPT" \
  --dataset-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
  --max-sample-ids 0 \
  --sequence-length 160 \
  --epochs 300 \
  --patience 30 \
  --batch-size 64 \
  --hidden-size 128 \
  --num-layers 2 \
  --learning-rate 1e-3 \
  --seed 42 \
  --device cuda \
  --inference-repeats 5 \
  2>&1 | tee "$OUT/train.log"

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_FULL_RUN.txt"
echo "Finished: $OUT"
