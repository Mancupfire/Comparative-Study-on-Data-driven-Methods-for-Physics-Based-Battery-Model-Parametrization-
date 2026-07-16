#!/usr/bin/env bash
# Official LHS full run: all 1000 sample_ids, all official models,
# official_clamped alignment, verified previous hyperparameters.
# Writes the final run path to outputs/lhs_1000_seed42/LATEST_OFFICIAL_RUN.txt.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_ZIP="$PROJECT_ROOT/data/lhs_1000_seed42.zip"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/emergency_lhs_train.py"
RUN_ID="lhs_official_full_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/outputs/lhs_1000_seed42/time_series/$RUN_ID"

PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"
GPU_ID="${GPU_ID:-7}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p "$DATA_DIR" "$OUT" "$PROJECT_ROOT/artifacts"

if [[ ! -f "$DATA_DIR/generated_dataset.h5" ]]; then
  [[ -f "$DATA_ZIP" ]] || { echo "Missing dataset and $DATA_ZIP" >&2; exit 1; }
  tmpdir="$(mktemp -d)"; trap 'rm -rf "$tmpdir"' EXIT
  unzip -q "$DATA_ZIP" -d "$tmpdir"
  cp -a "$tmpdir/lhs_1000_seed42/." "$DATA_DIR/"
fi
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

# Verified previous hyperparameters (match scripts/run_lhs_full.sh).
"$PYTHON" "$SCRIPT" \
  --dataset-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --alignment-mode official_clamped \
  --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
  --max-sample-ids 0 \
  --sequence-length 160 \
  --epochs 300 \
  --patience 30 \
  --batch-size 64 \
  --hidden-size 128 \
  --num-layers 2 \
  --learning-rate 1e-3 \
  --weight-decay 1e-5 \
  --dropout 0.10 \
  --seed 42 \
  --device "$DEVICE" \
  --inference-repeats 5 \
  2>&1 | tee "$OUT/train.log"

cp -f "$OUT/artifacts/excluded_sequences.csv" \
      "$PROJECT_ROOT/artifacts/excluded_sequences.csv" 2>/dev/null || true

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_OFFICIAL_RUN.txt"
echo "Finished official full run: $OUT"
