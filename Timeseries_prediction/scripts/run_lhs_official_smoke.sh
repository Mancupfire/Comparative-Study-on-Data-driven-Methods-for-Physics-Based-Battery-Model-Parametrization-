#!/usr/bin/env bash
# Official LHS smoke run: 30 sample_ids, all official models, 2 epochs,
# official_clamped alignment. Verifies the pipeline end-to-end quickly.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_ZIP="$PROJECT_ROOT/data/lhs_1000_seed42.zip"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/emergency_lhs_train.py"
RUN_ID="lhs_official_smoke_$(date +%Y%m%d_%H%M%S)"
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

"$PYTHON" "$SCRIPT" \
  --dataset-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --alignment-mode official_clamped \
  --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
  --max-sample-ids 30 \
  --sequence-length 160 \
  --epochs 2 \
  --patience 2 \
  --batch-size 64 \
  --hidden-size 128 \
  --num-layers 2 \
  --learning-rate 1e-3 \
  --seed 42 \
  --device "$DEVICE" \
  --inference-repeats 2 \
  2>&1 | tee "$OUT/train.log"

# Surface the run-scoped excluded-sequence audit at the repo-level path.
cp -f "$OUT/artifacts/excluded_sequences.csv" \
      "$PROJECT_ROOT/artifacts/excluded_sequences.csv" 2>/dev/null || true

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_OFFICIAL_SMOKE_RUN.txt"
echo "Finished smoke: $OUT"
