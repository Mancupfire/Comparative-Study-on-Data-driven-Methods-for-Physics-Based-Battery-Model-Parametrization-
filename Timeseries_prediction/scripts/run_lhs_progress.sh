#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
DATA_ZIP="$PROJECT_ROOT/data/lhs_1000_seed42.zip"
DATA_DIR="$PROJECT_ROOT/data/lhs_1000_seed42"
SCRIPT="$PROJECT_ROOT/scripts/emergency_lhs_train.py"
RUN_ID="lhs_progress_$(date +%Y%m%d_%H%M%S)"
OUT="$PROJECT_ROOT/outputs/lhs_1000_seed42/time_series/$RUN_ID"
GPU_ID="${GPU_ID:-7}"

cd "$PROJECT_ROOT"
mkdir -p "$DATA_DIR" "$OUT" logs/lhs_1000_seed42

if [[ ! -f "$DATA_DIR/generated_dataset.h5" ]]; then
  [[ -f "$DATA_ZIP" ]] || { echo "Missing $DATA_ZIP" >&2; exit 1; }
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT
  unzip -q "$DATA_ZIP" -d "$tmpdir"
  cp -a "$tmpdir/lhs_1000_seed42/." "$DATA_DIR/"
fi

python - <<'PY'
import h5py, matplotlib, numpy, pandas, sklearn, torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY

[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1

python "$SCRIPT" \
  --dataset-dir "$DATA_DIR" \
  --output-dir "$OUT" \
  --models mlp lstm bilstm \
  --max-sample-ids 300 \
  --sequence-length 160 \
  --epochs 12 \
  --patience 4 \
  --batch-size 64 \
  --hidden-size 128 \
  --num-layers 2 \
  --learning-rate 1e-3 \
  --seed 42 \
  --device cuda \
  --inference-repeats 3 \
  2>&1 | tee "$OUT/train.log"

echo "$OUT" | tee "$PROJECT_ROOT/outputs/lhs_1000_seed42/LATEST_PROGRESS_RUN.txt"
echo "Finished: $OUT"
