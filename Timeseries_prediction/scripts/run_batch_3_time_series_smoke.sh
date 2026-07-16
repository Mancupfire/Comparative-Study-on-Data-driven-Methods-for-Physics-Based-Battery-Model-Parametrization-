#!/usr/bin/env bash
# Batch 3 TIME-SERIES smoke test — Batch-2-comparable downsampled-160 dataset.
# All 7 per-case models on ONE case for 2 epochs.
# Writes ONLY to outputs_smoke/Data_Batch_3/... and logs/Data_Batch_3/...
# Never touches Batch 1, Batch 2, or any raw data.
#
# Usage:
#   bash scripts/run_batch_3_time_series_smoke.sh [run_id]
#   CUDA_VISIBLE_DEVICES=4 bash scripts/run_batch_3_time_series_smoke.sh batch3_ts_smoke_001
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch3_ts_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"   # single GPU only (GPU 0 busy w/ other users)

CONFIG="configs/batch_3/time_series/smoke.yaml"
DATA_ROOT="data/Data_Batch_3_downsampled_160"
SMOKE_ROOT="outputs_smoke/Data_Batch_3/time_series_downsampled_160/${RUN_ID}"
LOG_DIR="logs/Data_Batch_3/time_series_downsampled_160/${RUN_ID}"
SMOKE_CASE="CC_C_2p5_T25C"
MODELS=(mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp)

mkdir -p "$LOG_DIR" "$SMOKE_ROOT"
echo "================ BATCH 3 TS SMOKE ================"
echo "run_id     = $RUN_ID"
echo "GPU        = $CUDA_VISIBLE_DEVICES"
echo "data_root  = $DATA_ROOT"
echo "smoke_root = $SMOKE_ROOT"
echo "log_dir    = $LOG_DIR"
echo "smoke_case = $SMOKE_CASE"
echo "models     = ${MODELS[*]}"
echo "================================================="

python scripts/batch_preflight.py --dataset-name Data_Batch_3 \
  --data-dir "$DATA_ROOT" --output-root "$SMOKE_ROOT" --log-root "$LOG_DIR"

python scripts/batch2_write_run_manifest.py \
  --config "$CONFIG" --run-dir "$SMOKE_ROOT" --log-dir "$LOG_DIR" \
  --data-dir "$DATA_ROOT" --run-id "$RUN_ID" --mode smoke \
  --audit-dir "outputs/Data_Batch_3/data_audit"

LOG="$LOG_DIR/smoke.log"
echo "[smoke] training 2 epochs/model -> $LOG"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_all_cases.py \
  --data-root "$DATA_ROOT" --outputs-dir "$SMOKE_ROOT" \
  --cases "$SMOKE_CASE" --models "${MODELS[@]}" \
  --epochs 2 --batch-size 64 --lr 1e-3 --seed 42 --device auto --mc-samples 3 \
  2>&1 | tee "$LOG"

python scripts/batch2_smoke_report.py \
  --smoke-dir "$SMOKE_ROOT" --case-id "$SMOKE_CASE" --models "${MODELS[@]}" \
  2>&1 | tee -a "$LOG"

echo "[smoke] DONE. Report: $SMOKE_ROOT/smoke_test_report.md"
