#!/usr/bin/env bash
# Batch 2 TIME-SERIES smoke test — MAIN downsampled (Batch-1-comparable) dataset.
# Runs all 7 per-case models on one case for 2 epochs.
#
# Writes ONLY to outputs_smoke/Data_Batch_2/time_series_downsampled_160/<run_id>/
# and logs to logs/Data_Batch_2/time_series_downsampled_160/<run_id>/.
# Never touches Batch 1, raw Batch 2, or the native cleaned data.
#
# Usage:
#   bash scripts/run_batch_2_smoke.sh [run_id]
#   bash scripts/run_batch_2_smoke.sh batch2_smoke_001
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch2_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single GPU only

CONFIG="configs/batch_2/time_series_downsampled_160.yaml"
DATA_ROOT="data/Data_Batch_2_downsampled_160"
SMOKE_ROOT="outputs_smoke/Data_Batch_2/time_series_downsampled_160/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/time_series_downsampled_160/${RUN_ID}"
SMOKE_CASE="CC_C_2p5_T25C"   # any case (all 160 pts after downsampling)
MODELS=(mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp)

mkdir -p "$LOG_DIR" "$SMOKE_ROOT"
echo "================ BATCH 2 SMOKE ================"
echo "run_id      = $RUN_ID"
echo "GPU         = $CUDA_VISIBLE_DEVICES"
echo "data_root   = $DATA_ROOT"
echo "smoke_root  = $SMOKE_ROOT"
echo "log_dir     = $LOG_DIR"
echo "smoke_case  = $SMOKE_CASE"
echo "models      = ${MODELS[*]}"
echo "=============================================="

# --- isolation safeguard (aborts on any Batch 1 collision) ---
python scripts/batch2_preflight.py \
  --data-dir "$DATA_ROOT" --output-root "$SMOKE_ROOT" --log-root "$LOG_DIR"

# --- reproducibility artifacts ---
python scripts/batch2_write_run_manifest.py \
  --config "$CONFIG" --run-dir "$SMOKE_ROOT" --log-dir "$LOG_DIR" \
  --data-dir "$DATA_ROOT" --run-id "$RUN_ID" --mode smoke

LOG="$LOG_DIR/smoke.log"
echo "[smoke] training 2 epochs/model -> $LOG"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_all_cases.py \
  --data-root "$DATA_ROOT" \
  --outputs-dir "$SMOKE_ROOT" \
  --cases "$SMOKE_CASE" \
  --models "${MODELS[@]}" \
  --epochs 2 --batch-size 64 --lr 1e-3 --seed 42 --device auto --mc-samples 3 \
  2>&1 | tee "$LOG"

# --- pass/fail report (also exits non-zero on any failure) ---
python scripts/batch2_smoke_report.py \
  --smoke-dir "$SMOKE_ROOT" --case-id "$SMOKE_CASE" --models "${MODELS[@]}" \
  2>&1 | tee -a "$LOG"

echo "[smoke] DONE. Report: $SMOKE_ROOT/smoke_test_report.md"
