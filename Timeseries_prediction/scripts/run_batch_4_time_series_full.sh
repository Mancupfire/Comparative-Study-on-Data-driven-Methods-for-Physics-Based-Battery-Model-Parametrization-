#!/usr/bin/env bash
# Batch 4 TIME-SERIES full training — Batch-3-comparable downsampled-160.
# 7 per-case models x 12 cases = 84 runs. Same schedule as Batch 2/3
# (300 epochs, batch 64, lr 1e-3, seed 42).
# Writes ONLY to outputs/Data_Batch_4/time_series_downsampled_160/<run_id>/.
# Never touches Batch 1/2/3 or any raw data. Never resumes from a foreign ckpt.
#
# Usage:
#   bash scripts/run_batch_4_time_series_full.sh [run_id] [--allow-resume]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch4_ts_full_$(date +%Y%m%d_%H%M%S)}"
RESUME_FLAG=""
if [[ "${2:-}" == "--allow-resume" ]]; then RESUME_FLAG="--allow-resume"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"   # single free GPU

CONFIG="configs/batch_4/time_series/full.yaml"
DATA_ROOT="data/Data_Batch_4_downsampled_160"
OUT_ROOT="outputs/Data_Batch_4/time_series_downsampled_160/${RUN_ID}"
LOG_DIR="logs/Data_Batch_4/time_series_downsampled_160/${RUN_ID}"
MODELS=(mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp)

mkdir -p "$LOG_DIR"
echo "================ BATCH 4 TS FULL ================"
echo "run_id    = $RUN_ID"
echo "GPU       = $CUDA_VISIBLE_DEVICES"
echo "data_root = $DATA_ROOT"
echo "out_root  = $OUT_ROOT"
echo "log_dir   = $LOG_DIR"
echo "models    = ${MODELS[*]}  (x 12 discovered cases = 84 runs)"
echo "schedule  = epochs 300, batch 64, lr 1e-3, seed 42 (Batch 2/3 protocol)"
echo "================================================"

python scripts/batch_preflight.py --dataset-name Data_Batch_4 \
  --data-dir "$DATA_ROOT" --output-root "$OUT_ROOT" --log-root "$LOG_DIR" $RESUME_FLAG

python scripts/batch2_write_run_manifest.py \
  --config "$CONFIG" --run-dir "$OUT_ROOT" --log-dir "$LOG_DIR" \
  --data-dir "$DATA_ROOT" --run-id "$RUN_ID" --mode full \
  --audit-dir "outputs/Data_Batch_4/data_audit"

LOG="$LOG_DIR/full_train.log"
echo "[full] training -> $LOG"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_all_cases.py \
  --data-root "$DATA_ROOT" --outputs-dir "$OUT_ROOT" \
  --models "${MODELS[@]}" \
  --epochs 300 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 \
  --hidden-dim 256 --num-layers 2 --dropout 0.1 --lambda-temp 1.0 \
  --patience 30 --seed 42 --device auto --mc-samples 30 \
  2>&1 | tee "$LOG"

echo "[full] training finished. Aggregating..."
python scripts/summarize_batch2.py --run-dir "$OUT_ROOT" \
  --comparison-root "outputs/comparisons/Batch_2_vs_Batch_4" 2>&1 | tee -a "$LOG" || true
echo "[full] DONE. Summary: $OUT_ROOT/experiment_summary.md"
