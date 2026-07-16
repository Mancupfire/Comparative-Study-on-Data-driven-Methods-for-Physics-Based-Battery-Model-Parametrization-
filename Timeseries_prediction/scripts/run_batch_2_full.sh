#!/usr/bin/env bash
# Batch 2 TIME-SERIES full training — MAIN downsampled (Batch-1-comparable).
# 7 per-case models x 12 cases, EXACT Batch 1 schedule (300 epochs, batch 64,
# lr 1e-3, seed 42). Dataset = data/Data_Batch_2_downsampled_160 (160-pt grid).
#
# Writes ONLY to outputs/Data_Batch_2/time_series_downsampled_160/<run_id>/ and
# logs to logs/Data_Batch_2/time_series_downsampled_160/<run_id>/.
# Never touches Batch 1, raw Batch 2, or the native cleaned data.
# (The native-resolution ablation is a SEPARATE, not-yet-launched experiment.)
#
# Usage:
#   bash scripts/run_batch_2_full.sh [run_id] [--allow-resume]
#   bash scripts/run_batch_2_full.sh batch2_full_001
#   bash scripts/run_batch_2_full.sh batch2_full_001 --allow-resume   # safe resume
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch2_full_$(date +%Y%m%d_%H%M%S)}"
RESUME_FLAG=""
if [[ "${2:-}" == "--allow-resume" ]]; then RESUME_FLAG="--allow-resume"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # single GPU only

CONFIG="configs/batch_2/time_series_downsampled_160.yaml"
DATA_ROOT="data/Data_Batch_2_downsampled_160"
OUT_ROOT="outputs/Data_Batch_2/time_series_downsampled_160/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/time_series_downsampled_160/${RUN_ID}"
MODELS=(mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp)

mkdir -p "$LOG_DIR"
echo "================ BATCH 2 FULL ================"
echo "run_id     = $RUN_ID"
echo "GPU        = $CUDA_VISIBLE_DEVICES"
echo "data_root  = $DATA_ROOT"
echo "out_root   = $OUT_ROOT"
echo "log_dir    = $LOG_DIR"
echo "models     = ${MODELS[*]}  (x 12 discovered cases)"
echo "schedule   = epochs 300, batch 64, lr 1e-3, seed 42 (Batch 1 protocol)"
echo "============================================="

python scripts/batch2_preflight.py \
  --data-dir "$DATA_ROOT" --output-root "$OUT_ROOT" --log-root "$LOG_DIR" $RESUME_FLAG

python scripts/batch2_write_run_manifest.py \
  --config "$CONFIG" --run-dir "$OUT_ROOT" --log-dir "$LOG_DIR" \
  --data-dir "$DATA_ROOT" --run-id "$RUN_ID" --mode full

LOG="$LOG_DIR/full_train.log"
echo "[full] training -> $LOG"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_all_cases.py \
  --data-root "$DATA_ROOT" \
  --outputs-dir "$OUT_ROOT" \
  --models "${MODELS[@]}" \
  --epochs 300 --batch-size 64 --lr 1e-3 --weight-decay 1e-4 \
  --hidden-dim 256 --num-layers 2 --dropout 0.1 --lambda-temp 1.0 \
  --patience 30 --seed 42 --device auto --mc-samples 30 \
  2>&1 | tee "$LOG"

echo "[full] training finished. Aggregating..."
python scripts/summarize_batch2.py --run-dir "$OUT_ROOT" 2>&1 | tee -a "$LOG"
echo "[full] DONE. Summary: $OUT_ROOT/experiment_summary.md"
