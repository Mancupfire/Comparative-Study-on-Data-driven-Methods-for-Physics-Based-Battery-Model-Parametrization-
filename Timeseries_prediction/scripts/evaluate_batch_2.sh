#!/usr/bin/env bash
# Re-evaluate every trained Batch 2 time-series (case, model) in a run dir.
#
# Usage:
#   bash scripts/evaluate_batch_2.sh <run_id>
#   bash scripts/evaluate_batch_2.sh batch2_full_001
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:?usage: evaluate_batch_2.sh <run_id>}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_ROOT="data/Data_Batch_2_cleaned"
OUT_ROOT="outputs/Data_Batch_2/time_series/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/time_series/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "[eval] run_id=$RUN_ID  out_root=$OUT_ROOT  GPU=$CUDA_VISIBLE_DEVICES"
python scripts/batch2_preflight.py \
  --data-dir "$DATA_ROOT" --output-root "$OUT_ROOT" --log-root "$LOG_DIR" --allow-resume

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/evaluate_batch2.py \
  --data-root "$DATA_ROOT" --run-dir "$OUT_ROOT" \
  --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
  --device auto 2>&1 | tee "$LOG_DIR/evaluate.log"
echo "[eval] done."
