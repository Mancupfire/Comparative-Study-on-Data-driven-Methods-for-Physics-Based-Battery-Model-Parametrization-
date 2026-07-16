#!/usr/bin/env bash
# Task B (error-metric surrogate) FULL training: ExtraTrees baseline + MLP main.
#
# Writes ONLY to outputs/Data_Batch_2/error_metric/<run_id>/ and logs to
# logs/Data_Batch_2/error_metric/<run_id>/. Never touches Batch 1 or Task A.
#
# Usage: bash scripts/run_error_metric_full.sh [run_id]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-errmetric_full_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="data/Data_Batch_2"
OUT_ROOT="outputs/Data_Batch_2/error_metric/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/error_metric/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "============ TASK B (error-metric) FULL ============"
echo "run_id    = $RUN_ID"
echo "data_dir  = $DATA_DIR"
echo "out_root  = $OUT_ROOT"
echo "models    = extratrees mlp"
echo "===================================================="

python scripts/batch2_preflight.py \
  --data-dir "$DATA_DIR" --output-root "$OUT_ROOT" --log-root "$LOG_DIR"

# Optional read-only audit alongside the run.
python scripts/audit_error_metric.py --data-dir "$DATA_DIR" \
  --out "outputs/Data_Batch_2/data_audit/error_metric/data_quality_summary.json" \
  2>&1 | tee "$LOG_DIR/audit.log"

LOG="$LOG_DIR/full_train.log"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_error_metric.py \
  --data-dir "$DATA_DIR" --output-root "$OUT_ROOT" \
  --models extratrees mlp \
  --epochs 200 --batch-size 256 --lr 1e-3 --weight-decay 1e-4 \
  --hidden-dim 128 --num-layers 3 --dropout 0.1 --patience 20 \
  --n-estimators 300 --seed 42 --device auto \
  2>&1 | tee "$LOG"

echo "[task-b full] DONE. Summary: $OUT_ROOT/run_manifest.json"
