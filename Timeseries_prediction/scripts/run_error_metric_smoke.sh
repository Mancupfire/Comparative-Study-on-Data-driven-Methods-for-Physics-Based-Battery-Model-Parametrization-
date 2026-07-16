#!/usr/bin/env bash
# Task B (error-metric surrogate) SMOKE test: ExtraTrees + MLP, reduced settings.
#
# Writes ONLY to outputs_smoke/Data_Batch_2/error_metric/<run_id>/ and logs to
# logs/Data_Batch_2/error_metric/<run_id>/. Never touches Batch 1 or Task A.
#
# Usage: bash scripts/run_error_metric_smoke.sh [run_id]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-errmetric_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATA_DIR="data/Data_Batch_2"
SMOKE_ROOT="outputs_smoke/Data_Batch_2/error_metric/${RUN_ID}"
LOG_DIR="logs/Data_Batch_2/error_metric/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "============ TASK B (error-metric) SMOKE ============"
echo "run_id     = $RUN_ID"
echo "data_dir   = $DATA_DIR"
echo "smoke_root = $SMOKE_ROOT"
echo "models     = extratrees mlp (reduced: 50 trees / 5 epochs)"
echo "====================================================="

# Isolation guard (raw Batch 2 data dir + Data_Batch_2 output namespace).
python scripts/batch2_preflight.py \
  --data-dir "$DATA_DIR" --output-root "$SMOKE_ROOT" --log-root "$LOG_DIR"

LOG="$LOG_DIR/smoke.log"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_error_metric.py \
  --data-dir "$DATA_DIR" --output-root "$SMOKE_ROOT" \
  --models extratrees mlp \
  --epochs 5 --n-estimators 50 --seed 42 --device auto \
  2>&1 | tee "$LOG"

echo "[task-b smoke] DONE. Artifacts: $SMOKE_ROOT (run_manifest.json has join + leakage checks)"
