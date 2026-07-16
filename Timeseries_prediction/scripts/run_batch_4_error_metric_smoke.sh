#!/usr/bin/env bash
# Batch 4 ERROR-METRIC smoke: ExtraTrees + MLP, reduced settings (50 trees / 5 epochs).
# Writes ONLY to outputs_smoke/Data_Batch_4/error_metric/<run_id>/.
# Never touches Batch 1/2/3 or the time-series task.
#
# Usage: bash scripts/run_batch_4_error_metric_smoke.sh [run_id]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch4_em_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"   # single free GPU

DATA_DIR="data/Data_Batch_4_raw"          # symlink -> generate_training_data_1000samples (immutable)
SMOKE_ROOT="outputs_smoke/Data_Batch_4/error_metric/${RUN_ID}"
LOG_DIR="logs/Data_Batch_4/error_metric/${RUN_ID}"
mkdir -p "$LOG_DIR"

echo "============ BATCH 4 TASK B (error-metric) SMOKE ============"
echo "run_id     = $RUN_ID"
echo "data_dir   = $DATA_DIR"
echo "smoke_root = $SMOKE_ROOT"
echo "models     = extratrees mlp (reduced)"
echo "============================================================"

python scripts/batch_preflight.py --dataset-name Data_Batch_4 \
  --data-dir "$DATA_DIR" --output-root "$SMOKE_ROOT" --log-root "$LOG_DIR"

LOG="$LOG_DIR/smoke.log"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python scripts/train_error_metric.py \
  --data-dir "$DATA_DIR" --output-root "$SMOKE_ROOT" --dataset-name Data_Batch_4 \
  --models extratrees mlp \
  --epochs 5 --n-estimators 50 --seed 42 --device auto \
  2>&1 | tee "$LOG"

echo "[task-b smoke] DONE. Artifacts: $SMOKE_ROOT (run_manifest.json has join + leakage checks)"
