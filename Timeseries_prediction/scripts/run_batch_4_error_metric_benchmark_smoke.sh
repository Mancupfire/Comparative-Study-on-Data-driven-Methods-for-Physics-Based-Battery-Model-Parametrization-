#!/usr/bin/env bash
# Batch 4 ERROR-METRIC BENCHMARK (12 families) — SMOKE.
# All 12 model families, ONE seed, tiny budget. Confirms forward pass, training,
# checkpoint saving, prediction export, metrics export and no NaN/Inf.
# Writes ONLY to outputs_smoke/. Never touches Batch 1/2/3/4 full outputs.
#
# Usage: bash scripts/run_batch_4_error_metric_benchmark_smoke.sh [run_id]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-em_bench_smoke_$(date +%Y%m%d_%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CONFIG="configs/batch_4/error_metric_benchmark_smoke.yaml"
LOG_DIR="logs/Data_Batch_4/error_metric_benchmark/${RUN_ID}"
RUN_DIR="outputs_smoke/Data_Batch_4/error_metric_benchmark/${RUN_ID}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/smoke.log"

echo "============ BATCH 4 ERROR-METRIC BENCHMARK — SMOKE ============"
echo "run_id = $RUN_ID"
echo "config = $CONFIG"
echo "out    = $RUN_DIR  (outputs_smoke ONLY)"
echo "gpu    = CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "==============================================================="

python -m src.error_metric_benchmark.run \
  --config "$CONFIG" --run-id "$RUN_ID" --smoke 2>&1 | tee "$LOG"

# Summary tables for the smoke run.
python -m src.error_metric_benchmark.summarize "$RUN_DIR" 2>&1 | tee -a "$LOG"

echo "[smoke] DONE. Artifacts under: $RUN_DIR"
