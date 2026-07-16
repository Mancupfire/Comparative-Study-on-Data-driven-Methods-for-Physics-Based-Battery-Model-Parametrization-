#!/usr/bin/env bash
# Batch 4 ERROR-METRIC BENCHMARK (12 families) — FULL multi-seed run.
# All 12 families x 3 seeds. Safe resume: completed (model,seed) combos are
# skipped and never overwritten. Exits nonzero on any failure.
# Writes ONLY to outputs/Data_Batch_4/error_metric_benchmark/<RUN_ID>/.
# Never touches Batch 1/2/3 or the Batch-4 time-series / two-model error-metric runs.
#
# Usage:
#   bash scripts/run_batch_4_error_metric_benchmark_full.sh <RUN_ID> [protocol]
#   protocol: grouped_holdout (default) | legacy_reproduction
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${1:-batch4_em_bench_$(date +%Y%m%d_%H%M%S)}"
PROTOCOL="${2:-grouped_holdout}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"

CONFIG="configs/batch_4/error_metric_benchmark_full.yaml"
RUN_DIR="outputs/Data_Batch_4/error_metric_benchmark/${RUN_ID}"
LOG_DIR="logs/Data_Batch_4/error_metric_benchmark/${RUN_ID}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/full_${PROTOCOL}.log"

echo "============ BATCH 4 ERROR-METRIC BENCHMARK — FULL ============"
echo "run_id   = $RUN_ID"
echo "protocol = $PROTOCOL"
echo "config   = $CONFIG"
echo "out      = $RUN_DIR"
echo "log      = $LOG"
echo "gpu      = CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=============================================================="

python -m src.error_metric_benchmark.run \
  --config "$CONFIG" --run-id "$RUN_ID" --protocol "$PROTOCOL" 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"

# Always (re)build summary tables for whatever has completed.
python -m src.error_metric_benchmark.summarize "$RUN_DIR" 2>&1 | tee -a "$LOG" || true

if [[ "$STATUS" -ne 0 ]]; then
  echo "[full] FAILED (see failures in $RUN_DIR/run_manifest.json)"
  exit "$STATUS"
fi
echo "[full] DONE. Artifacts under: $RUN_DIR"
