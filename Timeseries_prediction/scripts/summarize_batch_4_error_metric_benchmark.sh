#!/usr/bin/env bash
# Summarize a completed Batch 4 error-metric benchmark run.
# Generates under <RUN_DIR>/tables/:
#   metrics_by_model_and_seed.csv, metrics_by_model.csv, metrics_by_target.csv,
#   ranking_table.csv, experiment_summary.md, resolved_config.yaml, split_audit.json
#
# Usage:
#   bash scripts/summarize_batch_4_error_metric_benchmark.sh <RUN_ID> [--smoke]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE=0
RUN_ID=""
for a in "$@"; do
  if [[ "$a" == "--smoke" ]]; then SMOKE=1; else RUN_ID="$a"; fi
done
if [[ -z "$RUN_ID" ]]; then echo "Usage: $0 <RUN_ID> [--smoke]"; exit 1; fi

if [[ "$SMOKE" -eq 1 ]]; then
  RUN_DIR="outputs_smoke/Data_Batch_4/error_metric_benchmark/$RUN_ID"
else
  RUN_DIR="outputs/Data_Batch_4/error_metric_benchmark/$RUN_ID"
fi
[[ -d "$RUN_DIR" ]] || { echo "ERROR: run dir not found: $RUN_DIR"; exit 1; }

python -m src.error_metric_benchmark.summarize "$RUN_DIR"
echo "[summarize] DONE -> $RUN_DIR/tables/"
