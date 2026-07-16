#!/usr/bin/env bash
# Compare a Batch 2 time-series run with a Batch 3 time-series run (read-only).
# Writes ONLY to outputs/comparisons/Batch_2_vs_Batch_3/<comparison_id>/.
#
# Usage:
#   bash scripts/compare_batch_2_vs_batch_3.sh <batch2_run_id> <batch3_run_id> [comparison_id]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

B2_RUN="${1:?usage: compare_batch_2_vs_batch_3.sh <batch2_run_id> <batch3_run_id> [comparison_id]}"
B3_RUN="${2:?need batch3 run id}"
CMP_ID="${3:-cmp_$(date +%Y%m%d_%H%M%S)}"

B2_DIR="outputs/Data_Batch_2/time_series_downsampled_160/${B2_RUN}"
B3_DIR="outputs/Data_Batch_3/time_series_downsampled_160/${B3_RUN}"

echo "[compare] Batch2=$B2_DIR  Batch3=$B3_DIR  id=$CMP_ID"
python scripts/compare_batch2_batch3.py \
  --batch2-run-dir "$B2_DIR" --batch3-run-dir "$B3_DIR" --comparison-id "$CMP_ID"
echo "[compare] done -> outputs/comparisons/Batch_2_vs_Batch_3/${CMP_ID}/"
