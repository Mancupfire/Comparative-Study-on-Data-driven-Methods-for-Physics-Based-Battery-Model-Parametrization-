#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/disk1/backup_user/minh.ntn/VInFast_BatteryPrediction/Timeseries_prediction"
cd "$ROOT"

python scripts/export_discussed_visualizations.py \
  --root "$ROOT" \
  --out reports/Data_Batch_4/final_filtered_protocol/final_visualizations_v3

echo
echo "===== GENERATED ZIP FILES ====="
ls -lh \
  reports/Data_Batch_4/final_filtered_protocol/final_visualizations_v3/final_visualization_results.zip \
  reports/Data_Batch_4/final_filtered_protocol/final_visualizations_v3/colleague_results.zip
