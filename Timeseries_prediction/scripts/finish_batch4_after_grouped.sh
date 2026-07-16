#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/disk1/backup_user/minh.ntn/VInFast_BatteryPrediction/Timeseries_prediction"

GROUP_RUN_ID="batch4_em_grouped_20260622_110539"
GROUP_PID="2305051"

TS_RUN_ID="batch4_full_20260621_140149"
POLL_SECONDS=300
EXPECTED_COMBINATIONS=36

cd "$ROOT"

GROUP_OUT="outputs/Data_Batch_4/error_metric_benchmark/$GROUP_RUN_ID"
GROUP_LOG_DIR="logs/Data_Batch_4/error_metric_benchmark/$GROUP_RUN_ID"

mkdir -p "$GROUP_LOG_DIR"
mkdir -p reports/Data_Batch_4/error_metric_benchmark

echo "============================================================"
echo "BATCH 4 AUTOMATIC COMPLETION PIPELINE"
echo "============================================================"
echo "Grouped run : $GROUP_RUN_ID"
echo "Grouped PID : $GROUP_PID"
echo "TS run      : $TS_RUN_ID"
echo "Started     : $(date)"
echo "============================================================"

echo
echo "[1/8] Waiting for grouped-holdout benchmark..."

while kill -0 "$GROUP_PID" 2>/dev/null
do
    echo
    echo "----- $(date) -----"
    bash scripts/check_batch4_error_metric_benchmark.sh "$GROUP_RUN_ID" || true
    echo "Next check in ${POLL_SECONDS}s..."
    sleep "$POLL_SECONDS"
done

sleep 10

echo
echo "[2/8] Grouped process stopped. Checking outputs..."

bash scripts/check_batch4_error_metric_benchmark.sh "$GROUP_RUN_ID" || true

GROUP_METRICS_COUNT=$(
    find "$GROUP_OUT" -type f -name "metrics.json" 2>/dev/null |
    wc -l |
    tr -d ' '
)

echo "Grouped metrics files found: $GROUP_METRICS_COUNT"

if [ "$GROUP_METRICS_COUNT" -lt "$EXPECTED_COMBINATIONS" ]
then
    echo "ERROR: Grouped benchmark is incomplete."
    echo "Expected at least $EXPECTED_COMBINATIONS completed model-seed combinations."
    echo "Inspect:"
    echo "  $GROUP_LOG_DIR"
    exit 1
fi

echo
echo "[3/8] Summarizing grouped-holdout benchmark..."

bash scripts/summarize_batch_4_error_metric_benchmark.sh "$GROUP_RUN_ID"

echo
echo "[4/8] Validating grouped-holdout benchmark..."

python3 scripts/validate_batch_4_error_metric_benchmark.py \
    --em-run-id "$GROUP_RUN_ID"

echo
echo "[5/8] Starting legacy-reproduction benchmark..."

LEGACY_RUN_ID="batch4_em_legacy_$(date +%Y%m%d_%H%M%S)"
LEGACY_OUT="outputs/Data_Batch_4/error_metric_benchmark/$LEGACY_RUN_ID"
LEGACY_LOG_DIR="logs/Data_Batch_4/error_metric_benchmark/$LEGACY_RUN_ID"

mkdir -p "$LEGACY_LOG_DIR"

cat > reports/Data_Batch_4/error_metric_benchmark/latest_run_ids.env <<EOF
GROUP_RUN_ID=$GROUP_RUN_ID
LEGACY_RUN_ID=$LEGACY_RUN_ID
TS_RUN_ID=$TS_RUN_ID
EOF

echo "Legacy run ID: $LEGACY_RUN_ID"

env CUDA_VISIBLE_DEVICES=7 \
    bash scripts/run_batch_4_error_metric_benchmark_full.sh \
    "$LEGACY_RUN_ID" \
    legacy_reproduction

echo
echo "Legacy benchmark process finished."

bash scripts/check_batch4_error_metric_benchmark.sh "$LEGACY_RUN_ID" || true

LEGACY_METRICS_COUNT=$(
    find "$LEGACY_OUT" -type f -name "metrics.json" 2>/dev/null |
    wc -l |
    tr -d ' '
)

echo "Legacy metrics files found: $LEGACY_METRICS_COUNT"

if [ "$LEGACY_METRICS_COUNT" -lt "$EXPECTED_COMBINATIONS" ]
then
    echo "ERROR: Legacy benchmark is incomplete."
    echo "Expected at least $EXPECTED_COMBINATIONS completed model-seed combinations."
    echo "Inspect:"
    echo "  $LEGACY_LOG_DIR"
    exit 1
fi

echo
echo "[6/8] Summarizing legacy benchmark..."

bash scripts/summarize_batch_4_error_metric_benchmark.sh "$LEGACY_RUN_ID"

echo
echo "[7/8] Exporting existing time-series predictions..."

env CUDA_VISIBLE_DEVICES=7 \
    python3 scripts/export_batch_4_time_series_predictions.py

TS_PRED_DIR="outputs/Data_Batch_4/time_series_downsampled_160/$TS_RUN_ID/predictions"

TS_PRED_COUNT=$(
    find "$TS_PRED_DIR" -type f -name "test_predictions.csv" 2>/dev/null |
    wc -l |
    tr -d ' '
)

echo "Time-series prediction files found: $TS_PRED_COUNT"

if [ "$TS_PRED_COUNT" -lt 84 ]
then
    echo "WARNING: Expected 84 prediction files but found $TS_PRED_COUNT."
    echo "The final builder will still run, but inspect checkpoint compatibility."
fi

echo
echo "[8/8] Building final publication-ready tables and figures..."

python3 scripts/build_batch_4_final_results.py \
    --em-run-id "$GROUP_RUN_ID" \
    --legacy-run-id "$LEGACY_RUN_ID"

FINAL_DIR="reports/Data_Batch_4/final_results/$GROUP_RUN_ID"
FINAL_ZIP="${FINAL_DIR}.zip"

echo
echo "============================================================"
echo "PIPELINE COMPLETED"
echo "============================================================"
echo "Grouped run : $GROUP_RUN_ID"
echo "Legacy run  : $LEGACY_RUN_ID"
echo "Final folder: $FINAL_DIR"
echo "Final ZIP   : $FINAL_ZIP"
echo "Completed   : $(date)"
echo "============================================================"

echo
echo "Generated tables:"
find "$FINAL_DIR/tables" -type f 2>/dev/null | sort || true

echo
echo "Generated figures:"
find "$FINAL_DIR/figures" -type f 2>/dev/null | sort || true

echo
echo "ZIP:"
ls -lh "$FINAL_ZIP" 2>/dev/null || true
