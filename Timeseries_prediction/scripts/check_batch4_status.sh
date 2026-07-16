#!/usr/bin/env bash
set -uo pipefail

ROOT="/mnt/disk1/backup_user/minh.ntn/VInFast_BatteryPrediction/Timeseries_prediction"
cd "$ROOT" || exit 1

EXPECTED_TS=84
EXPECTED_EM=2

# Optionally pass a RUN_ID:
# bash scripts/check_batch4_status.sh batch4_full_20260621_120000
RUN_ID="${1:-}"

if [[ -z "$RUN_ID" ]]; then
    RUN_ID="$(
        {
            find outputs/Data_Batch_4/time_series_downsampled_160 \
                -mindepth 1 -maxdepth 1 -type d \
                -name 'batch4_full_*' \
                -printf '%T@ %f\n' 2>/dev/null

            find logs/Data_Batch_4/time_series_downsampled_160 \
                -mindepth 1 -maxdepth 1 -type d \
                -name 'batch4_full_*' \
                -printf '%T@ %f\n' 2>/dev/null
        } |
        sort -nr |
        head -1 |
        cut -d' ' -f2-
    )"
fi

if [[ -z "$RUN_ID" ]]; then
    echo "ERROR: No Batch 4 full run found."
    exit 1
fi

TS_OUT="outputs/Data_Batch_4/time_series_downsampled_160/$RUN_ID"
EM_OUT="outputs/Data_Batch_4/error_metric/$RUN_ID"

TS_LOG_DIR="logs/Data_Batch_4/time_series_downsampled_160/$RUN_ID"
EM_LOG_DIR="logs/Data_Batch_4/error_metric/$RUN_ID"
ROOT_LOG_DIR="logs/Data_Batch_4/$RUN_ID"

find_log() {
    local result=""

    for candidate in \
        "$TS_LOG_DIR/full_train.log" \
        "$TS_LOG_DIR/console.log" \
        "$ROOT_LOG_DIR/console.log"
    do
        if [[ -f "$candidate" ]]; then
            result="$candidate"
            break
        fi
    done

    if [[ -z "$result" ]]; then
        result="$(
            find logs/Data_Batch_4 -type f \
                \( -name '*.log' -o -name '*.out' \) \
                -path "*${RUN_ID}*" \
                -printf '%T@ %p\n' 2>/dev/null |
            sort -nr |
            head -1 |
            cut -d' ' -f2-
        )"
    fi

    printf '%s' "$result"
}

MAIN_LOG="$(find_log)"

PROCESS_LINES="$(
    pgrep -af "$RUN_ID" 2>/dev/null |
    grep -v 'check_batch4_status.sh' || true
)"

if [[ -n "$PROCESS_LINES" ]]; then
    PROCESS_STATUS="RUNNING"
else
    PROCESS_STATUS="NOT RUNNING"
fi

TS_DONE=0
if [[ -d "$TS_OUT/metrics" ]]; then
    TS_DONE="$(
        find "$TS_OUT/metrics" -type f -name 'metrics.json' 2>/dev/null |
        wc -l
    )"
fi

EM_DONE=0
if [[ -d "$EM_OUT/metrics" ]]; then
    EM_DONE="$(
        find "$EM_OUT/metrics" -type f -name 'metrics.json' 2>/dev/null |
        wc -l
    )"
fi

ERROR_COUNT=0
if [[ -n "$MAIN_LOG" && -f "$MAIN_LOG" ]]; then
    ERROR_COUNT="$(
        grep -iEc \
        'Traceback|CUDA out of memory|OutOfMemory|RuntimeError|FAILED|nan loss|Killed' \
        "$MAIN_LOG" 2>/dev/null || true
    )"
fi

SUMMARY_OK="NO"
if [[ -f "$TS_OUT/metrics_summary.csv" \
   && -f "$TS_OUT/metrics_by_model.csv" \
   && -f "$TS_OUT/experiment_summary.md" ]]; then
    SUMMARY_OK="YES"
fi

FINAL_STATUS="INCOMPLETE"

if [[ "$TS_DONE" -ge "$EXPECTED_TS" \
   && "$EM_DONE" -ge "$EXPECTED_EM" \
   && "$SUMMARY_OK" == "YES" ]]; then
    FINAL_STATUS="COMPLETED"
elif [[ "$PROCESS_STATUS" == "RUNNING" ]]; then
    FINAL_STATUS="RUNNING"
elif [[ "$ERROR_COUNT" -gt 0 ]]; then
    FINAL_STATUS="FAILED_OR_INTERRUPTED"
else
    FINAL_STATUS="STOPPED_OR_WAITING_FOR_SUMMARY"
fi

echo
echo "======================================================================"
echo "BATCH 4 STATUS"
echo "======================================================================"
echo "Run ID                : $RUN_ID"
echo "Overall status        : $FINAL_STATUS"
echo "Process status        : $PROCESS_STATUS"
echo "Time-series completed : $TS_DONE / $EXPECTED_TS"
echo "Error-metric completed: $EM_DONE / $EXPECTED_EM"
echo "Summary files ready   : $SUMMARY_OK"
echo "Detected errors       : $ERROR_COUNT"
echo "Main log              : ${MAIN_LOG:-NOT FOUND}"
echo "Time-series output    : $TS_OUT"
echo "Error-metric output   : $EM_OUT"
echo "======================================================================"

echo
echo "===== RUNNING PROCESS ====="
if [[ -n "$PROCESS_LINES" ]]; then
    echo "$PROCESS_LINES"
else
    echo "No process found matching $RUN_ID"
fi

echo
echo "===== GPU STATUS ====="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
    --format=csv,noheader 2>/dev/null || nvidia-smi

echo
echo "===== COMPLETED MODELS BY CASE ====="
if [[ -d "$TS_OUT/metrics" ]]; then
    find "$TS_OUT/metrics" -type f -name 'metrics.json' 2>/dev/null |
    awk -F/ '
    {
        if (NF >= 3) {
            model=$(NF-1)
            count[model]++
        }
    }
    END {
        for (model in count) {
            printf "%-20s %3d\n", model, count[model]
        }
    }' | sort
else
    echo "metrics directory not found yet."
fi

echo
echo "===== ERROR-METRIC MODELS ====="
if [[ -d "$EM_OUT/metrics" ]]; then
    find "$EM_OUT/metrics" -type f -name 'metrics.json' -print
else
    echo "No error-metric results found yet."
fi

echo
echo "===== RECENT LOG ====="
if [[ -n "$MAIN_LOG" && -f "$MAIN_LOG" ]]; then
    tail -n 40 "$MAIN_LOG"
else
    echo "No log found."
fi

echo
echo "===== ERROR SCAN ====="
if [[ -n "$MAIN_LOG" && -f "$MAIN_LOG" ]]; then
    grep -niE \
    'Traceback|CUDA out of memory|OutOfMemory|RuntimeError|FAILED|nan loss|Killed' \
    "$MAIN_LOG" |
    tail -n 50 || echo "No serious errors detected."
else
    echo "No log to inspect."
fi

echo
echo "===== SUMMARY FILES ====="
find "$TS_OUT" "$EM_OUT" -maxdepth 2 -type f \
    \( -name 'metrics_summary.csv' \
       -o -name 'metrics_by_model.csv' \
       -o -name 'metrics_by_target.csv' \
       -o -name 'average_ranking.csv' \
       -o -name 'experiment_summary.md' \) \
    -print 2>/dev/null || true

echo
echo "===== DISK USAGE ====="
du -sh \
    "$TS_OUT" \
    "$EM_OUT" \
    "$TS_LOG_DIR" \
    "$EM_LOG_DIR" 2>/dev/null || true

df -h /mnt/disk1

echo
echo "======================================================================"
case "$FINAL_STATUS" in
    COMPLETED)
        echo "Batch 4 fully completed."
        ;;
    RUNNING)
        echo "Batch 4 still running. Do not launch a duplicate full run."
        ;;
    FAILED_OR_INTERRUPTED)
        echo "Run may have errored or been interrupted. Check ERROR SCAN and log."
        ;;
    *)
        echo "Process not running but artifacts incomplete. Inspect or resume."
        ;;
esac
echo "======================================================================"
