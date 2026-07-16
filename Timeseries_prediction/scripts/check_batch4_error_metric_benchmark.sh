#!/usr/bin/env bash
# Status monitor for a Batch 4 error-metric benchmark run.
# Usage: bash scripts/check_batch4_error_metric_benchmark.sh [RUN_ID] [--smoke]
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

SMOKE=0
RUN_ID=""
for a in "$@"; do
  if [[ "$a" == "--smoke" ]]; then SMOKE=1; else RUN_ID="$a"; fi
done

if [[ "$SMOKE" -eq 1 ]]; then
  BASE="outputs_smoke/Data_Batch_4/error_metric_benchmark"
else
  BASE="outputs/Data_Batch_4/error_metric_benchmark"
fi
LOGBASE="logs/Data_Batch_4/error_metric_benchmark"

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(find "$BASE" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %f\n' 2>/dev/null \
            | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -z "$RUN_ID" ]]; then echo "ERROR: no run found under $BASE"; exit 1; fi

RUN_DIR="$BASE/$RUN_ID"
MANIFEST="$RUN_DIR/run_manifest.json"
FAMILIES=(ann mlp wide_deep_mlp attention_mlp gated_mlp residual_mlp \
          multitask_mlp deep_ensemble_mlp rnn lstm bilstm extratrees)

echo "=================== ERROR-METRIC BENCHMARK STATUS ==================="
echo "RUN_ID   : $RUN_ID"

PROTOCOL="n/a"; SEEDS="n/a"
if [[ -f "$MANIFEST" ]]; then
  PROTOCOL="$(python -c "import json;print(json.load(open('$MANIFEST')).get('protocol'))" 2>/dev/null)"
  SEEDS="$(python -c "import json;print(json.load(open('$MANIFEST')).get('seeds'))" 2>/dev/null)"
fi
echo "Protocol : $PROTOCOL"
echo "Seeds    : $SEEDS"
echo "Out dir  : $RUN_DIR"

# Active process?
if pgrep -af "src.error_metric_benchmark.run" | grep -q "$RUN_ID"; then
  echo "Process  : RUNNING"
  pgrep -af "src.error_metric_benchmark.run" | grep "$RUN_ID"
else
  echo "Process  : not running"
fi

# Completed / missing combos.
echo "--------------------------------------------------------------------"
echo "Completed (model/seed) combinations:"
done_n=0; miss_n=0
if [[ -f "$MANIFEST" ]]; then
  mapfile -t SEEDLIST < <(python -c "import json;print('\n'.join(map(str,json.load(open('$MANIFEST')).get('seeds',[]))))" 2>/dev/null)
else
  SEEDLIST=(42)
fi
for fam in "${FAMILIES[@]}"; do
  for s in "${SEEDLIST[@]}"; do
    if [[ -f "$RUN_DIR/metrics/$fam/seed$s/metrics.json" ]]; then
      done_n=$((done_n+1))
    else
      miss_n=$((miss_n+1)); echo "  MISSING: $fam/seed$s"
    fi
  done
done
echo "  completed=$done_n  missing=$miss_n"

# Failures.
echo "--------------------------------------------------------------------"
if [[ -f "$MANIFEST" ]]; then
  NF="$(python -c "import json;print(json.load(open('$MANIFEST')).get('n_failures',0))" 2>/dev/null)"
  echo "Failures (run_manifest): $NF"
  python -c "import json;[print('  ',f) for f in json.load(open('$MANIFEST')).get('failures',[])]" 2>/dev/null
fi

# Detected errors in logs.
LATEST_LOG="$(find "$LOGBASE/$RUN_ID" -name '*.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
echo "--------------------------------------------------------------------"
if [[ -n "${LATEST_LOG:-}" && -f "$LATEST_LOG" ]]; then
  echo "Log: $LATEST_LOG"
  echo "Detected error lines:"
  errs="$(grep -E 'Traceback|\[FAIL\]|FAILED|Exception|Error:|Errno' "$LATEST_LOG" | tail -5)"
  if [[ -n "$errs" ]]; then echo "$errs" | sed 's/^/  /'; else echo "  (none)"; fi
  echo "Recent log tail:"
  tail -8 "$LATEST_LOG" | sed 's/^/  /'
else
  echo "Log: (none found under $LOGBASE/$RUN_ID)"
fi

# Summary availability + output locations.
echo "--------------------------------------------------------------------"
for f in tables/ranking_table.csv tables/experiment_summary.md split_manifest.csv; do
  [[ -e "$RUN_DIR/$f" ]] && echo "Summary present : $RUN_DIR/$f" || echo "Summary missing : $RUN_DIR/$f"
done

# GPU.
echo "--------------------------------------------------------------------"
echo "GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>/dev/null | sed 's/^/  /' || echo "  nvidia-smi unavailable"
echo "===================================================================="
