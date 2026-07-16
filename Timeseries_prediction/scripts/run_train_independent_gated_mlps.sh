#!/usr/bin/env bash
# Train the twelve independent scalar Gated-MLP RMSE surrogates
# (6 discharge conditions x 2 error metrics) and run the inference example.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA_DIR="${DATA_DIR:-ann_rmse_training_2500_physics_aligned}"
OUT_DIR="${OUT_DIR:-ann_rmse_training_2500_physics_aligned/gated_mlp_12models_results}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

# Optional smoke mode: SMOKE=1 ./scripts/run_train_independent_gated_mlps.sh
EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
  OUT_DIR="${OUT_DIR}_smoke"
  EXTRA="--max-epochs 6 --patience 6"
fi

python -m src.gated_mlp_independent.train_models \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --seed "$SEED" \
  --device "$DEVICE" \
  $EXTRA

echo "== inference example (single held-out vector) =="
python - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path
ex = Path(sys.argv[1]) / "example_unseen_prediction.json"
print(ex.read_text())
PY
