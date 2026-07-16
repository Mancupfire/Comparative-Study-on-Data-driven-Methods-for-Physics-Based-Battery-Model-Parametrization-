#!/usr/bin/env bash
# Verify an error-metric run: required CSVs/figures exist, and every saved model
# reloads and produces finite inference on the test features.
#   usage: check_lhs_error_metrics.sh [RUN_DIR]
# Default RUN_DIR: latest smoke run, else latest full run.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  for pointer in LATEST_ERROR_METRICS_SMOKE_RUN.txt LATEST_ERROR_METRICS_RUN.txt; do
    f="$PROJECT_ROOT/outputs/lhs_1000_seed42/$pointer"
    [[ -f "$f" ]] && { RUN_DIR="$(cat "$f")"; break; }
  done
fi
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "Run dir not found: '$RUN_DIR'" >&2; exit 1; }
echo "Checking run: $RUN_DIR"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
RUN_DIR="$RUN_DIR" "$PYTHON" - <<'PY'
import json, os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path("/data1/minhntn/nhatminh/VinFast/Timeseries_prediction")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import lhs_error_metrics_train as P

run = Path(os.environ["RUN_DIR"])
fail = []

# 1) Required output files.
req = ["metrics/model_metrics.csv", "metrics/model_ranking.csv",
       "metrics/model_timing.csv", "metrics/per_case_metrics.csv",
       "metrics/error_metric_predictions.csv", "figures/model_ranking_heatmap.png",
       "artifacts/scalers.json", "artifacts/hyperparameters.json", "SUMMARY.md"]
for r in req:
    if not (run / r).exists():
        fail.append(f"missing {r}")
print("[check] required files present:", not fail)

# 2) Per-model figures.
cfg = json.loads((run / "run_config.json").read_text())["resolved"]
models = cfg["models"]
for m in models:
    for fig in (f"{m}_voltage_parity.png", f"{m}_temperature_parity.png", f"{m}_residuals.png"):
        if not (run / "figures" / fig).exists():
            fail.append(f"missing figure {fig}")
print("[check] per-model figures present:", not any('figure' in f for f in fail))

# 3) Metrics are finite; ranking has one row per model.
mm = pd.read_csv(run / "metrics/model_metrics.csv")
if len(mm) != len(models):
    fail.append(f"model_metrics rows {len(mm)} != models {len(models)}")
for col in ["v_rmse", "t_rmse", "macro_rmse"]:
    if not np.all(np.isfinite(mm[col].to_numpy())):
        fail.append(f"non-finite {col} in model_metrics")

# 4) Reload every saved model and run inference on real test features.
scal = json.loads((run / "artifacts/scalers.json").read_text())
meta, X_raw, Y_raw, feats = P.load_features_targets(REPO / "data/lhs_1000_seed42")
Xs = (X_raw - np.array(scal["x_mean"])) / np.array(scal["x_scale"])
xin = Xs[:64].astype(np.float32)
device = torch.device("cpu")
y_mean, y_scale = np.array(scal["y_mean"]), np.array(scal["y_scale"])

def reload_neural(name):
    import joblib  # noqa
    if name == "deep_ensemble_mlp":
        paths = sorted(glob.glob(str(run / "models" / f"{name}_member*.pt")))
        preds = []
        for k, p in enumerate(paths):
            m = P.build_neural("deep_ensemble_mlp", xin.shape[1],
                               _cfg(), 42 + k)
            m.load_state_dict(torch.load(p, map_location=device)); m.eval()
            preds.append(P.neural_predict(m, xin, device))
        return np.stack(preds).mean(0)
    m = P.build_neural(name, xin.shape[1], _cfg(), 42)
    m.load_state_dict(torch.load(run / "models" / f"{name}_best.pt", map_location=device)); m.eval()
    return P.neural_predict(m, xin, device)

class _C:  # minimal cfg for build_neural
    hidden_size=cfg["hidden_size"]; dropout=cfg["dropout"]
def _cfg(): return _C()

import joblib
for m in models:
    try:
        if m in P.NEURAL_MODELS:
            out = reload_neural(m)
        elif m == "catboost":
            from catboost import CatBoostRegressor
            mdl = CatBoostRegressor(); mdl.load_model(str(run / "models" / f"{m}_best.cbm"))
            out = P.tree_predict(m, mdl, xin)
        elif m == "xgboost":
            mdl = joblib.load(run / "models" / f"{m}_best.joblib")
            out = P.tree_predict(m, mdl, xin)
        else:
            mdl = joblib.load(run / "models" / f"{m}_best.joblib")
            out = P.tree_predict(m, mdl, xin)
        if out.shape != (len(xin), 2) or not np.all(np.isfinite(out)):
            fail.append(f"{m}: bad reload inference shape/finite")
        else:
            print(f"[check] reload+inference OK: {m} -> {out.shape}")
    except Exception as e:
        fail.append(f"{m}: reload failed: {type(e).__name__}: {e}")

if fail:
    print("\nCHECK FAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("\nCHECK PASSED: all files, figures, metrics and model reloads verified.")
PY
