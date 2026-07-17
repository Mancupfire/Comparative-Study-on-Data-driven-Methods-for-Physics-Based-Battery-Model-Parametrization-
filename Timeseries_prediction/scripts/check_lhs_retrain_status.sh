#!/usr/bin/env bash
# Verify an LHS retrain pair (time-series run + error-metric run):
#   * required CSVs / JSON provenance / figures exist for both branches,
#   * timing fields are present and non-negative,
#   * every saved model reloads and produces finite inference.
#
# Usage:
#   check_lhs_retrain_status.sh [TS_RUN_DIR] [EM_RUN_DIR]
# With no arguments it resolves the latest smoke pointers, then the latest full
# pointers.
set -Eeuo pipefail

PROJECT_ROOT="/data1/minhntn/nhatminh/VinFast/Timeseries_prediction"
PYTHON="${PYTHON:-/data1/minhntn/miniconda3/envs/ai3090/bin/python}"

resolve() {  # $1 smoke-pointer $2 full-pointer
  for p in "$1" "$2"; do
    f="$PROJECT_ROOT/outputs/lhs_1000_seed42/$p"
    [[ -f "$f" ]] && { cat "$f"; return 0; }
  done
  return 0
}

TS_RUN="${1:-$(resolve LATEST_OFFICIAL_SMOKE_RUN.txt LATEST_OFFICIAL_RUN.txt)}"
EM_RUN="${2:-$(resolve LATEST_ERROR_METRICS_SMOKE_RUN.txt LATEST_ERROR_METRICS_RUN.txt)}"

[[ -n "$TS_RUN" && -d "$TS_RUN" ]] || { echo "Time-series run dir not found: '$TS_RUN'" >&2; exit 1; }
[[ -n "$EM_RUN" && -d "$EM_RUN" ]] || { echo "Error-metric run dir not found: '$EM_RUN'" >&2; exit 1; }
echo "Time-series run: $TS_RUN"
echo "Error-metric run: $EM_RUN"

cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
TS_RUN="$TS_RUN" EM_RUN="$EM_RUN" "$PYTHON" - <<'PY'
import json, os, sys, glob
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path("/data1/minhntn/nhatminh/VinFast/Timeseries_prediction")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import emergency_lhs_train as TS
import lhs_error_metrics_train as EM

ts, em = Path(os.environ["TS_RUN"]), Path(os.environ["EM_RUN"])
fail = []

def need(run, rel):
    if not (run / rel).exists():
        fail.append(f"{run.name}: missing {rel}")

# ---- required files (both branches) -------------------------------------- #
common = ["metrics/model_metrics.csv", "metrics/model_ranking.csv",
          "metrics/model_timing.csv", "metrics/per_case_metrics.csv",
          "metrics/predictions.csv", "run_config.json",
          "dataset_audit.json", "environment.json", "SUMMARY.md"]
for r in common:
    need(ts, r); need(em, r)

# time-series extras
for r in ["metrics/voltage_metrics.csv", "metrics/temperature_metrics.csv",
          "metrics/per_operation_metrics.csv", "metrics/per_c_rate_metrics.csv",
          "figures/model_ranking_heatmap.png"]:
    need(ts, r)

# ---- timing fields present and non-negative ------------------------------ #
TIMING_NONNEG = ["inference_seconds_total", "inference_ms_per_sequence",
                 "inference_ms_per_row", "throughput_sequences_per_second",
                 "test_batch_size", "total_training_seconds", "train_seconds",
                 "peak_gpu_memory_mb"]
TIMING_REQUIRED = ["device_name", "cuda_version", "gpu_name",
                   "throughput_sequences_per_second", "test_batch_size"]
for run in (ts, em):
    tp = run / "metrics/model_timing.csv"
    if not tp.exists():
        continue
    df = pd.read_csv(tp)
    for col in TIMING_REQUIRED:
        if col not in df.columns:
            fail.append(f"{run.name}: timing missing column {col}")
    for col in TIMING_NONNEG:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if vals.size and (vals < 0).any():
                fail.append(f"{run.name}: negative timing values in {col}")
print("[check] files + timing fields validated:", not fail)

# ---- reload + finite inference: error-metric models ---------------------- #
class _EMCfg:
    pass
emcfg_raw = json.loads((em / "run_config.json").read_text())["resolved"]
emcfg = _EMCfg(); emcfg.hidden_size = emcfg_raw["hidden_size"]; emcfg.dropout = emcfg_raw["dropout"]
meta, X_raw, Y_raw, feats = EM.load_features_targets(REPO / "data/lhs_1000_seed42")
scal = json.loads((em / "artifacts/scalers.json").read_text())
Xs = (X_raw - np.array(scal["x_mean"])) / np.array(scal["x_scale"])
xin = Xs[:64].astype(np.float32)
device = torch.device("cpu")
import joblib
for m in emcfg_raw["models"]:
    try:
        if m == "deep_ensemble_mlp":
            paths = sorted(glob.glob(str(em / "models" / f"{m}_member*.pt")))
            outs = []
            for k, p in enumerate(paths):
                mod = EM.build_neural(m, xin.shape[1], emcfg, 42 + k)
                mod.load_state_dict(torch.load(p, map_location=device)); mod.eval()
                outs.append(EM.neural_predict(mod, xin, device))
            out = np.stack(outs).mean(0)
        elif m in EM.NEURAL_MODELS:
            mod = EM.build_neural(m, xin.shape[1], emcfg, 42)
            mod.load_state_dict(torch.load(em / "models" / f"{m}_best.pt", map_location=device)); mod.eval()
            out = EM.neural_predict(mod, xin, device)
        elif m == "catboost":
            from catboost import CatBoostRegressor
            mdl = CatBoostRegressor(); mdl.load_model(str(em / "models" / f"{m}_best.cbm"))
            out = EM.tree_predict(m, mdl, xin)
        else:
            mdl = joblib.load(em / "models" / f"{m}_best.joblib")
            out = EM.tree_predict(m, mdl, xin)
        if out.shape != (len(xin), 2) or not np.all(np.isfinite(out)):
            fail.append(f"EM {m}: bad reload inference")
        else:
            print(f"[check] EM reload+inference OK: {m}")
    except Exception as e:
        fail.append(f"EM {m}: reload failed: {type(e).__name__}: {e}")

# ---- reload + finite inference: time-series models ----------------------- #
tscfg_raw = json.loads((ts / "run_config.json").read_text())
n_features = len(json.loads((ts / "artifacts/feature_names.json").read_text()))
seq_len = int(tscfg_raw["sequence_length"])
from types import SimpleNamespace
tscfg = SimpleNamespace(hidden_size=tscfg_raw["hidden_size"],
                        num_layers=tscfg_raw["num_layers"], dropout=tscfg_raw["dropout"])
xb = torch.zeros(4, n_features)
qb = torch.linspace(0, 1, seq_len).unsqueeze(0).repeat(4, 1)
for m in tscfg_raw["models"]:
    try:
        model = TS.build_model(m, n_features, tscfg)
        ckpt = torch.load(ts / "best_checkpoints" / f"{m}_best.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        with torch.no_grad():
            out = model(xb, qb)
        if out.shape != (4, seq_len, 2) or not torch.all(torch.isfinite(out)):
            fail.append(f"TS {m}: bad reload inference {tuple(out.shape)}")
        else:
            print(f"[check] TS reload+inference OK: {m}")
    except Exception as e:
        fail.append(f"TS {m}: reload failed: {type(e).__name__}: {e}")

if fail:
    print("\nCHECK FAILED:")
    for f in fail:
        print("  -", f)
    sys.exit(1)
print("\nCHECK PASSED: files, timing fields, figures, and model reloads verified for both branches.")
PY
