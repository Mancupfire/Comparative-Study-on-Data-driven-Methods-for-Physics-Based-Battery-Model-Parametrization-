"""Write the Phase-4 reproducibility artifacts for a Batch 2 run directory.

Creates, inside <run-dir>:
    resolved_config.yaml      run_manifest.json     environment.txt
    git_commit.txt            model_inventory.json  data_summary.json
    split_summary.json

Usage
-----
python scripts/batch2_write_run_manifest.py \
    --config configs/batch_2/time_series.yaml \
    --run-dir outputs/Data_Batch_2/time_series/<run_id> \
    --log-dir logs/Data_Batch_2/time_series/<run_id> \
    --data-dir data/Data_Batch_2_cleaned \
    --run-id <run_id> --mode full
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no_git_repository"


def _pip_freeze() -> str:
    try:
        out = subprocess.run(["python", "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=120)
        return out.stdout
    except Exception as exc:
        return f"pip freeze failed: {exc}\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", default="full", choices=["full", "smoke"])
    p.add_argument("--audit-dir", default="outputs/Data_Batch_2/data_audit")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())

    # resolved_config.yaml
    resolved = dict(cfg)
    resolved.update({"run_id": args.run_id, "mode": args.mode,
                     "resolved_data_root": args.data_dir,
                     "resolved_output_root": str(run_dir),
                     "resolved_log_dir": args.log_dir,
                     "resolved_at": datetime.now(timezone.utc).isoformat()})
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # environment + git
    (run_dir / "environment.txt").write_text(_pip_freeze())
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    # model_inventory.json
    model_inventory = {
        "task": "time_series",
        "models": cfg.get("models", []),
        "notes": {
            "mlp": "point model, output 2*t_last (concat V|T)",
            "bayesian_mlp": "MLP + MC-Dropout (mc_samples) at eval",
            "rnn/lstm/bilstm/cnn/cnn_bilstm": "sequence models, [B,T,n_params+1]->[B,T,2]",
        },
        "also_available_in_repo_not_in_this_run": [
            "shared_* (one model across all cases) via scripts/train_shared_model.py",
            "automl_trees / automl_trees_reduced via scripts/train_automl_baseline.py",
        ],
        "error_metric_task": "BLOCKED — no pipeline in repo (see configs/batch_2/error_metric.yaml)",
    }
    (run_dir / "model_inventory.json").write_text(json.dumps(model_inventory, indent=2))

    # data_summary.json
    data_summary = {"data_root": args.data_dir}
    for fn in ("dataset_summary.json", "build_stats.json", "cleaning_manifest.json",
               "source_checksums.json"):
        fp = Path(args.data_dir) / fn
        if fp.is_file():
            try:
                data_summary[fn] = json.loads(fp.read_text())
            except Exception:
                data_summary[fn] = "unparseable"
    (run_dir / "data_summary.json").write_text(json.dumps(data_summary, indent=2, default=str))

    # split_summary.json (reuse the audit's split integrity if present)
    split_src = Path(args.audit_dir) / "time_series" / "split_integrity_report.json"
    if split_src.is_file():
        shutil.copy2(split_src, run_dir / "split_summary.json")
    else:
        (run_dir / "split_summary.json").write_text(json.dumps(
            {"grouping": "per-case sample-level 70/15/15 seed 42",
             "note": "audit split report not found"}, indent=2))

    # run_manifest.json
    manifest = {
        "dataset_name": cfg.get("dataset_name", "Data_Batch_2"),
        "task": "time_series",
        "data_directory": str(Path(args.data_dir).resolve()),
        "run_id": args.run_id,
        "mode": args.mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "models": cfg.get("models", []),
        "features": "12 static LHS parameters (parameter_sets.csv), per case; "
                    "sequence models add a normalized-time channel",
        "targets": ["voltage_v", "temperature_c"],
        "sequence_length": "per-case t_last (678..7030, native resolution)",
        "prediction_horizon": "full curve (static->curve surrogate; no autoregressive horizon)",
        "split_strategy": {"type": "per-case sample-wise", "ratios": [0.7, 0.15, 0.15],
                           "seed": cfg.get("seed", 42),
                           "scalers": "fit on train split only (X,V,T separately)"},
        "random_seeds": [cfg.get("seed", 42)],
        "metrics": ["MAE_V", "RMSE_V", "R2_V", "MaxError_V",
                    "MAE_T", "RMSE_T", "R2_T", "MaxError_T",
                    "voltage_end_mae", "temperature_end_mae", "temperature_peak_mae",
                    "voltage_curve_rmse_mean", "temperature_curve_rmse_mean"],
        "output_directory": str(run_dir.resolve()),
        "log_directory": str(Path(args.log_dir).resolve()),
        "loss": "MSE(V) + lambda_temp*MSE(T)",
        "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau(min,0.5,patience//3)",
        "model_selection": "best validation total loss",
        "deviations_from_batch_1": [
            "Time grid kept at native ~1s resolution (t_last up to 7030) per user "
            "instruction, vs Batch 1's downsampled ~150-180 grid.",
            "Different operating points & parameter identities (inherent to Batch 2 data).",
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote reproducibility artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
