"""Optional sklearn tree-ensemble baseline (NOT a deep-learning model).

This baseline maps the parameter vector directly to the response with an
ExtraTrees / RandomForest multi-output regressor.  It exists only as a
reference point; it does **not** replace, and does not automatically include,
the RNN/LSTM/BiLSTM sequence models, which require a specialised AutoML system.

Two output modes:

* full (default) : predict the entire ``2 * t_last`` curve and report the same
  curve metrics as the deep models.
* ``--reduced-output`` : predict 10 summary scalars
  ``[V_start, V_mid, V_end, V_min, V_mean, T_start, T_mid, T_end, T_max, T_mean]``
  (recommended for large ``t_last`` where full curves are expensive).

Example
-------
python scripts/train_automl_baseline.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --estimator extratrees --reduced-output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.data import load_aligned_case_data, split_indices
from src.metrics import compute_metrics
from src.predict import checkpoint_dir, metrics_dir, scaler_dir
from src.utils import ensure_dir, save_json


def _reduced_targets(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Summarise each curve into 10 scalars (see module docstring for order)."""
    mid = V.shape[1] // 2
    feats = [
        V[:, 0], V[:, mid], V[:, -1], V.min(axis=1), V.mean(axis=1),
        T[:, 0], T[:, mid], T[:, -1], T.max(axis=1), T.mean(axis=1),
    ]
    return np.stack(feats, axis=1)  # [N, 10]


REDUCED_NAMES = [
    "V_start", "V_mid", "V_end", "V_min", "V_mean",
    "T_start", "T_mid", "T_end", "T_max", "T_mean",
]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="sklearn tree-ensemble baseline.")
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--case-id", required=True)
    p.add_argument("--estimator", choices=["extratrees", "randomforest"], default="extratrees")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--reduced-output", action="store_true",
                   help="Predict 10 summary scalars instead of full curves.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p


def main() -> int:
    args = build_parser().parse_args()
    model_name = "automl_trees_reduced" if args.reduced_output else "automl_trees"

    case = load_aligned_case_data(args.data_root, args.case_id)
    train_idx, _val_idx, test_idx = split_indices(
        case.n_samples, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    # Scale X on the train split only (no leakage).
    x_scaler = StandardScaler().fit(case.X[train_idx])
    X = x_scaler.transform(case.X)

    estimator_cls = ExtraTreesRegressor if args.estimator == "extratrees" else RandomForestRegressor
    # These ensembles support multi-output regression natively.
    model = estimator_cls(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )

    t_last = case.t_last
    if args.reduced_output:
        Y = np.concatenate([_reduced_targets(case.V, case.T)], axis=1)  # [N, 10]
    else:
        Y = np.concatenate([case.V, case.T], axis=1)                    # [N, 2*t_last]

    print(f"[{args.case_id}/{model_name}] fitting {args.estimator} "
          f"on {len(train_idx)} samples, target dim={Y.shape[1]} ...")
    model.fit(X[train_idx], Y[train_idx])
    pred = model.predict(X[test_idx])

    # Persist artefacts alongside the deep models.
    sdir = ensure_dir(scaler_dir(args.outputs_dir, args.case_id, model_name))
    joblib.dump(x_scaler, sdir / "x_scaler.joblib")
    cdir = ensure_dir(checkpoint_dir(args.outputs_dir, args.case_id, model_name))
    joblib.dump(model, cdir / "model.joblib")
    mdir = ensure_dir(metrics_dir(args.outputs_dir, args.case_id, model_name))

    if args.reduced_output:
        # Per-scalar MAE/RMSE (curve metrics are undefined for summary scalars).
        true, p = Y[test_idx], pred
        per = {
            name: {
                "MAE": float(np.mean(np.abs(true[:, i] - p[:, i]))),
                "RMSE": float(np.sqrt(np.mean((true[:, i] - p[:, i]) ** 2))),
            }
            for i, name in enumerate(REDUCED_NAMES)
        }
        record = {
            "case_id": args.case_id, "model_name": model_name,
            "mode": "reduced", "estimator": args.estimator,
            "n_test": int(len(test_idx)), "per_target": per,
        }
    else:
        v_pred, t_pred = pred[:, :t_last], pred[:, t_last:]
        metrics = compute_metrics(case.V[test_idx], v_pred, case.T[test_idx], t_pred)
        record = {
            "case_id": args.case_id, "model_name": model_name,
            "mode": "full", "estimator": args.estimator,
            "split": "test", "n_samples": int(len(test_idx)),
            "t_last": t_last, **metrics,
        }

    save_json(record, mdir / "metrics.json")
    print(f"[{args.case_id}/{model_name}] metrics -> {mdir / 'metrics.json'}")
    print(record if args.reduced_output else
          {k: record[k] for k in ("RMSE_V", "R2_V", "RMSE_T", "R2_T")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
