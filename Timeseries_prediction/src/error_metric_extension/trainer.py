"""Tree-based trainers for the error-metric final extension.

Random Forest, XGBoost and CatBoost, each producing a ``metrics.json`` in the
*exact* schema of the existing benchmark (``src.error_metric_benchmark.trainer``)
so the combined ranking can mix reused and new families transparently.

All three are multi-output regressors over the two standardized targets
``[rmse_voltage_mv, rmse_temperature_c]``; metrics are reported in physical
units after inverse transform.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.utils import ensure_dir, save_json, set_seed
from src.error_metric_benchmark.data import BenchmarkDataset, inverse_y
from src.error_metric_benchmark.trainer import _full_result, _save_predictions, _finite

NEW_FAMILIES = ["random_forest", "xgboost", "catboost"]

DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def _paths(run_dir: Path, family: str, seed: int) -> Dict[str, Path]:
    tag = f"seed{seed}"
    return {
        "ckpt": ensure_dir(run_dir / "checkpoints" / family / tag),
        "hist": ensure_dir(run_dir / "histories" / family / tag),
        "metrics": ensure_dir(run_dir / "metrics" / family / tag),
        "pred": ensure_dir(run_dir / "predictions" / family / tag),
    }


def is_complete(run_dir: Path, family: str, seed: int) -> bool:
    return (run_dir / "metrics" / family / f"seed{seed}" / "metrics.json").is_file()


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _build_random_forest(cfg: Dict, seed: int):
    return RandomForestRegressor(
        n_estimators=int(cfg.get("rf_n_estimators", cfg.get("n_estimators", 300))),
        max_depth=cfg.get("rf_max_depth", None),
        n_jobs=int(cfg.get("n_jobs", -1)),
        random_state=seed,
    )


def _build_xgboost(cfg: Dict, seed: int):
    from xgboost import XGBRegressor
    # XGBoost >= 1.6 supports native multi-output regression.
    return XGBRegressor(
        n_estimators=int(cfg.get("xgb_n_estimators", 400)),
        max_depth=int(cfg.get("xgb_max_depth", 6)),
        learning_rate=float(cfg.get("xgb_learning_rate", 0.05)),
        subsample=float(cfg.get("xgb_subsample", 0.8)),
        colsample_bytree=float(cfg.get("xgb_colsample_bytree", 0.8)),
        reg_lambda=float(cfg.get("xgb_reg_lambda", 1.0)),
        tree_method=str(cfg.get("xgb_tree_method", "hist")),
        multi_strategy="multi_output_tree",
        n_jobs=int(cfg.get("n_jobs", -1)),
        random_state=seed,
    )


def _build_catboost(cfg: Dict, seed: int):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(
        iterations=int(cfg.get("cat_iterations", 500)),
        depth=int(cfg.get("cat_depth", 6)),
        learning_rate=float(cfg.get("cat_learning_rate", 0.05)),
        loss_function="MultiRMSE",
        random_seed=seed,
        thread_count=int(cfg.get("n_jobs", -1)) if int(cfg.get("n_jobs", -1)) > 0 else -1,
        verbose=False,
        allow_writing_files=False,
    )


def _node_or_tree_count(family: str, model) -> int:
    try:
        if family == "random_forest":
            return int(sum(est.tree_.node_count for est in model.estimators_))
        if family == "xgboost":
            booster = model.get_booster()
            return int(len(booster.get_dump()))
        if family == "catboost":
            return int(model.tree_count_)
    except Exception:  # noqa: BLE001
        return 0
    return 0


# --------------------------------------------------------------------------- #
# Train one family/seed
# --------------------------------------------------------------------------- #
def train_tree(family: str, ds: BenchmarkDataset, run_dir: Path, cfg: Dict,
               seed: int) -> Dict:
    if family not in NEW_FAMILIES:
        raise ValueError(f"Unknown extension family '{family}'. Choices: {NEW_FAMILIES}")
    paths = _paths(run_dir, family, seed)
    set_seed(seed)

    if family == "random_forest":
        model = _build_random_forest(cfg, seed)
    elif family == "xgboost":
        model = _build_xgboost(cfg, seed)
    else:
        model = _build_catboost(cfg, seed)

    model.fit(ds.X_train, ds.Y_train)

    # Persist the fitted model (joblib works for all three sklearn-style APIs).
    joblib.dump(model, paths["ckpt"] / "model.joblib")

    preds = {s: np.asarray(model.predict(getattr(ds, f"X_{s}")), dtype=np.float64)
             for s in ("train", "val", "test")}
    for s in preds:
        if preds[s].ndim == 1:
            preds[s] = preds[s].reshape(-1, len(ds.target_names))

    t0 = time.perf_counter()
    model.predict(ds.X_test)
    infer_s = time.perf_counter() - t0

    if not _finite(preds["test"]):
        raise FloatingPointError(f"{family} seed{seed}: non-finite predictions")

    # Feature importance when available.
    try:
        imp = getattr(model, "feature_importances_", None)
        if imp is not None:
            pd.DataFrame({"feature": ds.feature_names, "importance": imp}) \
                .sort_values("importance", ascending=False) \
                .to_csv(paths["metrics"] / "feature_importance.csv", index=False)
    except Exception:  # noqa: BLE001
        pass

    y_test_phys = inverse_y(ds, ds.Y_test)
    p_test_phys = inverse_y(ds, preds["test"])
    _, n_rows = _save_predictions(paths["pred"], ds, y_test_phys, p_test_phys)

    res = _full_result(family, ds, preds, {
        "param_count": _node_or_tree_count(family, model),
        "inference_time_s": float(infer_s),
        "inference_ms_per_sample": 1e3 * infer_s / max(1, len(ds.X_test)),
        "n_test_rows": n_rows, "seed": seed, "device": "cpu",
        "family_kind": "classical_tree",
    })
    save_json(res, paths["metrics"] / "metrics.json")
    return res
