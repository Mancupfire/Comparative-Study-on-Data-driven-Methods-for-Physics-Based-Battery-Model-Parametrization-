"""Error-metric evaluation metrics (all reported in physical units).

Per target (voltage RMSE in mV, temperature RMSE in degC):
    MAE, RMSE, R2, MaxError, rel_mean, rel_median, rel_p95

Overall (scale-safe — never mixes mV and degC directly):
    norm_overall_RMSE : RMSE in standardized two-target space
    norm_overall_MAE  : MAE  in standardized two-target space
    mean_R2           : mean of the two per-target R2

The two "normalized overall" numbers are computed on z-scored targets using the
per-target standard deviation of the *true test values*, so each target
contributes on a comparable, unitless scale.  This is documented in the
experiment_summary / methods_note.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

REL_EPS = 1e-8


def per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       names: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(names):
        t = np.asarray(y_true[:, j], dtype=np.float64)
        p = np.asarray(y_pred[:, j], dtype=np.float64)
        err = t - p
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        max_err = float(np.max(np.abs(err)))
        rel = np.abs(err) / (np.abs(t) + REL_EPS)
        out[name] = {
            "MAE": mae, "RMSE": rmse, "R2": r2, "MaxError": max_err,
            "rel_mean": float(np.mean(rel)),
            "rel_median": float(np.median(rel)),
            "rel_p95": float(np.percentile(rel, 95)),
        }
    return out


def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    names: List[str], per_target: Dict[str, Dict[str, float]]
                    ) -> Dict[str, float]:
    """Scale-safe overall metrics using true-test per-target std for z-scoring."""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    std = t.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    tz = (t - t.mean(axis=0)) / std
    pz = (p - t.mean(axis=0)) / std
    errz = tz - pz
    return {
        "norm_overall_RMSE": float(np.sqrt(np.mean(errz ** 2))),
        "norm_overall_MAE": float(np.mean(np.abs(errz))),
        "mean_R2": float(np.mean([per_target[n]["R2"] for n in names])),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             names: List[str]) -> Dict[str, object]:
    pt = per_target_metrics(y_true, y_pred, names)
    ov = overall_metrics(y_true, y_pred, names, pt)
    return {"per_target": pt, "overall": ov}
