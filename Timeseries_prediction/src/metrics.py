"""Voltage / temperature curve metrics.

All metrics operate on *physical* (inverse-transformed) values with shape
``[N, t_last]`` so the numbers are directly interpretable in volts / degrees C.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    """Global R^2 across all (sample, timestep) entries."""
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _curve_rmse_mean(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean over samples of each curve's own RMSE (per-sample then averaged)."""
    per_sample = np.sqrt(np.mean((true - pred) ** 2, axis=1))  # [N]
    return float(np.mean(per_sample))


def compute_metrics(
    v_true: np.ndarray,
    v_pred: np.ndarray,
    t_true: np.ndarray,
    t_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute the full metric suite for one case/model on physical units.

    Parameters
    ----------
    v_true, v_pred, t_true, t_pred : ``[N, t_last]`` arrays (volts / deg C).
    """
    v_true = np.asarray(v_true, dtype=np.float64)
    v_pred = np.asarray(v_pred, dtype=np.float64)
    t_true = np.asarray(t_true, dtype=np.float64)
    t_pred = np.asarray(t_pred, dtype=np.float64)

    metrics: Dict[str, float] = {
        # Voltage
        "MAE_V": _mae(v_true, v_pred),
        "RMSE_V": _rmse(v_true, v_pred),
        "R2_V": _r2(v_true, v_pred),
        "MaxError_V": _max_error(v_true, v_pred),
        # Temperature
        "MAE_T": _mae(t_true, t_pred),
        "RMSE_T": _rmse(t_true, t_pred),
        "R2_T": _r2(t_true, t_pred),
        "MaxError_T": _max_error(t_true, t_pred),
        # Curve-specific
        "voltage_end_mae": _mae(v_true[:, -1], v_pred[:, -1]),
        "temperature_end_mae": _mae(t_true[:, -1], t_pred[:, -1]),
        # Peak temperature is a key safety quantity -> compare per-sample maxima.
        "temperature_peak_mae": _mae(t_true.max(axis=1), t_pred.max(axis=1)),
        "voltage_curve_rmse_mean": _curve_rmse_mean(v_true, v_pred),
        "temperature_curve_rmse_mean": _curve_rmse_mean(t_true, t_pred),
    }
    return metrics


# Canonical column order used by compare_models.py / model_comparison.csv.
METRIC_COLUMNS = [
    "MAE_V",
    "RMSE_V",
    "R2_V",
    "MaxError_V",
    "MAE_T",
    "RMSE_T",
    "R2_T",
    "MaxError_T",
    "voltage_end_mae",
    "temperature_end_mae",
    "temperature_peak_mae",
    "voltage_curve_rmse_mean",
    "temperature_curve_rmse_mean",
]
