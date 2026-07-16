"""Valid-time masking for filtered time-series trajectories.

Every aligned sequence shares the common length ``t_last = 160`` but each sample
was only simulated up to its own ``simulation_end_s``; beyond that point the
voltage / temperature curves are *held* (constant boundary extrapolation).  A
point is **valid** iff::

    time_s[j] <= simulation_end_s[i]

The functions here build that boolean mask and provide masked loss / metric
primitives.  The defining property — proven in ``tests/test_valid_time_mask.py``
— is that the value of any *invalid* (held / extrapolated) tail entry has **no
effect** on the loss, MAE, RMSE, R2, parity scatter or time-resolved metrics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

# Numerical tolerance so a point landing exactly on simulation_end_s counts as
# valid despite floating-point round-off in the shared time grid.
TIME_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Mask construction
# --------------------------------------------------------------------------- #
def compute_valid_mask(time_s: np.ndarray, sim_end_s: np.ndarray) -> np.ndarray:
    """Return a ``[N, T]`` boolean mask of simulated (non-extrapolated) points.

    Parameters
    ----------
    time_s   : ``[T]`` shared per-case time grid (seconds).
    sim_end_s: ``[N]`` per-sample simulation end time (seconds).
    """
    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    sim_end_s = np.asarray(sim_end_s, dtype=np.float64).reshape(-1)
    mask = time_s[None, :] <= (sim_end_s[:, None] + TIME_EPS)
    # The first point (t=0) is always part of the simulated interval.
    mask[:, 0] = True
    return mask


def last_valid_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last valid timestep per sample (``[N]``)."""
    m = np.asarray(mask, dtype=bool)
    # argmax on the reversed array finds the first True from the end.
    rev = m[:, ::-1]
    last_from_end = np.argmax(rev, axis=1)
    return m.shape[1] - 1 - last_from_end


# --------------------------------------------------------------------------- #
# Masked metric primitives (numpy, physical units)
# --------------------------------------------------------------------------- #
def _flat_valid(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    m = np.asarray(mask, dtype=bool)
    return np.asarray(a, dtype=np.float64)[m], np.asarray(b, dtype=np.float64)[m]


def masked_mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    t, p = _flat_valid(true, pred, mask)
    return float(np.mean(np.abs(t - p))) if t.size else 0.0


def masked_rmse(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    t, p = _flat_valid(true, pred, mask)
    return float(np.sqrt(np.mean((t - p) ** 2))) if t.size else 0.0


def masked_max_error(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    t, p = _flat_valid(true, pred, mask)
    return float(np.max(np.abs(t - p))) if t.size else 0.0


def masked_r2(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Global R^2 over valid (sample, timestep) entries only."""
    t, p = _flat_valid(true, pred, mask)
    if t.size == 0:
        return 0.0
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def masked_curve_rmse_mean(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Mean over samples of each curve's own RMSE on its valid points."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    n_valid = m.sum(axis=1)
    sq = np.where(m, (true - pred) ** 2, 0.0).sum(axis=1)
    per_sample = np.sqrt(sq / np.maximum(n_valid, 1))
    return float(np.mean(per_sample))


def masked_end_mae(true: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """MAE at each sample's last *valid* timestep (true curve end)."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    idx = last_valid_index(mask)
    rows = np.arange(true.shape[0])
    return float(np.mean(np.abs(true[rows, idx] - pred[rows, idx])))


def masked_peak_temperature_mae(t_true: np.ndarray, t_pred: np.ndarray,
                                mask: np.ndarray) -> float:
    """MAE of per-sample peak temperature over valid points only."""
    t_true = np.asarray(t_true, dtype=np.float64)
    t_pred = np.asarray(t_pred, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    very_neg = -np.inf
    true_peak = np.where(m, t_true, very_neg).max(axis=1)
    pred_peak = np.where(m, t_pred, very_neg).max(axis=1)
    return float(np.mean(np.abs(true_peak - pred_peak)))


def compute_masked_metrics(v_true, v_pred, t_true, t_pred, mask) -> Dict[str, float]:
    """Full masked metric suite (physical units), mirroring src.metrics."""
    return {
        "MAE_V": masked_mae(v_true, v_pred, mask),
        "RMSE_V": masked_rmse(v_true, v_pred, mask),
        "R2_V": masked_r2(v_true, v_pred, mask),
        "MaxError_V": masked_max_error(v_true, v_pred, mask),
        "MAE_T": masked_mae(t_true, t_pred, mask),
        "RMSE_T": masked_rmse(t_true, t_pred, mask),
        "R2_T": masked_r2(t_true, t_pred, mask),
        "MaxError_T": masked_max_error(t_true, t_pred, mask),
        "voltage_end_mae": masked_end_mae(v_true, v_pred, mask),
        "temperature_end_mae": masked_end_mae(t_true, t_pred, mask),
        "temperature_peak_mae": masked_peak_temperature_mae(t_true, t_pred, mask),
        "voltage_curve_rmse_mean": masked_curve_rmse_mean(v_true, v_pred, mask),
        "temperature_curve_rmse_mean": masked_curve_rmse_mean(t_true, t_pred, mask),
        "n_valid_points": int(np.asarray(mask, dtype=bool).sum()),
        "n_total_points": int(np.asarray(mask).size),
    }


# --------------------------------------------------------------------------- #
# Masked torch loss (training)
# --------------------------------------------------------------------------- #
def masked_mse_torch(pred, true, mask):
    """Mean squared error over valid entries only (torch tensors).

    ``pred`` / ``true`` are ``[B, T]``; ``mask`` is a ``[B, T]`` float/bool
    tensor of the same shape.  Returns a scalar tensor.  Invalid entries are
    zero-weighted, so their values cannot influence the gradient.
    """
    import torch

    if mask.dtype != pred.dtype:
        mask = mask.to(pred.dtype)
    diff2 = (pred - true) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom
