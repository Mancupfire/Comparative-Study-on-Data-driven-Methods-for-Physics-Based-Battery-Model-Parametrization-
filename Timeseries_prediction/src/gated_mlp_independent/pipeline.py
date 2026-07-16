"""Data loading, leakage-safe grouped splitting, distance bins, metrics.

Everything operates on the ``training_rmse_errors.csv`` long table (one row per
``sample_id`` x condition).  The 12 model targets are the cross product of the
six discharge conditions and the two error metrics ``rmse_v_mV`` / ``rmse_t_C``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 12 physics parameters used as model inputs (order is fixed and saved).
PARAM_COLUMNS: List[str] = [
    "Positive electrode reference diffusivity [m2.s-1]",
    "Negative electrode reference diffusivity [m2.s-1]",
    "Positive electrode reference reaction rate coefficient [m2.5.mol-0.5.s-1]",
    "Negative electrode reference reaction rate coefficient [m2.5.mol-0.5.s-1]",
    "Core surface thermal resistance [K.W-1]",
    "Environment thermal resistance [K.W-1]",
    "Casing heat capacity [J.K-1]",
    "Contact resistance [Ohm]",
    "Positive electrode reference diffusivity activation energy [J.mol-1]",
    "Negative electrode reference diffusivity activation energy [J.mol-1]",
    "Positive electrode reference reaction rate activation energy [J.mol-1]",
    "Negative electrode reference reaction rate activation energy [J.mol-1]",
]

# Four diffusivity / reaction-rate columns are log10-scaled before standardising.
LOG10_COLUMNS: List[str] = PARAM_COLUMNS[:4]

GROUP_KEY = "sample_id"
METRICS = ["rmse_v_mV", "rmse_t_C"]

# case_id_code -> canonical condition name (index == code; from dataset_summary).
CASE_CODE_TO_NAME: Dict[int, str] = {
    0: "cc_dchg_0p5_25degC",
    1: "cc_dchg_1C_25degC",
    2: "cc_dchg_2C_25degC",
    3: "cc_dchg_0p5_10degC",
    4: "cc_dchg_1C_10degC",
    5: "cc_dchg_2C_10degC",
}


@dataclass(frozen=True)
class TargetSpec:
    """One of the 12 independent models."""

    name: str          # e.g. cc_dchg_0p5_25degC__rmse_v_mV
    condition: str     # e.g. cc_dchg_0p5_25degC
    case_code: int     # 0..5
    metric: str        # rmse_v_mV | rmse_t_C


def build_target_specs() -> List[TargetSpec]:
    specs: List[TargetSpec] = []
    for code in sorted(CASE_CODE_TO_NAME):
        cond = CASE_CODE_TO_NAME[code]
        for metric in METRICS:
            specs.append(TargetSpec(f"{cond}__{metric}", cond, code, metric))
    return specs


def load_table(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "training_rmse_errors.csv")
    if "status" in df.columns:
        df = df[df["status"] == "ok"].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Grouped split (by parameter tuple / sample_id)
# --------------------------------------------------------------------------- #
def grouped_split(
    groups: np.ndarray,
    seed: int = 42,
    frac_train: float = 0.70,
    frac_val: float = 0.15,
) -> Dict[str, np.ndarray]:
    """Deterministically assign whole groups to train/val/test.

    Returns a dict of the group ids per split.  The same group never appears in
    more than one split.
    """
    unique = np.array(sorted(np.unique(groups)))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(unique))
    unique = unique[perm]
    n = len(unique)
    n_train = int(round(frac_train * n))
    n_val = int(round(frac_val * n))
    return {
        "train": np.array(sorted(unique[:n_train])),
        "val": np.array(sorted(unique[n_train : n_train + n_val])),
        "test": np.array(sorted(unique[n_train + n_val :])),
    }


def assert_no_group_overlap(split_groups: Dict[str, np.ndarray]) -> None:
    tr, va, te = (set(split_groups[k].tolist()) for k in ("train", "val", "test"))
    assert tr.isdisjoint(va), "train/val group overlap"
    assert tr.isdisjoint(te), "train/test group overlap"
    assert va.isdisjoint(te), "val/test group overlap"


# --------------------------------------------------------------------------- #
# Distance-to-training-set bins (in standardized input space)
# --------------------------------------------------------------------------- #
def nearest_train_distance(
    x_train_scaled: np.ndarray, x_query_scaled: np.ndarray
) -> np.ndarray:
    """Euclidean distance from each query row to its nearest training row."""
    out = np.empty(len(x_query_scaled), dtype=np.float64)
    chunk = 2048
    for i in range(0, len(x_query_scaled), chunk):
        q = x_query_scaled[i : i + chunk]
        # ||q - t||^2 = ||q||^2 + ||t||^2 - 2 q.t
        d2 = (
            (q ** 2).sum(1)[:, None]
            + (x_train_scaled ** 2).sum(1)[None, :]
            - 2.0 * q @ x_train_scaled.T
        )
        out[i : i + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def distance_bin_labels(dist: np.ndarray, edges: Tuple[float, float]) -> np.ndarray:
    """near / medium / far given two cut edges (e.g. tertiles)."""
    lo, hi = edges
    labels = np.where(dist <= lo, "near", np.where(dist <= hi, "medium", "far"))
    return labels


# --------------------------------------------------------------------------- #
# Scalar regression metrics (original target units)
# --------------------------------------------------------------------------- #
def scalar_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_true)
    if n == 0:
        return {k: float("nan") for k in
                ["n", "rmse", "mae", "r2", "smape", "bias", "max_abs_error", "pearson"]}
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = float(np.mean(np.where(denom > 1e-12, 2.0 * np.abs(err) / denom, 0.0)) * 100.0)
    bias = float(np.mean(err))
    max_abs = float(np.max(np.abs(err)))
    if n >= 2 and y_true.std() > 1e-12 and y_pred.std() > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")
    return {
        "n": n, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape,
        "bias": bias, "max_abs_error": max_abs, "pearson": pearson,
    }
