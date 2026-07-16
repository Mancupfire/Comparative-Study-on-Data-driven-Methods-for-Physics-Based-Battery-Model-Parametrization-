"""

Implements the README "Error-Metric Surrogate" spec for Data_Batch_2:

    error_metrics.csv  --(sequence_id)-->  sequence_manifest.csv
                       --(sample_id)----->  parameter_sets.csv

Predict two scalar targets per sequence::

    rmse_voltage_mv
    rmse_temperature_c

from the 12 physical parameters plus operating-condition features. Categorical
operating conditions (``operation_code``) are one-hot encoded explicitly. The
train/val/test split is over **unique ``sample_id``** (70/15/15, seed 42) so the
same parameter set can never appear in more than one split. Input and target
scalers are fit on the **training split only**.

This module performs NO training — it only builds, validates and splits arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PathLike = Union[str, Path]

SAMPLE_ID_COL = "sample_id"
SEQUENCE_ID_COL = "sequence_id"
TARGET_COLS = ["rmse_voltage_mv", "rmse_temperature_c"]
# Continuous operating-condition features taken from the manifest.
COND_NUMERIC = ["c_rate", "ambient_temp_C", "initial_temp_C"]
# Categorical operating-condition column (explicitly one-hot encoded).
COND_CATEGORICAL = "operation_code"


@dataclass
class JoinReport:
    n_error_metrics: int
    n_manifest: int
    n_parameters: int
    n_joined: int
    error_metrics_sequence_id_unique: bool
    manifest_sequence_id_unique: bool
    parameters_sample_id_unique: bool
    metrics_manifest_one_to_one: bool
    joined_params_complete: bool
    unmatched_metric_sequence_ids: List[str]
    unmatched_param_sample_ids: List[str]
    duplicate_joined_sequence_ids: List[str]

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class ErrorMetricDataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_val: np.ndarray
    Y_test: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    feature_names: List[str]
    target_names: List[str]
    continuous_feature_idx: List[int]
    categorical_feature_idx: List[int]
    split_sample_ids: Dict[str, List[str]]
    split_row_sample_ids: Dict[str, np.ndarray]   # sample_id per row, per split
    join_report: JoinReport
    raw_feature_frame: pd.DataFrame = field(repr=False, default=None)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required Task B file not found: {path}")
    return pd.read_csv(path)


def build_joined_frame(data_dir: PathLike) -> Tuple[pd.DataFrame, List[str], JoinReport]:
    """Join the three sources and validate cardinality. Returns (frame, param_cols, report)."""
    d = Path(data_dir)
    metrics = _read_csv(d / "error_metrics.csv")
    manifest = _read_csv(d / "sequence_manifest.csv")
    params = _read_csv(d / "parameter_sets.csv")

    for col in TARGET_COLS:
        if col not in metrics.columns:
            raise KeyError(f"error_metrics.csv missing target column '{col}'")
    if SAMPLE_ID_COL not in params.columns:
        raise KeyError("parameter_sets.csv missing 'sample_id'")
    param_cols = [c for c in params.columns if c != SAMPLE_ID_COL]

    em_unique = bool(metrics[SEQUENCE_ID_COL].is_unique)
    mf_unique = bool(manifest[SEQUENCE_ID_COL].is_unique)
    pr_unique = bool(params[SAMPLE_ID_COL].is_unique)

    # metrics <-> manifest : one-to-one on sequence_id
    em_ids = set(metrics[SEQUENCE_ID_COL])
    mf_ids = set(manifest[SEQUENCE_ID_COL])
    one_to_one = em_unique and mf_unique and (em_ids == mf_ids)
    unmatched_metric = sorted(em_ids - mf_ids)

    j1 = metrics.merge(
        manifest[[SEQUENCE_ID_COL, SAMPLE_ID_COL, COND_CATEGORICAL] + COND_NUMERIC],
        on=SEQUENCE_ID_COL, how="left", validate="one_to_one",
    )
    # joined <-> parameters : many sequences to one parameter row
    joined = j1.merge(params, on=SAMPLE_ID_COL, how="left", validate="many_to_one")

    params_complete = bool(joined[param_cols].notna().all().all())
    unmatched_param = sorted(
        set(j1[SAMPLE_ID_COL]) - set(params[SAMPLE_ID_COL])
    )
    dup_join = sorted(joined.loc[joined[SEQUENCE_ID_COL].duplicated(), SEQUENCE_ID_COL].unique())

    report = JoinReport(
        n_error_metrics=len(metrics),
        n_manifest=len(manifest),
        n_parameters=len(params),
        n_joined=len(joined),
        error_metrics_sequence_id_unique=em_unique,
        manifest_sequence_id_unique=mf_unique,
        parameters_sample_id_unique=pr_unique,
        metrics_manifest_one_to_one=one_to_one,
        joined_params_complete=params_complete,
        unmatched_metric_sequence_ids=unmatched_metric[:50],
        unmatched_param_sample_ids=unmatched_param[:50],
        duplicate_joined_sequence_ids=dup_join[:50],
    )
    return joined, param_cols, report


def _build_feature_matrix(
    joined: pd.DataFrame, param_cols: List[str]
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """Assemble [params | numeric cond | one-hot(operation_code)]."""
    continuous_cols = param_cols + COND_NUMERIC
    X_cont = joined[continuous_cols].to_numpy(dtype=np.float64)

    # Explicit one-hot of operation_code (deterministic, sorted code order).
    codes = np.sort(joined[COND_CATEGORICAL].unique())
    onehot = np.zeros((len(joined), len(codes)), dtype=np.float64)
    code_to_col = {c: i for i, c in enumerate(codes)}
    col_idx = joined[COND_CATEGORICAL].map(code_to_col).to_numpy()
    onehot[np.arange(len(joined)), col_idx] = 1.0
    onehot_names = [f"operation_code_{int(c)}" for c in codes]

    X = np.concatenate([X_cont, onehot], axis=1)
    feature_names = continuous_cols + onehot_names
    continuous_idx = list(range(len(continuous_cols)))
    categorical_idx = list(range(len(continuous_cols), len(feature_names)))
    return X, feature_names, continuous_idx, categorical_idx


def split_sample_ids(
    sample_ids: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Deterministically split UNIQUE sample_ids into train/val/test."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1.0")
    uniq = np.array(sorted(set(sample_ids)))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_train = int(round(train_ratio * len(uniq)))
    n_val = int(round(val_ratio * len(uniq)))
    return {
        "train": np.sort(uniq[perm[:n_train]]),
        "val": np.sort(uniq[perm[n_train:n_train + n_val]]),
        "test": np.sort(uniq[perm[n_train + n_val:]]),
    }


def build_dataset(
    data_dir: PathLike,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    scale_targets: bool = True,
) -> ErrorMetricDataset:
    """Build the full Task B dataset: join, encode, split-by-sample_id, scale-on-train."""
    joined, param_cols, report = build_joined_frame(data_dir)
    if not report.metrics_manifest_one_to_one:
        raise ValueError("metrics<->manifest join is not one-to-one; aborting (see join report).")
    if not report.joined_params_complete:
        raise ValueError("some joined rows have missing parameters; aborting (see join report).")

    X, feature_names, cont_idx, cat_idx = _build_feature_matrix(joined, param_cols)
    Y = joined[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sample_ids = joined[SAMPLE_ID_COL].to_numpy().astype(str)

    splits = split_sample_ids(row_sample_ids, train_ratio, val_ratio, test_ratio, seed)
    masks = {k: np.isin(row_sample_ids, v) for k, v in splits.items()}

    # ----- scalers fit on TRAIN ONLY -----
    x_scaler = StandardScaler()
    x_scaler.fit(X[masks["train"]][:, cont_idx])  # scale continuous cols only
    y_scaler = StandardScaler()
    if scale_targets:
        y_scaler.fit(Y[masks["train"]])           # per-column => targets scaled separately

    def _apply_x(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        out[:, cont_idx] = x_scaler.transform(arr[:, cont_idx])
        return out

    def _apply_y(arr: np.ndarray) -> np.ndarray:
        return y_scaler.transform(arr) if scale_targets else arr

    ds = ErrorMetricDataset(
        X_train=_apply_x(X[masks["train"]]),
        X_val=_apply_x(X[masks["val"]]),
        X_test=_apply_x(X[masks["test"]]),
        Y_train=_apply_y(Y[masks["train"]]),
        Y_val=_apply_y(Y[masks["val"]]),
        Y_test=_apply_y(Y[masks["test"]]),
        x_scaler=x_scaler,
        y_scaler=y_scaler if scale_targets else None,
        feature_names=feature_names,
        target_names=TARGET_COLS,
        continuous_feature_idx=cont_idx,
        categorical_feature_idx=cat_idx,
        split_sample_ids={k: v.astype(str).tolist() for k, v in splits.items()},
        split_row_sample_ids={k: row_sample_ids[masks[k]] for k in masks},
        join_report=report,
        raw_feature_frame=joined,
    )
    return ds


def leakage_check(ds: ErrorMetricDataset) -> Dict[str, object]:
    """Verify no sample_id crosses splits and counts are consistent."""
    s = {k: set(v) for k, v in ds.split_sample_ids.items()}
    pairwise_disjoint = (
        s["train"].isdisjoint(s["val"])
        and s["train"].isdisjoint(s["test"])
        and s["val"].isdisjoint(s["test"])
    )
    row_disjoint = True
    seen = {}
    for split, ids in ds.split_row_sample_ids.items():
        for sid in set(ids.tolist()):
            if sid in seen and seen[sid] != split:
                row_disjoint = False
            seen[sid] = split
    return {
        "sample_id_splits_pairwise_disjoint": bool(pairwise_disjoint),
        "row_level_sample_id_disjoint": bool(row_disjoint),
        "n_train_samples": len(s["train"]),
        "n_val_samples": len(s["val"]),
        "n_test_samples": len(s["test"]),
        "n_train_rows": int(ds.X_train.shape[0]),
        "n_val_rows": int(ds.X_val.shape[0]),
        "n_test_rows": int(ds.X_test.shape[0]),
        "n_total_rows": int(ds.X_train.shape[0] + ds.X_val.shape[0] + ds.X_test.shape[0]),
        "leakage_free": bool(pairwise_disjoint and row_disjoint),
    }
