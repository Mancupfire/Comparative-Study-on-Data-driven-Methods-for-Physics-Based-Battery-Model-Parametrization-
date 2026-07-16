"""Unified error-metric benchmark dataset (two split protocols).

Reuses the audited join + feature-matrix logic from
``src.error_metric_data`` so the feature set is *identical* to the existing
two-model Batch 4 pipeline:

    17 inputs = 12 physical parameters
              + [c_rate, ambient_temp_C, initial_temp_C]
              + one-hot(operation_code)   (2 codes -> 2 columns)

Targets: ``[rmse_voltage_mv, rmse_temperature_c]`` (standardized on train only).

Two protocols
-------------
``grouped_holdout``     : split by unique ``sample_id`` (all 12 conditions of a
                          parameter set stay in one split).  Scientifically
                          recommended; matches the current pipeline.
``legacy_reproduction`` : row-wise random split over individual sequences
                          (sample_ids leak across splits).  Provided only to
                          reproduce / explain the optimistic legacy numbers.

Both fit feature + target scalers on the TRAIN split only.  A per-row split
manifest (sample_id, experiment_id, sequence_id, split) is produced so every
model and seed evaluates the exact same rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.error_metric_data import (
    COND_CATEGORICAL, COND_NUMERIC, SAMPLE_ID_COL, SEQUENCE_ID_COL, TARGET_COLS,
    _build_feature_matrix, build_joined_frame,
)

PathLike = Union[str, Path]
PROTOCOLS = ("grouped_holdout", "legacy_reproduction")


@dataclass
class BenchmarkDataset:
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
    manifest: pd.DataFrame                 # sample_id, experiment_id, sequence_id, split
    protocol: str
    seed: int

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def split_frame(self, split: str) -> pd.DataFrame:
        return self.manifest[self.manifest["split"] == split].reset_index(drop=True)


def _grouped_split(sample_ids: np.ndarray, ratios, seed: int) -> Dict[str, set]:
    uniq = np.array(sorted(set(sample_ids.tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(ratios[0] * len(uniq)))
    n_va = int(round(ratios[1] * len(uniq)))
    return {
        "train": set(uniq[perm[:n_tr]].tolist()),
        "val": set(uniq[perm[n_tr:n_tr + n_va]].tolist()),
        "test": set(uniq[perm[n_tr + n_va:]].tolist()),
    }


def _row_split(n_rows: int, ratios, seed: int) -> np.ndarray:
    """Row-wise split labels (legacy, leaky)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_tr = int(round(ratios[0] * n_rows))
    n_va = int(round(ratios[1] * n_rows))
    labels = np.empty(n_rows, dtype=object)
    labels[perm[:n_tr]] = "train"
    labels[perm[n_tr:n_tr + n_va]] = "val"
    labels[perm[n_tr + n_va:]] = "test"
    return labels


def build_benchmark_dataset(
    data_dir: PathLike,
    protocol: str = "grouped_holdout",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> BenchmarkDataset:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}, got '{protocol}'")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("split ratios must sum to 1.0")

    joined, param_cols, report = build_joined_frame(data_dir)
    if not report.metrics_manifest_one_to_one:
        raise ValueError("metrics<->manifest join not one-to-one (see join report).")
    if not report.joined_params_complete:
        raise ValueError("joined rows have missing parameters (see join report).")

    X, feature_names, cont_idx, cat_idx = _build_feature_matrix(joined, param_cols)
    Y = joined[TARGET_COLS].to_numpy(dtype=np.float64)
    row_sample_ids = joined[SAMPLE_ID_COL].to_numpy().astype(str)
    row_seq_ids = joined[SEQUENCE_ID_COL].to_numpy().astype(str)
    # experiment_id is the operating-case string after the "__" separator.
    row_exp_ids = np.array([s.split("__", 1)[1] if "__" in s else "" for s in row_seq_ids])

    ratios = (train_ratio, val_ratio, test_ratio)
    if protocol == "grouped_holdout":
        groups = _grouped_split(row_sample_ids, ratios, seed)
        split_label = np.array(
            ["train" if s in groups["train"] else "val" if s in groups["val"] else "test"
             for s in row_sample_ids], dtype=object)
    else:  # legacy_reproduction
        split_label = _row_split(len(Y), ratios, seed)

    masks = {k: (split_label == k) for k in ("train", "val", "test")}

    # ----- scalers fit on TRAIN ONLY -----
    x_scaler = StandardScaler().fit(X[masks["train"]][:, cont_idx])
    y_scaler = StandardScaler().fit(Y[masks["train"]])

    def _ax(arr):
        out = arr.copy()
        out[:, cont_idx] = x_scaler.transform(arr[:, cont_idx])
        return out

    manifest = pd.DataFrame({
        "sample_id": row_sample_ids,
        "experiment_id": row_exp_ids,
        "sequence_id": row_seq_ids,
        "split": split_label.astype(str),
    })

    return BenchmarkDataset(
        X_train=_ax(X[masks["train"]]),
        X_val=_ax(X[masks["val"]]),
        X_test=_ax(X[masks["test"]]),
        Y_train=y_scaler.transform(Y[masks["train"]]),
        Y_val=y_scaler.transform(Y[masks["val"]]),
        Y_test=y_scaler.transform(Y[masks["test"]]),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_names=feature_names,
        target_names=list(TARGET_COLS),
        continuous_feature_idx=cont_idx,
        categorical_feature_idx=cat_idx,
        manifest=manifest,
        protocol=protocol,
        seed=seed,
    )


def inverse_y(ds: BenchmarkDataset, y_scaled: np.ndarray) -> np.ndarray:
    return ds.y_scaler.inverse_transform(np.asarray(y_scaled))


def split_audit(ds: BenchmarkDataset) -> Dict[str, object]:
    """Leakage + size audit for the built dataset."""
    m = ds.manifest
    by = {s: set(m.loc[m.split == s, "sample_id"]) for s in ("train", "val", "test")}
    overlap_tv = sorted(by["train"] & by["val"])
    overlap_tt = sorted(by["train"] & by["test"])
    overlap_vt = sorted(by["val"] & by["test"])
    disjoint = not (overlap_tv or overlap_tt or overlap_vt)
    return {
        "protocol": ds.protocol,
        "seed": ds.seed,
        "total_rows": int(len(m)),
        "unique_sample_ids": int(m["sample_id"].nunique()),
        "unique_experiment_ids": int(m["experiment_id"].nunique()),
        "n_train_rows": int((m.split == "train").sum()),
        "n_val_rows": int((m.split == "val").sum()),
        "n_test_rows": int((m.split == "test").sum()),
        "n_train_sample_ids": len(by["train"]),
        "n_val_sample_ids": len(by["val"]),
        "n_test_sample_ids": len(by["test"]),
        "overlap_train_val": len(overlap_tv),
        "overlap_train_test": len(overlap_tt),
        "overlap_val_test": len(overlap_vt),
        "overlap_examples": (overlap_tv + overlap_tt + overlap_vt)[:10],
        "sample_id_splits_disjoint": bool(disjoint),
        "scaler_fit_scope": "train_only",
        "target_stats_use_val_or_test": False,
    }
