"""Filtered time-series dataset with grouped splits and valid-time masks.

For one case (experiment) and one seed this builds train/val/test arrays that:

* keep only the duration-ratio-filtered samples
  (``time_series_kept_manifest.csv``);
* assign every kept sample to a split via the **grouped** sample-id holdout
  (700/150/150), reconstructed identically to the completed grouped
  error-metric benchmark (``_grouped_split`` below mirrors
  ``src.error_metric_benchmark.data._grouped_split``);
* carry a per-sample ``[T]`` valid mask (``time_s <= simulation_end_s``);
* standardize X per-feature and V/T globally on TRAIN-valid entries only.

The raw / downsampled ``outputs.npz`` arrays are read-only inputs; nothing here
mutates the source data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from src.data import load_aligned_case_data
from .masking import compute_valid_mask

PathLike = Union[str, Path]
REPO = Path(__file__).resolve().parents[2]

FILTERED_DIR = "data/Data_Batch_4_TSFiltered_0p8"
DEFAULT_RATIOS = (0.7, 0.15, 0.15)


# --------------------------------------------------------------------------- #
# Grouped split (identical contract to the EM benchmark)
# --------------------------------------------------------------------------- #
def grouped_split(sample_ids: np.ndarray, ratios=DEFAULT_RATIOS, seed: int = 42
                  ) -> Dict[str, set]:
    """Split UNIQUE sample_ids into train/val/test sets (grouped holdout).

    Mirrors ``src.error_metric_benchmark.data._grouped_split`` exactly so the
    filtered time-series split is interchangeable with the completed grouped
    error-metric benchmark splits.
    """
    uniq = np.array(sorted(set(np.asarray(sample_ids).astype(str).tolist())))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(ratios[0] * len(uniq)))
    n_va = int(round(ratios[1] * len(uniq)))
    return {
        "train": set(uniq[perm[:n_tr]].tolist()),
        "val": set(uniq[perm[n_tr:n_tr + n_va]].tolist()),
        "test": set(uniq[perm[n_tr + n_va:]].tolist()),
    }


def split_label_for(sample_id: str, groups: Dict[str, set]) -> str:
    if sample_id in groups["train"]:
        return "train"
    if sample_id in groups["val"]:
        return "val"
    return "test"


# --------------------------------------------------------------------------- #
# Global (scalar) standardizer for V / T over valid points
# --------------------------------------------------------------------------- #
@dataclass
class GlobalScaler:
    mean_: float
    scale_: float

    @classmethod
    def fit_valid(cls, values: np.ndarray, mask: np.ndarray) -> "GlobalScaler":
        v = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        mean = float(np.mean(v)) if v.size else 0.0
        std = float(np.std(v)) if v.size else 1.0
        return cls(mean_=mean, scale_=std if std > 1e-12 else 1.0)

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (np.asarray(arr, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64) * self.scale_ + self.mean_


# --------------------------------------------------------------------------- #
# Manifests
# --------------------------------------------------------------------------- #
def load_filter_meta(filtered_dir: PathLike = FILTERED_DIR) -> Dict:
    return json.loads((REPO / filtered_dir / "filter_meta.json").read_text())


def load_kept_manifest(filtered_dir: PathLike = FILTERED_DIR) -> pd.DataFrame:
    return pd.read_csv(REPO / filtered_dir / "time_series_kept_manifest.csv")


def load_source_manifest(filtered_dir: PathLike = FILTERED_DIR) -> pd.DataFrame:
    return pd.read_csv(REPO / filtered_dir / "time_series_source_manifest.csv")


def all_sample_ids(filtered_dir: PathLike = FILTERED_DIR) -> np.ndarray:
    """The full (unfiltered) set of unique sample ids — split is over all 1000."""
    src = load_source_manifest(filtered_dir)
    return np.array(sorted(src["sample_id"].astype(str).unique()))


# --------------------------------------------------------------------------- #
# Case bundle
# --------------------------------------------------------------------------- #
@dataclass
class FilteredCaseBundle:
    case_id: str
    seed: int
    time_s: np.ndarray                       # [T]
    t_last: int
    n_parameters: int
    param_names: List[str]
    # per split:
    X: Dict[str, np.ndarray]                 # scaled params [n, P]
    V_phys: Dict[str, np.ndarray]            # physical voltage [n, T]
    T_phys: Dict[str, np.ndarray]            # physical temperature [n, T]
    V_scaled: Dict[str, np.ndarray]
    T_scaled: Dict[str, np.ndarray]
    mask: Dict[str, np.ndarray]              # [n, T] bool
    sample_ids: Dict[str, np.ndarray]
    x_scaler: object
    v_scaler: GlobalScaler
    t_scaler: GlobalScaler

    def n(self, split: str) -> int:
        return self.X[split].shape[0]


def build_filtered_case(
    case_id: str,
    seed: int = 42,
    filtered_dir: PathLike = FILTERED_DIR,
    ratios=DEFAULT_RATIOS,
) -> FilteredCaseBundle:
    from sklearn.preprocessing import StandardScaler

    meta = load_filter_meta(filtered_dir)
    source_ts_dir = REPO / meta["source_time_series_dir"]

    # Aligned (params, V, T, time) for this case at native filtered grid.
    case = load_aligned_case_data(source_ts_dir, case_id)
    case_sample_ids = case.sample_ids.astype(str)

    # Kept sample ids + simulation_end_s for this experiment.
    src = load_source_manifest(filtered_dir)
    case_src = src[src["experiment_id"] == case_id].copy()
    case_src["sample_id"] = case_src["sample_id"].astype(str)
    kept_ids = set(case_src.loc[case_src["kept"], "sample_id"])
    sim_end_by_id = dict(zip(case_src["sample_id"], case_src["simulation_end_s"]))

    # Grouped split over ALL sample ids (full 1000), seed-dependent.
    groups = grouped_split(all_sample_ids(filtered_dir), ratios, seed)

    # Row selection per split: kept AND in this split's sample-id set.
    row_split = np.array([split_label_for(sid, groups) for sid in case_sample_ids])
    keep_row = np.array([sid in kept_ids for sid in case_sample_ids])

    sim_end = np.array([sim_end_by_id[sid] for sid in case_sample_ids], dtype=np.float64)
    full_mask = compute_valid_mask(case.time_s, sim_end)   # [N, T]

    def rows_for(split: str) -> np.ndarray:
        return np.where(keep_row & (row_split == split))[0]

    idx = {s: rows_for(s) for s in ("train", "val", "test")}

    # X scaler on train params; V/T global scalers on train-valid entries.
    x_scaler = StandardScaler().fit(case.X[idx["train"]])
    v_scaler = GlobalScaler.fit_valid(case.V[idx["train"]], full_mask[idx["train"]])
    t_scaler = GlobalScaler.fit_valid(case.T[idx["train"]], full_mask[idx["train"]])

    X, Vp, Tp, Vs, Ts, msk, sids = {}, {}, {}, {}, {}, {}, {}
    for s in ("train", "val", "test"):
        ix = idx[s]
        X[s] = x_scaler.transform(case.X[ix])
        Vp[s] = case.V[ix]
        Tp[s] = case.T[ix]
        Vs[s] = v_scaler.transform(case.V[ix])
        Ts[s] = t_scaler.transform(case.T[ix])
        msk[s] = full_mask[ix]
        sids[s] = case_sample_ids[ix]

    return FilteredCaseBundle(
        case_id=case_id, seed=seed, time_s=case.time_s, t_last=case.t_last,
        n_parameters=case.n_parameters, param_names=case.param_names,
        X=X, V_phys=Vp, T_phys=Tp, V_scaled=Vs, T_scaled=Ts,
        mask=msk, sample_ids=sids,
        x_scaler=x_scaler, v_scaler=v_scaler, t_scaler=t_scaler,
    )


def discover_filtered_cases(filtered_dir: PathLike = FILTERED_DIR) -> List[str]:
    src = load_source_manifest(filtered_dir)
    return sorted(src["experiment_id"].astype(str).unique())
