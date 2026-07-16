"""Dataset loading, alignment, splitting and PyTorch ``Dataset`` classes.

The simulation pipeline does not guarantee that every LHS parameter row
produces a successful run, so the only reliable way to line up inputs and
outputs is via the ``sample_ids`` stored inside each ``outputs.npz``.  Every
loader below aligns ``parameter_sets.csv`` to those ids and validates shapes
before any tensor is built.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

PathLike = Union[str, Path]

PARAM_CSV_NAME = "parameter_sets.csv"
CASES_DIRNAME = "cases"
NPZ_NAME = "outputs.npz"
SAMPLE_ID_COL = "sample_id"

# Required arrays inside each outputs.npz.
_REQUIRED_NPZ_KEYS = ("sample_ids", "time_s", "voltage_v", "temperature_c")


# --------------------------------------------------------------------------- #
# Discovery & raw loading
# --------------------------------------------------------------------------- #
def discover_cases(data_root: PathLike) -> List[str]:
    """Return the sorted ids of every case directory that contains an npz.

    A "case" is a sub-directory of ``<data_root>/cases`` holding ``outputs.npz``.
    """
    cases_dir = Path(data_root) / CASES_DIRNAME
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    cases = [
        p.name
        for p in sorted(cases_dir.iterdir())
        if p.is_dir() and (p / NPZ_NAME).is_file()
    ]
    if not cases:
        raise FileNotFoundError(
            f"No case folders containing '{NPZ_NAME}' were found under {cases_dir}"
        )
    return cases


def load_parameter_table(data_root: PathLike) -> pd.DataFrame:
    """Load ``parameter_sets.csv`` indexed by ``sample_id`` (the X matrix)."""
    csv_path = Path(data_root) / PARAM_CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Parameter file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if SAMPLE_ID_COL not in df.columns:
        raise KeyError(
            f"'{SAMPLE_ID_COL}' column missing from {csv_path}; "
            f"found columns: {list(df.columns)[:5]}..."
        )
    return df.set_index(SAMPLE_ID_COL)


def load_case_npz(data_root: PathLike, case_id: str) -> Dict[str, np.ndarray]:
    """Load and validate the raw arrays of one case's ``outputs.npz``."""
    npz_path = Path(data_root) / CASES_DIRNAME / case_id / NPZ_NAME
    if not npz_path.is_file():
        raise FileNotFoundError(f"outputs.npz not found for case '{case_id}': {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        missing = [k for k in _REQUIRED_NPZ_KEYS if k not in npz.files]
        if missing:
            raise KeyError(
                f"{npz_path} is missing required arrays {missing}; "
                f"found {list(npz.files)}"
            )
        data = {
            "sample_ids": np.asarray(npz["sample_ids"]).astype(str),
            "time_s": np.asarray(npz["time_s"], dtype=np.float64),
            "voltage_v": np.asarray(npz["voltage_v"], dtype=np.float64),
            "temperature_c": np.asarray(npz["temperature_c"], dtype=np.float64),
        }
    _validate_case_arrays(case_id, data)
    return data


def _validate_case_arrays(case_id: str, data: Dict[str, np.ndarray]) -> None:
    """Assert the shape contract described in the project spec."""
    sample_ids = data["sample_ids"]
    time_s = data["time_s"]
    voltage = data["voltage_v"]
    temperature = data["temperature_c"]

    if voltage.ndim != 2 or temperature.ndim != 2:
        raise ValueError(
            f"[{case_id}] voltage_v and temperature_c must be 2D, got "
            f"{voltage.shape} and {temperature.shape}"
        )
    if voltage.shape != temperature.shape:
        raise ValueError(
            f"[{case_id}] voltage_v {voltage.shape} and temperature_c "
            f"{temperature.shape} must have the same shape"
        )
    if sample_ids.shape[0] != voltage.shape[0]:
        raise ValueError(
            f"[{case_id}] sample_ids length ({sample_ids.shape[0]}) must match "
            f"the number of voltage rows ({voltage.shape[0]})"
        )
    if time_s.shape[0] != voltage.shape[1]:
        raise ValueError(
            f"[{case_id}] time_s length ({time_s.shape[0]}) must match the "
            f"number of voltage columns / time steps ({voltage.shape[1]})"
        )


# --------------------------------------------------------------------------- #
# Aligned case bundle
# --------------------------------------------------------------------------- #
@dataclass
class CaseData:
    """Aligned inputs/outputs for a single case (only successful samples)."""

    case_id: str
    sample_ids: np.ndarray          # [N_ok]
    X: np.ndarray                   # [N_ok, n_parameters]
    V: np.ndarray                   # [N_ok, t_last]
    T: np.ndarray                   # [N_ok, t_last]
    time_s: np.ndarray              # [t_last]
    param_names: List[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_parameters(self) -> int:
        return self.X.shape[1]

    @property
    def t_last(self) -> int:
        return self.time_s.shape[0]


def load_aligned_case_data(data_root: PathLike, case_id: str) -> CaseData:
    """Align ``parameter_sets.csv`` to a case's successful ``sample_ids``.

    This is the canonical entry point used by every dataset builder.  It
    guarantees ``X[i]`` corresponds to ``V[i]`` / ``T[i]`` for the same sample.
    """
    params = load_parameter_table(data_root)
    case = load_case_npz(data_root, case_id)
    sample_ids = case["sample_ids"]

    # Fail loudly if a simulated sample_id has no matching parameter row.
    missing = [sid for sid in sample_ids if sid not in params.index]
    if missing:
        preview = ", ".join(missing[:10])
        raise KeyError(
            f"[{case_id}] {len(missing)} sample_id(s) from outputs.npz are absent "
            f"from {PARAM_CSV_NAME}: {preview}"
            + (" ..." if len(missing) > 10 else "")
        )

    # .loc with the ordered ids reorders X to exactly match the npz row order.
    X_df = params.loc[sample_ids]
    return CaseData(
        case_id=case_id,
        sample_ids=sample_ids,
        X=X_df.to_numpy(dtype=np.float64),
        V=case["voltage_v"],
        T=case["temperature_c"],
        time_s=case["time_s"],
        param_names=list(X_df.columns),
    )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def split_indices(
    n_samples: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split *row* indices into train/val/test.

    The split is over samples (parameter sets), never over time points, so an
    entire response curve always stays within a single split.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = int(round(train_ratio * n_samples))
    n_val = int(round(val_ratio * n_samples))
    train_idx = np.sort(perm[:n_train])
    val_idx = np.sort(perm[n_train : n_train + n_val])
    test_idx = np.sort(perm[n_train + n_val :])
    return train_idx, val_idx, test_idx


def normalize_time(time_s: np.ndarray) -> np.ndarray:
    """Map a time vector to [0, 1]; constant-time edge case -> all zeros."""
    t = np.asarray(time_s, dtype=np.float64)
    span = float(t.max() - t.min())
    if span <= 0.0:
        return np.zeros_like(t)
    return (t - t.min()) / span


# --------------------------------------------------------------------------- #
# Dataset classes
# --------------------------------------------------------------------------- #
class BatteryMLPDataset(Dataset):
    """Tabular dataset: x = parameters, y = concatenated [V | T] curves."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class BatterySequenceDataset(Dataset):
    """Sequence dataset: x = [params..., time_norm] per step, y = [V, T] per step."""

    def __init__(self, X_seq: np.ndarray, Y_seq: np.ndarray):
        self.X = torch.as_tensor(X_seq, dtype=torch.float32)
        self.Y = torch.as_tensor(Y_seq, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


# --------------------------------------------------------------------------- #
# Bundles returned to the training code
# --------------------------------------------------------------------------- #
@dataclass
class DatasetBundle:
    """Everything the training / evaluation code needs for one case+model."""

    train: Dataset
    val: Dataset
    test: Dataset
    x_scaler: StandardScaler
    v_scaler: StandardScaler
    t_scaler: StandardScaler
    case: CaseData
    splits: Dict[str, np.ndarray]
    is_sequence: bool


def _fit_scalers(
    case: CaseData, train_idx: np.ndarray
) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Fit X / V / T scalers on the TRAIN split only (no leakage)."""
    x_scaler = StandardScaler().fit(case.X[train_idx])
    v_scaler = StandardScaler().fit(case.V[train_idx])  # per-timestep stats
    t_scaler = StandardScaler().fit(case.T[train_idx])
    return x_scaler, v_scaler, t_scaler


def create_mlp_datasets(
    data_root: PathLike,
    case_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    """Build train/val/test MLP datasets with train-only fitted scalers.

    Output target Y is ``concat([scaled_V, scaled_T], axis=1)`` of width
    ``2 * t_last`` so the network predicts both full curves at once.
    """
    case = load_aligned_case_data(data_root, case_id)
    train_idx, val_idx, test_idx = split_indices(
        case.n_samples, train_ratio, val_ratio, test_ratio, seed
    )
    x_scaler, v_scaler, t_scaler = _fit_scalers(case, train_idx)

    X_scaled = x_scaler.transform(case.X)
    V_scaled = v_scaler.transform(case.V)
    T_scaled = t_scaler.transform(case.T)
    Y = np.concatenate([V_scaled, T_scaled], axis=1)  # [N, 2*t_last]

    def subset(idx: np.ndarray) -> BatteryMLPDataset:
        return BatteryMLPDataset(X_scaled[idx], Y[idx])

    return DatasetBundle(
        train=subset(train_idx),
        val=subset(val_idx),
        test=subset(test_idx),
        x_scaler=x_scaler,
        v_scaler=v_scaler,
        t_scaler=t_scaler,
        case=case,
        splits={"train": train_idx, "val": val_idx, "test": test_idx},
        is_sequence=False,
    )


def create_sequence_datasets(
    data_root: PathLike,
    case_id: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    """Build train/val/test sequence datasets for RNN/LSTM/BiLSTM models.

    Input  : [N, t_last, n_parameters + 1]  (scaled params broadcast over time,
             plus a normalized-time channel in [0, 1]).
    Target : [N, t_last, 2]                 (scaled voltage, scaled temperature).
    """
    case = load_aligned_case_data(data_root, case_id)
    train_idx, val_idx, test_idx = split_indices(
        case.n_samples, train_ratio, val_ratio, test_ratio, seed
    )
    x_scaler, v_scaler, t_scaler = _fit_scalers(case, train_idx)

    X_scaled = x_scaler.transform(case.X)              # [N, P]
    V_scaled = v_scaler.transform(case.V)              # [N, t_last]
    T_scaled = t_scaler.transform(case.T)              # [N, t_last]

    n, p = X_scaled.shape
    t_last = case.t_last
    time_norm = normalize_time(case.time_s)            # [t_last]

    # Broadcast parameters across every time step and append the time channel.
    X_seq = np.empty((n, t_last, p + 1), dtype=np.float64)
    X_seq[:, :, :p] = X_scaled[:, None, :]             # params repeated per step
    X_seq[:, :, p] = time_norm[None, :]                # normalized time channel

    Y_seq = np.stack([V_scaled, T_scaled], axis=-1)    # [N, t_last, 2]

    def subset(idx: np.ndarray) -> BatterySequenceDataset:
        return BatterySequenceDataset(X_seq[idx], Y_seq[idx])

    return DatasetBundle(
        train=subset(train_idx),
        val=subset(val_idx),
        test=subset(test_idx),
        x_scaler=x_scaler,
        v_scaler=v_scaler,
        t_scaler=t_scaler,
        case=case,
        splits={"train": train_idx, "val": val_idx, "test": test_idx},
        is_sequence=True,
    )


def build_datasets(model_name: str, data_root: PathLike, case_id: str, **kwargs) -> DatasetBundle:
    """Dispatch to the MLP or sequence dataset builder based on ``model_name``.

    Point-wise models (``mlp``, ``bayesian_mlp``) use the tabular MLP datasets;
    every sequence model (``rnn``/``lstm``/``bilstm`` and the additive
    ``cnn``/``cnn_bilstm``) uses the per-step sequence datasets.
    """
    if model_name in {"mlp", "bayesian_mlp"}:
        return create_mlp_datasets(data_root, case_id, **kwargs)
    if model_name in {"rnn", "lstm", "bilstm", "cnn", "cnn_bilstm"}:
        return create_sequence_datasets(data_root, case_id, **kwargs)
    raise ValueError(f"Unknown model_name '{model_name}'")
