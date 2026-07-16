"""Shared, condition-aware dataset construction across **all** cases.

The per-case pipeline (``src/data.py``) trains one model per case and therefore
never has to reconcile different sequence lengths.  The *shared* pipeline trains
a single model on every case at once, so it must handle the fact that each case
has a different ``t_last``.

Instead of concatenating full curves (which would require a common length) we
model the problem **point-wise / conditionally**::

    f_theta([parameter_vector, c_rate, ambient_temp_C, time_norm]) -> [V(t), T(t)]

* ``parameter_vector`` — the LHS parameters for that sample (from
  ``parameter_sets.csv``, aligned through ``sample_ids``).
* ``c_rate`` / ``ambient_temp_C`` — operating-condition features parsed from the
  ``case_id``.
* ``time_norm`` — ``time_s / max(time_s)`` in ``[0, 1]``; this is what lets one
  model span cases of different length.

Two views of the same data are provided:

* **point-wise** (used by ``shared_mlp``): every ``(sample, timestep)`` pair is an
  independent training example.
* **sequence + mask** (used by ``shared_rnn`` / ``shared_lstm`` /
  ``shared_bilstm``): every curve is one variable-length sequence, padded in a
  custom ``collate_fn`` with a boolean ``mask`` so padded steps never contribute.

To avoid leakage the train/val/test split is performed at the **curve level**
(one ``(sample_id, case_id)`` pair = one curve); a curve always lands entirely in
a single split.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from . import data as data_mod
from .utils import ensure_dir

PathLike = Union[str, Path]

# Number of condition / time features appended after the parameter vector:
# [c_rate, ambient_temp_C, time_norm].
N_CONDITION_FEATURES = 3

SEQUENCE_MODELS = {
    "shared_rnn", "shared_lstm", "shared_bilstm", "shared_cnn", "shared_cnn_bilstm"
}
POINT_MODELS = {"shared_mlp", "shared_bayesian_mlp"}
ALL_SHARED_MODELS = POINT_MODELS | SEQUENCE_MODELS


# --------------------------------------------------------------------------- #
# Output-path layout (mirrors src/predict.py but under a ``shared`` namespace)
# --------------------------------------------------------------------------- #
def shared_checkpoint_dir(outputs_dir: PathLike, model_name: str) -> Path:
    return Path(outputs_dir) / "checkpoints" / "shared" / model_name


def shared_scaler_dir(outputs_dir: PathLike, model_name: str) -> Path:
    return Path(outputs_dir) / "scalers" / "shared" / model_name


def shared_metrics_dir(outputs_dir: PathLike, model_name: str) -> Path:
    return Path(outputs_dir) / "metrics" / "shared" / model_name


def shared_figures_dir(outputs_dir: PathLike, model_name: str, case_id: str) -> Path:
    return Path(outputs_dir) / "figures" / "shared" / model_name / case_id


# --------------------------------------------------------------------------- #
# Case-id parsing
# --------------------------------------------------------------------------- #
def parse_case_id(case_id: str) -> Dict[str, Optional[float]]:
    """Extract ``c_rate`` and ``ambient_temp_C`` from a ``case_id`` string.

    Examples
    --------
    >>> parse_case_id("cc_dchg_0p5_10degC")
    {'c_rate': 0.5, 'ambient_temp_C': 10.0}
    >>> parse_case_id("cc_dchg_1C_25degC")
    {'c_rate': 1.0, 'ambient_temp_C': 25.0}
    >>> parse_case_id("cc_dchg_2C_10degC")
    {'c_rate': 2.0, 'ambient_temp_C': 10.0}

    Future-style tokens such as ``1p5C`` or ``3C`` are also supported (``p`` is a
    decimal point and a trailing ``C`` is the C-rate unit marker).
    """
    ambient_temp_C: Optional[float] = None
    temp_match = re.search(r"(-?\d+(?:[p.]\d+)?)\s*degC", case_id, flags=re.IGNORECASE)
    if temp_match:
        ambient_temp_C = float(temp_match.group(1).replace("p", "."))

    c_rate: Optional[float] = None
    for token in case_id.split("_"):
        low = token.lower()
        if "degc" in low or low in {"cc", "dchg", "chg", "cv"}:
            continue
        if low.endswith("c"):
            # e.g. "1C", "2C", "1p5C" -> strip the trailing C-rate unit marker.
            candidate = token[:-1].replace("p", ".")
            if re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                c_rate = float(candidate)
                break
        elif "p" in low:
            # e.g. "0p5" uses 'p' as a decimal point and carries no unit marker.
            candidate = token.replace("p", ".")
            if re.fullmatch(r"-?\d+(?:\.\d+)?", candidate):
                c_rate = float(candidate)
                break

    return {"c_rate": c_rate, "ambient_temp_C": ambient_temp_C}


# --------------------------------------------------------------------------- #
# Curve-level representation shared by both views
# --------------------------------------------------------------------------- #
@dataclass
class Curve:
    """One simulated response curve (a single ``sample_id`` within a case)."""

    case_id: str
    sample_id: str
    c_rate: float
    ambient_temp_C: float
    params: np.ndarray      # [P]
    time_s: np.ndarray      # [T]
    time_norm: np.ndarray   # [T] in [0, 1]
    V: np.ndarray           # [T]
    T: np.ndarray           # [T]

    @property
    def t_last(self) -> int:
        return self.time_s.shape[0]


def _normalize_time(time_s: np.ndarray) -> np.ndarray:
    """Map a time vector to ``[0, 1]`` via division by its max (per the spec).

    Falls back to a span-based mapping when ``max == 0``; an all-constant time
    vector degenerates to zeros.
    """
    t = np.asarray(time_s, dtype=np.float64)
    t_max = float(t.max())
    if t_max > 0.0:
        return t / t_max
    span = float(t.max() - t.min())
    if span <= 0.0:
        return np.zeros_like(t)
    return (t - t.min()) / span


def load_all_curves(
    data_root: PathLike, cases: Optional[List[str]] = None
) -> Tuple[List[Curve], List[str]]:
    """Load every case and return a deterministic, ordered list of ``Curve``.

    Reuses ``src.data.load_aligned_case_data`` so the ``sample_ids``-based
    alignment between ``parameter_sets.csv`` and each ``outputs.npz`` is shared
    with the per-case pipeline.  Curves are ordered by ``(case_id, npz row)`` so
    every downstream split is reproducible.
    """
    case_ids = list(cases) if cases else data_mod.discover_cases(data_root)
    curves: List[Curve] = []
    param_names: List[str] = []

    for case_id in case_ids:
        cond = parse_case_id(case_id)
        if cond["c_rate"] is None or cond["ambient_temp_C"] is None:
            raise ValueError(
                f"Could not parse c_rate / ambient_temp_C from case_id '{case_id}'. "
                f"Got {cond}. Expected something like 'cc_dchg_1C_25degC'."
            )
        case = data_mod.load_aligned_case_data(data_root, case_id)
        if not param_names:
            param_names = list(case.param_names)
        elif param_names != list(case.param_names):
            raise ValueError(
                f"[{case_id}] parameter columns differ from the first case; "
                f"all cases must share the same parameter schema."
            )

        time_norm = _normalize_time(case.time_s)
        for row in range(case.n_samples):
            curves.append(
                Curve(
                    case_id=case_id,
                    sample_id=str(case.sample_ids[row]),
                    c_rate=float(cond["c_rate"]),
                    ambient_temp_C=float(cond["ambient_temp_C"]),
                    params=case.X[row].astype(np.float64),
                    time_s=case.time_s,
                    time_norm=time_norm,
                    V=case.V[row].astype(np.float64),
                    T=case.T[row].astype(np.float64),
                )
            )

    if not curves:
        raise FileNotFoundError(f"No curves loaded from {data_root}.")
    return curves, param_names


def build_curve_features(curve: Curve) -> np.ndarray:
    """Build the per-timestep raw feature matrix ``[T, P + 3]`` for one curve.

    Columns are ``[*parameter_vector, c_rate, ambient_temp_C, time_norm]``; the
    parameters and conditions are constant across time and only ``time_norm``
    varies.  This is the single source of truth used by both the point-wise
    builder and evaluation.
    """
    t = curve.t_last
    p = curve.params.shape[0]
    feats = np.empty((t, p + N_CONDITION_FEATURES), dtype=np.float64)
    feats[:, :p] = curve.params[None, :]
    feats[:, p] = curve.c_rate
    feats[:, p + 1] = curve.ambient_temp_C
    feats[:, p + 2] = curve.time_norm
    return feats


def build_curve_targets(curve: Curve) -> np.ndarray:
    """Build the per-timestep target matrix ``[T, 2]`` = ``[V, T]`` for a curve."""
    return np.stack([curve.V, curve.T], axis=1)


# --------------------------------------------------------------------------- #
# Point-wise loading
# --------------------------------------------------------------------------- #
@dataclass
class PointwiseData:
    """Flattened point-wise samples plus per-point metadata for splitting/eval."""

    X: np.ndarray              # [M, P + 3]   raw (unscaled) features
    Y: np.ndarray              # [M, 2]       raw (unscaled) [V, T]
    curve_idx: np.ndarray      # [M]          index into ``curves`` for each point
    case_id: np.ndarray        # [M]          str
    sample_id: np.ndarray      # [M]          str
    time_s: np.ndarray         # [M]          physical time of the point
    curves: List[Curve]
    param_names: List[str] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return self.X.shape[0]

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    @property
    def input_dim(self) -> int:
        return self.X.shape[1]


def _select_timesteps(
    t_last: int, max_points_per_curve: Optional[int], rng: np.random.Generator
) -> np.ndarray:
    """Choose the timestep indices to keep for one curve.

    Always keeps the first and last step; if ``max_points_per_curve`` is set and
    smaller than ``t_last`` the interior steps are randomly subsampled.
    """
    if max_points_per_curve is None or max_points_per_curve >= t_last:
        return np.arange(t_last)
    if max_points_per_curve <= 2:
        return np.array([0, t_last - 1], dtype=int)[:max_points_per_curve]
    interior = np.arange(1, t_last - 1)
    n_interior = max_points_per_curve - 2
    chosen = rng.choice(interior, size=n_interior, replace=False)
    idx = np.concatenate([[0], np.sort(chosen), [t_last - 1]])
    return idx


def load_all_cases_pointwise(
    data_root: PathLike,
    max_points_per_curve: Optional[int] = None,
    seed: int = 42,
    cases: Optional[List[str]] = None,
) -> PointwiseData:
    """Load all cases and flatten them into point-wise ``(X, Y)`` samples.

    For each curve a point is created for every selected timestep::

        X = [parameter_vector, c_rate, ambient_temp_C, time_norm]  -> Y = [V, T]

    ``max_points_per_curve``:
        * ``None``  -> keep every timestep.
        * ``int``   -> keep at most that many timesteps per curve, always
          including the first and last step (interior steps are subsampled with
          the given ``seed``).
    """
    curves, param_names = load_all_curves(data_root, cases=cases)
    rng = np.random.default_rng(seed)

    X_blocks: List[np.ndarray] = []
    Y_blocks: List[np.ndarray] = []
    curve_idx_blocks: List[np.ndarray] = []
    case_id_blocks: List[np.ndarray] = []
    sample_id_blocks: List[np.ndarray] = []
    time_blocks: List[np.ndarray] = []

    for ci, curve in enumerate(curves):
        steps = _select_timesteps(curve.t_last, max_points_per_curve, rng)
        feats = build_curve_features(curve)[steps]
        targs = build_curve_targets(curve)[steps]
        n = steps.shape[0]
        X_blocks.append(feats)
        Y_blocks.append(targs)
        curve_idx_blocks.append(np.full(n, ci, dtype=np.int64))
        case_id_blocks.append(np.full(n, curve.case_id, dtype=object))
        sample_id_blocks.append(np.full(n, curve.sample_id, dtype=object))
        time_blocks.append(curve.time_s[steps])

    return PointwiseData(
        X=np.concatenate(X_blocks, axis=0),
        Y=np.concatenate(Y_blocks, axis=0),
        curve_idx=np.concatenate(curve_idx_blocks, axis=0),
        case_id=np.concatenate(case_id_blocks, axis=0),
        sample_id=np.concatenate(sample_id_blocks, axis=0),
        time_s=np.concatenate(time_blocks, axis=0),
        curves=curves,
        param_names=param_names,
    )


# --------------------------------------------------------------------------- #
# Curve-level (leakage-free) splitting
# --------------------------------------------------------------------------- #
def split_by_sample_case(
    n_curves: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Split *curve* indices into train/val/test.

    The split is over whole curves (each a unique ``sample_id + case_id`` pair),
    never over individual time points, so a curve never straddles two splits.
    """
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_curves)
    n_train = int(round(train_ratio * n_curves))
    n_val = int(round(val_ratio * n_curves))
    return {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train : n_train + n_val]),
        "test": np.sort(perm[n_train + n_val :]),
    }


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class SharedPointDataset(Dataset):
    """Point-wise dataset: ``x = [params, c_rate, ambient_temp_C, time_norm]``."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.Y[idx]


class SharedSequenceDataset(Dataset):
    """Sequence dataset: one item is one variable-length curve.

    Each item is ``(x_seq [T, P+3], y_seq [T, 2])``.  Padding to a common length
    happens later in :func:`collate_pad_sequences`, which also returns the mask.
    """

    def __init__(self, x_seqs: List[np.ndarray], y_seqs: List[np.ndarray]):
        self.x_seqs = [torch.as_tensor(x, dtype=torch.float32) for x in x_seqs]
        self.y_seqs = [torch.as_tensor(y, dtype=torch.float32) for y in y_seqs]

    def __len__(self) -> int:
        return len(self.x_seqs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x_seqs[idx], self.y_seqs[idx]


def collate_pad_sequences(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length curves to a common length.

    Returns
    -------
    x_padded : [B, max_T, input_dim]
    y_padded : [B, max_T, 2]
    mask     : [B, max_T] (bool, True for valid timesteps)
    lengths  : [B] (long, original sequence lengths)
    """
    lengths = torch.tensor([x.shape[0] for x, _ in batch], dtype=torch.long)
    max_t = int(lengths.max().item())
    b = len(batch)
    input_dim = batch[0][0].shape[1]

    x_padded = torch.zeros(b, max_t, input_dim, dtype=torch.float32)
    y_padded = torch.zeros(b, max_t, 2, dtype=torch.float32)
    mask = torch.zeros(b, max_t, dtype=torch.bool)
    for i, (x_seq, y_seq) in enumerate(batch):
        t = x_seq.shape[0]
        x_padded[i, :t] = x_seq
        y_padded[i, :t] = y_seq
        mask[i, :t] = True
    return x_padded, y_padded, mask, lengths


# --------------------------------------------------------------------------- #
# Scaler fitting / persistence
# --------------------------------------------------------------------------- #
@dataclass
class SharedDataBundle:
    """Everything the shared training / evaluation code needs."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    x_scaler: StandardScaler
    v_scaler: StandardScaler
    t_scaler: StandardScaler
    splits: Dict[str, np.ndarray]
    curves: List[Curve]
    param_names: List[str]
    input_dim: int
    is_sequence: bool


def _fit_and_save_scalers(
    X_train: np.ndarray,
    V_train: np.ndarray,
    T_train: np.ndarray,
    outputs_dir: PathLike,
    model_name: str,
) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Fit X / V / T scalers on TRAIN data only and persist them to disk."""
    x_scaler = StandardScaler().fit(X_train)
    v_scaler = StandardScaler().fit(V_train.reshape(-1, 1))
    t_scaler = StandardScaler().fit(T_train.reshape(-1, 1))

    sdir = ensure_dir(shared_scaler_dir(outputs_dir, model_name))
    joblib.dump(x_scaler, sdir / "x_scaler.joblib")
    joblib.dump(v_scaler, sdir / "v_scaler.joblib")
    joblib.dump(t_scaler, sdir / "t_scaler.joblib")
    return x_scaler, v_scaler, t_scaler


def load_shared_scalers(
    outputs_dir: PathLike, model_name: str
) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Load the x / v / t scalers saved during shared training."""
    sdir = shared_scaler_dir(outputs_dir, model_name)
    paths = {
        "x": sdir / "x_scaler.joblib",
        "v": sdir / "v_scaler.joblib",
        "t": sdir / "t_scaler.joblib",
    }
    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {key}_scaler for shared/{model_name}: {p}. Train first."
            )
    return joblib.load(paths["x"]), joblib.load(paths["v"]), joblib.load(paths["t"])


def _apply_scalers(
    X: np.ndarray,
    Y: np.ndarray,
    x_scaler: StandardScaler,
    v_scaler: StandardScaler,
    t_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale a point block: features with ``x_scaler``, V/T with their scalers."""
    X_s = x_scaler.transform(X)
    v_s = v_scaler.transform(Y[:, 0:1])
    t_s = t_scaler.transform(Y[:, 1:2])
    return X_s, np.concatenate([v_s, t_s], axis=1)


# --------------------------------------------------------------------------- #
# Point-wise dataloaders (shared_mlp)
# --------------------------------------------------------------------------- #
def create_shared_point_dataloaders(
    data_root: PathLike,
    model_name: str = "shared_mlp",
    *,
    outputs_dir: PathLike = "outputs",
    batch_size: int = 8192,
    max_points_per_curve: Optional[int] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cases: Optional[List[str]] = None,
) -> SharedDataBundle:
    """Build point-wise train/val/test dataloaders with train-only scalers."""
    pdata = load_all_cases_pointwise(
        data_root, max_points_per_curve=max_points_per_curve, seed=seed, cases=cases
    )
    splits = split_by_sample_case(pdata.n_curves, train_ratio, val_ratio, test_ratio, seed)

    # Map curve-level split -> point-level boolean masks (no leakage).
    train_curves = set(splits["train"].tolist())
    val_curves = set(splits["val"].tolist())
    test_curves = set(splits["test"].tolist())
    in_train = np.fromiter((c in train_curves for c in pdata.curve_idx), dtype=bool, count=pdata.n_points)
    in_val = np.fromiter((c in val_curves for c in pdata.curve_idx), dtype=bool, count=pdata.n_points)
    in_test = np.fromiter((c in test_curves for c in pdata.curve_idx), dtype=bool, count=pdata.n_points)

    x_scaler, v_scaler, t_scaler = _fit_and_save_scalers(
        pdata.X[in_train], pdata.Y[in_train, 0], pdata.Y[in_train, 1], outputs_dir, model_name
    )

    def subset(mask: np.ndarray) -> SharedPointDataset:
        X_s, Y_s = _apply_scalers(pdata.X[mask], pdata.Y[mask], x_scaler, v_scaler, t_scaler)
        return SharedPointDataset(X_s, Y_s)

    train_loader = DataLoader(
        subset(in_train), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    )
    val_loader = DataLoader(
        subset(in_val), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        subset(in_test), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return SharedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        x_scaler=x_scaler,
        v_scaler=v_scaler,
        t_scaler=t_scaler,
        splits=splits,
        curves=pdata.curves,
        param_names=pdata.param_names,
        input_dim=pdata.input_dim,
        is_sequence=False,
    )


# --------------------------------------------------------------------------- #
# Sequence dataloaders (shared_rnn / shared_lstm / shared_bilstm)
# --------------------------------------------------------------------------- #
def create_shared_sequence_dataloaders(
    data_root: PathLike,
    model_name: str = "shared_rnn",
    *,
    outputs_dir: PathLike = "outputs",
    batch_size: int = 64,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    cases: Optional[List[str]] = None,
) -> SharedDataBundle:
    """Build sequence (padded + masked) dataloaders with train-only scalers.

    Each curve becomes one variable-length sequence; the scalers are fit on the
    flattened TRAIN timesteps only, then every curve is scaled and turned into a
    ``[T, P+3]`` / ``[T, 2]`` sequence.
    """
    curves, param_names = load_all_curves(data_root, cases=cases)
    n_curves = len(curves)
    splits = split_by_sample_case(n_curves, train_ratio, val_ratio, test_ratio, seed)

    # Fit scalers on flattened TRAIN timesteps only.
    train_X = np.concatenate([build_curve_features(curves[i]) for i in splits["train"]], axis=0)
    train_V = np.concatenate([curves[i].V for i in splits["train"]], axis=0)
    train_T = np.concatenate([curves[i].T for i in splits["train"]], axis=0)
    x_scaler, v_scaler, t_scaler = _fit_and_save_scalers(
        train_X, train_V, train_T, outputs_dir, model_name
    )

    input_dim = train_X.shape[1]

    def build_subset(indices: np.ndarray) -> SharedSequenceDataset:
        x_seqs, y_seqs = [], []
        for i in indices:
            curve = curves[i]
            feats = x_scaler.transform(build_curve_features(curve))
            v_s = v_scaler.transform(curve.V.reshape(-1, 1))
            t_s = t_scaler.transform(curve.T.reshape(-1, 1))
            x_seqs.append(feats)
            y_seqs.append(np.concatenate([v_s, t_s], axis=1))
        return SharedSequenceDataset(x_seqs, y_seqs)

    train_loader = DataLoader(
        build_subset(splits["train"]), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_pad_sequences,
        drop_last=False,
    )
    val_loader = DataLoader(
        build_subset(splits["val"]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_pad_sequences,
    )
    test_loader = DataLoader(
        build_subset(splits["test"]), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_pad_sequences,
    )

    return SharedDataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        x_scaler=x_scaler,
        v_scaler=v_scaler,
        t_scaler=t_scaler,
        splits=splits,
        curves=curves,
        param_names=param_names,
        input_dim=input_dim,
        is_sequence=True,
    )


def create_shared_dataloaders(model_name: str, data_root: PathLike, **kwargs) -> SharedDataBundle:
    """Dispatch to the point-wise or sequence dataloader builder by model name."""
    name = model_name.lower()
    if name in POINT_MODELS:
        return create_shared_point_dataloaders(data_root, name, **kwargs)
    if name in SEQUENCE_MODELS:
        # Point-only kwargs are silently irrelevant for sequence models.
        kwargs.pop("max_points_per_curve", None)
        return create_shared_sequence_dataloaders(data_root, name, **kwargs)
    raise ValueError(
        f"Unknown shared model '{model_name}'. Choices: {sorted(ALL_SHARED_MODELS)}"
    )
