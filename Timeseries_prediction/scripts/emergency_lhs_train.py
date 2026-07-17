#!/usr/bin/env python3
"""
Emergency LHS battery time-series training pipeline.

Purpose
-------
Produce a scientifically defensible smoke/preliminary result quickly from
lhs_1000_seed42 while Claude Code is unavailable.

Key safeguards
--------------
- Group-aware train/validation/test split by sample_id.
- One shared split for every model.
- Feature and target scalers fitted on training data only.
- Curves resampled on a normalized capacity grid.
- Loss and metrics ignore the held-constant tail after simulated_end_capacity_Ah.
- Best checkpoint selected using validation loss only.

Expected dataset files
----------------------
<dataset_dir>/sequence_manifest.csv
<dataset_dir>/parameter_sets_physical.csv
<dataset_dir>/generated_dataset.h5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Additive reproducibility/reporting helpers (no effect on training behaviour).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lhs_retrain_reporting as reporting  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


MODEL_NAMES = [
    "mlp",
    "rnn",
    "lstm",
    "bilstm",
    "cnn",
    "cnn_bilstm",
    "bayesian_mlp",
]

DISPLAY_NAMES = {
    "mlp": "ANN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn": "CNN",
    "cnn_bilstm": "CNN-BiLSTM",
    "bayesian_mlp": "Bayesian MLP",
}


@dataclass
class RunConfig:
    dataset_dir: str
    output_dir: str
    models: List[str]
    sequence_length: int
    max_sample_ids: Optional[int]
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    num_layers: int
    dropout: float
    train_fraction: float
    val_fraction: float
    test_fraction: float
    seed: int
    num_workers: int
    device: str
    inference_repeats: int
    parity_max_points: int
    min_valid_fraction: float
    alignment_mode: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emergency LHS battery sequence training pipeline."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mlp", "lstm", "bilstm"],
        choices=MODEL_NAMES,
    )
    parser.add_argument("--sequence-length", type=int, default=160)
    parser.add_argument(
        "--max-sample-ids",
        type=int,
        default=300,
        help="Limit unique sample IDs for a fast preliminary run; 0 uses all.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--inference-repeats", type=int, default=3)
    parser.add_argument("--parity-max-points", type=int, default=20000)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.0,
        help="Optional quality filter. 0.0 keeps all successful sequences.",
    )
    parser.add_argument(
        "--alignment-mode",
        type=str,
        default="official_clamped",
        choices=["official_clamped", "masked"],
        help=(
            "official_clamped reproduces the verified previous protocol: the "
            "endpoint-held horizontal tail is retained and the entire aligned "
            "sequence enters loss/metrics. masked excludes the held tail after "
            "simulated_end_capacity_Ah (sensitivity mode)."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dataset(dataset_dir: Path) -> Tuple[Path, Path, Path]:
    manifest = dataset_dir / "sequence_manifest.csv"
    params = dataset_dir / "parameter_sets_physical.csv"
    h5_path = dataset_dir / "generated_dataset.h5"
    missing = [p for p in (manifest, params, h5_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files:\n" + "\n".join(str(p) for p in missing)
        )
    return manifest, params, h5_path


def split_sample_ids(
    sample_ids: Sequence[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[str]]:
    total = train_fraction + val_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    unique_ids = np.array(sorted(set(sample_ids)))
    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=(1.0 - train_fraction),
        random_state=seed,
        shuffle=True,
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
    )
    result = {
        "train": sorted(train_ids.tolist()),
        "val": sorted(val_ids.tolist()),
        "test": sorted(test_ids.tolist()),
    }
    assert not (set(result["train"]) & set(result["val"]))
    assert not (set(result["train"]) & set(result["test"]))
    assert not (set(result["val"]) & set(result["test"]))
    return result


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_and_resample(
    dataset_dir: Path,
    sequence_length: int,
    max_sample_ids: Optional[int],
    seed: int,
    min_valid_fraction: float,
    alignment_mode: str,
    excluded_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return metadata, X raw, Y raw, mask, q-grid, feature names.

    Alignment onto the per-case common capacity grid is inherited from the
    dataset generation (``aligned/*`` arrays already carry endpoint-held tails).
    Here every sequence is reparametrized to a shared normalized capacity grid
    of length ``sequence_length`` so a single model can span all cases.

    ``alignment_mode``:
      * ``official_clamped`` — mask is all ones; the endpoint-held horizontal
        tail is retained and the entire aligned sequence enters loss/metrics.
      * ``masked`` — the held tail after ``simulated_end_capacity_Ah`` is
        excluded by a valid mask (sensitivity mode).

    Structurally invalid sequences are excluded (never silently kept) and their
    reasons are written to ``excluded_csv``. A sequence is invalid when it has
    fewer than two raw simulation points, a non-positive raw capacity span, or
    any NaN/Inf in its raw or aligned curve values.
    """
    if alignment_mode not in ("official_clamped", "masked"):
        raise ValueError(f"Unknown alignment_mode '{alignment_mode}'")
    manifest_path, params_path, h5_path = ensure_dataset(dataset_dir)
    manifest = pd.read_csv(manifest_path)
    params = pd.read_csv(params_path)

    manifest = manifest.loc[manifest["simulation_status"].eq("ok")].copy()
    if manifest["sequence_id"].duplicated().any():
        raise ValueError("Duplicate sequence_id detected")

    if max_sample_ids and max_sample_ids > 0:
        all_ids = np.array(sorted(manifest["sample_id"].unique()))
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(all_ids, size=min(max_sample_ids, len(all_ids)), replace=False))
        manifest = manifest.loc[manifest["sample_id"].isin(selected)].copy()

    param_cols = [c for c in params.columns if c != "sample_id"]
    if len(param_cols) != 10:
        raise ValueError(f"Expected 10 physical parameter columns, found {len(param_cols)}")

    manifest = manifest.merge(params, on="sample_id", how="left", validate="many_to_one")
    if manifest[param_cols].isna().any().any():
        raise ValueError("Missing physical parameters after manifest merge")

    operation_encoded = manifest["operation"].map({"discharge": 0.0, "charge": 1.0})
    if operation_encoded.isna().any():
        raise ValueError(f"Unknown operation values: {manifest['operation'].unique().tolist()}")

    feature_names = param_cols + ["operation_encoded", "c_rate", "initial_temperature_C"]
    X_all = np.column_stack(
        [
            manifest[param_cols].to_numpy(dtype=np.float64),
            operation_encoded.to_numpy(dtype=np.float64),
            manifest["c_rate"].to_numpy(dtype=np.float64),
            manifest["initial_temperature_C"].to_numpy(dtype=np.float64),
        ]
    )

    manifest = manifest.reset_index(drop=True)
    has_raw = {"raw_offset", "raw_length"}.issubset(manifest.columns)

    q_grid = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)
    n_rows = len(manifest)
    Y = np.empty((n_rows, sequence_length, 2), dtype=np.float32)
    mask = np.zeros((n_rows, sequence_length), dtype=np.float32)
    keep = np.ones(n_rows, dtype=bool)
    excluded_rows: List[Dict[str, object]] = []

    def exclude(row, reason: str) -> None:
        keep[row.Index] = False
        excluded_rows.append(
            {
                "sequence_id": row.sequence_id,
                "sample_id": row.sample_id,
                "experiment_id": row.experiment_id,
                "reason": reason,
            }
        )

    with h5py.File(h5_path, "r") as h5:
        cap_flat = h5["aligned/experimental_capacity_Ah"]
        v_flat = h5["aligned/simulated_voltage_V"]
        t_flat = h5["aligned/simulated_temperature_C"]
        raw_cap_flat = h5["raw/capacity_Ah"] if has_raw else None

        for row in manifest.itertuples(index=True):
            out_i = row.Index
            start = int(row.aligned_offset)
            end = start + int(row.aligned_length)
            cap = np.asarray(cap_flat[start:end], dtype=np.float64)
            voltage = np.asarray(v_flat[start:end], dtype=np.float64)
            temperature = np.asarray(t_flat[start:end], dtype=np.float64)

            if not (len(cap) == len(voltage) == len(temperature) == int(row.aligned_length)):
                raise ValueError(f"Length mismatch for {row.sequence_id}")

            # ---- Structural validity: exclude (never crash) -------------- #
            # (a) fewer than two raw simulation points.
            raw_length = int(row.raw_length) if has_raw else int(row.aligned_length)
            if raw_length < 2:
                exclude(row, "fewer_than_two_raw_points")
                continue
            # (b) non-positive raw capacity span.
            if raw_cap_flat is not None:
                rstart = int(row.raw_offset)
                raw_cap = np.asarray(raw_cap_flat[rstart:rstart + raw_length], dtype=np.float64)
            else:
                raw_cap = cap
            if not np.all(np.isfinite(raw_cap)) or float(raw_cap.max() - raw_cap.min()) <= 0.0:
                exclude(row, "non_positive_raw_capacity_span")
                continue
            # (c) NaN/Inf in raw or aligned curve values.
            if not (
                np.all(np.isfinite(cap))
                and np.all(np.isfinite(voltage))
                and np.all(np.isfinite(temperature))
                and np.all(np.isfinite(raw_cap))
            ):
                exclude(row, "nan_or_inf")
                continue

            experimental_end = float(row.experimental_end_capacity_Ah)
            simulated_end = float(row.simulated_end_capacity_Ah)
            if not np.isfinite(experimental_end) or experimental_end <= 0:
                experimental_end = float(cap[-1])
            if not np.isfinite(experimental_end) or experimental_end <= 0:
                exclude(row, "non_positive_raw_capacity_span")
                continue

            q = np.clip(cap / experimental_end, 0.0, 1.0)
            # Remove duplicate coordinates to keep np.interp well-defined.
            q_unique, unique_idx = np.unique(q, return_index=True)
            v_unique = voltage[unique_idx]
            t_unique = temperature[unique_idx]
            if len(q_unique) < 2:
                exclude(row, "non_positive_raw_capacity_span")
                continue

            # np.interp holds the endpoint value for q outside [q_unique[0],
            # q_unique[-1]], reproducing the horizontal tail of the protocol.
            Y[out_i, :, 0] = np.interp(q_grid, q_unique, v_unique).astype(np.float32)
            Y[out_i, :, 1] = np.interp(q_grid, q_unique, t_unique).astype(np.float32)

            if alignment_mode == "official_clamped":
                # Entire aligned sequence (incl. endpoint-held tail) is valid.
                mask[out_i] = 1.0
            else:  # masked sensitivity mode
                valid_fraction = float(np.clip(simulated_end / experimental_end, 0.0, 1.0))
                mask[out_i] = (q_grid <= valid_fraction + 1e-12).astype(np.float32)
                # Always retain at least the first point so the denominator
                # cannot be zero.
                mask[out_i, 0] = 1.0

    excluded_df = pd.DataFrame(
        excluded_rows, columns=["sequence_id", "sample_id", "experiment_id", "reason"]
    )
    if excluded_csv is not None:
        excluded_csv.parent.mkdir(parents=True, exist_ok=True)
        excluded_df.to_csv(excluded_csv, index=False)
    if excluded_rows:
        print(
            f"[preprocess] excluded {len(excluded_rows)} structurally invalid "
            f"sequence(s); reasons: "
            f"{excluded_df['reason'].value_counts().to_dict()}",
            flush=True,
        )

    manifest = manifest.loc[keep].reset_index(drop=True)
    X = X_all[keep]
    Y = Y[keep]
    mask = mask[keep]

    manifest["valid_fraction"] = mask.mean(axis=1)
    if min_valid_fraction > 0:
        vkeep = manifest["valid_fraction"].to_numpy() >= min_valid_fraction
        manifest = manifest.loc[vkeep].reset_index(drop=True)
        X = X[vkeep]
        Y = Y[vkeep]
        mask = mask[vkeep]

    return manifest, X.astype(np.float32), Y, mask, q_grid.astype(np.float32), feature_names


class SequenceDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        mask: np.ndarray,
        q_grid: np.ndarray,
        indices: np.ndarray,
        metadata: pd.DataFrame,
    ) -> None:
        self.X = torch.from_numpy(X[indices]).float()
        self.Y = torch.from_numpy(Y[indices]).float()
        self.mask = torch.from_numpy(mask[indices]).float()
        self.q = torch.from_numpy(q_grid).float()
        self.indices = np.asarray(indices)
        self.metadata = metadata.iloc[indices].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.X[idx], self.q, self.Y[idx], self.mask[idx], idx


class PointwiseMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float, bayesian: bool = False):
        super().__init__()
        p = dropout if bayesian else 0.0
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_static: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(-1)
        x = x_static.unsqueeze(1).expand(-1, q.shape[1], -1)
        return self.net(torch.cat([x, q], dim=-1))


class RecurrentModel(nn.Module):
    def __init__(
        self,
        kind: str,
        n_features: int,
        hidden: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        input_size = n_features + 1
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if kind == "rnn":
            self.core = nn.RNN(
                input_size,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                nonlinearity="tanh",
                bidirectional=bidirectional,
            )
        elif kind == "lstm":
            self.core = nn.LSTM(
                input_size,
                hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=recurrent_dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(kind)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 2)

    def forward(self, x_static: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(-1)
        x = x_static.unsqueeze(1).expand(-1, q.shape[1], -1)
        out, _ = self.core(torch.cat([x, q], dim=-1))
        return self.head(out)


class CNNModel(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float):
        super().__init__()
        in_channels = n_features + 1
        mid = max(32, hidden // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(mid, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, mid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(mid, 2, kernel_size=1),
        )

    def forward(self, x_static: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(-1)
        x = x_static.unsqueeze(1).expand(-1, q.shape[1], -1)
        z = torch.cat([x, q], dim=-1).transpose(1, 2)
        return self.net(z).transpose(1, 2)


class CNNBiLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int, num_layers: int, dropout: float):
        super().__init__()
        in_channels = n_features + 1
        conv_channels = max(32, hidden // 2)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            conv_channels,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden * 2, 2)

    def forward(self, x_static: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q = q.unsqueeze(-1)
        x = x_static.unsqueeze(1).expand(-1, q.shape[1], -1)
        z = torch.cat([x, q], dim=-1).transpose(1, 2)
        z = self.conv(z).transpose(1, 2)
        z, _ = self.lstm(z)
        return self.head(z)


def build_model(name: str, n_features: int, cfg: RunConfig) -> nn.Module:
    if name == "mlp":
        return PointwiseMLP(n_features, cfg.hidden_size, cfg.dropout, bayesian=False)
    if name == "bayesian_mlp":
        return PointwiseMLP(n_features, cfg.hidden_size, cfg.dropout, bayesian=True)
    if name == "rnn":
        return RecurrentModel("rnn", n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if name == "lstm":
        return RecurrentModel("lstm", n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    if name == "bilstm":
        return RecurrentModel(
            "lstm", n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout, bidirectional=True
        )
    if name == "cnn":
        return CNNModel(n_features, cfg.hidden_size, cfg.dropout)
    if name == "cnn_bilstm":
        return CNNBiLSTM(n_features, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    raise ValueError(f"Unknown model: {name}")


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.unsqueeze(-1).expand_as(pred)
    squared = (pred - target) ** 2
    denom = expanded.sum().clamp_min(1.0)
    return (squared * expanded).sum() / denom


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_weighted_loss = 0.0
    total_valid_values = 0.0

    for x, q, y, mask, _ in loader:
        x = x.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x, q)
        loss = masked_mse(pred, y, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("NaN/Inf loss detected")
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        valid_values = float(mask.sum().item() * y.shape[-1])
        total_weighted_loss += float(loss.item()) * valid_values
        total_valid_values += valid_values

    return total_weighted_loss / max(total_valid_values, 1.0)


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, targets, masks, local_indices = [], [], [], []
    with torch.no_grad():
        for x, q, y, mask, idx in loader:
            pred = model(x.to(device), q.to(device))
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
            masks.append(mask.numpy())
            local_indices.append(np.asarray(idx))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(masks, axis=0),
        np.concatenate(local_indices, axis=0),
    )


def inverse_targets(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 2)).reshape(original_shape)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    valid = mask.astype(bool)
    vt = y_true[:, :, 0][valid]
    vp = y_pred[:, :, 0][valid]
    tt = y_true[:, :, 1][valid]
    tp = y_pred[:, :, 1][valid]

    def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
        return float(r2_score(a, b)) if len(a) >= 2 and np.var(a) > 0 else float("nan")

    return {
        "v_mae": float(mean_absolute_error(vt, vp)),
        "v_rmse": float(math.sqrt(mean_squared_error(vt, vp))),
        "v_r2": safe_r2(vt, vp),
        "t_mae": float(mean_absolute_error(tt, tp)),
        "t_rmse": float(math.sqrt(mean_squared_error(tt, tp))),
        "t_r2": safe_r2(tt, tp),
        "n_valid_points": int(valid.sum()),
    }


def model_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int,
) -> Tuple[float, float]:
    model.eval()
    batches = list(loader)
    if not batches:
        return float("nan"), float("nan")

    with torch.no_grad():
        # One warm-up pass.
        for x, q, _, _, _ in batches:
            _ = model(x.to(device), q.to(device))
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = []
        for _ in range(repeats):
            start = time.perf_counter()
            for x, q, _, _, _ in batches:
                _ = model(x.to(device), q.to(device))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter() - start)

    seconds_total = float(np.mean(elapsed))
    ms_per_sequence = seconds_total * 1000.0 / len(loader.dataset)
    return seconds_total, ms_per_sequence


def plot_learning_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train")
    ax.plot(history["epoch"], history["val_loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized masked MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    channel: int,
    path: Path,
    max_points: int,
    seed: int,
) -> None:
    valid = mask.astype(bool)
    truth = y_true[:, :, channel][valid]
    pred = y_pred[:, :, channel][valid]
    if len(truth) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(truth), size=max_points, replace=False)
        truth, pred = truth[idx], pred[idx]
    rmse = math.sqrt(mean_squared_error(truth, pred))
    r2 = r2_score(truth, pred)
    label = "Voltage (V)" if channel == 0 else "Temperature (°C)"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(truth, pred, s=7, alpha=0.25)
    low = min(float(truth.min()), float(pred.min()))
    high = max(float(truth.max()), float(pred.max()))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2)
    ax.set_xlabel(f"Target (aligned simulation) {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{label} parity\nRMSE={rmse:.4f}, R²={r2:.4f}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_response_examples(
    q_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    metadata: pd.DataFrame,
    path: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    # Only sequences with at least two valid grid points can produce a
    # non-blank response curve; never select invalid or one-point sequences.
    candidates = np.flatnonzero(mask.astype(bool).sum(axis=1) >= 2)
    if candidates.size == 0:
        return
    n = min(3, candidates.size)
    chosen = rng.choice(candidates, size=n, replace=False)
    target_label = "Target (aligned simulation)"
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.5 * n), squeeze=False)
    for row_idx, i in enumerate(chosen):
        end_cap = float(metadata.iloc[i]["experimental_end_capacity_Ah"])
        cap = q_grid * end_cap
        valid = mask[i].astype(bool)
        title = str(metadata.iloc[i]["sequence_id"])

        axes[row_idx, 0].plot(cap[valid], y_true[i, valid, 0], label=target_label)
        axes[row_idx, 0].plot(cap[valid], y_pred[i, valid, 0], label="Predicted")
        axes[row_idx, 0].set_xlabel("Capacity (Ah)")
        axes[row_idx, 0].set_ylabel("Voltage (V)")
        axes[row_idx, 0].set_title(title)
        axes[row_idx, 0].grid(alpha=0.2)
        axes[row_idx, 0].legend()

        axes[row_idx, 1].plot(cap[valid], y_true[i, valid, 1], label=target_label)
        axes[row_idx, 1].plot(cap[valid], y_pred[i, valid, 1], label="Predicted")
        axes[row_idx, 1].set_xlabel("Capacity (Ah)")
        axes[row_idx, 1].set_ylabel("Temperature (°C)")
        axes[row_idx, 1].set_title(title)
        axes[row_idx, 1].grid(alpha=0.2)
        axes[row_idx, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_preprocessing_diagnostic(
    dataset_dir: Path,
    metadata: pd.DataFrame,
    sequence_length: int,
    path: Path,
) -> None:
    """Overlay raw, aligned/interpolated and experimental curves per operation.

    One row per representative case (a charge and a discharge case). For each the
    left panel shows voltage and the right panel temperature, with three traces:
    the raw simulation on its native capacity axis, the aligned/interpolated
    simulation on the common capacity grid (carrying the endpoint-held tail), and
    the experimental reference on the same grid.
    """
    _, _, h5_path = ensure_dataset(dataset_dir)
    q_grid = np.linspace(0.0, 1.0, sequence_length, dtype=np.float64)

    def pick(operation: str):
        sub = metadata.loc[metadata["operation"].eq(operation)]
        if sub.empty:
            return None
        # Prefer a sequence with a visible early-termination tail.
        if "end_capacity_error_fraction" in sub.columns:
            sub = sub.sort_values("end_capacity_error_fraction")
        return sub.iloc[0]

    rows = [(op, pick(op)) for op in ("charge", "discharge")]
    rows = [(op, r) for op, r in rows if r is not None]
    if not rows:
        return

    with h5py.File(h5_path, "r") as h5:
        acap = h5["aligned/experimental_capacity_Ah"]
        av = h5["aligned/simulated_voltage_V"]
        at = h5["aligned/simulated_temperature_C"]
        ev = h5["aligned/experimental_voltage_V"]
        et = h5["aligned/experimental_temperature_C"]
        has_raw = {"raw_offset", "raw_length"}.issubset(metadata.columns)
        rcap = h5["raw/capacity_Ah"] if has_raw else None
        rv = h5["raw/voltage_V"] if has_raw else None
        rt = h5["raw/temperature_C"] if has_raw else None

        n = len(rows)
        fig, axes = plt.subplots(n, 2, figsize=(13, 4.0 * n), squeeze=False)
        for r_idx, (operation, row) in enumerate(rows):
            a0 = int(row["aligned_offset"])
            a1 = a0 + int(row["aligned_length"])
            cap_g = np.asarray(acap[a0:a1], dtype=np.float64)
            sim_v = np.asarray(av[a0:a1], dtype=np.float64)
            sim_t = np.asarray(at[a0:a1], dtype=np.float64)
            exp_v = np.asarray(ev[a0:a1], dtype=np.float64)
            exp_t = np.asarray(et[a0:a1], dtype=np.float64)
            title = f"{operation}: {row['sequence_id']}"

            if rcap is not None:
                r0 = int(row["raw_offset"])
                r1 = r0 + int(row["raw_length"])
                raw_cap = np.asarray(rcap[r0:r1], dtype=np.float64)
                raw_v = np.asarray(rv[r0:r1], dtype=np.float64)
                raw_t = np.asarray(rt[r0:r1], dtype=np.float64)
            else:
                raw_cap = raw_v = raw_t = None

            axV, axT = axes[r_idx, 0], axes[r_idx, 1]
            if raw_cap is not None:
                axV.plot(raw_cap, raw_v, "o", ms=2.5, alpha=0.5, label="Raw simulation")
                axT.plot(raw_cap, raw_t, "o", ms=2.5, alpha=0.5, label="Raw simulation")
            axV.plot(cap_g, sim_v, "-", lw=1.6, label="Aligned/interpolated simulation")
            axT.plot(cap_g, sim_t, "-", lw=1.6, label="Aligned/interpolated simulation")
            axV.plot(cap_g, exp_v, "--", lw=1.2, label="Experimental reference")
            axT.plot(cap_g, exp_t, "--", lw=1.2, label="Experimental reference")
            sim_end = float(row.get("simulated_end_capacity_Ah", np.nan))
            if np.isfinite(sim_end):
                for ax in (axV, axT):
                    ax.axvline(sim_end, color="grey", ls=":", lw=1.0,
                               label="Simulated end capacity")
            axV.set_xlabel("Capacity (Ah)"); axV.set_ylabel("Voltage (V)")
            axT.set_xlabel("Capacity (Ah)"); axT.set_ylabel("Temperature (°C)")
            axV.set_title(title); axT.set_title(title)
            axV.grid(alpha=0.2); axT.grid(alpha=0.2)
            axV.legend(fontsize=8); axT.legend(fontsize=8)

    fig.suptitle("Preprocessing diagnostic: raw vs aligned vs experimental", y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    rank_cols = []
    for col in ["v_mae", "v_rmse", "t_mae", "t_rmse"]:
        rc = f"rank_{col}"
        ranked[rc] = ranked[col].rank(method="average", ascending=True)
        rank_cols.append(rc)
    for col in ["v_r2", "t_r2"]:
        rc = f"rank_{col}"
        ranked[rc] = ranked[col].rank(method="average", ascending=False)
        rank_cols.append(rc)
    ranked["average_rank"] = ranked[rank_cols].mean(axis=1)
    return ranked.sort_values("average_rank").reset_index(drop=True)


def plot_ranking_heatmap(ranked: pd.DataFrame, path: Path) -> None:
    cols = ["v_mae", "v_rmse", "v_r2", "t_mae", "t_rmse", "t_r2", "average_rank"]
    labels = ["V MAE", "V RMSE", "V R²", "T MAE", "T RMSE", "T R²", "Average rank"]
    values = ranked[cols].to_numpy(dtype=float)
    # Heatmap color is based on metric-specific rank, not incomparable raw scales.
    color_values = np.column_stack(
        [
            ranked["rank_v_mae"],
            ranked["rank_v_rmse"],
            ranked["rank_v_r2"],
            ranked["rank_t_mae"],
            ranked["rank_t_rmse"],
            ranked["rank_t_r2"],
            ranked["average_rank"],
        ]
    )
    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(ranked) + 1.5)))
    im = ax.imshow(color_values, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ranked)), labels=ranked["display_name"].tolist())
    ax.set_title("Time-series response model ranking")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text = f"{value:.4f}" if j != len(cols) - 1 else f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric-specific rank (1 = best)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_split(metadata: pd.DataFrame, split_ids: Dict[str, List[str]]) -> Dict[str, int]:
    counts = {}
    seen: Dict[str, str] = {}
    for split, ids in split_ids.items():
        for sample_id in ids:
            if sample_id in seen:
                raise ValueError(f"sample_id leakage: {sample_id} in {seen[sample_id]} and {split}")
            seen[sample_id] = split
        counts[f"{split}_sample_ids"] = len(ids)
        counts[f"{split}_sequences"] = int(metadata["sample_id"].isin(ids).sum())
    if set(metadata["sample_id"].unique()) != set(seen):
        raise ValueError("Split IDs do not exactly cover the selected metadata")
    return counts


def main() -> int:
    args = parse_args()
    max_sample_ids = None if args.max_sample_ids == 0 else args.max_sample_ids
    cfg = RunConfig(
        dataset_dir=str(args.dataset_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        models=list(dict.fromkeys(args.models)),
        sequence_length=args.sequence_length,
        max_sample_ids=max_sample_ids,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        inference_repeats=args.inference_repeats,
        parity_max_points=args.parity_max_points,
        min_valid_fraction=args.min_valid_fraction,
        alignment_mode=args.alignment_mode,
    )

    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    dirs = {
        "checkpoints": output_dir / "best_checkpoints",
        "figures": output_dir / "figures",
        "metrics": output_dir / "metrics",
        "predictions": output_dir / "predictions",
        "artifacts": output_dir / "artifacts",
        "logs": output_dir / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", asdict(cfg))

    print(f"[1/7] Loading and resampling dataset (alignment_mode={cfg.alignment_mode})...", flush=True)
    metadata, X_raw, Y_raw, mask, q_grid, feature_names = load_and_resample(
        Path(cfg.dataset_dir),
        cfg.sequence_length,
        cfg.max_sample_ids,
        cfg.seed,
        cfg.min_valid_fraction,
        cfg.alignment_mode,
        excluded_csv=dirs["artifacts"] / "excluded_sequences.csv",
    )
    metadata.to_csv(dirs["artifacts"] / "selected_manifest.csv", index=False)
    save_json(dirs["artifacts"] / "feature_names.json", feature_names)
    plot_preprocessing_diagnostic(
        Path(cfg.dataset_dir),
        metadata,
        cfg.sequence_length,
        dirs["figures"] / "preprocessing_diagnostic.png",
    )

    print("[2/7] Creating group-aware sample_id split...", flush=True)
    split_ids = split_sample_ids(
        metadata["sample_id"].tolist(),
        cfg.train_fraction,
        cfg.val_fraction,
        cfg.test_fraction,
        cfg.seed,
    )
    split_counts = validate_split(metadata, split_ids)
    save_json(dirs["artifacts"] / "split_sample_ids.json", split_ids)
    save_json(dirs["artifacts"] / "split_counts.json", split_counts)

    split_indices = {
        split: np.flatnonzero(metadata["sample_id"].isin(ids).to_numpy())
        for split, ids in split_ids.items()
    }

    print("[3/7] Fitting train-only scalers...", flush=True)
    x_scaler = StandardScaler().fit(X_raw[split_indices["train"]])
    train_mask = mask[split_indices["train"]].astype(bool)
    train_targets = Y_raw[split_indices["train"]][train_mask]
    y_scaler = StandardScaler().fit(train_targets.reshape(-1, 2))

    X = x_scaler.transform(X_raw).astype(np.float32)
    Y = y_scaler.transform(Y_raw.reshape(-1, 2)).reshape(Y_raw.shape).astype(np.float32)
    save_json(
        dirs["artifacts"] / "scalers.json",
        {
            "x_mean": x_scaler.mean_.tolist(),
            "x_scale": x_scaler.scale_.tolist(),
            "y_mean": y_scaler.mean_.tolist(),
            "y_scale": y_scaler.scale_.tolist(),
            "fit_on": "train split only",
        },
    )

    # Reproducibility provenance (additive; does not affect training).
    reporting.write_environment(output_dir / "environment.json", REPO)
    reporting.write_dataset_audit(
        output_dir / "dataset_audit.json",
        Path(cfg.dataset_dir),
        split_counts=split_counts,
        selected={
            "unique_sample_ids": int(metadata["sample_id"].nunique()),
            "sequences": int(len(metadata)),
            "sequence_length": cfg.sequence_length,
            "fixed_length_method": (
                "reparametrize each aligned curve onto q=clip(experimental_capacity/"
                "experimental_end_capacity,0,1), then np.interp onto "
                "linspace(0,1,sequence_length); np.interp holds endpoints so the "
                "endpoint-held aligned tail is preserved"
            ),
            "alignment_mode": cfg.alignment_mode,
        },
    )

    datasets = {
        split: SequenceDataset(X, Y, mask, q_grid, idx, metadata)
        for split, idx in split_indices.items()
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    all_metrics: List[Dict[str, object]] = []
    all_timing: List[Dict[str, object]] = []
    all_per_case: List[Dict[str, object]] = []
    all_per_operation: List[Dict[str, object]] = []
    all_per_c_rate: List[Dict[str, object]] = []
    all_predictions: List[Dict[str, object]] = []
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]] = {}

    print("[4/7] Training models...", flush=True)
    for model_name in cfg.models:
        print(f"\n===== {DISPLAY_NAMES[model_name]} =====", flush=True)
        set_seed(cfg.seed)
        model = build_model(model_name, X.shape[1], cfg).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(2, cfg.patience // 2)
        )
        checkpoint_path = dirs["checkpoints"] / f"{model_name}_best.pt"
        history_rows = []
        best_val = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = run_epoch(model, loaders["train"], device, optimizer)
            with torch.no_grad():
                val_loss = run_epoch(model, loaders["val"], device, optimizer=None)
            scheduler.step(val_loss)
            elapsed_epoch = time.perf_counter() - epoch_start
            lr = optimizer.param_groups[0]["lr"]
            history_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": lr,
                    "epoch_seconds": elapsed_epoch,
                }
            )
            print(
                f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
                f"lr={lr:.2e} time={elapsed_epoch:.1f}s",
                flush=True,
            )

            if val_loss < best_val - 1e-8:
                best_val = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val,
                        "run_config": asdict(cfg),
                        "feature_names": feature_names,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= cfg.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        total_training_seconds = time.perf_counter() - train_start
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(dirs["logs"] / f"{model_name}_history.csv", index=False)
        plot_learning_curve(
            history_df,
            dirs["figures"] / f"{model_name}_learning_curves.png",
            f"{DISPLAY_NAMES[model_name]} learning curves",
        )

        pred_scaled, target_scaled, test_mask, local_idx = predict_loader(
            model, loaders["test"], device
        )
        pred = inverse_targets(pred_scaled, y_scaler)
        target = inverse_targets(target_scaled, y_scaler)
        test_meta = datasets["test"].metadata.iloc[local_idx].reset_index(drop=True)
        metrics = metric_dict(target, pred, test_mask)
        metrics.update(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "parameter_count": model_parameter_count(model),
            }
        )
        all_metrics.append(metrics)

        inference_seconds_total, inference_ms_per_sequence = measure_inference(
            model, loaders["test"], device, cfg.inference_repeats
        )
        all_timing.append(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES[model_name],
                "best_epoch": best_epoch,
                "checkpoint_selection_metric": "validation_normalized_masked_mse",
                "best_validation_score": best_val,
                "total_training_seconds": total_training_seconds,
                "seconds_per_epoch": total_training_seconds / len(history_df),
                "inference_seconds_total": inference_seconds_total,
                "inference_ms_per_sequence": inference_ms_per_sequence,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "parameter_count": model_parameter_count(model),
            }
        )

        for experiment_id, group in test_meta.groupby("experiment_id"):
            idx = group.index.to_numpy()
            case_metrics = metric_dict(target[idx], pred[idx], test_mask[idx])
            case_metrics.update(
                {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES[model_name],
                    "experiment_id": experiment_id,
                    "operation": group["operation"].iloc[0],
                    "c_rate": float(group["c_rate"].iloc[0]),
                    "n_sequences": len(group),
                }
            )
            all_per_case.append(case_metrics)

        # Per-operation (charge/discharge) and per-C-rate aggregates.
        for op_val, op_group in test_meta.groupby("operation"):
            gidx = op_group.index.to_numpy()
            om = metric_dict(target[gidx], pred[gidx], test_mask[gidx])
            om.update(
                {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES[model_name],
                    "operation": op_val,
                    "n_sequences": len(op_group),
                }
            )
            all_per_operation.append(om)
        for c_val, c_group in test_meta.groupby("c_rate"):
            gidx = c_group.index.to_numpy()
            cm = metric_dict(target[gidx], pred[gidx], test_mask[gidx])
            cm.update(
                {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES[model_name],
                    "c_rate": float(c_val),
                    "n_sequences": len(c_group),
                }
            )
            all_per_c_rate.append(cm)

        # Per-sequence long-form predictions row (canonical predictions.csv).
        for local_row in range(len(test_meta)):
            seq_mask = test_mask[local_row].astype(bool)
            meta_row = test_meta.iloc[local_row]
            if seq_mask.any():
                v_rmse = float(
                    math.sqrt(
                        mean_squared_error(
                            target[local_row, :, 0][seq_mask],
                            pred[local_row, :, 0][seq_mask],
                        )
                    )
                )
                t_rmse = float(
                    math.sqrt(
                        mean_squared_error(
                            target[local_row, :, 1][seq_mask],
                            pred[local_row, :, 1][seq_mask],
                        )
                    )
                )
            else:
                v_rmse = t_rmse = float("nan")
            all_predictions.append(
                {
                    "model": model_name,
                    "display_name": DISPLAY_NAMES[model_name],
                    "sequence_id": meta_row["sequence_id"],
                    "sample_id": meta_row["sample_id"],
                    "experiment_id": meta_row["experiment_id"],
                    "operation": meta_row["operation"],
                    "c_rate": float(meta_row["c_rate"]),
                    "v_rmse": v_rmse,
                    "t_rmse": t_rmse,
                    "n_valid_points": int(seq_mask.sum()),
                }
            )

        np.savez_compressed(
            dirs["predictions"] / f"{model_name}_test_predictions.npz",
            prediction=pred.astype(np.float32),
            target=target.astype(np.float32),
            mask=test_mask.astype(np.uint8),
            q_grid=q_grid.astype(np.float32),
            sequence_id=test_meta["sequence_id"].astype(str).to_numpy(),
        )
        test_meta.to_csv(dirs["predictions"] / f"{model_name}_test_metadata.csv", index=False)
        plot_parity(
            target,
            pred,
            test_mask,
            channel=0,
            path=dirs["figures"] / f"{model_name}_voltage_parity.png",
            max_points=cfg.parity_max_points,
            seed=cfg.seed,
        )
        plot_parity(
            target,
            pred,
            test_mask,
            channel=1,
            path=dirs["figures"] / f"{model_name}_temperature_parity.png",
            max_points=cfg.parity_max_points,
            seed=cfg.seed,
        )
        plot_response_examples(
            q_grid,
            target,
            pred,
            test_mask,
            test_meta,
            dirs["figures"] / f"{model_name}_response_curves_vs_capacity.png",
            cfg.seed,
        )
        model_predictions[model_name] = (target, pred, test_mask, test_meta)
        print(
            f"Test: V RMSE={metrics['v_rmse']:.5f} V R²={metrics['v_r2']:.5f} | "
            f"T RMSE={metrics['t_rmse']:.5f} T R²={metrics['t_r2']:.5f}",
            flush=True,
        )

    print("[5/7] Aggregating metrics and ranking...", flush=True)
    metrics_df = pd.DataFrame(all_metrics)
    timing_df = pd.DataFrame(all_timing)
    # Additive: device/CUDA/GPU/throughput/test-batch columns.
    timing_df = reporting.enrich_timing(
        timing_df, device, cfg.batch_size, len(datasets["test"])
    )
    per_case_df = pd.DataFrame(all_per_case)
    per_operation_df = pd.DataFrame(all_per_operation)
    per_c_rate_df = pd.DataFrame(all_per_c_rate)
    predictions_df = pd.DataFrame(all_predictions)
    ranked_df = make_ranking(metrics_df)

    metrics_df.to_csv(dirs["metrics"] / "model_metrics.csv", index=False)
    timing_df.to_csv(dirs["metrics"] / "model_timing.csv", index=False)
    per_case_df.to_csv(dirs["metrics"] / "per_case_metrics.csv", index=False)
    per_operation_df.to_csv(dirs["metrics"] / "per_operation_metrics.csv", index=False)
    per_c_rate_df.to_csv(dirs["metrics"] / "per_c_rate_metrics.csv", index=False)
    predictions_df.to_csv(dirs["metrics"] / "predictions.csv", index=False)
    ranked_df.to_csv(dirs["metrics"] / "model_ranking.csv", index=False)
    # Dedicated voltage/temperature metric tables (subsets of model_metrics).
    v_cols = ["model", "display_name", "v_mae", "v_rmse", "v_r2", "n_valid_points"]
    t_cols = ["model", "display_name", "t_mae", "t_rmse", "t_r2", "n_valid_points"]
    metrics_df[v_cols].to_csv(dirs["metrics"] / "voltage_metrics.csv", index=False)
    metrics_df[t_cols].to_csv(dirs["metrics"] / "temperature_metrics.csv", index=False)
    plot_ranking_heatmap(ranked_df, dirs["figures"] / "model_ranking_heatmap.png")

    best_name = str(ranked_df.iloc[0]["model"])
    target, pred, best_mask, best_meta = model_predictions[best_name]
    plot_response_examples(
        q_grid,
        target,
        pred,
        best_mask,
        best_meta,
        dirs["figures"] / "best_model_response_curves_vs_capacity.png",
        cfg.seed + 1,
    )

    print("[6/7] Writing summary...", flush=True)
    summary_lines = [
        "# Emergency LHS Time-series Training Summary",
        "",
        f"- Dataset: `{cfg.dataset_dir}`",
        f"- Selected unique sample IDs: {metadata['sample_id'].nunique()}",
        f"- Selected successful sequences: {len(metadata)}",
        f"- Sequence length: {cfg.sequence_length}",
        f"- Models: {', '.join(DISPLAY_NAMES[m] for m in cfg.models)}",
        f"- Split: {cfg.train_fraction:.0%}/{cfg.val_fraction:.0%}/{cfg.test_fraction:.0%} by `sample_id`",
        f"- Best preliminary model: **{DISPLAY_NAMES[best_name]}**",
        f"- Alignment mode: **{cfg.alignment_mode}**",
        "- Best checkpoints selected using validation normalized masked MSE only.",
        (
            "- official_clamped: endpoint-held horizontal tails retained; the "
            "entire aligned sequence enters loss and metrics."
            if cfg.alignment_mode == "official_clamped"
            else "- masked: held-constant tails after simulated end capacity "
            "were excluded by a valid mask (sensitivity mode)."
        ),
        "",
        "## Ranking",
        "",
        "```text",
        ranked_df[["display_name", "v_rmse", "v_r2", "t_rmse", "t_r2", "average_rank"]].to_string(index=False),
        "```",
        "",
        "## Split counts",
        "",
        "```json",
        json.dumps(split_counts, indent=2),
        "```",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("[7/7] Done.", flush=True)
    print(f"Output: {output_dir}")
    print(ranked_df[["display_name", "v_rmse", "v_r2", "t_rmse", "t_r2", "average_rank"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
