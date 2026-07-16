"""Inference helpers: checkpoint / scaler loading and curve reconstruction.

These functions are deliberately model-agnostic.  They reconstruct the exact
train/val/test split used during training (via the saved ``run_config.json``),
so predictions can be lined up against ground truth for any split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import torch.nn as nn

from . import data as data_mod
from .models import build_model, split_prediction
from .utils import load_json, resolve_device

PathLike = Union[str, Path]

# Dropout module types whose stochasticity we re-enable for MC-Dropout.
_DROPOUT_TYPES = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)


# --------------------------------------------------------------------------- #
# Standard output-path layout
# --------------------------------------------------------------------------- #
def checkpoint_dir(outputs_dir: PathLike, case_id: str, model_name: str) -> Path:
    return Path(outputs_dir) / "checkpoints" / case_id / model_name


def scaler_dir(outputs_dir: PathLike, case_id: str, model_name: str) -> Path:
    return Path(outputs_dir) / "scalers" / case_id / model_name


def metrics_dir(outputs_dir: PathLike, case_id: str, model_name: str) -> Path:
    return Path(outputs_dir) / "metrics" / case_id / model_name


def figures_dir(outputs_dir: PathLike, case_id: str, model_name: str) -> Path:
    return Path(outputs_dir) / "figures" / case_id / model_name


def predictions_dir(outputs_dir: PathLike, case_id: str, model_name: str) -> Path:
    return Path(outputs_dir) / "predictions" / case_id / model_name


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_checkpoint(
    checkpoint_path: PathLike, device: str = "auto"
) -> Tuple[torch.nn.Module, Dict]:
    """Rebuild a model from a checkpoint and return ``(model, checkpoint)``.

    The checkpoint stores ``model_name`` and ``model_kwargs`` so the exact
    architecture can be reconstructed without external configuration.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dev = resolve_device(device)
    checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model = build_model(checkpoint["model_name"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev).eval()
    return model, checkpoint


def load_scalers(
    outputs_dir: PathLike, case_id: str, model_name: str
) -> Tuple[StandardScaler, StandardScaler, StandardScaler]:
    """Load the x / v / t scalers saved during training."""
    sdir = scaler_dir(outputs_dir, case_id, model_name)
    paths = {
        "x": sdir / "x_scaler.joblib",
        "v": sdir / "v_scaler.joblib",
        "t": sdir / "t_scaler.joblib",
    }
    for key, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {key}_scaler for {case_id}/{model_name}: {p}")
    return (
        joblib.load(paths["x"]),
        joblib.load(paths["v"]),
        joblib.load(paths["t"]),
    )


# --------------------------------------------------------------------------- #
# Output reconstruction
# --------------------------------------------------------------------------- #
def split_mlp_output(output: np.ndarray, t_last: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split an MLP output ``[N, 2*t_last]`` into ``(V, T)`` halves."""
    return output[:, :t_last], output[:, t_last:]


def inverse_transform_predictions(
    v_scaled: np.ndarray,
    t_scaled: np.ndarray,
    v_scaler: StandardScaler,
    t_scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map scaled voltage / temperature back to physical units."""
    v_phys = v_scaler.inverse_transform(np.asarray(v_scaled))
    t_phys = t_scaler.inverse_transform(np.asarray(t_scaled))
    return v_phys, t_phys


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    dataset,
    model_name: str,
    t_last: int,
    device: str,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a model over a dataset and return scaled ``(V, T)`` predictions."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    v_chunks, t_chunks = [], []
    model.eval()
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        v, t = split_prediction(model_name, out, t_last)
        v_chunks.append(v.cpu().numpy())
        t_chunks.append(t.cpu().numpy())
    return np.concatenate(v_chunks, axis=0), np.concatenate(t_chunks, axis=0)


# --------------------------------------------------------------------------- #
# Monte-Carlo Dropout (approximate Bayesian inference)
# --------------------------------------------------------------------------- #
def enable_mc_dropout(model: torch.nn.Module) -> torch.nn.Module:
    """Put the model in eval mode but re-enable *only* its dropout layers.

    This is the core trick behind MC-Dropout: BatchNorm / other layers stay in
    eval mode (using running statistics) while dropout keeps sampling, so each
    forward pass yields a different stochastic prediction.  Non-Bayesian models
    are unaffected unless this helper is explicitly called.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, _DROPOUT_TYPES):
            module.train()
    return model


@torch.no_grad()
def predict_mc_dropout(
    model: torch.nn.Module, inputs: torch.Tensor, mc_samples: int = 30
):
    """Run ``mc_samples`` stochastic forward passes and return ``(mean, std)``.

    ``inputs`` is any tensor the model accepts; the returned tensors have the
    model's output shape.  Dropout is enabled via :func:`enable_mc_dropout`.
    """
    enable_mc_dropout(model)
    preds = [model(inputs) for _ in range(max(1, mc_samples))]
    stacked = torch.stack(preds, dim=0)            # [mc, *out_shape]
    return stacked.mean(dim=0), stacked.std(dim=0)


@torch.no_grad()
def _run_inference_mc(
    model: torch.nn.Module,
    dataset,
    model_name: str,
    t_last: int,
    device: str,
    mc_samples: int,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """MC-Dropout inference over a dataset.

    Returns scaled ``(v_mean, v_std, t_mean, t_std)`` each ``[N, t_last]``,
    computed across ``mc_samples`` stochastic passes.
    """
    enable_mc_dropout(model)
    v_passes, t_passes = [], []
    for _ in range(max(1, mc_samples)):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        v_chunks, t_chunks = [], []
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            v, t = split_prediction(model_name, out, t_last)
            v_chunks.append(v.cpu().numpy())
            t_chunks.append(t.cpu().numpy())
        v_passes.append(np.concatenate(v_chunks, axis=0))
        t_passes.append(np.concatenate(t_chunks, axis=0))
    v_stack = np.stack(v_passes, axis=0)           # [mc, N, t_last]
    t_stack = np.stack(t_passes, axis=0)
    return (
        v_stack.mean(axis=0), v_stack.std(axis=0),
        t_stack.mean(axis=0), t_stack.std(axis=0),
    )


def predict_case_uncertainty(
    data_root: PathLike,
    case_id: str,
    model_name: str,
    outputs_dir: PathLike = "outputs",
    checkpoint_name: str = "best_model.pt",
    split: str = "test",
    device: str = "auto",
    batch_size: int = 256,
    seed: Optional[int] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    mc_samples: int = 30,
) -> Dict[str, np.ndarray]:
    """Like :func:`predict_case` but with MC-Dropout mean + std (physical units).

    Returns the same keys as :func:`predict_case` (with ``v_pred``/``t_pred``
    being the MC mean) plus ``v_std`` / ``t_std`` standard-deviation arrays in
    physical units.  Intended for ``bayesian_mlp`` but works for any model with
    dropout layers.
    """
    dev = resolve_device(device)

    if seed is None or ratios is None:
        run_cfg = load_json(metrics_dir(outputs_dir, case_id, model_name) / "run_config.json")
        seed = run_cfg["seed"] if seed is None else seed
        ratios = (
            (run_cfg["train_ratio"], run_cfg["val_ratio"], run_cfg["test_ratio"])
            if ratios is None
            else ratios
        )

    bundle = data_mod.build_datasets(
        model_name, data_root, case_id,
        train_ratio=ratios[0], val_ratio=ratios[1], test_ratio=ratios[2], seed=seed,
    )
    if split not in bundle.splits:
        raise ValueError(f"Unknown split '{split}'. Choices: {list(bundle.splits)}")
    dataset = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[split]
    idx = bundle.splits[split]
    case = bundle.case
    t_last = case.t_last

    model, _ = load_checkpoint(
        checkpoint_dir(outputs_dir, case_id, model_name) / checkpoint_name, device=dev
    )
    v_scaler, t_scaler = bundle.v_scaler, bundle.t_scaler

    v_mean_s, v_std_s, t_mean_s, t_std_s = _run_inference_mc(
        model, dataset, model_name, t_last, dev, mc_samples, batch_size
    )
    # Mean maps back through the full affine inverse; std only scales (the shift
    # cancels in a difference), so multiply by the per-feature scale_.
    v_pred, t_pred = inverse_transform_predictions(v_mean_s, t_mean_s, v_scaler, t_scaler)
    v_std = np.asarray(v_std_s) * np.asarray(v_scaler.scale_)[None, :]
    t_std = np.asarray(t_std_s) * np.asarray(t_scaler.scale_)[None, :]

    return {
        "v_pred": v_pred,
        "t_pred": t_pred,
        "v_std": v_std,
        "t_std": t_std,
        "v_true": case.V[idx],
        "t_true": case.T[idx],
        "time_s": case.time_s,
        "indices": idx,
        "sample_ids": case.sample_ids[idx],
    }


def predict_case(
    data_root: PathLike,
    case_id: str,
    model_name: str,
    outputs_dir: PathLike = "outputs",
    checkpoint_name: str = "best_model.pt",
    split: str = "test",
    device: str = "auto",
    batch_size: int = 256,
    seed: Optional[int] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
) -> Dict[str, np.ndarray]:
    """Predict physical V/T curves for one split of a case.

    The split is reconstructed from the training ``run_config.json`` (seed and
    ratios) unless ``seed`` / ``ratios`` are supplied explicitly.

    Returns a dict with ``v_pred``, ``t_pred``, ``v_true``, ``t_true`` (all
    ``[n_split, t_last]``), plus ``indices``, ``sample_ids`` and ``time_s``.
    """
    dev = resolve_device(device)

    # Recover the split configuration used at training time.
    if seed is None or ratios is None:
        run_cfg = load_json(metrics_dir(outputs_dir, case_id, model_name) / "run_config.json")
        seed = run_cfg["seed"] if seed is None else seed
        ratios = (
            (run_cfg["train_ratio"], run_cfg["val_ratio"], run_cfg["test_ratio"])
            if ratios is None
            else ratios
        )

    bundle = data_mod.build_datasets(
        model_name,
        data_root,
        case_id,
        train_ratio=ratios[0],
        val_ratio=ratios[1],
        test_ratio=ratios[2],
        seed=seed,
    )
    if split not in bundle.splits:
        raise ValueError(f"Unknown split '{split}'. Choices: {list(bundle.splits)}")
    dataset = {"train": bundle.train, "val": bundle.val, "test": bundle.test}[split]
    idx = bundle.splits[split]
    case = bundle.case
    t_last = case.t_last

    model, _ = load_checkpoint(
        checkpoint_dir(outputs_dir, case_id, model_name) / checkpoint_name, device=dev
    )
    v_scaler, t_scaler = bundle.v_scaler, bundle.t_scaler

    v_scaled, t_scaled = _run_inference(model, dataset, model_name, t_last, dev, batch_size)
    v_pred, t_pred = inverse_transform_predictions(v_scaled, t_scaled, v_scaler, t_scaler)

    return {
        "v_pred": v_pred,
        "t_pred": t_pred,
        "v_true": case.V[idx],
        "t_true": case.T[idx],
        "time_s": case.time_s,
        "indices": idx,
        "sample_ids": case.sample_ids[idx],
    }
