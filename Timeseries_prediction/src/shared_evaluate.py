"""Evaluation of shared models on the held-out test curves.

All metrics are computed in **physical units** after inverse-transforming the
network's normalized predictions.  Because cases have different ``t_last`` the
test set is a list of variable-length curves, so:

* "global" metrics (MAE/RMSE/R2/MaxError) are computed over every
  ``(curve, timestep)`` point pooled together;
* "curve-specific" metrics (end / peak / per-curve RMSE) are computed per curve
  then averaged.

In addition to the overall metrics the suite is recomputed for each group:
by ``case_id``, by ``ambient_temp_C`` and by ``c_rate``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .shared_data import (
    build_curve_features,
    load_all_curves,
    load_shared_scalers,
    shared_metrics_dir,
    split_by_sample_case,
)
from .predict import enable_mc_dropout
from .shared_models import build_shared_model
from .utils import ensure_dir, load_json, resolve_device, save_json

PathLike = Union[str, Path]

# Metric columns (physical units), matching the per-case metric suite.
METRIC_COLUMNS = [
    "MAE_V", "RMSE_V", "R2_V", "MaxError_V",
    "MAE_T", "RMSE_T", "R2_T", "MaxError_T",
    "voltage_end_mae", "temperature_end_mae", "temperature_peak_mae",
    "voltage_curve_rmse_mean", "temperature_curve_rmse_mean",
]


# --------------------------------------------------------------------------- #
# Metric helpers (operate on lists of variable-length 1D curves)
# --------------------------------------------------------------------------- #
def _r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 0.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def compute_shared_metrics(records: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Compute the full physical-unit metric suite from per-curve records.

    Each record holds 1D arrays ``v_true, v_pred, t_true, t_pred`` of that
    curve's own length.
    """
    if not records:
        return {col: float("nan") for col in METRIC_COLUMNS}

    v_true_all = np.concatenate([r["v_true"] for r in records])
    v_pred_all = np.concatenate([r["v_pred"] for r in records])
    t_true_all = np.concatenate([r["t_true"] for r in records])
    t_pred_all = np.concatenate([r["t_pred"] for r in records])

    # Per-curve aggregates.
    v_end = np.array([abs(r["v_true"][-1] - r["v_pred"][-1]) for r in records])
    t_end = np.array([abs(r["t_true"][-1] - r["t_pred"][-1]) for r in records])
    t_peak = np.array([abs(r["t_true"].max() - r["t_pred"].max()) for r in records])
    v_curve_rmse = np.array([np.sqrt(np.mean((r["v_true"] - r["v_pred"]) ** 2)) for r in records])
    t_curve_rmse = np.array([np.sqrt(np.mean((r["t_true"] - r["t_pred"]) ** 2)) for r in records])

    return {
        "MAE_V": float(np.mean(np.abs(v_true_all - v_pred_all))),
        "RMSE_V": float(np.sqrt(np.mean((v_true_all - v_pred_all) ** 2))),
        "R2_V": _r2(v_true_all, v_pred_all),
        "MaxError_V": float(np.max(np.abs(v_true_all - v_pred_all))),
        "MAE_T": float(np.mean(np.abs(t_true_all - t_pred_all))),
        "RMSE_T": float(np.sqrt(np.mean((t_true_all - t_pred_all) ** 2))),
        "R2_T": _r2(t_true_all, t_pred_all),
        "MaxError_T": float(np.max(np.abs(t_true_all - t_pred_all))),
        "voltage_end_mae": float(np.mean(v_end)),
        "temperature_end_mae": float(np.mean(t_end)),
        "temperature_peak_mae": float(np.mean(t_peak)),
        "voltage_curve_rmse_mean": float(np.mean(v_curve_rmse)),
        "temperature_curve_rmse_mean": float(np.mean(t_curve_rmse)),
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _load_shared_checkpoint(
    checkpoint_path: PathLike, device: str
) -> Tuple[torch.nn.Module, Dict]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_shared_model(checkpoint["model_name"], checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def predict_shared_curves(
    data_root: PathLike,
    model_name: str,
    *,
    outputs_dir: PathLike = "outputs",
    checkpoint_name: str = "best_model.pt",
    split: str = "test",
    device: str = "auto",
    seed: Optional[int] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    cases: Optional[List[str]] = None,
) -> List[Dict]:
    """Predict physical V/T curves for one split with a shared model.

    The split is reconstructed from the training ``run_config.json`` (seed and
    ratios) unless overridden.  Returns one record per curve with physical
    ``v_true/v_pred/t_true/t_pred`` plus ``case_id``, ``c_rate`` etc.
    """
    dev = resolve_device(device)
    mdir = shared_metrics_dir(outputs_dir, model_name)

    if seed is None or ratios is None:
        run_cfg = load_json(mdir / "run_config.json")
        seed = run_cfg["seed"] if seed is None else seed
        ratios = (
            (run_cfg["train_ratio"], run_cfg["val_ratio"], run_cfg["test_ratio"])
            if ratios is None
            else ratios
        )

    curves, _ = load_all_curves(data_root, cases=cases)
    splits = split_by_sample_case(len(curves), ratios[0], ratios[1], ratios[2], seed)
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Choices: {list(splits)}")

    x_scaler, v_scaler, t_scaler = load_shared_scalers(outputs_dir, model_name)
    ckpt_path = Path(outputs_dir) / "checkpoints" / "shared" / model_name / checkpoint_name
    model, checkpoint = _load_shared_checkpoint(ckpt_path, dev)
    is_sequence = bool(checkpoint.get("is_sequence", False))

    records: List[Dict] = []
    for ci in splits[split]:
        curve = curves[ci]
        feats = x_scaler.transform(build_curve_features(curve))
        x = torch.as_tensor(feats, dtype=torch.float32, device=dev)
        if is_sequence:
            lengths = torch.tensor([curve.t_last], dtype=torch.long)
            out = model(x.unsqueeze(0), lengths).squeeze(0)  # [T, 2]
        else:
            out = model(x)  # [T, 2]
        out = out.cpu().numpy()
        v_pred = v_scaler.inverse_transform(out[:, 0:1]).ravel()
        t_pred = t_scaler.inverse_transform(out[:, 1:2]).ravel()

        records.append(
            {
                "case_id": curve.case_id,
                "sample_id": curve.sample_id,
                "c_rate": curve.c_rate,
                "ambient_temp_C": curve.ambient_temp_C,
                "time_s": curve.time_s,
                "v_true": curve.V,
                "v_pred": v_pred,
                "t_true": curve.T,
                "t_pred": t_pred,
            }
        )
    return records


@torch.no_grad()
def predict_shared_curves_uncertainty(
    data_root: PathLike,
    model_name: str,
    *,
    outputs_dir: PathLike = "outputs",
    checkpoint_name: str = "best_model.pt",
    split: str = "test",
    device: str = "auto",
    seed: Optional[int] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    cases: Optional[List[str]] = None,
    mc_samples: int = 30,
) -> List[Dict]:
    """Like :func:`predict_shared_curves` but with MC-Dropout mean + std.

    Each returned record additionally carries physical-unit ``v_std`` / ``t_std``
    arrays.  ``v_pred`` / ``t_pred`` are the MC mean.  Used for shared Bayesian
    models; works for any shared model that contains dropout layers.
    """
    dev = resolve_device(device)
    mdir = shared_metrics_dir(outputs_dir, model_name)

    if seed is None or ratios is None:
        run_cfg = load_json(mdir / "run_config.json")
        seed = run_cfg["seed"] if seed is None else seed
        ratios = (
            (run_cfg["train_ratio"], run_cfg["val_ratio"], run_cfg["test_ratio"])
            if ratios is None
            else ratios
        )

    curves, _ = load_all_curves(data_root, cases=cases)
    splits = split_by_sample_case(len(curves), ratios[0], ratios[1], ratios[2], seed)
    if split not in splits:
        raise ValueError(f"Unknown split '{split}'. Choices: {list(splits)}")

    x_scaler, v_scaler, t_scaler = load_shared_scalers(outputs_dir, model_name)
    ckpt_path = Path(outputs_dir) / "checkpoints" / "shared" / model_name / checkpoint_name
    model, checkpoint = _load_shared_checkpoint(ckpt_path, dev)
    is_sequence = bool(checkpoint.get("is_sequence", False))
    # Keep BatchNorm etc. in eval mode but re-enable dropout for MC sampling.
    enable_mc_dropout(model)

    # std only scales under the affine inverse; scalers are point-wise (scalar).
    v_scale = float(np.asarray(v_scaler.scale_).ravel()[0])
    t_scale = float(np.asarray(t_scaler.scale_).ravel()[0])

    records: List[Dict] = []
    for ci in splits[split]:
        curve = curves[ci]
        feats = x_scaler.transform(build_curve_features(curve))
        x = torch.as_tensor(feats, dtype=torch.float32, device=dev)
        passes = []
        for _ in range(max(1, mc_samples)):
            if is_sequence:
                lengths = torch.tensor([curve.t_last], dtype=torch.long)
                out = model(x.unsqueeze(0), lengths).squeeze(0)  # [T, 2]
            else:
                out = model(x)  # [T, 2]
            passes.append(out.cpu().numpy())
        stack = np.stack(passes, axis=0)         # [mc, T, 2]
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)

        v_pred = v_scaler.inverse_transform(mean[:, 0:1]).ravel()
        t_pred = t_scaler.inverse_transform(mean[:, 1:2]).ravel()
        v_std = std[:, 0] * v_scale
        t_std = std[:, 1] * t_scale

        records.append(
            {
                "case_id": curve.case_id,
                "sample_id": curve.sample_id,
                "c_rate": curve.c_rate,
                "ambient_temp_C": curve.ambient_temp_C,
                "time_s": curve.time_s,
                "v_true": curve.V,
                "v_pred": v_pred,
                "t_true": curve.T,
                "t_pred": t_pred,
                "v_std": v_std,
                "t_std": t_std,
            }
        )
    return records


def _shared_uncertainty_summary(records: List[Dict]) -> Dict[str, float]:
    """Aggregate per-curve MC-Dropout std arrays into summary statistics."""
    v_std_all = np.concatenate([r["v_std"] for r in records])
    t_std_all = np.concatenate([r["t_std"] for r in records])
    v_end = np.array([r["v_std"][-1] for r in records])
    t_end = np.array([r["t_std"][-1] for r in records])
    return {
        "mean_std_V": float(np.mean(v_std_all)),
        "mean_std_T": float(np.mean(t_std_all)),
        "max_std_V": float(np.max(v_std_all)),
        "max_std_T": float(np.max(t_std_all)),
        "mean_std_V_end": float(np.mean(v_end)),
        "mean_std_T_end": float(np.mean(t_end)),
    }


# --------------------------------------------------------------------------- #
# Grouped metric tables
# --------------------------------------------------------------------------- #
def _grouped_metrics(records: List[Dict], key: str) -> pd.DataFrame:
    """Compute the metric suite within each distinct value of ``records[key]``."""
    rows = []
    for value in sorted({r[key] for r in records}):
        subset = [r for r in records if r[key] == value]
        metrics = compute_shared_metrics(subset)
        rows.append({key: value, "n_curves": len(subset), **metrics})
    return pd.DataFrame(rows)


def evaluate_shared_model(
    data_root: PathLike,
    model_name: str,
    *,
    outputs_dir: PathLike = "outputs",
    checkpoint_name: str = "best_model.pt",
    split: str = "test",
    device: str = "auto",
    seed: Optional[int] = None,
    ratios: Optional[Tuple[float, float, float]] = None,
    save: bool = True,
    mc_samples: int = 30,
) -> Dict[str, float]:
    """Evaluate a shared model and write overall + grouped metric files.

    For Bayesian shared models (name contains ``bayesian``) predictions use
    MC-Dropout: standard metrics are computed on the MC *mean* and an
    ``uncertainty_summary.json`` with std statistics is also written.
    """
    bayesian = "bayesian" in model_name.lower()
    if bayesian:
        records = predict_shared_curves_uncertainty(
            data_root, model_name,
            outputs_dir=outputs_dir, checkpoint_name=checkpoint_name, split=split,
            device=device, seed=seed, ratios=ratios, mc_samples=mc_samples,
        )
    else:
        records = predict_shared_curves(
            data_root, model_name,
            outputs_dir=outputs_dir, checkpoint_name=checkpoint_name, split=split,
            device=device, seed=seed, ratios=ratios,
        )
    if not records:
        raise RuntimeError(f"No '{split}' curves to evaluate for shared/{model_name}.")

    overall = compute_shared_metrics(records)
    case_ids = sorted({r["case_id"] for r in records})
    metrics_record = {
        "model_name": model_name,
        "split": split,
        "n_curves": len(records),
        "n_cases": len(case_ids),
        "cases": case_ids,
        **overall,
    }

    if save:
        mdir = ensure_dir(shared_metrics_dir(outputs_dir, model_name))
        save_json(metrics_record, mdir / "metrics.json")
        _grouped_metrics(records, "case_id").to_csv(mdir / "metrics_by_case.csv", index=False)
        _grouped_metrics(records, "ambient_temp_C").to_csv(mdir / "metrics_by_temp.csv", index=False)
        _grouped_metrics(records, "c_rate").to_csv(mdir / "metrics_by_c_rate.csv", index=False)
        print(f"[shared/{model_name}] metrics -> {mdir}")

    if bayesian:
        summary = {
            "model_name": model_name,
            "split": split,
            "mc_samples": int(mc_samples),
            "n_curves": len(records),
            **_shared_uncertainty_summary(records),
        }
        if save:
            mdir = ensure_dir(shared_metrics_dir(outputs_dir, model_name))
            save_json(summary, mdir / "uncertainty_summary.json")
            print(f"[shared/{model_name}] uncertainty -> {mdir / 'uncertainty_summary.json'}")
        print(
            f"[shared/{model_name}] MC-Dropout (n={mc_samples}) "
            f"mean_std_V={summary['mean_std_V']:.4f} mean_std_T={summary['mean_std_T']:.4f}"
        )
        metrics_record["uncertainty"] = summary

    print(
        f"[shared/{model_name}] ({split}) over {len(records)} curves / {len(case_ids)} cases | "
        f"RMSE_V={overall['RMSE_V']:.4f} R2_V={overall['R2_V']:.4f} | "
        f"RMSE_T={overall['RMSE_T']:.4f} R2_T={overall['R2_T']:.4f} | "
        f"T_peak_MAE={overall['temperature_peak_mae']:.4f}"
    )
    return metrics_record
