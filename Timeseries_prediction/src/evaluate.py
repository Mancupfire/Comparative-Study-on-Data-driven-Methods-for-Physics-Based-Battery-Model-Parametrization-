"""Evaluation: run a trained model on a split and compute physical metrics.

All metrics are computed on inverse-transformed (physical) predictions so the
reported errors are in volts / degrees Celsius.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from .metrics import compute_metrics
from .predict import metrics_dir, predict_case, predict_case_uncertainty
from .utils import save_json

PathLike = Union[str, Path]


def is_bayesian_model(model_name: str) -> bool:
    """A model uses MC-Dropout uncertainty when its name contains 'bayesian'."""
    return "bayesian" in model_name.lower()


def _uncertainty_summary(v_std: np.ndarray, t_std: np.ndarray) -> Dict[str, float]:
    """Aggregate per-point MC-Dropout std arrays into summary statistics.

    ``v_std`` / ``t_std`` are ``[N, t_last]`` physical-unit standard deviations.
    """
    return {
        "mean_std_V": float(np.mean(v_std)),
        "mean_std_T": float(np.mean(t_std)),
        "max_std_V": float(np.max(v_std)),
        "max_std_T": float(np.max(t_std)),
        "mean_std_V_end": float(np.mean(v_std[:, -1])),
        "mean_std_T_end": float(np.mean(t_std[:, -1])),
    }


def evaluate_case(
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
    save: bool = True,
    mc_samples: int = 30,
) -> Dict[str, float]:
    """Evaluate a trained model and (optionally) write ``metrics.json``.

    Returns the metric dict.  When ``save`` is True the metrics are persisted to
    ``outputs/metrics/<case_id>/<model_name>/metrics.json``.

    For Bayesian models (name contains ``bayesian``) predictions use MC-Dropout:
    the standard metrics are computed on the MC *mean*, and an
    ``uncertainty_summary.json`` with the std statistics is also written.
    """
    bayesian = is_bayesian_model(model_name)

    if bayesian:
        pred = predict_case_uncertainty(
            data_root, case_id, model_name,
            outputs_dir=outputs_dir, checkpoint_name=checkpoint_name, split=split,
            device=device, batch_size=batch_size, seed=seed, ratios=ratios,
            mc_samples=mc_samples,
        )
    else:
        pred = predict_case(
            data_root, case_id, model_name,
            outputs_dir=outputs_dir, checkpoint_name=checkpoint_name, split=split,
            device=device, batch_size=batch_size, seed=seed, ratios=ratios,
        )

    metrics = compute_metrics(
        pred["v_true"], pred["v_pred"], pred["t_true"], pred["t_pred"]
    )
    metrics_record = {
        "case_id": case_id,
        "model_name": model_name,
        "split": split,
        "n_samples": int(pred["v_true"].shape[0]),
        "t_last": int(pred["v_true"].shape[1]),
        **metrics,
    }

    if save:
        out_path = metrics_dir(outputs_dir, case_id, model_name) / "metrics.json"
        save_json(metrics_record, out_path)
        print(f"[{case_id}/{model_name}] metrics -> {out_path}")

    if bayesian:
        summary = {
            "case_id": case_id,
            "model_name": model_name,
            "split": split,
            "mc_samples": int(mc_samples),
            **_uncertainty_summary(pred["v_std"], pred["t_std"]),
        }
        if save:
            unc_path = metrics_dir(outputs_dir, case_id, model_name) / "uncertainty_summary.json"
            save_json(summary, unc_path)
            print(f"[{case_id}/{model_name}] uncertainty -> {unc_path}")
        print(
            f"[{case_id}/{model_name}] MC-Dropout (n={mc_samples}) "
            f"mean_std_V={summary['mean_std_V']:.4f} mean_std_T={summary['mean_std_T']:.4f}"
        )
        metrics_record["uncertainty"] = summary

    _print_metrics(case_id, model_name, split, metrics)
    return metrics_record


def _print_metrics(case_id: str, model_name: str, split: str, metrics: Dict[str, float]) -> None:
    print(f"[{case_id}/{model_name}] ({split}) "
          f"RMSE_V={metrics['RMSE_V']:.4f} R2_V={metrics['R2_V']:.4f} | "
          f"RMSE_T={metrics['RMSE_T']:.4f} R2_T={metrics['R2_T']:.4f} | "
          f"T_peak_MAE={metrics['temperature_peak_mae']:.4f}")
