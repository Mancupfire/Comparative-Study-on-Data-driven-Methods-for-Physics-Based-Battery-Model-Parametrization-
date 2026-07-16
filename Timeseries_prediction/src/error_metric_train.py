"""Task B — error-metric surrogate: models, training, evaluation, persistence.

Two multi-output regressors predicting [rmse_voltage_mv, rmse_temperature_c]:

  a) ExtraTrees baseline  (sklearn.ensemble.ExtraTreesRegressor, native multi-output)
  b) MLP main model       (small torch MLP, AdamW + early stopping)

Both train on train-only-fitted scaled features/targets, are evaluated on
physical units (inverse-transformed), persist all artifacts in a Task-B
namespace, and run a checkpoint reload-equivalence verification.

No new external dependency is introduced (numpy / pandas / sklearn / torch /
joblib are already used by the repo).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor

from .error_metric_data import ErrorMetricDataset, leakage_check
from .utils import ensure_dir, resolve_device, save_json, set_seed

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Metrics (physical units)
# --------------------------------------------------------------------------- #
def _per_target_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: List[str]) -> Dict:
    out: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(names):
        t, p = y_true[:, j], y_pred[:, j]
        rmse = float(np.sqrt(np.mean((t - p) ** 2)))
        mae = float(np.mean(np.abs(t - p)))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        out[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    # Overall (averaged across targets).
    out["overall"] = {
        "RMSE": float(np.mean([out[n]["RMSE"] for n in names])),
        "MAE": float(np.mean([out[n]["MAE"] for n in names])),
        "R2": float(np.mean([out[n]["R2"] for n in names])),
    }
    return out


def _inverse_y(ds: ErrorMetricDataset, y_scaled: np.ndarray) -> np.ndarray:
    return ds.y_scaler.inverse_transform(y_scaled) if ds.y_scaler is not None else y_scaled


# --------------------------------------------------------------------------- #
# MLP main model
# --------------------------------------------------------------------------- #
class ErrorMetricMLP(nn.Module):
    """Multi-output MLP: n_features -> hidden* -> n_targets."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _mlp_predict(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(t).cpu().numpy()


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #
def _save_predictions(out_dir: Path, model_name: str, split: str,
                      sample_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                      target_names: List[str]) -> Path:
    pred_dir = ensure_dir(out_dir / "predictions" / model_name)
    df = pd.DataFrame({"sample_id": sample_ids})
    for j, name in enumerate(target_names):
        df[f"{name}_true"] = y_true[:, j]
        df[f"{name}_pred"] = y_pred[:, j]
    path = pred_dir / f"{split}_predictions.csv"
    df.to_csv(path, index=False)
    return path


def _save_scalers(out_dir: Path, ds: ErrorMetricDataset) -> None:
    sdir = ensure_dir(out_dir / "scalers")
    joblib.dump(ds.x_scaler, sdir / "x_scaler.joblib")
    if ds.y_scaler is not None:
        joblib.dump(ds.y_scaler, sdir / "y_scaler.joblib")


# --------------------------------------------------------------------------- #
# Trainers
# --------------------------------------------------------------------------- #
def train_extratrees(ds: ErrorMetricDataset, out_dir: Path, config: Dict) -> Dict:
    set_seed(config.get("seed", 42))
    model = ExtraTreesRegressor(
        n_estimators=config.get("n_estimators", 300),
        max_depth=config.get("max_depth", None),
        n_jobs=config.get("n_jobs", -1),
        random_state=config.get("seed", 42),
    )
    t0 = time.time()
    model.fit(ds.X_train, ds.Y_train)
    elapsed = time.time() - t0

    mdir = ensure_dir(out_dir / "models" / "extratrees")
    model_path = mdir / "model.joblib"
    joblib.dump(model, model_path)

    # Predictions in physical units.
    pred_test_scaled = model.predict(ds.X_test)
    y_test_phys = _inverse_y(ds, ds.Y_test)
    p_test_phys = _inverse_y(ds, pred_test_scaled)
    metrics = {s: _per_target_metrics(
        _inverse_y(ds, getattr(ds, f"Y_{s}")),
        _inverse_y(ds, model.predict(getattr(ds, f"X_{s}"))),
        ds.target_names) for s in ("train", "val", "test")}

    _save_predictions(out_dir, "extratrees", "test", ds.split_row_sample_ids["test"],
                      y_test_phys, p_test_phys, ds.target_names)

    # ----- reload-equivalence verification -----
    reloaded = joblib.load(model_path)
    reload_ok = bool(np.allclose(reloaded.predict(ds.X_test), pred_test_scaled))

    save_json({"model": "extratrees", "elapsed_s": elapsed,
               "params": {"n_estimators": config.get("n_estimators", 300)},
               "metrics": metrics, "reload_equivalence_ok": reload_ok},
              out_dir / "metrics" / "extratrees" / "metrics.json")
    return {"model": "extratrees", "metrics": metrics, "reload_equivalence_ok": reload_ok}


def train_mlp(ds: ErrorMetricDataset, out_dir: Path, config: Dict) -> Dict:
    set_seed(config.get("seed", 42))
    device = resolve_device(config.get("device", "auto"))
    arch = {"in_dim": ds.n_features, "out_dim": len(ds.target_names),
            "hidden_dim": config.get("hidden_dim", 128),
            "num_layers": config.get("num_layers", 3),
            "dropout": config.get("dropout", 0.1)}
    model = ErrorMetricMLP(**arch).to(device)

    Xtr = torch.as_tensor(ds.X_train, dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(ds.Y_train, dtype=torch.float32, device=device)
    Xva = torch.as_tensor(ds.X_val, dtype=torch.float32, device=device)
    Yva = torch.as_tensor(ds.Y_val, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3),
                            weight_decay=config.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=config.get("patience", 20) // 2)
    crit = nn.MSELoss()
    batch_size = config.get("batch_size", 256)
    epochs = config.get("epochs", 200)
    patience = config.get("patience", 20)

    n = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    bad = 0
    history = []
    g = torch.Generator(device="cpu").manual_seed(config.get("seed", 42))
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            opt.step()
            tot += loss.detach().item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xva), Yva).item()
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss})
        if val_loss < best_val - 1e-9:
            best_val, best_state, bad = val_loss, {k: v.detach().cpu().clone()
                                                   for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    elapsed = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    cdir = ensure_dir(out_dir / "models" / "mlp")
    ckpt_path = cdir / "best_model.pt"
    torch.save({"state_dict": model.state_dict(), "arch": arch,
                "model": "error_metric_mlp"}, ckpt_path)
    pd.DataFrame(history).to_csv(ensure_dir(out_dir / "metrics" / "mlp") / "history.csv", index=False)

    pred_test_scaled = _mlp_predict(model, ds.X_test, device)
    y_test_phys = _inverse_y(ds, ds.Y_test)
    p_test_phys = _inverse_y(ds, pred_test_scaled)
    metrics = {s: _per_target_metrics(
        _inverse_y(ds, getattr(ds, f"Y_{s}")),
        _inverse_y(ds, _mlp_predict(model, getattr(ds, f"X_{s}"), device)),
        ds.target_names) for s in ("train", "val", "test")}
    _save_predictions(out_dir, "mlp", "test", ds.split_row_sample_ids["test"],
                      y_test_phys, p_test_phys, ds.target_names)

    # ----- reload-equivalence verification -----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    reloaded = ErrorMetricMLP(**ckpt["arch"]).to(device)
    reloaded.load_state_dict(ckpt["state_dict"])
    reload_ok = bool(np.allclose(_mlp_predict(reloaded, ds.X_test, device),
                                 pred_test_scaled, atol=1e-5))

    save_json({"model": "mlp", "arch": arch, "elapsed_s": elapsed,
               "best_val_loss_scaled": best_val, "epochs_ran": len(history),
               "metrics": metrics, "reload_equivalence_ok": reload_ok},
              out_dir / "metrics" / "mlp" / "metrics.json")
    return {"model": "mlp", "metrics": metrics, "reload_equivalence_ok": reload_ok}


def run_task_b(ds: ErrorMetricDataset, out_dir: PathLike, config: Dict,
               models: List[str]) -> Dict:
    """Train requested models and write run-level manifest (join + leakage + config)."""
    out = ensure_dir(Path(out_dir))
    _save_scalers(out, ds)
    save_json(ds.split_sample_ids, out / "split_sample_ids.json")

    leak = leakage_check(ds)
    results = {}
    if "extratrees" in models:
        results["extratrees"] = train_extratrees(ds, out, config)
    if "mlp" in models:
        results["mlp"] = train_mlp(ds, out, config)

    manifest = {
        "task": "error_metric_surrogate",
        "dataset_name": config.get("dataset_name", "Data_Batch_2"),
        "data_dir": str(config.get("data_dir")),
        "feature_names": ds.feature_names,
        "n_features": ds.n_features,
        "target_names": ds.target_names,
        "continuous_feature_idx": ds.continuous_feature_idx,
        "categorical_feature_idx": ds.categorical_feature_idx,
        "split": {"train_ratio": config.get("train_ratio", 0.7),
                  "val_ratio": config.get("val_ratio", 0.15),
                  "test_ratio": config.get("test_ratio", 0.15),
                  "seed": config.get("seed", 42), "split_by": "sample_id"},
        "join_report": ds.join_report.as_dict(),
        "leakage_check": leak,
        "config": config,
        "results": results,
    }
    save_json(manifest, out / "run_manifest.json")
    save_json(config, out / "run_config.json")
    return manifest
