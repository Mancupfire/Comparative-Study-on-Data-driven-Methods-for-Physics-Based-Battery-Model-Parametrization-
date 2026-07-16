"""Masked training + evaluation for one filtered case / model / seed.

Loss and every reported metric use the valid-time mask, so a held /
extrapolated tail never contributes.  Artifacts are written under an isolated
run dir::

    <run_dir>/checkpoints/<case>/<model>/best_model.pt
    <run_dir>/scalers/<case>/<model>/*.joblib
    <run_dir>/metrics/<case>/<model>/metrics.json
    <run_dir>/predictions/<case>/<model>/test_predictions.csv

Each (case, model, seed) is independent and resumable (presence of metrics.json
marks completion).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import normalize_time
from src.utils import ensure_dir, resolve_device, save_json, set_seed

from . import models as M
from .data import build_filtered_case, FilteredCaseBundle
from .masking import compute_masked_metrics, masked_mse_torch

PathLike = Union[str, Path]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _ckpt_dir(run_dir, case, model): return ensure_dir(Path(run_dir) / "checkpoints" / case / model)
def _scaler_dir(run_dir, case, model): return ensure_dir(Path(run_dir) / "scalers" / case / model)
def _metrics_dir(run_dir, case, model): return ensure_dir(Path(run_dir) / "metrics" / case / model)
def _pred_dir(run_dir, case, model): return ensure_dir(Path(run_dir) / "predictions" / case / model)


def is_complete(run_dir: PathLike, case: str, model: str) -> bool:
    return (Path(run_dir) / "metrics" / case / model / "metrics.json").is_file()


# --------------------------------------------------------------------------- #
# Tensor assembly
# --------------------------------------------------------------------------- #
def _make_inputs(bundle: FilteredCaseBundle, split: str, model_name: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_model, V_scaled, T_scaled, mask) for a split.

    ``X_model`` is ``[n, P]`` for point models or ``[n, T, P+1]`` for sequence
    models (params broadcast over time + a normalized-time channel).
    """
    Xp = bundle.X[split]                         # [n, P] scaled params
    Vs = bundle.V_scaled[split]                  # [n, T]
    Ts = bundle.T_scaled[split]                  # [n, T]
    mask = bundle.mask[split]                    # [n, T]
    if not M.is_sequence_model(model_name):
        return Xp, Vs, Ts, mask
    n, p = Xp.shape
    t_last = bundle.t_last
    time_norm = normalize_time(bundle.time_s)    # [T]
    Xseq = np.empty((n, t_last, p + 1), dtype=np.float64)
    Xseq[:, :, :p] = Xp[:, None, :]
    Xseq[:, :, p] = time_norm[None, :]
    return Xseq, Vs, Ts, mask


def _split_pred(model_name, out, t_last):
    return M.split_prediction(model_name, out, t_last)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _epoch(model, Xt, Vt, Tt, Mt, model_name, t_last, lambda_temp, device,
           batch_size, optimizer=None, generator=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    n = Xt.shape[0]
    if train_mode and generator is not None:
        order = torch.randperm(n, generator=generator).to(device)
    else:
        order = torch.arange(n, device=device)
    tot = tv = tt = 0.0
    seen = 0
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        x = Xt[idx]; v = Vt[idx]; t = Tt[idx]; m = Mt[idx]
        with torch.set_grad_enabled(train_mode):
            out = model(x)
            pv, pt = _split_pred(model_name, out, t_last)
            loss_v = masked_mse_torch(pv, v, m)
            loss_t = masked_mse_torch(pt, t, m)
            loss = loss_v + lambda_temp * loss_t
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        bs = len(idx)
        tot += loss.item() * bs; tv += loss_v.item() * bs; tt += loss_t.item() * bs
        seen += bs
    d = max(seen, 1)
    return {"total": tot / d, "v": tv / d, "t": tt / d}


def train_and_eval(
    case_id: str,
    model_name: str,
    *,
    run_dir: PathLike,
    seed: int = 42,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    ann_hidden_dim: int = 128,
    lambda_temp: float = 1.0,
    patience: int = 30,
    device: str = "auto",
    mc_samples: int = 30,
) -> Dict:
    model_name = model_name.lower()
    if model_name not in M.FINAL_TS_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {M.FINAL_TS_MODELS}")

    set_seed(seed)
    dev = resolve_device(device)
    bundle = build_filtered_case(case_id, seed=seed)
    t_last = bundle.t_last
    print(f"[{case_id}/{model_name}] seed={seed} device={dev} "
          f"train/val/test={bundle.n('train')}/{bundle.n('val')}/{bundle.n('test')}")

    # Persist scalers.
    sdir = _scaler_dir(run_dir, case_id, model_name)
    joblib.dump(bundle.x_scaler, sdir / "x_scaler.joblib")
    joblib.dump(bundle.v_scaler, sdir / "v_scaler.joblib")
    joblib.dump(bundle.t_scaler, sdir / "t_scaler.joblib")

    # Build tensors per split.
    tensors = {}
    for s in ("train", "val", "test"):
        Xn, Vn, Tn, Mn = _make_inputs(bundle, s, model_name)
        tensors[s] = (
            torch.as_tensor(Xn, dtype=torch.float32, device=dev),
            torch.as_tensor(Vn, dtype=torch.float32, device=dev),
            torch.as_tensor(Tn, dtype=torch.float32, device=dev),
            torch.as_tensor(Mn.astype(np.float32), device=dev),
        )

    model_kwargs = M.make_model_kwargs(
        model_name, bundle.n_parameters, t_last, hidden_dim, num_layers,
        dropout, ann_hidden_dim=ann_hidden_dim,
    )
    model = M.build_model(model_name, model_kwargs).to(dev)
    n_params = M.count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3))
    gen = torch.Generator(device="cpu").manual_seed(seed)

    cdir = _ckpt_dir(run_dir, case_id, model_name)
    meta = {"model_name": model_name, "model_kwargs": model_kwargs,
            "case_id": case_id, "t_last": t_last, "seed": seed,
            "n_parameters_model": n_params,
            "v_scaler": asdict(bundle.v_scaler), "t_scaler": asdict(bundle.t_scaler)}

    history = []
    best_val = float("inf"); best_epoch = -1; bad = 0
    start = time.time()
    Xtr, Vtr, Ttr, Mtr = tensors["train"]
    Xva, Vva, Tva, Mva = tensors["val"]
    for epoch in range(1, epochs + 1):
        tr = _epoch(model, Xtr, Vtr, Ttr, Mtr, model_name, t_last, lambda_temp,
                    dev, batch_size, optimizer, gen)
        va = _epoch(model, Xva, Vva, Tva, Mva, model_name, t_last, lambda_temp,
                    dev, batch_size, optimizer=None)
        scheduler.step(va["total"])
        history.append({"epoch": epoch, "train_loss": tr["total"],
                        "val_loss": va["total"], "lr": optimizer.param_groups[0]["lr"]})
        if va["total"] < best_val - 1e-9:
            best_val, best_epoch, bad = va["total"], epoch, 0
            torch.save({**meta, "epoch": epoch, "val_loss": best_val,
                        "model_state_dict": model.state_dict()},
                       cdir / "best_model.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    elapsed = time.time() - start
    pd.DataFrame(history).to_csv(_metrics_dir(run_dir, case_id, model_name) / "history.csv",
                                 index=False)

    # ---- evaluate the best checkpoint (reload to prove round-trip) ----
    ckpt = torch.load(cdir / "best_model.pt", map_location=dev, weights_only=False)
    eval_model = M.build_model(model_name, ckpt["model_kwargs"]).to(dev)
    eval_model.load_state_dict(ckpt["model_state_dict"])
    eval_model.eval()

    metrics_record = _evaluate(
        eval_model, model_name, bundle, tensors, t_last, dev, mc_samples,
        run_dir=run_dir, case_id=case_id,
    )
    metrics_record.update({
        "case_id": case_id, "model_name": model_name, "seed": seed,
        "display": M.DISPLAY_NAMES[model_name],
        "param_count": n_params, "best_epoch": best_epoch,
        "best_val_loss_masked": best_val, "epochs_ran": len(history),
        "elapsed_s": elapsed, "device": dev,
        "n_train": bundle.n("train"), "n_val": bundle.n("val"),
        "n_test": bundle.n("test"),
        "model_kwargs": model_kwargs,
    })
    save_json(metrics_record, _metrics_dir(run_dir, case_id, model_name) / "metrics.json")

    # Save run config for reproducibility.
    save_json({
        "case_id": case_id, "model_name": model_name, "seed": seed,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "hidden_dim": hidden_dim,
        "num_layers": num_layers, "dropout": dropout,
        "ann_hidden_dim": ann_hidden_dim, "lambda_temp": lambda_temp,
        "patience": patience, "protocol": "filtered_grouped_masked",
        "ratios": [0.7, 0.15, 0.15],
    }, _metrics_dir(run_dir, case_id, model_name) / "run_config.json")

    print(f"[{case_id}/{model_name}] done {elapsed:.1f}s | params={n_params} | "
          f"RMSE_V={metrics_record['test']['RMSE_V']:.4f} "
          f"R2_V={metrics_record['test']['R2_V']:.4f} | "
          f"RMSE_T={metrics_record['test']['RMSE_T']:.4f} "
          f"R2_T={metrics_record['test']['R2_T']:.4f}")
    return metrics_record


@torch.no_grad()
def _predict_scaled(model, X, model_name, t_last, device, batch_size=512,
                    mc_samples=0):
    """Return scaled (V, T) predictions; if mc_samples>0 also (V_std, T_std)."""
    from src.predict import enable_mc_dropout
    n = X.shape[0]
    if mc_samples and mc_samples > 0:
        enable_mc_dropout(model)
        passes_v, passes_t = [], []
        for _ in range(mc_samples):
            vc, tc = [], []
            for i in range(0, n, batch_size):
                out = model(X[i:i + batch_size])
                pv, pt = _split_pred(model_name, out, t_last)
                vc.append(pv.cpu().numpy()); tc.append(pt.cpu().numpy())
            passes_v.append(np.concatenate(vc, 0)); passes_t.append(np.concatenate(tc, 0))
        vs = np.stack(passes_v, 0); ts = np.stack(passes_t, 0)
        return vs.mean(0), ts.mean(0), vs.std(0), ts.std(0)
    model.eval()
    vc, tc = [], []
    for i in range(0, n, batch_size):
        out = model(X[i:i + batch_size])
        pv, pt = _split_pred(model_name, out, t_last)
        vc.append(pv.cpu().numpy()); tc.append(pt.cpu().numpy())
    return np.concatenate(vc, 0), np.concatenate(tc, 0), None, None


def _evaluate(model, model_name, bundle, tensors, t_last, device, mc_samples,
              run_dir, case_id) -> Dict:
    bayesian = "bayesian" in model_name.lower()
    out = {}
    for split in ("val", "test"):
        X = tensors[split][0]
        mc = mc_samples if bayesian else 0
        v_s, t_s, v_std, t_std = _predict_scaled(model, X, model_name, t_last, device,
                                                 mc_samples=mc)
        v_pred = bundle.v_scaler.inverse_transform(v_s)
        t_pred = bundle.t_scaler.inverse_transform(t_s)
        if not (np.all(np.isfinite(v_pred)) and np.all(np.isfinite(t_pred))):
            raise FloatingPointError(f"{model_name}: non-finite predictions on {split}")
        mask = bundle.mask[split]
        mt = compute_masked_metrics(bundle.V_phys[split], v_pred,
                                    bundle.T_phys[split], t_pred, mask)
        if bayesian and v_std is not None:
            # Physical-unit std (the scalar GlobalScaler scale_ rescales std).
            v_std_phys = np.asarray(v_std) * bundle.v_scaler.scale_
            t_std_phys = np.asarray(t_std) * bundle.t_scaler.scale_
            mt["calibration"] = _calibration(
                bundle.V_phys[split], v_pred, v_std_phys,
                bundle.T_phys[split], t_pred, t_std_phys, mask)
        out[split] = mt
        if split == "test":
            _write_test_predictions(run_dir, case_id, model_name, bundle,
                                    v_pred, t_pred)
    return out


def _calibration(v_true, v_pred, v_std, t_true, t_pred, t_std, mask) -> Dict:
    """MC-Dropout calibration: 95% interval coverage on valid points."""
    m = np.asarray(mask, dtype=bool)

    def _cov(true, pred, std):
        z = 1.959963985
        lo, hi = pred - z * std, pred + z * std
        inside = (true >= lo) & (true <= hi)
        return float(inside[m].mean()) if m.any() else 0.0

    return {
        "coverage95_V": _cov(v_true, v_pred, v_std),
        "coverage95_T": _cov(t_true, t_pred, t_std),
        "mean_std_V": float(v_std[m].mean()) if m.any() else 0.0,
        "mean_std_T": float(t_std[m].mean()) if m.any() else 0.0,
    }


def _write_test_predictions(run_dir, case_id, model_name, bundle, v_pred, t_pred):
    """Per-sample masked test summary (curve end + peak temperature)."""
    from .masking import last_valid_index
    mask = bundle.mask["test"]
    idx = last_valid_index(mask)
    rows = np.arange(v_pred.shape[0])
    very_neg = -np.inf
    df = pd.DataFrame({
        "sample_id": bundle.sample_ids["test"],
        "v_end_true": bundle.V_phys["test"][rows, idx],
        "v_end_pred": v_pred[rows, idx],
        "t_end_true": bundle.T_phys["test"][rows, idx],
        "t_end_pred": t_pred[rows, idx],
        "t_peak_true": np.where(mask, bundle.T_phys["test"], very_neg).max(axis=1),
        "t_peak_pred": np.where(mask, t_pred, very_neg).max(axis=1),
        "n_valid": mask.sum(axis=1),
    })
    path = _pred_dir(run_dir, case_id, model_name) / "test_predictions.csv"
    df.to_csv(path, index=False)
    return path
