"""Training / evaluation for the unified error-metric benchmark.

One entry point :func:`train_one` dispatches a (family, seed) job to a neural,
deep-ensemble, or ExtraTrees trainer.  All artifacts are written under::

    <run_dir>/checkpoints/<family>/seed<seed>/
    <run_dir>/histories/<family>/seed<seed>/
    <run_dir>/metrics/<family>/seed<seed>/
    <run_dir>/predictions/<family>/seed<seed>/

so each (family, seed) combination is independent and resumable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesRegressor

from src.utils import ensure_dir, resolve_device, save_json, set_seed

from . import models as M
from .data import BenchmarkDataset, inverse_y
from .metrics import evaluate


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def _paths(run_dir: Path, family: str, seed: int) -> Dict[str, Path]:
    tag = f"seed{seed}"
    return {
        "ckpt": ensure_dir(run_dir / "checkpoints" / family / tag),
        "hist": ensure_dir(run_dir / "histories" / family / tag),
        "metrics": ensure_dir(run_dir / "metrics" / family / tag),
        "pred": ensure_dir(run_dir / "predictions" / family / tag),
    }


def is_complete(run_dir: Path, family: str, seed: int) -> bool:
    p = run_dir / "metrics" / family / f"seed{seed}" / "metrics.json"
    return p.is_file()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _finite(*arrs: np.ndarray) -> bool:
    return all(np.all(np.isfinite(a)) for a in arrs)


def _save_predictions(pred_dir: Path, ds: BenchmarkDataset,
                      y_true_phys: np.ndarray, y_pred_phys: np.ndarray) -> Tuple[Path, int]:
    test_mf = ds.split_frame("test")
    df = test_mf[["sample_id", "experiment_id", "sequence_id"]].copy()
    for j, name in enumerate(ds.target_names):
        df[f"{name}_true"] = y_true_phys[:, j]
        df[f"{name}_pred"] = y_pred_phys[:, j]
    path = pred_dir / "test_predictions.csv"
    df.to_csv(path, index=False)
    return path, len(df)


def _full_result(family: str, ds: BenchmarkDataset, preds: Dict[str, np.ndarray],
                 extra: Dict) -> Dict:
    """Compute physical-unit metrics for every split from scaled predictions."""
    metrics = {}
    for split in ("train", "val", "test"):
        yt = inverse_y(ds, getattr(ds, f"Y_{split}"))
        yp = inverse_y(ds, preds[split])
        metrics[split] = evaluate(yt, yp, ds.target_names)
    res = {"model": family, "metrics": metrics, "target_names": ds.target_names}
    res.update(extra)
    return res


# --------------------------------------------------------------------------- #
# Neural training
# --------------------------------------------------------------------------- #
def _predict_torch(model: nn.Module, X: np.ndarray, device: str,
                   batch_size: int = 4096) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            t = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(t).cpu().numpy())
    return np.concatenate(outs, axis=0)


def _train_torch_model(model: nn.Module, ds: BenchmarkDataset, cfg: Dict,
                       seed: int, device: str) -> Tuple[nn.Module, List[Dict], Dict]:
    set_seed(seed)
    model = model.to(device)
    Xtr = torch.as_tensor(ds.X_train, dtype=torch.float32, device=device)
    Ytr = torch.as_tensor(ds.Y_train, dtype=torch.float32, device=device)
    Xva = torch.as_tensor(ds.X_val, dtype=torch.float32, device=device)
    Yva = torch.as_tensor(ds.Y_val, dtype=torch.float32, device=device)

    opt_name = str(cfg.get("optimizer", "adamw")).lower()
    OptCls = torch.optim.AdamW if opt_name == "adamw" else torch.optim.Adam
    opt = OptCls(model.parameters(), lr=float(cfg.get("lr", 1e-3)),
                 weight_decay=float(cfg.get("weight_decay", 1e-4)))
    patience = int(cfg.get("patience", 20))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(1, patience // 2))
    crit = nn.MSELoss()
    clip = float(cfg.get("grad_clip", 1.0))
    batch_size = int(cfg.get("batch_size", 256))
    epochs = int(cfg.get("epochs", 200))

    n = Xtr.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    best_val, best_state, bad = float("inf"), None, 0
    best_epoch = -1
    history: List[Dict] = []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = crit(model(Xtr[idx]), Ytr[idx])
            loss.backward()
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            tot += loss.detach().item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_loss = crit(model(Xva), Yva).item()
            # RMSE in standardized target space (matches loss scale, for plots).
            tr_rmse = float(np.sqrt(tot / n))
            va_rmse = float(np.sqrt(val_loss))
        sched.step(val_loss)
        history.append({
            "epoch": epoch,
            "train_loss": tot / n, "val_loss": val_loss,
            "train_rmse": tr_rmse, "val_rmse": va_rmse,
            "lr": opt.param_groups[0]["lr"],
        })
        if val_loss < best_val - 1e-9:
            best_val, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, {"best_val_loss_scaled": best_val, "best_epoch": best_epoch,
                            "epochs_ran": len(history)}


def train_neural(family: str, ds: BenchmarkDataset, run_dir: Path, cfg: Dict,
                 seed: int, device: str) -> Dict:
    paths = _paths(run_dir, family, seed)
    arch = M.default_arch(family, ds.n_features, len(ds.target_names), cfg)
    model = M.build_torch_model(family, arch)
    n_params = M.count_parameters(model)

    model, history, info = _train_torch_model(model, ds, cfg, seed, device)

    preds = {s: _predict_torch(model, getattr(ds, f"X_{s}"), device)
             for s in ("train", "val", "test")}
    # Inference time on the test split (median of 3 timed runs).
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _predict_torch(model, ds.X_test, device)
        times.append(time.perf_counter() - t0)
    infer_s = float(np.median(times))

    if not _finite(preds["test"]):
        raise FloatingPointError(f"{family} seed{seed}: non-finite predictions")

    torch.save({"family": family, "arch": arch, "state_dict": model.state_dict(),
                "best_epoch": info["best_epoch"], "seed": seed},
               paths["ckpt"] / "best_model.pt")
    pd.DataFrame(history).to_csv(paths["hist"] / "history.csv", index=False)

    y_test_phys = inverse_y(ds, ds.Y_test)
    p_test_phys = inverse_y(ds, preds["test"])
    _, n_rows = _save_predictions(paths["pred"], ds, y_test_phys, p_test_phys)

    res = _full_result(family, ds, preds, {
        "arch": arch, "param_count": n_params,
        "inference_time_s": infer_s,
        "inference_ms_per_sample": 1e3 * infer_s / max(1, len(ds.X_test)),
        "best_val_loss_scaled": info["best_val_loss_scaled"],
        "best_epoch": info["best_epoch"], "epochs_ran": info["epochs_ran"],
        "n_test_rows": n_rows, "seed": seed, "device": device,
    })
    save_json(res, paths["metrics"] / "metrics.json")
    return res


# --------------------------------------------------------------------------- #
# Deep ensemble
# --------------------------------------------------------------------------- #
def train_deep_ensemble(ds: BenchmarkDataset, run_dir: Path, cfg: Dict,
                        seed: int, device: str) -> Dict:
    family = "deep_ensemble_mlp"
    n_members = int(cfg.get("ensemble_size", 5))
    paths = _paths(run_dir, family, seed)
    arch = M.default_arch("mlp", ds.n_features, len(ds.target_names), cfg)

    member_test, member_train, member_val = [], [], []
    histories = []
    total_params = 0
    t_infer0 = time.perf_counter()
    for m in range(n_members):
        member_seed = seed * 100 + m
        model = M.build_torch_model("mlp", arch)
        total_params += M.count_parameters(model)
        model, hist, info = _train_torch_model(model, ds, cfg, member_seed, device)
        torch.save({"family": family, "arch": arch, "state_dict": model.state_dict(),
                    "member": m, "member_seed": member_seed, "seed": seed,
                    "best_epoch": info["best_epoch"]},
                   paths["ckpt"] / f"member_{m}.pt")
        for h in hist:
            h["member"] = m
        histories.extend(hist)
        member_train.append(_predict_torch(model, ds.X_train, device))
        member_val.append(_predict_torch(model, ds.X_val, device))
        member_test.append(_predict_torch(model, ds.X_test, device))
    infer_s = time.perf_counter() - t_infer0

    stack_test = np.stack(member_test, axis=0)            # [M, N, 2]
    preds = {
        "train": np.mean(np.stack(member_train, 0), 0),
        "val": np.mean(np.stack(member_val, 0), 0),
        "test": stack_test.mean(0),
    }
    ens_std = stack_test.std(0)                            # epistemic std (scaled)
    if not _finite(preds["test"]):
        raise FloatingPointError("deep_ensemble_mlp: non-finite predictions")

    pd.DataFrame(histories).to_csv(paths["hist"] / "history.csv", index=False)
    np.savez(paths["pred"] / "member_predictions.npz",
             member_test_scaled=stack_test,
             ensemble_mean_scaled=preds["test"], ensemble_std_scaled=ens_std)

    y_test_phys = inverse_y(ds, ds.Y_test)
    p_test_phys = inverse_y(ds, preds["test"])
    _, n_rows = _save_predictions(paths["pred"], ds, y_test_phys, p_test_phys)

    res = _full_result(family, ds, preds, {
        "arch": arch, "param_count": total_params, "ensemble_size": n_members,
        "inference_time_s": float(infer_s),
        "inference_ms_per_sample": 1e3 * infer_s / max(1, len(ds.X_test)),
        "n_test_rows": n_rows, "seed": seed, "device": device,
    })
    save_json(res, paths["metrics"] / "metrics.json")
    return res


# --------------------------------------------------------------------------- #
# ExtraTrees
# --------------------------------------------------------------------------- #
def train_extratrees(ds: BenchmarkDataset, run_dir: Path, cfg: Dict,
                     seed: int, device: str) -> Dict:
    family = "extratrees"
    paths = _paths(run_dir, family, seed)
    set_seed(seed)
    model = ExtraTreesRegressor(
        n_estimators=int(cfg.get("n_estimators", 300)),
        max_depth=cfg.get("max_depth", None),
        n_jobs=int(cfg.get("n_jobs", -1)),
        random_state=seed,
    )
    model.fit(ds.X_train, ds.Y_train)
    joblib.dump(model, paths["ckpt"] / "model.joblib")

    preds = {s: model.predict(getattr(ds, f"X_{s}")) for s in ("train", "val", "test")}
    t0 = time.perf_counter()
    model.predict(ds.X_test)
    infer_s = time.perf_counter() - t0
    if not _finite(preds["test"]):
        raise FloatingPointError("extratrees: non-finite predictions")

    # Feature importance.
    imp = pd.DataFrame({"feature": ds.feature_names,
                        "importance": model.feature_importances_})
    imp.sort_values("importance", ascending=False).to_csv(
        paths["metrics"] / "feature_importance.csv", index=False)

    # Learning curve vs training-set size (NOT epochs).
    lc = _extratrees_learning_curve(ds, cfg, seed)
    lc.to_csv(paths["hist"] / "learning_curve.csv", index=False)

    y_test_phys = inverse_y(ds, ds.Y_test)
    p_test_phys = inverse_y(ds, preds["test"])
    _, n_rows = _save_predictions(paths["pred"], ds, y_test_phys, p_test_phys)

    res = _full_result(family, ds, preds, {
        "param_count": int(_extratrees_node_count(model)),
        "n_estimators": int(cfg.get("n_estimators", 300)),
        "inference_time_s": float(infer_s),
        "inference_ms_per_sample": 1e3 * infer_s / max(1, len(ds.X_test)),
        "n_test_rows": n_rows, "seed": seed, "device": "cpu",
    })
    save_json(res, paths["metrics"] / "metrics.json")
    return res


def _extratrees_node_count(model: ExtraTreesRegressor) -> int:
    return int(sum(est.tree_.node_count for est in model.estimators_))


def _extratrees_learning_curve(ds: BenchmarkDataset, cfg: Dict, seed: int) -> pd.DataFrame:
    fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    rng = np.random.default_rng(seed)
    n = len(ds.X_train)
    order = rng.permutation(n)
    rows = []
    yt_test = inverse_y(ds, ds.Y_test)
    for f in fracs:
        k = max(2, int(round(f * n)))
        idx = order[:k]
        mdl = ExtraTreesRegressor(
            n_estimators=int(cfg.get("n_estimators", 300)),
            max_depth=cfg.get("max_depth", None),
            n_jobs=int(cfg.get("n_jobs", -1)), random_state=seed)
        mdl.fit(ds.X_train[idx], ds.Y_train[idx])
        yp_test = inverse_y(ds, mdl.predict(ds.X_test))
        yp_tr = inverse_y(ds, mdl.predict(ds.X_train[idx]))
        yt_tr = inverse_y(ds, ds.Y_train[idx])
        rows.append({
            "train_fraction": f, "n_train": k,
            "test_rmse_voltage_mv": float(np.sqrt(np.mean((yt_test[:, 0] - yp_test[:, 0]) ** 2))),
            "test_rmse_temperature_c": float(np.sqrt(np.mean((yt_test[:, 1] - yp_test[:, 1]) ** 2))),
            "train_rmse_voltage_mv": float(np.sqrt(np.mean((yt_tr[:, 0] - yp_tr[:, 0]) ** 2))),
            "train_rmse_temperature_c": float(np.sqrt(np.mean((yt_tr[:, 1] - yp_tr[:, 1]) ** 2))),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def train_one(family: str, ds: BenchmarkDataset, run_dir: Path, cfg: Dict,
              seed: int, device: str) -> Dict:
    if family == "extratrees":
        return train_extratrees(ds, run_dir, cfg, seed, device)
    if family == "deep_ensemble_mlp":
        return train_deep_ensemble(ds, run_dir, cfg, seed, device)
    if family in M._TORCH_BUILDERS:
        return train_neural(family, ds, run_dir, cfg, seed, device)
    raise ValueError(f"Unknown family '{family}'")
