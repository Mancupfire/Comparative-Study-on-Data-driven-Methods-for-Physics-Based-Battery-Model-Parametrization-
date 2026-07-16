"""Reusable training loop for the MLP / RNN / LSTM / BiLSTM models.

A single ``train_model`` entry point handles every architecture.  It fits and
persists the scalers, trains with AdamW + ReduceLROnPlateau, applies early
stopping on the validation loss and writes checkpoints, history and the run
configuration to the standard ``outputs/`` layout.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple, Union

import joblib
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import data as data_mod
from .models import ALL_MODELS, build_model, make_model_kwargs, split_prediction
from .predict import checkpoint_dir, metrics_dir, scaler_dir
from .utils import ensure_dir, resolve_device, save_json, set_seed

PathLike = Union[str, Path]


def _compute_losses(
    model_name: str,
    output: torch.Tensor,
    target: torch.Tensor,
    t_last: int,
    lambda_temp: float,
    criterion: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, loss_v, loss_t) with total = loss_v + lambda * loss_t."""
    pred_v, pred_t = split_prediction(model_name, output, t_last)
    true_v, true_t = split_prediction(model_name, target, t_last)
    loss_v = criterion(pred_v, true_v)
    loss_t = criterion(pred_t, true_t)
    total = loss_v + lambda_temp * loss_t
    return total, loss_v, loss_t


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    model_name: str,
    t_last: int,
    lambda_temp: float,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    """Run one epoch (train if ``optimizer`` given, else evaluate). Loss-weighted."""
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"total": 0.0, "v": 0.0, "t": 0.0}
    n_seen = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        bs = x.shape[0]
        with torch.set_grad_enabled(train_mode):
            out = model(x)
            total, loss_v, loss_t = _compute_losses(
                model_name, out, y, t_last, lambda_temp, criterion
            )
            if train_mode:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()
        totals["total"] += total.item() * bs
        totals["v"] += loss_v.item() * bs
        totals["t"] += loss_t.item() * bs
        n_seen += bs

    return {k: v / max(n_seen, 1) for k, v in totals.items()}


def _save_checkpoint(path: Path, model: nn.Module, meta: Dict) -> None:
    ensure_dir(path.parent)
    torch.save({**meta, "model_state_dict": model.state_dict()}, path)


def train_model(
    data_root: PathLike,
    case_id: str,
    model_name: str,
    *,
    outputs_dir: PathLike = "outputs",
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    lambda_temp: float = 1.0,
    patience: int = 30,
    seed: int = 42,
    device: str = "auto",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    norm: str = "layernorm",
    num_workers: int = 0,
) -> Dict:
    """Train one model on one case and persist all artefacts.

    Returns a dict with the best epoch, best validation loss and the paths of
    the artefacts written under ``outputs/``.
    """
    model_name = model_name.lower()
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {sorted(ALL_MODELS)}")

    set_seed(seed)
    dev = resolve_device(device)
    print(f"[{case_id}/{model_name}] device={dev}")

    # --- data -------------------------------------------------------------- #
    bundle = data_mod.build_datasets(
        model_name,
        data_root,
        case_id,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    case = bundle.case
    t_last = case.t_last
    print(
        f"[{case_id}/{model_name}] n_samples={case.n_samples} "
        f"n_parameters={case.n_parameters} t_last={t_last} "
        f"(train/val/test = {len(bundle.splits['train'])}/"
        f"{len(bundle.splits['val'])}/{len(bundle.splits['test'])})"
    )

    # --- persist scalers (fit on train only inside build_datasets) --------- #
    sdir = ensure_dir(scaler_dir(outputs_dir, case_id, model_name))
    joblib.dump(bundle.x_scaler, sdir / "x_scaler.joblib")
    joblib.dump(bundle.v_scaler, sdir / "v_scaler.joblib")
    joblib.dump(bundle.t_scaler, sdir / "t_scaler.joblib")

    # --- model / optim ----------------------------------------------------- #
    model_kwargs = make_model_kwargs(
        model_name, case.n_parameters, t_last, hidden_dim, num_layers, dropout, norm
    )
    model = build_model(model_name, model_kwargs).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
    )
    criterion = nn.MSELoss()

    pin = dev == "cuda"
    train_loader = DataLoader(
        bundle.train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    val_loader = DataLoader(
        bundle.val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    ckpt_dir = ensure_dir(checkpoint_dir(outputs_dir, case_id, model_name))
    mdir = ensure_dir(metrics_dir(outputs_dir, case_id, model_name))

    ckpt_meta = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "case_id": case_id,
        "t_last": t_last,
        "n_parameters": case.n_parameters,
        "param_names": case.param_names,
    }

    # --- training loop ----------------------------------------------------- #
    history = []
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(
            model, train_loader, model_name, t_last, lambda_temp, criterion, dev, optimizer
        )
        va = _run_epoch(
            model, val_loader, model_name, t_last, lambda_temp, criterion, dev, optimizer=None
        )
        scheduler.step(va["total"])
        cur_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["total"],
                "val_loss": va["total"],
                "train_loss_v": tr["v"],
                "train_loss_t": tr["t"],
                "val_loss_v": va["v"],
                "val_loss_t": va["t"],
                "lr": cur_lr,
            }
        )
        print(
            f"[{case_id}/{model_name}] epoch {epoch:4d} | "
            f"train {tr['total']:.5f} | val {va['total']:.5f} | "
            f"loss_v {va['v']:.5f} | loss_t {va['t']:.5f} | lr {cur_lr:.2e}"
        )

        # --- checkpoint best + early stopping ---------------------------- #
        if va["total"] < best_val - 1e-8:
            best_val = va["total"]
            best_epoch = epoch
            epochs_no_improve = 0
            _save_checkpoint(
                ckpt_dir / "best_model.pt",
                model,
                {**ckpt_meta, "epoch": epoch, "val_loss": best_val},
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"[{case_id}/{model_name}] early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs; best epoch {best_epoch})"
                )
                break

    elapsed = time.time() - start

    # --- persist final artefacts ------------------------------------------ #
    _save_checkpoint(
        ckpt_dir / "final_model.pt",
        model,
        {**ckpt_meta, "epoch": history[-1]["epoch"], "val_loss": history[-1]["val_loss"]},
    )
    pd.DataFrame(history).to_csv(mdir / "history.csv", index=False)

    run_config = {
        "data_root": str(data_root),
        "outputs_dir": str(outputs_dir),
        "case_id": case_id,
        "model_name": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "lambda_temp": lambda_temp,
        "patience": patience,
        "seed": seed,
        "device": dev,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "norm": norm,
        "model_kwargs": model_kwargs,
        "n_samples": case.n_samples,
        "n_parameters": case.n_parameters,
        "t_last": t_last,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_s": elapsed,
    }
    save_json(run_config, mdir / "run_config.json")

    print(
        f"[{case_id}/{model_name}] done in {elapsed:.1f}s | "
        f"best_val={best_val:.5f} @ epoch {best_epoch}"
    )
    return {
        "case_id": case_id,
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_checkpoint": str(ckpt_dir / "best_model.pt"),
        "final_checkpoint": str(ckpt_dir / "final_model.pt"),
        "history_csv": str(mdir / "history.csv"),
        "run_config": str(mdir / "run_config.json"),
        "elapsed_s": elapsed,
    }
