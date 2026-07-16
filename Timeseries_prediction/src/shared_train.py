"""Reusable training loop for the shared (all-cases) models.

A single ``train_shared_model`` entry point handles both modes:

* **point mode** (``shared_mlp``): batches are ``(x, y)`` with ``x [B, D]`` and
  ``y [B, 2]``.
* **sequence mode** (``shared_rnn`` / ``shared_lstm`` / ``shared_bilstm``):
  batches are ``(x, y, mask, lengths)``; the loss is masked so padded timesteps
  do not contribute.

Loss is MSE on the *normalized* targets, split per channel::

    total = loss_v + lambda_temp * loss_t

Training uses AdamW + ReduceLROnPlateau with early stopping, and writes
checkpoints / history / run-config to ``outputs/.../shared/<model_name>/``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn

from .shared_data import (
    ALL_SHARED_MODELS,
    SEQUENCE_MODELS,
    create_shared_dataloaders,
    shared_checkpoint_dir,
    shared_metrics_dir,
)
from .shared_models import build_shared_model, make_shared_model_kwargs
from .utils import ensure_dir, resolve_device, save_json, set_seed

PathLike = Union[str, Path]


def _point_losses(
    out: torch.Tensor, target: torch.Tensor, lambda_temp: float, criterion: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MSE losses for the point-wise mode: ``out``/``target`` are ``[B, 2]``."""
    loss_v = criterion(out[:, 0], target[:, 0])
    loss_t = criterion(out[:, 1], target[:, 1])
    return loss_v + lambda_temp * loss_t, loss_v, loss_t


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error computed only over valid (mask=True) positions."""
    diff2 = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1)
    return diff2.sum() / denom


def _sequence_losses(
    out: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, lambda_temp: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Masked MSE for sequence mode: ``out``/``target`` are ``[B, T, 2]``."""
    loss_v = _masked_mse(out[..., 0], target[..., 0], mask)
    loss_t = _masked_mse(out[..., 1], target[..., 1], mask)
    return loss_v + lambda_temp * loss_t, loss_v, loss_t


def _run_epoch(
    model: nn.Module,
    loader,
    is_sequence: bool,
    lambda_temp: float,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer = None,
) -> Dict[str, float]:
    """Run one epoch (train if ``optimizer`` given, else evaluate)."""
    train_mode = optimizer is not None
    model.train(train_mode)
    totals = {"total": 0.0, "v": 0.0, "t": 0.0}
    n_seen = 0

    for batch in loader:
        with torch.set_grad_enabled(train_mode):
            if is_sequence:
                x, y, mask, lengths = batch
                x, y, mask = x.to(device), y.to(device), mask.to(device)
                out = model(x, lengths)
                total, loss_v, loss_t = _sequence_losses(out, y, mask, lambda_temp)
                # Weight epoch averages by valid timesteps (sequence) ...
                weight = int(mask.sum().item())
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = model(x)
                total, loss_v, loss_t = _point_losses(out, y, lambda_temp, criterion)
                weight = x.shape[0]  # ... or by points (point mode).

            if train_mode:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

        totals["total"] += total.item() * weight
        totals["v"] += loss_v.item() * weight
        totals["t"] += loss_t.item() * weight
        n_seen += weight

    return {k: v / max(n_seen, 1) for k, v in totals.items()}


def _save_checkpoint(path: Path, model: nn.Module, meta: Dict) -> None:
    ensure_dir(path.parent)
    torch.save({**meta, "model_state_dict": model.state_dict()}, path)


def train_shared_model(
    data_root: PathLike,
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
    max_points_per_curve: Union[int, None] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    num_workers: int = 0,
) -> Dict:
    """Train one shared model on all cases and persist all artefacts."""
    model_name = model_name.lower()
    if model_name not in ALL_SHARED_MODELS:
        raise ValueError(
            f"Unknown shared model '{model_name}'. Choices: {sorted(ALL_SHARED_MODELS)}"
        )

    set_seed(seed)
    dev = resolve_device(device)
    is_sequence = model_name in SEQUENCE_MODELS
    print(f"[shared/{model_name}] device={dev} mode={'sequence' if is_sequence else 'point'}")

    # --- data -------------------------------------------------------------- #
    pin = dev == "cuda"
    loader_kwargs = dict(
        outputs_dir=outputs_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin,
    )
    if not is_sequence:
        loader_kwargs["max_points_per_curve"] = max_points_per_curve
    bundle = create_shared_dataloaders(model_name, data_root, **loader_kwargs)

    n_cases = len({c.case_id for c in bundle.curves})
    print(
        f"[shared/{model_name}] n_curves={len(bundle.curves)} across {n_cases} case(s) "
        f"| input_dim={bundle.input_dim} "
        f"| train/val/test curves = {len(bundle.splits['train'])}/"
        f"{len(bundle.splits['val'])}/{len(bundle.splits['test'])}"
    )

    # --- model / optim ----------------------------------------------------- #
    model_kwargs = make_shared_model_kwargs(
        model_name, bundle.input_dim, hidden_dim, num_layers, dropout
    )
    model = build_shared_model(model_name, model_kwargs).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
    )
    criterion = nn.MSELoss()

    ckpt_dir = ensure_dir(shared_checkpoint_dir(outputs_dir, model_name))
    mdir = ensure_dir(shared_metrics_dir(outputs_dir, model_name))

    ckpt_meta = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "is_sequence": is_sequence,
        "input_dim": bundle.input_dim,
        "param_names": bundle.param_names,
        "n_parameters": len(bundle.param_names),
    }

    # --- training loop ----------------------------------------------------- #
    history = []
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    start = time.time()

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(
            model, bundle.train_loader, is_sequence, lambda_temp, criterion, dev, optimizer
        )
        va = _run_epoch(
            model, bundle.val_loader, is_sequence, lambda_temp, criterion, dev, optimizer=None
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
            f"[shared/{model_name}] epoch {epoch:4d} | train {tr['total']:.5f} | "
            f"val {va['total']:.5f} | loss_v {va['v']:.5f} | loss_t {va['t']:.5f} | "
            f"lr {cur_lr:.2e}"
        )

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
                    f"[shared/{model_name}] early stopping at epoch {epoch} "
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
        "model_name": model_name,
        "is_sequence": is_sequence,
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
        "max_points_per_curve": max_points_per_curve,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "model_kwargs": model_kwargs,
        "input_dim": bundle.input_dim,
        "n_parameters": len(bundle.param_names),
        "n_curves": len(bundle.curves),
        "n_cases": n_cases,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "elapsed_s": elapsed,
    }
    save_json(run_config, mdir / "run_config.json")

    print(
        f"[shared/{model_name}] done in {elapsed:.1f}s | "
        f"best_val={best_val:.5f} @ epoch {best_epoch}"
    )
    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "best_checkpoint": str(ckpt_dir / "best_model.pt"),
        "final_checkpoint": str(ckpt_dir / "final_model.pt"),
        "history_csv": str(mdir / "history.csv"),
        "run_config": str(mdir / "run_config.json"),
        "elapsed_s": elapsed,
    }
