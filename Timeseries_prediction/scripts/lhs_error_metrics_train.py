#!/usr/bin/env python3
"""Official LHS error-metric prediction pipeline.

Predicts the two scalar error metrics ``rmse_v_mV`` and ``rmse_t_C`` for each
successful ``(sample_id, experiment case)`` from the same 13 physical/operating
features used by the official time-series pipeline
(``scripts/emergency_lhs_train.py``).

Nine model families are trained under one identical, stored sample_id split:

  neural  : ann, mlp, wide_deep_mlp, gated_mlp, deep_ensemble_mlp
  trees   : random_forest, extra_trees, xgboost, catboost

Architectures are reused from the repository where they already exist:
  * ``ann``       -> ``src.final_filtered.models.ANN`` (shallow, 1 hidden layer)
  * ``mlp``       -> ``src.models.MLP`` (3 layers h->2h->4h, LayerNorm)
  * ``gated_mlp`` -> ``src.gated_mlp_independent.model.GatedMLP`` (gated residual)
  * ``deep_ensemble_mlp`` -> K independently-seeded ``src.models.MLP`` members.
Only ``wide_deep_mlp`` is a new (documented) architecture, as the repo has no
Wide & Deep model.

Contract highlights
-------------------
* One identical split for every model, reused from the time-series run.
* Feature and target scalers fit on TRAIN rows only.
* Neural checkpoints selected by validation loss; tree/boosting hyper-parameters
  or iteration counts selected on the validation split only.
* The test split is scored exactly once, after all model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.models import MLP  # noqa: E402  official 3-layer MLP
from src.final_filtered.models import ANN  # noqa: E402  official shallow ANN
from src.gated_mlp_independent.model import GatedMLP  # noqa: E402  official gated MLP

# Reuse the exact split routine of the time-series pipeline for the fallback.
sys.path.insert(0, str(REPO / "scripts"))
from emergency_lhs_train import split_sample_ids  # noqa: E402

# Additive reproducibility/reporting helpers (no effect on training behaviour).
import lhs_retrain_reporting as reporting  # noqa: E402


NEURAL_MODELS = ["ann", "mlp", "wide_deep_mlp", "gated_mlp", "deep_ensemble_mlp"]
TREE_MODELS = ["random_forest", "extra_trees", "xgboost", "catboost"]
ALL_MODELS = NEURAL_MODELS + TREE_MODELS

DISPLAY_NAMES = {
    "ann": "ANN",
    "mlp": "MLP",
    "wide_deep_mlp": "Wide & Deep MLP",
    "gated_mlp": "Gated MLP",
    "deep_ensemble_mlp": "Deep Ensemble MLP",
    "random_forest": "Random Forest",
    "extra_trees": "ExtraTrees",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

TARGET_COLS = ["rmse_v_mV", "rmse_t_C"]
TARGET_LABELS = {"rmse_v_mV": "Voltage RMSE (mV)", "rmse_t_C": "Temperature RMSE (°C)"}


# --------------------------------------------------------------------------- #
# Wide & Deep MLP (new; the repo has no such model)
# --------------------------------------------------------------------------- #
class WideDeepMLP(nn.Module):
    """Classic Wide & Deep regressor (Cheng et al., 2016).

    A *wide* linear branch (memorization) on the raw feature vector is summed
    with a *deep* MLP tower (generalization). Distinct from ``src.models.MLP``,
    which is a purely deep stack with no wide skip connection.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 n_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        self.wide = nn.Linear(input_dim, output_dim)
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.deep = nn.Sequential(*layers)
        self.deep_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wide(x) + self.deep_head(self.deep(x))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    data_dir: str
    output_dir: str
    models: List[str]
    split_json: Optional[str]
    max_sample_ids: Optional[int]
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_size: int
    dropout: float
    ensemble_size: int
    seed: int
    device: str
    inference_repeats: int
    smoke: bool


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official LHS error-metric pipeline.")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    p.add_argument("--split-json", type=Path, default=None,
                   help="Stored split_sample_ids.json from the time-series run.")
    p.add_argument("--max-sample-ids", type=int, default=0,
                   help="Limit unique sample IDs (smoke). 0 uses all.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden-size", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--inference-repeats", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="Shrink tree/boosting settings for a fast smoke run.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features_targets(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """Build the identical 13 features + the 2 targets from the EM table.

    Features (exact time-series order): the 10 physical parameters from
    ``parameter_sets_physical.csv`` followed by ``operation_encoded``,
    ``c_rate`` and ``initial_temperature_C``.
    """
    em_path = data_dir / "error_metrics_by_case.csv"
    params_path = data_dir / "parameter_sets_physical.csv"
    for pth in (em_path, params_path):
        if not pth.exists():
            raise FileNotFoundError(pth)

    em = pd.read_csv(em_path)
    em = em.loc[em["simulation_status"].eq("ok")].copy()
    params = pd.read_csv(params_path)
    param_cols = [c for c in params.columns if c != "sample_id"]
    if len(param_cols) != 10:
        raise ValueError(f"Expected 10 physical params, found {len(param_cols)}")

    meta_cols = ["sample_id", "sequence_id", "experiment_id", "operation",
                 "c_rate", "initial_temperature_C", *TARGET_COLS]
    df = em[meta_cols].merge(params, on="sample_id", how="left", validate="many_to_one")
    if df[param_cols].isna().any().any():
        raise ValueError("Missing physical parameters after merge")

    op = df["operation"].map({"discharge": 0.0, "charge": 1.0})
    if op.isna().any():
        raise ValueError(f"Unknown operation values: {df['operation'].unique().tolist()}")

    feature_names = param_cols + ["operation_encoded", "c_rate", "initial_temperature_C"]
    X = np.column_stack([
        df[param_cols].to_numpy(dtype=np.float64),
        op.to_numpy(dtype=np.float64),
        df["c_rate"].to_numpy(dtype=np.float64),
        df["initial_temperature_C"].to_numpy(dtype=np.float64),
    ])
    Y = df[TARGET_COLS].to_numpy(dtype=np.float64)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise ValueError("Non-finite feature/target values detected")
    meta = df[["sample_id", "sequence_id", "experiment_id", "operation", "c_rate"]].reset_index(drop=True)
    return meta, X, Y, feature_names


def resolve_split(split_json: Optional[Path]) -> Optional[Dict[str, List[str]]]:
    """Return the stored split, preferring an explicit path, else LATEST run."""
    candidates: List[Path] = []
    if split_json is not None:
        candidates.append(split_json)
    latest = REPO / "outputs/lhs_1000_seed42/LATEST_OFFICIAL_RUN.txt"
    if latest.exists():
        candidates.append(Path(latest.read_text().strip()) / "artifacts" / "split_sample_ids.json")
    for c in candidates:
        if c.exists():
            print(f"[split] loading stored split: {c}", flush=True)
            return json.loads(c.read_text())
    return None


def assign_splits(meta: pd.DataFrame, split_ids: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    lookup: Dict[str, str] = {}
    for name, ids in split_ids.items():
        for sid in ids:
            lookup[str(sid)] = name
    labels = meta["sample_id"].astype(str).map(lookup)
    if labels.isna().any():
        missing = meta.loc[labels.isna(), "sample_id"].unique()[:5]
        raise ValueError(f"{int(labels.isna().sum())} rows have sample_ids absent "
                         f"from the stored split, e.g. {missing.tolist()}")
    return {s: np.flatnonzero((labels == s).to_numpy()) for s in ("train", "val", "test")}


def validate_no_leakage(meta: pd.DataFrame, idx: Dict[str, np.ndarray]) -> Dict[str, int]:
    sets = {s: set(meta.iloc[i]["sample_id"]) for s, i in idx.items()}
    assert not (sets["train"] & sets["val"]), "train/val sample_id leakage"
    assert not (sets["train"] & sets["test"]), "train/test sample_id leakage"
    assert not (sets["val"] & sets["test"]), "val/test sample_id leakage"
    return {f"{s}_sample_ids": len(sets[s]) for s in sets} | \
           {f"{s}_rows": int(len(idx[s])) for s in idx}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    return float(r2_score(a, b)) if len(a) >= 2 and np.var(a) > 0 else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Target-wise + macro MAE/RMSE/R2 (targets in physical units)."""
    out: Dict[str, float] = {}
    per_target = []
    for ch, prefix in ((0, "v"), (1, "t")):
        yt, yp = y_true[:, ch], y_pred[:, ch]
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(math.sqrt(mean_squared_error(yt, yp)))
        r2 = _safe_r2(yt, yp)
        out[f"{prefix}_mae"] = mae
        out[f"{prefix}_rmse"] = rmse
        out[f"{prefix}_r2"] = r2
        per_target.append((mae, rmse, r2))
    out["macro_mae"] = float(np.mean([p[0] for p in per_target]))
    out["macro_rmse"] = float(np.mean([p[1] for p in per_target]))
    out["macro_r2"] = float(np.nanmean([p[2] for p in per_target]))
    out["n_rows"] = int(len(y_true))
    return out


def grouped_metrics(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                    by: str, model: str) -> List[Dict[str, object]]:
    rows = []
    for key, sub in meta.groupby(by):
        idx = sub.index.to_numpy()
        m = compute_metrics(y_true[idx], y_pred[idx])
        m.update({"model": model, "display_name": DISPLAY_NAMES[model],
                  "group_kind": by, "group": str(key)})
        rows.append(m)
    return rows


def make_ranking(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics_df.copy()
    rank_cols = []
    for col in ["v_mae", "v_rmse", "t_mae", "t_rmse", "macro_mae", "macro_rmse"]:
        rc = f"rank_{col}"
        ranked[rc] = ranked[col].rank(method="average", ascending=True)
        rank_cols.append(rc)
    for col in ["v_r2", "t_r2", "macro_r2"]:
        rc = f"rank_{col}"
        ranked[rc] = ranked[col].rank(method="average", ascending=False)
        rank_cols.append(rc)
    ranked["average_rank"] = ranked[rank_cols].mean(axis=1)
    return ranked.sort_values("average_rank").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Neural training
# --------------------------------------------------------------------------- #
def build_neural(name: str, in_dim: int, cfg: RunConfig, seed: int) -> nn.Module:
    set_seed(seed)
    if name in ("mlp", "deep_ensemble_mlp"):
        return MLP(in_dim, 2, hidden_dim=cfg.hidden_size, dropout=cfg.dropout)
    if name == "ann":
        return ANN(in_dim, 2, hidden_dim=128, dropout=cfg.dropout)
    if name == "wide_deep_mlp":
        return WideDeepMLP(in_dim, 2, hidden_dim=128, n_layers=3, dropout=cfg.dropout)
    if name == "gated_mlp":
        return GatedMLP(in_dim, hidden_dim=128, n_blocks=4, dropout=cfg.dropout, out_dim=2)
    raise ValueError(name)


def train_one_neural(model: nn.Module, loaders: Dict[str, DataLoader], cfg: RunConfig,
                     device: torch.device, ckpt_path: Path) -> Tuple[float, int, List[Dict]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, cfg.patience // 2))
    loss_fn = nn.MSELoss()
    best_val, best_epoch, no_improve = float("inf"), 0, 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb).reshape(yb.shape), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        val_loss = _eval_loss(model, loaders["val"], device, loss_fn)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "val_loss": val_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        if val_loss < best_val - 1e-9:
            best_val, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            break
    return best_val, best_epoch, history


def _eval_loss(model: nn.Module, loader: DataLoader, device: torch.device,
               loss_fn: nn.Module) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).reshape(yb.shape)
            tot += float(loss_fn(pred, yb).item()) * len(xb)
            n += len(xb)
    return tot / max(n, 1)


def neural_predict(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32)).to(device)
        return model(t).reshape(len(X), 2).cpu().numpy()


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_parity(y_true, y_pred, channel, model, path, std=None):
    yt, yp = y_true[:, channel], y_pred[:, channel]
    rmse = math.sqrt(mean_squared_error(yt, yp))
    r2 = _safe_r2(yt, yp)
    label = TARGET_LABELS[TARGET_COLS[channel]]
    fig, ax = plt.subplots(figsize=(6, 6))
    if std is not None:
        ax.errorbar(yt, yp, yerr=std[:, channel], fmt="o", ms=4, alpha=0.4,
                    ecolor="grey", elinewidth=0.6, capsize=1.5, label="±1σ")
    else:
        ax.scatter(yt, yp, s=12, alpha=0.4)
    lo = float(min(yt.min(), yp.min())); hi = float(max(yt.max(), yp.max()))
    ax.plot([lo, hi], [lo, hi], "--", lw=1.2, color="k")
    ax.set_xlabel(f"Target {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(f"{DISPLAY_NAMES[model]} — {label}\nRMSE={rmse:.4f}  R²={r2:.4f}")
    ax.grid(alpha=0.2)
    if std is not None:
        ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_residuals(y_true, y_pred, model, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ch, ax in enumerate(axes):
        yt, yp = y_true[:, ch], y_pred[:, ch]
        ax.scatter(yp, yp - yt, s=10, alpha=0.4)
        ax.axhline(0.0, ls="--", color="k", lw=1)
        ax.set_xlabel(f"Predicted {TARGET_LABELS[TARGET_COLS[ch]]}")
        ax.set_ylabel("Residual (pred − target)")
        ax.set_title(f"{DISPLAY_NAMES[model]} residuals")
        ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                            model: str, path: Path):
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(feature_names))))
    ax.barh(np.arange(len(order))[::-1], importances[order])
    ax.set_yticks(np.arange(len(order))[::-1])
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"{DISPLAY_NAMES[model]} feature importance")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_ensemble_uncertainty(y_true, y_pred, std, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ch, ax in enumerate(axes):
        abs_err = np.abs(y_pred[:, ch] - y_true[:, ch])
        ax.scatter(std[:, ch], abs_err, s=12, alpha=0.4)
        lim = max(float(std[:, ch].max()), float(abs_err.max()))
        ax.plot([0, lim], [0, lim], "--", color="k", lw=1, label="ideal |err|=σ")
        ax.set_xlabel(f"Predictive σ — {TARGET_LABELS[TARGET_COLS[ch]]}")
        ax.set_ylabel("Absolute error")
        ax.set_title("Deep Ensemble uncertainty calibration")
        ax.grid(alpha=0.2); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def plot_ranking_heatmap(ranked: pd.DataFrame, path: Path):
    metric_cols = ["v_mae", "v_rmse", "v_r2", "t_mae", "t_rmse", "t_r2",
                   "macro_rmse", "average_rank"]
    rank_lookup = {"v_mae": "rank_v_mae", "v_rmse": "rank_v_rmse", "v_r2": "rank_v_r2",
                   "t_mae": "rank_t_mae", "t_rmse": "rank_t_rmse", "t_r2": "rank_t_r2",
                   "macro_rmse": "rank_macro_rmse", "average_rank": "average_rank"}
    color = np.column_stack([ranked[rank_lookup[c]] for c in metric_cols])
    values = ranked[metric_cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.7 * len(ranked) + 1.5)))
    im = ax.imshow(color, aspect="auto", cmap="viridis_r")
    ax.set_xticks(np.arange(len(metric_cols)), labels=metric_cols, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ranked)), labels=ranked["display_name"].tolist())
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            fmt = f"{values[i, j]:.3f}" if metric_cols[j] != "average_rank" else f"{values[i, j]:.2f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8,
                    color="white")
    ax.set_title("Error-metric model ranking (color = per-metric rank, 1=best)")
    fig.colorbar(im, ax=ax, label="Rank (1 = best)")
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


# --------------------------------------------------------------------------- #
# Tree / boosting training (val-selected)
# --------------------------------------------------------------------------- #
def train_forest(name: str, Xtr, Ytr, Xva, Yva, cfg: RunConfig):
    Ctor = RandomForestRegressor if name == "random_forest" else ExtraTreesRegressor
    if cfg.smoke:
        grid = [{"n_estimators": 50, "max_depth": None}]
    else:
        grid = [{"n_estimators": 400, "max_depth": None},
                {"n_estimators": 400, "max_depth": 16},
                {"n_estimators": 800, "max_depth": 24}]
    best, best_val, best_hp = None, float("inf"), None
    for hp in grid:
        m = Ctor(n_jobs=-1, random_state=cfg.seed, **hp).fit(Xtr, Ytr)
        v = math.sqrt(mean_squared_error(Yva, m.predict(Xva)))
        if v < best_val:
            best, best_val, best_hp = m, v, hp
    return best, best_hp


def train_xgboost(Xtr, Ytr, Xva, Yva, cfg: RunConfig):
    import xgboost as xgb
    n_est = 50 if cfg.smoke else 3000
    stop = 10 if cfg.smoke else 50
    models = []
    for ch in range(2):
        m = xgb.XGBRegressor(
            n_estimators=n_est, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=cfg.seed, n_jobs=-1, tree_method="hist",
            early_stopping_rounds=stop, eval_metric="rmse")
        m.fit(Xtr, Ytr[:, ch], eval_set=[(Xva, Yva[:, ch])], verbose=False)
        models.append(m)
    hp = {"n_estimators": n_est, "learning_rate": 0.05, "max_depth": 6,
          "best_iteration": [int(getattr(m, "best_iteration", -1)) for m in models]}
    return models, hp


def train_catboost(Xtr, Ytr, Xva, Yva, cfg: RunConfig):
    from catboost import CatBoostRegressor, Pool
    iters = 50 if cfg.smoke else 3000
    wait = 10 if cfg.smoke else 50
    m = CatBoostRegressor(
        iterations=iters, learning_rate=0.05, depth=6, loss_function="MultiRMSE",
        random_seed=cfg.seed, od_type="Iter", od_wait=wait, verbose=False,
        allow_writing_files=False)
    m.fit(Pool(Xtr, Ytr), eval_set=Pool(Xva, Yva), use_best_model=True)
    hp = {"iterations": iters, "learning_rate": 0.05, "depth": 6,
          "best_iteration": int(m.get_best_iteration())}
    return m, hp


def tree_predict(name: str, model, X: np.ndarray) -> np.ndarray:
    if name == "xgboost":
        return np.column_stack([model[0].predict(X), model[1].predict(X)])
    pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64).reshape(len(X), 2)


def tree_importance(name: str, model) -> Optional[np.ndarray]:
    try:
        if name == "xgboost":
            return np.mean([m.feature_importances_ for m in model], axis=0)
        if name == "catboost":
            return np.asarray(model.get_feature_importance(), dtype=np.float64)
        return np.asarray(model.feature_importances_, dtype=np.float64)
    except Exception:
        return None


def save_tree(name: str, model, path_stub: Path) -> Path:
    import joblib
    if name == "catboost":
        p = path_stub.with_suffix(".cbm"); model.save_model(str(p)); return p
    if name == "xgboost":
        p = path_stub.with_suffix(".joblib"); joblib.dump(model, p); return p
    p = path_stub.with_suffix(".joblib"); joblib.dump(model, p); return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    cfg = RunConfig(
        data_dir=str(args.data_dir.resolve()), output_dir=str(args.output_dir.resolve()),
        models=list(dict.fromkeys(args.models)),
        split_json=str(args.split_json) if args.split_json else None,
        max_sample_ids=None if args.max_sample_ids == 0 else args.max_sample_ids,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        hidden_size=args.hidden_size, dropout=args.dropout,
        ensemble_size=args.ensemble_size, seed=args.seed, device=args.device,
        inference_repeats=args.inference_repeats, smoke=args.smoke,
    )
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    dirs = {k: out / k for k in ("models", "figures", "metrics", "artifacts")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    with (out / "run_config.json").open("w") as f:
        json.dump(vars(args) | {"resolved": cfg.__dict__}, f, indent=2, default=str)

    print("[1/6] Loading features and targets...", flush=True)
    meta, X_raw, Y_raw, feature_names = load_features_targets(Path(cfg.data_dir))

    split_ids = resolve_split(Path(cfg.split_json) if cfg.split_json else None)
    if split_ids is None:
        print("[split] no stored split found; reconstructing with the "
              "time-series split routine (seed=42, 0.70/0.15/0.15).", flush=True)
        split_ids = split_sample_ids(sorted(meta["sample_id"].unique()),
                                     0.70, 0.15, 0.15, cfg.seed)

    if cfg.max_sample_ids:
        # Smoke: keep the first N sample_ids present in the split order, but
        # never break the split — just subsample sample_ids within each split.
        rng = np.random.default_rng(cfg.seed)
        chosen = set()
        for s in ("train", "val", "test"):
            ids = [i for i in split_ids[s] if i in set(meta["sample_id"])]
            k = max(1, int(round(cfg.max_sample_ids * len(ids) / max(
                sum(len(split_ids[q]) for q in ("train", "val", "test")), 1))))
            chosen |= set(rng.choice(ids, size=min(k, len(ids)), replace=False))
        keep = meta["sample_id"].isin(chosen).to_numpy()
        meta = meta.loc[keep].reset_index(drop=True)
        X_raw, Y_raw = X_raw[keep], Y_raw[keep]

    idx = assign_splits(meta, split_ids)
    split_counts = validate_no_leakage(meta, idx)
    with (dirs["artifacts"] / "split_counts.json").open("w") as f:
        json.dump(split_counts, f, indent=2)
    print(f"[split] {split_counts}", flush=True)

    print("[2/6] Fitting train-only scalers...", flush=True)
    x_scaler = StandardScaler().fit(X_raw[idx["train"]])
    y_scaler = StandardScaler().fit(Y_raw[idx["train"]])
    Xs = x_scaler.transform(X_raw).astype(np.float64)
    Ys = y_scaler.transform(Y_raw).astype(np.float64)
    with (dirs["artifacts"] / "scalers.json").open("w") as f:
        json.dump({"x_mean": x_scaler.mean_.tolist(), "x_scale": x_scaler.scale_.tolist(),
                   "y_mean": y_scaler.mean_.tolist(), "y_scale": y_scaler.scale_.tolist(),
                   "feature_names": feature_names, "targets": TARGET_COLS}, f, indent=2)

    # Reproducibility provenance (additive; does not affect training).
    reporting.write_environment(out / "environment.json", REPO)
    reporting.write_dataset_audit(
        out / "dataset_audit.json", Path(cfg.data_dir),
        split_counts=split_counts,
        selected={"rows": int(len(meta)), "features": len(feature_names),
                  "targets": TARGET_COLS, "unique_sample_ids": int(meta["sample_id"].nunique())})

    device = torch.device(cfg.device)
    print(f"Device: {device}", flush=True)

    def make_loader(split, shuffle):
        ds = TensorDataset(torch.from_numpy(Xs[idx[split]].astype(np.float32)),
                           torch.from_numpy(Ys[idx[split]].astype(np.float32)))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)
    loaders = {"train": make_loader("train", True), "val": make_loader("val", False)}

    test_meta = meta.iloc[idx["test"]].reset_index(drop=True)
    Y_test = Y_raw[idx["test"]]
    predictions_wide = test_meta.copy()
    predictions_wide["rmse_v_mV_true"] = Y_test[:, 0]
    predictions_wide["rmse_t_C_true"] = Y_test[:, 1]

    all_metrics: List[Dict] = []
    all_timing: List[Dict] = []
    all_per_case: List[Dict] = []
    hyperparams: Dict[str, object] = {}

    print("[3/6] Training models...", flush=True)
    for name in cfg.models:
        print(f"\n===== {DISPLAY_NAMES[name]} =====", flush=True)
        ens_std = None
        best_epoch: Optional[int] = None
        peak_gpu_mb = 0.0
        if name in NEURAL_MODELS:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            if name == "deep_ensemble_mlp":
                members, member_preds, member_epochs = [], [], []
                for k in range(cfg.ensemble_size):
                    m = build_neural("deep_ensemble_mlp", Xs.shape[1], cfg, cfg.seed + k).to(device)
                    ckpt = dirs["models"] / f"{name}_member{k}.pt"
                    bv, be, _ = train_one_neural(m, loaders, cfg, device, ckpt)
                    m.load_state_dict(torch.load(ckpt, map_location=device))
                    members.append(m)
                    member_epochs.append(be)
                    member_preds.append(y_scaler.inverse_transform(
                        neural_predict(m, Xs[idx["test"]], device)))
                    print(f"  member {k}: best_val={bv:.5f} @epoch {be}", flush=True)
                stack = np.stack(member_preds, axis=0)          # [K, N, 2]
                pred = stack.mean(axis=0)
                ens_std = stack.std(axis=0)
                best_epoch = int(round(float(np.mean(member_epochs)))) if member_epochs else None
                hyperparams[name] = {"ensemble_size": cfg.ensemble_size,
                                     "member": "src.models.MLP", "hidden": cfg.hidden_size}
            else:
                model = build_neural(name, Xs.shape[1], cfg, cfg.seed).to(device)
                ckpt = dirs["models"] / f"{name}_best.pt"
                bv, be, _ = train_one_neural(model, loaders, cfg, device, ckpt)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                pred = y_scaler.inverse_transform(neural_predict(model, Xs[idx["test"]], device))
                best_epoch = be
                hyperparams[name] = _neural_hp(name, cfg)
                print(f"  best_val={bv:.5f} @epoch {be}", flush=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            train_secs = time.perf_counter() - t0
            n_params = _neural_param_count(name, members if name == "deep_ensemble_mlp" else model)
            importance = None
        else:
            t0 = time.perf_counter()
            Xtr, Ytr = Xs[idx["train"]], Y_raw[idx["train"]]
            Xva, Yva = Xs[idx["val"]], Y_raw[idx["val"]]
            if name in ("random_forest", "extra_trees"):
                model, hp = train_forest(name, Xtr, Ytr, Xva, Yva, cfg)
            elif name == "xgboost":
                model, hp = train_xgboost(Xtr, Ytr, Xva, Yva, cfg)
            else:
                model, hp = train_catboost(Xtr, Ytr, Xva, Yva, cfg)
            train_secs = time.perf_counter() - t0
            hyperparams[name] = hp
            saved = save_tree(name, model, dirs["models"] / f"{name}_best")
            pred = tree_predict(name, model, Xs[idx["test"]])
            importance = tree_importance(name, model)
            n_params = 0

        # ---- inference timing (repeated) ----------------------------------- #
        Xte = Xs[idx["test"]]
        inf_times = []
        for _ in range(cfg.inference_repeats):
            ti = time.perf_counter()
            if name in NEURAL_MODELS and name != "deep_ensemble_mlp":
                _ = neural_predict(model, Xte, device)
            elif name == "deep_ensemble_mlp":
                _ = np.stack([neural_predict(m, Xte, device) for m in members], 0).mean(0)
            else:
                _ = tree_predict(name, model, Xte)
            inf_times.append(time.perf_counter() - ti)
        inf_total = float(np.mean(inf_times))

        metrics = compute_metrics(Y_test, pred)
        metrics.update({"model": name, "display_name": DISPLAY_NAMES[name]})
        all_metrics.append(metrics)
        all_timing.append({
            "model": name, "display_name": DISPLAY_NAMES[name],
            "best_epoch": best_epoch,
            "train_seconds": train_secs, "inference_seconds_total": inf_total,
            "inference_ms_per_row": inf_total * 1000.0 / max(len(Xte), 1),
            "peak_gpu_memory_mb": peak_gpu_mb,
            "parameter_count": n_params})
        for by in ("experiment_id", "operation", "c_rate"):
            all_per_case.extend(grouped_metrics(test_meta, Y_test, pred, by, name))

        predictions_wide[f"{name}__rmse_v_mV_pred"] = pred[:, 0]
        predictions_wide[f"{name}__rmse_t_C_pred"] = pred[:, 1]
        if ens_std is not None:
            predictions_wide[f"{name}__rmse_v_mV_std"] = ens_std[:, 0]
            predictions_wide[f"{name}__rmse_t_C_std"] = ens_std[:, 1]

        # ---- figures ------------------------------------------------------- #
        plot_parity(Y_test, pred, 0, name, dirs["figures"] / f"{name}_voltage_parity.png",
                    std=ens_std)
        plot_parity(Y_test, pred, 1, name, dirs["figures"] / f"{name}_temperature_parity.png",
                    std=ens_std)
        plot_residuals(Y_test, pred, name, dirs["figures"] / f"{name}_residuals.png")
        if importance is not None:
            plot_feature_importance(importance, feature_names, name,
                                    dirs["figures"] / f"{name}_feature_importance.png")
        if ens_std is not None:
            plot_ensemble_uncertainty(Y_test, pred, ens_std,
                                      dirs["figures"] / f"{name}_uncertainty.png")
        print(f"  test macro RMSE={metrics['macro_rmse']:.4f} "
              f"(V={metrics['v_rmse']:.3f} mV, T={metrics['t_rmse']:.3f} °C) "
              f"train={train_secs:.1f}s", flush=True)

    print("\n[4/6] Aggregating metrics...", flush=True)
    metrics_df = pd.DataFrame(all_metrics)
    ranked_df = make_ranking(metrics_df)
    timing_df = pd.DataFrame(all_timing)
    # Additive: device/CUDA/GPU/throughput/test-batch columns.
    timing_df = reporting.enrich_timing(
        timing_df, device, cfg.batch_size, int(len(idx["test"])))
    per_case_df = pd.DataFrame(all_per_case)

    metrics_df.to_csv(dirs["metrics"] / "model_metrics.csv", index=False)
    ranked_df.to_csv(dirs["metrics"] / "model_ranking.csv", index=False)
    timing_df.to_csv(dirs["metrics"] / "model_timing.csv", index=False)
    per_case_df.to_csv(dirs["metrics"] / "per_case_metrics.csv", index=False)
    predictions_wide.to_csv(dirs["metrics"] / "error_metric_predictions.csv", index=False)
    # Canonical predictions.csv name required by the retrain protocol.
    predictions_wide.to_csv(dirs["metrics"] / "predictions.csv", index=False)
    with (dirs["artifacts"] / "hyperparameters.json").open("w") as f:
        json.dump(hyperparams, f, indent=2, default=str)

    print("[5/6] Ranking heatmap...", flush=True)
    plot_ranking_heatmap(ranked_df, dirs["figures"] / "model_ranking_heatmap.png")

    print("[6/6] Summary...", flush=True)
    best = str(ranked_df.iloc[0]["model"])
    lines = [
        "# LHS Error-Metric Prediction Summary", "",
        f"- Data: `{cfg.data_dir}`",
        f"- Rows (sample-cases): {len(meta)}  |  features: {len(feature_names)}  targets: {TARGET_COLS}",
        f"- Split (rows): {split_counts}",
        f"- Models: {', '.join(DISPLAY_NAMES[m] for m in cfg.models)}",
        f"- Best model: **{DISPLAY_NAMES[best]}** (lowest average rank)",
        "", "## Ranking", "", "```text",
        ranked_df[["display_name", "v_rmse", "v_r2", "t_rmse", "t_r2",
                   "macro_rmse", "average_rank"]].to_string(index=False),
        "```",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + ranked_df[["display_name", "macro_rmse", "v_rmse", "t_rmse",
                            "average_rank"]].to_string(index=False))
    print(f"\nOutput: {out}")
    return 0


def _neural_hp(name: str, cfg: RunConfig) -> Dict[str, object]:
    common = {"epochs": cfg.epochs, "patience": cfg.patience, "batch_size": cfg.batch_size,
              "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay,
              "dropout": cfg.dropout, "optimizer": "AdamW", "loss": "MSE(standardized)"}
    arch = {"ann": {"impl": "src.final_filtered.models.ANN", "hidden": 128, "layers": 1},
            "mlp": {"impl": "src.models.MLP", "hidden": cfg.hidden_size, "widths": "h,2h,4h"},
            "wide_deep_mlp": {"impl": "WideDeepMLP", "hidden": 128, "deep_layers": 3},
            "gated_mlp": {"impl": "src.gated_mlp_independent.model.GatedMLP",
                          "hidden": 128, "n_blocks": 4}}
    return common | arch.get(name, {})


def _neural_param_count(name, model_or_list) -> int:
    if name == "deep_ensemble_mlp":
        return int(sum(sum(p.numel() for p in m.parameters()) for m in model_or_list))
    return int(sum(p.numel() for p in model_or_list.parameters()))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
