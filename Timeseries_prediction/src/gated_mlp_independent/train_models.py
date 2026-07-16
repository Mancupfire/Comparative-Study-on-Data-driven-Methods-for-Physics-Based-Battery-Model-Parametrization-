"""Train twelve fully independent scalar Gated-MLP RMSE surrogates.

Each (condition x metric) target gets its own model instance, optimizer, target
scaler, early-stopping and checkpoint.  The input scaler and the leakage-safe
grouped split (by ``sample_id``) are shared across all twelve models.

Usage::

    python -m src.gated_mlp_independent.train_models \
        --data-dir ann_rmse_training_2500_physics_aligned \
        --output-dir ann_rmse_training_2500_physics_aligned/gated_mlp_12models_results \
        --seed 42 --device auto
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow ``python src/gated_mlp_independent/train_models.py`` as well as -m.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.utils import ensure_dir, resolve_device, save_json, set_seed  # noqa: E402
from src.gated_mlp_independent.model import (  # noqa: E402
    GatedMLP,
    StandardScaler,
    TargetTransform,
)
from src.gated_mlp_independent import pipeline as P  # noqa: E402


# --------------------------------------------------------------------------- #
# Training of a single model
# --------------------------------------------------------------------------- #
def _loss_fn(name: str):
    import torch

    return {
        "smooth_l1": torch.nn.SmoothL1Loss(),
        "mse": torch.nn.MSELoss(),
        "l1": torch.nn.L1Loss(),
    }[name]


def train_one_model(
    x_train: np.ndarray, y_train_scaled: np.ndarray,
    x_val: np.ndarray, y_val_scaled: np.ndarray,
    hp: Dict, device: str, init_seed: int,
) -> Tuple[object, List[Dict], int, float]:
    """Train one GatedMLP; restore the best-val checkpoint. Returns model."""
    import torch

    set_seed(init_seed)
    model = GatedMLP(
        in_dim=x_train.shape[1], hidden_dim=hp["hidden_dim"],
        n_blocks=hp["n_blocks"], dropout=hp["dropout"],
        activation=hp["activation"], out_dim=1,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=hp["lr_factor"],
        patience=hp["lr_patience"], min_lr=hp["min_lr"],
    )
    loss_fn = _loss_fn(hp["loss"])

    xt = torch.tensor(x_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    xv = torch.tensor(x_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    n = len(xt)
    bs = hp["batch_size"]
    g = torch.Generator(device="cpu").manual_seed(init_seed)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    history: List[Dict] = []
    bad = 0

    for epoch in range(hp["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            pred = model(xt[idx])
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        sched.step(val_loss)
        history.append({
            "epoch": epoch, "train_loss": epoch_loss, "val_loss": val_loss,
            "lr": opt.param_groups[0]["lr"],
        })

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad = 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def _predict_scaled(model, x: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(x, dtype=torch.float32, device=device))
    return out.cpu().numpy().reshape(-1)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def baseline_predictions(
    x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray
) -> Dict[str, np.ndarray]:
    """Mean predictor + Ridge regression (fit on training only)."""
    from sklearn.linear_model import Ridge

    preds = {"mean": np.full(len(x_eval), float(np.mean(y_train)))}
    ridge = Ridge(alpha=1.0, random_state=0)
    ridge.fit(x_train, y_train)
    preds["ridge"] = ridge.predict(x_eval)
    return preds


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _make_plots(plot_dir: Path, name: str, history, y_true, y_pred, dist):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(ep, [h["train_loss"] for h in history], label="train")
    plt.plot(ep, [h["val_loss"] for h in history], label="val")
    plt.xlabel("epoch"); plt.ylabel("scaled loss"); plt.yscale("log")
    plt.title(f"learning curve: {name}"); plt.legend(); plt.tight_layout()
    plt.savefig(plot_dir / f"learning_curve_{name}.png", dpi=110); plt.close()

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.4)
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel("true"); plt.ylabel("predicted"); plt.title(f"parity (test): {name}")
    plt.tight_layout(); plt.savefig(plot_dir / f"parity_test_{name}.png", dpi=110)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, y_pred - y_true, s=8, alpha=0.4)
    plt.axhline(0, color="r", ls="--", lw=1)
    plt.xlabel("predicted"); plt.ylabel("residual (pred - true)")
    plt.title(f"residuals (test): {name}"); plt.tight_layout()
    plt.savefig(plot_dir / f"residual_test_{name}.png", dpi=110); plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(dist, np.abs(y_pred - y_true), s=8, alpha=0.4)
    plt.xlabel("normalized distance to nearest training point")
    plt.ylabel("absolute error"); plt.title(f"error vs train distance: {name}")
    plt.tight_layout(); plt.savefig(plot_dir / f"error_vs_training_distance_{name}.png", dpi=110)
    plt.close()


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def build_hparams(args) -> Dict:
    return {
        "hidden_dim": args.hidden_dim, "n_blocks": args.n_blocks,
        "dropout": args.dropout, "activation": args.activation,
        "loss": args.loss, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay, "batch_size": args.batch_size,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "lr_patience": args.lr_patience, "lr_factor": args.lr_factor,
        "min_lr": args.min_lr, "grad_clip": args.grad_clip,
    }


def run(args) -> None:
    t0 = time.time()
    set_seed(args.seed)
    device = resolve_device(args.device)
    out = ensure_dir(args.output_dir)
    ensure_dir(out / "checkpoints")
    ensure_dir(out / "preprocessing")
    ensure_dir(out / "predictions")
    ensure_dir(out / "plots")

    df = P.load_table(Path(args.data_dir))
    specs = P.build_target_specs()
    hp = build_hparams(args)

    # ---- grouped split on parameter tuples (sample_id) ----
    all_groups = np.array(sorted(df[P.GROUP_KEY].unique()))
    split_groups = P.grouped_split(all_groups, seed=args.seed)
    P.assert_no_group_overlap(split_groups)
    group_to_split = {}
    for sp in ("train", "val", "test"):
        for gid in split_groups[sp]:
            group_to_split[gid] = sp
    df["split"] = df[P.GROUP_KEY].map(group_to_split)

    # ---- shared input scaler: fit on TRAIN sample parameters only ----
    train_param_rows = (
        df[df["split"] == "train"]
        .drop_duplicates(P.GROUP_KEY)[P.PARAM_COLUMNS]
        .to_numpy()
    )
    input_scaler = StandardScaler(P.PARAM_COLUMNS, P.LOG10_COLUMNS).fit(train_param_rows)
    save_json(input_scaler.to_dict(), out / "preprocessing" / "input_scaler.json")

    # Pre-scale all rows once (inputs identical across metrics for same condition).
    df_scaled_X = input_scaler.transform(df[P.PARAM_COLUMNS].to_numpy())

    # ---- distance bins reference: scaled unique training params ----
    train_X_unique = input_scaler.transform(train_param_rows)

    # ---- data audit ----
    audit = {
        "data_dir": str(Path(args.data_dir).resolve()),
        "n_rows": int(len(df)),
        "n_unique_sample_id": int(df[P.GROUP_KEY].nunique()),
        "n_conditions": int(df["case_id_code"].nunique()),
        "input_columns": P.PARAM_COLUMNS,
        "log10_input_columns": P.LOG10_COLUMNS,
        "metrics": P.METRICS,
        "n_models": len(specs),
        "model_targets": [s.name for s in specs],
        "missing_values": {
            c: int(df[c].isna().sum())
            for c in P.PARAM_COLUMNS + P.METRICS if c in df.columns
        },
        "n_duplicate_rows": int(df.duplicated().sum()),
        "split_group_counts": {k: int(len(v)) for k, v in split_groups.items()},
        "target_distributions": {
            m: {
                "min": float(df[m].min()), "max": float(df[m].max()),
                "mean": float(df[m].mean()), "std": float(df[m].std()),
                "skew": float(df[m].skew()), "n_negative": int((df[m] < 0).sum()),
            } for m in P.METRICS
        },
    }
    save_json(audit, out / "data_audit.json")

    # ---- split assignments ----
    split_df = (
        df.drop_duplicates(P.GROUP_KEY)[[P.GROUP_KEY, "split"]]
        .sort_values(P.GROUP_KEY).reset_index(drop=True)
    )
    split_df.to_csv(out / "split_assignments.csv", index=False)

    # ---- parameter range report (per split, original units) ----
    range_rows = []
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp].drop_duplicates(P.GROUP_KEY)
        for c in P.PARAM_COLUMNS:
            range_rows.append({
                "split": sp, "parameter": c, "n_groups": len(sub),
                "min": float(sub[c].min()), "max": float(sub[c].max()),
                "mean": float(sub[c].mean()),
            })
    pd.DataFrame(range_rows).to_csv(out / "parameter_range_report.csv", index=False)

    # ---- distance-bin edges (tertiles over val+test of metric-agnostic distance) ----
    eval_mask = df["split"].isin(["val", "test"]).to_numpy()
    eval_unique = df.loc[eval_mask].drop_duplicates(P.GROUP_KEY)
    eval_unique_X = input_scaler.transform(eval_unique[P.PARAM_COLUMNS].to_numpy())
    eval_unique_dist = P.nearest_train_distance(train_X_unique, eval_unique_X)
    edges = (
        float(np.quantile(eval_unique_dist, 1 / 3)),
        float(np.quantile(eval_unique_dist, 2 / 3)),
    )
    sid_to_dist = dict(zip(eval_unique[P.GROUP_KEY].to_numpy(), eval_unique_dist))

    # =================================================================== #
    # Train the twelve models
    # =================================================================== #
    per_model_rows: List[Dict] = []
    baseline_rows: List[Dict] = []
    dist_bin_rows: List[Dict] = []
    pred_records = {"train": [], "validation": [], "test": []}
    target_meta = {}

    for ti, spec in enumerate(specs):
        cond_mask = (df["case_id_code"] == spec.case_code).to_numpy()
        sub = df.loc[cond_mask]
        Xc = df_scaled_X[cond_mask]
        yc = sub[spec.metric].to_numpy(dtype=np.float64)
        sids = sub[P.GROUP_KEY].to_numpy()
        sp = sub["split"].to_numpy()

        tr, va, te = sp == "train", sp == "val", sp == "test"

        # auto target transform: log1p for strongly right-skewed non-negative
        skew = float(pd.Series(yc[tr]).skew())
        use_log1p = bool(skew > 1.0 and yc[tr].min() >= 0) and not args.no_log1p
        tfm = TargetTransform(spec.name, use_log1p=use_log1p).fit(yc[tr])
        save_json(tfm.to_dict(), out / "preprocessing" / f"target_scaler_{spec.name}.json")
        target_meta[spec.name] = {"use_log1p": use_log1p, "train_skew": skew}

        model, history, best_epoch, best_val = train_one_model(
            Xc[tr], tfm.transform(yc[tr]), Xc[va], tfm.transform(yc[va]),
            hp, device, init_seed=args.seed + ti + 1,
        )

        # checkpoint (weights + config + scalers needed for standalone inference)
        import torch
        torch.save({
            "state_dict": model.state_dict(),
            "model_config": model.config,
            "target_name": spec.name,
            "condition": spec.condition,
            "metric": spec.metric,
            "input_scaler": input_scaler.to_dict(),
            "target_transform": tfm.to_dict(),
            "best_epoch": best_epoch,
        }, out / "checkpoints" / f"gated_mlp_{spec.name}.pt")

        # loss history
        pd.DataFrame(history).to_csv(
            out / "plots" / f"loss_history_{spec.name}.csv", index=False)

        # predictions per split (original units, clamped >= 0: RMSE cannot be negative)
        split_pred = {}
        for split_name, m in (("train", tr), ("validation", va), ("test", te)):
            ypred = np.clip(tfm.inverse(_predict_scaled(model, Xc[m], device)), 0.0, None)
            ytrue = yc[m]
            split_pred[split_name] = (ytrue, ypred, sids[m], Xc[m])
            mt = P.scalar_metrics(ytrue, ypred)
            row = {"target": spec.name, "condition": spec.condition,
                   "metric": spec.metric, "split": split_name,
                   "use_log1p": use_log1p, "epochs_ran": len(history),
                   "best_epoch": best_epoch, **mt}
            per_model_rows.append(row)

            # prediction records
            dvals = np.array([sid_to_dist.get(s, 0.0) for s in sids[m]])
            for k in range(m.sum()):
                pred_records[split_name].append({
                    "sample_id": int(sids[m][k]), "split": split_name,
                    "condition": spec.condition, "metric": spec.metric,
                    "target": spec.name, "y_true": float(ytrue[k]),
                    "y_pred": float(ypred[k]),
                    "abs_error": float(abs(ypred[k] - ytrue[k])),
                    "nearest_train_distance": float(dvals[k]),
                })

        # baselines (train + ridge) evaluated on val & test
        for split_name, m in (("validation", va), ("test", te)):
            bp = baseline_predictions(Xc[tr], yc[tr], Xc[m])
            for bname, bpred in bp.items():
                bm = P.scalar_metrics(yc[m], np.clip(bpred, 0.0, None))
                baseline_rows.append({
                    "target": spec.name, "condition": spec.condition,
                    "metric": spec.metric, "split": split_name,
                    "baseline": bname, **bm})

        # distance-bin metrics on test
        ytrue_te, ypred_te, sids_te, _ = split_pred["test"]
        dist_te = np.array([sid_to_dist.get(s, 0.0) for s in sids_te])
        labels = P.distance_bin_labels(dist_te, edges)
        for b in ("near", "medium", "far"):
            bm = labels == b
            mt = P.scalar_metrics(ytrue_te[bm], ypred_te[bm])
            dist_bin_rows.append({
                "target": spec.name, "condition": spec.condition,
                "metric": spec.metric, "bin": b,
                "dist_lo": edges[0], "dist_hi": edges[1], **mt})

        _make_plots(out / "plots", spec.name, history, ytrue_te, ypred_te, dist_te)
        print(f"[{ti+1:2d}/12] {spec.name:32s} "
              f"test_rmse={[r for r in per_model_rows if r['target']==spec.name and r['split']=='test'][0]['rmse']:.4g} "
              f"test_r2={[r for r in per_model_rows if r['target']==spec.name and r['split']=='test'][0]['r2']:.4f} "
              f"log1p={use_log1p} epochs={len(history)}")

    # ---- write tables ----
    per_model_df = pd.DataFrame(per_model_rows)
    per_model_df.to_csv(out / "metrics_per_model.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out / "baseline_metrics.csv", index=False)
    pd.DataFrame(dist_bin_rows).to_csv(out / "distance_bin_metrics.csv", index=False)

    for split_name in ("train", "validation", "test"):
        pd.DataFrame(pred_records[split_name]).to_csv(
            out / "predictions" / f"predictions_{split_name}.csv", index=False)

    # ---- summary (macro averages over the 12 models) ----
    def macro(split):
        sub = per_model_df[per_model_df["split"] == split]
        return {k: float(sub[k].mean()) for k in ["rmse", "mae", "r2", "smape",
                                                  "bias", "max_abs_error", "pearson"]}
    summary = {
        "n_models": len(specs),
        "device": device, "seed": args.seed,
        "split_group_counts": {k: int(len(v)) for k, v in split_groups.items()},
        "split_row_counts": {
            "train": int((df["split"] == "train").sum()),
            "validation": int((df["split"] == "val").sum()),
            "test": int((df["split"] == "test").sum()),
        },
        "macro_average": {s: macro(s) for s in ("train", "validation", "test")},
        "distance_bin_edges": {"near<=": edges[0], "medium<=": edges[1]},
        "target_transforms": target_meta,
        "elapsed_s": time.time() - t0,
    }
    save_json(summary, out / "metrics_summary.json")

    # ---- config ----
    save_json({
        "data_dir": str(Path(args.data_dir).resolve()),
        "output_dir": str(out.resolve()),
        "seed": args.seed, "device": device,
        "input_columns": P.PARAM_COLUMNS,
        "log10_input_columns": P.LOG10_COLUMNS,
        "metrics": P.METRICS,
        "model_targets": [s.name for s in specs],
        "group_key": P.GROUP_KEY,
        "split_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "hyperparameters": hp,
    }, out / "config.json")

    _write_readme(out, summary, specs)
    _write_example(out, df, df_scaled_X, specs, input_scaler, device)
    print(f"\nDone in {summary['elapsed_s']:.1f}s. Artifacts -> {out}")


def _write_example(out, df, df_scaled_X, specs, input_scaler, device):
    """Predict all 12 errors for one held-out (test) parameter vector."""
    import torch

    test_sid = df[df["split"] == "test"][P.GROUP_KEY].iloc[0]
    row = df[df[P.GROUP_KEY] == test_sid].iloc[0]
    pvec = {c: float(row[c]) for c in P.PARAM_COLUMNS}
    x = input_scaler.transform(np.array([[pvec[c] for c in P.PARAM_COLUMNS]]))
    preds = {}
    truths = {}
    for spec in specs:
        ckpt = torch.load(out / "checkpoints" / f"gated_mlp_{spec.name}.pt",
                          map_location=device, weights_only=False)
        model = GatedMLP(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        tfm = TargetTransform.from_dict(ckpt["target_transform"])
        model.eval()
        with torch.no_grad():
            z = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
        preds[spec.name] = float(np.clip(tfm.inverse(z), 0.0, None)[0])
        truth_row = df[(df[P.GROUP_KEY] == test_sid) &
                       (df["case_id_code"] == spec.case_code)]
        truths[spec.name] = float(truth_row[spec.metric].iloc[0])
    save_json({
        "note": "Example prediction for one held-out TEST parameter vector.",
        "sample_id": int(test_sid),
        "parameter_vector": pvec,
        "predicted_errors": preds,
        "ground_truth_errors": truths,
    }, out / "example_unseen_prediction.json")


def _write_readme(out, summary, specs):
    macro_te = summary["macro_average"]["test"]
    lines = [
        "# Independent Gated-MLP RMSE surrogates (12 models)",
        "",
        "Twelve fully independent scalar Gated-MLP regressors, one per",
        "(discharge condition x error metric) target.  Each model has its own",
        "weights, optimizer, target scaler, early-stopping and checkpoint.",
        "",
        "## Targets",
        "",
        "| # | condition | metric |",
        "|---|-----------|--------|",
    ]
    for i, s in enumerate(specs, 1):
        lines.append(f"| {i} | {s.condition} | {s.metric} |")
    lines += [
        "",
        "## Split (leakage-safe, grouped by `sample_id`)",
        "",
        f"- groups: train={summary['split_group_counts']['train']}, "
        f"val={summary['split_group_counts']['val']}, "
        f"test={summary['split_group_counts']['test']}",
        f"- rows: train={summary['split_row_counts']['train']}, "
        f"val={summary['split_row_counts']['validation']}, "
        f"test={summary['split_row_counts']['test']}",
        "",
        "## Macro-average TEST metrics (original units, across 12 models)",
        "",
        f"- RMSE={macro_te['rmse']:.4g}, MAE={macro_te['mae']:.4g}, "
        f"R2={macro_te['r2']:.4f}, sMAPE={macro_te['smape']:.2f}%, "
        f"bias={macro_te['bias']:.4g}",
        "",
        "See `metrics_per_model.csv`, `metrics_summary.json`,",
        "`baseline_metrics.csv`, `distance_bin_metrics.csv`,",
        "`parameter_range_report.csv`, `split_assignments.csv` and `plots/`.",
        "",
        "## Inference",
        "",
        "```bash",
        "python -m src.gated_mlp_independent.predict_models \\",
        f"    --models-dir {out.name} \\",
        "    --vector '{\"Positive electrode reference diffusivity [m2.s-1]\": 1e-14, ...}'",
        "```",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", default="silu", choices=["silu", "gelu"])
    p.add_argument("--loss", default="smooth_l1", choices=["smooth_l1", "mse", "l1"])
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=80)
    p.add_argument("--lr-patience", type=int, default=30)
    p.add_argument("--lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--no-log1p", action="store_true",
                   help="disable automatic log1p target transform")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
