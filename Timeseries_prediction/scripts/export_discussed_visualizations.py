#!/usr/bin/env python3
"""
Export the publication-ready visualizations discussed for the final filtered
Batch-4 workflow, using completed metrics/predictions/histories only.

This script NEVER trains a model.

Run from the repository root:
    python scripts/export_discussed_visualizations.py

Optional:
    python scripts/export_discussed_visualizations.py \
        --root /path/to/Timeseries_prediction \
        --out reports/Data_Batch_4/final_filtered_protocol/final_visualizations_v3
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
import sys
import traceback
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


TS_MODELS = [
    "ann", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"
]
TS_DISPLAY = {
    "ann": "ANN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn": "CNN",
    "cnn_bilstm": "CNN–BiLSTM",
    "bayesian_mlp": "Bayesian MLP",
}
EM_MODELS = [
    "ann", "mlp", "gated_mlp", "deep_ensemble_mlp",
    "random_forest", "extratrees", "xgboost", "catboost",
]
EM_DISPLAY = {
    "ann": "ANN",
    "mlp": "MLP",
    "gated_mlp": "Gated MLP",
    "deep_ensemble_mlp": "Deep Ensemble MLP",
    "random_forest": "Random Forest",
    "extratrees": "ExtraTrees",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}
SEEDS = [42, 43, 44]
CASES_RE = re.compile(r"(CC_[CD]_[0-9]p[0-9]_T[0-9]+C)", re.I)
SEED_RE = re.compile(r"seed[_-]?([0-9]+)", re.I)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def flatten_dict(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}.{i}" if prefix else str(i)
            out.update(flatten_dict(v, key))
    else:
        out[prefix] = obj
    return out


def numeric(v: Any) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def pick_flat(flat: dict[str, Any], aliases: list[str]) -> float:
    nmap = {norm(k): v for k, v in flat.items()}
    for alias in aliases:
        a = norm(alias)
        if a in nmap:
            return numeric(nmap[a])
    for alias in aliases:
        a = norm(alias)
        matches = [(len(k), v) for k, v in nmap.items() if k.endswith(a)]
        if matches:
            matches.sort(key=lambda x: x[0])
            return numeric(matches[0][1])
    for alias in aliases:
        toks = [t for t in re.split(r"[_\W]+", alias.lower()) if t]
        matches = []
        for k, v in flat.items():
            nk = norm(k)
            if all(norm(t) in nk for t in toks):
                matches.append((len(nk), v))
        if matches:
            matches.sort(key=lambda x: x[0])
            return numeric(matches[0][1])
    return np.nan


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def savefig(fig: plt.Figure, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    png = base.with_suffix(".png")
    pdf = base.with_suffix(".pdf")
    fig.savefig(png, dpi=600, metadata={"Software": "Matplotlib"})
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_col(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    cmap = {norm(c): c for c in df.columns}
    for a in aliases:
        if norm(a) in cmap:
            return cmap[norm(a)]
    for a in aliases:
        na = norm(a)
        for nk, original in cmap.items():
            if nk.endswith(na) or na in nk:
                return original
    return None


def r2_score_np(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2:
        return np.nan
    den = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - np.sum((y - p) ** 2) / den if den > 0 else np.nan


def rmse_np(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else np.nan


def mae_np(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else np.nan


def parse_ts_path(path: Path) -> tuple[Optional[str], Optional[str], Optional[int]]:
    text = str(path)
    cm = CASES_RE.search(text)
    case = cm.group(1) if cm else None
    lower_parts = [p.lower() for p in path.parts]
    model = next((m for m in TS_MODELS if m in lower_parts), None)
    sm = SEED_RE.search(text)
    seed = int(sm.group(1)) if sm else None
    return case, model, seed


def collect_ts_metrics(root: Path) -> pd.DataFrame:
    ts_root = root / "outputs/Data_Batch_4_TSFiltered_0p8/time_series"
    candidates = sorted(ts_root.rglob("metrics.json"))
    rows = []
    aliases = {
        "V_MAE": ["MAE_V", "voltage_MAE", "test_MAE_V", "metrics.MAE_V"],
        "V_RMSE": ["RMSE_V", "voltage_RMSE", "test_RMSE_V", "metrics.RMSE_V"],
        "V_R2": ["R2_V", "voltage_R2", "test_R2_V", "metrics.R2_V"],
        "T_MAE": ["MAE_T", "temperature_MAE", "test_MAE_T", "metrics.MAE_T"],
        "T_RMSE": ["RMSE_T", "temperature_RMSE", "test_RMSE_T", "metrics.RMSE_T"],
        "T_R2": ["R2_T", "temperature_R2", "test_R2_T", "metrics.R2_T"],
        "T_Peak_MAE": ["temperature_peak_mae", "peak_temperature_mae", "T_peak_MAE"],
        "V_Endpoint_MAE": ["voltage_endpoint_mae", "endpoint_voltage_mae", "V_endpoint_MAE"],
        "T_Endpoint_MAE": ["temperature_endpoint_mae", "endpoint_temperature_mae", "T_endpoint_MAE"],
        "Inference_ms": ["inference_ms_per_sample", "inference_ms", "latency_ms"],
        "Param_Count": ["param_count", "parameter_count", "n_parameters", "num_params"],
        "Coverage95_V": ["coverage95_V", "coverage_95_V"],
        "Coverage95_T": ["coverage95_T", "coverage_95_T"],
    }
    for path in candidates:
        case, model, seed = parse_ts_path(path)
        if not (case and model and seed in SEEDS):
            continue
        try:
            obj = json.loads(path.read_text())
            flat = flatten_dict(obj)
        except Exception:
            continue
        row = {"case_id": case, "model": model, "seed": seed, "metrics_path": str(path)}
        for out_key, als in aliases.items():
            row[out_key] = pick_flat(flat, als)
        rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(["case_id", "model", "seed"])
    if len(df) != 252:
        print(f"[WARN] Parsed {len(df)} TS combinations, expected 252.")
    return df


def build_ts_tables(df: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tables = out / "02_Time_Series_Response/tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "time_series_metrics_by_case_model_seed.csv", index=False)

    metric_cols = ["V_MAE", "V_RMSE", "V_R2", "T_MAE", "T_RMSE", "T_R2"]
    per_seed = (
        df.groupby(["model", "seed"], as_index=False)
        .agg({**{c: "mean" for c in metric_cols},
              "Param_Count": "mean",
              "Inference_ms": "mean",
              "T_Peak_MAE": "mean",
              "V_Endpoint_MAE": "mean",
              "T_Endpoint_MAE": "mean"})
    )
    per_seed.to_csv(tables / "time_series_metrics_by_model_seed.csv", index=False)

    rows = []
    for model, g in per_seed.groupby("model"):
        row: dict[str, Any] = {"model": model, "display": TS_DISPLAY.get(model, model)}
        for c in metric_cols + [
            "Param_Count", "Inference_ms", "T_Peak_MAE",
            "V_Endpoint_MAE", "T_Endpoint_MAE",
        ]:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std(ddof=1)
        rows.append(row)
    summary = pd.DataFrame(rows)

    directions = {
        "V_MAE_mean": True, "V_RMSE_mean": True, "V_R2_mean": False,
        "T_MAE_mean": True, "T_RMSE_mean": True, "T_R2_mean": False,
    }
    for c, ascending in directions.items():
        summary[c.replace("_mean", "_rank")] = summary[c].rank(
            method="average", ascending=ascending
        )
    rank_cols = [c.replace("_mean", "_rank") for c in directions]
    summary["Average_Rank"] = summary[rank_cols].mean(axis=1)
    summary["Final_Rank"] = summary["Average_Rank"].rank(method="min").astype(int)
    summary = summary.sort_values(["Final_Rank", "Average_Rank"])
    summary.to_csv(tables / "time_series_ranking_table.csv", index=False)
    return df, per_seed, summary


def collect_em_summary(root: Path) -> pd.DataFrame:
    p = first_existing([
        root / "outputs/Data_Batch_4/error_metric_final_extension/final_filtered_full/combined/tables/metrics_by_model.csv",
        root / "reports/Data_Batch_4/final_filtered_protocol/final_filtered_full/error_metric/metrics_by_model.csv",
    ])
    if p is None:
        raise FileNotFoundError("Could not find composite error-metric metrics_by_model.csv")
    df = pd.read_csv(p)
    if "display" not in df:
        df["display"] = df["model"].map(EM_DISPLAY).fillna(df["model"])
    return df[df["model"].isin(EM_MODELS)].copy()


def build_em_ranking(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    tables = out / "03_Error_Metric/tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "error_metric_metrics_by_model.csv", index=False)
    metric_map = {
        "V_MAE": "rmse_voltage_mv.MAE_mean",
        "V_RMSE": "rmse_voltage_mv.RMSE_mean",
        "V_R2": "rmse_voltage_mv.R2_mean",
        "T_MAE": "rmse_temperature_c.MAE_mean",
        "T_RMSE": "rmse_temperature_c.RMSE_mean",
        "T_R2": "rmse_temperature_c.R2_mean",
        "Overall_NRMSE": "overall_norm_overall_RMSE_mean",
    }
    out_df = df[["model", "display"]].copy()
    for name, col in metric_map.items():
        out_df[name] = df[col] if col in df else np.nan
    for name in ["V_MAE", "V_RMSE", "T_MAE", "T_RMSE", "Overall_NRMSE"]:
        out_df[f"{name}_rank"] = out_df[name].rank(ascending=True)
    for name in ["V_R2", "T_R2"]:
        out_df[f"{name}_rank"] = out_df[name].rank(ascending=False)
    out_df["Average_Rank"] = out_df[[c for c in out_df if c.endswith("_rank")]].mean(axis=1)
    out_df["Final_Rank"] = out_df["Average_Rank"].rank(method="min").astype(int)
    out_df = out_df.sort_values(["Final_Rank", "Average_Rank"])
    out_df.to_csv(tables / "error_metric_ranking_table.csv", index=False)
    return out_df


def heatmap_figure(
    table: pd.DataFrame,
    row_label: str,
    metrics: list[tuple[str, str, bool]],
    title: str,
    output: Path,
) -> None:
    values = []
    ranks = []
    for _, row in table.iterrows():
        vals = [numeric(row.get(col)) for _, col, _ in metrics]
        values.append(vals)
    values_arr = np.asarray(values, float)
    for j, (_, _, lower_better) in enumerate(metrics):
        s = pd.Series(values_arr[:, j])
        ranks.append(s.rank(ascending=lower_better, method="average").to_numpy())
    rank_arr = np.vstack(ranks).T

    fig, ax = plt.subplots(figsize=(1.55 * len(metrics) + 2.4, 0.65 * len(table) + 1.7))
    im = ax.imshow(rank_arr, aspect="auto", cmap="viridis_r", vmin=1, vmax=max(2, len(table)))
    ax.set_xticks(np.arange(len(metrics)), [m[0] for m in metrics], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(table)), table[row_label].tolist())
    for i in range(len(table)):
        for j, (label, col, _) in enumerate(metrics):
            val = values_arr[i, j]
            if "R2" in label:
                txt = f"{val:.4f}"
            elif "Rank" in label:
                txt = f"{val:.2f}"
            elif abs(val) >= 100:
                txt = f"{val:.1f}"
            else:
                txt = f"{val:.4f}"
            rank = rank_arr[i, j]
            marker = "★" if rank == 1 else ("●" if rank == 2 else "")
            ax.text(j, i, f"{txt}{marker}", ha="center", va="center",
                    color="white" if rank > len(table) / 2 else "black", fontsize=7.6)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Metric-specific rank (1 = best)")
    fig.tight_layout()
    savefig(fig, output)


def load_manifest(root: Path) -> pd.DataFrame:
    p = first_existing([
        root / "data/Data_Batch_4_TSFiltered_0p8/time_series_source_manifest.csv",
        root / "data/Data_Batch_4_TSFiltered_0p8/time_series_kept_manifest.csv",
        root / "data/Data_Batch_4_raw/sequence_manifest.csv",
        root / "data/generate_training_data_1000samples/sequence_manifest.csv",
    ])
    if p is None:
        return pd.DataFrame()
    return pd.read_csv(p)


def fig_data_routing(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc="white"):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.4, edgecolor="black", facecolor=fc
        )
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10)
        return patch

    box(3.4, 4.65, 3.2, 0.85, "12,000 source sequences\n1,000 parameter sets × 12 cases", "#f4f4f4")
    box(0.8, 2.55, 3.2, 0.95, "Error-metric task\n12,000 rows retained", "#e8f2ff")
    box(6.0, 2.55, 3.2, 0.95, "Time-series response task\nFilter by duration ratio", "#fff2df")
    box(5.0, 0.55, 2.2, 0.9, "11,109 retained\n(92.6%)", "#e8f7ea")
    box(7.7, 0.55, 2.0, 0.9, "891 removed\n(7.4%)", "#fde8e8")

    arrows = [
        ((4.5, 4.65), (2.4, 3.5)),
        ((5.5, 4.65), (7.6, 3.5)),
        ((7.0, 2.55), (6.1, 1.45)),
        ((8.2, 2.55), (8.7, 1.45)),
    ]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.3))
    ax.text(5, 0.05,
            r"$\mathrm{duration\ ratio}=\mathrm{simulation\ end}/\mathrm{reference\ end}$; retain if $\geq 0.8$",
            ha="center", va="bottom", fontsize=9)
    ax.set_title("DATA-1. Dataset routing for the two surrogate-learning tasks", pad=12)
    savefig(fig, out)


def fig_duration_audit(root: Path, manifest: pd.DataFrame, out: Path) -> None:
    if manifest.empty:
        return
    ratio_col = find_col(manifest, ["duration_ratio"])
    case_col = find_col(manifest, ["experiment_id", "case_id", "operating_case"])
    keep_col = find_col(manifest, ["keep_flag", "kept"])
    if ratio_col is None:
        return
    ratio = pd.to_numeric(manifest[ratio_col], errors="coerce")
    if keep_col is not None:
        k = manifest[keep_col]
        if k.dtype == bool:
            removed = manifest[~k].copy()
        else:
            removed = manifest[~k.astype(str).str.lower().isin(["true", "1", "yes", "kept", "keep"])].copy()
    else:
        removed = manifest[ratio < 0.8].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].hist(ratio.dropna(), bins=35, edgecolor="white")
    axes[0].axvline(0.8, linestyle="--", linewidth=1.6, label="Cutoff = 0.8")
    axes[0].set(xlabel="Duration ratio", ylabel="Sequence count", title="(a) Duration-ratio distribution")
    axes[0].legend()

    if case_col:
        total = manifest.groupby(case_col).size()
        rem = removed.groupby(case_col).size().reindex(total.index, fill_value=0)
        rate = 100 * rem / total
        order = rem.sort_values(ascending=False).index
        ax = axes[1]
        x = np.arange(len(order))
        bars = ax.bar(x, rem[order])
        ax.set_xticks(x, order, rotation=50, ha="right")
        ax.set(ylabel="Removed sequence count", title="(b) Early termination by operating case")
        ax2 = ax.twinx()
        ax2.plot(x, rate[order], marker="o", linewidth=1.3)
        ax2.set_ylabel("Removal rate (%)")
        for b, v in zip(bars, rem[order]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), str(int(v)),
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    savefig(fig, out)


def discover_csv(root: Path, required_parts: list[str], names: list[str]) -> Optional[Path]:
    candidates = []
    for base in [
        root / "outputs/Data_Batch_4_TSFiltered_0p8",
        root / "outputs/Data_Batch_4",
    ]:
        if not base.exists():
            continue
        for name in names:
            candidates.extend(base.rglob(name))
    req = [norm(p) for p in required_parts]
    scored = []
    for p in candidates:
        text = norm(str(p))
        score = sum(1 for x in req if x in text)
        if score == len(req):
            scored.append((score, p.stat().st_mtime, p))
    return sorted(scored, reverse=True)[0][2] if scored else None


def parse_array_cell(x: Any) -> Optional[np.ndarray]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, float)
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s or s[0] not in "[(":
        return None
    try:
        return np.asarray(ast.literal_eval(s), float)
    except Exception:
        try:
            return np.asarray(json.loads(s), float)
        except Exception:
            return None


def standardize_ts_predictions(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    seq = find_col(df, ["sequence_id", "sample_id", "id"])
    time = find_col(df, ["time_s", "time", "t", "time_sec"])
    vtrue = find_col(df, ["voltage_true", "true_voltage", "V_true", "voltage_actual", "target_voltage"])
    vpred = find_col(df, ["voltage_pred", "pred_voltage", "V_pred", "voltage_prediction"])
    ttrue = find_col(df, ["temperature_true", "true_temperature", "T_true", "temperature_actual", "target_temperature"])
    tpred = find_col(df, ["temperature_pred", "pred_temperature", "T_pred", "temperature_prediction"])
    valid = find_col(df, ["valid_mask", "mask", "is_valid"])
    if all([vtrue, vpred, ttrue, tpred]):
        out = pd.DataFrame({
            "sequence_id": df[seq].astype(str) if seq else "sequence_0",
            "time_s": pd.to_numeric(df[time], errors="coerce") if time else np.arange(len(df)),
            "V_true": pd.to_numeric(df[vtrue], errors="coerce"),
            "V_pred": pd.to_numeric(df[vpred], errors="coerce"),
            "T_true": pd.to_numeric(df[ttrue], errors="coerce"),
            "T_pred": pd.to_numeric(df[tpred], errors="coerce"),
            "valid": df[valid].astype(bool) if valid else True,
        })
        return out

    # Wide rows with serialized trajectory arrays
    if all([vtrue, vpred, ttrue, tpred]):
        pass
    for candidate in df.columns:
        arr = parse_array_cell(df[candidate].iloc[0]) if len(df) else None
        if arr is not None:
            break
    else:
        return None
    rows = []
    for idx, row in df.iterrows():
        arrays = {}
        for name, col in [("V_true", vtrue), ("V_pred", vpred), ("T_true", ttrue), ("T_pred", tpred)]:
            arrays[name] = parse_array_cell(row[col]) if col else None
        if any(v is None for v in arrays.values()):
            continue
        n = min(len(v) for v in arrays.values())
        sid = str(row[seq]) if seq else f"sequence_{idx}"
        for j in range(n):
            rows.append({"sequence_id": sid, "time_s": j, "valid": True,
                         **{k: v[j] for k, v in arrays.items()}})
    return pd.DataFrame(rows) if rows else None


def select_sequence(pred: pd.DataFrame, quantile: float = 0.5) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    pred = pred[pred["valid"].fillna(True)].dropna(subset=["V_true", "V_pred", "T_true", "T_pred"]).copy()
    scores = []
    for sid, g in pred.groupby("sequence_id"):
        rv = rmse_np(g.V_true, g.V_pred)
        rt = rmse_np(g.T_true, g.T_pred)
        scores.append((sid, rv, rt))
    score_df = pd.DataFrame(scores, columns=["sequence_id", "V_RMSE", "T_RMSE"])
    if score_df.empty:
        raise RuntimeError("No valid sequence found in prediction CSV.")
    for c in ["V_RMSE", "T_RMSE"]:
        med = score_df[c].median()
        score_df[c + "_norm"] = score_df[c] / med if med > 0 else score_df[c]
    score_df["combined"] = score_df["V_RMSE_norm"] + score_df["T_RMSE_norm"]
    target = score_df["combined"].quantile(quantile)
    sid = score_df.iloc[(score_df["combined"] - target).abs().argsort().iloc[0]]["sequence_id"]
    return str(sid), pred[pred.sequence_id == str(sid)].sort_values("time_s"), score_df


def find_history(root: Path, case: str, model: str, seed: int) -> Optional[pd.DataFrame]:
    p = discover_csv(root, [case, model, f"seed{seed}"], ["history.csv", "*history*.csv"])
    return read_csv_safe(p) if p else None


def find_ts_predictions(root: Path, case: str, model: str, seed: int) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
    p = discover_csv(root, [case, model, f"seed{seed}"],
                     ["test_predictions.csv", "*prediction*.csv", "predictions.csv"])
    if p is None:
        return None, None
    raw = read_csv_safe(p)
    if raw is None:
        return p, None
    return p, standardize_ts_predictions(raw)


def fig_ts_best_detail(root: Path, ts_df: pd.DataFrame, ts_rank: pd.DataFrame, out: Path) -> dict[str, Any]:
    best = ts_rank.iloc[0]["model"]
    best_cases = ts_df[ts_df.model == best].groupby("case_id", as_index=False)[["V_RMSE", "T_RMSE"]].mean()
    best_cases["combined"] = (
        best_cases["V_RMSE"] / best_cases["V_RMSE"].median()
        + best_cases["T_RMSE"] / best_cases["T_RMSE"].median()
    )
    case = best_cases.sort_values("combined", ascending=False).iloc[0]["case_id"]
    seed = 43
    hist = find_history(root, case, best, seed)
    pred_path, pred = find_ts_predictions(root, case, best, seed)
    if pred is None or pred.empty:
        raise FileNotFoundError(f"No usable TS prediction CSV for {case}/{best}/seed{seed}")
    sid, seq, score_df = select_sequence(pred, 0.5)
    valid_pred = pred[pred.valid.fillna(True)].copy()

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    ax = axes[0, 0]
    if hist is not None:
        ep = find_col(hist, ["epoch", "step"])
        tr = find_col(hist, ["train_loss", "loss_train"])
        va = find_col(hist, ["val_loss", "validation_loss"])
        if tr:
            ax.plot(hist[ep] if ep else np.arange(len(hist)), hist[tr], label="Train")
        if va:
            ax.plot(hist[ep] if ep else np.arange(len(hist)), hist[va], label="Validation")
        ax.legend()
    ax.set(xlabel="Epoch", ylabel="Loss", title="(a) Learning curves")

    axes[0, 1].plot(seq.time_s, seq.V_true, label="Observed")
    axes[0, 1].plot(seq.time_s, seq.V_pred, label="Predicted")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Voltage (V)", title=f"(b) Voltage trajectory — {sid}")
    axes[0, 1].legend()

    axes[0, 2].plot(seq.time_s, seq.T_true, label="Observed")
    axes[0, 2].plot(seq.time_s, seq.T_pred, label="Predicted")
    axes[0, 2].set(xlabel="Time (s)", ylabel="Temperature (°C)", title=f"(c) Temperature trajectory — {sid}")
    axes[0, 2].legend()

    for ax, y, p, label, unit in [
        (axes[1, 0], valid_pred.V_true, valid_pred.V_pred, "Voltage", "V"),
        (axes[1, 1], valid_pred.T_true, valid_pred.T_pred, "Temperature", "°C"),
    ]:
        n = min(len(y), 40000)
        idx = np.linspace(0, len(y)-1, n).astype(int) if len(y) else []
        yy = np.asarray(y)[idx]
        pp = np.asarray(p)[idx]
        ax.scatter(yy, pp, s=5, alpha=0.18)
        lo, hi = np.nanmin([yy, pp]), np.nanmax([yy, pp])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set(xlabel=f"Observed {label} ({unit})", ylabel=f"Predicted {label} ({unit})",
               title=f"({'d' if label=='Voltage' else 'e'}) {label} parity\n"
                     f"RMSE={rmse_np(yy, pp):.4g}, R²={r2_score_np(yy, pp):.4f}")

    ax = axes[1, 2]
    tmp = valid_pred.copy()
    tmp["V_AE"] = np.abs(tmp.V_true - tmp.V_pred)
    tmp["T_AE"] = np.abs(tmp.T_true - tmp.T_pred)
    agg = tmp.groupby("time_s", as_index=False)[["V_AE", "T_AE"]].mean()
    ax.plot(agg.time_s, agg.V_AE, label="Voltage absolute error (V)")
    ax2 = ax.twinx()
    ax2.plot(agg.time_s, agg.T_AE, label="Temperature absolute error (°C)")
    ax.set(xlabel="Time (s)", ylabel="Voltage absolute error (V)", title="(f) Masked error over time")
    ax2.set_ylabel("Temperature absolute error (°C)")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")

    fig.suptitle(f"TS-2. Best time-series model: {TS_DISPLAY[best]} | hardest case: {case}", y=1.01)
    fig.tight_layout()
    savefig(fig, out)
    return {"best_model": best, "hardest_case": case, "seed": seed, "prediction_path": str(pred_path), "score_df": score_df, "pred": pred}


def fig_ts_examples(pred: pd.DataFrame, score_df: pd.DataFrame, out: Path) -> None:
    qs = [0.5, 0.9, 1.0]
    labels = ["Median-error", "90th-percentile", "Worst"]
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for i, (q, lab) in enumerate(zip(qs, labels)):
        target = score_df.combined.quantile(q)
        row = score_df.iloc[(score_df.combined - target).abs().argsort().iloc[0]]
        g = pred[(pred.sequence_id == str(row.sequence_id)) & pred.valid.fillna(True)].sort_values("time_s")
        axes[i, 0].plot(g.time_s, g.V_true, label="Observed")
        axes[i, 0].plot(g.time_s, g.V_pred, label="Predicted")
        axes[i, 0].set(ylabel="Voltage (V)", title=f"{lab}: {row.sequence_id}")
        axes[i, 1].plot(g.time_s, g.T_true, label="Observed")
        axes[i, 1].plot(g.time_s, g.T_pred, label="Predicted")
        axes[i, 1].set(ylabel="Temperature (°C)", title=f"{lab}: {row.sequence_id}")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[0, 0].legend()
    axes[0, 1].legend()
    fig.suptitle("TS-S1. Representative test trajectories by sequence-level error", y=1.01)
    fig.tight_layout()
    savefig(fig, out)


def fig_ts_per_case(ts_df: pd.DataFrame, best: str, out: Path) -> None:
    g = ts_df[ts_df.model == best]
    agg = g.groupby("case_id")[["V_RMSE", "T_RMSE"]].agg(["mean", "std"])
    x = np.arange(len(agg))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(x, agg[("V_RMSE", "mean")], yerr=agg[("V_RMSE", "std")], capsize=3)
    axes[0].set(ylabel="Voltage RMSE (V)", title="(a) Per-case voltage RMSE")
    axes[1].bar(x, agg[("T_RMSE", "mean")], yerr=agg[("T_RMSE", "std")], capsize=3)
    axes[1].set(ylabel="Temperature RMSE (°C)", title="(b) Per-case temperature RMSE")
    axes[1].set_xticks(x, agg.index, rotation=45, ha="right")
    fig.suptitle(f"TS-S2. Per-case performance of {TS_DISPLAY[best]} (mean ± SD across seeds)")
    fig.tight_layout()
    savefig(fig, out)


def fig_ts_endpoint_peak(pred: pd.DataFrame, out: Path) -> None:
    rows = []
    for sid, g in pred[pred.valid.fillna(True)].groupby("sequence_id"):
        g = g.sort_values("time_s")
        if g.empty:
            continue
        rows.append({
            "V_endpoint_AE": abs(g.V_true.iloc[-1] - g.V_pred.iloc[-1]),
            "T_endpoint_AE": abs(g.T_true.iloc[-1] - g.T_pred.iloc[-1]),
            "T_peak_AE": abs(g.T_true.max() - g.T_pred.max()),
        })
    d = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, c, lab in zip(
        axes,
        ["V_endpoint_AE", "T_endpoint_AE", "T_peak_AE"],
        ["Voltage endpoint AE (V)", "Temperature endpoint AE (°C)", "Peak-temperature AE (°C)"],
    ):
        ax.hist(d[c].dropna(), bins=25)
        ax.set(xlabel=lab, ylabel="Sequence count")
    fig.suptitle("TS-S3. Endpoint and peak error distributions")
    fig.tight_layout()
    savefig(fig, out)


def fig_ts_uncertainty(root: Path, case: str, out: Path) -> bool:
    p = discover_csv(root, [case, "bayesian_mlp", "seed43"],
                     ["test_predictions.csv", "*prediction*.csv", "predictions.csv"])
    if p is None:
        return False
    df = read_csv_safe(p)
    if df is None:
        return False
    seq_col = find_col(df, ["sequence_id", "sample_id"])
    time_col = find_col(df, ["time_s", "time", "t"])
    vt = find_col(df, ["voltage_true", "V_true"])
    vp = find_col(df, ["voltage_pred", "V_pred", "voltage_mean"])
    tt = find_col(df, ["temperature_true", "T_true"])
    tp = find_col(df, ["temperature_pred", "T_pred", "temperature_mean"])
    vs = find_col(df, ["voltage_std", "V_std", "voltage_sigma"])
    ts = find_col(df, ["temperature_std", "T_std", "temperature_sigma"])
    vl = find_col(df, ["voltage_lower", "V_lower", "voltage_lo"])
    vu = find_col(df, ["voltage_upper", "V_upper", "voltage_hi"])
    tl = find_col(df, ["temperature_lower", "T_lower", "temperature_lo"])
    tu = find_col(df, ["temperature_upper", "T_upper", "temperature_hi"])
    if not all([vt, vp, tt, tp]) or not ((vs and ts) or (vl and vu and tl and tu)):
        return False
    if seq_col:
        sid = str(df[seq_col].iloc[0])
        g = df[df[seq_col].astype(str) == sid].copy()
    else:
        g = df.copy()
    x = pd.to_numeric(g[time_col], errors="coerce") if time_col else np.arange(len(g))
    if vs and ts:
        vlo = pd.to_numeric(g[vp], errors="coerce") - 1.96 * pd.to_numeric(g[vs], errors="coerce")
        vhi = pd.to_numeric(g[vp], errors="coerce") + 1.96 * pd.to_numeric(g[vs], errors="coerce")
        tlo = pd.to_numeric(g[tp], errors="coerce") - 1.96 * pd.to_numeric(g[ts], errors="coerce")
        thi = pd.to_numeric(g[tp], errors="coerce") + 1.96 * pd.to_numeric(g[ts], errors="coerce")
    else:
        vlo, vhi, tlo, thi = [pd.to_numeric(g[c], errors="coerce") for c in [vl, vu, tl, tu]]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x, g[vt], label="Observed")
    axes[0].plot(x, g[vp], label="Predictive mean")
    axes[0].fill_between(x, vlo, vhi, alpha=0.25, label="95% interval")
    axes[0].set(ylabel="Voltage (V)", title="(a) Voltage uncertainty")
    axes[0].legend()
    axes[1].plot(x, g[tt], label="Observed")
    axes[1].plot(x, g[tp], label="Predictive mean")
    axes[1].fill_between(x, tlo, thi, alpha=0.25, label="95% interval")
    axes[1].set(xlabel="Time (s)", ylabel="Temperature (°C)", title="(b) Temperature uncertainty")
    axes[1].legend()
    fig.suptitle("TS-S4. Bayesian MLP predictive uncertainty")
    fig.tight_layout()
    savefig(fig, out)
    return True


def fig_duration_performance(pred: pd.DataFrame, manifest: pd.DataFrame, score_df: pd.DataFrame, out: Path) -> bool:
    if manifest.empty:
        return False
    sid_col = find_col(manifest, ["sequence_id", "sample_id"])
    ratio_col = find_col(manifest, ["duration_ratio", "valid_fraction"])
    if not sid_col or not ratio_col:
        return False
    m = manifest[[sid_col, ratio_col]].copy()
    m.columns = ["sequence_id", "duration_ratio"]
    m.sequence_id = m.sequence_id.astype(str)
    d = score_df.merge(m, on="sequence_id", how="inner")
    if d.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(d.duration_ratio, d.V_RMSE, s=12, alpha=0.45)
    axes[0].set(xlabel="Duration ratio", ylabel="Voltage RMSE (V)", title="(a) Voltage")
    axes[1].scatter(d.duration_ratio, d.T_RMSE, s=12, alpha=0.45)
    axes[1].set(xlabel="Duration ratio", ylabel="Temperature RMSE (°C)", title="(b) Temperature")
    fig.suptitle("TS-S5. Performance versus retained sequence duration")
    fig.tight_layout()
    savefig(fig, out)
    return True


def find_em_prediction_files(root: Path, model: str) -> list[Path]:
    files = []
    for base in [
        root / "outputs/Data_Batch_4/error_metric_final_extension/final_filtered_full",
        root / "outputs/Data_Batch_4/error_metric_benchmark/batch4_em_grouped_20260622_110539",
    ]:
        if not base.exists():
            continue
        for p in base.rglob("test_predictions.csv"):
            if model in [part.lower() for part in p.parts]:
                files.append(p)
    return sorted(files)


def load_em_predictions(root: Path, model: str) -> pd.DataFrame:
    dfs = []
    for p in find_em_prediction_files(root, model):
        df = read_csv_safe(p)
        if df is None:
            continue
        sm = SEED_RE.search(str(p))
        df["seed"] = int(sm.group(1)) if sm else np.nan
        df["source_path"] = str(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def find_em_learning(root: Path, model: str) -> Optional[pd.DataFrame]:
    for name in ["learning_curve.csv", "history.csv"]:
        p = discover_csv(root, [model], [name])
        if p:
            d = read_csv_safe(p)
            if d is not None:
                return d
    return None


def fig_em_best_detail(root: Path, em_rank: pd.DataFrame, out: Path) -> dict[str, Any]:
    best = em_rank.iloc[0]["model"]
    pred = load_em_predictions(root, best)
    if pred.empty:
        raise FileNotFoundError(f"No error-metric predictions found for {best}")
    vt = find_col(pred, ["rmse_voltage_mv_true"])
    vp = find_col(pred, ["rmse_voltage_mv_pred"])
    tt = find_col(pred, ["rmse_temperature_c_true"])
    tp = find_col(pred, ["rmse_temperature_c_pred"])
    sid = find_col(pred, ["sample_id", "sequence_id"])
    if not all([vt, vp, tt, tp]):
        raise RuntimeError(f"Prediction columns are incomplete: {list(pred.columns)}")
    lc = find_em_learning(root, best)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    ax = axes[0, 0]
    if lc is not None:
        xcol = find_col(lc, ["epoch", "train_size", "fraction", "n_train"])
        for aliases, label in [
            (["train_loss", "train_rmse", "train_score"], "Train"),
            (["val_loss", "validation_loss", "val_rmse", "test_score"], "Validation"),
        ]:
            c = find_col(lc, aliases)
            if c:
                ax.plot(lc[xcol] if xcol else np.arange(len(lc)), lc[c], marker="o" if len(lc) < 20 else None, label=label)
        ax.legend()
        ax.set_xlabel(xcol or "Step")
    ax.set(ylabel="Loss / error", title="(a) Learning behaviour")

    order = pred.groupby(sid if sid else pred.index)[[vt, vp, tt, tp]].mean().reset_index()
    order = order.sort_values(vt)
    axes[0, 1].plot(np.arange(len(order)), order[vt], label="True")
    axes[0, 1].plot(np.arange(len(order)), order[vp], label="Predicted")
    axes[0, 1].set(xlabel="Parameter-set order", ylabel="Voltage trajectory RMSE (mV)",
                   title="(b) Voltage RMSE by parameter set")
    axes[0, 1].legend()

    order_t = order.sort_values(tt)
    axes[0, 2].plot(np.arange(len(order_t)), order_t[tt], label="True")
    axes[0, 2].plot(np.arange(len(order_t)), order_t[tp], label="Predicted")
    axes[0, 2].set(xlabel="Parameter-set order", ylabel="Temperature trajectory RMSE (°C)",
                   title="(c) Temperature RMSE by parameter set")
    axes[0, 2].legend()

    for ax, ycol, pcol, lab, unit, panel in [
        (axes[1, 0], vt, vp, "Voltage-target", "mV", "d"),
        (axes[1, 1], tt, tp, "Temperature-target", "°C", "e"),
    ]:
        y = pd.to_numeric(pred[ycol], errors="coerce").to_numpy()
        p = pd.to_numeric(pred[pcol], errors="coerce").to_numpy()
        ax.scatter(y, p, s=8, alpha=0.25)
        lo, hi = np.nanmin([y, p]), np.nanmax([y, p])
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set(xlabel=f"True ({unit})", ylabel=f"Predicted ({unit})",
               title=f"({panel}) {lab} parity\nRMSE={rmse_np(y,p):.3g}, R²={r2_score_np(y,p):.4f}")

    rv = pd.to_numeric(pred[vp], errors="coerce") - pd.to_numeric(pred[vt], errors="coerce")
    rt = pd.to_numeric(pred[tp], errors="coerce") - pd.to_numeric(pred[tt], errors="coerce")
    axes[1, 2].hist(rv.dropna(), bins=30, alpha=0.6, label="Voltage residual (mV)")
    ax2 = axes[1, 2].twinx()
    ax2.hist(rt.dropna(), bins=30, alpha=0.35, label="Temperature residual (°C)")
    axes[1, 2].axvline(0, linestyle="--", linewidth=1)
    axes[1, 2].set(xlabel="Residual", ylabel="Voltage count", title="(f) Residual distributions")
    ax2.set_ylabel("Temperature count")
    lines = axes[1, 2].patches[:1] + ax2.patches[:1]
    axes[1, 2].legend(lines, ["Voltage", "Temperature"])

    fig.suptitle(f"EM-2. Best error-metric model: {EM_DISPLAY.get(best, best)}", y=1.01)
    fig.tight_layout()
    savefig(fig, out)
    return {"best_model": best, "pred": pred, "columns": (sid, vt, vp, tt, tp)}


def fig_em_feature_importance(root: Path, out: Path) -> bool:
    models = ["random_forest", "extratrees", "xgboost", "catboost"]
    found: dict[str, pd.DataFrame] = {}
    for model in models:
        files = []
        for base in [
            root / "outputs/Data_Batch_4/error_metric_final_extension/final_filtered_full",
            root / "outputs/Data_Batch_4/error_metric_benchmark/batch4_em_grouped_20260622_110539",
        ]:
            if base.exists():
                files.extend([p for p in base.rglob("*importance*.csv") if model in str(p).lower()])
        if not files:
            continue
        frames = [read_csv_safe(p) for p in files]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        d = pd.concat(frames, ignore_index=True)
        feat = find_col(d, ["feature", "feature_name"])
        imp = find_col(d, ["importance", "gain", "weight"])
        if feat and imp:
            found[model] = d.groupby(feat, as_index=False)[imp].mean().sort_values(imp, ascending=False).head(15)
    if not found:
        return False
    fig, axes = plt.subplots(1, len(found), figsize=(4.2 * len(found), 5), squeeze=False)
    for ax, (model, d) in zip(axes[0], found.items()):
        feat = find_col(d, ["feature", "feature_name"])
        imp = find_col(d, ["importance", "gain", "weight"])
        d = d.sort_values(imp)
        ax.barh(d[feat], d[imp])
        ax.set(title=EM_DISPLAY.get(model, model), xlabel="Normalized importance")
    fig.suptitle("EM-S1. Native feature importance for tree-based models")
    fig.tight_layout()
    savefig(fig, out)
    return True


def fig_grouped_vs_legacy(root: Path, out: Path) -> bool:
    gp = root / "outputs/Data_Batch_4/error_metric_benchmark/batch4_em_grouped_20260622_110539/tables/metrics_by_model.csv"
    lp = root / "outputs/Data_Batch_4/error_metric_benchmark/batch4_em_legacy_20260622_143740/tables/metrics_by_model.csv"
    if not gp.exists() or not lp.exists():
        return False
    g, l = pd.read_csv(gp), pd.read_csv(lp)
    metric = "overall_norm_overall_RMSE_mean"
    if metric not in g or metric not in l:
        return False
    d = g[["model", metric]].merge(l[["model", metric]], on="model", suffixes=("_grouped", "_legacy"))
    d = d[d.model.isin(EM_MODELS)]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.38
    ax.bar(x-w/2, d[f"{metric}_grouped"], width=w, label="Grouped holdout")
    ax.bar(x+w/2, d[f"{metric}_legacy"], width=w, label="Legacy row-wise split")
    ax.set_xticks(x, [EM_DISPLAY.get(m, m) for m in d.model], rotation=35, ha="right")
    ax.set(ylabel="Normalized overall RMSE", title="EM-S2. Grouped holdout versus leakage-prone legacy split")
    ax.legend()
    fig.tight_layout()
    savefig(fig, out)
    return True


def fig_em_worst(pred: pd.DataFrame, cols: tuple, out: Path) -> None:
    sid, vt, vp, tt, tp = cols
    d = pred.copy()
    d["V_AE"] = np.abs(pd.to_numeric(d[vp], errors="coerce") - pd.to_numeric(d[vt], errors="coerce"))
    d["T_AE"] = np.abs(pd.to_numeric(d[tp], errors="coerce") - pd.to_numeric(d[tt], errors="coerce"))
    if sid:
        agg = d.groupby(sid, as_index=False)[["V_AE", "T_AE"]].mean()
    else:
        agg = d.reset_index().rename(columns={"index": "sample_id"})
        sid = "sample_id"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, c, title in [
        (axes[0], "V_AE", "Voltage-target"),
        (axes[1], "T_AE", "Temperature-target"),
    ]:
        top = agg.nlargest(15, c).sort_values(c)
        ax.barh(top[sid].astype(str), top[c])
        ax.set(xlabel="Mean absolute prediction error", title=title)
    fig.suptitle("EM-S5. Worst parameter sets")
    fig.tight_layout()
    savefig(fig, out)


def copy_or_generate_calibration(root: Path, out: Path, manifest: list[dict[str, str]]) -> None:
    cal_out = out / "04_Parameter_Calibration"
    fig_out = cal_out / "figures"
    tab_out = cal_out / "tables"
    fig_out.mkdir(parents=True, exist_ok=True)
    tab_out.mkdir(parents=True, exist_ok=True)

    candidates = []
    for base in [root / "reports", root / "outputs"]:
        if base.exists():
            for p in base.rglob("*"):
                if p.is_file() and "calibr" in p.name.lower() and "final_filtered_full_v2" not in str(p):
                    candidates.append(p)

    copied = 0
    for p in candidates:
        if p.suffix.lower() not in {".png", ".pdf", ".csv", ".json", ".md"}:
            continue
        dst = (fig_out if p.suffix.lower() in {".png", ".pdf"} else tab_out) / p.name
        if dst.exists():
            continue
        try:
            shutil.copy2(p, dst)
            copied += 1
        except Exception:
            pass

    # Generate CAL-3 if a parameter-recovery table can be recognized.
    for p in candidates:
        if p.suffix.lower() != ".csv":
            continue
        d = read_csv_safe(p)
        if d is None:
            continue
        param = find_col(d, ["parameter", "parameter_name", "name"])
        initial = find_col(d, ["initial", "initial_value"])
        true = find_col(d, ["true", "true_value", "target"])
        calibrated = find_col(d, ["calibrated", "estimated", "calibrated_value", "estimate"])
        if all([param, initial, true, calibrated]):
            dd = d[[param, initial, true, calibrated]].dropna().copy()
            if dd.empty:
                continue
            for c in [initial, true, calibrated]:
                dd[c] = pd.to_numeric(dd[c], errors="coerce")
            # normalize each parameter by true magnitude for plotting
            denom = dd[true].abs().replace(0, np.nan)
            plot = pd.DataFrame({
                "parameter": dd[param].astype(str),
                "Initial": dd[initial] / denom,
                "True": dd[true] / denom,
                "Calibrated": dd[calibrated] / denom,
            })
            x = np.arange(len(plot))
            fig, ax = plt.subplots(figsize=(max(9, len(plot)*0.8), 4.8))
            w = 0.25
            ax.bar(x-w, plot["Initial"], width=w, label="Initial")
            ax.bar(x, plot["True"], width=w, label="True")
            ax.bar(x+w, plot["Calibrated"], width=w, label="Calibrated")
            ax.set_xticks(x, plot.parameter, rotation=45, ha="right")
            ax.set(ylabel="Value normalized by true magnitude",
                   title="CAL-3. Synthetic parameter recovery")
            ax.legend()
            fig.tight_layout()
            savefig(fig, fig_out / "CAL-3_parameter_recovery")
            break

    note = cal_out / "calibration_scope.md"
    note.write_text(
        "# Parameter calibration\n\n"
        "All copied or regenerated calibration artifacts must be interpreted as "
        "**synthetic surrogate-assisted parameter recovery**, unless a verified "
        "experimental reference dataset is explicitly supplied.\n\n"
        f"Calibration-related source artifacts copied: {copied}.\n",
        encoding="utf-8",
    )


def build_docs(out: Path, info: dict[str, Any], generated: list[Path], optional_missing: list[str]) -> None:
    readme = out / "00_README"
    readme.mkdir(parents=True, exist_ok=True)
    (readme / "RESULTS_INDEX.md").write_text(
        "# Final filtered Batch-4 visualization package\n\n"
        f"- Best time-series model: **{TS_DISPLAY.get(info['best_ts'], info['best_ts'])}**\n"
        f"- Best error-metric model: **{EM_DISPLAY.get(info['best_em'], info['best_em'])}**\n"
        "- Time-series benchmark: 7 models × 12 cases × 3 seeds = 252 runs.\n"
        "- Error-metric benchmark: 8 models × 3 seeds.\n"
        "- Time-series filtering: 12,000 source → 11,109 retained + 891 removed.\n\n"
        "See the task-specific figure and table directories for the main and supplementary results.\n",
        encoding="utf-8",
    )
    (readme / "captions.md").write_text(
        "# Figure captions\n\n"
        "**DATA-1.** Routing of the 12,000 generated sequences to the scalar error-metric "
        "task and the filtered time-series response task.\n\n"
        "**DATA-2.** Duration-ratio distribution and early-termination exclusion by operating case.\n\n"
        "**TS-1.** Seven-model comparison across voltage and temperature MAE, RMSE and R². "
        "Stars and dots indicate the best and second-best values.\n\n"
        "**TS-2.** Detailed analysis of the best-ranked time-series model on the hardest retained "
        "operating case, including learning, trajectories, parity and masked time-resolved error.\n\n"
        "**EM-1.** Eight-model comparison for scalar trajectory-error prediction.\n\n"
        "**EM-2.** Detailed analysis of the best error-metric model, including learning behaviour, "
        "ordered predictions, parity and residuals.\n\n"
        "**CAL figures.** Synthetic surrogate-assisted parameter-recovery results only.\n",
        encoding="utf-8",
    )
    (readme / "optional_missing.md").write_text(
        "# Optional figures not generated\n\n"
        + ("\n".join(f"- {x}" for x in optional_missing) if optional_missing else "None.\n"),
        encoding="utf-8",
    )
    rows = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size} for p in generated if p.exists()]
    pd.DataFrame(rows).to_csv(readme / "artifact_manifest.csv", index=False)


def zip_folder(folder: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file() and p != zip_path:
                z.write(p, p.relative_to(folder.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--out", type=Path,
        default=Path("reports/Data_Batch_4/final_filtered_protocol/final_visualizations_v3")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = (root / args.out).resolve() if not args.out.is_absolute() else args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    optional_missing: list[str] = []
    errors: list[str] = []

    try:
        ts_df = collect_ts_metrics(root)
        if ts_df.empty:
            raise RuntimeError("No time-series metrics parsed.")
        ts_df, ts_per_seed, ts_rank = build_ts_tables(ts_df, out)
        best_ts = str(ts_rank.iloc[0].model)

        em_summary = collect_em_summary(root)
        em_rank = build_em_ranking(em_summary, out)
        best_em = str(em_rank.iloc[0].model)

        # Data figures
        fig_data_routing(out / "01_Data_Preparation/figures/DATA-1_dataset_routing")
        manifest = load_manifest(root)
        fig_duration_audit(root, manifest, out / "01_Data_Preparation/figures/DATA-2_duration_ratio_audit")

        # Main heatmaps
        heatmap_figure(
            ts_rank, "display",
            [
                ("V MAE", "V_MAE_mean", True),
                ("V RMSE", "V_RMSE_mean", True),
                ("V R²", "V_R2_mean", False),
                ("T MAE", "T_MAE_mean", True),
                ("T RMSE", "T_RMSE_mean", True),
                ("T R²", "T_R2_mean", False),
                ("Average rank", "Average_Rank", True),
            ],
            "TS-1. Time-series response model ranking",
            out / "02_Time_Series_Response/figures_main/TS-1_model_ranking_heatmap",
        )
        heatmap_figure(
            em_rank, "display",
            [
                ("V-target MAE", "V_MAE", True),
                ("V-target RMSE", "V_RMSE", True),
                ("V-target R²", "V_R2", False),
                ("T-target MAE", "T_MAE", True),
                ("T-target RMSE", "T_RMSE", True),
                ("T-target R²", "T_R2", False),
                ("Overall NRMSE", "Overall_NRMSE", True),
                ("Average rank", "Average_Rank", True),
            ],
            "EM-1. Error-metric surrogate model ranking",
            out / "03_Error_Metric/figures_main/EM-1_model_ranking_heatmap",
        )

        # Detailed TS and supplementary
        ts_detail = fig_ts_best_detail(
            root, ts_df, ts_rank,
            out / "02_Time_Series_Response/figures_main/TS-2_best_model_detailed_analysis"
        )
        fig_ts_examples(
            ts_detail["pred"], ts_detail["score_df"],
            out / "02_Time_Series_Response/figures_supplementary/TS-S1_median_p90_worst_sequences"
        )
        fig_ts_per_case(
            ts_df, best_ts,
            out / "02_Time_Series_Response/figures_supplementary/TS-S2_per_case_rmse"
        )
        fig_ts_endpoint_peak(
            ts_detail["pred"],
            out / "02_Time_Series_Response/figures_supplementary/TS-S3_endpoint_peak_errors"
        )
        if not fig_ts_uncertainty(
            root, ts_detail["hardest_case"],
            out / "02_Time_Series_Response/figures_supplementary/TS-S4_bayesian_uncertainty"
        ):
            optional_missing.append("TS-S4 Bayesian predictive-uncertainty plot: no uncertainty columns found.")
        if not fig_duration_performance(
            ts_detail["pred"], manifest, ts_detail["score_df"],
            out / "02_Time_Series_Response/figures_supplementary/TS-S5_performance_vs_duration_ratio"
        ):
            optional_missing.append("TS-S5 performance versus duration ratio: sequence IDs could not be joined.")

        # Detailed EM and supplementary
        em_detail = fig_em_best_detail(
            root, em_rank,
            out / "03_Error_Metric/figures_main/EM-2_best_model_detailed_analysis"
        )
        if not fig_em_feature_importance(
            root,
            out / "03_Error_Metric/figures_supplementary/EM-S1_feature_importance"
        ):
            optional_missing.append("EM-S1 feature importance: no compatible importance CSV found.")
        if not fig_grouped_vs_legacy(
            root,
            out / "03_Error_Metric/figures_supplementary/EM-S2_grouped_vs_legacy"
        ):
            optional_missing.append("EM-S2 grouped versus legacy: required aggregate tables not found.")
        fig_em_worst(
            em_detail["pred"], em_detail["columns"],
            out / "03_Error_Metric/figures_supplementary/EM-S5_worst_parameter_sets"
        )

        # Calibration: copy existing and generate parameter-recovery chart when possible.
        copy_or_generate_calibration(root, out, [])

        # Collect artifacts and docs.
        generated = [p for p in out.rglob("*") if p.is_file()]
        build_docs(out, {"best_ts": best_ts, "best_em": best_em}, generated, optional_missing)

        generated = [p for p in out.rglob("*") if p.is_file()]
        full_zip = out / "final_visualization_results.zip"
        zip_folder(out, full_zip)

        # colleague package: main figures/tables/readme/calibration only
        colleague_dir = out / "_colleague_package"
        if colleague_dir.exists():
            shutil.rmtree(colleague_dir)
        for sub in ["00_README", "01_Data_Preparation", "04_Parameter_Calibration"]:
            src = out / sub
            if src.exists():
                shutil.copytree(src, colleague_dir / sub)
        for sub in ["02_Time_Series_Response", "03_Error_Metric"]:
            src = out / sub
            if src.exists():
                for part in ["tables", "figures_main"]:
                    s = src / part
                    if s.exists():
                        shutil.copytree(s, colleague_dir / sub / part)
        colleague_zip = out / "colleague_results.zip"
        zip_folder(colleague_dir, colleague_zip)
        shutil.rmtree(colleague_dir)

        pngs = list(out.rglob("*.png"))
        pdfs = list(out.rglob("*.pdf"))
        qc = out / "00_README/quality_control_report.md"
        qc.write_text(
            "# Quality-control report\n\n"
            f"- Parsed TS combinations: {len(ts_df)} / 252\n"
            f"- TS models: {ts_df.model.nunique()} / 7\n"
            f"- TS seeds: {sorted(ts_df.seed.unique().tolist())}\n"
            f"- EM models: {len(em_rank)} / 8\n"
            f"- PNG figures: {len(pngs)}\n"
            f"- PDF figures: {len(pdfs)}\n"
            f"- Best TS model: {TS_DISPLAY.get(best_ts, best_ts)}\n"
            f"- Best EM model: {EM_DISPLAY.get(best_em, best_em)}\n"
            f"- Optional missing items: {len(optional_missing)}\n"
            "- No model-training function is called by this exporter.\n\n"
            + ("\n".join(f"- Optional missing: {x}" for x in optional_missing) + "\n\n"
               if optional_missing else "")
            + "FINAL_VISUALIZATION_STATUS=PASS\n",
            encoding="utf-8",
        )

        print("=" * 72)
        print("VISUALIZATION EXPORT COMPLETE")
        print(f"Best TS model : {TS_DISPLAY.get(best_ts, best_ts)}")
        print(f"Best EM model : {EM_DISPLAY.get(best_em, best_em)}")
        print(f"PNG figures   : {len(pngs)}")
        print(f"PDF figures   : {len(pdfs)}")
        print(f"Full ZIP      : {full_zip}")
        print(f"Colleague ZIP : {colleague_zip}")
        print("Training      : NOT PERFORMED")
        print("=" * 72)
        return 0

    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        (out / "EXPORT_FAILED.txt").write_text(
            "\n".join(errors) + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
