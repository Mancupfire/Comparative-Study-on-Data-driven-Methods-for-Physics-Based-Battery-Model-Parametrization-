#!/usr/bin/env python3
"""Build publication-ready tables and figures for Batch 4.

Consumes:
  * a completed time-series run  (default batch4_full_20260621_140149)
  * a completed error-metric benchmark run (--em-run-id)

Produces, under reports/Data_Batch_4/final_results/<RUN_ID>/:
  tables/{csv,markdown,latex}/Table_1_time_series.*  Table_2_error_metric.*
  figures/{png,pdf}/Figure_1..8 (those whose inputs are available)
  captions.md  methods_note.md  results_summary.md  artifact_inventory.txt
and a ZIP of the report folder.

No model is retrained.  Time-series predictions are read from the export CSVs
(or reconstructed on the fly from existing checkpoints via predict_case).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.error_metric_benchmark.models import DISPLAY_NAMES, FAMILY_ORDER  # noqa: E402
from src.error_metric_benchmark.summarize import _load_run  # noqa: E402
from src.utils import ensure_dir  # noqa: E402

plt.rcParams.update({"figure.dpi": 120, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.bbox": "tight"})

TS_MODELS = ["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]
TS_DISPLAY = {"mlp": "MLP", "rnn": "RNN", "lstm": "LSTM", "bilstm": "BiLSTM",
              "cnn": "CNN", "cnn_bilstm": "CNN-BiLSTM", "bayesian_mlp": "Bayesian MLP"}


# =========================================================================== #
# Ranking helper
# =========================================================================== #
def average_rank(rows: Dict[str, Dict[str, float]], higher_better: set) -> Dict[str, float]:
    """rows: model -> {metric: value}.  Returns model -> mean rank (1=best)."""
    metrics = list(next(iter(rows.values())).keys())
    models = list(rows.keys())
    ranks = {m: [] for m in models}
    for met in metrics:
        vals = [(m, rows[m][met]) for m in models]
        rev = met in higher_better
        ordered = sorted(vals, key=lambda kv: kv[1], reverse=rev)
        for r, (m, _) in enumerate(ordered, start=1):
            ranks[m].append(r)
    return {m: float(np.mean(rs)) for m, rs in ranks.items()}


# =========================================================================== #
# TABLE 1 — time series
# =========================================================================== #
def load_ts_metrics(ts_run: Path) -> pd.DataFrame:
    """Average per-case test metrics over all cases, per model."""
    mroot = ts_run / "metrics"
    cases = sorted(p.name for p in mroot.iterdir() if p.is_dir())
    recs = []
    for model in TS_MODELS:
        per = []
        for case in cases:
            mp = mroot / case / model / "metrics.json"
            if mp.is_file():
                per.append(json.loads(mp.read_text()))
        if not per:
            continue
        recs.append({
            "model": model, "display": TS_DISPLAY[model], "n_cases": len(per),
            "V_MAE": np.mean([d["MAE_V"] for d in per]),
            "V_RMSE": np.mean([d["RMSE_V"] for d in per]),
            "V_R2": np.mean([d["R2_V"] for d in per]),
            "T_MAE": np.mean([d["MAE_T"] for d in per]),
            "T_RMSE": np.mean([d["RMSE_T"] for d in per]),
            "T_R2": np.mean([d["R2_T"] for d in per]),
        })
    df = pd.DataFrame(recs)
    rank_rows = {r["model"]: {k: r[k] for k in
                 ("V_MAE", "V_RMSE", "V_R2", "T_MAE", "T_RMSE", "T_R2")}
                 for _, r in df.iterrows()}
    ar = average_rank(rank_rows, higher_better={"V_R2", "T_R2"})
    df["Average_Rank"] = df["model"].map(ar)
    order = ["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]
    df["__o"] = df["model"].map({m: i for i, m in enumerate(order)})
    return df.sort_values("__o").drop(columns="__o").reset_index(drop=True)


# =========================================================================== #
# TABLE 2 — error metric
# =========================================================================== #
def load_em_table(em_run: Path) -> pd.DataFrame:
    df = _load_run(em_run)
    if df.empty:
        raise SystemExit(f"No metrics under {em_run}/metrics")
    g = df.groupby("model")
    rows = []
    for model in FAMILY_ORDER:
        if model not in g.groups:
            continue
        sub = g.get_group(model)
        def ms(col):
            return float(sub[col].mean()), float(sub[col].std(ddof=0))
        rec = {"model": model, "display": DISPLAY_NAMES[model], "n_seeds": len(sub)}
        rec["V_MAE_m"], rec["V_MAE_s"] = ms("rmse_voltage_mv.MAE")
        rec["V_RMSE_m"], rec["V_RMSE_s"] = ms("rmse_voltage_mv.RMSE")
        rec["V_R2_m"], rec["V_R2_s"] = ms("rmse_voltage_mv.R2")
        rec["T_MAE_m"], rec["T_MAE_s"] = ms("rmse_temperature_c.MAE")
        rec["T_RMSE_m"], rec["T_RMSE_s"] = ms("rmse_temperature_c.RMSE")
        rec["T_R2_m"], rec["T_R2_s"] = ms("rmse_temperature_c.R2")
        rec["NORM_RMSE_m"], rec["NORM_RMSE_s"] = ms("overall_norm_overall_RMSE")
        rec["MEAN_R2_m"], rec["MEAN_R2_s"] = ms("overall_mean_R2")
        rec["PARAMS"] = float(sub["param_count"].mean())
        rec["INFER_MS_m"], rec["INFER_MS_s"] = ms("inference_ms_per_sample")
        rows.append(rec)
    df2 = pd.DataFrame(rows)
    rank_rows = {r["model"]: {"V_MAE": r["V_MAE_m"], "V_RMSE": r["V_RMSE_m"],
                              "V_R2": r["V_R2_m"], "T_MAE": r["T_MAE_m"],
                              "T_RMSE": r["T_RMSE_m"], "T_R2": r["T_R2_m"]}
                 for _, r in df2.iterrows()}
    ar = average_rank(rank_rows, higher_better={"V_R2", "T_R2"})
    df2["Average_Rank"] = df2["model"].map(ar)
    return df2.reset_index(drop=True)


# =========================================================================== #
# Table writers
# =========================================================================== #
def _fmt(v, nd=4):
    return f"{v:.{nd}f}"


def _ms(m, s, nd=4):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def write_table1(df: pd.DataFrame, dirs: Dict[str, Path]):
    cols = ["V_MAE", "V_RMSE", "V_R2", "T_MAE", "T_RMSE", "T_R2", "Average_Rank"]
    head = ["Model", "V MAE", "V RMSE", "V R²", "T MAE", "T RMSE", "T R²", "Avg Rank"]
    # CSV
    csv = df[["display"] + cols].copy()
    csv.columns = head
    csv.to_csv(dirs["csv"] / "Table_1_time_series.csv", index=False)
    # best/2nd per column for highlighting
    lower = {"V_MAE", "V_RMSE", "T_MAE", "T_RMSE", "Average_Rank"}
    best, second = {}, {}
    for c in cols:
        order = df[c].sort_values(ascending=(c in lower)).index.tolist()
        best[c], second[c] = order[0], (order[1] if len(order) > 1 else order[0])
    # Markdown
    md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for i, r in df.iterrows():
        cells = [r["display"]]
        for c in cols:
            txt = _fmt(r[c], 4 if c != "Average_Rank" else 2)
            if i == best[c]:
                txt = f"**{txt}**"
            cells.append(txt)
        md.append("| " + " | ".join(cells) + " |")
    (dirs["md"] / "Table_1_time_series.md").write_text("\n".join(md) + "\n")
    # LaTeX
    tex = _latex_table(df, "display", cols, head, best, second, nd_map={"Average_Rank": 2},
                       caption="Time-series prediction performance (test set, averaged "
                       "over 12 operating cases). Best in \\textbf{bold}, second-best "
                       "\\underline{underlined}. Voltage in V, temperature in °C.",
                       label="tab:time_series")
    (dirs["tex"] / "Table_1_time_series.tex").write_text(tex)


def write_table2(df: pd.DataFrame, dirs: Dict[str, Path]):
    head = ["Model", "V MAE (mV)", "V RMSE (mV)", "V R²", "T MAE (°C)",
            "T RMSE (°C)", "T R²", "Norm. RMSE", "Mean R²", "Avg Rank",
            "Params", "Infer (ms)"]
    # CSV (with mean/std columns expanded)
    csv = pd.DataFrame({
        "Model": df["display"],
        "V MAE (mV)": [_ms(m, s) for m, s in zip(df.V_MAE_m, df.V_MAE_s)],
        "V RMSE (mV)": [_ms(m, s) for m, s in zip(df.V_RMSE_m, df.V_RMSE_s)],
        "V R2": [_ms(m, s) for m, s in zip(df.V_R2_m, df.V_R2_s)],
        "T MAE (C)": [_ms(m, s) for m, s in zip(df.T_MAE_m, df.T_MAE_s)],
        "T RMSE (C)": [_ms(m, s) for m, s in zip(df.T_RMSE_m, df.T_RMSE_s)],
        "T R2": [_ms(m, s) for m, s in zip(df.T_R2_m, df.T_R2_s)],
        "Norm Overall RMSE": [_ms(m, s) for m, s in zip(df.NORM_RMSE_m, df.NORM_RMSE_s)],
        "Mean R2": [_ms(m, s) for m, s in zip(df.MEAN_R2_m, df.MEAN_R2_s)],
        "Average Rank": [f"{v:.2f}" for v in df.Average_Rank],
        "Parameter Count": [f"{int(v)}" for v in df.PARAMS],
        "Inference Time (ms/sample)": [_ms(m, s) for m, s in zip(df.INFER_MS_m, df.INFER_MS_s)],
    })
    csv.to_csv(dirs["csv"] / "Table_2_error_metric.csv", index=False)

    metric_cols = ["V_MAE_m", "V_RMSE_m", "V_R2_m", "T_MAE_m", "T_RMSE_m",
                   "T_R2_m", "NORM_RMSE_m", "MEAN_R2_m", "Average_Rank"]
    lower = {"V_MAE_m", "V_RMSE_m", "T_MAE_m", "T_RMSE_m", "NORM_RMSE_m", "Average_Rank"}
    best, second = {}, {}
    for c in metric_cols:
        order = df[c].sort_values(ascending=(c in lower)).index.tolist()
        best[c], second[c] = order[0], (order[1] if len(order) > 1 else order[0])

    def cell(r, c, nd=4):
        sm = c.replace("_m", "_s")
        txt = _ms(r[c], r[sm], nd) if sm in df.columns else _fmt(r[c], nd)
        return txt

    # Markdown
    md = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for i, r in df.iterrows():
        cells = [r["display"]]
        for c in metric_cols[:-1]:
            t = cell(r, c)
            if i == best[c]:
                t = f"**{t}**"
            cells.append(t)
        ar = f"{r['Average_Rank']:.2f}"
        if i == best["Average_Rank"]:
            ar = f"**{ar}**"
        cells.append(ar)
        cells.append(f"{int(r['PARAMS'])}")
        cells.append(_ms(r["INFER_MS_m"], r["INFER_MS_s"]))
        md.append("| " + " | ".join(cells) + " |")
    (dirs["md"] / "Table_2_error_metric.md").write_text("\n".join(md) + "\n")

    # LaTeX
    ncol = len(head)
    tex = ["\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Error-metric prediction performance (grouped-holdout test "
           "set, mean $\\pm$ std over 3 seeds). Best in \\textbf{bold}, "
           "second-best \\underline{underlined}.}",
           "\\label{tab:error_metric}",
           "\\begin{tabular}{l" + "r" * (ncol - 1) + "}", "\\toprule",
           " & ".join(_texesc(h) for h in head) + " \\\\", "\\midrule"]
    for i, r in df.iterrows():
        cells = [_texesc(r["display"])]
        for c in metric_cols[:-1]:
            t = cell(r, c)
            t = _hl(t, i, best[c], second[c])
            cells.append(t)
        ar = f"{r['Average_Rank']:.2f}"
        ar = _hl(ar, i, best["Average_Rank"], second["Average_Rank"])
        cells.append(ar)
        cells.append(f"{int(r['PARAMS'])}")
        cells.append(_ms(r["INFER_MS_m"], r["INFER_MS_s"]))
        tex.append(" & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (dirs["tex"] / "Table_2_error_metric.tex").write_text("\n".join(tex))


def _texesc(s):
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("²", "$^2$")
            .replace("±", "$\\pm$").replace("°C", "$^\\circ$C"))


def _hl(txt, i, best_i, second_i):
    txt = _texesc(txt)
    if i == best_i:
        return f"\\textbf{{{txt}}}"
    if i == second_i:
        return f"\\underline{{{txt}}}"
    return txt


def _latex_table(df, name_col, cols, head, best, second, nd_map, caption, label):
    ncol = len(head)
    out = ["\\begin{table}[t]", "\\centering", "\\small",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{l" + "r" * (ncol - 1) + "}", "\\toprule",
           " & ".join(_texesc(h) for h in head) + " \\\\", "\\midrule"]
    for i, r in df.iterrows():
        cells = [_texesc(r[name_col])]
        for c in cols:
            nd = nd_map.get(c, 4)
            cells.append(_hl(_fmt(r[c], nd), i, best[c], second[c]))
        out.append(" & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


# =========================================================================== #
# Figure helpers
# =========================================================================== #
def savefig(fig, figdirs, name):
    try:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:  # noqa: BLE001
        pass
    fig.savefig(figdirs["png"] / f"{name}.png", dpi=300)
    fig.savefig(figdirs["pdf"] / f"{name}.pdf")
    plt.close(fig)
    return name


def _linfit(x, y):
    x = np.asarray(x); y = np.asarray(y)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((y - x) ** 2))
    mae = np.mean(np.abs(y - x))
    return a, b, r2, rmse, mae


# ---- EM figures ---- #
def fig1_learning(em_run, best_neural, figdirs) -> Optional[str]:
    hp = em_run / "histories" / best_neural / "seed42" / "history.csv"
    if not hp.is_file():
        return None
    h = pd.read_csv(hp)
    if "member" in h.columns:        # deep ensemble: use member 0
        h = h[h["member"] == 0]
    best_epoch = int(h.loc[h["val_loss"].idxmin(), "epoch"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(h["epoch"], h["train_rmse"], label="train RMSE")
    ax[0].plot(h["epoch"], h["val_rmse"], label="val RMSE")
    ax[0].axvline(best_epoch, color="k", ls="--", lw=1, label=f"best epoch {best_epoch}")
    ax[0].set(xlabel="epoch", ylabel="RMSE (standardized)", title="(a) Train/val RMSE")
    ax[0].legend()
    ax[1].plot(h["epoch"], h["train_loss"], label="train loss")
    ax[1].plot(h["epoch"], h["val_loss"], label="val loss")
    ax[1].axvline(best_epoch, color="k", ls="--", lw=1)
    ax[1].set(xlabel="epoch", ylabel="MSE loss (standardized)", title="(b) Train/val loss")
    ax[1].legend()
    ax[2].plot(h["epoch"], h["lr"], color="tab:green")
    ax[2].axvline(best_epoch, color="k", ls="--", lw=1)
    ax[2].set(xlabel="epoch", ylabel="learning rate", title="(c) LR schedule")
    fig.suptitle(f"Figure 1 — Learning behaviour: {DISPLAY_NAMES[best_neural]}")
    return savefig(fig, figdirs, "Figure_1_error_metric_learning")


def _load_em_pred(em_run, model):
    p = em_run / "predictions" / model / "seed42" / "test_predictions.csv"
    return pd.read_csv(p) if p.is_file() else None


def fig2_parity(em_run, best_model, figdirs, relative_inset=False) -> Optional[str]:
    df = _load_em_pred(em_run, best_model)
    if df is None:
        return None
    specs = [("rmse_voltage_mv", "Voltage RMSE (mV)"),
             ("rmse_temperature_c", "Temperature RMSE (°C)")]
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    for j, (col, lab) in enumerate(specs):
        t = df[f"{col}_true"].to_numpy(); p = df[f"{col}_pred"].to_numpy()
        a, b, r2, rmse, mae = _linfit(t, p)
        axp = ax[0, j]
        axp.scatter(t, p, s=8, alpha=0.4)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        axp.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        xs = np.linspace(lo, hi, 100)
        axp.plot(xs, a * xs + b, "r-", lw=1.2, label="linear fit")
        axp.set(xlabel=f"true {lab}", ylabel=f"predicted {lab}",
                title=f"({chr(97+j)}) Parity — {lab}")
        axp.text(0.04, 0.96, f"slope={a:.3f}\nintercept={b:.3f}\nR²={r2:.3f}\n"
                 f"RMSE={rmse:.3f}\nMAE={mae:.3f}", transform=axp.transAxes,
                 va="top", ha="left", fontsize=8,
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))
        axp.legend(loc="lower right", fontsize=8)
        # residuals
        res = p - t
        axr = ax[1, j]
        axr.hist(res, bins=40, color="tab:blue", alpha=0.7)
        axr.axvline(0, color="k", ls="--", lw=1, label="zero error")
        axr.axvline(res.mean(), color="r", lw=1.2, label=f"mean={res.mean():.3f}")
        axr.axvline(np.median(res), color="g", lw=1.2, label=f"median={np.median(res):.3f}")
        p2, p97 = np.percentile(res, [2.5, 97.5])
        axr.axvspan(p2, p97, color="orange", alpha=0.15, label="95% interval")
        axr.set(xlabel=f"residual {lab}", ylabel="count",
                title=f"({chr(99+j)}) Residuals — {lab}")
        axr.legend(fontsize=7)
        if relative_inset:
            rel = np.abs(res) / (np.abs(t) + 1e-8)
            ins = axr.inset_axes([0.62, 0.45, 0.35, 0.5])
            ins.hist(rel * 100, bins=30, color="purple", alpha=0.6)
            ins.set_title("rel. err (%)", fontsize=7)
            ins.tick_params(labelsize=6)
    fig.suptitle(f"Figure 2 — Parity & residuals: {DISPLAY_NAMES[best_model]}")
    name = "Figure_2_error_metric_parity" + ("_relinset" if relative_inset else "")
    return savefig(fig, figdirs, name)


def fig3_comparison(by_model_df, figdirs) -> Optional[str]:
    df = by_model_df.copy()
    metrics = [("V_MAE_m", "V MAE", "low"), ("V_RMSE_m", "V RMSE", "low"),
               ("V_R2_m", "V R²", "high"), ("T_MAE_m", "T MAE", "low"),
               ("T_RMSE_m", "T RMSE", "low"), ("T_R2_m", "T R²", "high"),
               ("NORM_RMSE_m", "Norm RMSE", "low"), ("MEAN_R2_m", "Mean R²", "high")]
    M = np.zeros((len(df), len(metrics)))
    for j, (c, _, d) in enumerate(metrics):
        v = df[c].to_numpy(dtype=float)
        rng = v.max() - v.min()
        norm = (v - v.min()) / rng if rng > 0 else np.zeros_like(v)
        # convert so 1=best, 0=worst for every column
        M[:, j] = (1 - norm) if d == "low" else norm
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([f"{m[1]}\n({'↓' if m[2]=='low' else '↑'})" for m in metrics])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["display"])
    for i in range(len(df)):
        for j, (c, _, _) in enumerate(metrics):
            ax.text(j, i, f"{df.iloc[i][c]:.3g}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="normalized score (1 = best)")
    ax.set_title("Figure 3 — Error-metric model comparison\n"
                 "(cells show raw values; color normalized so brighter = better; "
                 "mV and °C never aggregated raw)")
    return savefig(fig, figdirs, "Figure_3_error_metric_comparison")


def fig4_by_sample(em_run, best_model, figdirs) -> Optional[str]:
    df = _load_em_pred(em_run, best_model)
    if df is None:
        return None
    for variant, sortcol in [("by_id", "sample_id"), ("by_true", "true")]:
        fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        for k, (col, lab, unit) in enumerate([
                ("rmse_voltage_mv", "Voltage RMSE", "mV"),
                ("rmse_temperature_c", "Temperature RMSE", "°C")]):
            d = df.copy()
            if sortcol == "sample_id":
                d = d.sort_values("sample_id")
            else:
                d = d.sort_values(f"{col}_true")
            x = np.arange(len(d))
            ax[k].plot(x, d[f"{col}_true"], label="true", lw=1)
            ax[k].plot(x, d[f"{col}_pred"], label="predicted", lw=1, alpha=0.8)
            ax[k].set(ylabel=f"{lab} ({unit})",
                      title=f"({chr(97+k)}) {lab} across test parameter sets")
            ax[k].legend()
        ax[1].set_xlabel("test sample index "
                         + ("(sorted by sample_id)" if sortcol == "sample_id"
                            else "(sorted by true value)"))
        fig.suptitle(f"Figure 4 — Per-sample prediction: {DISPLAY_NAMES[best_model]} ({variant})")
        savefig(fig, figdirs, f"Figure_4_error_metric_by_sample_{variant}")
    return "Figure_4_error_metric_by_sample"


# ---- TS figures ---- #
def _ts_pred(ts_run, data_root, case, model):
    p = ts_run / "predictions" / case / model / "test_predictions.csv"
    if p.is_file():
        df = pd.read_csv(p)
        n_seq = df["sequence_id"].nunique()
        t_last = len(df) // n_seq
        order = df.sort_values(["sequence_id", "time_index"])
        vt = order["voltage_true"].to_numpy().reshape(n_seq, t_last)
        vp = order["voltage_pred"].to_numpy().reshape(n_seq, t_last)
        tt = order["temperature_true"].to_numpy().reshape(n_seq, t_last)
        tp = order["temperature_pred"].to_numpy().reshape(n_seq, t_last)
        time_s = order.groupby("sequence_id")["time_s"].first()  # not exact order
        time_s = order[order["sequence_id"] == order["sequence_id"].iloc[0]].sort_values("time_index")["time_s"].to_numpy()
        seq_ids = order["sequence_id"].drop_duplicates().to_numpy()
        return dict(v_true=vt, v_pred=vp, t_true=tt, t_pred=tp, time_s=time_s,
                    seq_ids=seq_ids)
    # fallback: reconstruct via checkpoint (no retrain)
    from src.predict import predict_case
    out = predict_case(data_root, case, model, outputs_dir=str(ts_run), split="test")
    seq_ids = np.array([f"{s}__{case}" for s in np.asarray(out["sample_ids"]).astype(str)])
    return dict(v_true=out["v_true"], v_pred=out["v_pred"], t_true=out["t_true"],
                t_pred=out["t_pred"], time_s=out["time_s"], seq_ids=seq_ids)


def pick_best_ts(ts_df: pd.DataFrame) -> str:
    return ts_df.sort_values("Average_Rank").iloc[0]["model"]


def pick_hardest_case(ts_run: Path, model: str) -> str:
    mroot = ts_run / "metrics"
    cases = sorted(p.name for p in mroot.iterdir() if p.is_dir())
    # combined normalized V+T RMSE across cases
    vr, tr = {}, {}
    for c in cases:
        mp = mroot / c / model / "metrics.json"
        if mp.is_file():
            d = json.loads(mp.read_text())
            vr[c], tr[c] = d["RMSE_V"], d["RMSE_T"]
    va = np.array([vr[c] for c in cases]); ta = np.array([tr[c] for c in cases])
    vn = (va - va.min()) / (np.ptp(va) or 1); tn = (ta - ta.min()) / (np.ptp(ta) or 1)
    comb = vn + tn
    return cases[int(np.argmax(comb))]


def fig5_six_panel(ts_run, data_root, model, case, figdirs) -> Optional[str]:
    d = _ts_pred(ts_run, data_root, case, model)
    t = d["time_s"]
    fig, ax = plt.subplots(3, 2, figsize=(14, 12))
    # (a) observed voltage curves
    for i in range(min(40, d["v_true"].shape[0])):
        ax[0, 0].plot(t, d["v_true"][i], color="tab:blue", alpha=0.15)
    ax[0, 0].set(xlabel="time (s)", ylabel="voltage (V)", title="(a) Observed voltage curves")
    # (b) observed temperature curves
    for i in range(min(40, d["t_true"].shape[0])):
        ax[0, 1].plot(t, d["t_true"][i], color="tab:red", alpha=0.15)
    ax[0, 1].set(xlabel="time (s)", ylabel="temperature (°C)", title="(b) Observed temperature curves")
    # (c) normalized MAE over time
    vmae = np.mean(np.abs(d["v_pred"] - d["v_true"]), axis=0)
    tmae = np.mean(np.abs(d["t_pred"] - d["t_true"]), axis=0)
    ax[1, 0].plot(t, vmae / (np.abs(d["v_true"]).mean() + 1e-9), label="voltage")
    ax[1, 0].plot(t, tmae / (np.abs(d["t_true"]).mean() + 1e-9), label="temperature")
    ax[1, 0].set(xlabel="time (s)", ylabel="normalized MAE", title="(c) Normalized MAE over time")
    ax[1, 0].legend()
    # (d) voltage obs vs pred mean + 5/95
    for arr, lab, c in [(d["v_true"], "observed", "tab:blue"), (d["v_pred"], "predicted", "tab:orange")]:
        m = arr.mean(0); p5 = np.percentile(arr, 5, 0); p95 = np.percentile(arr, 95, 0)
        ax[1, 1].plot(t, m, color=c, label=f"{lab} mean")
        ax[1, 1].fill_between(t, p5, p95, color=c, alpha=0.2)
    ax[1, 1].set(xlabel="time (s)", ylabel="voltage (V)", title="(d) Voltage mean & 5–95%")
    ax[1, 1].legend()
    # (e) temperature obs vs pred mean + 5/95
    for arr, lab, c in [(d["t_true"], "observed", "tab:blue"), (d["t_pred"], "predicted", "tab:orange")]:
        m = arr.mean(0); p5 = np.percentile(arr, 5, 0); p95 = np.percentile(arr, 95, 0)
        ax[2, 0].plot(t, m, color=c, label=f"{lab} mean")
        ax[2, 0].fill_between(t, p5, p95, color=c, alpha=0.2)
    ax[2, 0].set(xlabel="time (s)", ylabel="temperature (°C)", title="(e) Temperature mean & 5–95%")
    ax[2, 0].legend()
    # (f) RMSE over time
    vrmse = np.sqrt(np.mean((d["v_pred"] - d["v_true"]) ** 2, axis=0))
    trmse = np.sqrt(np.mean((d["t_pred"] - d["t_true"]) ** 2, axis=0))
    ax[2, 1].plot(t, vrmse, label="voltage RMSE (V)")
    ax[2, 1].plot(t, trmse, label="temperature RMSE (°C)")
    ax[2, 1].set(xlabel="time (s)", ylabel="RMSE", title="(f) RMSE over time")
    ax[2, 1].legend()
    fig.suptitle(f"Figure 5 — Time-series summary: {TS_DISPLAY[model]}, hardest case {case}")
    return savefig(fig, figdirs, "Figure_5_time_series_six_panel")


def fig6_examples(ts_run, data_root, model, figdirs) -> Optional[str]:
    mroot = ts_run / "metrics"
    cases = sorted(p.name for p in mroot.iterdir() if p.is_dir())
    # gather per-sequence combined error across all cases
    rows = []
    cache = {}
    for c in cases:
        d = _ts_pred(ts_run, data_root, c, model)
        cache[c] = d
        vr = np.sqrt(np.mean((d["v_pred"] - d["v_true"]) ** 2, axis=1))
        tr = np.sqrt(np.mean((d["t_pred"] - d["t_true"]) ** 2, axis=1))
        for i, sid in enumerate(d["seq_ids"]):
            rows.append((c, i, sid, vr[i], tr[i]))
    rdf = pd.DataFrame(rows, columns=["case", "idx", "seq", "vr", "tr"])
    vn = (rdf.vr - rdf.vr.min()) / (np.ptp(rdf.vr.to_numpy()) or 1)
    tn = (rdf.tr - rdf.tr.min()) / (np.ptp(rdf.tr.to_numpy()) or 1)
    rdf["comb"] = vn + tn
    rdf = rdf.sort_values("comb").reset_index(drop=True)
    picks = {"median": rdf.iloc[len(rdf) // 2], "p90": rdf.iloc[int(0.9 * (len(rdf) - 1))],
             "worst": rdf.iloc[-1]}
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for r, (lab, row) in enumerate(picks.items()):
        d = cache[row["case"]]; i = int(row["idx"]); t = d["time_s"]
        ax[r, 0].plot(t, d["v_true"][i], label="true")
        ax[r, 0].plot(t, d["v_pred"][i], label="pred", alpha=0.8)
        ax[r, 0].set(ylabel="voltage (V)",
                     title=f"({chr(97+2*r)}) {lab}: {row['seq']}")
        ax[r, 0].legend(fontsize=8)
        ax[r, 1].plot(t, d["t_true"][i], label="true")
        ax[r, 1].plot(t, d["t_pred"][i], label="pred", alpha=0.8)
        ax[r, 1].set(ylabel="temperature (°C)",
                     title=f"({chr(98+2*r)}) {lab}: {row['case']}")
        ax[r, 1].legend(fontsize=8)
    for a in ax[2]:
        a.set_xlabel("time (s)")
    fig.suptitle(f"Figure 6 — Ground truth vs predicted examples: {TS_DISPLAY[model]}")
    return savefig(fig, figdirs, "Figure_6_time_series_examples")


def fig7_ts_learning(ts_run, model, case, figdirs) -> Optional[str]:
    hp = ts_run / "metrics" / case / model / "history.csv"
    if not hp.is_file():
        return None
    h = pd.read_csv(hp)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(h["epoch"], h["train_loss"], label="train loss")
    ax[0].plot(h["epoch"], h["val_loss"], label="val loss")
    if "train_loss_v" in h:
        ax[0].plot(h["epoch"], h["train_loss_v"], "--", label="train V loss", alpha=0.7)
        ax[0].plot(h["epoch"], h["train_loss_t"], "--", label="train T loss", alpha=0.7)
        ax[0].plot(h["epoch"], h["val_loss_v"], ":", label="val V loss", alpha=0.7)
        ax[0].plot(h["epoch"], h["val_loss_t"], ":", label="val T loss", alpha=0.7)
    be = int(h.loc[h["val_loss"].idxmin(), "epoch"])
    ax[0].axvline(be, color="k", ls="--", lw=1, label=f"best epoch {be}")
    ax[0].set(xlabel="epoch", ylabel="loss", title="(a) Loss curves")
    ax[0].legend(fontsize=8)
    if "lr" in h:
        ax[1].plot(h["epoch"], h["lr"], color="tab:green")
        ax[1].axvline(be, color="k", ls="--", lw=1)
    ax[1].set(xlabel="epoch", ylabel="learning rate", title="(b) LR schedule")
    fig.suptitle(f"Figure 7 — Learning curve: {TS_DISPLAY[model]}, hardest case {case}")
    return savefig(fig, figdirs, "Figure_7_time_series_learning")


def fig8_legacy_vs_grouped(grouped_run, legacy_run, figdirs) -> Optional[str]:
    if legacy_run is None or not Path(legacy_run).is_dir():
        return None
    g = _load_run(Path(grouped_run)).groupby("model")["overall_norm_overall_RMSE"].mean()
    l = _load_run(Path(legacy_run)).groupby("model")["overall_norm_overall_RMSE"].mean()
    models = [m for m in FAMILY_ORDER if m in g.index and m in l.index]
    x = np.arange(len(models)); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w / 2, [l[m] for m in models], w, label="legacy_reproduction (leaky)")
    ax.bar(x + w / 2, [g[m] for m in models], w, label="grouped_holdout (no leakage)")
    ax.set_xticks(x); ax.set_xticklabels([DISPLAY_NAMES[m] for m in models], rotation=30, ha="right")
    ax.set(ylabel="test normalized overall RMSE", title="Figure 8 — Legacy vs grouped split")
    ax.legend()
    return savefig(fig, figdirs, "Figure_8_legacy_vs_grouped")


# =========================================================================== #
# Orchestration
# =========================================================================== #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ts-run-id", default="batch4_full_20260621_140149")
    p.add_argument("--ts-root", default="outputs/Data_Batch_4/time_series_downsampled_160")
    p.add_argument("--ts-data-root", default="data/Data_Batch_4_downsampled_160")
    p.add_argument("--em-run-id", required=True)
    p.add_argument("--em-root", default="outputs/Data_Batch_4/error_metric_benchmark")
    p.add_argument("--em-smoke", action="store_true")
    p.add_argument("--legacy-run-id", default=None)
    p.add_argument("--out-id", default=None)
    a = p.parse_args(argv)

    ts_run = Path(a.ts_root) / a.ts_run_id
    em_base = ("outputs_smoke/Data_Batch_4/error_metric_benchmark"
               if a.em_smoke else a.em_root)
    em_run = Path(em_base) / a.em_run_id
    legacy_run = (Path(em_base) / a.legacy_run_id) if a.legacy_run_id else None
    out_id = a.out_id or a.em_run_id
    report = Path("reports/Data_Batch_4/final_results") / out_id

    tdirs = {k: ensure_dir(report / "tables" / v)
             for k, v in {"csv": "csv", "md": "markdown", "tex": "latex"}.items()}
    fdirs = {k: ensure_dir(report / "figures" / k) for k in ("png", "pdf")}

    print(f"[final] ts_run={ts_run}\n[final] em_run={em_run}\n[final] report={report}")

    # Tables
    ts_df = load_ts_metrics(ts_run)
    em_df = load_em_table(em_run)
    write_table1(ts_df, tdirs)
    write_table2(em_df, tdirs)
    print("[final] tables written")

    # Best models
    best_overall = em_df.sort_values("NORM_RMSE_m").iloc[0]["model"]
    neural = em_df[em_df["model"] != "extratrees"]
    best_neural = neural.sort_values("NORM_RMSE_m").iloc[0]["model"]
    best_ts = pick_best_ts(ts_df)
    hardest = pick_hardest_case(ts_run, best_ts)
    print(f"[final] best_overall_em={best_overall} best_neural_em={best_neural} "
          f"best_ts={best_ts} hardest_case={hardest}")

    made = []
    for fn in [
        lambda: fig1_learning(em_run, best_neural, fdirs),
        lambda: fig2_parity(em_run, best_overall, fdirs, False),
        lambda: fig2_parity(em_run, best_overall, fdirs, True),
        lambda: fig3_comparison(em_df, fdirs),
        lambda: fig4_by_sample(em_run, best_overall, fdirs),
        lambda: fig5_six_panel(ts_run, a.ts_data_root, best_ts, hardest, fdirs),
        lambda: fig6_examples(ts_run, a.ts_data_root, best_ts, fdirs),
        lambda: fig7_ts_learning(ts_run, best_ts, hardest, fdirs),
        lambda: fig8_legacy_vs_grouped(em_run, legacy_run, fdirs),
    ]:
        try:
            r = fn()
            if r:
                made.append(r); print(f"[fig] {r}")
            else:
                print("[fig] skipped (inputs unavailable)")
        except Exception as exc:  # noqa: BLE001
            print(f"[fig] FAILED: {exc}")
            import traceback; traceback.print_exc()

    _write_docs(report, ts_df, em_df, best_overall, best_neural, best_ts, hardest,
                a.em_run_id, a.ts_run_id, made, legacy_run)
    _zip(report)
    print(f"[final] DONE -> {report}")
    return 0


def _write_docs(report, ts_df, em_df, best_overall, best_neural, best_ts, hardest,
                em_run_id, ts_run_id, made, legacy_run):
    (report / "captions.md").write_text(_captions(best_overall, best_neural, best_ts, hardest))
    (report / "methods_note.md").write_text(_methods())
    (report / "results_summary.md").write_text(
        _results_summary(ts_df, em_df, best_overall, best_ts, em_run_id, ts_run_id, legacy_run))
    inv = []
    for pth in sorted(report.rglob("*")):
        if pth.is_file():
            inv.append(f"{pth.relative_to(report)}\t{pth.stat().st_size}")
    (report / "artifact_inventory.txt").write_text("\n".join(inv) + "\n")


def _captions(bo, bn, bts, hardest):
    return f"""# Figure & table captions

**Table 1.** Time-series prediction performance on the Batch-4 test set, averaged
over the 12 operating cases. Columns: voltage/temperature MAE, RMSE, R²
(voltage in V, temperature in °C) and the average rank over the six metrics
(lower better for MAE/RMSE, higher better for R²).

**Table 2.** Error-metric prediction performance (grouped-holdout test set,
mean ± std over seeds 42/43/44). Targets: voltage RMSE (mV), temperature RMSE
(°C). "Norm. RMSE" is the scale-safe normalized overall RMSE in standardized
two-target space. Best in bold, second-best underlined (LaTeX).

**Figure 1.** Learning behaviour of the best neural error-metric model ({DISPLAY_NAMES.get(bn,bn)}):
(a) train/val RMSE, (b) train/val loss, (c) learning-rate schedule, with best epoch marked.

**Figure 2.** Parity and residual analysis for the best error-metric model
({DISPLAY_NAMES.get(bo,bo)}): (a,b) true-vs-predicted voltage/temperature RMSE with
y=x, linear fit, slope/intercept/R²/RMSE/MAE; (c,d) residual distributions with
zero line, mean, median and 95% interval. A variant adds a relative-error inset.

**Figure 3.** Error-metric model comparison heatmap; cells show raw values,
colours normalized per metric so brighter = better. mV and °C are never combined
into a raw aggregate.

**Figure 4.** Per-parameter-set true vs predicted voltage/temperature RMSE for
{DISPLAY_NAMES.get(bo,bo)}, ordered by sample id and (variant) by true value.

**Figure 5.** Six-panel time-series summary for the best time-series model
({TS_DISPLAY.get(bts,bts)}) on the hardest case ({hardest}).

**Figure 6.** Ground-truth vs predicted voltage and temperature for median-,
90th-percentile- and worst-error test sequences ({TS_DISPLAY.get(bts,bts)}).

**Figure 7.** Training/validation loss (incl. per-target) and LR schedule for
{TS_DISPLAY.get(bts,bts)} on the hardest case ({hardest}).

**Figure 8.** Legacy (leaky, row-wise) vs grouped-holdout split: effect of the
split protocol on apparent error-metric performance.
"""


def _methods():
    return """# Methods note — error-metric benchmark

**Data.** Batch 4: 1000 physical parameter sets × 12 operating conditions =
12,000 sequences (0 failed). Inputs (17): 12 physical parameters +
[c_rate, ambient_temp_C, initial_temp_C] + one-hot(operation_code). Targets:
voltage-prediction RMSE (mV) and temperature-prediction RMSE (°C).

**Split.** Recommended protocol `grouped_holdout`: split by unique `sample_id`
(70/15/15, seed 42) so all 12 conditions of a parameter set stay in one split —
zero sample_id overlap. A `legacy_reproduction` protocol (row-wise random split)
is provided only to reproduce/explain the optimistic legacy numbers; it leaks
sample_ids across splits.

**Scaling.** Feature StandardScaler and per-target StandardScaler are fit on the
training split only. All metrics are reported after inverse transform, in
physical units.

**Overall metric (scale-safe).** Normalized overall RMSE/MAE are computed in
standardized two-target space (per-target z-score using the true-test std), so
mV and °C are never mixed in a raw aggregate. Mean R² is the mean of the two
per-target R².

**Models (12).** ANN, MLP, Wide&Deep MLP, Attention MLP, Gated MLP, Residual
MLP, Multitask MLP, Deep Ensemble MLP (5 members), RNN, LSTM, BiLSTM,
ExtraTrees. Neural models: AdamW, MSE on standardized targets, ReduceLROnPlateau,
gradient clipping, early stopping on val loss, best-val checkpoint, deterministic
seeds. RNN/LSTM/BiLSTM consume the ordered physical-feature vector as a
length-17 feature sequence (one scalar per step); these are NOT temporal models.
Three seeds (42/43/44) → mean ± std.

**Time-series task.** The 7 time-series models are NOT retrained; metrics come
from the completed run and predictions are reconstructed from saved checkpoints
by forward inference only.
"""


def _results_summary(ts_df, em_df, best_overall, best_ts, em_run_id, ts_run_id, legacy_run):
    em = em_df.sort_values("NORM_RMSE_m")
    lines = ["# Results summary", "",
             f"- Error-metric run: `{em_run_id}`",
             f"- Time-series run: `{ts_run_id}`",
             f"- Best error-metric model: **{DISPLAY_NAMES.get(best_overall,best_overall)}**",
             f"- Best time-series model: **{TS_DISPLAY.get(best_ts,best_ts)}**",
             f"- Legacy comparison run: {'yes' if legacy_run else 'not provided'}", "",
             "## Error-metric ranking (test normalized overall RMSE)", ""]
    for _, r in em.iterrows():
        lines.append(f"  {r['display']}: norm_RMSE={r['NORM_RMSE_m']:.4f}"
                     f" ± {r['NORM_RMSE_s']:.4f}, mean_R²={r['MEAN_R2_m']:.4f}")
    lines += ["", "## Time-series ranking (average rank)", ""]
    for _, r in ts_df.sort_values("Average_Rank").iterrows():
        lines.append(f"  {r['display']}: avg_rank={r['Average_Rank']:.2f}, "
                     f"V_RMSE={r['V_RMSE']:.4f} V, T_RMSE={r['T_RMSE']:.4f} °C")
    return "\n".join(lines) + "\n"


def _zip(report: Path):
    zpath = report.parent / f"{report.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(report.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(report.parent))
    print(f"[final] zip -> {zpath}")


if __name__ == "__main__":
    sys.exit(main())
