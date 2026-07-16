"""Finalize the filtered-protocol results: rankings, figures, calibration, ZIP.

Reads the completed filtered time-series run and the combined error-metric
results, then writes:

    <reports_dir>/time_series/ranking_table.csv      (mean over cases x seeds)
    <reports_dir>/time_series/metrics_by_model_and_case.csv
    <reports_dir>/time_series/figures/ts_rmse_by_model.png
    <reports_dir>/time_series/calibration_bayesian.csv (MC-Dropout 95% coverage)
    <reports_dir>/error_metric/ranking_table.csv     (copied from combined)
    <reports_dir>/error_metric/figures/em_rmse_by_model.png
    <reports_dir>/final_filtered_results.zip          (all of the above)

Designed to be called by launch_full_pipeline.sh after training completes; it
is read-only w.r.t. the training outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TS_MODEL_ORDER = ["ann", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]
TS_DISPLAY = {"ann": "ANN", "rnn": "RNN", "lstm": "LSTM", "bilstm": "BiLSTM",
              "cnn": "CNN", "cnn_bilstm": "CNN-BiLSTM", "bayesian_mlp": "Bayesian MLP"}


def _load_ts(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for seed_dir in sorted(run_dir.glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        for mp in sorted((seed_dir / "metrics").glob("*/*/metrics.json")):
            d = json.loads(mp.read_text())
            te = d["test"]
            rows.append({
                "case_id": d["case_id"], "model": d["model_name"], "seed": seed,
                "param_count": d.get("param_count"),
                "RMSE_V": te["RMSE_V"], "R2_V": te["R2_V"],
                "RMSE_T": te["RMSE_T"], "R2_T": te["R2_T"],
                "temperature_peak_mae": te["temperature_peak_mae"],
                "coverage95_V": te.get("calibration", {}).get("coverage95_V"),
                "coverage95_T": te.get("calibration", {}).get("coverage95_T"),
            })
    return pd.DataFrame(rows)


def finalize_ts(run_dir: Path, out_dir: Path) -> Dict[str, Path]:
    df = _load_ts(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = (out_dir / "figures"); figs.mkdir(exist_ok=True)
    paths = {}
    if df.empty:
        return paths

    by_case = df.groupby(["model", "case_id"]).mean(numeric_only=True).reset_index()
    by_case.to_csv(out_dir / "metrics_by_model_and_case.csv", index=False)
    paths["by_case"] = out_dir / "metrics_by_model_and_case.csv"

    order = {m: i for i, m in enumerate(TS_MODEL_ORDER)}
    rank = (df.groupby("model").mean(numeric_only=True).reset_index())
    rank["display"] = rank["model"].map(lambda m: TS_DISPLAY.get(m, m))
    rank = rank.sort_values("RMSE_V").reset_index(drop=True)
    rank.insert(0, "rank", np.arange(1, len(rank) + 1))
    rank.to_csv(out_dir / "ranking_table.csv", index=False)
    paths["ranking"] = out_dir / "ranking_table.csv"

    # Calibration (bayesian only).
    cal = df[df["model"] == "bayesian_mlp"][
        ["case_id", "seed", "coverage95_V", "coverage95_T"]].dropna()
    if not cal.empty:
        cal.to_csv(out_dir / "calibration_bayesian.csv", index=False)
        paths["calibration"] = out_dir / "calibration_bayesian.csv"

    # Figure: RMSE_V / RMSE_T by model.
    g = df.groupby("model").mean(numeric_only=True).reindex(TS_MODEL_ORDER)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar([TS_DISPLAY[m] for m in g.index], g["RMSE_V"], color="#3b6")
    ax[0].set_title("Voltage RMSE by model (masked, test)"); ax[0].tick_params(axis="x", rotation=45)
    ax[1].bar([TS_DISPLAY[m] for m in g.index], g["RMSE_T"], color="#c64")
    ax[1].set_title("Temperature RMSE by model (masked, test)"); ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout(); fig.savefig(figs / "ts_rmse_by_model.png", dpi=120); plt.close(fig)
    paths["figure"] = figs / "ts_rmse_by_model.png"
    return paths


def finalize_em(combined_dir: Path, out_dir: Path) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = (out_dir / "figures"); figs.mkdir(exist_ok=True)
    paths = {}
    rk = combined_dir / "tables" / "ranking_table.csv"
    if rk.is_file():
        shutil.copy2(rk, out_dir / "ranking_table.csv")
        paths["ranking"] = out_dir / "ranking_table.csv"
        r = pd.read_csv(rk)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(r["display"], r["overall_norm_overall_RMSE_mean"], color="#558")
        ax.set_title("Error-metric normalized overall RMSE (8 families)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout(); fig.savefig(figs / "em_rmse_by_model.png", dpi=120); plt.close(fig)
        paths["figure"] = figs / "em_rmse_by_model.png"
    for extra in ("metrics_by_model.csv", "experiment_summary.md"):
        src = combined_dir / "tables" / extra
        if src.is_file():
            shutil.copy2(src, out_dir / extra)
    return paths


def make_zip(reports_dir: Path, zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(reports_dir.rglob("*")):
            if p.is_file() and p.resolve() != zip_path.resolve():
                z.write(p, p.relative_to(reports_dir.parent))
    return zip_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ts-run-dir", required=True)
    p.add_argument("--em-combined-dir", required=True)
    p.add_argument("--reports-dir", required=True)
    a = p.parse_args(argv)
    reports = Path(a.reports_dir)
    ts_paths = finalize_ts(Path(a.ts_run_dir), reports / "time_series")
    em_paths = finalize_em(Path(a.em_combined_dir), reports / "error_metric")
    zip_path = make_zip(reports, reports / "final_filtered_results.zip")
    print(f"[finalize] ts: {list(ts_paths)}")
    print(f"[finalize] em: {list(em_paths)}")
    print(f"[finalize] zip: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
