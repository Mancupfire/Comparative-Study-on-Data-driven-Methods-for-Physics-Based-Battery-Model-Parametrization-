"""Aggregate Batch 2 time-series metrics into summary tables + comparison report.

Reads <run-dir>/metrics/<case_id>/<model>/metrics.json (written by the existing
evaluate_case) and produces, inside <run-dir>:

    metrics_summary.csv     # one row per (case, model) with all 13 metrics
    metrics_by_model.csv    # mean over cases, per model
    metrics_by_target.csv   # voltage vs temperature blocks, per model
    experiment_summary.md

Also writes a clearly separated batch comparison under:
    outputs/comparisons/Batch_1_vs_Batch_2/<comparison_id>/

Usage
-----
python scripts/summarize_batch2.py --run-dir outputs/Data_Batch_2/time_series/<run_id>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

METRICS = ["MAE_V", "RMSE_V", "R2_V", "MaxError_V",
           "MAE_T", "RMSE_T", "R2_T", "MaxError_T",
           "voltage_end_mae", "temperature_end_mae", "temperature_peak_mae",
           "voltage_curve_rmse_mean", "temperature_curve_rmse_mean"]
V_METRICS = ["MAE_V", "RMSE_V", "R2_V", "MaxError_V", "voltage_end_mae",
             "voltage_curve_rmse_mean"]
T_METRICS = ["MAE_T", "RMSE_T", "R2_T", "MaxError_T", "temperature_end_mae",
             "temperature_peak_mae", "temperature_curve_rmse_mean"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--batch1-outputs", default="outputs",
                   help="Batch 1 outputs root (read-only, for comparison).")
    p.add_argument("--comparison-root", default="outputs/comparisons/Batch_1_vs_Batch_2")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mdir = run_dir / "metrics"
    rows = []
    if mdir.is_dir():
        for mj in sorted(mdir.glob("*/*/metrics.json")):
            try:
                rec = json.loads(mj.read_text())
            except Exception:
                continue
            row = {"case_id": rec.get("case_id"), "model_name": rec.get("model_name"),
                   "n_samples": rec.get("n_samples"), "t_last": rec.get("t_last")}
            for m in METRICS:
                row[m] = rec.get(m)
            rows.append(row)

    if not rows:
        print(f"[summarize] no metrics.json found under {mdir} yet.")
        # Still emit empty files so downstream never crashes.
        pd.DataFrame(columns=["case_id", "model_name", *METRICS]).to_csv(
            run_dir / "metrics_summary.csv", index=False)
        return 0

    df = pd.DataFrame(rows).sort_values(["case_id", "model_name"])
    df.to_csv(run_dir / "metrics_summary.csv", index=False)

    by_model = df.groupby("model_name")[METRICS].mean().reset_index()
    by_model.insert(1, "n_cases", df.groupby("model_name").size().values)
    by_model = by_model.sort_values("RMSE_V")
    by_model.to_csv(run_dir / "metrics_by_model.csv", index=False)

    # by target: long format
    bt = []
    for _, r in by_model.iterrows():
        for tgt, mlist in (("voltage", V_METRICS), ("temperature", T_METRICS)):
            rec = {"model_name": r["model_name"], "target": tgt}
            for m in mlist:
                rec[m] = r[m]
            bt.append(rec)
    pd.DataFrame(bt).to_csv(run_dir / "metrics_by_target.csv", index=False)

    # experiment summary
    n_cases = df["case_id"].nunique()
    n_models = df["model_name"].nunique()
    md = [
        "# Batch 2 — time-series experiment summary",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        f"\nRun dir: `{run_dir}`",
        f"\nCompleted (case, model) pairs: **{len(df)}**  "
        f"({n_cases} cases x {n_models} models)\n",
        "## Mean metrics per model (averaged across cases, ranked by RMSE_V)\n",
        by_model.round(4).to_markdown(index=False),
        "\n## Per (case, model)\n",
        df.round(4).to_markdown(index=False),
    ]
    (run_dir / "experiment_summary.md").write_text("\n".join(md) + "\n")
    print(f"[summarize] wrote metrics_summary.csv ({len(df)} rows), "
          f"metrics_by_model.csv, metrics_by_target.csv, experiment_summary.md")

    # ---- separated Batch 1 vs Batch 2 comparison ------------------------- #
    cid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    cdir = Path(args.comparison_root) / cid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "batch_2_metrics_summary.csv").write_text(
        (run_dir / "metrics_summary.csv").read_text())
    (cdir / "batch_2_metrics_by_model.csv").write_text(
        (run_dir / "metrics_by_model.csv").read_text())
    b1_cmp = Path(args.batch1_outputs) / "model_comparison.csv"
    note = ["# Batch 1 vs Batch 2 — SEPARATE experiment batches", "",
            "These are two DISTINCT datasets/experiments, not repeated runs of the "
            "same setup. Compare with care:",
            "* Different operating points (Batch 2 adds charge; c-rates 0.5/1.5/2.5 "
            "vs 0.5/1/2; temps 25/45 vs 10/25; only discharge-0.5C-25C overlaps).",
            "* Different parameter identities (12 vs 15 parameters).",
            "* Batch 2 uses native ~1s time grid (t_last up to 7030) vs Batch 1's "
            "downsampled ~150-180 grid.",
            "* Same modeling protocol, split, seed, loss and metric definitions.",
            "",
            "Files here are READ-ONLY COPIES; neither batch's model directories were "
            "modified."]
    if b1_cmp.is_file():
        (cdir / "batch_1_model_comparison.csv").write_text(b1_cmp.read_text())
        note.append(f"\nBatch 1 source: `{b1_cmp}` (copied).")
    else:
        note.append("\nBatch 1 `model_comparison.csv` not found; only Batch 2 copied.")
    (cdir / "README.md").write_text("\n".join(note) + "\n")
    print(f"[summarize] comparison written to {cdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
