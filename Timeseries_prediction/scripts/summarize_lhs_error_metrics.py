#!/usr/bin/env python3
"""Summarize an LHS error-metric run.

Reads a run directory produced by ``scripts/lhs_error_metrics_train.py`` and
prints (and writes ``ERROR_METRICS_SUMMARY.md``) the ranking, timing and the
grouped (per-case / charge-vs-discharge / by-C-rate) metrics.

Usage
-----
python scripts/summarize_lhs_error_metrics.py                 # latest run
python scripts/summarize_lhs_error_metrics.py --run-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
POINTERS = ["LATEST_ERROR_METRICS_RUN.txt", "LATEST_ERROR_METRICS_SMOKE_RUN.txt"]


def resolve_run(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    for p in POINTERS:
        f = REPO / "outputs/lhs_1000_seed42" / p
        if f.exists():
            return Path(f.read_text().strip()).resolve()
    raise SystemExit("No --run-dir and no LATEST_ERROR_METRICS pointer found.")


def _csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = resolve_run(args.run_dir)
    if not run.is_dir():
        raise SystemExit(f"Run dir not found: {run}")

    ranking = _csv(run / "metrics" / "model_ranking.csv")
    timing = _csv(run / "metrics" / "model_timing.csv")
    per_case = _csv(run / "metrics" / "per_case_metrics.csv")
    counts_p = run / "artifacts" / "split_counts.json"
    counts = json.loads(counts_p.read_text()) if counts_p.exists() else {}

    lines: list[str] = [f"# Error-metric run summary — {run.name}", "",
                        f"- Run directory: `{run}`"]
    if counts:
        lines.append(f"- Split (rows): {counts}")

    if ranking is not None:
        cols = [c for c in ["display_name", "v_mae", "v_rmse", "v_r2",
                            "t_mae", "t_rmse", "t_r2", "macro_rmse", "average_rank"]
                if c in ranking.columns]
        lines += ["", "## Ranking (lowest average_rank = best)", "```text",
                  ranking[cols].to_string(index=False), "```",
                  f"\nBest model: **{ranking.iloc[0]['display_name']}**"]

    if timing is not None:
        cols = [c for c in ["display_name", "train_seconds",
                            "inference_ms_per_row", "parameter_count"]
                if c in timing.columns]
        lines += ["", "## Timing", "```text",
                  timing[cols].to_string(index=False), "```"]

    if per_case is not None and not per_case.empty:
        for kind, title in [("operation", "Charge vs discharge"),
                            ("c_rate", "By C-rate"),
                            ("experiment_id", "Per experiment case")]:
            sub = per_case[per_case["group_kind"] == kind]
            if sub.empty:
                continue
            # Best model per group by macro_rmse for a compact view.
            piv = (sub.pivot_table(index="group", columns="display_name",
                                   values="macro_rmse")
                   .round(3))
            lines += ["", f"## {title} — macro RMSE by model", "```text",
                      piv.to_string(), "```"]

    text = "\n".join(lines)
    print(text)
    out = Path(args.out) if args.out else (run / "ERROR_METRICS_SUMMARY.md")
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
