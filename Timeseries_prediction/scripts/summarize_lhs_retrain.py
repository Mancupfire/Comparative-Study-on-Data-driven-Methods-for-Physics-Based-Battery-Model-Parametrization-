#!/usr/bin/env python
"""Combine one time-series run and one error-metric run into a retrain bundle.

Reads the per-branch ``metrics/model_timing.csv`` and ``metrics/model_ranking.csv``
frames, tags them with their branch, and writes the combined bundle artifacts:

    <bundle>/time_series_run.txt
    <bundle>/error_metrics_run.txt
    <bundle>/combined_model_timing.csv
    <bundle>/combined_model_ranking.csv
    <bundle>/COMBINED_SUMMARY.md

This is a pure post-processing/reporting step; it never retrains anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() else None


def _tag(df: Optional[pd.DataFrame], branch: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    out = df.copy()
    out.insert(0, "branch", branch)
    return out


def _ranking_line(run: Path, branch: str) -> str:
    rk = _read_csv(run / "metrics" / "model_ranking.csv")
    if rk is None or rk.empty:
        return f"- {branch}: (no ranking found)"
    best = rk.iloc[0]
    name = best.get("display_name", best.get("model", "?"))
    return f"- {branch}: best model **{name}** (avg rank {best.get('average_rank', float('nan')):.3f})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize an LHS retrain bundle.")
    ap.add_argument("--time-series-run", type=Path, required=True)
    ap.add_argument("--error-metrics-run", type=Path, required=True)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    args = ap.parse_args()

    ts, em, bundle = args.time_series_run, args.error_metrics_run, args.bundle_dir
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "time_series_run.txt").write_text(str(ts) + "\n", encoding="utf-8")
    (bundle / "error_metrics_run.txt").write_text(str(em) + "\n", encoding="utf-8")

    # Combined timing.
    timings = [
        _tag(_read_csv(ts / "metrics" / "model_timing.csv"), "time_series"),
        _tag(_read_csv(em / "metrics" / "model_timing.csv"), "error_metrics"),
    ]
    timings = [t for t in timings if t is not None]
    if timings:
        pd.concat(timings, ignore_index=True).to_csv(
            bundle / "combined_model_timing.csv", index=False
        )

    # Combined ranking.
    rankings = [
        _tag(_read_csv(ts / "metrics" / "model_ranking.csv"), "time_series"),
        _tag(_read_csv(em / "metrics" / "model_ranking.csv"), "error_metrics"),
    ]
    rankings = [r for r in rankings if r is not None]
    if rankings:
        pd.concat(rankings, ignore_index=True).to_csv(
            bundle / "combined_model_ranking.csv", index=False
        )

    # Combined summary.
    def _facts(run: Path) -> str:
        aud = run / "dataset_audit.json"
        if not aud.exists():
            return "(no dataset_audit.json)"
        a = json.loads(aud.read_text())
        gs = a.get("generation_summary", {})
        sc = a.get("split_counts", {})
        return (
            f"samples={gs.get('n_requested_samples')} "
            f"sequences={gs.get('n_successful_sequences')} "
            f"failed={gs.get('n_failed_sequences')} "
            f"split={sc}"
        )

    lines = [
        "# Combined LHS Retraining Summary",
        "",
        f"- Time-series run: `{ts}`",
        f"- Error-metric run: `{em}`",
        "",
        "## Dataset audit",
        f"- time_series: {_facts(ts)}",
        f"- error_metrics: {_facts(em)}",
        "",
        "## Best models",
        _ranking_line(ts, "time_series"),
        _ranking_line(em, "error_metrics"),
        "",
        "## Artifacts",
        "- combined_model_timing.csv",
        "- combined_model_ranking.csv",
        "- time_series_run.txt / error_metrics_run.txt",
    ]
    (bundle / "COMBINED_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summarize] wrote bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
