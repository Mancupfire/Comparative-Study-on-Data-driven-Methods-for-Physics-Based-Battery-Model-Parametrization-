#!/usr/bin/env python3
"""Summarize an official LHS time-series run.

Reads a run directory produced by ``scripts/emergency_lhs_train.py`` and prints
(and writes) a compact summary: alignment mode, split counts, excluded-sequence
audit, the model ranking and per-case metrics.

Usage
-----
# Summarize the latest official full run (path from LATEST_OFFICIAL_RUN.txt):
python scripts/summarize_lhs_official.py

# Or a specific run directory:
python scripts/summarize_lhs_official.py --run-dir outputs/lhs_1000_seed42/time_series/<run_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POINTER = REPO / "outputs/lhs_1000_seed42/LATEST_OFFICIAL_RUN.txt"


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def resolve_run_dir(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    if DEFAULT_POINTER.exists():
        return Path(DEFAULT_POINTER.read_text().strip()).resolve()
    raise SystemExit(
        f"No --run-dir given and pointer not found: {DEFAULT_POINTER}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize an official LHS run.")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--out", default=None, help="Markdown output path (optional).")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    ranking = _read_csv(run_dir / "metrics" / "model_ranking.csv")
    per_case = _read_csv(run_dir / "metrics" / "per_case_metrics.csv")
    split_counts_path = run_dir / "artifacts" / "split_counts.json"
    split_counts = json.loads(split_counts_path.read_text()) if split_counts_path.exists() else {}
    excluded = _read_csv(run_dir / "artifacts" / "excluded_sequences.csv")
    selected = _read_csv(run_dir / "artifacts" / "selected_manifest.csv")

    lines: list[str] = []
    lines.append(f"# Official LHS run summary — {run_dir.name}")
    lines.append("")
    lines.append(f"- Run directory: `{run_dir}`")
    lines.append(f"- Alignment mode: **{cfg.get('alignment_mode', 'unknown')}**")
    lines.append(f"- Sequence length: {cfg.get('sequence_length', '?')}")
    lines.append(f"- Models: {', '.join(cfg.get('models', []))}")
    if selected is not None:
        lines.append(f"- Selected sequences: {len(selected)} "
                     f"({selected['sample_id'].nunique()} sample_ids)")
    n_excl = 0 if excluded is None else len(excluded)
    lines.append(f"- Excluded (structurally invalid) sequences: {n_excl}")
    if excluded is not None and n_excl:
        counts = excluded["reason"].value_counts().to_dict()
        lines.append(f"  - reasons: {counts}")
    if split_counts:
        lines.append("")
        lines.append("## Split counts")
        lines.append("```json")
        lines.append(json.dumps(split_counts, indent=2))
        lines.append("```")

    if ranking is not None:
        cols = [c for c in ["display_name", "v_rmse", "v_r2", "t_rmse", "t_r2", "average_rank"]
                if c in ranking.columns]
        lines.append("")
        lines.append("## Model ranking")
        lines.append("```text")
        lines.append(ranking[cols].to_string(index=False))
        lines.append("```")
        lines.append(f"\nBest model: **{ranking.iloc[0].get('display_name', '?')}**")

    if per_case is not None and not per_case.empty:
        cols = [c for c in ["model", "experiment_id", "operation", "c_rate",
                            "v_rmse", "t_rmse", "n_sequences"] if c in per_case.columns]
        lines.append("")
        lines.append("## Per-case metrics")
        lines.append("```text")
        lines.append(per_case[cols].to_string(index=False))
        lines.append("```")

    text = "\n".join(lines)
    print(text)

    out_path = Path(args.out) if args.out else (run_dir / "OFFICIAL_SUMMARY.md")
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
