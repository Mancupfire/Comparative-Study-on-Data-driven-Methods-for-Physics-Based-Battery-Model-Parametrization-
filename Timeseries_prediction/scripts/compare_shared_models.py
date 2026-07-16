"""Compare trained SHARED models against each other.

Reads, for every model under ``outputs/metrics/shared/<model_name>/``:

* ``metrics.json``            -> overall physical-unit metrics
* ``metrics_by_case.csv``     -> per-case metrics
* ``metrics_by_temp.csv``     -> per-ambient-temperature metrics
* ``metrics_by_c_rate.csv``   -> per-C-rate metrics

and writes the aggregated comparison tables:

* ``outputs/shared_model_comparison_overall.csv``
* ``outputs/shared_model_comparison_by_case.csv``
* ``outputs/shared_model_comparison_by_temp.csv``
* ``outputs/shared_model_comparison_by_c_rate.csv``

Usage
-----
python scripts/compare_shared_models.py --outputs-dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import load_json

# Columns carried into every comparison table.
COMPARE_COLUMNS = [
    "model_name", "n_cases",
    "MAE_V", "RMSE_V", "R2_V",
    "MAE_T", "RMSE_T", "R2_T",
    "voltage_end_mae", "temperature_end_mae", "temperature_peak_mae",
    "score_RMSE_V_plus_RMSE_T", "rank",
]

_FLOAT_FMT = lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else v


def _discover_models(metrics_root: Path) -> List[str]:
    return [p.name for p in sorted(metrics_root.iterdir())
            if p.is_dir() and (p / "metrics.json").is_file()]


def _row_from_metrics(model_name: str, metrics: dict) -> dict:
    rmse_v = float(metrics.get("RMSE_V", np.nan))
    rmse_t = float(metrics.get("RMSE_T", np.nan))
    return {
        "model_name": model_name,
        "n_cases": metrics.get("n_cases", np.nan),
        "MAE_V": metrics.get("MAE_V", np.nan),
        "RMSE_V": rmse_v,
        "R2_V": metrics.get("R2_V", np.nan),
        "MAE_T": metrics.get("MAE_T", np.nan),
        "RMSE_T": rmse_t,
        "R2_T": metrics.get("R2_T", np.nan),
        "voltage_end_mae": metrics.get("voltage_end_mae", np.nan),
        "temperature_end_mae": metrics.get("temperature_end_mae", np.nan),
        "temperature_peak_mae": metrics.get("temperature_peak_mae", np.nan),
        "score_RMSE_V_plus_RMSE_T": rmse_v + rmse_t,
    }


def _rank(df: pd.DataFrame, within: Optional[str] = None) -> pd.DataFrame:
    """Add a 1-based ``rank`` column by ``score`` (lower is better)."""
    df = df.copy()
    if within is None:
        df = df.sort_values("score_RMSE_V_plus_RMSE_T").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    else:
        df["rank"] = (
            df.groupby(within)["score_RMSE_V_plus_RMSE_T"]
            .rank(method="min", ascending=True)
            .astype(int)
        )
        df = df.sort_values([within, "rank"]).reset_index(drop=True)
    return df


def _grouped_table(
    metrics_root: Path, models: List[str], filename: str, group_col: str
) -> pd.DataFrame:
    """Stack a per-group CSV across models and rank within each group value."""
    rows = []
    for model_name in models:
        csv_path = metrics_root / model_name / filename
        if not csv_path.is_file():
            print(f"WARNING: missing {csv_path}; skipping for {model_name}.")
            continue
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            rmse_v = float(r.get("RMSE_V", np.nan))
            rmse_t = float(r.get("RMSE_T", np.nan))
            rows.append({
                "model_name": model_name,
                group_col: r[group_col],
                "n_curves": r.get("n_curves", np.nan),
                "MAE_V": r.get("MAE_V", np.nan),
                "RMSE_V": rmse_v,
                "R2_V": r.get("R2_V", np.nan),
                "MAE_T": r.get("MAE_T", np.nan),
                "RMSE_T": rmse_t,
                "R2_T": r.get("R2_T", np.nan),
                "voltage_end_mae": r.get("voltage_end_mae", np.nan),
                "temperature_end_mae": r.get("temperature_end_mae", np.nan),
                "temperature_peak_mae": r.get("temperature_peak_mae", np.nan),
                "score_RMSE_V_plus_RMSE_T": rmse_v + rmse_t,
            })
    if not rows:
        return pd.DataFrame()
    return _rank(pd.DataFrame(rows), within=group_col)


def _print_overall(df: pd.DataFrame) -> None:
    cols = ["model_name", "n_cases", "RMSE_V", "RMSE_T", "R2_V", "R2_T",
            "score_RMSE_V_plus_RMSE_T", "rank"]
    with pd.option_context("display.float_format", _FLOAT_FMT,
                           "display.max_rows", None, "display.width", 200):
        print("\nA. Shared-model overall ranking (sorted by RMSE_V + RMSE_T, lower is better):")
        print(df[cols].to_string(index=False))


def _print_grouped(title: str, df: pd.DataFrame, group_col: str) -> None:
    if df.empty:
        return
    cols = ["model_name", group_col, "RMSE_V", "RMSE_T", "R2_V", "R2_T",
            "score_RMSE_V_plus_RMSE_T", "rank"]
    with pd.option_context("display.float_format", _FLOAT_FMT,
                           "display.max_rows", None, "display.width", 200):
        print(f"\n{title}")
        print(df[cols].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare shared models from metrics files.")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics_root = outputs_dir / "metrics" / "shared"
    if not metrics_root.is_dir():
        print(f"No shared metrics directory found at {metrics_root}. Train shared models first.")
        return 1

    models = _discover_models(metrics_root)
    if not models:
        print(f"No shared model metrics.json found under {metrics_root}.")
        return 1

    written: List[Path] = []

    # --- Overall ----------------------------------------------------------- #
    overall_rows = [
        _row_from_metrics(m, load_json(metrics_root / m / "metrics.json")) for m in models
    ]
    overall = _rank(pd.DataFrame(overall_rows))[COMPARE_COLUMNS]
    overall_csv = outputs_dir / "shared_model_comparison_overall.csv"
    overall.to_csv(overall_csv, index=False)
    written.append(overall_csv)

    # --- Grouped tables ---------------------------------------------------- #
    by_case = _grouped_table(metrics_root, models, "metrics_by_case.csv", "case_id")
    by_case_csv = outputs_dir / "shared_model_comparison_by_case.csv"
    by_case.to_csv(by_case_csv, index=False)
    written.append(by_case_csv)

    by_temp = _grouped_table(metrics_root, models, "metrics_by_temp.csv", "ambient_temp_C")
    by_temp_csv = outputs_dir / "shared_model_comparison_by_temp.csv"
    by_temp.to_csv(by_temp_csv, index=False)
    written.append(by_temp_csv)

    by_c_rate = _grouped_table(metrics_root, models, "metrics_by_c_rate.csv", "c_rate")
    by_c_rate_csv = outputs_dir / "shared_model_comparison_by_c_rate.csv"
    by_c_rate.to_csv(by_c_rate_csv, index=False)
    written.append(by_c_rate_csv)

    # --- Console reporting ------------------------------------------------- #
    _print_overall(overall)
    _print_grouped("B. Shared-model ranking by ambient temperature:", by_temp, "ambient_temp_C")
    _print_grouped("C. Shared-model ranking by C-rate:", by_c_rate, "c_rate")

    if not by_case.empty:
        best_idx = by_case.groupby("case_id")["score_RMSE_V_plus_RMSE_T"].idxmin()
        best = by_case.loc[best_idx].sort_values("case_id")
        best = best[["case_id", "model_name", "RMSE_V", "RMSE_T", "R2_V", "R2_T",
                     "score_RMSE_V_plus_RMSE_T"]].rename(
            columns={"model_name": "best_model", "score_RMSE_V_plus_RMSE_T": "score"}
        )
        with pd.option_context("display.float_format", _FLOAT_FMT,
                               "display.max_rows", None, "display.width", 200):
            print("\nD. Best shared model per case:")
            print(best.to_string(index=False))

    print("\nGenerated CSV files:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
