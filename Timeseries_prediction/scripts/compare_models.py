"""Aggregate every metrics.json into clean comparison tables.

Reads ``outputs/metrics/<case_id>/<model_name>/metrics.json`` recursively and
produces two families of reports:

* **Deep learning full-curve models** (mlp, rnn, lstm, bilstm) are compared on
  the full voltage/temperature curve (RMSE_V / RMSE_T / R2_V / R2_T ...). These
  get a per-case table plus aggregated tables averaged across cases, by ambient
  temperature, and by C-rate.
* **AutoML reduced-output baseline** (automl_trees_reduced) predicts only 10
  summary targets, so it is reported *separately* and is never forced into the
  full-curve ranking.

Usage
-----
python scripts/compare_models.py --outputs-dir outputs
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils import load_json

# ---------------------------------------------------------------------------
# Column layouts
# ---------------------------------------------------------------------------

# Deep-learning full-curve metrics carried straight from metrics.json.
DEEP_METRIC_COLUMNS = [
    "MAE_V",
    "RMSE_V",
    "R2_V",
    "MaxError_V",
    "MAE_T",
    "RMSE_T",
    "R2_T",
    "MaxError_T",
    "voltage_end_mae",
    "temperature_end_mae",
    "temperature_peak_mae",
    "voltage_curve_rmse_mean",
    "temperature_curve_rmse_mean",
]

# Ordered per-case columns for the full-curve table.
FULL_CURVE_COLUMNS = [
    "case_id",
    "model_name",
    "c_rate",
    "ambient_temp_C",
    *DEEP_METRIC_COLUMNS,
    "score_RMSE_V_plus_RMSE_T",
]

# AutoML reduced summary targets (order matters for the output table).
AUTOML_TARGETS = [
    "V_start",
    "V_mid",
    "V_end",
    "V_min",
    "V_mean",
    "T_start",
    "T_mid",
    "T_end",
    "T_max",
    "T_mean",
]

# Reference list of known full-curve deep models (documentation only). The
# comparison is data-driven: any metrics.json exposing RMSE_V/RMSE_T/R2_V/R2_T
# is included automatically (see ``is_deep_metric``), so new models such as
# cnn / cnn_bilstm / bayesian_mlp appear without code changes.
DEEP_MODEL_NAMES = [
    "mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp",
]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_case_id(case_id: str) -> Dict[str, Optional[float]]:
    """Extract C-rate and ambient temperature from a case_id string.

    Examples
    --------
    cc_dchg_0p5_10degC -> c_rate=0.5,  ambient_temp_C=10
    cc_dchg_1C_25degC  -> c_rate=1.0,  ambient_temp_C=25
    cc_dchg_2C_10degC  -> c_rate=2.0,  ambient_temp_C=10

    Also supports future-style IDs such as ``3C`` or ``1p5C``.
    """
    c_rate: Optional[float] = None
    ambient_temp_C: Optional[float] = None

    # Ambient temperature: a number immediately followed by "degC".
    temp_match = re.search(r"(-?\d+(?:[p.]\d+)?)\s*degC", case_id, flags=re.IGNORECASE)
    if temp_match:
        ambient_temp_C = float(temp_match.group(1).replace("p", "."))

    # C-rate: a number token optionally followed by a "C" marker.  "0p5" or
    # "1p5C" use 'p' as a decimal point; a trailing "C" is the C-rate unit.
    for token in case_id.split("_"):
        candidate = token
        if candidate.lower().endswith("c") and candidate.lower() != "cc":
            candidate = candidate[:-1]  # strip the C-rate unit marker
        normalized = candidate.replace("p", ".")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", normalized):
            # Skip the temperature token (it carries the "degC" suffix, already
            # consumed above) and the leading "cc"/"dchg" descriptors.
            if "degc" in token.lower():
                continue
            # A bare number with no 'C' marker is only a C-rate when it used the
            # 'p' decimal form (e.g. 0p5); plain integers like the temperature
            # are excluded by the degC check above.
            if token.lower().endswith("c") or "p" in token.lower():
                c_rate = float(normalized)
                break

    return {"c_rate": c_rate, "ambient_temp_C": ambient_temp_C}


def load_metric_files(outputs_dir: Path) -> List[Dict[str, Any]]:
    """Recursively load every metrics.json under ``outputs/metrics``.

    Returns a list of raw json records, each annotated with ``case_id`` and
    ``model_name`` derived from the directory layout (robust to file content).
    """
    metrics_root = Path(outputs_dir) / "metrics"
    records: List[Dict[str, Any]] = []
    for metrics_file in sorted(metrics_root.glob("*/*/metrics.json")):
        # The per-case layout is metrics/<case_id>/<model_name>/. The shared
        # pipeline writes metrics/shared/<model_name>/ which also matches this
        # glob but is NOT a per-case result (its "case_id" would be "shared",
        # yielding NaN c_rate/ambient_temp_C and breaking the grouped rankings).
        # Shared models are compared separately by compare_shared_models.py.
        if metrics_file.parent.parent.name == "shared":
            continue
        try:
            record = load_json(metrics_file)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"WARNING: could not read {metrics_file}: {exc}")
            continue
        record.setdefault("case_id", metrics_file.parent.parent.name)
        record.setdefault("model_name", metrics_file.parent.name)
        records.append(record)
    return records


def is_deep_metric(record: Dict[str, Any]) -> bool:
    """Deep full-curve metrics expose RMSE_V / RMSE_T / R2_V / R2_T."""
    return all(key in record for key in ("RMSE_V", "RMSE_T", "R2_V", "R2_T"))


def is_automl_reduced_metric(record: Dict[str, Any]) -> bool:
    """AutoML reduced metrics carry mode == 'reduced' and a per_target block."""
    return record.get("mode") == "reduced" and isinstance(record.get("per_target"), dict)


def flatten_deep_metrics(metric_json: Dict[str, Any]) -> Dict[str, Any]:
    """Build one full-curve table row from a deep-learning metrics record."""
    case_id = metric_json.get("case_id", "")
    parsed = parse_case_id(case_id)
    row: Dict[str, Any] = {
        "case_id": case_id,
        "model_name": metric_json.get("model_name", ""),
        "c_rate": parsed["c_rate"],
        "ambient_temp_C": parsed["ambient_temp_C"],
    }
    for col in DEEP_METRIC_COLUMNS:
        # Missing optional deep metrics become NaN rather than crashing.
        value = metric_json.get(col, np.nan)
        row[col] = float(value) if value is not None else np.nan

    rmse_v = row.get("RMSE_V", np.nan)
    rmse_t = row.get("RMSE_T", np.nan)
    row["score_RMSE_V_plus_RMSE_T"] = float(rmse_v) + float(rmse_t)
    return row


def flatten_automl_reduced_metrics(metric_json: Dict[str, Any]) -> Dict[str, Any]:
    """Build one row from an AutoML reduced metrics record."""
    case_id = metric_json.get("case_id", "")
    parsed = parse_case_id(case_id)
    row: Dict[str, Any] = {
        "case_id": case_id,
        "model_name": metric_json.get("model_name", "automl_trees_reduced"),
        "mode": metric_json.get("mode"),
        "estimator": metric_json.get("estimator"),
        "c_rate": parsed["c_rate"],
        "ambient_temp_C": parsed["ambient_temp_C"],
        "n_test": metric_json.get("n_test"),
    }
    per_target = metric_json.get("per_target", {})
    for target in AUTOML_TARGETS:
        target_metrics = per_target.get(target, {})
        row[f"{target}_MAE"] = target_metrics.get("MAE", np.nan)
        row[f"{target}_RMSE"] = target_metrics.get("RMSE", np.nan)
    return row


# ---------------------------------------------------------------------------
# Deep-learning aggregation
# ---------------------------------------------------------------------------

# Metrics summarised as mean+std in the overall table.
_MEANSTD_METRICS = ["MAE_V", "RMSE_V", "R2_V", "MAE_T", "RMSE_T", "R2_T"]
# Metrics summarised as mean-only.
_MEAN_ONLY_METRICS = [
    "voltage_end_mae",
    "temperature_end_mae",
    "temperature_peak_mae",
    "voltage_curve_rmse_mean",
    "temperature_curve_rmse_mean",
    "score_RMSE_V_plus_RMSE_T",
]


def _aggregate(df: pd.DataFrame, group_cols: List[str],
               meanstd: List[str], mean_only: List[str]) -> pd.DataFrame:
    """Group ``df`` and compute mean (+optional std) for the chosen metrics."""
    grouped = df.groupby(group_cols, dropna=False)
    out = pd.DataFrame({"n_cases": grouped.size()})
    for metric in meanstd:
        out[f"{metric}_mean"] = grouped[metric].mean()
        out[f"{metric}_std"] = grouped[metric].std()
    for metric in mean_only:
        out[f"{metric}_mean"] = grouped[metric].mean()
    return out.reset_index()


def summarize_deep_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Average across all cases per deep model, ranked by combined RMSE."""
    out = _aggregate(df, ["model_name"], _MEANSTD_METRICS, _MEAN_ONLY_METRICS)
    out = out.sort_values("score_RMSE_V_plus_RMSE_T_mean").reset_index(drop=True)
    out["rank_by_score"] = np.arange(1, len(out) + 1)

    columns = [
        "model_name", "n_cases",
        "MAE_V_mean", "MAE_V_std", "RMSE_V_mean", "RMSE_V_std",
        "R2_V_mean", "R2_V_std",
        "MAE_T_mean", "MAE_T_std", "RMSE_T_mean", "RMSE_T_std",
        "R2_T_mean", "R2_T_std",
        "voltage_end_mae_mean", "temperature_end_mae_mean",
        "temperature_peak_mae_mean", "voltage_curve_rmse_mean_mean",
        "temperature_curve_rmse_mean_mean", "score_RMSE_V_plus_RMSE_T_mean",
        "rank_by_score",
    ]
    return out[columns]


def _summarize_deep_by(df: pd.DataFrame, group_col: str, rank_col: str) -> pd.DataFrame:
    """Shared aggregation for the by-temp / by-c-rate tables."""
    meanstd: List[str] = []  # these tables use mean-only columns
    mean_only = [
        "RMSE_V", "RMSE_T", "R2_V", "R2_T", "MAE_V", "MAE_T",
        "voltage_end_mae", "temperature_end_mae", "temperature_peak_mae",
        "score_RMSE_V_plus_RMSE_T",
    ]
    out = _aggregate(df, ["model_name", group_col], meanstd, mean_only)
    # Rank models within each group by combined RMSE (lower is better).
    out[rank_col] = (
        out.groupby(group_col)["score_RMSE_V_plus_RMSE_T_mean"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    out = out.sort_values([group_col, rank_col]).reset_index(drop=True)

    columns = [
        "model_name", group_col, "n_cases",
        "RMSE_V_mean", "RMSE_T_mean", "R2_V_mean", "R2_T_mean",
        "MAE_V_mean", "MAE_T_mean",
        "voltage_end_mae_mean", "temperature_end_mae_mean",
        "temperature_peak_mae_mean", "score_RMSE_V_plus_RMSE_T_mean",
        rank_col,
    ]
    return out[columns]


def summarize_deep_by_temp(df: pd.DataFrame) -> pd.DataFrame:
    """Average by model and ambient temperature, ranked within each temp."""
    return _summarize_deep_by(df, "ambient_temp_C", "rank_within_temp")


def summarize_deep_by_c_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Average by model and C-rate, ranked within each C-rate."""
    return _summarize_deep_by(df, "c_rate", "rank_within_c_rate")


# ---------------------------------------------------------------------------
# AutoML aggregation
# ---------------------------------------------------------------------------

def _automl_metric_columns() -> List[str]:
    cols: List[str] = []
    for target in AUTOML_TARGETS:
        cols.append(f"{target}_MAE")
        cols.append(f"{target}_RMSE")
    return cols


def summarize_automl(df: pd.DataFrame, group_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Average AutoML reduced metrics, optionally grouped.

    With ``group_cols=None`` returns a single overall-average row; otherwise
    averages within each group (e.g. by ambient_temp_C).
    """
    metric_cols = _automl_metric_columns()
    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
        out = pd.DataFrame({"n_cases": grouped.size()})
        for col in metric_cols:
            out[col] = grouped[col].mean()
        return out.reset_index()

    summary = {"n_cases": len(df)}
    for col in metric_cols:
        summary[col] = df[col].mean()
    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Console printing
# ---------------------------------------------------------------------------

_FLOAT_FMT = lambda v: f"{v:.4f}" if isinstance(v, (int, float, np.floating)) else v


def _print_ranking(title: str, frame: pd.DataFrame) -> None:
    cols = ["model_name", "n_cases", "RMSE_V_mean", "RMSE_T_mean",
            "R2_V_mean", "R2_T_mean", "score_RMSE_V_plus_RMSE_T_mean"]
    rank_col = "rank_by_score" if "rank_by_score" in frame.columns else frame.columns[-1]
    cols = cols + [rank_col]
    with pd.option_context("display.float_format", _FLOAT_FMT,
                           "display.max_rows", None, "display.width", 200):
        print(f"\n{title}")
        print(frame[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Compare trained models from metrics.json files.")
    parser.add_argument("--outputs-dir", default="outputs")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    metrics_root = outputs_dir / "metrics"
    if not metrics_root.is_dir():
        print(f"No metrics directory found at {metrics_root}. Train some models first.")
        return 1

    records = load_metric_files(outputs_dir)
    if not records:
        print(f"No metrics.json files found under {metrics_root}.")
        return 1

    deep_rows = [flatten_deep_metrics(r) for r in records if is_deep_metric(r)]
    automl_rows = [flatten_automl_reduced_metrics(r) for r in records if is_automl_reduced_metric(r)]

    if not deep_rows:
        print("WARNING: no deep-learning full-curve metrics found "
              "(expected keys RMSE_V/RMSE_T/R2_V/R2_T).")
    if not automl_rows:
        print("WARNING: no AutoML reduced metrics found "
              "(expected mode=='reduced' with a per_target block).")

    written: List[Path] = []

    # --- Deep learning tables -------------------------------------------------
    deep_overall = deep_by_temp = None
    if deep_rows:
        deep_df = pd.DataFrame(deep_rows, columns=FULL_CURVE_COLUMNS)
        deep_df = deep_df.sort_values(["case_id", "model_name"]).reset_index(drop=True)

        full_curve_csv = outputs_dir / "model_comparison_full_curve.csv"
        deep_df.to_csv(full_curve_csv, index=False)
        written.append(full_curve_csv)

        # Backward-compatible alias (same content, no AutoML NaN rows).
        legacy_csv = outputs_dir / "model_comparison.csv"
        deep_df.to_csv(legacy_csv, index=False)
        written.append(legacy_csv)

        deep_overall = summarize_deep_overall(deep_df)
        overall_csv = outputs_dir / "model_comparison_by_model_overall.csv"
        deep_overall.to_csv(overall_csv, index=False)
        written.append(overall_csv)

        deep_by_temp = summarize_deep_by_temp(deep_df)
        by_temp_csv = outputs_dir / "model_comparison_by_temp.csv"
        deep_by_temp.to_csv(by_temp_csv, index=False)
        written.append(by_temp_csv)

        deep_by_c_rate = summarize_deep_by_c_rate(deep_df)
        by_c_rate_csv = outputs_dir / "model_comparison_by_c_rate.csv"
        deep_by_c_rate.to_csv(by_c_rate_csv, index=False)
        written.append(by_c_rate_csv)

    # --- AutoML reduced tables ------------------------------------------------
    if automl_rows:
        automl_df = pd.DataFrame(automl_rows)
        automl_df = automl_df.sort_values(["case_id"]).reset_index(drop=True)

        automl_csv = outputs_dir / "automl_reduced_comparison.csv"
        automl_df.to_csv(automl_csv, index=False)
        written.append(automl_csv)

        automl_by_temp = summarize_automl(automl_df, group_cols=["ambient_temp_C"])
        automl_by_temp_csv = outputs_dir / "automl_reduced_summary_by_temp.csv"
        automl_by_temp.to_csv(automl_by_temp_csv, index=False)
        written.append(automl_by_temp_csv)

        automl_overall = summarize_automl(automl_df)
        automl_overall_csv = outputs_dir / "automl_reduced_summary_overall.csv"
        automl_overall.to_csv(automl_overall_csv, index=False)
        written.append(automl_overall_csv)

    # --- Console reporting ----------------------------------------------------
    if deep_overall is not None:
        _print_ranking("A. Full-curve deep learning overall ranking "
                        "(sorted by RMSE_V + RMSE_T, lower is better):", deep_overall)

        for temp, label in ((25.0, "B. Full-curve ranking at 25degC:"),
                            (10.0, "C. Full-curve ranking at 10degC:")):
            subset = deep_by_temp[deep_by_temp["ambient_temp_C"] == temp]
            if not subset.empty:
                ranked = subset.sort_values("rank_within_temp")
                _print_ranking(label, ranked)

        # D. Best model per case.
        deep_df_full = pd.DataFrame(deep_rows, columns=FULL_CURVE_COLUMNS)
        best_idx = deep_df_full.groupby("case_id")["score_RMSE_V_plus_RMSE_T"].idxmin()
        best = deep_df_full.loc[best_idx].sort_values("case_id")
        best_cols = ["case_id", "model_name", "RMSE_V", "RMSE_T", "R2_V", "R2_T",
                     "score_RMSE_V_plus_RMSE_T"]
        best = best[best_cols].rename(columns={
            "model_name": "best_model",
            "score_RMSE_V_plus_RMSE_T": "score",
        })
        with pd.option_context("display.float_format", _FLOAT_FMT,
                               "display.max_rows", None, "display.width", 200):
            print("\nD. Best model per case:")
            print(best.to_string(index=False))

    # E. AutoML note.
    print("\nE. AutoML note:")
    print("AutoML reduced-output baseline is reported separately because it "
          "predicts summary targets rather than full voltage/temperature "
          "curves. It is not directly ranked against full-curve deep learning "
          "models.")

    # --- Written file list ----------------------------------------------------
    print("\nGenerated CSV files:")
    for path in written:
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
