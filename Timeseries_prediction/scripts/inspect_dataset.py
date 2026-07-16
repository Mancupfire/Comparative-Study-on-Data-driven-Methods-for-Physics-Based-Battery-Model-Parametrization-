"""Inspect and validate the generated training dataset.

Usage
-----
python scripts/inspect_dataset.py --data-root generate_training_data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src`` importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data import (
    PARAM_CSV_NAME,
    discover_cases,
    load_aligned_case_data,
    load_parameter_table,
)


def _failed_counts(data_root: Path) -> dict:
    """Return ``{experiment_id: failed_count}`` from failed_cases.csv if present."""
    path = data_root / "failed_cases.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - reporting only
        print(f"  (could not read {path.name}: {exc})")
        return {}
    if "experiment_id" not in df.columns:
        return {}
    return df["experiment_id"].value_counts().to_dict()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the battery time-series dataset.")
    parser.add_argument("--data-root", default="generate_training_data")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    print(f"Data root: {data_root.resolve()}")

    params = load_parameter_table(data_root)
    print(f"{PARAM_CSV_NAME}: {params.shape[0]} samples x {params.shape[1]} parameters")

    cases = discover_cases(data_root)
    print(f"Discovered {len(cases)} case(s): {cases}\n")

    failed = _failed_counts(data_root)
    all_ok = True

    for case_id in cases:
        print(f"=== {case_id} ===")
        try:
            case = load_aligned_case_data(data_root, case_id)
        except Exception as exc:  # noqa: BLE001 - report and continue
            all_ok = False
            print(f"  VALIDATION FAILED: {exc}\n")
            continue

        print(f"  samples (N_ok)     : {case.n_samples}")
        print(f"  time steps (t_last): {case.t_last}")
        print(f"  parameters         : {case.n_parameters}")
        print(f"  voltage     min/max: {case.V.min():.4f} / {case.V.max():.4f} V")
        print(f"  temperature min/max: {case.T.min():.4f} / {case.T.max():.4f} C")
        print(f"  time        min/max: {case.time_s.min():.4f} / {case.time_s.max():.4f} s")
        if case_id in failed:
            print(f"  failed simulations : {failed[case_id]}")

        # Explicit alignment checks (load_aligned_case_data already validates,
        # but we re-assert here so the report is self-contained).
        assert case.sample_ids.shape[0] == case.V.shape[0] == case.T.shape[0]
        assert case.V.shape == case.T.shape
        assert case.time_s.shape[0] == case.V.shape[1]
        if np.isnan(case.V).any() or np.isnan(case.T).any():
            all_ok = False
            print("  WARNING: NaNs detected in voltage/temperature arrays")
        print("  alignment + shape checks: OK\n")

    print("All cases validated successfully." if all_ok else "Some cases FAILED validation.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
