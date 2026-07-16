"""Read-only validation of the Data_Batch_4 raw dataset (no modification).

Checks every requirement in the Batch 4 setup brief:
  * time-series data rows == 4,001,000
  * parameter sets == 1000
  * sequences == 12000
  * cases == 12
  * failed sequences == 0
  * no missing / non-finite values in time_series.csv
  * manifest <-> parameter <-> error-metric joins all match
  * n_extrapolated_points preserved and reported (held boundary values).

Usage:
  python scripts/validate_batch_4_raw.py --data-dir data/Data_Batch_4_raw
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECT = dict(ts_rows=4_001_000, params=1000, sequences=12000, cases=12, failed=0)
TS_DTYPES = {"sequence_id": "string", "time_index": "int32",
             "time_s": "float64", "voltage_v": "float64", "temperature_c": "float64"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Data_Batch_4_raw")
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()
    d = Path(args.data_dir)

    report: dict = {"data_dir": str(d.resolve()), "checks": {}, "n_extrapolated_points": {}}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        report["checks"][name] = {"pass": bool(cond), "detail": str(detail)}
        ok = ok and bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    params = pd.read_csv(d / "parameter_sets.csv")
    manifest = pd.read_csv(d / "sequence_manifest.csv")
    metrics = pd.read_csv(d / "error_metrics.csv")
    failed = pd.read_csv(d / "failed_cases.csv")

    check("parameter_sets == 1000", len(params) == EXPECT["params"], f"{len(params)} rows")
    check("sequences (manifest) == 12000", len(manifest) == EXPECT["sequences"], f"{len(manifest)} rows")
    check("error_metrics rows == 12000", len(metrics) == EXPECT["sequences"], f"{len(metrics)} rows")
    check("cases == 12", manifest["experiment_id"].nunique() == EXPECT["cases"],
          f"{manifest['experiment_id'].nunique()} unique experiment_id")
    check("failed sequences == 0", len(failed) == EXPECT["failed"], f"{len(failed)} rows in failed_cases.csv")

    # ---- joins must match ----
    man_seq = set(manifest["sequence_id"])
    met_seq = set(metrics["sequence_id"])
    check("manifest<->error_metric sequence_id 1:1",
          man_seq == met_seq and len(man_seq) == len(manifest) == len(metrics),
          f"manifest_only={len(man_seq-met_seq)} metric_only={len(met_seq-man_seq)}")
    man_samples = set(manifest["sample_id"])
    par_samples = set(params["sample_id"])
    check("every manifest sample_id present in parameter_sets",
          man_samples.issubset(par_samples),
          f"missing={len(man_samples - par_samples)} param_samples={len(par_samples)}")
    check("parameter_sets sample_id unique", params["sample_id"].is_unique,
          f"{params['sample_id'].nunique()} unique / {len(params)}")
    check("sequences == n_samples * n_cases",
          len(manifest) == EXPECT["params"] * EXPECT["cases"],
          f"{EXPECT['params']}*{EXPECT['cases']}={EXPECT['params']*EXPECT['cases']}")

    # ---- n_extrapolated_points (REQ 7): preserve + report, never clip ----
    nx = manifest["n_extrapolated_points"]
    report["n_extrapolated_points"] = {
        "present": True,
        "total": int(nx.sum()), "min": int(nx.min()), "max": int(nx.max()),
        "mean": float(nx.mean()),
        "n_sequences_with_extrapolation": int((nx > 0).sum()),
        "n_sequences_zero": int((nx == 0).sum()),
        "by_case_total": {k: int(v) for k, v in
                          manifest.groupby("experiment_id")["n_extrapolated_points"].sum().items()},
        "note": "Boundary values are held outside the simulated interval (per dataset_summary "
                "alignment). Extrapolated points are preserved end-to-end; never clipped.",
    }
    check("n_extrapolated_points column present and finite",
          nx.notna().all() and np.isfinite(nx).all(),
          f"total={int(nx.sum())} max={int(nx.max())} seqs_with_extrap={int((nx>0).sum())}")

    # ---- streaming pass over time_series.csv ----
    n_rows = 0
    nonfinite = 0
    nan_cells = 0
    ts_seq_ids: set[str] = set()
    cols_ok = True
    for chunk in pd.read_csv(d / "time_series.csv", chunksize=args.chunksize, dtype=TS_DTYPES):
        if list(chunk.columns) != list(TS_DTYPES.keys()):
            cols_ok = False
        n_rows += len(chunk)
        num = chunk[["time_s", "voltage_v", "temperature_c"]].to_numpy()
        nan_cells += int(np.isnan(num).sum())
        nonfinite += int((~np.isfinite(num)).sum())
        ts_seq_ids.update(chunk["sequence_id"].dropna().unique().tolist())

    check("time-series data rows == 4,001,000", n_rows == EXPECT["ts_rows"], f"{n_rows:,} rows")
    check("time_series.csv schema matches", cols_ok, str(list(TS_DTYPES.keys())))
    check("no NaN cells in time_series", nan_cells == 0, f"{nan_cells} NaN")
    check("no non-finite cells in time_series", nonfinite == 0, f"{nonfinite} non-finite")
    check("time_series sequence_ids == manifest sequence_ids",
          ts_seq_ids == man_seq, f"ts={len(ts_seq_ids)} manifest={len(man_seq)} "
          f"sym_diff={len(ts_seq_ids ^ man_seq)}")

    report["counts"] = {"ts_rows": n_rows, "params": len(params), "sequences": len(manifest),
                        "cases": int(manifest["experiment_id"].nunique()), "failed": len(failed),
                        "ts_unique_sequence_ids": len(ts_seq_ids)}
    report["overall_pass"] = ok

    out = d / "_batch4_validation_report.json"
    # write report to OUTPUTS, never into the immutable raw dir
    print("\n==== n_extrapolated_points ====")
    print(json.dumps(report["n_extrapolated_points"], indent=2))
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    print(json.dumps(report, indent=2, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
