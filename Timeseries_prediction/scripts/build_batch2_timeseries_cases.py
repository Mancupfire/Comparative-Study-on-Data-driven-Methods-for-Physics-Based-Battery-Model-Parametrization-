"""Adapter: convert Batch 2 RAW long-format ``time_series.csv`` into the per-case
``cases/<case_id>/outputs.npz`` matrix layout that the established Batch 1
pipeline (``src/data.py``) consumes.

Why this exists
---------------
Batch 1 shipped pre-aligned per-case matrices (``outputs.npz`` with
``sample_ids``/``time_s``/``voltage_v``/``temperature_c``).  Batch 2 instead
ships a single long ``time_series.csv`` (one row per time point).  This script
reconstructs the *exact same matrix contract* per experiment case so the whole
downstream pipeline (split, scalers, models, metrics, evaluation) runs
**unchanged** and remains scientifically comparable to Batch 1.

Key facts validated up-front (and re-asserted here):
* Within every experiment case ``n_time_points`` is constant -> a clean
  ``[N_ok, t_last]`` matrix exists with no padding.
* 0 failed sequences; 1000 samples x 12 cases = 12000 sequences.

The source long file is the *verified full* extract
``data/Data_Batch_2_cleaned/time_series.csv`` (the original
``data/Data_Batch_2/time_series.csv`` is truncated and is left untouched).

Output (immutable processed dataset, NOT the raw Batch 2 folder)::

    data/Data_Batch_2_cleaned/
    ├── time_series.csv                 (already present: verified full extract)
    ├── parameter_sets.csv              (copied from raw)
    ├── sequence_manifest.csv           (copied from raw)
    ├── failed_cases.csv                (copied from raw)
    ├── dataset_summary.json            (copied from raw)
    ├── cases/<case_id>/outputs.npz     (BUILT here)
    ├── source_checksums.json
    ├── cleaning_manifest.json
    ├── cleaning_report.md
    └── build_stats.json                (per-case stats for the audit)

Usage
-----
python scripts/build_batch2_timeseries_cases.py \
    --raw-dir data/Data_Batch_2 \
    --full-timeseries data/Data_Batch_2_cleaned/time_series.csv \
    --out-dir data/Data_Batch_2_cleaned
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TS_COLS = ["sequence_id", "time_index", "time_s", "voltage_v", "temperature_c"]


def sha256(path: Path, buf: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Batch 2 per-case outputs.npz matrices.")
    p.add_argument("--raw-dir", default="data/Data_Batch_2",
                   help="Immutable raw Batch 2 directory (metadata source).")
    p.add_argument("--full-timeseries",
                   default="data/Data_Batch_2_cleaned/time_series.csv",
                   help="Verified full long-format time_series.csv.")
    p.add_argument("--out-dir", default="data/Data_Batch_2_cleaned")
    p.add_argument("--chunksize", type=int, default=2_000_000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    raw = Path(args.raw_dir)
    ts_full = Path(args.full_timeseries)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)

    t0 = time.time()
    print(f"[adapter] raw_dir={raw}  full_timeseries={ts_full}  out_dir={out}")

    # ----- manifest -> per-case skeletons ---------------------------------- #
    manifest = pd.read_csv(raw / "sequence_manifest.csv")
    print(f"[adapter] manifest rows={len(manifest)} cases={manifest['experiment_id'].nunique()}")

    cases = {}
    for exp, g in manifest.groupby("experiment_id"):
        n_tp = g["n_time_points"].unique()
        if len(n_tp) != 1:
            raise ValueError(
                f"[{exp}] n_time_points is NOT constant ({sorted(n_tp)}); "
                f"a fixed-length matrix cannot be built without resampling."
            )
        t_last = int(n_tp[0])
        sids = sorted(g["sample_id"].unique())  # ascending, deterministic
        row_of = {sid: i for i, sid in enumerate(sids)}
        cases[exp] = {
            "t_last": t_last,
            "sids": sids,
            "row_of": row_of,
            "ref_sid": sids[0],
            "V": np.full((len(sids), t_last), np.nan, dtype=np.float64),
            "T": np.full((len(sids), t_last), np.nan, dtype=np.float64),
            "time": np.full(t_last, np.nan, dtype=np.float64),
        }
    print(f"[adapter] {len(cases)} cases; t_last "
          f"{min(c['t_last'] for c in cases.values())}.."
          f"{max(c['t_last'] for c in cases.values())}")

    # ----- single streaming pass over the long file ------------------------ #
    n_rows = 0
    dtypes = {"sequence_id": "string", "time_index": "int32",
              "time_s": "float64", "voltage_v": "float64", "temperature_c": "float64"}
    reader = pd.read_csv(ts_full, chunksize=args.chunksize, dtype=dtypes)
    for ci, chunk in enumerate(reader):
        if list(chunk.columns) != TS_COLS:
            raise ValueError(f"Unexpected time_series columns: {list(chunk.columns)}")
        seq = chunk["sequence_id"].str.rsplit("__", n=1, expand=True)
        chunk = chunk.assign(sample_id=seq[0], experiment_id=seq[1])
        for exp, g in chunk.groupby("experiment_id", observed=True):
            c = cases.get(exp)
            if c is None:
                raise KeyError(f"time_series has unknown experiment_id '{exp}'")
            rows = g["sample_id"].map(c["row_of"]).to_numpy()
            if np.any(pd.isna(rows)):
                bad = g.loc[pd.isna(rows), "sample_id"].unique()[:5]
                raise KeyError(f"[{exp}] sample_id(s) not in manifest: {bad}")
            rows = rows.astype(np.int64)
            ti = g["time_index"].to_numpy()
            c["V"][rows, ti] = g["voltage_v"].to_numpy()
            c["T"][rows, ti] = g["temperature_c"].to_numpy()
            # time grid taken from the reference sample only (deterministic)
            ref = g["sample_id"].to_numpy() == c["ref_sid"]
            if ref.any():
                c["time"][ti[ref]] = g["time_s"].to_numpy()[ref]
        n_rows += len(chunk)
        print(f"[adapter] chunk {ci}: cumulative rows={n_rows:,}", flush=True)

    # ----- validate + save each case --------------------------------------- #
    build_stats = {}
    for exp, c in cases.items():
        V, T, tvec = c["V"], c["T"], c["time"]
        if np.isnan(V).any() or np.isnan(T).any():
            raise ValueError(f"[{exp}] missing cells after fill: "
                             f"V_nan={int(np.isnan(V).sum())} T_nan={int(np.isnan(T).sum())}")
        if np.isnan(tvec).any():
            raise ValueError(f"[{exp}] time grid has NaN (reference sample incomplete)")
        if not np.all(np.diff(tvec) > 0):
            raise ValueError(f"[{exp}] time grid not strictly increasing")
        if not (np.isfinite(V).all() and np.isfinite(T).all()):
            raise ValueError(f"[{exp}] non-finite voltage/temperature present")

        case_dir = out / "cases" / exp
        case_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = np.array(c["sids"], dtype="U7")
        np.savez_compressed(
            case_dir / "outputs.npz",
            sample_ids=sample_ids,
            time_s=tvec.astype(np.float64),
            voltage_v=V.astype(np.float64),
            temperature_c=T.astype(np.float64),
        )
        build_stats[exp] = {
            "n_samples": int(V.shape[0]),
            "t_last": int(V.shape[1]),
            "time_start_s": float(tvec[0]),
            "time_end_s": float(tvec[-1]),
            "voltage_min": float(V.min()), "voltage_max": float(V.max()),
            "voltage_mean": float(V.mean()), "voltage_std": float(V.std()),
            "temperature_min": float(T.min()), "temperature_max": float(T.max()),
            "temperature_mean": float(T.mean()), "temperature_std": float(T.std()),
        }
        print(f"[adapter] saved {exp}: V/T shape {V.shape}  "
              f"V[{V.min():.3f},{V.max():.3f}] T[{T.min():.3f},{T.max():.3f}]")

    # ----- copy metadata files (self-contained processed dataset) ---------- #
    copied = []
    for name in ("parameter_sets.csv", "sequence_manifest.csv",
                 "failed_cases.csv", "dataset_summary.json", "README.md"):
        src = raw / name
        if src.is_file():
            shutil.copy2(src, out / name)
            copied.append(name)

    # ----- provenance / checksums ------------------------------------------ #
    source_checksums = {}
    for name in ("parameter_sets.csv", "sequence_manifest.csv", "error_metrics.csv",
                 "failed_cases.csv", "dataset_summary.json", "README.md",
                 "time_series.csv"):
        src = raw / name
        if src.is_file():
            source_checksums[f"raw/{name}"] = {"sha256": sha256(src),
                                               "bytes": src.stat().st_size}
    source_checksums["cleaned/time_series.csv"] = {
        "sha256": sha256(ts_full), "bytes": ts_full.stat().st_size,
        "n_rows_incl_header": int(n_rows + 1),
    }
    with (out / "source_checksums.json").open("w") as f:
        json.dump(source_checksums, f, indent=2)

    with (out / "build_stats.json").open("w") as f:
        json.dump(build_stats, f, indent=2)

    now = datetime.now(timezone.utc).isoformat()
    cleaning_manifest = {
        "source_dataset": "Data_Batch_2",
        "source_files": sorted(source_checksums.keys()),
        "source_checksums": source_checksums,
        "cleaned_files": ["time_series.csv (verified full extract)",
                          "cases/<case_id>/outputs.npz"] + copied,
        "operations": [
            "Extracted full time_series.csv from generate_training_data.zip "
            "(raw on-disk copy was truncated to 23.4M of 37.87M rows).",
            "Reconstructed per-case [N_ok, t_last] voltage/temperature matrices "
            "from the long format, grouped by experiment_id, sorted by time_index, "
            "with sample_ids sorted ascending.",
            "Saved per-case outputs.npz in the Batch 1 matrix contract.",
            "Copied parameter/manifest/summary metadata; no values altered.",
        ],
        "rows_before": int(n_rows),
        "rows_after": int(n_rows),
        "cases_before": len(cases),
        "cases_after": len(cases),
        "reason_for_each_operation": [
            "Original time_series.csv truncated -> use authoritative zip copy.",
            "Downstream Batch 1 pipeline requires per-case matrix layout.",
            "No outlier/imputation/unit/value changes were made.",
        ],
        "created_at": now,
    }
    with (out / "cleaning_manifest.json").open("w") as f:
        json.dump(cleaning_manifest, f, indent=2)

    report = [
        "# Data_Batch_2 cleaning / build report",
        "",
        f"Created: {now}",
        "",
        "## What was done",
        "1. The on-disk `data/Data_Batch_2/time_series.csv` was found **truncated** "
        f"(23,407,726 rows). The authoritative full copy was extracted from "
        "`data/generate_training_data.zip` to "
        "`data/Data_Batch_2_cleaned/time_series.csv` "
        f"({n_rows:,} data rows, verified complete).",
        "2. Per-case `cases/<case_id>/outputs.npz` matrices were reconstructed from "
        "the long format using the same matrix contract as Batch 1.",
        "3. Metadata files were copied unchanged.",
        "",
        "## Guarantees",
        "* No voltage/temperature/parameter values were modified, clipped, imputed, "
        "smoothed, resampled or unit-converted.",
        "* The raw `data/Data_Batch_2/` directory was left untouched (immutable).",
        "* Per case, `n_time_points` is constant, so matrices contain **no padding** "
        "and **no missing cells**.",
        "",
        "## Per-case summary",
        "",
        "| case_id | n_samples | t_last | V[min,max] | T[min,max] |",
        "|---|---|---|---|---|",
    ]
    for exp in sorted(build_stats):
        s = build_stats[exp]
        report.append(
            f"| {exp} | {s['n_samples']} | {s['t_last']} | "
            f"[{s['voltage_min']:.3f}, {s['voltage_max']:.3f}] | "
            f"[{s['temperature_min']:.3f}, {s['temperature_max']:.3f}] |"
        )
    (out / "cleaning_report.md").write_text("\n".join(report) + "\n")

    print(f"[adapter] DONE in {time.time() - t0:.1f}s. "
          f"Built {len(cases)} cases under {out / 'cases'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
