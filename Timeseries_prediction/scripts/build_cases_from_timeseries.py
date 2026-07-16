"""Generic adapter: convert a long-format ``time_series.csv`` into the per-case
``cases/<case_id>/outputs.npz`` matrix layout consumed by ``src/data.py``.

Faithful, parameterised reimplementation of the Batch 2 adapter
(``scripts/build_batch2_timeseries_cases.py``) with correct, dataset-agnostic
provenance text. Memory-safe: a single chunked streaming pass over the long CSV.

Contract per case (same as Batch 1/2):
  outputs.npz = {sample_ids, time_s[t_last], voltage_v[N,t_last], temperature_c[N,t_last]}
Requires ``n_time_points`` constant within each experiment_id (no padding).

Usage:
    python scripts/build_cases_from_timeseries.py \
        --raw-dir data/generate_training_data \
        --full-timeseries data/generate_training_data/time_series.csv \
        --out-dir data/Data_Batch_3_cleaned \
        --dataset-name Data_Batch_3
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
    p = argparse.ArgumentParser(description="Build per-case outputs.npz from a long time_series.csv.")
    p.add_argument("--raw-dir", required=True, help="Dir with sequence_manifest.csv + metadata (immutable).")
    p.add_argument("--full-timeseries", required=True, help="Verified full long-format time_series.csv.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset-name", default="dataset")
    p.add_argument("--chunksize", type=int, default=1_000_000)
    return p


def main() -> int:
    args = build_parser().parse_args()
    raw = Path(args.raw_dir)
    ts_full = Path(args.full_timeseries)
    out = Path(args.out_dir)
    if out.resolve() == raw.resolve():
        raise SystemExit("out-dir must differ from raw-dir (never modify raw data).")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases").mkdir(exist_ok=True)

    t0 = time.time()
    print(f"[adapter] raw={raw} full_timeseries={ts_full} out={out}")

    manifest = pd.read_csv(raw / "sequence_manifest.csv")
    cases = {}
    for exp, g in manifest.groupby("experiment_id"):
        n_tp = g["n_time_points"].unique()
        if len(n_tp) != 1:
            raise ValueError(f"[{exp}] n_time_points not constant ({sorted(n_tp)}); cannot build fixed matrix.")
        t_last = int(n_tp[0])
        sids = sorted(g["sample_id"].unique())
        cases[exp] = {
            "t_last": t_last, "sids": sids, "row_of": {s: i for i, s in enumerate(sids)},
            "ref_sid": sids[0],
            "V": np.full((len(sids), t_last), np.nan, dtype=np.float64),
            "T": np.full((len(sids), t_last), np.nan, dtype=np.float64),
            "time": np.full(t_last, np.nan, dtype=np.float64),
        }
    print(f"[adapter] {len(cases)} cases; t_last "
          f"{min(c['t_last'] for c in cases.values())}..{max(c['t_last'] for c in cases.values())}")

    dtypes = {"sequence_id": "string", "time_index": "int32",
              "time_s": "float64", "voltage_v": "float64", "temperature_c": "float64"}
    n_rows = 0
    for ci, chunk in enumerate(pd.read_csv(ts_full, chunksize=args.chunksize, dtype=dtypes)):
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
            ref = g["sample_id"].to_numpy() == c["ref_sid"]
            if ref.any():
                c["time"][ti[ref]] = g["time_s"].to_numpy()[ref]
        n_rows += len(chunk)
        print(f"[adapter] chunk {ci}: cumulative rows={n_rows:,}", flush=True)

    build_stats = {}
    for exp, c in cases.items():
        V, T, tvec = c["V"], c["T"], c["time"]
        if np.isnan(V).any() or np.isnan(T).any():
            raise ValueError(f"[{exp}] missing cells after fill")
        if np.isnan(tvec).any():
            raise ValueError(f"[{exp}] time grid incomplete")
        if not np.all(np.diff(tvec) > 0):
            raise ValueError(f"[{exp}] time grid not strictly increasing")
        if not (np.isfinite(V).all() and np.isfinite(T).all()):
            raise ValueError(f"[{exp}] non-finite voltage/temperature present")
        case_dir = out / "cases" / exp
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            case_dir / "outputs.npz",
            sample_ids=np.array(c["sids"], dtype="U7"),
            time_s=tvec.astype(np.float64),
            voltage_v=V.astype(np.float64),
            temperature_c=T.astype(np.float64),
        )
        build_stats[exp] = {
            "n_samples": int(V.shape[0]), "t_last": int(V.shape[1]),
            "time_start_s": float(tvec[0]), "time_end_s": float(tvec[-1]),
            "voltage_min": float(V.min()), "voltage_max": float(V.max()),
            "temperature_min": float(T.min()), "temperature_max": float(T.max()),
        }
        print(f"[adapter] saved {exp}: V/T {V.shape} V[{V.min():.3f},{V.max():.3f}] T[{T.min():.3f},{T.max():.3f}]")

    copied = []
    for name in ("parameter_sets.csv", "sequence_manifest.csv", "failed_cases.csv",
                 "dataset_summary.json", "README.md", "error_metrics.csv"):
        src = raw / name
        if src.is_file():
            shutil.copy2(src, out / name)
            copied.append(name)

    source_checksums = {}
    for name in ("parameter_sets.csv", "sequence_manifest.csv", "error_metrics.csv",
                 "failed_cases.csv", "dataset_summary.json", "README.md", "time_series.csv"):
        src = raw / name
        if src.is_file():
            source_checksums[f"raw/{name}"] = {"sha256": sha256(src), "bytes": src.stat().st_size}
    source_checksums["full/time_series.csv"] = {
        "sha256": sha256(ts_full), "bytes": ts_full.stat().st_size,
        "n_rows_incl_header": int(n_rows + 1),
    }
    (out / "source_checksums.json").write_text(json.dumps(source_checksums, indent=2))
    (out / "build_stats.json").write_text(json.dumps(build_stats, indent=2))

    now = datetime.now(timezone.utc).isoformat()
    cleaning_manifest = {
        "dataset_name": args.dataset_name,
        "source_dataset": str(raw),
        "source_files": sorted(source_checksums.keys()),
        "source_checksums": source_checksums,
        "cleaned_files": ["cases/<case_id>/outputs.npz"] + copied,
        "operations": [
            "Reconstructed per-case [N, t_last] voltage/temperature matrices from the "
            "long-format time_series.csv, grouped by experiment_id, sorted by time_index, "
            "sample_ids sorted ascending.",
            "Saved per-case outputs.npz (Batch 1/2 matrix contract).",
            "Copied parameter/manifest/summary/error-metric metadata unchanged.",
        ],
        "rows_before": int(n_rows), "rows_after": int(n_rows),
        "cases_before": len(cases), "cases_after": len(cases),
        "value_changes": "none (no clip/impute/smooth/resample/unit-convert)",
        "created_at": now,
    }
    (out / "cleaning_manifest.json").write_text(json.dumps(cleaning_manifest, indent=2))

    rep = [f"# {args.dataset_name} cases build report", "", f"Created: {now}", "",
           f"Built {len(cases)} per-case outputs.npz from {n_rows:,} data rows of {ts_full}.",
           "No values modified. Raw dir left immutable.", "",
           "| case_id | n_samples | t_last | V[min,max] | T[min,max] |", "|---|---|---|---|---|"]
    for exp in sorted(build_stats):
        s = build_stats[exp]
        rep.append(f"| {exp} | {s['n_samples']} | {s['t_last']} | "
                   f"[{s['voltage_min']:.3f},{s['voltage_max']:.3f}] | "
                   f"[{s['temperature_min']:.3f},{s['temperature_max']:.3f}] |")
    (out / "cleaning_report.md").write_text("\n".join(rep) + "\n")

    print(f"[adapter] DONE in {time.time()-t0:.1f}s. {len(cases)} cases -> {out/'cases'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
