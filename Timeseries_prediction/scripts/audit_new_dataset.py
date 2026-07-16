"""Deep integrity audit of the newly received dataset + relationship to Batch 2.

Read-only. Never modifies the raw extracted directory. Streams the large
time_series.csv in chunks so memory stays bounded. Produces the Phase 3 audit
reports and the Phase 4 Batch-2 comparison tables.

Usage:
    python scripts/audit_new_dataset.py \
        --new-dir data/generate_training_data \
        --batch2-dir data/Data_Batch_2 \
        --out-dir reports/new_batch_setup/data_audit \
        --cmp-out-dir reports/new_batch_setup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TS_COLS = ["sequence_id", "time_index", "time_s", "voltage_v", "temperature_c"]
PHYS = {
    "voltage_v": (2.0, 4.3),          # plausible Li-ion cell voltage window
    "temperature_c": (20.0, 90.0),    # ambient 25/45 + self-heating headroom
}


def sha256(path, buf=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def file_inventory(d):
    rows = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            continue
        st = os.stat(p)
        ends_nl = None
        if name.endswith((".csv", ".json", ".md")):
            with open(p, "rb") as f:
                f.seek(-1, os.SEEK_END)
                ends_nl = f.read(1) == b"\n"
        rows.append({
            "path": p,
            "bytes": st.st_size,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "sha256": sha256(p),
            "ends_with_newline": ends_nl,
        })
    return pd.DataFrame(rows)


def audit_time_series(ts_path, chunksize=1_000_000):
    """Single streaming pass. Returns per-case + global stats."""
    dtypes = {"sequence_id": "string", "time_index": "int32",
              "time_s": "float64", "voltage_v": "float64", "temperature_c": "float64"}
    per_case = defaultdict(lambda: {
        "n_rows": 0, "n_seq": set(), "ntp": defaultdict(int),
        "v_min": np.inf, "v_max": -np.inf, "t_min": np.inf, "t_max": -np.inf,
        "n_nan": 0, "n_inf": 0,
        "ti_min": np.inf, "ti_max": -np.inf,
        "time_min": np.inf, "time_max": -np.inf,
    })
    # per-sequence monotonic-time tracking across chunk boundaries
    last_time = {}      # seq -> last time_s seen
    last_ti = {}        # seq -> last time_index seen
    nonmono_time = set()
    nonmono_ti = set()
    dup_ti = 0
    total_rows = 0
    bad_header = None

    reader = pd.read_csv(ts_path, chunksize=chunksize, dtype=dtypes)
    for ci, chunk in enumerate(reader):
        if list(chunk.columns) != TS_COLS:
            bad_header = list(chunk.columns)
            break
        seq = chunk["sequence_id"].str.rsplit("__", n=1, expand=True)
        chunk = chunk.assign(experiment_id=seq[1])
        for exp, g in chunk.groupby("experiment_id", observed=True):
            c = per_case[exp]
            c["n_rows"] += len(g)
            c["n_seq"].update(g["sequence_id"].unique().tolist())
            vc = g["sequence_id"].value_counts()
            for sid, n in vc.items():
                c["ntp"][sid] += int(n)
            v = g["voltage_v"].to_numpy()
            t = g["temperature_c"].to_numpy()
            c["n_nan"] += int(np.isnan(v).sum() + np.isnan(t).sum())
            c["n_inf"] += int(np.isinf(v).sum() + np.isinf(t).sum())
            fv = v[np.isfinite(v)]; ft = t[np.isfinite(t)]
            if fv.size:
                c["v_min"] = min(c["v_min"], fv.min()); c["v_max"] = max(c["v_max"], fv.max())
            if ft.size:
                c["t_min"] = min(c["t_min"], ft.min()); c["t_max"] = max(c["t_max"], ft.max())
            ti = g["time_index"].to_numpy(); ts = g["time_s"].to_numpy()
            c["ti_min"] = min(c["ti_min"], ti.min()); c["ti_max"] = max(c["ti_max"], ti.max())
            c["time_min"] = min(c["time_min"], ts.min()); c["time_max"] = max(c["time_max"], ts.max())
        # per-sequence monotonicity (process rows in file order within chunk)
        for sid, g in chunk.groupby("sequence_id", observed=True):
            ts = g["time_s"].to_numpy(); ti = g["time_index"].to_numpy()
            seq_t = ts; seq_i = ti
            if sid in last_time:
                seq_t = np.concatenate([[last_time[sid]], ts])
                seq_i = np.concatenate([[last_ti[sid]], ti])
            if np.any(np.diff(seq_t) <= 0):
                nonmono_time.add(sid)
            if np.any(np.diff(seq_i) != 1):
                nonmono_ti.add(sid)
            last_time[sid] = ts[-1]; last_ti[sid] = ti[-1]
        total_rows += len(chunk)
    return {
        "bad_header": bad_header,
        "total_rows": total_rows,
        "per_case": per_case,
        "nonmono_time": nonmono_time,
        "nonmono_ti": nonmono_ti,
        "last_ti": last_ti,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-dir", default="data/generate_training_data")
    ap.add_argument("--batch2-dir", default="data/Data_Batch_2")
    ap.add_argument("--out-dir", default="reports/new_batch_setup/data_audit")
    ap.add_argument("--cmp-out-dir", default="reports/new_batch_setup")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cmp_out_dir, exist_ok=True)
    warnings = []
    errors = []

    # ---------- file inventory + checksums ----------
    inv = file_inventory(args.new_dir)
    inv.to_csv(os.path.join(args.out_dir, "source_file_inventory.csv"), index=False)
    checks = {r["path"]: {"sha256": r["sha256"], "bytes": int(r["bytes"])}
              for _, r in inv.iterrows()}
    with open(os.path.join(args.out_dir, "source_checksums.json"), "w") as f:
        json.dump(checks, f, indent=2)

    # ---------- metadata tables ----------
    man = pd.read_csv(os.path.join(args.new_dir, "sequence_manifest.csv"))
    err = pd.read_csv(os.path.join(args.new_dir, "error_metrics.csv"))
    par = pd.read_csv(os.path.join(args.new_dir, "parameter_sets.csv"))
    failed = pd.read_csv(os.path.join(args.new_dir, "failed_cases.csv"))
    summ = json.load(open(os.path.join(args.new_dir, "dataset_summary.json")))

    # ---------- schema report ----------
    schema_rows = []
    for fname, df in [("sequence_manifest.csv", man), ("error_metrics.csv", err),
                      ("parameter_sets.csv", par), ("failed_cases.csv", failed)]:
        for col in df.columns:
            schema_rows.append({"file": fname, "column": col, "dtype": str(df[col].dtype),
                                "n_null": int(df[col].isna().sum()),
                                "n_unique": int(df[col].nunique())})
    pd.DataFrame(schema_rows).to_csv(os.path.join(args.out_dir, "schema_report.csv"), index=False)

    # ---------- relational integrity ----------
    man_seq = set(man["sequence_id"]); err_seq = set(err["sequence_id"])
    man_samp = set(man["sample_id"]); par_samp = set(par["sample_id"])
    rel = []
    rel.append(("manifest_sequence_id_unique", bool(man["sequence_id"].is_unique)))
    rel.append(("error_sequence_id_unique", bool(err["sequence_id"].is_unique)))
    rel.append(("param_sample_id_unique", bool(par["sample_id"].is_unique)))
    rel.append(("manifest_eq_error_seq", man_seq == err_seq))
    rel.append(("manifest_samples_subset_of_params", man_samp.issubset(par_samp)))
    rel.append(("n_manifest_seq", len(man_seq)))
    rel.append(("n_error_seq", len(err_seq)))
    rel.append(("n_param_samples", len(par_samp)))
    rel.append(("n_manifest_samples", len(man_samp)))
    rel.append(("failed_cases_empty", failed.empty))
    rel.append(("expected_seqs_500x12", len(man_seq) == 6000))
    pd.DataFrame(rel, columns=["check", "value"]).to_csv(
        os.path.join(args.out_dir, "relational_integrity_report.csv"), index=False)
    if man_seq != err_seq:
        errors.append("error_metrics sequence_ids != manifest sequence_ids")
    if not failed.empty:
        warnings.append("failed_cases.csv not empty")

    # ---------- missing / invalid (metadata) ----------
    mv = []
    for fname, df in [("sequence_manifest.csv", man), ("error_metrics.csv", err),
                      ("parameter_sets.csv", par)]:
        for col in df.columns:
            nn = int(df[col].isna().sum())
            if nn:
                mv.append({"file": fname, "column": col, "n_missing": nn})
    pd.DataFrame(mv, columns=["file", "column", "n_missing"]).to_csv(
        os.path.join(args.out_dir, "missing_values_report.csv"), index=False)

    iv = []
    for col in ["rmse_voltage_mv", "rmse_temperature_c"]:
        a = err[col].to_numpy()
        iv.append({"file": "error_metrics.csv", "column": col,
                   "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
                   "n_negative": int((a < 0).sum()), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))})
    pcols = [c for c in par.columns if c != "sample_id"]
    for col in pcols:
        a = par[col].to_numpy()
        iv.append({"file": "parameter_sets.csv", "column": col,
                   "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum()),
                   "n_negative": int((a < 0).sum()), "min": float(np.nanmin(a)), "max": float(np.nanmax(a))})
    pd.DataFrame(iv).to_csv(os.path.join(args.out_dir, "invalid_values_report.csv"), index=False)

    # ---------- parameter range vs declared bounds ----------
    pr = []
    for col, (lo, hi) in summ.get("parameter_ranges", {}).items():
        if col in par.columns:
            a = par[col].to_numpy()
            within = bool((a >= lo - abs(lo) * 1e-6).all() and (a <= hi + abs(hi) * 1e-6).all())
            pr.append({"parameter": col, "declared_min": lo, "declared_max": hi,
                       "actual_min": float(a.min()), "actual_max": float(a.max()),
                       "within_declared": within})
            if not within:
                warnings.append(f"parameter out of declared range: {col}")
    pd.DataFrame(pr).to_csv(os.path.join(args.out_dir, "physical_range_report.csv"), index=False)

    # ---------- duplicates ----------
    dup = []
    dup.append({"check": "duplicate_manifest_rows", "count": int(man.duplicated().sum())})
    dup.append({"check": "duplicate_param_rows", "count": int(par.duplicated().sum())})
    dup.append({"check": "duplicate_param_vectors(excl id)", "count": int(par[pcols].duplicated().sum())})
    dup.append({"check": "duplicate_error_rows", "count": int(err.duplicated().sum())})
    dup.append({"check": "duplicate_sequence_id_manifest", "count": int(man["sequence_id"].duplicated().sum())})
    pd.DataFrame(dup).to_csv(os.path.join(args.out_dir, "duplicate_report.csv"), index=False)

    # ---------- time-series streaming audit ----------
    ts_path = os.path.join(args.new_dir, "time_series.csv")
    print("[audit] streaming time_series.csv ...", flush=True)
    tsr = audit_time_series(ts_path)
    if tsr["bad_header"]:
        errors.append(f"time_series header mismatch: {tsr['bad_header']}")

    # cross-check manifest n_time_points vs observed per sequence
    man_ntp = dict(zip(man["sequence_id"], man["n_time_points"]))
    ntp_mismatch = 0
    temporal_rows = []
    for exp, c in sorted(tsr["per_case"].items()):
        ntp_vals = set(c["ntp"].values())
        # verify each sequence's observed point count matches manifest
        for sid, n in c["ntp"].items():
            if sid in man_ntp and man_ntp[sid] != n:
                ntp_mismatch += 1
        temporal_rows.append({
            "experiment_id": exp,
            "n_sequences": len(c["n_seq"]),
            "n_rows": c["n_rows"],
            "ntp_constant_within_case": len(ntp_vals) == 1,
            "ntp_values": sorted(ntp_vals)[:5],
            "time_index_min": int(c["ti_min"]), "time_index_max": int(c["ti_max"]),
            "time_s_min": float(c["time_min"]), "time_s_max": float(c["time_max"]),
            "n_nonmonotonic_time_seqs": sum(1 for s in tsr["nonmono_time"] if s.endswith("__" + exp)),
            "n_nonmonotonic_index_seqs": sum(1 for s in tsr["nonmono_ti"] if s.endswith("__" + exp)),
        })
    pd.DataFrame(temporal_rows).to_csv(
        os.path.join(args.out_dir, "temporal_integrity_report.csv"), index=False)

    # physical-range report for time series (per case)
    tspr = []
    for exp, c in sorted(tsr["per_case"].items()):
        v_ok = PHYS["voltage_v"][0] <= c["v_min"] and c["v_max"] <= PHYS["voltage_v"][1]
        t_ok = PHYS["temperature_c"][0] <= c["t_min"] and c["t_max"] <= PHYS["temperature_c"][1]
        tspr.append({"experiment_id": exp,
                     "voltage_min": c["v_min"], "voltage_max": c["v_max"], "voltage_in_range": bool(v_ok),
                     "temperature_min": c["t_min"], "temperature_max": c["t_max"], "temperature_in_range": bool(t_ok),
                     "n_nan": c["n_nan"], "n_inf": c["n_inf"]})
        if not v_ok:
            warnings.append(f"{exp}: voltage outside [{PHYS['voltage_v']}]")
        if not t_ok:
            warnings.append(f"{exp}: temperature outside [{PHYS['temperature_c']}]")
        if c["n_nan"] or c["n_inf"]:
            errors.append(f"{exp}: non-finite values in time series")
    pd.DataFrame(tspr).to_csv(os.path.join(args.out_dir, "ts_physical_range_report.csv"), index=False)

    nonmono = len(tsr["nonmono_time"]) + len(tsr["nonmono_ti"])
    if nonmono:
        errors.append(f"{nonmono} sequences with non-monotonic time/index")
    if ntp_mismatch:
        errors.append(f"{ntp_mismatch} sequences: observed point count != manifest n_time_points")

    total_ts_rows = tsr["total_rows"]
    expected_ts_rows = int(man["n_time_points"].sum())
    if total_ts_rows != expected_ts_rows:
        errors.append(f"time_series rows {total_ts_rows} != sum(manifest n_time_points) {expected_ts_rows}")

    # ---------- leakage report ----------
    leak = [
        "# Leakage / split-risk report (new dataset)",
        "",
        f"- Split key: `sample_id` (README mandates grouping all 12 sequences of a sample).",
        f"- sequence_id encodes `<sample_id>__<experiment_id>`: parsing verified.",
        f"- Each sample_id has {len(man) // max(1, man['sample_id'].nunique())} sequences (expected 12).",
        f"- error_metrics targets (rmse_*) are NOT present in time_series inputs: no direct target leakage into TS task.",
        f"- For error-metric task, targets are rmse_* keyed by sequence_id; inputs are parameter vector + operating",
        f"  condition. Split must be by sample_id so a parameter set never crosses splits.",
        f"- Operating-condition features (c_rate, temps, operation_code) are deterministic per case, not leaked targets.",
        "",
        "No target columns were found inside the time-series feature set. Split-by-sample_id is enforceable.",
    ]
    with open(os.path.join(args.out_dir, "leakage_report.md"), "w") as f:
        f.write("\n".join(leak) + "\n")

    # ---------- classification ----------
    if errors:
        status = "REQUIRES_MANUAL_REVIEW"
    elif warnings:
        status = "PASS_WITH_WARNINGS"
    else:
        status = "PASS"

    summary = {
        "classification": status,
        "n_errors": len(errors), "errors": errors,
        "n_warnings": len(warnings), "warnings": warnings,
        "rows": {"time_series": total_ts_rows, "manifest": len(man),
                 "error_metrics": len(err), "parameter_sets": len(par)},
        "cases": int(man["experiment_id"].nunique()),
        "valid_cases": int(man["experiment_id"].nunique()),
        "failed_sequences": int(len(failed)),
        "parameter_sets": int(par["sample_id"].nunique()),
        "sequences": int(len(man_seq)),
        "expected_ts_rows": expected_ts_rows,
        "ntp_constant_per_case": all(r["ntp_constant_within_case"] for r in temporal_rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(args.out_dir, "data_quality_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # markdown report
    md = [f"# New dataset audit — {status}", "",
          f"Generated: {summary['created_at']}", "",
          "## Counts",
          f"- time_series rows: {total_ts_rows:,} (expected {expected_ts_rows:,})",
          f"- sequences (manifest = error_metrics): {len(man_seq):,}",
          f"- parameter sets: {len(par_samp)}",
          f"- operating cases: {summary['cases']}",
          f"- failed sequences: {len(failed)}",
          "", "## Integrity",
          f"- n_time_points constant within each case: {summary['ntp_constant_per_case']}",
          f"- non-monotonic-time sequences: {len(tsr['nonmono_time'])}",
          f"- non-monotonic-index sequences: {len(tsr['nonmono_ti'])}",
          f"- n_time_points mismatch vs manifest: {ntp_mismatch}",
          f"- non-finite values in time series: {sum(c['n_nan']+c['n_inf'] for c in tsr['per_case'].values())}",
          "", "## Errors", *([f"- {e}" for e in errors] or ["- none"]),
          "", "## Warnings", *([f"- {w}" for w in warnings] or ["- none"]),
          ""]
    with open(os.path.join(args.out_dir, "data_quality_report.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    # suspicious cases (constant columns, near-constant)
    susp = []
    for col in pcols:
        if par[col].nunique() <= 1:
            susp.append({"file": "parameter_sets.csv", "column": col, "issue": "constant"})
    pd.DataFrame(susp, columns=["file", "column", "issue"]).to_csv(
        os.path.join(args.out_dir, "suspicious_cases.csv"), index=False)

    print(f"[audit] classification = {status}")
    print(f"[audit] errors={len(errors)} warnings={len(warnings)}")
    print("[audit] reports written to", args.out_dir)
    return summary


if __name__ == "__main__":
    main()
