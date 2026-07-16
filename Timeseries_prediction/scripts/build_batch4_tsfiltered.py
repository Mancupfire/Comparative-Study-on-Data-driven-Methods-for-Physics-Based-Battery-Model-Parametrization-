"""Build the duration-ratio-filtered time-series dataset manifests.

Final filtered protocol (Data_Batch_4).  The *raw* and *downsampled* arrays are
never modified: this script only derives manifests that record which of the
12000 source sequences are retained for time-series training.

Filter rule
-----------
    duration_ratio = simulation_end_s / reference_end_s
    keep  iff  duration_ratio >= KEEP_THRESHOLD   (default 0.8)

The error-metric task is unaffected (it keeps all 12000 rows); only the
time-series trajectory models use the filtered set, because a sequence whose
simulation terminated early has a long held / extrapolated tail that would
otherwise dominate trajectory metrics.

Outputs (under data/Data_Batch_4_TSFiltered_0p8/)
    time_series_source_manifest.csv    all 12000 sequences + duration_ratio + kept flag
    time_series_kept_manifest.csv      kept subset
    time_series_removed_manifest.csv   removed subset
    removed_sequences_by_case.csv      per experiment_id removed / kept / total
    duration_ratio_summary.csv         overall + per-case duration-ratio stats
    filtering_audit.md                 human-readable audit
    filter_meta.json                   machine-readable provenance (source dirs, checksums)

All counts are derived from data; nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Source of the simulation/reference end times (immutable, audited dataset).
SOURCE_DS = "data/Data_Batch_4_downsampled_160"
RAW_DS = "data/Data_Batch_4_raw"
OUT_DIR = "data/Data_Batch_4_TSFiltered_0p8"
KEEP_THRESHOLD = 0.8


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(source_ds: str, out_dir: str, threshold: float) -> dict:
    src = REPO / source_ds
    out = REPO / out_dir
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = src / "sequence_manifest.csv"
    man = pd.read_csv(manifest_path)

    required = {"sequence_id", "sample_id", "experiment_id",
                "reference_end_s", "simulation_end_s"}
    missing = required - set(man.columns)
    if missing:
        raise KeyError(f"sequence_manifest.csv missing columns: {sorted(missing)}")

    # duration_ratio derived from data; guard against zero/negative reference.
    ref = man["reference_end_s"].to_numpy(dtype=np.float64)
    sim = man["simulation_end_s"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref > 0, sim / ref, np.nan)
    man = man.copy()
    man["duration_ratio"] = ratio
    man["kept"] = man["duration_ratio"] >= threshold

    source = man.sort_values("sequence_id").reset_index(drop=True)
    kept = source[source["kept"]].reset_index(drop=True)
    removed = source[~source["kept"]].reset_index(drop=True)

    # ---- per-case removal table (derived) ----
    by_case = (
        source.groupby("experiment_id")
        .agg(total=("sequence_id", "size"),
             kept=("kept", "sum"))
        .reset_index()
    )
    by_case["removed"] = by_case["total"] - by_case["kept"]
    by_case = by_case.sort_values("removed", ascending=False).reset_index(drop=True)
    by_case = by_case[["experiment_id", "total", "kept", "removed"]]

    # ---- duration-ratio summary (overall + per case) ----
    def _stats(s: pd.Series, label: str) -> dict:
        v = s.to_numpy(dtype=np.float64)
        v = v[np.isfinite(v)]
        return {
            "scope": label,
            "n": int(v.size),
            "min": float(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "mean": float(np.mean(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
            "n_below_threshold": int(np.sum(v < threshold)),
        }

    summ_rows = [_stats(source["duration_ratio"], "ALL")]
    for exp, sub in source.groupby("experiment_id"):
        summ_rows.append(_stats(sub["duration_ratio"], exp))
    summary = pd.DataFrame(summ_rows)

    # ---- write csvs ----
    cols = ["sequence_id", "sample_id", "experiment_id", "reference_end_s",
            "simulation_end_s", "duration_ratio", "kept"]
    source[cols].to_csv(out / "time_series_source_manifest.csv", index=False)
    kept[cols].to_csv(out / "time_series_kept_manifest.csv", index=False)
    removed[cols].to_csv(out / "time_series_removed_manifest.csv", index=False)
    by_case.to_csv(out / "removed_sequences_by_case.csv", index=False)
    summary.to_csv(out / "duration_ratio_summary.csv", index=False)

    # ---- provenance ----
    meta = {
        "dataset_role": "time_series_duration_ratio_filtered",
        "keep_rule": "duration_ratio = simulation_end_s / reference_end_s >= threshold",
        "keep_threshold": threshold,
        "source_time_series_dir": source_ds,
        "raw_dir": RAW_DS,
        "error_metric_data_dir": RAW_DS,
        "error_metric_rows_unfiltered": int(len(source)),
        "n_source": int(len(source)),
        "n_kept": int(len(kept)),
        "n_removed": int(len(removed)),
        "n_unique_sample_ids_source": int(source["sample_id"].nunique()),
        "n_unique_sample_ids_kept": int(kept["sample_id"].nunique()),
        "source_sequence_manifest_sha256": _sha256(manifest_path),
        "note": ("Time-series trajectory models train on the kept subset with "
                 "valid-time masking; the error-metric surrogate keeps all rows. "
                 "Raw and downsampled arrays are never modified."),
    }
    (out / "filter_meta.json").write_text(json.dumps(meta, indent=2))

    # ---- audit markdown ----
    md = []
    md.append("# Time-series duration-ratio filtering audit\n")
    md.append(f"- **Source manifest**: `{source_ds}/sequence_manifest.csv`")
    md.append(f"- **Keep rule**: `duration_ratio = simulation_end_s / reference_end_s >= {threshold}`")
    md.append(f"- **Source sequences**: {len(source)}")
    md.append(f"- **Kept**: {len(kept)}")
    md.append(f"- **Removed**: {len(removed)}")
    md.append(f"- **Unique sample_ids (source / kept)**: "
              f"{source['sample_id'].nunique()} / {kept['sample_id'].nunique()}")
    md.append("- **Error-metric task**: unaffected — all "
              f"{len(source)} rows retained.\n")
    md.append("## Removed sequences by case\n")
    md.append("| experiment_id | total | kept | removed |")
    md.append("|---|---|---|---|")
    for _, r in by_case.iterrows():
        md.append(f"| {r['experiment_id']} | {int(r['total'])} | "
                  f"{int(r['kept'])} | {int(r['removed'])} |")
    md.append("\n## Duration-ratio summary (overall)\n")
    s0 = summ_rows[0]
    md.append(f"min={s0['min']:.4f}  median={s0['median']:.4f}  "
              f"mean={s0['mean']:.4f}  max={s0['max']:.4f}  "
              f"n_below_{threshold}={s0['n_below_threshold']}")
    md.append("")
    (out / "filtering_audit.md").write_text("\n".join(md))

    return {
        "n_source": len(source),
        "n_kept": len(kept),
        "n_removed": len(removed),
        "by_case": by_case,
        "out_dir": str(out),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-ds", default=SOURCE_DS)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--threshold", type=float, default=KEEP_THRESHOLD)
    a = p.parse_args()
    res = build(a.source_ds, a.out_dir, a.threshold)
    print(f"[filter] source={res['n_source']} kept={res['n_kept']} "
          f"removed={res['n_removed']}  -> {res['out_dir']}")
    print("[filter] top removed-by-case:")
    print(res["by_case"].head(6).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
