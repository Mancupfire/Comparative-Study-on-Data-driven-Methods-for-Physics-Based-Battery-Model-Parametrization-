"""Build a Batch 2 vs Batch 3 comparison bundle (read-only on both runs).

Writes ONLY under outputs/comparisons/Batch_2_vs_Batch_3/<comparison_id>/.
Never writes into any Batch 2 or Batch 3 model directory.
"""
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone

import pandas as pd


def _load(run_dir, name):
    p = os.path.join(run_dir, name)
    return pd.read_csv(p) if os.path.isfile(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch2-run-dir", required=True)
    ap.add_argument("--batch3-run-dir", required=True)
    ap.add_argument("--comparison-id", default="cmp_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--out-root", default="outputs/comparisons/Batch_2_vs_Batch_3")
    args = ap.parse_args()

    out = os.path.join(args.out_root, args.comparison_id)
    os.makedirs(out, exist_ok=True)

    # dataset-level relationship (from Phase 4 artifacts, if present)
    for src in ["reports/new_batch_setup/dataset_relationship_report.md",
                "reports/new_batch_setup/batch2_vs_new_dataset_comparison.csv"]:
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out, os.path.basename(src)))

    rows = []
    for name in ["metrics_by_model.csv", "metrics_summary.csv", "metrics_by_target.csv"]:
        b2 = _load(args.batch2_run_dir, name)
        b3 = _load(args.batch3_run_dir, name)
        if b2 is not None:
            b2.insert(0, "batch", "Batch_2")
        if b3 is not None:
            b3.insert(0, "batch", "Batch_3")
        merged = pd.concat([x for x in (b2, b3) if x is not None], ignore_index=True) \
            if (b2 is not None or b3 is not None) else None
        if merged is not None:
            merged.to_csv(os.path.join(out, f"compare_{name}"), index=False)
            if name == "metrics_by_model.csv":
                rows = merged

    md = [f"# Batch 2 vs Batch 3 comparison — {args.comparison_id}", "",
          f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
          f"- Batch 2 run: `{args.batch2_run_dir}`",
          f"- Batch 3 run: `{args.batch3_run_dir}`", "",
          "## Important",
          "Batch 2 and Batch 3 are **separate experiments** on **independent datasets**",
          "(0 parameter-vector overlap; 500 vs 1000 samples; coarser c-rate-dependent",
          "native sampling in Batch 3). Per-case numbers are NOT paired across batches",
          "because the parameter sets differ. Treat this as a cross-dataset model-behaviour",
          "comparison, not a same-data ablation.", "",
          "See `compare_metrics_by_model.csv`, `compare_metrics_summary.csv`,",
          "`dataset_relationship_report.md` in this folder.", ""]
    open(os.path.join(out, "comparison_summary.md"), "w").write("\n".join(md) + "\n")
    print(f"[compare] wrote {out}")


if __name__ == "__main__":
    main()
