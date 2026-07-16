"""Task B — read-only schema/join/split/leakage audit (NO training).

Reports the joined dataset shape, feature list, target stats, join cardinality
(unmatched / duplicated rows) and the sample_id split + leakage check. Writes a
JSON summary under the data-audit namespace.

Usage:
  python scripts/audit_error_metric.py \
    --data-dir data/Data_Batch_2 \
    --out outputs/Data_Batch_2/data_audit/error_metric/data_quality_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.error_metric_data import build_dataset, leakage_check
from src.utils import save_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Data_Batch_2")
    ap.add_argument("--out", default="outputs/Data_Batch_2/data_audit/error_metric/data_quality_summary.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = build_dataset(args.data_dir, seed=args.seed)
    leak = leakage_check(ds)
    jr = ds.join_report.as_dict()

    # Physical target stats from the joined frame.
    tgt = ds.raw_feature_frame[ds.target_names]
    target_stats = {c: {"min": float(tgt[c].min()), "max": float(tgt[c].max()),
                        "mean": float(tgt[c].mean()), "std": float(tgt[c].std()),
                        "n_nonfinite": int((~np.isfinite(tgt[c])).sum())}
                    for c in ds.target_names}

    summary = {
        "data_dir": args.data_dir,
        "n_features": ds.n_features,
        "feature_names": ds.feature_names,
        "continuous_feature_idx": ds.continuous_feature_idx,
        "categorical_feature_idx": ds.categorical_feature_idx,
        "target_names": ds.target_names,
        "target_stats": target_stats,
        "join_report": jr,
        "split_counts": {
            "samples": {k: len(v) for k, v in ds.split_sample_ids.items()},
            "rows": {"train": int(ds.X_train.shape[0]), "val": int(ds.X_val.shape[0]),
                     "test": int(ds.X_test.shape[0])},
        },
        "leakage_check": leak,
        "feature_nonfinite_total": int((~np.isfinite(
            np.concatenate([ds.X_train, ds.X_val, ds.X_test]))).sum()),
    }
    save_json(summary, args.out)
    print(f"[audit-b] joined rows={jr['n_joined']} features={ds.n_features} "
          f"one_to_one={jr['metrics_manifest_one_to_one']} "
          f"params_complete={jr['joined_params_complete']}")
    print(f"[audit-b] split rows train/val/test="
          f"{summary['split_counts']['rows']['train']}/"
          f"{summary['split_counts']['rows']['val']}/"
          f"{summary['split_counts']['rows']['test']}  "
          f"leakage_free={leak['leakage_free']}")
    print(f"[audit-b] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
