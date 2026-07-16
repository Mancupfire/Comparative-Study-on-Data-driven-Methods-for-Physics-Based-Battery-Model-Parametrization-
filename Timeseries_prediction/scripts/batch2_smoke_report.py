"""Scan a Batch 2 smoke output dir and emit a pass/fail table + smoke_test_report.md.

Checks per model (on the smoke case):
  Train       -> metrics/<case>/<model>/history.csv exists & non-empty
  Validation  -> history.csv has a val_loss column with finite values
  Checkpoint  -> checkpoints/<case>/<model>/best_model.pt exists
  Prediction  -> metrics.json present (eval ran inference)
  Metrics     -> metrics.json has finite RMSE_V / RMSE_T

Usage
-----
python scripts/batch2_smoke_report.py --smoke-dir outputs_smoke/Data_Batch_2/time_series/<run_id> \
    --case-id CC_C_2p5_T25C --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke-dir", required=True)
    p.add_argument("--case-id", required=True)
    p.add_argument("--models", nargs="+", required=True)
    args = p.parse_args()
    sd = Path(args.smoke_dir)

    rows, all_pass = [], True
    for m in args.models:
        ck = sd / "checkpoints" / args.case_id / m / "best_model.pt"
        hist = sd / "metrics" / args.case_id / m / "history.csv"
        mj = sd / "metrics" / args.case_id / m / "metrics.json"

        train_ok = hist.is_file() and hist.stat().st_size > 0
        val_ok = False
        if train_ok:
            try:
                h = pd.read_csv(hist)
                val_ok = "val_loss" in h.columns and h["val_loss"].apply(_finite).all() \
                    and len(h) > 0
            except Exception:
                val_ok = False
        ckpt_ok = ck.is_file()
        pred_ok = mj.is_file()
        met_ok = False
        if pred_ok:
            try:
                rec = json.loads(mj.read_text())
                met_ok = _finite(rec.get("RMSE_V")) and _finite(rec.get("RMSE_T"))
            except Exception:
                met_ok = False
        status = all([train_ok, val_ok, ckpt_ok, pred_ok, met_ok])
        all_pass = all_pass and status
        rows.append({"model": m, "train": train_ok, "validation": val_ok,
                     "checkpoint": ckpt_ok, "prediction": pred_ok,
                     "metrics": met_ok, "status": "PASS" if status else "FAIL"})

    df = pd.DataFrame(rows)
    table = df.to_string(index=False)
    print("\nSMOKE TEST RESULTS (case = %s)" % args.case_id)
    print(table)
    print("\nOVERALL:", "ALL PASS" if all_pass else "FAILURE(S) PRESENT")

    md = ["# Batch 2 time-series smoke test report",
          f"\nCase: `{args.case_id}`\n",
          df.to_markdown(index=False),
          f"\n**Overall: {'ALL PASS' if all_pass else 'FAILURE(S) PRESENT'}**"]
    (sd / "smoke_test_report.md").write_text("\n".join(md) + "\n")
    return 0 if all_pass else 4


if __name__ == "__main__":
    raise SystemExit(main())
