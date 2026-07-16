#!/usr/bin/env python3
"""Export Batch-4 time-series test predictions from EXISTING checkpoints.

Loads the already-trained ``best_model.pt`` for every (case, model) in the
completed run ``batch4_full_20260621_140149`` and writes long-format test-set
prediction CSVs.  NO retraining occurs: the model is rebuilt from the checkpoint
and only forward inference is run.  The deterministic train-only split/scalers
are reconstructed from each model's saved ``run_config.json`` (seed + ratios),
exactly as the existing src/predict.predict_case does.

If a checkpoint cannot be loaded, the exact incompatibility is reported (and
recorded in the export report) instead of being silently skipped.

Output (one file per case/model):
    <ts_run>/predictions/<case>/<model>/test_predictions.csv
Columns:
    sequence_id, sample_id, case_id, model_name, time_index, time_s,
    voltage_true, voltage_pred, temperature_true, temperature_pred
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.predict import predict_case  # noqa: E402
from src.utils import ensure_dir, save_json  # noqa: E402

DEFAULT_RUN = "batch4_full_20260621_140149"
DEFAULT_TS_ROOT = "outputs/Data_Batch_4/time_series_downsampled_160"
DEFAULT_DATA_ROOT = "data/Data_Batch_4_downsampled_160"
MODELS = ["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]


def discover_cases(run_dir: Path) -> list[str]:
    cdir = run_dir / "checkpoints"
    return sorted(p.name for p in cdir.iterdir() if p.is_dir())


def export_one(data_root, run_dir: Path, case_id: str, model: str,
               device: str) -> dict:
    ck = run_dir / "checkpoints" / case_id / model / "best_model.pt"
    if not ck.is_file():
        return {"case": case_id, "model": model, "status": "missing_checkpoint",
                "detail": str(ck), "rows": 0}
    try:
        out = predict_case(data_root, case_id, model,
                           outputs_dir=str(run_dir), split="test", device=device)
    except Exception as exc:  # noqa: BLE001
        return {"case": case_id, "model": model, "status": "load_failed",
                "detail": f"{type(exc).__name__}: {exc}", "rows": 0,
                "traceback": traceback.format_exc()}

    v_pred, t_pred = out["v_pred"], out["t_pred"]
    v_true, t_true = out["v_true"], out["t_true"]
    time_s, sids = out["time_s"], np.asarray(out["sample_ids"]).astype(str)
    n, t_last = v_pred.shape

    ti = np.tile(np.arange(t_last), n)
    ts = np.tile(np.asarray(time_s), n)
    sid_rep = np.repeat(sids, t_last)
    seq = np.array([f"{s}__{case_id}" for s in sid_rep])
    df = pd.DataFrame({
        "sequence_id": seq, "sample_id": sid_rep, "case_id": case_id,
        "model_name": model, "time_index": ti, "time_s": ts,
        "voltage_true": v_true.reshape(-1), "voltage_pred": v_pred.reshape(-1),
        "temperature_true": t_true.reshape(-1), "temperature_pred": t_pred.reshape(-1),
    })
    finite = bool(np.all(np.isfinite(df[["voltage_pred", "temperature_pred"]].to_numpy())))
    pdir = ensure_dir(run_dir / "predictions" / case_id / model)
    df.to_csv(pdir / "test_predictions.csv", index=False)
    return {"case": case_id, "model": model, "status": "ok", "rows": len(df),
            "n_sequences": int(n), "t_last": int(t_last), "finite": finite}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default=DEFAULT_RUN)
    p.add_argument("--ts-root", default=DEFAULT_TS_ROOT)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p.add_argument("--device", default="auto")
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    a = p.parse_args(argv)

    run_dir = Path(a.ts_root) / a.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}"); return 1
    cases = a.cases or discover_cases(run_dir)
    models = a.models or MODELS

    print(f"[ts-export] run={a.run_id} cases={len(cases)} models={len(models)}")
    report = []
    for case_id in cases:
        for model in models:
            r = export_one(a.data_root, run_dir, case_id, model, a.device)
            report.append(r)
            flag = "OK " if r["status"] == "ok" else "!! "
            print(f"  {flag}{case_id}/{model}: {r['status']} rows={r['rows']}"
                  + ("" if r["status"] == "ok" else f"  -> {r.get('detail','')}"))

    n_ok = sum(1 for r in report if r["status"] == "ok")
    n_fail = len(report) - n_ok
    total_rows = sum(r["rows"] for r in report)
    summary = {"run_id": a.run_id, "n_combinations": len(report),
               "n_ok": n_ok, "n_failed": n_fail, "total_rows": total_rows,
               "results": report}
    save_json(summary, run_dir / "predictions" / "export_report.json")
    print(f"[ts-export] ok={n_ok} failed={n_fail} total_rows={total_rows}")
    print(f"[ts-export] report -> {run_dir/'predictions'/'export_report.json'}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
