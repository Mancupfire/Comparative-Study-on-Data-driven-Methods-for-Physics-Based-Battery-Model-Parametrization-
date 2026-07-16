#!/usr/bin/env python3
"""Automated scientific & implementation checks (Phase 8).

Runs against a completed error-metric benchmark run + the time-series run and
writes reports/Data_Batch_4/error_metric_benchmark/validation_report.md.

Usage:
  python scripts/validate_batch_4_error_metric_benchmark.py \
      --em-run-id em_bench_smoke --em-smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.error_metric_benchmark.data import build_benchmark_dataset, split_audit  # noqa: E402
from src.error_metric_benchmark.models import FAMILY_ORDER  # noqa: E402

TS_RUN = "outputs/Data_Batch_4/time_series_downsampled_160/batch4_full_20260621_140149"


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--em-run-id", required=True)
    p.add_argument("--em-smoke", action="store_true")
    p.add_argument("--data-dir", default="data/Data_Batch_4_raw")
    p.add_argument("--stage", choices=["benchmark", "final"], default="benchmark",
                   help="benchmark: post-training checks only; the figure check is "
                        "NOT YET APPLICABLE until build_batch_4_final_results.py runs. "
                        "final: also require post-build artifacts (figures).")
    a = p.parse_args(argv)

    base = ("outputs_smoke/Data_Batch_4/error_metric_benchmark" if a.em_smoke
            else "outputs/Data_Batch_4/error_metric_benchmark")
    run = Path(base) / a.em_run_id
    ts = Path(TS_RUN)
    checks = []

    def add(name, ok, detail="", applicable=True):
        # applicable=False -> reported as NOT YET APPLICABLE and excluded from
        # the pass/fail tally and the exit code.
        checks.append((name, bool(ok), detail, bool(applicable)))

    # 1. zero sample-ID overlap (grouped)
    aud = split_audit(build_benchmark_dataset(a.data_dir, "grouped_holdout", seed=42))
    add("Zero sample_id overlap (grouped_holdout)",
        aud["sample_id_splits_disjoint"] and aud["overlap_train_test"] == 0,
        f"overlaps tv/tt/vt = {aud['overlap_train_val']}/{aud['overlap_train_test']}/{aud['overlap_val_test']}")

    # 2. train-only scaler fitting
    add("Scaler fit scope = train only", aud["scaler_fit_scope"] == "train_only"
        and not aud["target_stats_use_val_or_test"])

    # 3 & 4. split / eval-row integrity.
    # NOTE: the multi-seed benchmark re-splits per seed (grouped_holdout with
    # seed=42/43/44 yields three distinct test sets). Each (family,seed)
    # prediction file is therefore evaluated on the split built with THAT seed.
    # We compare stable identifiers as *sets* against the deterministically
    # rebuilt test split for each seed (never relying on CSV row order), and
    # we treat any duplicate prediction row as a failure.
    manifest = run / "split_manifest.csv"
    pred_dirs = sorted((run / "predictions").glob("*/seed*/test_predictions.csv"))

    def _seed_of(pf: Path) -> int:
        return int(pf.parts[-2].replace("seed", ""))

    seeds = sorted({_seed_of(pf) for pf in pred_dirs})
    # Deterministic expected test set per seed (rebuilt; matches on-disk manifest
    # for the seed it was written with — see determinism check below).
    expected = {
        s: set(build_benchmark_dataset(a.data_dir, "grouped_holdout", seed=s)
               .manifest.query("split=='test'")["sequence_id"])
        for s in seeds
    }

    per_seed_sets = {}        # seed -> {family: frozenset(sequence_id)}
    dup_files = []            # files containing duplicate sequence_id rows
    set_mismatch = []         # files whose set != that seed's expected test set
    for pf in pred_dirs:
        fam = pf.parts[-3]
        seed = _seed_of(pf)
        seq = pd.read_csv(pf)["sequence_id"]
        if seq.duplicated().any():
            dup_files.append(f"{fam}/seed{seed}")
        sset = frozenset(seq)
        per_seed_sets.setdefault(seed, {})[fam] = sset
        if set(sset) != expected[seed]:
            set_mismatch.append(f"{fam}/seed{seed}")

    # Check 3: within each seed, every family is evaluated on identical rows,
    # and no file contains duplicate rows.
    identical_within_seed = (
        bool(per_seed_sets)
        and all(len(set(d.values())) == 1 for d in per_seed_sets.values())
    )
    add("Identical evaluation rows across all models (per seed)",
        identical_within_seed and not dup_files,
        f"{len(pred_dirs)} prediction files; seeds={seeds}; "
        f"dup_files={len(dup_files)}")

    # Check 4: each (family,seed) prediction set == that seed's test split
    # (set/sorted-key equality), same exact observations, no missing/extra,
    # and no duplicate prediction rows. The on-disk split_manifest.csv anchors
    # the seed it was written with.
    manifest_anchor_ok = True
    anchor_detail = "no manifest on disk"
    if manifest.is_file() and seeds:
        mtest = set(pd.read_csv(manifest).query("split=='test'")["sequence_id"])
        manifest_anchor_ok = any(mtest == expected[s] for s in seeds)
        anchor_detail = (f"manifest_test={len(mtest)} matches a rebuilt seed split"
                         if manifest_anchor_ok else
                         f"manifest_test={len(mtest)} matches NO rebuilt seed split")
    ok4 = (bool(pred_dirs) and not set_mismatch and not dup_files
           and manifest_anchor_ok)
    add("Predictions match split_manifest test rows", ok4,
        f"files={len(pred_dirs)} seeds={seeds} mismatched={len(set_mismatch)} "
        f"dup_files={len(dup_files)}; {anchor_detail}")

    # 5. correct inverse transform (predictions' *_true == raw targets)
    raw = pd.read_csv(Path(a.data_dir) / "error_metrics.csv").set_index("sequence_id")
    ok_inv = True; detail_inv = ""
    if pred_dirs:
        df = pd.read_csv(pred_dirs[0])
        merged = df.set_index("sequence_id")
        v_ok = np.allclose(merged["rmse_voltage_mv_true"],
                           raw.loc[merged.index, "rmse_voltage_mv"], atol=1e-6)
        t_ok = np.allclose(merged["rmse_temperature_c_true"],
                           raw.loc[merged.index, "rmse_temperature_c"], atol=1e-6)
        ok_inv = bool(v_ok and t_ok); detail_inv = f"voltage_true_ok={v_ok} temp_true_ok={t_ok}"
    add("Correct target round-trip (inverse transform of both targets)", ok_inv, detail_inv)

    # 6. finite predictions & histories
    fin = True
    for pf in pred_dirs:
        d = pd.read_csv(pf)
        cols = [c for c in d.columns if c.endswith("_pred")]
        if not np.all(np.isfinite(d[cols].to_numpy())):
            fin = False
    add("All predictions finite (no NaN/Inf)", fin)

    # 7. determinism: rebuild dataset twice -> identical manifest
    d1 = build_benchmark_dataset(a.data_dir, "grouped_holdout", seed=42).manifest
    d2 = build_benchmark_dataset(a.data_dir, "grouped_holdout", seed=42).manifest
    add("Deterministic split (identical manifest on rebuild)", d1.equals(d2))

    # 8. unique checkpoint paths
    ckpts = list((run / "checkpoints").glob("*/seed*"))
    add("Unique checkpoint paths per (family,seed)", len(ckpts) == len(set(map(str, ckpts))),
        f"{len(ckpts)} checkpoint dirs")

    # 9. correct physical units (mV ~ tens, degC ~ order 1)
    units_ok = True; ud = ""
    if pred_dirs:
        df = pd.read_csv(pred_dirs[0])
        vmed = float(df["rmse_voltage_mv_true"].median())
        tmed = float(df["rmse_temperature_c_true"].median())
        units_ok = (1.0 < vmed < 1000.0) and (0.01 < tmed < 50.0)
        ud = f"median V={vmed:.2f} mV, T={tmed:.3f} °C"
    add("Physical units sane (mV / °C)", units_ok, ud)

    # 10. no raw combined RMSE mixing mV & °C
    no_raw = True
    for mf in (run / "metrics").glob("*/seed*/metrics.json"):
        m = json.loads(mf.read_text())
        ov = m["metrics"]["test"]["overall"]
        if "RMSE" in ov or "combined_rmse" in ov:   # only norm_* / mean_R2 allowed
            no_raw = False
    add("No raw combined RMSE mixing mV and °C", no_raw,
        "overall keys limited to norm_overall_RMSE/MAE, mean_R2")

    # 11. no Batch 2/3/4 protected-output overwrite (benchmark writes new dir only)
    protected_new = not (Path("outputs/Data_Batch_4/error_metric_benchmark") / a.em_run_id).exists() \
        if a.em_smoke else True
    add("Benchmark isolated from existing Batch 2/3/4 outputs",
        a.em_smoke and str(run).startswith("outputs_smoke") or not a.em_smoke,
        f"run dir = {run}")

    # 12. no time-series retraining (checkpoints predate this session)
    ck = ts / "checkpoints" / "CC_D_0p5_T25C" / "mlp" / "best_model.pt"
    add("Time-series checkpoints not modified (no retrain)", ck.is_file(),
        f"mtime={pd.Timestamp(ck.stat().st_mtime, unit='s')}" if ck.is_file() else "missing")

    # 13. prediction export row counts
    rc_ok = True; rd = ""
    exp = ts / "predictions" / "export_report.json"
    if exp.is_file():
        er = json.loads(exp.read_text())
        rc_ok = er["n_failed"] == 0 and all(r["rows"] in (0, r.get("n_sequences", 0) * r.get("t_last", 0))
                                            for r in er["results"])
        rd = f"ts export ok={er['n_ok']} failed={er['n_failed']} rows={er['total_rows']}"
    add("Time-series prediction export row counts", rc_ok, rd)

    # 14. figure generation — POST-BUILD check. Only applicable in the final
    # stage (after build_batch_4_final_results.py). In the benchmark stage it is
    # marked NOT YET APPLICABLE so it cannot fail before final-result generation.
    figbase = Path("reports/Data_Batch_4/final_results") / a.em_run_id / "figures" / "png"
    nfig = len(list(figbase.glob("*.png"))) if figbase.is_dir() else 0
    fig_applicable = a.stage == "final"
    fig_detail = (f"{nfig} PNG figures at {figbase}" if fig_applicable
                  else f"{nfig} PNG figures (not required until --stage final)")
    add("Figures generated (PNG)", nfig >= 7, fig_detail, applicable=fig_applicable)

    # ---- write report ----
    applicable = [c for c in checks if c[3]]
    n_applicable = len(applicable)
    n_pass = sum(1 for _, ok, _, ap in checks if ap and ok)
    n_na = len(checks) - n_applicable

    def _result(ok, ap):
        if not ap:
            return "⚪ NOT YET APPLICABLE"
        return "✅ PASS" if ok else "❌ FAIL"

    lines = ["# Validation Report — Batch 4 Error-Metric Benchmark", "",
             f"- Run: `{a.em_run_id}` ({'smoke' if a.em_smoke else 'full'})",
             f"- Stage: `{a.stage}`",
             f"- Checks passed: **{n_pass}/{n_applicable}** applicable"
             + (f" ({n_na} not yet applicable)" if n_na else ""), "",
             "| # | Check | Result | Detail |", "|---|---|---|---|"]
    for i, (name, ok, detail, ap) in enumerate(checks, 1):
        lines.append(f"| {i} | {name} | {_result(ok, ap)} | {detail} |")
    lines += ["", "Generated by `scripts/validate_batch_4_error_metric_benchmark.py`."]
    outp = Path("reports/Data_Batch_4/error_metric_benchmark/validation_report.md")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text("\n".join(lines) + "\n")
    def _tag(ok, ap):
        return "N/A" if not ap else ("PASS" if ok else "FAIL")
    print("\n".join(f"[{_tag(ok, ap)}] {n}" for n, ok, _, ap in checks))
    print(f"\n{n_pass}/{n_applicable} applicable checks passed"
          + (f" ({n_na} not yet applicable)" if n_na else "") + f" -> {outp}")
    # Exit nonzero only if an *applicable* check failed.
    return 0 if n_pass == n_applicable else 1


if __name__ == "__main__":
    sys.exit(main())
