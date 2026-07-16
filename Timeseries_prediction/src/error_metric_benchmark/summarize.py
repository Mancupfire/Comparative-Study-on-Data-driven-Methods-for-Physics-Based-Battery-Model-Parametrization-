"""Aggregate a completed benchmark run into summary tables.

Reads every ``metrics/<family>/seed<seed>/metrics.json`` under a run dir and
writes (into ``<run_dir>/tables/`` by default):

    metrics_by_model_and_seed.csv   one row per (model, seed)
    metrics_by_model.csv            mean +/- std across seeds, per model
    metrics_by_target.csv           per-target metrics, mean across seeds
    ranking_table.csv               models ranked by mean test norm_overall_RMSE
    experiment_summary.md           human-readable summary
    resolved_config.yaml            copy of the run's resolved config
    split_audit.json                copy of the run's split audit

Also callable as a module: ``python -m src.error_metric_benchmark.summarize <run_dir>``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .models import DISPLAY_NAMES, FAMILY_ORDER

TEST_FIELDS = ["norm_overall_RMSE", "norm_overall_MAE", "mean_R2"]


def _load_run(run_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    mroot = run_dir / "metrics"
    for family in FAMILY_ORDER:
        fdir = mroot / family
        if not fdir.is_dir():
            continue
        for seed_dir in sorted(fdir.glob("seed*")):
            mp = seed_dir / "metrics.json"
            if not mp.is_file():
                continue
            d = json.loads(mp.read_text())
            seed = int(seed_dir.name.replace("seed", ""))
            te = d["metrics"]["test"]
            pt = te["per_target"]
            ov = te["overall"]
            row = {"model": family, "display": DISPLAY_NAMES.get(family, family),
                   "seed": seed,
                   "param_count": d.get("param_count"),
                   "inference_time_s": d.get("inference_time_s"),
                   "inference_ms_per_sample": d.get("inference_ms_per_sample"),
                   **{f"overall_{k}": ov[k] for k in TEST_FIELDS}}
            for tname, tm in pt.items():
                for mk, mv in tm.items():
                    row[f"{tname}.{mk}"] = mv
            rows.append(row)
    return pd.DataFrame(rows)


def _agg_by_model(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    num_cols = [c for c in df.columns if c not in ("model", "display", "seed")]
    g = df.groupby("model")
    out = g[num_cols].agg(["mean", "std"]).reset_index()
    out.columns = ["model"] + [f"{a}_{b}" for a, b in out.columns[1:]]
    out["display"] = out["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    out["n_seeds"] = g.size().values
    order = {m: i for i, m in enumerate(FAMILY_ORDER)}
    return out.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def _by_target(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    targets = ["rmse_voltage_mv", "rmse_temperature_c"]
    metrics = ["MAE", "RMSE", "R2", "MaxError", "rel_mean", "rel_median", "rel_p95"]
    for model, sub in df.groupby("model"):
        for t in targets:
            row = {"model": model, "display": DISPLAY_NAMES.get(model, model), "target": t}
            for mk in metrics:
                col = f"{t}.{mk}"
                if col in sub:
                    row[f"{mk}_mean"] = float(sub[col].mean())
                    row[f"{mk}_std"] = float(sub[col].std(ddof=0))
            rows.append(row)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(FAMILY_ORDER)}
    return out.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def _ranking(by_model: pd.DataFrame) -> pd.DataFrame:
    if by_model.empty:
        return by_model
    key = "overall_norm_overall_RMSE_mean"
    cols = ["model", "display", key, "overall_mean_R2_mean", "param_count_mean",
            "inference_ms_per_sample_mean", "n_seeds"]
    cols = [c for c in cols if c in by_model.columns]
    out = by_model[cols].sort_values(key).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def summarize(run_dir: Path, tables_dir: Path | None = None) -> Dict[str, Path]:
    run_dir = Path(run_dir)
    tables_dir = tables_dir or (run_dir / "tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = _load_run(run_dir)
    if df.empty:
        raise SystemExit(f"No metrics found under {run_dir}/metrics")

    by_seed = df.sort_values(["model", "seed"])
    by_model = _agg_by_model(df)
    by_target = _by_target(df)
    ranking = _ranking(by_model)

    paths = {}
    paths["by_seed"] = tables_dir / "metrics_by_model_and_seed.csv"
    paths["by_model"] = tables_dir / "metrics_by_model.csv"
    paths["by_target"] = tables_dir / "metrics_by_target.csv"
    paths["ranking"] = tables_dir / "ranking_table.csv"
    by_seed.to_csv(paths["by_seed"], index=False)
    by_model.to_csv(paths["by_model"], index=False)
    by_target.to_csv(paths["by_target"], index=False)
    ranking.to_csv(paths["ranking"], index=False)

    # Copy resolved config + split audit if present.
    for name in ("resolved_config.yaml", "split_audit.json"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, tables_dir / name)
            paths[name] = tables_dir / name

    # Experiment summary markdown.
    md = ["# Error-metric benchmark — experiment summary", ""]
    mani = run_dir / "run_manifest.json"
    if mani.is_file():
        m = json.loads(mani.read_text())
        md += [f"- **run_id**: `{m.get('run_id')}`",
               f"- **protocol**: `{m.get('protocol')}`",
               f"- **seeds**: {m.get('seeds')}",
               f"- **models**: {len(m.get('models', []))}",
               f"- **failures**: {m.get('n_failures')}", ""]
    md += ["## Ranking (by mean test normalized overall RMSE, lower is better)", ""]
    md += ["| Rank | Model | Norm. overall RMSE | Mean R² | Params | Infer ms/sample |",
           "|---|---|---|---|---|---|"]
    for _, r in ranking.iterrows():
        md.append(
            f"| {int(r['rank'])} | {r['display']} | "
            f"{r.get('overall_norm_overall_RMSE_mean', float('nan')):.4f} | "
            f"{r.get('overall_mean_R2_mean', float('nan')):.4f} | "
            f"{r.get('param_count_mean', float('nan')):.0f} | "
            f"{r.get('inference_ms_per_sample_mean', float('nan')):.4f} |")
    md += ["", "Overall metrics are scale-safe: normalized overall RMSE/MAE are "
           "computed in standardized two-target space (per-target z-score using "
           "true-test std); mV and °C are never mixed in a raw aggregate."]
    summ = tables_dir / "experiment_summary.md"
    summ.write_text("\n".join(md))
    paths["experiment_summary"] = summ
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Summarize an error-metric benchmark run")
    p.add_argument("run_dir")
    p.add_argument("--tables-dir", default=None)
    a = p.parse_args(argv)
    out = summarize(Path(a.run_dir), Path(a.tables_dir) if a.tables_dir else None)
    for k, v in out.items():
        print(f"[summary] {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
