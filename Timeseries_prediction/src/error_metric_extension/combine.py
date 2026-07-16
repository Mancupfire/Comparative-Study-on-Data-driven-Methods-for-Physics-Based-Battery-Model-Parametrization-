"""Combine reused + new error-metric families into one final ranking.

Final error-metric model set (8 families)::

    reused : ann, mlp, gated_mlp, deep_ensemble_mlp, extratrees
    new    : random_forest, xgboost, catboost

Reused metrics are copied verbatim from the completed grouped benchmark run;
new metrics come from the extension run.  The combined run dir then holds a
``metrics/<family>/seed<seed>/metrics.json`` tree for all eight families, from
which ranking tables are produced.  Nothing in the source runs is modified.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import ensure_dir, save_json  # noqa: E402

REUSED_FAMILIES = ["ann", "mlp", "gated_mlp", "deep_ensemble_mlp", "extratrees"]
NEW_FAMILIES = ["random_forest", "xgboost", "catboost"]
FINAL_FAMILY_ORDER = REUSED_FAMILIES + NEW_FAMILIES

DISPLAY_NAMES = {
    "ann": "ANN", "mlp": "MLP", "gated_mlp": "Gated MLP",
    "deep_ensemble_mlp": "Deep Ensemble MLP", "extratrees": "ExtraTrees",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "catboost": "CatBoost",
}

TEST_FIELDS = ["norm_overall_RMSE", "norm_overall_MAE", "mean_R2"]


def _copy_family_seeds(src_metrics_root: Path, dst_metrics_root: Path,
                       family: str, seeds: List[int]) -> int:
    n = 0
    for seed in seeds:
        src = src_metrics_root / family / f"seed{seed}" / "metrics.json"
        if not src.is_file():
            continue
        dst = ensure_dir(dst_metrics_root / family / f"seed{seed}")
        shutil.copy2(src, dst / "metrics.json")
        n += 1
    return n


def assemble(combined_dir: Path, reused_run: Path, new_run: Path,
             seeds: List[int]) -> Dict:
    metrics_root = ensure_dir(combined_dir / "metrics")
    counts = {}
    for fam in REUSED_FAMILIES:
        counts[fam] = _copy_family_seeds(reused_run / "metrics", metrics_root, fam, seeds)
    for fam in NEW_FAMILIES:
        counts[fam] = _copy_family_seeds(new_run / "metrics", metrics_root, fam, seeds)
    return counts


def _load(combined_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    mroot = combined_dir / "metrics"
    for fam in FINAL_FAMILY_ORDER:
        fdir = mroot / fam
        if not fdir.is_dir():
            continue
        for seed_dir in sorted(fdir.glob("seed*")):
            mp = seed_dir / "metrics.json"
            if not mp.is_file():
                continue
            d = json.loads(mp.read_text())
            te = d["metrics"]["test"]
            ov, pt = te["overall"], te["per_target"]
            row = {"model": fam, "display": DISPLAY_NAMES.get(fam, fam),
                   "seed": int(seed_dir.name.replace("seed", "")),
                   "param_count": d.get("param_count"),
                   "inference_ms_per_sample": d.get("inference_ms_per_sample"),
                   **{f"overall_{k}": ov[k] for k in TEST_FIELDS}}
            for tname, tm in pt.items():
                for mk, mv in tm.items():
                    row[f"{tname}.{mk}"] = mv
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(combined_dir: Path) -> Dict[str, Path]:
    tables = ensure_dir(combined_dir / "tables")
    df = _load(combined_dir)
    if df.empty:
        raise SystemExit(f"No metrics under {combined_dir}/metrics")
    order = {m: i for i, m in enumerate(FINAL_FAMILY_ORDER)}
    df = df.sort_values(["model", "seed"], key=lambda s: s.map(order) if s.name == "model" else s)

    num_cols = [c for c in df.columns if c not in ("model", "display", "seed")]
    g = df.groupby("model")
    by_model = g[num_cols].agg(["mean", "std"]).reset_index()
    by_model.columns = ["model"] + [f"{a}_{b}" for a, b in by_model.columns[1:]]
    by_model["display"] = by_model["model"].map(lambda m: DISPLAY_NAMES.get(m, m))
    by_model["n_seeds"] = g.size().values
    by_model = by_model.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)

    key = "overall_norm_overall_RMSE_mean"
    rank_cols = ["model", "display", key, "overall_mean_R2_mean",
                 "param_count_mean", "inference_ms_per_sample_mean", "n_seeds"]
    rank_cols = [c for c in rank_cols if c in by_model.columns]
    ranking = by_model[rank_cols].sort_values(key).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    p_seed = tables / "metrics_by_model_and_seed.csv"
    p_model = tables / "metrics_by_model.csv"
    p_rank = tables / "ranking_table.csv"
    df.to_csv(p_seed, index=False)
    by_model.to_csv(p_model, index=False)
    ranking.to_csv(p_rank, index=False)

    md = ["# Final error-metric ranking (8 families: 5 reused + 3 new)", ""]
    md += ["| Rank | Model | Norm. overall RMSE | Mean R² | Params | Infer ms/sample | Source |",
           "|---|---|---|---|---|---|---|"]
    for _, r in ranking.iterrows():
        src = "reused" if r["model"] in REUSED_FAMILIES else "new"
        md.append(f"| {int(r['rank'])} | {r['display']} | "
                  f"{r.get(key, float('nan')):.4f} | "
                  f"{r.get('overall_mean_R2_mean', float('nan')):.4f} | "
                  f"{r.get('param_count_mean', float('nan')):.0f} | "
                  f"{r.get('inference_ms_per_sample_mean', float('nan')):.4f} | {src} |")
    (tables / "experiment_summary.md").write_text("\n".join(md))
    return {"by_seed": p_seed, "by_model": p_model, "ranking": p_rank}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Combine reused + new EM families")
    p.add_argument("--combined-dir", required=True)
    p.add_argument("--reused-run", required=True,
                   help="Completed grouped benchmark run dir.")
    p.add_argument("--new-run", required=True,
                   help="Extension run dir (RF/XGB/CatBoost).")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    a = p.parse_args(argv)
    combined = ensure_dir(Path(a.combined_dir))
    counts = assemble(combined, Path(a.reused_run), Path(a.new_run), a.seeds)
    save_json({"copied_seed_counts": counts, "reused_run": a.reused_run,
               "new_run": a.new_run, "seeds": a.seeds,
               "final_family_order": FINAL_FAMILY_ORDER},
              combined / "combine_manifest.json")
    out = summarize(combined)
    print(f"[combine] families copied: {counts}")
    for k, v in out.items():
        print(f"[combine] {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
