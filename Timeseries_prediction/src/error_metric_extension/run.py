"""Train the three NEW error-metric families (RF / XGBoost / CatBoost).

Uses the same grouped dataset builder as the completed benchmark
(``build_benchmark_dataset``) so the split / features / targets / scalers are
identical and the new results are directly comparable with the reused families.

Example
-------
    python -m src.error_metric_extension.run \
        --config configs/batch_4/final_filtered_protocol/error_metric_extension.yaml \
        --run-id em_ext_full --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import ensure_dir, save_json  # noqa: E402
from src.error_metric_benchmark.data import build_benchmark_dataset, split_audit  # noqa: E402

from .trainer import NEW_FAMILIES, is_complete, train_tree  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str, run_id: str, smoke: bool, seeds, models) -> int:
    cfg = load_config(config_path)
    protocol = cfg.get("protocol", "grouped_holdout")
    root_key = "smoke_output_root" if smoke else "output_root"
    out_root = Path(cfg[root_key])
    run_dir = ensure_dir(out_root / run_id)

    models = models or cfg.get("new_models") or NEW_FAMILIES
    seeds = seeds or cfg.get("seeds") or [42, 43, 44]
    ratios = (float(cfg.get("train_ratio", 0.7)),
              float(cfg.get("val_ratio", 0.15)),
              float(cfg.get("test_ratio", 0.15)))

    resolved = dict(cfg)
    resolved.update({"run_id": run_id, "protocol": protocol, "new_models": models,
                     "seeds": seeds, "smoke": smoke})
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    print(f"[ext] run_id={run_id} protocol={protocol} models={models} seeds={seeds}")
    print(f"[ext] out_dir={run_dir}")

    ds_cache, audits, failures, done = {}, {}, [], 0

    def get_ds(seed):
        if seed not in ds_cache:
            ds = build_benchmark_dataset(cfg["data_dir"], protocol=protocol,
                                         train_ratio=ratios[0], val_ratio=ratios[1],
                                         test_ratio=ratios[2], seed=seed)
            ds_cache[seed] = ds
            audits[seed] = split_audit(ds)
            if seed == seeds[0]:
                ds.manifest.to_csv(run_dir / "split_manifest.csv", index=False)
                save_json(audits[seed], run_dir / "split_audit.json")
        return ds_cache[seed]

    for family in models:
        for seed in seeds:
            key = f"{family}/seed{seed}"
            if is_complete(run_dir, family, seed):
                print(f"[skip] {key} already complete")
                continue
            try:
                ds = get_ds(seed)
                print(f"[train] {key} ...", flush=True)
                res = train_tree(family, ds, run_dir, cfg, seed)
                te = res["metrics"]["test"]["overall"]
                print(f"[done] {key} norm_overall_RMSE={te['norm_overall_RMSE']:.4f} "
                      f"mean_R2={te['mean_R2']:.4f}")
                done += 1
            except Exception as exc:  # noqa: BLE001
                failures.append({"combo": key, "error": repr(exc)})
                print(f"[FAIL] {key}: {exc}")
                traceback.print_exc()

    save_json({
        "run_id": run_id, "protocol": protocol, "smoke": smoke,
        "new_models": models, "seeds": seeds, "data_dir": cfg["data_dir"],
        "split_audit_by_seed": audits, "n_done": done,
        "n_failures": len(failures), "failures": failures,
    }, run_dir / "run_manifest.json")
    print(f"[ext] complete. done={done} failures={len(failures)}")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Error-metric extension runner (RF/XGB/CatBoost)")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    a = p.parse_args(argv)
    return run(a.config, a.run_id, a.smoke, a.seeds, a.models)


if __name__ == "__main__":
    raise SystemExit(main())
