"""CLI orchestration for the Batch-4 error-metric benchmark.

Runs every requested (family, seed) combination under one RUN_ID, with safe
resume (already-completed combinations are skipped) and deterministic,
identical splits/scalers across all models.

Example
-------
python -m src.error_metric_benchmark.run \
    --config configs/batch_4/error_metric_benchmark_full.yaml \
    --run-id batch4_errmetric_full_20260622 \
    --protocol grouped_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List

import joblib
import yaml

from src.utils import ensure_dir, resolve_device, save_json

from . import models as M
from .data import build_benchmark_dataset, split_audit
from .trainer import is_complete, train_one


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_dataset_artifacts(run_dir: Path, ds) -> Dict:
    ds.manifest.to_csv(run_dir / "split_manifest.csv", index=False)
    sdir = ensure_dir(run_dir / "scalers")
    joblib.dump(ds.x_scaler, sdir / "x_scaler.joblib")
    joblib.dump(ds.y_scaler, sdir / "y_scaler.joblib")
    audit = split_audit(ds)
    save_json(audit, run_dir / "split_audit.json")
    return audit


def run_benchmark(config_path: str, run_id: str, protocol: str | None,
                  smoke: bool, models: List[str] | None,
                  seeds: List[int] | None) -> int:
    cfg = load_config(config_path)
    protocol = protocol or cfg.get("protocol", "grouped_holdout")
    root_key = "smoke_output_root" if smoke else "output_root"
    out_root = Path(cfg[root_key])
    run_dir = ensure_dir(out_root / run_id)

    models = models or cfg.get("models") or M.FAMILY_ORDER
    seeds = seeds or cfg.get("seeds") or [int(cfg.get("seed", 42))]
    device = resolve_device(cfg.get("device", "auto"))
    ratios = (float(cfg.get("train_ratio", 0.7)),
              float(cfg.get("val_ratio", 0.15)),
              float(cfg.get("test_ratio", 0.15)))

    resolved = dict(cfg)
    resolved.update({"run_id": run_id, "protocol": protocol, "models": models,
                     "seeds": seeds, "resolved_device": device, "smoke": smoke})
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)

    print(f"[run] run_id={run_id} protocol={protocol} device={device}")
    print(f"[run] models={models}")
    print(f"[run] seeds={seeds}  out_dir={run_dir}")

    # Build the dataset once per seed (split is seed-dependent); cache by seed.
    ds_cache = {}
    audits = {}

    def get_ds(seed: int):
        if seed not in ds_cache:
            ds = build_benchmark_dataset(cfg["data_dir"], protocol=protocol,
                                         train_ratio=ratios[0], val_ratio=ratios[1],
                                         test_ratio=ratios[2], seed=seed)
            ds_cache[seed] = ds
            audits[seed] = split_audit(ds)
            # Persist artifacts for the primary (first) seed.
            if seed == seeds[0]:
                _write_dataset_artifacts(run_dir, ds)
        return ds_cache[seed]

    results = {}
    failures = []
    for family in models:
        if family not in M.ALL_FAMILIES:
            print(f"[warn] unknown family '{family}', skipping")
            continue
        for seed in seeds:
            key = f"{family}/seed{seed}"
            if is_complete(run_dir, family, seed):
                print(f"[skip] {key} already complete")
                continue
            try:
                ds = get_ds(seed)
                print(f"[train] {key} ...", flush=True)
                res = train_one(family, ds, run_dir, cfg, seed, device)
                te = res["metrics"]["test"]["overall"]
                print(f"[done] {key} norm_overall_RMSE={te['norm_overall_RMSE']:.4f} "
                      f"mean_R2={te['mean_R2']:.4f}")
                results[key] = "ok"
            except Exception as exc:  # noqa: BLE001
                failures.append({"combo": key, "error": repr(exc)})
                print(f"[FAIL] {key}: {exc}")
                traceback.print_exc()

    manifest = {
        "run_id": run_id, "protocol": protocol, "smoke": smoke,
        "models": models, "seeds": seeds, "device": device,
        "config_path": config_path, "data_dir": cfg["data_dir"],
        "split_audit_by_seed": audits,
        "n_failures": len(failures), "failures": failures,
    }
    save_json(manifest, run_dir / "run_manifest.json")
    print(f"[run] complete. failures={len(failures)}")
    return 1 if failures else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Batch-4 error-metric benchmark runner")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--protocol", choices=["grouped_holdout", "legacy_reproduction"],
                   default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    a = p.parse_args(argv)
    return run_benchmark(a.config, a.run_id, a.protocol, a.smoke, a.models, a.seeds)


if __name__ == "__main__":
    sys.exit(main())
