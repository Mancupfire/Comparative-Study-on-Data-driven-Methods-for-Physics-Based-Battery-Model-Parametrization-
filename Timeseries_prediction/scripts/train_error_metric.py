"""Task B entrypoint — train the error-metric surrogate(s).

Builds the joined dataset (error_metrics + manifest + parameter_sets), splits by
sample_id (70/15/15, seed 42), fits scalers on train only, then trains the
ExtraTrees baseline and/or the MLP main model. Writes everything under an
isolated Task-B namespace.

Example (full):
  python scripts/train_error_metric.py \
    --data-dir data/Data_Batch_2 \
    --output-root outputs/Data_Batch_2/error_metric/run1 \
    --models extratrees mlp --epochs 200

Example (smoke):
  python scripts/train_error_metric.py \
    --data-dir data/Data_Batch_2 \
    --output-root outputs_smoke/Data_Batch_2/error_metric/smoke1 \
    --models extratrees mlp --epochs 5 --n-estimators 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.error_metric_data import build_dataset
from src.error_metric_train import run_task_b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Task B error-metric surrogate.")
    p.add_argument("--data-dir", default="data/Data_Batch_2",
                   help="Dir holding error_metrics.csv, sequence_manifest.csv, parameter_sets.csv")
    p.add_argument("--output-root", required=True)
    p.add_argument("--dataset-name", default="Data_Batch_2",
                   help="Dataset label recorded in the run manifest (e.g. Data_Batch_3).")
    p.add_argument("--models", nargs="+", default=["extratrees", "mlp"],
                   choices=["extratrees", "mlp"])
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    # MLP
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--device", default="auto")
    # ExtraTrees
    p.add_argument("--n-estimators", type=int, default=300)
    return p


def main() -> int:
    args = build_parser().parse_args()
    config = {
        "dataset_name": args.dataset_name, "task": "error_metric",
        "data_dir": args.data_dir,
        "train_ratio": args.train_ratio, "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio, "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "dropout": args.dropout,
        "patience": args.patience, "device": args.device,
        "n_estimators": args.n_estimators,
    }
    print(f"[task-b] building dataset from {args.data_dir} ...")
    ds = build_dataset(args.data_dir, args.train_ratio, args.val_ratio,
                       args.test_ratio, args.seed)
    print(f"[task-b] features={ds.n_features} targets={ds.target_names} "
          f"rows train/val/test={ds.X_train.shape[0]}/{ds.X_val.shape[0]}/{ds.X_test.shape[0]}")
    manifest = run_task_b(ds, args.output_root, config, args.models)
    for m, r in manifest["results"].items():
        t = r["metrics"]["test"]
        print(f"[task-b] {m}: test overall RMSE={t['overall']['RMSE']:.4f} "
              f"R2={t['overall']['R2']:.4f} reload_ok={r['reload_equivalence_ok']}")
    print(f"[task-b] leakage_free={manifest['leakage_check']['leakage_free']}")
    print(f"[task-b] DONE -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
