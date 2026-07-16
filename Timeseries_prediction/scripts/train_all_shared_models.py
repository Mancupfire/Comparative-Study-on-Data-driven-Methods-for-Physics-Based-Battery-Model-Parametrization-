"""Train every requested SHARED model across all cases, then evaluate each.

Point-wise (``shared_mlp``) and sequence (``shared_rnn`` / ``shared_lstm`` /
``shared_bilstm``) models want very different batch sizes, so the per-model batch
size is chosen automatically: ``--mlp-batch-size`` for ``shared_mlp`` and
``--sequence-batch-size`` for the recurrent models (unless ``--batch-size`` is
given explicitly, which then overrides both).

Example
-------
python scripts/train_all_shared_models.py \
    --data-root generate_training_data \
    --models shared_mlp shared_rnn shared_lstm shared_bilstm \
    --epochs 300 --lr 1e-3 --seed 42 --device auto
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared_data import SEQUENCE_MODELS
from src.shared_evaluate import evaluate_shared_model
from src.shared_train import train_shared_model


def _max_points(value: str):
    if value is None or value.lower() in {"none", "all"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train all shared models across all cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument(
        "--models",
        nargs="+",
        default=["shared_mlp", "shared_rnn", "shared_lstm", "shared_bilstm"],
        choices=["shared_mlp", "shared_rnn", "shared_lstm", "shared_bilstm",
                 "shared_cnn", "shared_cnn_bilstm", "shared_bayesian_mlp"],
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override both per-mode batch sizes for every model.",
    )
    p.add_argument("--mlp-batch-size", type=int, default=8192)
    p.add_argument("--sequence-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-temp", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-points-per-curve", type=_max_points, default=None)
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="MC-Dropout passes for Bayesian shared models at evaluation.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    print(f"Training shared models {args.models}\n")

    failures = []
    for model_name in args.models:
        if args.batch_size is not None:
            batch_size = args.batch_size
        elif model_name in SEQUENCE_MODELS:
            batch_size = args.sequence_batch_size
        else:
            batch_size = args.mlp_batch_size

        banner = f"### shared/{model_name} (batch_size={batch_size}) ###"
        print("\n" + "#" * len(banner) + f"\n{banner}\n" + "#" * len(banner))
        try:
            train_shared_model(
                data_root=args.data_root,
                model_name=model_name,
                outputs_dir=args.outputs_dir,
                epochs=args.epochs,
                batch_size=batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                lambda_temp=args.lambda_temp,
                patience=args.patience,
                seed=args.seed,
                device=args.device,
                max_points_per_curve=args.max_points_per_curve,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                num_workers=args.num_workers,
            )
            evaluate_shared_model(
                data_root=args.data_root,
                model_name=model_name,
                outputs_dir=args.outputs_dir,
                checkpoint_name="best_model.pt",
                split="test",
                device=args.device,
                mc_samples=args.mc_samples,
            )
        except Exception as exc:  # noqa: BLE001 - keep going across the grid
            failures.append((model_name, str(exc)))
            print(f"!! FAILED shared/{model_name}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for model_name, err in failures:
            print(f"  - shared/{model_name}: {err}")
        return 1
    print("All shared training runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
