"""Train one SHARED model across all cases, then evaluate it on the test curves.

A shared model learns ``f([parameters, c_rate, ambient_temp_C, time_norm]) ->
[V(t), T(t)]`` for every case at once, so it copes with cases of different
sequence length via the normalized-time feature.

Examples
--------
python scripts/train_shared_model.py \
    --data-root generate_training_data --model shared_mlp \
    --epochs 300 --batch-size 8192 --lr 1e-3 --max-points-per-curve 80 --device auto

python scripts/train_shared_model.py \
    --data-root generate_training_data --model shared_rnn \
    --epochs 300 --batch-size 64 --lr 1e-3 --device auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared_evaluate import evaluate_shared_model
from src.shared_train import train_shared_model


def _max_points(value: str):
    """Parse ``--max-points-per-curve``: ``None``/``all`` -> None, else int."""
    if value is None or value.lower() in {"none", "all"}:
        return None
    return int(value)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train one shared model across all cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument(
        "--model",
        required=True,
        choices=["shared_mlp", "shared_rnn", "shared_lstm", "shared_bilstm",
                 "shared_cnn", "shared_cnn_bilstm", "shared_bayesian_mlp"],
    )
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda-temp", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--max-points-per-curve",
        type=_max_points,
        default=None,
        help="Point models only: cap timesteps per curve (int) or 'none' for all.",
    )
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

    train_shared_model(
        data_root=args.data_root,
        model_name=args.model,
        outputs_dir=args.outputs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
        model_name=args.model,
        outputs_dir=args.outputs_dir,
        checkpoint_name="best_model.pt",
        split="test",
        device=args.device,
        mc_samples=args.mc_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
