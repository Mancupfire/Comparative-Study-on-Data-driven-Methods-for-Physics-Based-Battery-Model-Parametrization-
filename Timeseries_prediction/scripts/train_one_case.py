"""Train a single model on a single case, then evaluate it on the test split.

Example
-------
python scripts/train_one_case.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model mlp --epochs 300 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import evaluate_case
from src.train import train_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one model on one case.")
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--case-id", required=True)
    p.add_argument(
        "--model",
        required=True,
        choices=["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"],
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
    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--test-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--mc-samples",
        type=int,
        default=30,
        help="MC-Dropout passes for Bayesian models at evaluation (ignored otherwise).",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    train_model(
        data_root=args.data_root,
        case_id=args.case_id,
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
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )

    # Evaluate the best checkpoint on the held-out test split.
    evaluate_case(
        data_root=args.data_root,
        case_id=args.case_id,
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
