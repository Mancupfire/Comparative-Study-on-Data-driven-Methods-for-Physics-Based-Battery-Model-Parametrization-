"""Train every requested model on every discovered case, then evaluate each.

Example
-------
python scripts/train_all_cases.py --data-root generate_training_data \
    --models mlp rnn lstm bilstm --epochs 300 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import discover_cases
from src.evaluate import evaluate_case
from src.train import train_model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train all models on all cases.")
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--cases", nargs="+", default=None,
                   help="Subset of case ids; default = all discovered cases.")
    p.add_argument("--models", nargs="+", default=["mlp", "rnn", "lstm", "bilstm"],
                   choices=["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm",
                            "bayesian_mlp"])
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
    cases = args.cases or discover_cases(args.data_root)
    print(f"Training {args.models} on {len(cases)} case(s): {cases}\n")

    failures = []
    for case_id in cases:
        for model_name in args.models:
            banner = f"### {case_id} / {model_name} ###"
            print("\n" + "#" * len(banner) + f"\n{banner}\n" + "#" * len(banner))
            try:
                train_model(
                    data_root=args.data_root,
                    case_id=case_id,
                    model_name=model_name,
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
                evaluate_case(
                    data_root=args.data_root,
                    case_id=case_id,
                    model_name=model_name,
                    outputs_dir=args.outputs_dir,
                    checkpoint_name="best_model.pt",
                    split="test",
                    device=args.device,
                    mc_samples=args.mc_samples,
                )
            except Exception as exc:  # noqa: BLE001 - keep going across the grid
                failures.append((case_id, model_name, str(exc)))
                print(f"!! FAILED {case_id}/{model_name}: {exc}")
                traceback.print_exc()

    print("\n" + "=" * 60)
    if failures:
        print(f"Completed with {len(failures)} failure(s):")
        for case_id, model_name, err in failures:
            print(f"  - {case_id}/{model_name}: {err}")
        return 1
    print("All training runs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
