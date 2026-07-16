"""Plot true-vs-predicted voltage and temperature for random test samples.

Recreates the exact test split (same seed/ratios from run_config.json), runs
the trained model and saves one voltage plot and one temperature plot per
sampled curve under ``outputs/figures/<case_id>/<model_name>/``.

Example
-------
python scripts/plot_predictions.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model mlp --num-samples 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; no display required
import matplotlib.pyplot as plt
import numpy as np

from src.predict import figures_dir, predict_case
from src.utils import ensure_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot predicted vs true curves on test samples.")
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--case-id", required=True)
    p.add_argument("--model", required=True, choices=["mlp", "rnn", "lstm", "bilstm"])
    p.add_argument("--checkpoint", default="best_model.pt")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p


def _plot_curve(time_s, true_curve, pred_curve, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_s, true_curve, label="true", linewidth=2)
    ax.plot(time_s, pred_curve, label="predicted", linewidth=2, linestyle="--")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()

    pred = predict_case(
        data_root=args.data_root,
        case_id=args.case_id,
        model_name=args.model,
        outputs_dir=args.outputs_dir,
        checkpoint_name=args.checkpoint,
        split="test",
        device=args.device,
    )

    time_s = pred["time_s"]
    n_test = pred["v_true"].shape[0]
    if n_test == 0:
        print("Test split is empty; nothing to plot.")
        return 1

    # Pick random test rows reproducibly (independent of training seed usage).
    rng = np.random.default_rng(args.seed)
    k = min(args.num_samples, n_test)
    chosen = rng.choice(n_test, size=k, replace=False)

    fig_dir = ensure_dir(figures_dir(args.outputs_dir, args.case_id, args.model))
    print(f"Plotting {k} test sample(s) -> {fig_dir}")

    for row in chosen:
        sid = str(pred["sample_ids"][row])
        _plot_curve(
            time_s, pred["v_true"][row], pred["v_pred"][row],
            f"{args.case_id} | {args.model} | {sid} | voltage",
            "voltage [V]",
            fig_dir / f"{sid}_voltage.png",
        )
        _plot_curve(
            time_s, pred["t_true"][row], pred["t_pred"][row],
            f"{args.case_id} | {args.model} | {sid} | temperature",
            "temperature [C]",
            fig_dir / f"{sid}_temperature.png",
        )
        print(f"  {sid}: saved voltage + temperature plots")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
