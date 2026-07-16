"""2x3 prediction-summary figure for a SHARED model on one selected case.

A shared model is trained across all cases at once.  This script loads that one
shared checkpoint, reconstructs the global (all-cases) test split exactly as used
at training time, keeps only the test curves belonging to ``--case-id`` and
renders the same publication-style 2x3 summary as ``plot_summary_figure.py``.

Saved to::

    outputs/figures/shared/<model_name>/<case_id>/summary_figure.{png,pdf}

Example
-------
python scripts/plot_shared_summary_figure.py \
    --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC \
    --model shared_rnn \
    --num-curves 80 --device auto
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent))
sys.path.insert(0, str(_SCRIPTS_DIR))

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np

# Reuse the per-panel renderers and colours from the per-case summary script.
import plot_summary_figure as psf

from src.shared_data import load_shared_scalers, shared_checkpoint_dir, shared_figures_dir
from src.shared_evaluate import predict_shared_curves
from src.utils import ensure_dir

MODEL_CHOICES = ("shared_mlp", "shared_rnn", "shared_lstm", "shared_bilstm",
                 "shared_cnn", "shared_cnn_bilstm", "shared_bayesian_mlp")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a 2x3 prediction summary for a shared model on one case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--case-id", required=True)
    p.add_argument("--model", default="shared_rnn", choices=list(MODEL_CHOICES))
    p.add_argument("--num-curves", type=int, default=80,
                   help="Max number of raw test curves to overlay in panels (a)/(b).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the random subset of curves shown in (a)/(b).")
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'.")
    p.add_argument("--save-path", default=None,
                   help="PNG output path. Default: "
                        "<outputs-dir>/figures/shared/<model>/<case-id>/summary_figure.png")
    p.add_argument("--dpi", type=int, default=300)
    return p


def main() -> int:
    args = build_parser().parse_args()
    warnings: list[str] = []

    ckpt_path = (shared_checkpoint_dir(args.outputs_dir, args.model) / "best_model.pt").resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nTrain the shared model first."
        )
    # Verify scalers exist (loud failure on an incomplete run).
    load_shared_scalers(args.outputs_dir, args.model)

    # Reconstruct the GLOBAL test split (all cases) then keep this case only --
    # filtering after the split avoids any train/test leakage for the case.
    records = predict_shared_curves(
        data_root=args.data_root,
        model_name=args.model,
        outputs_dir=args.outputs_dir,
        checkpoint_name="best_model.pt",
        split="test",
        device=args.device,
    )
    case_records = [r for r in records if r["case_id"] == args.case_id]
    if not case_records:
        raise ValueError(
            f"No test curves for case '{args.case_id}'. "
            f"Available cases in test split: {sorted({r['case_id'] for r in records})}"
        )

    time_s = np.asarray(case_records[0]["time_s"], dtype=np.float64)
    v_true = np.stack([r["v_true"] for r in case_records], axis=0)
    v_pred = np.stack([r["v_pred"] for r in case_records], axis=0)
    t_true = np.stack([r["t_true"] for r in case_records], axis=0)
    t_pred = np.stack([r["t_pred"] for r in case_records], axis=0)
    n_test = v_true.shape[0]

    abs_err_v = np.abs(v_pred - v_true)
    abs_err_t = np.abs(t_pred - t_true)

    rng = np.random.default_rng(args.seed)
    k = min(args.num_curves, n_test)
    if args.num_curves > n_test:
        warnings.append(
            f"--num-curves={args.num_curves} exceeds test size for this case "
            f"({n_test}); showing all {n_test} curves."
        )
    chosen = rng.choice(n_test, size=k, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    psf._plot_observed_curves(ax_a, time_s, v_true[chosen], psf.C_VOLT,
                              "(a) Observed Voltage Curves", "Voltage (V)")
    psf._plot_observed_curves(ax_b, time_s, t_true[chosen], psf.C_TEMP,
                              "(b) Observed Temperature Curves", "Temperature (°C)")
    psf._plot_normalized_error(ax_c, time_s, abs_err_v, abs_err_t)
    psf._plot_percentile_band(ax_d, time_s, v_true, v_pred,
                              "(d) Voltage: Observed vs Predicted", "Voltage", "V")
    psf._plot_percentile_band(ax_e, time_s, t_true, t_pred,
                              "(e) Temperature: Observed vs Predicted", "Temperature", "°C")
    psf._plot_rmse_over_time(ax_f, time_s, abs_err_v, abs_err_t)

    fig.suptitle(
        f"shared/{args.model} on {args.case_id}: "
        f"Voltage and Temperature Prediction Summary",
        fontsize=14, fontweight="bold",
    )

    if args.save_path is not None:
        png_path = Path(args.save_path).expanduser()
    else:
        png_path = shared_figures_dir(args.outputs_dir, args.model, args.case_id) / "summary_figure.png"
    ensure_dir(png_path.parent)
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Rendered 2x3 shared-model summary from {n_test} test curves for "
          f"{args.case_id} (overlaid {k} in panels a/b).")
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
