"""Publication-style 2x3 summary figure for a battery surrogate model.

Loads a trained checkpoint, reconstructs the exact test split used at training
time, runs inference, inverse-transforms predictions back to physical units
(Voltage [V], Temperature [degC]) and renders a single 2x3 matplotlib figure.

The reference paper figure has a third column showing capacity degradation over
charge/discharge cycles.  Our datasets contain only per-curve voltage and
temperature time-series -- there is NO aging / cycle-capacity signal -- so we do
NOT fabricate one.  The capacity panels are replaced with honest error/RMSE
panels derived from the model's own predictions:

    (a) Observed voltage curves          (d) Voltage: observed vs predicted
    (b) Observed temperature curves       (e) Temperature: observed vs predicted
    (c) Normalized prediction error       (f) RMSE over time

Example
-------
python scripts/plot_summary_figure.py \
    --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC \
    --model rnn \
    --num-curves 80 \
    --device auto
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

from src.predict import (
    checkpoint_dir,
    figures_dir,
    load_scalers,
    predict_case,
)
from src.utils import ensure_dir

MODEL_CHOICES = ("mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp")

# Consistent colours across panels.
C_VOLT = "#1f77b4"   # blue  -> voltage
C_TEMP = "#d62728"   # red   -> temperature
C_OBS = "#2c3e50"    # dark  -> observed
C_PRED = "#e67e22"   # orange-> predicted


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render a 2x3 publication-style prediction summary figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-root", default="generate_training_data")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--case-id", required=True)
    p.add_argument("--model", default="rnn", choices=list(MODEL_CHOICES))
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint .pt path. Default: "
        "<outputs-dir>/checkpoints/<case-id>/<model>/best_model.pt",
    )
    p.add_argument("--num-curves", type=int, default=80,
                   help="Max number of raw test curves to overlay in panels (a)/(b).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for the random subset of curves shown in (a)/(b).")
    p.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'.")
    p.add_argument(
        "--save-path",
        default=None,
        help="PNG output path. Default: "
        "<outputs-dir>/figures/<case-id>/<model>/summary_figure.png",
    )
    p.add_argument("--dpi", type=int, default=300)
    return p


# --------------------------------------------------------------------------- #
# Per-panel renderers
# --------------------------------------------------------------------------- #
def _plot_observed_curves(ax, time_s, curves, color, title, ylabel):
    """Overlay a sample of raw observed curves with thin translucent lines."""
    for row in range(curves.shape[0]):
        ax.plot(time_s, curves[row], color=color, linewidth=0.7, alpha=0.35)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def _plot_normalized_error(ax, time_s, abs_err_v, abs_err_t):
    """Mean absolute error per timestep, each channel normalized by its own max.

    Voltage and temperature have very different magnitudes/units, so normalizing
    each curve to [0, 1] lets both share a single axis for a clean comparison of
    *where in time* the model struggles most.
    """
    mae_v = abs_err_v.mean(axis=0)
    mae_t = abs_err_t.mean(axis=0)

    def _norm(x):
        peak = float(np.max(x))
        return x / peak if peak > 0 else x

    ax.plot(time_s, _norm(mae_v), color=C_VOLT, linewidth=1.8,
            label="Voltage MAE (norm.)")
    ax.plot(time_s, _norm(mae_t), color=C_TEMP, linewidth=1.8,
            label="Temperature MAE (norm.)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Mean Absolute Error")
    ax.set_title("(c) Normalized Prediction Error")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def _plot_percentile_band(ax, time_s, true_curves, pred_curves, title, ylabel,
                          unit):
    """Observed vs predicted 5th / mean / 95th percentile envelopes."""
    obs_lo = np.percentile(true_curves, 5, axis=0)
    obs_mid = true_curves.mean(axis=0)
    obs_hi = np.percentile(true_curves, 95, axis=0)
    prd_lo = np.percentile(pred_curves, 5, axis=0)
    prd_mid = pred_curves.mean(axis=0)
    prd_hi = np.percentile(pred_curves, 95, axis=0)

    # Shaded observed envelope for visual anchoring.
    ax.fill_between(time_s, obs_lo, obs_hi, color=C_OBS, alpha=0.12)

    ax.plot(time_s, obs_mid, color=C_OBS, linewidth=1.9, label="Observed mean")
    ax.plot(time_s, obs_lo, color=C_OBS, linewidth=1.0, alpha=0.8,
            label="Observed 5th/95th")
    ax.plot(time_s, obs_hi, color=C_OBS, linewidth=1.0, alpha=0.8)

    ax.plot(time_s, prd_mid, color=C_PRED, linewidth=1.9, linestyle="--",
            label="Predicted mean")
    ax.plot(time_s, prd_lo, color=C_PRED, linewidth=1.0, linestyle="--",
            alpha=0.8, label="Predicted 5th/95th")
    ax.plot(time_s, prd_hi, color=C_PRED, linewidth=1.0, linestyle="--",
            alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{ylabel} ({unit})")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def _plot_rmse_over_time(ax, time_s, abs_err_v, abs_err_t):
    """Per-timestep RMSE for V (left axis) and T (right axis)."""
    rmse_v = np.sqrt((abs_err_v ** 2).mean(axis=0))
    rmse_t = np.sqrt((abs_err_t ** 2).mean(axis=0))

    line_v, = ax.plot(time_s, rmse_v, color=C_VOLT, linewidth=1.8,
                      label="Voltage RMSE")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage RMSE (V)", color=C_VOLT)
    ax.tick_params(axis="y", labelcolor=C_VOLT)
    ax.grid(True, alpha=0.25)

    ax_r = ax.twinx()
    line_t, = ax_r.plot(time_s, rmse_t, color=C_TEMP, linewidth=1.8,
                        linestyle="--", label="Temperature RMSE")
    ax_r.set_ylabel("Temperature RMSE (°C)", color=C_TEMP)
    ax_r.tick_params(axis="y", labelcolor=C_TEMP)

    ax.set_title("(f) RMSE Over Time")
    ax.legend([line_v, line_t], [line_v.get_label(), line_t.get_label()],
              fontsize=8, loc="best")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = build_parser().parse_args()
    warnings: list[str] = []

    # --- Resolve & validate the checkpoint path (clear error if missing) ---
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint).expanduser().resolve()
    else:
        ckpt_path = (
            checkpoint_dir(args.outputs_dir, args.case_id, args.model)
            / "best_model.pt"
        ).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Train the model first or pass --checkpoint explicitly."
        )

    # --- Validate scalers exist on disk (clear error if missing) ---
    # predict_case re-fits identical train-only scalers from the deterministic
    # split, but the project contract is to persist them; verify their presence
    # so a stale/incomplete run fails loudly rather than silently.
    load_scalers(args.outputs_dir, args.case_id, args.model)

    # --- Inference on the test split (same split logic as training) ---
    # Pass an absolute checkpoint path as checkpoint_name: predict_case builds
    # `checkpoint_dir(...) / checkpoint_name`, and Path division with an absolute
    # right-hand side returns that absolute path unchanged.
    pred = predict_case(
        data_root=args.data_root,
        case_id=args.case_id,
        model_name=args.model,
        outputs_dir=args.outputs_dir,
        checkpoint_name=str(ckpt_path),
        split="test",
        device=args.device,
    )

    time_s = np.asarray(pred["time_s"], dtype=np.float64)
    v_true, v_pred = pred["v_true"], pred["v_pred"]
    t_true, t_pred = pred["t_true"], pred["t_pred"]
    n_test = v_true.shape[0]
    if n_test == 0:
        print("Test split is empty; nothing to plot.")
        return 1

    abs_err_v = np.abs(v_pred - v_true)
    abs_err_t = np.abs(t_pred - t_true)

    # --- Reproducible subset of raw curves for panels (a)/(b) ---
    rng = np.random.default_rng(args.seed)
    k = min(args.num_curves, n_test)
    if args.num_curves > n_test:
        warnings.append(
            f"--num-curves={args.num_curves} exceeds test size ({n_test}); "
            f"showing all {n_test} test curves."
        )
    chosen = rng.choice(n_test, size=k, replace=False)

    # --- Figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    (ax_a, ax_b, ax_c), (ax_d, ax_e, ax_f) = axes

    _plot_observed_curves(ax_a, time_s, v_true[chosen], C_VOLT,
                          "(a) Observed Voltage Curves", "Voltage (V)")
    _plot_observed_curves(ax_b, time_s, t_true[chosen], C_TEMP,
                          "(b) Observed Temperature Curves", "Temperature (°C)")
    _plot_normalized_error(ax_c, time_s, abs_err_v, abs_err_t)
    _plot_percentile_band(ax_d, time_s, v_true, v_pred,
                          "(d) Voltage: Observed vs Predicted", "Voltage", "V")
    _plot_percentile_band(ax_e, time_s, t_true, t_pred,
                          "(e) Temperature: Observed vs Predicted",
                          "Temperature", "°C")
    _plot_rmse_over_time(ax_f, time_s, abs_err_v, abs_err_t)

    fig.suptitle(
        f"{args.case_id} / {args.model}: "
        f"Voltage and Temperature Prediction Summary",
        fontsize=14, fontweight="bold",
    )

    # --- Save PNG + PDF ---
    if args.save_path is not None:
        png_path = Path(args.save_path).expanduser()
    else:
        png_path = (
            figures_dir(args.outputs_dir, args.case_id, args.model)
            / "summary_figure.png"
        )
    ensure_dir(png_path.parent)
    pdf_path = png_path.with_suffix(".pdf")

    fig.savefig(png_path, dpi=args.dpi)
    fig.savefig(pdf_path)  # vector PDF; dpi only affects rasterized elements
    plt.close(fig)

    print(f"Rendered 2x3 summary from {n_test} test curves "
          f"(overlaid {k} in panels a/b).")
    print(f"Saved PNG: {png_path.resolve()}")
    print(f"Saved PDF: {pdf_path.resolve()}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
