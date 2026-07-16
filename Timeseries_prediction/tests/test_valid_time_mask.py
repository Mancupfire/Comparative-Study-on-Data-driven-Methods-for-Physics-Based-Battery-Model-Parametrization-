"""Valid-time masking invariance tests.

Proves the defining property of the mask: the values of the *invalid*
(held / extrapolated) tail entries have NO effect on the loss or on any reported
metric (MAE, RMSE, R2, parity inputs, time-resolved curve RMSE, peak/end).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.final_filtered.masking import (  # noqa: E402
    compute_valid_mask, compute_masked_metrics, last_valid_index,
    masked_mse_torch,
)

RNG = np.random.default_rng(0)


def _make_case(n=40, t=160):
    time_s = np.linspace(0.0, 1590.0, t)
    # Each sample ends somewhere between 40% and 100% of the horizon.
    sim_end = RNG.uniform(0.4, 1.0, size=n) * time_s[-1]
    mask = compute_valid_mask(time_s, sim_end)
    v_true = RNG.normal(3.7, 0.2, size=(n, t))
    t_true = RNG.normal(30.0, 3.0, size=(n, t))
    v_pred = v_true + RNG.normal(0, 0.05, size=(n, t))
    t_pred = t_true + RNG.normal(0, 0.5, size=(n, t))
    return time_s, sim_end, mask, v_true, v_pred, t_true, t_pred


def _corrupt_tail(arr, mask):
    """Replace every invalid (masked-out) entry with wild garbage."""
    out = arr.copy()
    invalid = ~mask
    out[invalid] = RNG.uniform(-1e6, 1e6, size=int(invalid.sum()))
    return out


def test_mask_shapes_and_monotonic():
    time_s, sim_end, mask, *_ = _make_case()
    assert mask.shape == (40, 160)
    # t=0 always valid; mask is monotone non-increasing along time.
    assert mask[:, 0].all()
    diffs = np.diff(mask.astype(int), axis=1)
    assert (diffs <= 0).all(), "valid mask must not switch back on after the tail"


def test_metrics_invariant_to_invalid_tail():
    _, _, mask, v_true, v_pred, t_true, t_pred = _make_case()
    base = compute_masked_metrics(v_true, v_pred, t_true, t_pred, mask)

    # Corrupt BOTH the truth and the prediction in their invalid tails.
    v_true_c = _corrupt_tail(v_true, mask)
    v_pred_c = _corrupt_tail(v_pred, mask)
    t_true_c = _corrupt_tail(t_true, mask)
    t_pred_c = _corrupt_tail(t_pred, mask)
    corrupted = compute_masked_metrics(v_true_c, v_pred_c, t_true_c, t_pred_c, mask)

    for k in base:
        assert np.isclose(base[k], corrupted[k], rtol=0, atol=1e-9), (
            f"masked metric '{k}' changed when invalid tail was corrupted: "
            f"{base[k]} != {corrupted[k]}")


def test_masking_actually_changes_unmasked_result():
    """Sanity: without masking, corrupting the tail WOULD change the numbers."""
    _, _, mask, v_true, v_pred, t_true, t_pred = _make_case()
    full = np.ones_like(mask, dtype=bool)
    base = compute_masked_metrics(v_true, v_pred, t_true, t_pred, full)
    v_pred_c = _corrupt_tail(v_pred, mask)
    corrupted = compute_masked_metrics(v_true, v_pred_c, t_true, t_pred, full)
    assert not np.isclose(base["RMSE_V"], corrupted["RMSE_V"]), (
        "test is vacuous: tail corruption did not move the unmasked RMSE")


def test_masked_torch_loss_invariant_to_invalid_tail():
    import torch

    _, _, mask, _, v_pred, _, t_true = _make_case()
    m = torch.as_tensor(mask.astype(np.float32))
    pred = torch.as_tensor(v_pred.astype(np.float32))
    true = torch.as_tensor(t_true.astype(np.float32))  # arbitrary same-shape target

    base = masked_mse_torch(pred, true, m).item()
    pred_c = torch.as_tensor(_corrupt_tail(v_pred, mask).astype(np.float32))
    true_c = torch.as_tensor(_corrupt_tail(t_true, mask).astype(np.float32))
    corrupted = masked_mse_torch(pred_c, true_c, m).item()
    assert np.isclose(base, corrupted, atol=1e-6), (
        f"masked torch loss changed under tail corruption: {base} != {corrupted}")


def test_masked_torch_loss_zero_gradient_on_invalid_tail():
    """Gradient w.r.t. invalid-tail predictions must be exactly zero."""
    import torch

    _, _, mask, _, v_pred, _, _ = _make_case()
    m = torch.as_tensor(mask.astype(np.float32))
    pred = torch.as_tensor(v_pred.astype(np.float32), dtype=torch.float32).clone()
    pred.requires_grad_(True)
    true = torch.zeros_like(pred)
    loss = masked_mse_torch(pred, true, m)
    loss.backward()
    invalid = ~mask
    assert np.allclose(pred.grad.numpy()[invalid], 0.0), (
        "invalid-tail predictions received non-zero gradient")


def test_last_valid_index():
    time_s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    sim_end = np.array([4.0, 2.0, 0.5, 10.0])
    mask = compute_valid_mask(time_s, sim_end)
    lvi = last_valid_index(mask)
    assert lvi.tolist() == [4, 2, 0, 4]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL VALID-TIME-MASK TESTS PASSED")
