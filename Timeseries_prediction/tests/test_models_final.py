"""Final time-series model-set tests: ANN baseline is shallow and distinct.

The ANN must (a) exist as one of the seven final families, (b) be a *shallow*
feed-forward network (single hidden layer), and (c) NOT be the same architecture
as the existing 3-hidden-layer MLP.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.final_filtered import models as M  # noqa: E402
from src.models import MLP  # noqa: E402


def _count_linear_layers(module) -> int:
    return sum(1 for m in module.modules() if isinstance(m, torch.nn.Linear))


def test_final_family_set():
    assert M.FINAL_TS_MODELS == [
        "ann", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]
    assert len(M.FINAL_TS_MODELS) == 7


def test_ann_is_shallow_and_distinct_from_mlp():
    n_params, t_last = 12, 160
    ann_kwargs = M.make_model_kwargs("ann", n_params, t_last, 256, 2, 0.1,
                                     ann_hidden_dim=128)
    ann = M.build_model("ann", ann_kwargs)
    mlp = MLP(input_dim=n_params, output_dim=2 * t_last, hidden_dim=256)

    # ANN: exactly 2 Linear layers (1 hidden). MLP: 4 Linear layers (3 hidden).
    assert _count_linear_layers(ann) == 2, "ANN must be a single-hidden-layer net"
    assert _count_linear_layers(mlp) == 4
    assert _count_linear_layers(ann) != _count_linear_layers(mlp)

    ann_p = M.count_parameters(ann)
    mlp_p = M.count_parameters(mlp)
    assert ann_p != mlp_p, "ANN and MLP must not share the same parameter count"
    print(f"ANN params={ann_p}  MLP params={mlp_p}")


def test_ann_forward_shape():
    n_params, t_last, b = 12, 160, 8
    ann = M.build_model("ann", M.make_model_kwargs("ann", n_params, t_last,
                                                   256, 2, 0.1))
    out = ann(torch.zeros(b, n_params))
    assert out.shape == (b, 2 * t_last)
    v, t = M.split_prediction("ann", out, t_last)
    assert v.shape == (b, t_last) and t.shape == (b, t_last)


def test_sequence_model_shapes():
    n_params, t_last, b = 12, 160, 4
    for name in ("rnn", "lstm", "bilstm", "cnn", "cnn_bilstm"):
        kw = M.make_model_kwargs(name, n_params, t_last, 32, 1, 0.1)
        model = M.build_model(name, kw)
        x = torch.zeros(b, t_last, n_params + 1)
        out = model(x)
        assert out.shape == (b, t_last, 2), f"{name} bad shape {out.shape}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL FINAL-MODEL TESTS PASSED")
