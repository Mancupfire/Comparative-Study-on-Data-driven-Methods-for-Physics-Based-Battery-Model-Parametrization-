"""Final time-series model set (7 families) for the filtered protocol.

Six families are reused unchanged from ``src.models`` (RNN, LSTM, BiLSTM, CNN,
CNN-BiLSTM, Bayesian MLP).  The seventh, **ANN**, is added here as a *shallow
feed-forward trajectory baseline*.  It is deliberately NOT identical to the
existing ``MLP``:

    MLP  (src.models.MLP) : 3 hidden layers, widths h -> 2h -> 4h (h=256),
                            LayerNorm, ~ wide & deep.
    ANN  (this module)    : single hidden layer of width ``ann_hidden_dim``
                            (default 128), ReLU + dropout, then a linear head to
                            ``2 * t_last``.  A classic shallow ANN baseline.

Like the MLP it is a *point* model: input ``[B, n_parameters]`` ->
``[B, 2*t_last]`` (concat of the voltage and temperature curves).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models import (
    SEQUENCE_MODELS, build_model as _build_seq_or_point,
    make_model_kwargs as _make_kwargs_existing,
)

# Final family set for the filtered time-series protocol.
FINAL_TS_MODELS = ["ann", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"]
POINT_MODELS = {"ann", "bayesian_mlp"}

DISPLAY_NAMES: Dict[str, str] = {
    "ann": "ANN",
    "rnn": "RNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "cnn": "CNN",
    "cnn_bilstm": "CNN-BiLSTM",
    "bayesian_mlp": "Bayesian MLP",
}


class ANN(nn.Module):
    """Shallow single-hidden-layer feed-forward trajectory baseline.

    Architecture::

        Linear(n_parameters -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 2*t_last)

    Distinct from :class:`src.models.MLP` (which has three progressively wider
    hidden layers with normalization).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128,
                 dropout: float = 0.1, **_: object):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_model_kwargs(model_name: str, n_parameters: int, t_last: int,
                      hidden_dim: int, num_layers: int, dropout: float,
                      ann_hidden_dim: int = 128) -> Dict[str, object]:
    name = model_name.lower()
    if name == "ann":
        return {
            "input_dim": n_parameters,
            "output_dim": 2 * t_last,
            "hidden_dim": ann_hidden_dim,
            "dropout": dropout,
        }
    return _make_kwargs_existing(
        name, n_parameters, t_last, hidden_dim, num_layers, dropout
    )


def build_model(model_name: str, model_kwargs: Dict[str, object]) -> nn.Module:
    name = model_name.lower()
    if name == "ann":
        return ANN(**model_kwargs)
    return _build_seq_or_point(name, model_kwargs)


def is_sequence_model(model_name: str) -> bool:
    return model_name.lower() in SEQUENCE_MODELS


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def split_prediction(model_name: str, output: torch.Tensor, t_last: int):
    """Split raw output into (voltage, temperature) ``[B, T]`` tensors."""
    if model_name.lower() in POINT_MODELS:
        return output[:, :t_last], output[:, t_last:]
    return output[..., 0], output[..., 1]
