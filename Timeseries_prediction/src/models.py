"""Model architectures and a factory for the four PyTorch models.

All models share a common contract so the training / prediction code can stay
model-agnostic:

* ``mlp``        : input ``[B, n_parameters]``          -> output ``[B, 2*t_last]``
* ``rnn``        : input ``[B, t_last, n_parameters+1]`` -> output ``[B, t_last, 2]``
* ``lstm``       : same shapes as ``rnn``
* ``bilstm``     : same shapes as ``rnn``
* ``cnn``        : same shapes as ``rnn`` (Conv1d temporal model)
* ``cnn_bilstm`` : same shapes as ``rnn`` (Conv1d front-end + BiLSTM)
* ``bayesian_mlp``: same shapes as ``mlp`` (MC-Dropout uncertainty)

The per-step output channels are ordered ``[voltage, temperature]``; the MLP /
Bayesian-MLP output is ``concat([voltage_curve, temperature_curve])``.

The ``cnn`` / ``cnn_bilstm`` / ``bayesian_mlp`` models are additive options; the
original four models keep their exact previous behaviour.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

# Sequence models consume ``[B, T, F]`` and emit ``[B, T, 2]``.
SEQUENCE_MODELS = {"rnn", "lstm", "bilstm", "cnn", "cnn_bilstm"}
# Point models consume ``[B, n_parameters]`` and emit ``[B, 2*t_last]``.
POINT_MODELS = {"mlp", "bayesian_mlp"}
ALL_MODELS = POINT_MODELS | SEQUENCE_MODELS


class MLP(nn.Module):
    """Fully-connected network predicting both full curves in one shot.

    Architecture (widths derived from ``hidden_dim``)::

        Linear(in -> h) -> ReLU -> Norm -> Dropout
        Linear(h  -> 2h) -> ReLU -> Dropout
        Linear(2h -> 4h) -> ReLU
        Linear(4h -> out)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        norm: str = "layernorm",
        **_: object,
    ):
        super().__init__()
        if norm == "batchnorm":
            norm_layer: nn.Module = nn.BatchNorm1d(hidden_dim)
        elif norm == "layernorm":
            norm_layer = nn.LayerNorm(hidden_dim)
        else:
            norm_layer = nn.Identity()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            norm_layer,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RecurrentRegressor(nn.Module):
    """Shared wrapper around nn.RNN / nn.LSTM for per-step [V, T] regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        kind: str = "lstm",
        bidirectional: bool = False,
        output_dim: int = 2,
        **_: object,
    ):
        super().__init__()
        # PyTorch only applies recurrent dropout when num_layers > 1.
        rnn_dropout = dropout if num_layers > 1 else 0.0
        common = dict(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=bidirectional,
        )
        if kind == "rnn":
            self.rnn: nn.Module = nn.RNN(nonlinearity="tanh", **common)
        elif kind == "lstm":
            self.rnn = nn.LSTM(**common)
        else:
            raise ValueError(f"Unknown recurrent kind '{kind}'")

        head_in = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # [B, t_last, hidden*(2 if bi)]
        return self.head(out)         # [B, t_last, 2]


class RNNModel(_RecurrentRegressor):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="rnn", bidirectional=False, **kwargs)


class LSTMModel(_RecurrentRegressor):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="lstm", bidirectional=False, **kwargs)


class BiLSTMModel(_RecurrentRegressor):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="lstm", bidirectional=True, **kwargs)


class TimeSeriesCNN(nn.Module):
    """1-D temporal CNN for per-step ``[V, T]`` regression.

    Input is the same sequence tensor used by the recurrent models,
    ``[B, T, input_dim]``; Conv1d expects ``[B, channels, time]`` so the tensor
    is transposed in/out of the convolution stack.  Several Conv1d blocks learn
    local temporal patterns in the voltage / temperature curves before a
    ``kernel_size=1`` projection maps to the two output channels.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        output_dim: int = 2,
        **_: object,
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # 1x1 conv == per-timestep linear projection to [V, T].
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # [B, T, F] -> [B, F, T]
        h = self.body(x)               # [B, hidden, T]
        out = self.head(h)             # [B, output_dim, T]
        return out.transpose(1, 2)     # [B, T, output_dim]


class CNNBiLSTM(nn.Module):
    """Conv1d temporal feature extractor followed by a BiLSTM.

    The CNN captures local time features; the BiLSTM captures long-range
    temporal dependencies.  Shapes match the recurrent models:
    ``[B, T, input_dim]`` -> ``[B, T, 2]``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 2,
        **_: object,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # PyTorch only applies recurrent dropout when num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # [B, T, F] -> [B, F, T]
        h = self.cnn(x)                # [B, hidden, T]
        h = h.transpose(1, 2)          # [B, T, hidden]
        out, _ = self.lstm(h)          # [B, T, 2*hidden]
        return self.head(out)          # [B, T, output_dim]


class BayesianMLP(nn.Module):
    """Approximate Bayesian MLP via Monte-Carlo Dropout.

    The architecture mirrors :class:`MLP` (predicting both full curves at once)
    but places a dropout layer after every hidden activation and uses no
    normalization layer, so dropout can be kept *active at inference time* to
    draw stochastic forward passes.  Training itself is ordinary MSE; the
    Bayesian behaviour comes entirely from MC-Dropout sampling at evaluation
    time (see :func:`src.predict.predict_mc_dropout`).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        **_: object,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(model_name: str, model_kwargs: Dict[str, object]) -> nn.Module:
    """Instantiate a model from its name and a kwargs dict.

    ``model_kwargs`` is stored verbatim in the checkpoint, so the same call can
    rebuild the architecture at inference time.
    """
    name = model_name.lower()
    if name == "mlp":
        return MLP(**model_kwargs)
    if name == "rnn":
        return RNNModel(**model_kwargs)
    if name == "lstm":
        return LSTMModel(**model_kwargs)
    if name == "bilstm":
        return BiLSTMModel(**model_kwargs)
    if name == "cnn":
        return TimeSeriesCNN(**model_kwargs)
    if name == "cnn_bilstm":
        return CNNBiLSTM(**model_kwargs)
    if name == "bayesian_mlp":
        return BayesianMLP(**model_kwargs)
    raise ValueError(f"Unknown model_name '{model_name}'. Choices: {sorted(ALL_MODELS)}")


def make_model_kwargs(
    model_name: str,
    n_parameters: int,
    t_last: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    norm: str = "layernorm",
) -> Dict[str, object]:
    """Build the architecture kwargs for ``build_model`` for a given case."""
    name = model_name.lower()
    if name == "mlp":
        return {
            "input_dim": n_parameters,
            "output_dim": 2 * t_last,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "norm": norm,
        }
    if name == "bayesian_mlp":
        # Same I/O contract as the MLP; no norm layer so MC-Dropout stays clean.
        return {
            "input_dim": n_parameters,
            "output_dim": 2 * t_last,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
        }
    if name in SEQUENCE_MODELS:
        return {
            "input_dim": n_parameters + 1,  # +1 normalized-time channel
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "output_dim": 2,
        }
    raise ValueError(f"Unknown model_name '{model_name}'")


def split_prediction(model_name: str, output: torch.Tensor, t_last: int):
    """Split a raw model output into (voltage, temperature) tensors.

    * point (mlp / bayesian_mlp) : ``[B, 2*t_last]``  -> two ``[B, t_last]``
    * sequence (rnn/lstm/bilstm/cnn/cnn_bilstm) : ``[B, t_last, 2]`` -> two ``[B, t_last]``
    """
    if model_name.lower() in POINT_MODELS:
        return output[:, :t_last], output[:, t_last:]
    return output[..., 0], output[..., 1]
