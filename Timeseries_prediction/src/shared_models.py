"""Shared, condition-aware model architectures.

All shared models map a parameter+condition+time feature vector to the voltage
and temperature at that time::

    f_theta([parameter_vector, c_rate, ambient_temp_C, time_norm]) -> [V(t), T(t)]

``input_dim`` is therefore ``n_parameters + 3`` for every model.

* ``shared_mlp``         : point-wise.  input ``[B, input_dim]`` -> output ``[B, 2]``.
* ``shared_rnn``         : sequence.    input ``[B, T, input_dim]`` -> output ``[B, T, 2]``.
* ``shared_lstm``        : sequence (nn.LSTM).
* ``shared_bilstm``      : sequence (nn.LSTM, ``bidirectional=True``).
* ``shared_cnn``         : sequence (Conv1d temporal model).
* ``shared_cnn_bilstm``  : sequence (Conv1d front-end + BiLSTM).
* ``shared_bayesian_mlp``: point-wise MC-Dropout MLP.

The recurrent sequence models accept an optional ``lengths`` tensor and use
``pack_padded_sequence`` so padded timesteps never influence the recurrence
(this matters in particular for the bidirectional backward pass).  The new
sequence models keep the same ``forward(x, lengths=None)`` signature so the
shared training / evaluation code stays unchanged; see each class for how
padding is handled.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

SHARED_SEQUENCE_MODELS = {
    "shared_rnn", "shared_lstm", "shared_bilstm", "shared_cnn", "shared_cnn_bilstm"
}
SHARED_POINT_MODELS = {"shared_mlp", "shared_bayesian_mlp"}
ALL_SHARED_MODELS = SHARED_POINT_MODELS | SHARED_SEQUENCE_MODELS


class SharedMLP(nn.Module):
    """Point-wise MLP predicting ``[V(t), T(t)]`` from one feature vector.

    Architecture (per the project spec)::

        Linear(input_dim -> h)  -> ReLU -> LayerNorm -> Dropout
        Linear(h  -> 2h)        -> ReLU -> Dropout
        Linear(2h -> 2h)        -> ReLU
        Linear(2h -> 2)
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
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedRecurrent(nn.Module):
    """Shared wrapper over nn.RNN / nn.LSTM for per-step ``[V, T]`` regression.

    ``forward`` accepts an optional ``lengths`` tensor; when given the input is
    packed so padded timesteps are excluded from the recurrence.
    """

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
        # PyTorch only applies inter-layer dropout when num_layers > 1.
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

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if lengths is not None:
            max_t = x.shape[1]
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=max_t)
        else:
            out, _ = self.rnn(x)
        return self.head(out)  # [B, T, 2]


class SharedRNN(SharedRecurrent):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="rnn", bidirectional=False, **kwargs)


class SharedLSTM(SharedRecurrent):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="lstm", bidirectional=False, **kwargs)


class SharedBiLSTM(SharedRecurrent):
    def __init__(self, input_dim: int, **kwargs):
        kwargs.pop("kind", None)
        kwargs.pop("bidirectional", None)
        super().__init__(input_dim, kind="lstm", bidirectional=True, **kwargs)


class SharedCNN(nn.Module):
    """Shared 1-D temporal CNN over padded sequences.

    Mirrors :class:`src.models.TimeSeriesCNN` but supports the shared pipeline's
    padded ``[B, max_T, input_dim]`` batches.

    Limitation: the convolution *does* see padded timesteps (Conv1d has no
    length-aware masking), so padded positions can leak into the receptive field
    of nearby valid positions.  This is accepted by design — correctness is
    preserved because the training / evaluation **loss and metrics are masked**
    so padded outputs never contribute to gradients or scores.  ``lengths`` is
    accepted for signature compatibility but is not used here.
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
        self.head = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x.transpose(1, 2)          # [B, max_T, F] -> [B, F, max_T]
        h = self.body(x)               # [B, hidden, max_T]
        out = self.head(h)             # [B, output_dim, max_T]
        return out.transpose(1, 2)     # [B, max_T, output_dim]


class SharedCNNBiLSTM(nn.Module):
    """Shared Conv1d front-end followed by a BiLSTM, for padded sequences.

    The Conv1d feature extractor runs over the padded sequence (same limitation
    as :class:`SharedCNN`: it can see padded steps).  The BiLSTM, however, uses
    ``pack_padded_sequence`` when ``lengths`` is provided so padded timesteps do
    not influence the recurrence (important for the bidirectional backward
    pass).  Either way the loss is masked downstream.
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

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        max_t = x.shape[1]
        h = x.transpose(1, 2)          # [B, max_T, F] -> [B, F, max_T]
        h = self.cnn(h)                # [B, hidden, max_T]
        h = h.transpose(1, 2)          # [B, max_T, hidden]
        if lengths is not None:
            packed = pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out_packed, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=max_t)
        else:
            out, _ = self.lstm(h)
        return self.head(out)          # [B, max_T, output_dim]


class SharedBayesianMLP(nn.Module):
    """Point-wise MC-Dropout MLP: ``[B, input_dim]`` -> ``[B, 2]``.

    Same point-wise contract as :class:`SharedMLP`, but with a dropout layer
    after every hidden activation and no normalization layer, so dropout can be
    kept active at inference time for Monte-Carlo uncertainty sampling.  Trained
    with ordinary MSE.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        output_dim: int = 2,
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
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_shared_model(model_name: str, model_kwargs: Dict[str, object]) -> nn.Module:
    """Instantiate a shared model from its name and a kwargs dict.

    ``model_kwargs`` is stored verbatim in the checkpoint so the architecture can
    be rebuilt at inference time without external configuration.
    """
    name = model_name.lower()
    if name == "shared_mlp":
        return SharedMLP(**model_kwargs)
    if name == "shared_rnn":
        return SharedRNN(**model_kwargs)
    if name == "shared_lstm":
        return SharedLSTM(**model_kwargs)
    if name == "shared_bilstm":
        return SharedBiLSTM(**model_kwargs)
    if name == "shared_cnn":
        return SharedCNN(**model_kwargs)
    if name == "shared_cnn_bilstm":
        return SharedCNNBiLSTM(**model_kwargs)
    if name == "shared_bayesian_mlp":
        return SharedBayesianMLP(**model_kwargs)
    raise ValueError(
        f"Unknown shared model '{model_name}'. Choices: {sorted(ALL_SHARED_MODELS)}"
    )


def make_shared_model_kwargs(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> Dict[str, object]:
    """Build the architecture kwargs for ``build_shared_model``."""
    name = model_name.lower()
    if name in {"shared_mlp", "shared_bayesian_mlp"}:
        return {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "output_dim": 2,
        }
    if name in SHARED_SEQUENCE_MODELS:
        return {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "output_dim": 2,
        }
    raise ValueError(f"Unknown shared model '{model_name}'")
