"""Unified error-metric model registry (12 families).

All neural families share one contract::

    forward(x: [B, n_features]) -> [B, n_targets]

so the training / evaluation loop stays model-agnostic.  Targets are the two
standardized error metrics ``[rmse_voltage_mv, rmse_temperature_c]``.

Recovery status (see code_recovery_report.md): no prior error-metric benchmark
implementation existed anywhere in the repository (no git history, no notebooks,
no archived configs/outputs).  ``mlp`` and ``extratrees`` are adapted from the
existing ``src/error_metric_train.py`` two-model pipeline; the remaining ten
families are clean reimplementations following standard architectures.

The recurrent families (``rnn`` / ``lstm`` / ``bilstm``) consume the static
physical-feature vector as an *ordered feature sequence* of shape
``[B, n_features, 1]`` (one scalar per timestep, sequence length = n_features).
These are NOT temporal-input models; they process the fixed feature order from
the dataset metadata.  This is documented explicitly because the original
benchmark's sequence representation could not be recovered.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

# Families whose forward() expects a sequence view [B, n_features, 1].
SEQUENCE_FAMILIES = {"rnn", "lstm", "bilstm"}
# Pure-torch point families.
POINT_FAMILIES = {
    "ann", "mlp", "wide_deep_mlp", "attention_mlp", "gated_mlp",
    "residual_mlp", "multitask_mlp",
}
# Composite / non-standard torch families.
SPECIAL_TORCH_FAMILIES = {"deep_ensemble_mlp"}
# Classical (sklearn) families.
CLASSICAL_FAMILIES = {"extratrees"}

NEURAL_FAMILIES = POINT_FAMILIES | SEQUENCE_FAMILIES | SPECIAL_TORCH_FAMILIES
ALL_FAMILIES = NEURAL_FAMILIES | CLASSICAL_FAMILIES

# Canonical ordering used in tables / reports.
FAMILY_ORDER: List[str] = [
    "ann", "mlp", "wide_deep_mlp", "attention_mlp", "gated_mlp",
    "residual_mlp", "multitask_mlp", "deep_ensemble_mlp",
    "rnn", "lstm", "bilstm", "extratrees",
]

DISPLAY_NAMES: Dict[str, str] = {
    "ann": "ANN",
    "mlp": "MLP",
    "wide_deep_mlp": "Wide & Deep MLP",
    "attention_mlp": "Attention MLP",
    "gated_mlp": "Gated MLP",
    "residual_mlp": "Residual MLP",
    "multitask_mlp": "Multitask MLP",
    "deep_ensemble_mlp": "Deep Ensemble MLP",
    "rnn": "RNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "extratrees": "ExtraTrees",
}


# --------------------------------------------------------------------------- #
# Point models
# --------------------------------------------------------------------------- #
class ANN(nn.Module):
    """Shallow single-hidden-layer network (classic ANN baseline)."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 64,
                 dropout: float = 0.1, **_: object):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """Multi-layer perceptron (mirrors src/error_metric_train.ErrorMetricMLP)."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WideDeepMLP(nn.Module):
    """Wide & Deep: a linear 'wide' path summed with a deep MLP 'deep' path."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        self.wide = nn.Linear(in_dim, out_dim)
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.deep = nn.Sequential(*layers)
        self.deep_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wide(x) + self.deep_head(self.deep(x))


class _SelfFeatureAttention(nn.Module):
    """Feature-token self-attention block over the [B, F, d] token embeddings."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(tokens, tokens, tokens)
        return self.norm(tokens + a)


class AttentionMLP(nn.Module):
    """Embed each scalar feature as a token, apply self-attention, then MLP head.

    Each of the ``in_dim`` features is projected to ``embed_dim`` and treated as
    a token; a multi-head self-attention layer mixes information across features
    before mean-pooling and a small MLP head produces the two targets.
    """

    def __init__(self, in_dim: int, out_dim: int = 2, embed_dim: int = 32,
                 num_heads: int = 4, dropout: float = 0.1, **_: object):
        super().__init__()
        self.in_dim = in_dim
        self.embed = nn.Linear(1, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, in_dim, embed_dim))
        self.block = _SelfFeatureAttention(embed_dim, num_heads, dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(x.unsqueeze(-1)) + self.pos     # [B, F, e]
        tokens = self.block(tokens)
        pooled = tokens.mean(dim=1)                          # [B, e]
        return self.head(pooled)


class _GatedBlock(nn.Module):
    """GLU-style gated feed-forward block with a residual connection."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.value(x) * torch.sigmoid(self.gate(x))
        return self.norm(x + self.drop(h))


class GatedMLP(nn.Module):
    """MLP built from gated (GLU) residual blocks."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_GatedBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class ResidualMLP(nn.Module):
    """Pre-projection followed by residual MLP blocks (ResNet-style)."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.proj(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)


class MultitaskMLP(nn.Module):
    """Shared trunk with two independent single-output heads (one per target).

    The two targets (voltage RMSE, temperature RMSE) are produced by separate
    heads off a shared representation, the canonical hard-parameter-sharing
    multitask design.  Output is concatenated to ``[B, 2]`` to keep the common
    contract.
    """

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.1, **_: object):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                          nn.Linear(hidden_dim // 2, 1))
            for _ in range(out_dim)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)


# --------------------------------------------------------------------------- #
# Sequence-over-feature models (NOT temporal)
# --------------------------------------------------------------------------- #
class _RecurrentRegressor(nn.Module):
    """RNN/LSTM/BiLSTM over the ordered feature sequence [B, F, 1] -> [B, 2]."""

    def __init__(self, in_dim: int, out_dim: int = 2, hidden_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.1,
                 kind: str = "lstm", bidirectional: bool = False, **_: object):
        super().__init__()
        rnn_dropout = dropout if num_layers > 1 else 0.0
        common = dict(input_size=1, hidden_size=hidden_dim, num_layers=num_layers,
                      batch_first=True, dropout=rnn_dropout,
                      bidirectional=bidirectional)
        if kind == "rnn":
            self.rnn: nn.Module = nn.RNN(nonlinearity="tanh", **common)
        else:
            self.rnn = nn.LSTM(**common)
        head_in = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x arrives as [B, F]; view as a length-F sequence of scalars.
        if x.dim() == 2:
            x = x.unsqueeze(-1)            # [B, F, 1]
        out, _ = self.rnn(x)              # [B, F, hidden*(2 if bi)]
        return self.head(out[:, -1, :])  # last "timestep" -> [B, out_dim]


class RNNRegressor(_RecurrentRegressor):
    def __init__(self, in_dim: int, **kw):
        kw.pop("kind", None); kw.pop("bidirectional", None)
        super().__init__(in_dim, kind="rnn", bidirectional=False, **kw)


class LSTMRegressor(_RecurrentRegressor):
    def __init__(self, in_dim: int, **kw):
        kw.pop("kind", None); kw.pop("bidirectional", None)
        super().__init__(in_dim, kind="lstm", bidirectional=False, **kw)


class BiLSTMRegressor(_RecurrentRegressor):
    def __init__(self, in_dim: int, **kw):
        kw.pop("kind", None); kw.pop("bidirectional", None)
        super().__init__(in_dim, kind="lstm", bidirectional=True, **kw)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_TORCH_BUILDERS = {
    "ann": ANN,
    "mlp": MLP,
    "wide_deep_mlp": WideDeepMLP,
    "attention_mlp": AttentionMLP,
    "gated_mlp": GatedMLP,
    "residual_mlp": ResidualMLP,
    "multitask_mlp": MultitaskMLP,
    "rnn": RNNRegressor,
    "lstm": LSTMRegressor,
    "bilstm": BiLSTMRegressor,
}


def default_arch(family: str, in_dim: int, out_dim: int, cfg: Dict) -> Dict:
    """Architecture kwargs for a family, drawing defaults from ``cfg``."""
    h = int(cfg.get("hidden_dim", 128))
    nl = int(cfg.get("num_layers", 3))
    do = float(cfg.get("dropout", 0.1))
    base = {"in_dim": in_dim, "out_dim": out_dim}
    if family == "ann":
        return {**base, "hidden_dim": int(cfg.get("ann_hidden_dim", 64)), "dropout": do}
    if family in {"mlp", "wide_deep_mlp", "gated_mlp", "residual_mlp", "multitask_mlp"}:
        return {**base, "hidden_dim": h, "num_layers": nl, "dropout": do}
    if family == "attention_mlp":
        return {**base, "embed_dim": int(cfg.get("attn_embed_dim", 32)),
                "num_heads": int(cfg.get("attn_num_heads", 4)), "dropout": do}
    if family in SEQUENCE_FAMILIES:
        return {**base, "hidden_dim": int(cfg.get("rnn_hidden_dim", 64)),
                "num_layers": int(cfg.get("rnn_num_layers", 1)), "dropout": do}
    raise ValueError(f"No arch spec for family '{family}'")


def build_torch_model(family: str, arch: Dict) -> nn.Module:
    if family not in _TORCH_BUILDERS:
        raise ValueError(f"'{family}' is not a buildable single torch model. "
                         f"Choices: {sorted(_TORCH_BUILDERS)}")
    return _TORCH_BUILDERS[family](**arch)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
