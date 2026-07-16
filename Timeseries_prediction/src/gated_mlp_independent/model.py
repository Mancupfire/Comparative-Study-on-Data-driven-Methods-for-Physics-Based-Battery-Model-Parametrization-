"""Scalar Gated-MLP architecture plus lightweight preprocessing helpers.

The architecture follows the task specification:

* standardized parameter-vector input
* an input projection
* 3-4 gated residual MLP blocks (LayerNorm, SiLU/GELU activation, sigmoid gate,
  residual connection, small dropout)
* a final *scalar* regression head.

Preprocessing helpers (`StandardScaler`, `TargetTransform`) are deliberately
dependency-light and JSON-serialisable so checkpoints and metadata can be
reloaded for inference without re-fitting anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
_ACTIVATIONS = {"silu": nn.SiLU, "gelu": nn.GELU}


class GatedResidualBlock(nn.Module):
    """Gated feed-forward block with LayerNorm and a residual connection.

    ``h = activation(value(x)) * sigmoid(gate(x))`` followed by a projection,
    dropout and a residual add normalised by LayerNorm.
    """

    def __init__(self, dim: int, dropout: float, activation: str = "silu"):
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.act = _ACTIVATIONS[activation]()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.value(x)) * torch.sigmoid(self.gate(x))
        h = self.drop(self.proj(h))
        return self.norm(x + h)


class GatedMLP(nn.Module):
    """Single scalar-output Gated-MLP regressor."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        out_dim: int = 1,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {list(_ACTIVATIONS)}")
        self.config = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "n_blocks": n_blocks,
            "dropout": dropout,
            "activation": activation,
            "out_dim": out_dim,
        }
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            _ACTIVATIONS[activation](),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, dropout, activation) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.head(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class StandardScaler:
    """z-score scaler fit on a 2-D array; JSON serialisable."""

    columns: List[str]
    log10_columns: List[str] = field(default_factory=list)
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None

    def _apply_log10(self, x: np.ndarray) -> np.ndarray:
        if not self.log10_columns:
            return x
        x = x.astype(np.float64).copy()
        idx = [self.columns.index(c) for c in self.log10_columns]
        x[:, idx] = np.log10(x[:, idx])
        return x

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = self._apply_log10(np.asarray(x, dtype=np.float64))
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale == 0.0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._apply_log10(np.asarray(x, dtype=np.float64))
        return (x - self.mean) / self.scale

    def to_dict(self) -> Dict:
        return {
            "columns": list(self.columns),
            "log10_columns": list(self.log10_columns),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StandardScaler":
        s = cls(columns=d["columns"], log10_columns=d.get("log10_columns", []))
        s.mean = np.asarray(d["mean"], dtype=np.float64)
        s.scale = np.asarray(d["scale"], dtype=np.float64)
        return s


@dataclass
class TargetTransform:
    """Per-target transform: optional log1p, then z-score standardisation.

    ``log1p`` is only valid for non-negative targets and is intended for
    strongly right-skewed ones.  ``inverse`` reverses the full pipeline so
    metrics are reported in the original target units.
    """

    name: str
    use_log1p: bool = False
    mean: float = 0.0
    scale: float = 1.0

    def fit(self, y: np.ndarray) -> "TargetTransform":
        y = np.asarray(y, dtype=np.float64)
        if self.use_log1p:
            if np.any(y < 0):
                raise ValueError(f"log1p target '{self.name}' has negative values")
            y = np.log1p(y)
        self.mean = float(y.mean())
        s = float(y.std())
        self.scale = s if s > 0 else 1.0
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.use_log1p:
            y = np.log1p(y)
        return (y - self.mean) / self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        y = z * self.scale + self.mean
        if self.use_log1p:
            y = np.expm1(y)
        return y

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "use_log1p": self.use_log1p,
            "mean": self.mean,
            "scale": self.scale,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TargetTransform":
        return cls(
            name=d["name"],
            use_log1p=bool(d["use_log1p"]),
            mean=float(d["mean"]),
            scale=float(d["scale"]),
        )
