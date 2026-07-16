"""Generic helpers: reproducibility, device selection and small IO utilities.

Everything here is dependency-light so the rest of the codebase can rely on a
single consistent way of seeding, choosing a device and writing JSON.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN trades a little speed for reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for some scripts (e.g. inspect_dataset)
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string. ``auto`` -> ``cuda`` if available else ``cpu``."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


def ensure_dir(path: PathLike) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: PathLike, indent: int = 2) -> Path:
    """Write a dict to JSON, creating parent directories as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    return p


def load_json(path: PathLike) -> Dict[str, Any]:
    """Load a JSON file into a dict, with a clear error if it is missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(value: Any) -> Any:
    """Make NumPy scalars / arrays JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")
