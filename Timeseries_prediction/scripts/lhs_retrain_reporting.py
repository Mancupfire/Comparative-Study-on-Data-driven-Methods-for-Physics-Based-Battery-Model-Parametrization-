"""Shared, additive reproducibility/reporting helpers for the official LHS
pipelines (``emergency_lhs_train.py`` and ``lhs_error_metrics_train.py``).

These helpers only *record* provenance and enrich reporting tables. They do NOT
touch model architectures, hyperparameters, optimizers, seeds, the sample_id
split, the fixed-length resampling, or the aligned-tail protocol. Every function
here is safe to call after training has produced its metrics/timing frames.

Exports written by these helpers:
  * ``environment.json``   -- Python/PyTorch/CUDA/sklearn/xgboost/catboost/... versions + git commit.
  * ``dataset_audit.json`` -- SHA256 checksums + sizes of the four dataset files
                              plus the confirmed generation-summary facts.
And ``enrich_timing`` adds device/CUDA/GPU/throughput/test-batch columns to an
existing model_timing frame.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# The four dataset files whose SHA256 checksums the protocol requires.
AUDIT_FILES: List[str] = [
    "generation_summary.json",
    "sequence_manifest.csv",
    "error_metrics_by_case.csv",
    "generated_dataset.h5",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 so the multi-hundred-MB HDF5 file is not read at once."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path) -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _ver(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def build_environment() -> Dict[str, object]:
    env: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _ver("numpy"),
        "pandas": _ver("pandas"),
        "sklearn": _ver("sklearn"),
        "scipy": _ver("scipy"),
        "xgboost": _ver("xgboost"),
        "catboost": _ver("catboost"),
        "h5py": _ver("h5py"),
        "matplotlib": _ver("matplotlib"),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_version"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cudnn_version"] = (
            torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available()
            else None
        )
        env["gpu_name"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception:
        env["torch"] = None
    return env


def write_environment(path: Path, repo: Path) -> Dict[str, object]:
    env = build_environment()
    env["git_commit"] = git_commit(repo)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(env, indent=2), encoding="utf-8")
    return env


def build_dataset_audit(
    data_dir: Path,
    split_counts: Optional[Dict[str, int]] = None,
    selected: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    data_dir = Path(data_dir)
    files: Dict[str, object] = {}
    for fn in AUDIT_FILES:
        p = data_dir / fn
        files[fn] = (
            {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            if p.exists()
            else None
        )
    audit: Dict[str, object] = {"data_dir": str(data_dir), "files": files}
    gsp = data_dir / "generation_summary.json"
    if gsp.exists():
        gs = json.loads(gsp.read_text())
        audit["generation_summary"] = {
            k: gs.get(k)
            for k in (
                "n_requested_samples",
                "n_successful_sequences",
                "n_failed_sequences",
                "grid_mode",
                "sampling_mode",
                "seed",
                "sequence_length",
                "case_ids",
            )
        }
    if split_counts is not None:
        audit["split_counts"] = split_counts
    if selected is not None:
        audit["selected"] = selected
    return audit


def write_dataset_audit(
    path: Path,
    data_dir: Path,
    split_counts: Optional[Dict[str, int]] = None,
    selected: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    audit = build_dataset_audit(data_dir, split_counts, selected)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def enrich_timing(timing_df, device, test_batch_size: int, test_rows: int):
    """Return a copy of ``timing_df`` with the protocol timing/device columns.

    Adds: ``test_batch_size``, ``test_rows``, ``device``, ``device_name``,
    ``cuda_version``, ``gpu_name`` and ``throughput_sequences_per_second``
    (derived from an existing ``inference_seconds_total`` column when present).
    Never overwrites columns the trainer already populated.
    """
    import torch

    df = timing_df.copy()
    if "test_batch_size" not in df:
        df["test_batch_size"] = int(test_batch_size)
    if "test_rows" not in df:
        df["test_rows"] = int(test_rows)

    dev = torch.device(device) if not isinstance(device, torch.device) else device
    is_cuda = dev.type == "cuda" and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(dev) if is_cuda else str(dev)
    df["device"] = str(dev)
    df["device_name"] = device_name
    df["cuda_version"] = torch.version.cuda
    df["gpu_name"] = torch.cuda.get_device_name(dev) if is_cuda else None

    if "inference_seconds_total" in df and "throughput_sequences_per_second" not in df:
        secs = df["inference_seconds_total"].clip(lower=1e-12)
        df["throughput_sequences_per_second"] = float(test_rows) / secs
    return df
