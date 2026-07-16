"""Isolation / overwrite safeguard for every Batch 2 execution path.

Run BEFORE any training or evaluation writes happen.  It enforces the critical
data-isolation requirements so a Batch 2 run can never collide with, overwrite,
or be confused for Batch 1 results.

Checks
------
1. The resolved data directory contains ``Data_Batch_2`` (and NOT ``Data_Batch_1``).
2. The resolved output and log directories contain ``Data_Batch_2`` and do NOT
   contain ``Data_Batch_1``.
3. The output directory is not equal to, and not inside, a known Batch 1 output
   root (the repo-level ``outputs/`` / ``outputs_smoke/`` / ``logs/`` used by
   Batch 1 live directly under those roots, e.g. ``outputs/checkpoints/...``).
4. A non-empty run directory is refused unless ``--allow-resume`` is given.

Exit code 0 = safe, non-zero = abort.

Usage
-----
python scripts/batch2_preflight.py \
    --data-dir data/Data_Batch_2_cleaned \
    --output-root outputs/Data_Batch_2/time_series/<run_id> \
    --log-root logs/Data_Batch_2/time_series/<run_id> \
    [--allow-resume]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Path fragments that, if present in an OUTPUT/LOG target, mean we are about to
# touch Batch 1 territory.
BATCH1_FORBIDDEN = ("Data_Batch_1",)
# Batch 1 wrote per-case artifacts directly under these repo roots
# (e.g. outputs/checkpoints/<case>/<model>).  A Batch 2 output root must live
# under a Data_Batch_2 namespace, never directly at these roots.
BATCH1_ROOT_NAMES = ("outputs", "outputs_smoke", "logs")


def _fail(msg: str) -> None:
    print(f"[preflight] ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def has_batch2_namespace(path: Path) -> bool:
    """True if any path component belongs to a Batch 2 namespace.

    Substring (not exact-component) match so that processed variants such as
    ``Data_Batch_2_cleaned`` and ``Data_Batch_2_downsampled_160`` are accepted,
    not only the bare ``Data_Batch_2`` directory.
    """
    return any("Data_Batch_2" in part for part in path.parts)


def has_batch1_namespace(path: Path) -> bool:
    """True if any path component belongs to a Batch 1 namespace (forbidden)."""
    return any("Data_Batch_1" in part for part in path.parts)


def _check_contains_batch2(label: str, path: Path) -> None:
    parts = path.parts
    if "Data_Batch_2" not in parts:
        _fail(f"{label} '{path}' does not contain a 'Data_Batch_2' namespace.")
    for bad in BATCH1_FORBIDDEN:
        if bad in parts:
            _fail(f"{label} '{path}' contains forbidden Batch 1 fragment '{bad}'.")


def _check_not_batch1_root(label: str, path: Path, repo: Path) -> None:
    """Refuse an output/log root that sits *directly* at a Batch 1 repo root."""
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return  # outside repo; the Data_Batch_2 check already passed
    parts = rel.parts
    # e.g. outputs/checkpoints/... (Batch 1) has parts[1] != 'Data_Batch_2'
    if parts and parts[0] in BATCH1_ROOT_NAMES:
        if len(parts) < 2 or parts[1] != "Data_Batch_2":
            _fail(f"{label} '{path}' would write directly under the Batch 1 root "
                  f"'{parts[0]}/' instead of '{parts[0]}/Data_Batch_2/...'.")


def _check_run_dir_empty(label: str, path: Path, allow_resume: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not allow_resume:
            _fail(f"{label} run directory '{path}' is non-empty. Refusing to "
                  f"overwrite. Pass --allow-resume to continue intentionally.")
        print(f"[preflight] NOTE: {label} '{path}' is non-empty; resume allowed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch 2 isolation preflight.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--log-root", required=True)
    p.add_argument("--allow-resume", action="store_true")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo_root)
    data_dir = Path(args.data_dir)
    out_root = Path(args.output_root)
    log_root = Path(args.log_root)

    # 1. data dir must point at a Batch 2 namespace (raw, cleaned or any
    #    processed variant such as Data_Batch_2_downsampled_160), never Batch 1.
    if has_batch1_namespace(data_dir):
        _fail(f"data-dir '{data_dir}' points at Batch 1.")
    if not has_batch2_namespace(data_dir):
        _fail(f"data-dir '{data_dir}' is not a Data_Batch_2 directory.")
    if not data_dir.exists():
        _fail(f"data-dir '{data_dir}' does not exist.")

    # 2/3. output + log roots must be inside a Data_Batch_2 namespace.
    _check_contains_batch2("output-root", out_root)
    _check_contains_batch2("log-root", log_root)
    _check_not_batch1_root("output-root", out_root, repo)
    _check_not_batch1_root("log-root", log_root, repo)

    # 4. refuse clobbering a non-empty run dir.
    _check_run_dir_empty("output-root", out_root, args.allow_resume)

    print("[preflight] OK — isolation checks passed.")
    print(f"[preflight]   data_dir    = {data_dir.resolve()}")
    print(f"[preflight]   output_root = {out_root.resolve()}")
    print(f"[preflight]   log_root    = {log_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
