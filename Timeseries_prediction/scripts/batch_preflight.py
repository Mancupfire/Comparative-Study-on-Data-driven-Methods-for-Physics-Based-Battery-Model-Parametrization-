"""Generic isolation / overwrite safeguard for any dataset batch.

Backward-compatible companion to ``scripts/batch2_preflight.py`` (which stays
Batch-2-locked and is used by the Batch 2 wrappers). This generic version is
parameterised by ``--dataset-name`` so Batch 3 (and future batches) get the same
guarantees without weakening Batch 1/2 protection:

* data/output/log paths must contain the requested ``--dataset-name`` namespace,
* must NOT contain any *other* known dataset namespace (Data_Batch_1 / Data_Batch_2),
* must not write directly at a Batch 1 repo root (outputs/ outputs_smoke/ logs/),
* refuse a non-empty run dir unless ``--allow-resume``.

Usage:
    python scripts/batch_preflight.py --dataset-name Data_Batch_3 \
        --data-dir data/Data_Batch_3_downsampled_160 \
        --output-root outputs/Data_Batch_3/time_series_downsampled_160/<run_id> \
        --log-root logs/Data_Batch_3/time_series_downsampled_160/<run_id> \
        [--allow-resume]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KNOWN_NAMESPACES = ("Data_Batch_1", "Data_Batch_2", "Data_Batch_3")
BATCH1_ROOT_NAMES = ("outputs", "outputs_smoke", "logs")


def _fail(msg: str) -> None:
    print(f"[preflight] ABORT: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _ns_of(name: str) -> str:
    m = re.match(r"(Data_Batch_\d+)", name)
    return m.group(1) if m else name


def _has_ns(path: Path, ns: str) -> bool:
    return any(ns in part for part in path.parts)


def _check_namespace(label: str, path: Path, ns: str) -> None:
    if not _has_ns(path, ns):
        _fail(f"{label} '{path}' does not contain the '{ns}' namespace.")
    for other in KNOWN_NAMESPACES:
        if other == ns:
            continue
        if any(other in part for part in path.parts):
            _fail(f"{label} '{path}' contains a foreign namespace fragment '{other}'.")


def _check_not_batch1_root(label: str, path: Path, repo: Path, ns: str) -> None:
    try:
        rel = path.resolve().relative_to(repo.resolve())
    except ValueError:
        return
    parts = rel.parts
    if parts and parts[0] in BATCH1_ROOT_NAMES:
        if len(parts) < 2 or parts[1] != ns:
            _fail(f"{label} '{path}' would write directly under the root "
                  f"'{parts[0]}/' instead of '{parts[0]}/{ns}/...'.")


def _check_run_dir_empty(label: str, path: Path, allow_resume: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not allow_resume:
            _fail(f"{label} run directory '{path}' is non-empty. Refusing to "
                  f"overwrite. Pass --allow-resume to continue intentionally.")
        print(f"[preflight] NOTE: {label} '{path}' is non-empty; resume allowed.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generic dataset-batch isolation preflight.")
    p.add_argument("--dataset-name", required=True, help="e.g. Data_Batch_3")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--log-root", required=True)
    p.add_argument("--allow-resume", action="store_true")
    p.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    return p


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo_root)
    ns = _ns_of(args.dataset_name)
    data_dir = Path(args.data_dir)
    out_root = Path(args.output_root)
    log_root = Path(args.log_root)

    # data-dir: must belong to this dataset namespace, never a foreign one.
    _check_namespace("data-dir", data_dir, ns)
    if not data_dir.exists():
        _fail(f"data-dir '{data_dir}' does not exist.")

    _check_namespace("output-root", out_root, ns)
    _check_namespace("log-root", log_root, ns)
    _check_not_batch1_root("output-root", out_root, repo, ns)
    _check_not_batch1_root("log-root", log_root, repo, ns)
    _check_run_dir_empty("output-root", out_root, args.allow_resume)

    print(f"[preflight] OK — isolation checks passed for namespace '{ns}'.")
    print(f"[preflight]   data_dir    = {data_dir.resolve()}")
    print(f"[preflight]   output_root = {out_root.resolve()}")
    print(f"[preflight]   log_root    = {log_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
