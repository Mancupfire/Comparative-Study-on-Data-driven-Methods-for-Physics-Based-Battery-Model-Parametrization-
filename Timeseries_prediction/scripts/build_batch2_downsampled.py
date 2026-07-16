"""Build the Batch 2 **main** (Batch-1-comparable) time-series dataset by
deterministically downsampling the cleaned native-resolution cases to a fixed
number of uniformly-spaced points.

Why
---
Batch 1 trained on a short fixed time grid (t_last ~147-178). The cleaned
Batch 2 cases are at native ~1 s resolution (t_last 678-7030). Training the
*main* Batch 2 experiment on a comparable fixed grid keeps the modelling
protocol aligned with Batch 1; native resolution is kept as a SEPARATE ablation
(``data/Data_Batch_2_cleaned``), never overwritten by this script.

What it does (per case)
-----------------------
* Reads ``data/Data_Batch_2_cleaned/cases/<case>/outputs.npz`` (immutable input).
* Builds a uniform target grid ``linspace(t[0], t[-1], n_points)`` over the
  **complete** physical time range (no truncation).
* Linearly interpolates every sample's V(t) and T(t) onto that grid
  (deterministic ``np.interp``; the native grid is shared per case).
* Endpoints are preserved exactly (grid starts at t[0] and ends at t[-1]).
* Writes ``<out>/cases/<case>/outputs.npz`` in the same matrix contract
  (``sample_ids``, ``time_s``, ``voltage_v``, ``temperature_c``).
* Copies metadata unchanged and emits a full ``downsample_manifest.json``.

Nothing in ``data/Data_Batch_2`` or ``data/Data_Batch_2_cleaned`` is modified.

Usage
-----
python scripts/build_batch2_downsampled.py \
    --src-dir data/Data_Batch_2_cleaned \
    --out-dir data/Data_Batch_2_downsampled_160 \
    --n-points 160
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

NPZ_KEYS = ("sample_ids", "time_s", "voltage_v", "temperature_c")
METADATA_FILES = ("parameter_sets.csv", "sequence_manifest.csv", "failed_cases.csv",
                  "dataset_summary.json", "README.md")


def sha256(path: Path, buf: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downsample Batch 2 cleaned cases to a fixed grid.")
    p.add_argument("--src-dir", default="data/Data_Batch_2_cleaned",
                   help="Cleaned native-resolution dataset (read-only input).")
    p.add_argument("--out-dir", default="data/Data_Batch_2_downsampled_160")
    p.add_argument("--n-points", type=int, default=160,
                   help="Number of uniformly spaced points per sequence.")
    return p


def discover_case_npzs(src: Path) -> list[Path]:
    cases_dir = src / "cases"
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")
    npzs = sorted(p / "outputs.npz" for p in sorted(cases_dir.iterdir())
                  if p.is_dir() and (p / "outputs.npz").is_file())
    if not npzs:
        raise FileNotFoundError(f"No outputs.npz under {cases_dir}")
    return npzs


def downsample_case(npz_path: Path, n_points: int) -> tuple[dict, dict]:
    """Return (arrays_for_npz, validation_stats) for one case."""
    case_id = npz_path.parent.name
    with np.load(npz_path, allow_pickle=True) as z:
        sample_ids = np.asarray(z["sample_ids"]).astype(str)
        t_src = np.asarray(z["time_s"], dtype=np.float64)        # [t_last]
        V_src = np.asarray(z["voltage_v"], dtype=np.float64)     # [N, t_last]
        T_src = np.asarray(z["temperature_c"], dtype=np.float64)

    n, t_last = V_src.shape
    if t_src.shape[0] != t_last:
        raise ValueError(f"[{case_id}] time/V length mismatch {t_src.shape} vs {V_src.shape}")
    if not np.all(np.diff(t_src) > 0):
        raise ValueError(f"[{case_id}] source time grid not strictly increasing")
    if not (np.isfinite(V_src).all() and np.isfinite(T_src).all() and np.isfinite(t_src).all()):
        raise ValueError(f"[{case_id}] non-finite values in source")

    # Uniform target grid over the FULL physical range; endpoints included.
    t_new = np.linspace(t_src[0], t_src[-1], n_points, dtype=np.float64)
    V_new = np.empty((n, n_points), dtype=np.float64)
    T_new = np.empty((n, n_points), dtype=np.float64)
    for i in range(n):
        V_new[i] = np.interp(t_new, t_src, V_src[i])
        T_new[i] = np.interp(t_new, t_src, T_src[i])

    # ----- validation -----
    if not np.all(np.diff(t_new) > 0):
        raise ValueError(f"[{case_id}] target time grid not strictly increasing")
    if not (np.isfinite(V_new).all() and np.isfinite(T_new).all()):
        raise ValueError(f"[{case_id}] non-finite values after interpolation")
    # Endpoint preservation (np.interp returns exact endpoints).
    if not (np.allclose(V_new[:, 0], V_src[:, 0]) and np.allclose(V_new[:, -1], V_src[:, -1])):
        raise ValueError(f"[{case_id}] voltage endpoints not preserved")
    if not (np.allclose(T_new[:, 0], T_src[:, 0]) and np.allclose(T_new[:, -1], T_src[:, -1])):
        raise ValueError(f"[{case_id}] temperature endpoints not preserved")
    if not (t_new[0] == t_src[0] and t_new[-1] == t_src[-1]):
        raise ValueError(f"[{case_id}] time endpoints not preserved")
    if V_new.shape != (n, n_points) or T_new.shape != (n, n_points):
        raise ValueError(f"[{case_id}] unexpected output shape")

    arrays = {
        "sample_ids": sample_ids,
        "time_s": t_new,
        "voltage_v": V_new,
        "temperature_c": T_new,
    }
    stats = {
        "case_id": case_id,
        "n_samples": int(n),
        "t_last_source": int(t_last),
        "t_last_target": int(n_points),
        "time_start_s": float(t_new[0]),
        "time_end_s": float(t_new[-1]),
        "source_sha256": sha256(npz_path),
        "monotonic_time": True,
        "finite_values": True,
        "endpoints_preserved": True,
        "row_alignment_ok": bool(sample_ids.shape[0] == V_new.shape[0]),
        "voltage_min": float(V_new.min()), "voltage_max": float(V_new.max()),
        "temperature_min": float(T_new.min()), "temperature_max": float(T_new.max()),
    }
    return arrays, stats


def main() -> int:
    args = build_parser().parse_args()
    src = Path(args.src_dir)
    out = Path(args.out_dir)
    if out.resolve() == src.resolve():
        raise SystemExit("out-dir must differ from src-dir (never overwrite cleaned data).")
    (out / "cases").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[downsample] src={src}  out={out}  n_points={args.n_points}")

    npzs = discover_case_npzs(src)
    per_case = {}
    for npz_path in npzs:
        arrays, stats = downsample_case(npz_path, args.n_points)
        case_dir = out / "cases" / stats["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(case_dir / "outputs.npz", **arrays)
        per_case[stats["case_id"]] = stats
        print(f"[downsample] {stats['case_id']}: "
              f"{stats['t_last_source']}->{stats['t_last_target']} pts, "
              f"V[{stats['voltage_min']:.3f},{stats['voltage_max']:.3f}] "
              f"T[{stats['temperature_min']:.3f},{stats['temperature_max']:.3f}]")

    # ----- copy metadata unchanged + checksum -----
    copied, src_checksums = [], {}
    for name in METADATA_FILES:
        s = src / name
        if s.is_file():
            shutil.copy2(s, out / name)
            copied.append(name)
            src_checksums[name] = {"sha256": sha256(s), "bytes": s.stat().st_size}

    manifest = {
        "dataset_role": "time_series_main_downsampled (Batch-1-comparable)",
        "source_dataset": str(src),
        "source_kind": "cleaned native-resolution per-case outputs.npz",
        "interpolation_method": "deterministic linear (numpy.interp) on the shared per-case time grid",
        "grid": "uniform linspace(time_start, time_end, n_points); endpoints preserved; full physical range (no truncation)",
        "n_points": int(args.n_points),
        "n_cases": len(per_case),
        "metadata_copied": copied,
        "source_metadata_checksums": src_checksums,
        "validation": {
            "all_monotonic_time": all(c["monotonic_time"] for c in per_case.values()),
            "all_finite": all(c["finite_values"] for c in per_case.values()),
            "all_endpoints_preserved": all(c["endpoints_preserved"] for c in per_case.values()),
            "all_row_aligned": all(c["row_alignment_ok"] for c in per_case.values()),
            "all_shapes_ok": all(c["t_last_target"] == args.n_points for c in per_case.values()),
        },
        "per_case": per_case,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (out / "downsample_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[downsample] DONE in {time.time()-t0:.1f}s. "
          f"{len(per_case)} cases -> {out/'cases'}. "
          f"Validation: {manifest['validation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
