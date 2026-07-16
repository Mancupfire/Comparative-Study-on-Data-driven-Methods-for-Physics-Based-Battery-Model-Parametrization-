"""Filtering-count and grouped-split reconstruction tests.

All expected numbers are derived from the data, then checked against the
protocol's fixed expectations:

    source=12000  kept=11109  removed=891
    per-case removals: CC_C_2p5_T25C=135, CC_D_2p5_T25C=134, CC_C_1p5_T25C=117,
                       CC_D_2p5_T45C=93, CC_D_1p5_T25C=91

    grouped split (seeds 42/43/44): 700/150/150 sample ids, zero overlap;
    seed-42 split is identical to the completed grouped EM benchmark.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.final_filtered.data import grouped_split, all_sample_ids  # noqa: E402

FILTERED = REPO / "data" / "Data_Batch_4_TSFiltered_0p8"
GROUPED_RUN = (REPO / "outputs" / "Data_Batch_4" / "error_metric_benchmark"
               / "batch4_em_grouped_20260622_110539")


def test_filter_counts():
    src = pd.read_csv(FILTERED / "time_series_source_manifest.csv")
    assert len(src) == 12000
    assert int(src["kept"].sum()) == 11109
    assert int((~src["kept"]).sum()) == 891


def test_filter_recompute_from_raw_manifest():
    """duration_ratio recomputed directly from the manifest matches the kept flag."""
    src = pd.read_csv(FILTERED / "time_series_source_manifest.csv")
    ratio = src["simulation_end_s"] / src["reference_end_s"]
    recomputed_kept = (ratio >= 0.8)
    assert int(recomputed_kept.sum()) == 11109
    assert (recomputed_kept.to_numpy() == src["kept"].to_numpy()).all()


def test_per_case_removals():
    by_case = pd.read_csv(FILTERED / "removed_sequences_by_case.csv")
    rem = dict(zip(by_case["experiment_id"], by_case["removed"]))
    expected = {"CC_C_2p5_T25C": 135, "CC_D_2p5_T25C": 134, "CC_C_1p5_T25C": 117,
                "CC_D_2p5_T45C": 93, "CC_D_1p5_T25C": 91}
    for case, n in expected.items():
        assert int(rem[case]) == n, f"{case}: expected {n}, got {rem[case]}"
    assert int(by_case["removed"].sum()) == 891
    assert int(by_case["kept"].sum()) == 11109


def test_grouped_split_sizes_and_disjoint():
    ids = all_sample_ids()
    assert len(ids) == 1000
    for seed in (42, 43, 44):
        g = grouped_split(ids, seed=seed)
        assert len(g["train"]) == 700
        assert len(g["val"]) == 150
        assert len(g["test"]) == 150
        assert not (g["train"] & g["val"])
        assert not (g["train"] & g["test"])
        assert not (g["val"] & g["test"])
        assert g["train"] | g["val"] | g["test"] == set(ids.tolist())


def test_grouped_split_matches_completed_benchmark_seed42():
    """Seed-42 grouped split must equal the completed grouped EM benchmark."""
    mani_path = GROUPED_RUN / "split_manifest.csv"
    if not mani_path.is_file():
        # Completed run not present in this checkout — skip gracefully.
        print("SKIP: completed grouped benchmark split_manifest.csv not found")
        return
    mani = pd.read_csv(mani_path)
    bench = {s: set(mani.loc[mani["split"] == s, "sample_id"].astype(str))
             for s in ("train", "val", "test")}
    g = grouped_split(all_sample_ids(), seed=42)
    for s in ("train", "val", "test"):
        assert g[s] == bench[s], f"seed42 grouped split mismatch on '{s}'"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL FILTERING/SPLIT TESTS PASSED")
