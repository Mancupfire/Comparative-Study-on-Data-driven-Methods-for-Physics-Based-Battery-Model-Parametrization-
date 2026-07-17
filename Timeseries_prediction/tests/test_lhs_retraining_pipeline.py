"""Focused unit tests for the LHS retraining pipeline update.

These cover the pieces the retrain protocol depends on, and run fast (no GPU, no
full training):

  * reproducibility/reporting helpers (environment, dataset audit, timing enrichment);
  * the fixed-length capacity resampling contract (endpoint-held tail preserved);
  * the group-aware sample_id split integrity of the regenerated dataset
    (699/150/151, zero leakage, all six cases of a sample_id in one split);
  * the error-metric feature/target construction (13 features, two targets).

Tests that need the real ``data/lhs_1000_seed42`` dataset are skipped when it is
absent, so this file is safe to run anywhere.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import emergency_lhs_train as elt  # noqa: E402
import lhs_error_metrics_train as emt  # noqa: E402
import lhs_retrain_reporting as reporting  # noqa: E402

# Reuse the synthetic-dataset builder from the alignment test suite.
sys.path.insert(0, str(REPO / "tests"))
from test_lhs_alignment import _build_dataset  # noqa: E402

DATA_DIR = REPO / "data" / "lhs_1000_seed42"
HAS_REAL = (DATA_DIR / "sequence_manifest.csv").exists()
STORED_SPLIT = (
    REPO / "outputs/lhs_1000_seed42/time_series/"
    "lhs_official_full_20260715_134335/artifacts/split_sample_ids.json"
)


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def test_build_environment_has_core_keys():
    env = reporting.build_environment()
    for key in ("python", "numpy", "pandas", "sklearn", "torch"):
        assert key in env
    assert env["python"].count(".") >= 1


def test_sha256_is_deterministic(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"vinfast-lhs-retrain")
    a = reporting.sha256_file(p)
    b = reporting.sha256_file(p)
    assert a == b and len(a) == 64


def test_dataset_audit_contract(tmp_path):
    ds = _build_dataset(tmp_path / "ds", n_samples=6)
    audit = reporting.build_dataset_audit(ds, split_counts={"train": 4, "val": 1, "test": 1})
    assert set(reporting.AUDIT_FILES) == set(audit["files"].keys())
    # generated_dataset.h5 exists in the synthetic dataset -> hashed.
    assert audit["files"]["generated_dataset.h5"]["sha256"]
    assert audit["split_counts"]["train"] == 4


def test_enrich_timing_adds_required_columns():
    df = pd.DataFrame(
        {"model": ["a", "b"], "inference_seconds_total": [0.1, 0.2]}
    )
    out = reporting.enrich_timing(df, "cpu", test_batch_size=64, test_rows=500)
    for col in ("device_name", "cuda_version", "gpu_name", "test_batch_size",
                "throughput_sequences_per_second"):
        assert col in out.columns
    # throughput = rows / inference_seconds_total.
    np.testing.assert_allclose(out["throughput_sequences_per_second"].iloc[0], 5000.0)
    assert (out["test_batch_size"] == 64).all()


# --------------------------------------------------------------------------- #
# Fixed-length resampling contract
# --------------------------------------------------------------------------- #
def test_fixed_length_resampling_shape_and_tail(tmp_path):
    ds = _build_dataset(tmp_path / "ds", n_samples=8, add_invalid=False)
    meta, X, Y, mask, q_grid, feats = elt.load_and_resample(
        ds, sequence_length=160, max_sample_ids=None, seed=42,
        min_valid_fraction=0.0, alignment_mode="official_clamped",
    )
    assert Y.shape[1:] == (160, 2)
    assert q_grid.shape == (160,)
    assert np.all(np.isfinite(Y))
    # official_clamped keeps the entire aligned sequence (endpoint-held tail).
    assert np.allclose(mask, 1.0)
    assert len(feats) == 13


def test_split_routine_is_group_disjoint(tmp_path):
    ids = [f"S{i:04d}" for i in range(1, 41)]
    split = elt.split_sample_ids(ids, 0.70, 0.15, 0.15, seed=42)
    s = {k: set(v) for k, v in split.items()}
    assert not (s["train"] & s["val"])
    assert not (s["train"] & s["test"])
    assert not (s["val"] & s["test"])
    assert sum(len(v) for v in s.values()) == len(ids)


# --------------------------------------------------------------------------- #
# Real-dataset split integrity (skipped if the dataset is absent)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_REAL, reason="regenerated dataset not present")
def test_regenerated_dataset_facts():
    m = pd.read_csv(DATA_DIR / "sequence_manifest.csv")
    m = m[m["simulation_status"] == "ok"]
    assert m["sample_id"].nunique() == 1000
    assert len(m) == 6000
    # exactly six cases per sample_id.
    assert (m.groupby("sample_id").size() == 6).all()


@pytest.mark.skipif(not (HAS_REAL and STORED_SPLIT.exists()),
                    reason="stored official split not present")
def test_stored_split_membership_and_no_leakage():
    split = json.loads(STORED_SPLIT.read_text())
    counts = {k: len(v) for k, v in split.items()}
    assert counts == {"train": 699, "val": 150, "test": 151}
    all_ids = set().union(*(set(v) for v in split.values()))
    assert len(all_ids) == 1000
    # pairwise disjoint.
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))


@pytest.mark.skipif(not (HAS_REAL and STORED_SPLIT.exists()),
                    reason="stored official split not present")
def test_all_six_cases_share_one_split():
    split = json.loads(STORED_SPLIT.read_text())
    lookup = {sid: name for name, ids in split.items() for sid in ids}
    m = pd.read_csv(DATA_DIR / "sequence_manifest.csv")
    m = m[m["simulation_status"] == "ok"].copy()
    m["split"] = m["sample_id"].astype(str).map(lookup)
    assert not m["split"].isna().any()
    # every sample_id maps all its rows to a single split.
    per_sample = m.groupby("sample_id")["split"].nunique()
    assert (per_sample == 1).all()


# --------------------------------------------------------------------------- #
# Error-metric feature/target construction
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HAS_REAL, reason="regenerated dataset not present")
def test_error_metric_features_and_targets():
    meta, X, Y, feats = emt.load_features_targets(DATA_DIR)
    assert len(feats) == 13
    assert X.shape[0] == Y.shape[0] == len(meta)
    assert Y.shape[1] == 2
    assert emt.TARGET_COLS == ["rmse_v_mV", "rmse_t_C"]
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))


@pytest.mark.skipif(not (HAS_REAL and STORED_SPLIT.exists()),
                    reason="stored official split not present")
def test_error_metric_split_assignment_no_leakage():
    meta, X, Y, feats = emt.load_features_targets(DATA_DIR)
    split = json.loads(STORED_SPLIT.read_text())
    idx = emt.assign_splits(meta, split)
    counts = emt.validate_no_leakage(meta, idx)
    assert counts["train_sample_ids"] == 699
    assert counts["val_sample_ids"] == 150
    assert counts["test_sample_ids"] == 151
