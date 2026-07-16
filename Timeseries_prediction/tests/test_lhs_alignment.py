"""Focused tests for the LHS capacity-alignment pipeline.

These build a tiny synthetic dataset (same file contract as
``data/lhs_1000_seed42``) so the two alignment modes, the endpoint-held tail,
the structural exclusions, the sample_id split and the diagnostic/response
plots can be checked deterministically and fast — no GPU, no real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import emergency_lhs_train as elt  # noqa: E402

# Two synthetic cases; grid length is constant within a case, differs across.
CASES = {
    "cc_chg_test": {"operation": "charge", "c_rate": 2.0, "L": 50, "exp_end": 100.0},
    "cc_dchg_test": {"operation": "discharge", "c_rate": 2.0, "L": 40, "exp_end": 80.0},
}
PARAM_COLS = [f"physical__p{i}" for i in range(10)]


def _curve(length: int, hi: float, lo: float) -> np.ndarray:
    return np.linspace(hi, lo, length)


def _build_dataset(tmp: Path, n_samples: int = 12, add_invalid: bool = True) -> Path:
    """Write manifest, params and h5 into ``tmp``; return the dataset dir."""
    tmp.mkdir(parents=True, exist_ok=True)
    sample_ids = [f"S{i:04d}" for i in range(1, n_samples + 1)]

    params = pd.DataFrame({"sample_id": sample_ids})
    rng = np.random.default_rng(0)
    for c in PARAM_COLS:
        params[c] = rng.uniform(0.1, 1.0, size=n_samples)

    a_cap, a_sv, a_st, a_ev, a_et = [], [], [], [], []
    r_cap, r_v, r_t = [], [], []
    rows = []
    a_off = r_off = 0

    for exp, spec in CASES.items():
        L = spec["L"]
        exp_end = spec["exp_end"]
        grid = np.linspace(0.0, exp_end, L)
        exp_v = _curve(L, 4.2, 3.0)
        exp_t = _curve(L, 25.0, 40.0)
        for si, sid in enumerate(sample_ids):
            # Every third sample terminates early -> endpoint-held flat tail.
            early = (si % 3 == 0)
            cutoff = L // 2 if early else L
            sim_v = exp_v.copy()
            sim_t = exp_t.copy()
            if early:
                sim_v[cutoff:] = sim_v[cutoff - 1]      # held voltage tail
                sim_t[cutoff:] = sim_t[cutoff - 1]      # held temperature tail
            sim_end_cap = grid[cutoff - 1]
            frac = (sim_end_cap - exp_end) / exp_end

            raw_len = 20
            raw_capacity = np.linspace(0.0, sim_end_cap, raw_len)
            rows.append({
                "sequence_id": f"{sid}__{exp}",
                "sample_id": sid,
                "experiment_id": exp,
                "operation": spec["operation"],
                "c_rate": spec["c_rate"],
                "initial_temperature_C": 25.0,
                "simulation_status": "ok",
                "aligned_offset": a_off,
                "aligned_length": L,
                "raw_offset": r_off,
                "raw_length": raw_len,
                "experimental_end_capacity_Ah": exp_end,
                "simulated_end_capacity_Ah": sim_end_cap,
                "end_capacity_error_fraction": frac,
            })
            a_cap.append(grid); a_sv.append(sim_v); a_st.append(sim_t)
            a_ev.append(exp_v); a_et.append(exp_t)
            r_cap.append(raw_capacity)
            r_v.append(_curve(raw_len, 4.2, sim_v[cutoff - 1]))
            r_t.append(_curve(raw_len, 25.0, sim_t[cutoff - 1]))
            a_off += L
            r_off += raw_len

    if add_invalid:
        # A structurally invalid sequence: one raw point only.
        sid = sample_ids[0]
        L = 10
        grid = np.linspace(0.0, 50.0, L)
        rows.append({
            "sequence_id": f"{sid}__cc_chg_bad",
            "sample_id": sid,
            "experiment_id": "cc_chg_bad",
            "operation": "charge",
            "c_rate": 2.0,
            "initial_temperature_C": 25.0,
            "simulation_status": "ok",
            "aligned_offset": a_off,
            "aligned_length": L,
            "raw_offset": r_off,
            "raw_length": 1,
            "experimental_end_capacity_Ah": 50.0,
            "simulated_end_capacity_Ah": 5.0,
            "end_capacity_error_fraction": -0.9,
        })
        a_cap.append(grid); a_sv.append(_curve(L, 4.2, 3.0)); a_st.append(_curve(L, 25.0, 40.0))
        a_ev.append(_curve(L, 4.2, 3.0)); a_et.append(_curve(L, 25.0, 40.0))
        r_cap.append(np.array([0.0])); r_v.append(np.array([4.2])); r_t.append(np.array([25.0]))
        a_off += L
        r_off += 1

    manifest = pd.DataFrame(rows)
    manifest.to_csv(tmp / "sequence_manifest.csv", index=False)
    params.to_csv(tmp / "parameter_sets_physical.csv", index=False)

    with h5py.File(tmp / "generated_dataset.h5", "w") as h5:
        h5.create_dataset("aligned/experimental_capacity_Ah", data=np.concatenate(a_cap))
        h5.create_dataset("aligned/simulated_voltage_V", data=np.concatenate(a_sv))
        h5.create_dataset("aligned/simulated_temperature_C", data=np.concatenate(a_st))
        h5.create_dataset("aligned/experimental_voltage_V", data=np.concatenate(a_ev))
        h5.create_dataset("aligned/experimental_temperature_C", data=np.concatenate(a_et))
        h5.create_dataset("raw/capacity_Ah", data=np.concatenate(r_cap))
        h5.create_dataset("raw/voltage_V", data=np.concatenate(r_v))
        h5.create_dataset("raw/temperature_C", data=np.concatenate(r_t))
    return tmp


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory) -> Path:
    return _build_dataset(tmp_path_factory.mktemp("lhs_synth"))


SEQ_LEN = 32


def _load(dataset_dir: Path, mode: str, excluded: Path | None = None):
    return elt.load_and_resample(
        dataset_dir, SEQ_LEN, max_sample_ids=None, seed=42,
        min_valid_fraction=0.0, alignment_mode=mode, excluded_csv=excluded,
    )


def test_shared_grid_consistency_within_case(dataset_dir):
    """Every sample of a case shares one physical capacity grid."""
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    assert q_grid.shape == (SEQ_LEN,)
    assert np.allclose(q_grid, np.linspace(0.0, 1.0, SEQ_LEN))
    for exp, g in meta.groupby("experiment_id"):
        ends = g["experimental_end_capacity_Ah"].round(6).nunique()
        assert ends == 1, f"{exp} has non-unique grid end"
        # Physical grid = q_grid * exp_end is therefore identical for the case.


def test_endpoint_held_flat_tail(dataset_dir):
    """An early-terminating sequence keeps a flat voltage/temperature tail."""
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    early = meta.index[meta["end_capacity_error_fraction"] < -1e-6].tolist()
    assert early, "fixture must contain early-terminating sequences"
    i = early[0]
    # Last quarter of the grid lies in the held tail -> near-constant values.
    tail = Y[i, -SEQ_LEN // 4:, 0]
    assert np.ptp(tail) < 1e-4, "voltage tail is not flat/held"
    tail_t = Y[i, -SEQ_LEN // 4:, 1]
    assert np.ptp(tail_t) < 1e-4, "temperature tail is not flat/held"


def test_official_clamped_uses_full_sequence(dataset_dir):
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    assert mask.shape == (len(meta), SEQ_LEN)
    assert np.all(mask == 1.0), "official_clamped must mark the whole sequence valid"
    assert np.allclose(meta["valid_fraction"].to_numpy(), 1.0)


def test_masked_mode_excludes_tail(dataset_dir):
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "masked")
    early = meta.index[meta["end_capacity_error_fraction"] < -1e-6].tolist()
    i = early[0]
    # The held tail must be masked out -> strictly fewer valid points than full.
    assert mask[i].sum() < SEQ_LEN
    assert mask[i, 0] == 1.0                       # first point always retained
    assert mask[i, -1] == 0.0                      # tail excluded
    # And official_clamped keeps strictly more valid points on the same seq.
    _, _, _, mask_off, _, _ = _load(dataset_dir, "official_clamped")
    assert mask_off[i].sum() > mask[i].sum()


def test_structural_exclusion_recorded(dataset_dir, tmp_path):
    excluded = tmp_path / "excluded_sequences.csv"
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped", excluded)
    assert excluded.exists()
    ex = pd.read_csv(excluded)
    assert "cc_chg_bad" in set(ex["experiment_id"])
    assert (ex["reason"] == "fewer_than_two_raw_points").any()
    # Excluded sequence must not survive into the training arrays.
    assert "cc_chg_bad" not in set(meta["experiment_id"])


def test_no_sample_id_leakage(dataset_dir):
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    split = elt.split_sample_ids(meta["sample_id"].tolist(), 0.7, 0.15, 0.15, 42)
    counts = elt.validate_split(meta, split)          # raises on leakage
    tr, va, te = set(split["train"]), set(split["val"]), set(split["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert counts["train_sample_ids"] > 0


def test_response_plot_skips_invalid_and_one_point(dataset_dir, tmp_path):
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    # Real (mixed) masks -> a plot with populated panels is produced.
    out = tmp_path / "response.png"
    elt.plot_response_examples(q_grid, Y, Y, mask, meta, out, seed=1)
    assert out.exists() and out.stat().st_size > 1000
    # If NO sequence has >=2 valid points, the plot is skipped (no blank panel).
    empty_mask = np.zeros_like(mask)
    empty_mask[:, 0] = 1.0                            # one-point sequences only
    out2 = tmp_path / "response_empty.png"
    elt.plot_response_examples(q_grid, Y, Y, empty_mask, meta, out2, seed=1)
    assert not out2.exists()


def test_preprocessing_diagnostic_plot(dataset_dir, tmp_path):
    meta, X, Y, mask, q_grid, feats = _load(dataset_dir, "official_clamped")
    out = tmp_path / "diagnostic.png"
    elt.plot_preprocessing_diagnostic(dataset_dir, meta, SEQ_LEN, out)
    assert out.exists() and out.stat().st_size > 1000
