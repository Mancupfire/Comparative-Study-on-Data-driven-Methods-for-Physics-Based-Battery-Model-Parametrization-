"""Validation tests for the twelve independent Gated-MLP surrogates.

A single fast smoke training run (few epochs) is executed once and shared by
all checks via a module-scoped fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.gated_mlp_independent import pipeline as P  # noqa: E402
from src.gated_mlp_independent import train_models as TM  # noqa: E402
from src.gated_mlp_independent.predict_models import SurrogateEnsemble  # noqa: E402

DATA_DIR = _REPO / "ann_rmse_training_2500_physics_aligned"


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("gmlp12")
    args = TM.build_parser().parse_args([
        "--data-dir", str(DATA_DIR), "--output-dir", str(out),
        "--device", "cpu", "--max-epochs", "6", "--patience", "6",
        "--seed", "42",
    ])
    TM.run(args)
    return out


def test_exactly_twelve_checkpoints(trained):
    ckpts = sorted((trained / "checkpoints").glob("gated_mlp_*.pt"))
    assert len(ckpts) == 12


def test_models_are_independent_instances(trained):
    import torch
    ckpts = sorted((trained / "checkpoints").glob("gated_mlp_*.pt"))
    first = torch.load(ckpts[0], map_location="cpu", weights_only=False)["state_dict"]
    second = torch.load(ckpts[1], map_location="cpu", weights_only=False)["state_dict"]
    key = "head.weight"
    assert not torch.allclose(first[key], second[key]), "two models share weights"


def test_no_group_overlap(trained):
    split = pd.read_csv(trained / "split_assignments.csv")
    groups = {s: set(split[split["split"] == s]["sample_id"]) for s in
              ("train", "val", "test")}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert sum(len(v) for v in groups.values()) == split["sample_id"].nunique()


def test_input_scaler_fit_on_train_only(trained):
    import json
    df = P.load_table(DATA_DIR)
    split = pd.read_csv(trained / "split_assignments.csv")
    train_ids = set(split[split["split"] == "train"]["sample_id"])
    train_rows = (df[df["sample_id"].isin(train_ids)]
                  .drop_duplicates("sample_id")[P.PARAM_COLUMNS].to_numpy())
    expected = TM.StandardScaler(P.PARAM_COLUMNS, P.LOG10_COLUMNS).fit(train_rows)
    saved = json.loads((trained / "preprocessing" / "input_scaler.json").read_text())
    assert np.allclose(saved["mean"], expected.mean, rtol=1e-5)
    assert np.allclose(saved["scale"], expected.scale, rtol=1e-5)


def test_predictions_finite_and_nonneg(trained):
    for split in ("train", "validation", "test"):
        df = pd.read_csv(trained / "predictions" / f"predictions_{split}.csv")
        assert np.isfinite(df["y_pred"]).all()
        assert (df["y_pred"] >= 0).all(), "predicted RMSE must be non-negative"


def test_no_target_column_in_inputs(trained):
    for bad in P.METRICS + ["rmse_v", "rmse_t"]:
        assert bad not in P.PARAM_COLUMNS


def test_inference_reloads_and_returns_twelve(trained):
    ens = SurrogateEnsemble(trained, device="cpu")
    assert len(ens.target_names) == 12
    df = P.load_table(DATA_DIR)
    row = df.iloc[0]
    vector = {c: float(row[c]) for c in P.PARAM_COLUMNS}
    preds = ens.predict_vector(vector)
    assert len(preds) == 12
    assert all(np.isfinite(v) and v >= 0 for v in preds.values())


def test_inverse_scaling_roundtrip():
    from src.gated_mlp_independent.model import TargetTransform
    y = np.array([0.5, 2.0, 7.0, 30.0])
    for log1p in (False, True):
        t = TargetTransform("x", use_log1p=log1p).fit(y)
        z = t.transform(y)
        assert np.allclose(t.inverse(z), y, rtol=1e-6, atol=1e-6)


def test_prediction_row_order_matches_source(trained):
    # predict_matrix must preserve input row order
    ens = SurrogateEnsemble(trained, device="cpu")
    df = P.load_table(DATA_DIR).drop_duplicates("sample_id").head(20)
    params = df[P.PARAM_COLUMNS].to_numpy()
    out = ens.predict_matrix(params)
    assert len(out) == len(df)
    # re-predicting a single row matches the batched result at that index
    single = ens.predict_matrix(params[5:6])
    name = ens.target_names[0]
    assert np.isclose(out[name].iloc[5], single[name].iloc[0])
