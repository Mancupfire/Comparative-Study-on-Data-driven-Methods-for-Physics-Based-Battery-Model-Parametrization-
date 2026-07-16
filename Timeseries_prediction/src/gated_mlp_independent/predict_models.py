"""Inference with the twelve independent Gated-MLP surrogates.

Reloads all twelve checkpoints and predicts the twelve error values
(``<condition>__<metric>``) for new parameter vectors supplied either as a CSV
of rows or as a single JSON / inline parameter vector.

Examples::

    # CSV of parameter vectors -> CSV of 12 predictions per row
    python -m src.gated_mlp_independent.predict_models \
        --models-dir ann_rmse_training_2500_physics_aligned/gated_mlp_12models_results \
        --input-csv new_params.csv --output-csv preds.csv

    # single vector as JSON (file or inline)
    python -m src.gated_mlp_independent.predict_models \
        --models-dir .../gated_mlp_12models_results \
        --vector '{"Positive electrode reference diffusivity [m2.s-1]": 1e-14, ...}'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.gated_mlp_independent.model import (  # noqa: E402
    GatedMLP,
    StandardScaler,
    TargetTransform,
)
from src.gated_mlp_independent import pipeline as P  # noqa: E402


class SurrogateEnsemble:
    """Holds the twelve loaded models + shared input scaler."""

    def __init__(self, models_dir: Path, device: str = "cpu"):
        import torch

        self.device = device
        self.models_dir = Path(models_dir)
        ckpt_dir = self.models_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("gated_mlp_*.pt"))
        if len(ckpts) != 12:
            raise RuntimeError(
                f"expected 12 checkpoints in {ckpt_dir}, found {len(ckpts)}")
        self.target_names: List[str] = []
        self.models = {}
        self.transforms: Dict[str, TargetTransform] = {}
        self.input_scaler = None
        for path in ckpts:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            name = ckpt["target_name"]
            model = GatedMLP(**ckpt["model_config"]).to(device)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.models[name] = model
            self.transforms[name] = TargetTransform.from_dict(ckpt["target_transform"])
            self.target_names.append(name)
            if self.input_scaler is None:
                self.input_scaler = StandardScaler.from_dict(ckpt["input_scaler"])
        self.target_names.sort()

    def predict_matrix(self, params: np.ndarray) -> pd.DataFrame:
        """params: [N, 12] in original units -> DataFrame [N, 12] predictions."""
        import torch

        x = self.input_scaler.transform(params)
        xt = torch.tensor(x, dtype=torch.float32, device=self.device)
        out = {}
        with torch.no_grad():
            for name in self.target_names:
                z = self.models[name](xt).cpu().numpy().reshape(-1)
                # RMSE is non-negative by definition -> clamp.
                out[name] = np.clip(self.transforms[name].inverse(z), 0.0, None)
        return pd.DataFrame(out)

    def predict_vector(self, vector: Dict[str, float]) -> Dict[str, float]:
        missing = [c for c in P.PARAM_COLUMNS if c not in vector]
        if missing:
            raise ValueError(f"missing parameters: {missing}")
        params = np.array([[float(vector[c]) for c in P.PARAM_COLUMNS]])
        df = self.predict_matrix(params)
        return {name: float(df[name].iloc[0]) for name in self.target_names}


def _load_vector(arg: str) -> Dict[str, float]:
    p = Path(arg)
    if p.exists():
        return json.loads(p.read_text())
    return json.loads(arg)


def run(args) -> None:
    ens = SurrogateEnsemble(args.models_dir, device=args.device)

    if args.vector is not None:
        preds = ens.predict_vector(_load_vector(args.vector))
        print(json.dumps({"predicted_errors": preds}, indent=2))
        if args.output_csv:
            pd.DataFrame([preds]).to_csv(args.output_csv, index=False)
        return

    if args.input_csv is not None:
        df = pd.read_csv(args.input_csv)
        missing = [c for c in P.PARAM_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"input CSV missing parameter columns: {missing}")
        params = df[P.PARAM_COLUMNS].to_numpy(dtype=np.float64)
        preds = ens.predict_matrix(params)
        result = pd.concat([df.reset_index(drop=True), preds], axis=1)
        out_path = args.output_csv or "predictions_new_params.csv"
        result.to_csv(out_path, index=False)
        print(f"Wrote {len(result)} rows x {len(ens.target_names)} predictions -> {out_path}")
        return

    raise SystemExit("Provide either --vector or --input-csv.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", required=True,
                   help="gated_mlp_12models_results directory")
    p.add_argument("--input-csv", help="CSV with the 12 parameter columns")
    p.add_argument("--vector", help="single parameter vector as JSON file or inline")
    p.add_argument("--output-csv", help="where to write predictions")
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
