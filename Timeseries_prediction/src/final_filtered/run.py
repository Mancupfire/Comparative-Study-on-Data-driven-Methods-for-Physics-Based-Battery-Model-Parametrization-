"""CLI runner for the filtered time-series final protocol.

Trains every requested (case, model, seed) under one run dir with safe resume
(already-complete combinations are skipped).  Writes ONLY under the isolated
filtered namespace.

Example
-------
    python -m src.final_filtered.run \
        --run-dir outputs/Data_Batch_4_TSFiltered_0p8/time_series/run1 \
        --models ann rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
        --seeds 42 43 44 --epochs 300
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import save_json  # noqa: E402

from . import models as M  # noqa: E402
from .data import discover_filtered_cases  # noqa: E402
from .train import is_complete, train_and_eval  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Filtered time-series final protocol")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--models", nargs="+", default=M.FINAL_TS_MODELS,
                   choices=M.FINAL_TS_MODELS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--ann-hidden-dim", type=int, default=128)
    p.add_argument("--lambda-temp", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--mc-samples", type=int, default=30)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = a.cases or discover_filtered_cases()
    print(f"[run] cases={len(cases)} models={a.models} seeds={a.seeds}")
    print(f"[run] run_dir={run_dir}")

    failures, done, skipped = [], 0, 0
    for case_id in cases:
        for model_name in a.models:
            for seed in a.seeds:
                tag = f"{case_id}/{model_name}/seed{seed}"
                if is_complete(run_dir / f"seed{seed}", case_id, model_name):
                    skipped += 1
                    print(f"[skip] {tag} already complete")
                    continue
                try:
                    train_and_eval(
                        case_id, model_name,
                        run_dir=run_dir / f"seed{seed}", seed=seed,
                        epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                        weight_decay=a.weight_decay, hidden_dim=a.hidden_dim,
                        num_layers=a.num_layers, dropout=a.dropout,
                        ann_hidden_dim=a.ann_hidden_dim, lambda_temp=a.lambda_temp,
                        patience=a.patience, device=a.device, mc_samples=a.mc_samples,
                    )
                    done += 1
                except Exception as exc:  # noqa: BLE001
                    failures.append({"combo": tag, "error": repr(exc)})
                    print(f"[FAIL] {tag}: {exc}")
                    traceback.print_exc()

    save_json({
        "run_dir": str(run_dir), "cases": cases, "models": a.models,
        "seeds": a.seeds, "n_done": done, "n_skipped": skipped,
        "n_failures": len(failures), "failures": failures,
        "protocol": "filtered_grouped_masked",
    }, run_dir / "run_manifest.json")
    print(f"[run] complete. done={done} skipped={skipped} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
