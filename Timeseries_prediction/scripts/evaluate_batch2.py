"""Standalone (re-)evaluation of every trained Batch 2 (case, model) in a run dir.

The training driver already evaluates each model on the test split, but this
provides an idempotent re-evaluation entry point (e.g. after a resume) that
re-uses the exact saved split via run_config.json.

Usage
-----
python scripts/evaluate_batch2.py \
    --data-root data/Data_Batch_2_cleaned \
    --run-dir outputs/Data_Batch_2/time_series/<run_id> \
    --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
    --device auto
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import discover_cases
from src.evaluate import evaluate_case


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--models", nargs="+",
                   default=["mlp", "rnn", "lstm", "bilstm", "cnn", "cnn_bilstm", "bayesian_mlp"])
    p.add_argument("--cases", nargs="+", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--mc-samples", type=int, default=30)
    args = p.parse_args()

    cases = args.cases or discover_cases(args.data_root)
    failures, done = [], 0
    for cid in cases:
        for m in args.models:
            ckpt = Path(args.run_dir) / "checkpoints" / cid / m / "best_model.pt"
            if not ckpt.is_file():
                print(f"[eval] skip {cid}/{m}: no checkpoint")
                continue
            try:
                evaluate_case(data_root=args.data_root, case_id=cid, model_name=m,
                              outputs_dir=args.run_dir, checkpoint_name="best_model.pt",
                              split="test", device=args.device, mc_samples=args.mc_samples)
                done += 1
            except Exception as exc:  # noqa: BLE001
                failures.append((cid, m, str(exc)))
                print(f"!! eval FAILED {cid}/{m}: {exc}")
                traceback.print_exc()

    print(f"\n[eval] evaluated {done} model(s); {len(failures)} failure(s)")
    for cid, m, e in failures:
        print(f"  - {cid}/{m}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
