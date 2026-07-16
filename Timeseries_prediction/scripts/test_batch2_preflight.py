"""Read-only unit tests for the Batch 2 isolation preflight.

No filesystem writes, no training. Verifies that:
* valid Batch 2 data dirs (raw / cleaned / downsampled variants) are accepted;
* any Batch 1 path is rejected;
* output/log isolation checks still reject Batch 1 roots and bare repo roots.

Run::

    python scripts/test_batch2_preflight.py        # plain asserts, exit 0 == pass
    pytest scripts/test_batch2_preflight.py         # also works if pytest present
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent

# Import the preflight module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "batch2_preflight", _HERE / "batch2_preflight.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _expect_fail(fn, *args) -> bool:
    try:
        fn(*args)
        return False
    except SystemExit:
        return True


def test_has_batch2_namespace_accepts_variants():
    for p in (
        "data/Data_Batch_2",
        "data/Data_Batch_2_cleaned",
        "data/Data_Batch_2_downsampled_160",
        "outputs/Data_Batch_2/time_series_downsampled_160/run1",
    ):
        assert pf.has_batch2_namespace(Path(p)), p


def test_has_batch2_namespace_rejects_non_batch2():
    for p in ("data/Data_Batch_1", "data/generate_training_data", "outputs/checkpoints"):
        assert not pf.has_batch2_namespace(Path(p)), p


def test_has_batch1_namespace_detects_batch1():
    assert pf.has_batch1_namespace(Path("data/Data_Batch_1"))
    assert pf.has_batch1_namespace(Path("data/Data_Batch_1_cleaned"))
    assert not pf.has_batch1_namespace(Path("data/Data_Batch_2_cleaned"))


def test_output_isolation_still_rejects_batch1_fragment():
    # _check_contains_batch2 must still reject a Batch 1 fragment in an output root.
    assert _expect_fail(
        pf._check_contains_batch2, "output-root",
        Path("outputs/Data_Batch_2/Data_Batch_1/run1"),
    )
    # ...and must reject a root with no Data_Batch_2 namespace at all.
    assert _expect_fail(
        pf._check_contains_batch2, "output-root", Path("outputs/checkpoints/run1")
    )
    # A correctly namespaced root passes.
    pf._check_contains_batch2("output-root", Path("outputs/Data_Batch_2/time_series/run1"))


def test_not_batch1_root_rejects_bare_repo_root():
    # Writing directly under outputs/ (Batch 1 territory) must abort.
    assert _expect_fail(
        pf._check_not_batch1_root, "output-root",
        Path("outputs/checkpoints/case/model"), _REPO,
    )
    # Under outputs/Data_Batch_2/... is allowed.
    pf._check_not_batch1_root(
        "output-root", Path("outputs/Data_Batch_2/time_series/run1"), _REPO
    )


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"[test] {passed}/{len(tests)} preflight tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
