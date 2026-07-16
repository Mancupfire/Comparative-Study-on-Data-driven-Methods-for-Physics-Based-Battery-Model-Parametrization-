"""Unified Batch-4 error-metric benchmark package.

Public surface:
    models   : 12-family registry + factory
    data     : grouped_holdout / legacy_reproduction dataset builder
    metrics  : physical-unit + scale-safe overall metrics
    trainer  : per-(family, seed) training & evaluation
    run      : CLI orchestration with resume
"""

from .models import ALL_FAMILIES, FAMILY_ORDER, DISPLAY_NAMES  # noqa: F401
