"""Independent scalar Gated-MLP RMSE surrogates.

Twelve fully independent single-output Gated-MLP regressors, one per
(discharge condition x error metric) target, for the
``ann_rmse_training_2500_physics_aligned`` dataset.

Each model has its own weights, optimizer, target scaler, early-stopping,
checkpoint, metrics and prediction files.  The shared input scaler and the
leakage-safe grouped train/val/test split (grouped by ``sample_id``) are
identical across all models.
"""
