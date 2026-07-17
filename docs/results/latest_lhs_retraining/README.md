# LHS Battery Modelling — Latest Retraining Progress

## Status

The complete retraining workflow has been validated for both experimental setups:

1. **Time-series response prediction**
2. **Error-metric prediction**

## Dataset

- Latin Hypercube Sampling with seed 42
- 1,000 unique physical parameter sets
- 6 experiment cases
- 6,000 successful sequences
- Group-aware train/validation/test split by `sample_id`
- No `sample_id` leakage between splits
- Experimental-grid alignment protocol

## Model families

### Time-series response prediction

- MLP
- RNN
- LSTM
- BiLSTM
- CNN
- CNN-BiLSTM
- Bayesian MLP

### Error-metric prediction

- ANN
- MLP
- Wide & Deep MLP
- Gated MLP
- Deep Ensemble MLP
- Random Forest
- ExtraTrees
- XGBoost
- CatBoost

## Verification

- New LHS unit tests: **19 passed**
- Smoke workflow: **passed**
- Model save/reload/inference: **passed for all 16 model families**
- Metrics, timing fields and figures: **verified**
- Training time and inference time are recorded for model comparison

## Results

- [Combined summary](combined/COMBINED_SUMMARY.md)
- [Combined ranking](combined/combined_model_ranking.csv)
- [Combined timing](combined/combined_model_timing.csv)
- [Time-series metrics](time_series/metrics/)
- [Time-series figures](time_series/figures/)
- [Error-metric metrics](error_metrics/metrics/)
- [Error-metric figures](error_metrics/figures/)

## Reproducibility

The repository contains the training scripts, model implementations, grouped
split logic, evaluation pipeline, runtime tracking and retraining wrappers.
Large datasets and trained checkpoints are distributed separately rather than
committed directly to Git.
