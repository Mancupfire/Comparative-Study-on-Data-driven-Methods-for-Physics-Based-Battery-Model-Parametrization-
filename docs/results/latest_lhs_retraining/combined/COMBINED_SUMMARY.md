# Combined LHS Retraining Summary

- Time-series run: `/data1/minhntn/nhatminh/VinFast/Timeseries_prediction/outputs/lhs_1000_seed42/time_series/lhs_official_full_20260717_100115`
- Error-metric run: `/data1/minhntn/nhatminh/VinFast/Timeseries_prediction/outputs/lhs_1000_seed42/error_metrics/lhs_error_metrics_full_20260717_101334`

## Dataset audit
- time_series: samples=1000 sequences=6000 failed=0 split={'train_sample_ids': 699, 'train_sequences': 4194, 'val_sample_ids': 150, 'val_sequences': 900, 'test_sample_ids': 151, 'test_sequences': 906}
- error_metrics: samples=1000 sequences=6000 failed=0 split={'train_sample_ids': 699, 'val_sample_ids': 150, 'test_sample_ids': 151, 'train_rows': 4194, 'val_rows': 900, 'test_rows': 906}

## Best models
- time_series: best model **CNN-BiLSTM** (avg rank 1.000)
- error_metrics: best model **Wide & Deep MLP** (avg rank 1.889)

## Artifacts
- combined_model_timing.csv
- combined_model_ranking.csv
- time_series_run.txt / error_metrics_run.txt
