# Data Generation

This folder generates two model-agnostic ML training datasets from the existing
PyBaMM physics workflow in `Multi_optimization_parallel/Scripts`.

It deliberately does not implement optimization or ML training. The only
conceptual replacement is:

```text
iL-SHADE candidate generation -> Latin Hypercube Sampling
```

Everything else is kept compatible with `parameter_optimization.py`:

- same parameter bounds and log10 transformed optimizer space
- same optimizer-to-physical parameter conversion
- same PyBaMM simulation construction
- same experiment protocols
- same experimental CSV loading
- same target-step extraction
- same `align_sim_to_exp_capacity` / `align_sim_to_exp_time`
- same `rmse` formula

The primary scalar targets in `error_metrics_by_case.csv` are:

- `rmse_v_mV`
- `rmse_t_C`

These are intended to match `evaluate_one_case(...)` within floating-point
tolerance for the same physical parameter vector and experiment case.

## Outputs

One command creates:

```text
Data_generation/outputs/<dataset_name>/
|-- generated_dataset.h5
|-- parameter_sets_optimizer.csv
|-- parameter_sets_physical.csv
|-- sequence_manifest.csv
|-- failed_cases.csv
|-- error_metrics_by_case.csv
|-- error_metrics_wide.csv
|-- generation_summary.json
`-- analysis/
    |-- case_summary.csv
    |-- rmse_distributions.png
    `-- alignment_examples/
```

`generated_dataset.h5` stores:

- aligned experimental coordinate and aligned voltage/temperature curves
- optimizer-space and physical-space parameter vectors
- raw simulation outputs by default, using flattened arrays plus offsets

Default `--grid-mode experimental_grid` stores aligned curves at the experimental
points used by the optimization code. This is the compatibility mode. Optional
`--grid-mode uniform_capacity` stores fixed-length resampled curves for quick
inspection, but RMSE is still calculated from the experimental-grid alignment.

## Smoke Example

From `F:\Parameter_Estimation\Scripts_Gotion`:

```powershell
..\ .venv\Scripts\python.exe -m Data_generation.generate ^
    --n-samples 2 ^
    --seed 42 ^
    --sampling-mode lhs ^
    --grid-mode experimental_grid ^
    --case-ids cc_dchg_2C_25degC ^
    --n-workers 1 ^
    --solver-threads 1 ^
    --task-batch-size 2 ^
    --flush-every 1 ^
    --output-dir Data_generation\outputs\smoke_2_seed42 ^
    --overwrite
```

Remove the space in `..\ .venv` when running the command:
`..\.venv\Scripts\python.exe`.

## Full Example

```powershell
python -m Data_generation.generate ^
    --n-samples 5000 ^
    --seed 42 ^
    --sampling-mode lhs ^
    --grid-mode experimental_grid ^
    --n-workers 48 ^
    --solver-threads 1 ^
    --task-batch-size 100 ^
    --flush-every 25 ^
    --output-dir Data_generation\outputs\lhs_5000_seed42 ^
    --overwrite
```

## Analysis

```powershell
python -m Data_generation.analyze Data_generation\outputs\lhs_5000_seed42
```

## Resume

Use `--resume` to skip sequence IDs already found in `sequence_manifest.csv` or
`failed_cases.csv`. Use either `--resume` or `--overwrite`, never both.


## Latest LHS Retraining Results

The latest validated results for both the time-series response and error-metric
prediction setups are available here:

- [Retraining progress, metrics, timing and figures](docs/results/latest_lhs_retraining/README.md)

The current workflow covers 7 time-series model families and 9 error-metric
model families using a common group-aware split by `sample_id`.
