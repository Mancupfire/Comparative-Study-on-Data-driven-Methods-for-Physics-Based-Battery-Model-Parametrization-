# Battery Simulation Time-Series Surrogate Models

A clean, production-ready PyTorch training system for learning battery
simulation **surrogate / meta-models**:

```
M : x(i)  ->  y(i, t)
```

* `x(i)` — one LHS-sampled parameter vector (a row of `parameter_sets.csv`).
* `y(i, t)` — the simulated response time-series.
* Current outputs: **voltage** and **temperature**. Aging is **not** modelled.

The system trains **one model per case** and supports four architectures
(`mlp`, `rnn`, `lstm`, `bilstm`) plus an optional sklearn tree-ensemble baseline.

---

## 1. Project goal

Replace expensive physics simulations with fast neural surrogates that predict
the full voltage and temperature curves from the input parameter vector, per
operating case (e.g. discharge C-rate × ambient temperature).

## 2. Dataset format

```
generate_training_data/
├── parameter_sets.csv          # X: one LHS parameter set per row (indexed by sample_id)
├── dataset_summary.json
├── failed_cases.csv            # simulations that failed (per experiment)
├── sequence_manifest.csv
└── cases/<case_id>/outputs.npz # aligned simulation outputs for that case
```

Each `cases/<case_id>/outputs.npz` contains:

| key             | shape              | meaning                          |
|-----------------|--------------------|----------------------------------|
| `sample_ids`    | `[N_ok]`           | ids of **successful** samples    |
| `time_s`        | `[t_last]`         | shared time vector (seconds)     |
| `voltage_v`     | `[N_ok, t_last]`   | voltage curves                   |
| `temperature_c` | `[N_ok, t_last]`   | temperature curves               |

### Why `sample_ids` must be used for alignment

Not every parameter row simulates successfully (`failed_cases.csv`), so
`N_ok ≤ N`. The **only** reliable alignment is via `sample_ids`:

```python
params = pd.read_csv("parameter_sets.csv").set_index("sample_id")
npz    = np.load("cases/<case_id>/outputs.npz")
X      = params.loc[npz["sample_ids"]]   # reorders X to match the npz rows exactly
V, T   = npz["voltage_v"], npz["temperature_c"]
```

Indexing by position (`params.iloc[:N_ok]`) would silently misalign inputs and
outputs. `src/data.load_aligned_case_data` performs this alignment and raises a
clear error if any simulated `sample_id` is missing from `parameter_sets.csv`.
`t_last` is read from the data and **varies per case** — nothing is hard-coded.

## 3. Modeling strategy

* **One model per case.** Each case has its own scalers, checkpoints and metrics.
* **MLP (`mlp`)** predicts both **full curves at once**:
  input `[B, n_parameters]` → output `[B, 2*t_last]` = `concat(voltage, temperature)`.
* **RNN / LSTM / BiLSTM** predict voltage & temperature **at each time step**:
  input `[B, t_last, n_parameters + 1]` (parameters broadcast over time **plus a
  normalized-time channel**) → output `[B, t_last, 2]` with channels `[V, T]`.
* **AutoML / sklearn baseline** (`scripts/train_automl_baseline.py`) is a
  *separate reference baseline only*. It maps parameters → outputs with an
  ExtraTrees/RandomForest multi-output regressor. **It does not automatically
  include the RNN/LSTM/BiLSTM sequence models** — those require a specialised
  AutoML system. Use `--reduced-output` to predict 10 summary scalars
  (`V_start, V_mid, V_end, V_min, V_mean, T_start, T_mid, T_end, T_max, T_mean`)
  instead of full curves when `t_last` is large.

### Preprocessing (no data leakage)

* `StandardScaler` for X; voltage and temperature scaled **separately**.
* Scalers are **fit on the train split only**, then applied to val/test.
* Sample-wise split (`train 70% / val 15% / test 15%`) with a fixed seed; an
  entire curve always stays inside one split.
* Metrics are computed on **inverse-transformed (physical) predictions**.
* Saved scalers:
  `outputs/scalers/<case_id>/<model_name>/{x,v,t}_scaler.joblib`.

## 4. Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x (CPU or CUDA — GPU is **not** required).

## 5. Inspect the dataset

```bash
python scripts/inspect_dataset.py --data-root generate_training_data
```

Prints, per case: `case_id`, number of samples, time steps, voltage/temperature
min-max, failed count, and validates `sample_id` alignment and array shapes.

## 6. Train one model

```bash
python scripts/train_one_case.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model mlp --epochs 300 --batch-size 64 --lr 1e-3
```

Trains, early-stops on validation loss, then evaluates the best checkpoint on
the test split. Per-epoch logs show `epoch, train_loss, val_loss, loss_v,
loss_t, lr`.

Key flags (defaults in brackets): `--model {mlp,rnn,lstm,bilstm,cnn,cnn_bilstm,bayesian_mlp}`,
`--epochs 300`, `--batch-size 64`, `--lr 1e-3`, `--weight-decay 1e-4`,
`--hidden-dim 256`, `--num-layers 2`, `--dropout 0.1`, `--lambda-temp 1.0`,
`--patience 30`, `--seed 42`, `--device auto`,
`--train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15`.

## 7. Train all models on all cases

```bash
python scripts/train_all_cases.py --data-root generate_training_data \
    --models mlp rnn lstm bilstm --epochs 300 --batch-size 64 --lr 1e-3
```

Discovers all cases under `generate_training_data/cases/` and trains every
requested model on each. A failure on one (case, model) is reported and does
not stop the rest.

## Additional time-series models

Three extra architectures are available as **optional** choices on top of the
original `mlp / rnn / lstm / bilstm` (and their `shared_*` counterparts). They
are fully additive: every existing command keeps working unchanged.

- **CNN** (`cnn`, `shared_cnn`) — a 1-D temporal convolutional network. Stacked
  `Conv1d` blocks (kernel sizes 5/5/3) learn *local* temporal patterns in the
  voltage/temperature curves, followed by a `1x1` conv projecting to `[V, T]`.
  Same sequence I/O as the recurrent models (`[B, T, n_params+1]` per case).
- **CNN-BiLSTM** (`cnn_bilstm`, `shared_cnn_bilstm`) — a `Conv1d` front-end
  feeds a bidirectional LSTM: the CNN captures local features while the BiLSTM
  models long-range temporal dependencies. A final `Linear(2*hidden -> 2)`
  emits `[V, T]` per step.
- **Bayesian MLP / MC-Dropout** (`bayesian_mlp`, `shared_bayesian_mlp`) — an
  approximate Bayesian network using **Monte-Carlo Dropout** (no extra
  probabilistic framework). It trains with ordinary MSE, but at evaluation time
  dropout is kept active and the model is run `--mc-samples` times (default 30).
  The standard metrics use the prediction **mean**; an additional
  `outputs/metrics/<case_id>/bayesian_mlp/uncertainty_summary.json` records the
  predictive std (`mean_std_V/T`, `max_std_V/T`, `mean_std_V/T_end`). The shared
  variant writes the same under `outputs/metrics/shared/shared_bayesian_mlp/`.

Per-case training:

```bash
# CNN
python scripts/train_one_case.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model cnn \
    --epochs 300 --batch-size 64 --lr 1e-3 --seed 42 --device auto

# CNN-BiLSTM
python scripts/train_one_case.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model cnn_bilstm \
    --epochs 300 --batch-size 64 --lr 1e-3 --seed 42 --device auto

# Bayesian MLP (MC-Dropout)
python scripts/train_one_case.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model bayesian_mlp \
    --epochs 300 --batch-size 64 --lr 1e-3 --seed 42 --device auto --mc-samples 30
```

Train all per-case models (new ones included):

```bash
python scripts/train_all_cases.py --data-root generate_training_data \
    --models mlp rnn lstm bilstm cnn cnn_bilstm bayesian_mlp \
    --epochs 300 --batch-size 64 --lr 1e-3 --seed 42 --device auto
```

Shared (all-cases) training:

```bash
python scripts/train_shared_model.py --data-root generate_training_data \
    --model shared_cnn --epochs 300 --batch-size 64 --lr 1e-3 --seed 42 --device auto

python scripts/train_shared_model.py --data-root generate_training_data \
    --model shared_cnn_bilstm --epochs 300 --batch-size 64 --lr 1e-3 \
    --num-layers 2 --seed 42 --device auto

python scripts/train_shared_model.py --data-root generate_training_data \
    --model shared_bayesian_mlp --epochs 300 --batch-size 8192 --lr 1e-3 \
    --max-points-per-curve 80 --seed 42 --device auto --mc-samples 30

# Or all shared models at once:
python scripts/train_all_shared_models.py --data-root generate_training_data \
    --models shared_mlp shared_rnn shared_lstm shared_bilstm \
             shared_cnn shared_cnn_bilstm shared_bayesian_mlp \
    --epochs 300 --lr 1e-3 --seed 42 --device auto
```

Note on `shared_cnn` / `shared_cnn_bilstm`: cases have different lengths, so
sequences are padded. The convolution does see padded steps (Conv1d has no
length masking), but the **loss and metrics are masked** so padded positions
never affect training or evaluation. `shared_cnn_bilstm` additionally uses
`pack_padded_sequence` for its LSTM so padding never enters the recurrence.

`compare_models.py` / `compare_shared_models.py` and the summary-figure scripts
pick up the new models automatically once their metrics exist.

## 8. Compare models

```bash
python scripts/compare_models.py --outputs-dir outputs
```

Aggregates every `metrics.json` into `outputs/model_comparison.csv` and prints a
ranking sorted by `RMSE_V + RMSE_T` (lower is better).

## 9. Plot predictions

```bash
python scripts/plot_predictions.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --model mlp --num-samples 5
```

Recreates the same test split, predicts on random test samples and saves one
voltage and one temperature plot per sample to
`outputs/figures/<case_id>/<model_name>/`.

## 10. Optional: sklearn baseline

```bash
python scripts/train_automl_baseline.py --data-root generate_training_data \
    --case-id cc_dchg_1C_25degC --estimator extratrees --reduced-output
```

## 11. Shared models across all cases

Everything above trains **one model per case**. The shared pipeline instead
trains **one model across all 6 cases at once**, conditioned on the operating
point. This is a separate, additive pipeline — it does **not** touch or retrain
the per-case models.

### Why a shared model needs a different formulation

Each case has a different number of time steps (`t_last` ranges from 147 to
178), so the full curves cannot simply be concatenated. The shared model is
therefore **point-wise / conditional**:

```
f_theta([parameter_vector, c_rate, ambient_temp_C, time_norm]) -> [V(t), T(t)]
```

* `parameter_vector` — the LHS battery parameters (from `parameter_sets.csv`,
  aligned through `sample_ids`).
* `c_rate`, `ambient_temp_C` — operating-condition features parsed from the
  `case_id` (e.g. `cc_dchg_1C_25degC` → `c_rate=1.0`, `ambient_temp_C=25`).
* `time_norm = time_s / max(time_s)` ∈ `[0, 1]` — the key feature that lets a
  single model span cases of **different length**, since every curve is
  expressed on the same normalized time axis.
* Output is `[V(t), T(t)]` at that time.

### `shared_mlp` vs `shared_rnn` / `shared_lstm` / `shared_bilstm`

* **`shared_mlp`** is **point-wise**: every `(sample, timestep)` is an
  independent example with input `[n_parameters + 3]` → output `[2]`. Because
  examples are independent, very large batch sizes (4096–8192) are efficient, and
  `--max-points-per-curve` can subsample timesteps per curve (the first and last
  step are always kept).
* **`shared_rnn` / `shared_lstm` / `shared_bilstm`** are **sequence models**:
  each curve is one variable-length sequence `[T, n_parameters + 3]` → `[T, 2]`,
  where the parameters and conditions are repeated across timesteps and only
  `time_norm` changes. Variable lengths are handled with **padding + a boolean
  mask** in a custom `collate_fn`; sequences are packed (`pack_padded_sequence`)
  so padded steps never affect the recurrence, and the **loss is computed only on
  valid (unmasked) timesteps**. Use small batch sizes (32–64).

### No leakage / preprocessing

* The train/val/test split (70/15/15, fixed seed) is performed at the
  **curve level** (one `sample_id + case_id` pair = one curve); a curve never
  straddles two splits.
* `StandardScaler` is fit on the **train split only** for X features, voltage and
  temperature separately, and saved to
  `outputs/scalers/shared/<model_name>/{x,v,t}_scaler.joblib`.
* Metrics are computed on **inverse-transformed (physical) predictions** and are
  reported overall and grouped by `case_id`, `ambient_temp_C` and `c_rate`.

### Commands

Inspect current cases:

```bash
python scripts/inspect_dataset.py --data-root generate_training_data
```

Train one shared RNN:

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_shared_model.py \
  --data-root generate_training_data \
  --model shared_rnn \
  --epochs 300 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42 \
  --device auto
```

Train one shared MLP (point-wise; large batch + subsampled timesteps):

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_shared_model.py \
  --data-root generate_training_data \
  --model shared_mlp \
  --epochs 300 \
  --batch-size 8192 \
  --lr 1e-3 \
  --max-points-per-curve 80 \
  --seed 42 \
  --device auto
```

Train all shared models (per-mode batch sizes are chosen automatically:
`--mlp-batch-size 8192`, `--sequence-batch-size 64`):

```bash
CUDA_VISIBLE_DEVICES=7 python scripts/train_all_shared_models.py \
  --data-root generate_training_data \
  --models shared_mlp shared_rnn shared_lstm shared_bilstm \
  --epochs 300 \
  --lr 1e-3 \
  --seed 42 \
  --device auto
```

Compare shared models:

```bash
python scripts/compare_shared_models.py --outputs-dir outputs
```

Plot a shared-model prediction summary for one case (reconstructs the global
test split, then keeps only that case's test curves):

```bash
python scripts/plot_shared_summary_figure.py \
  --data-root generate_training_data \
  --case-id cc_dchg_1C_25degC \
  --model shared_rnn \
  --num-curves 80 \
  --device auto
```

### Shared-model outputs

```
outputs/
├── checkpoints/shared/<model_name>/{best_model.pt, final_model.pt}
├── scalers/shared/<model_name>/{x_scaler,v_scaler,t_scaler}.joblib
├── metrics/shared/<model_name>/{history.csv, run_config.json, metrics.json,
│                                metrics_by_case.csv, metrics_by_temp.csv,
│                                metrics_by_c_rate.csv}
├── figures/shared/<model_name>/<case_id>/summary_figure.{png,pdf}
└── shared_model_comparison_{overall,by_case,by_temp,by_c_rate}.csv
```

## Metrics

Per case/model (computed in physical units on the test split):

* **Voltage:** `MAE_V, RMSE_V, R2_V, MaxError_V`
* **Temperature:** `MAE_T, RMSE_T, R2_T, MaxError_T`
* **Curve-specific:** `voltage_end_mae, temperature_end_mae,
  temperature_peak_mae, voltage_curve_rmse_mean, temperature_curve_rmse_mean`

Saved to `outputs/metrics/<case_id>/<model_name>/metrics.json`.

## Expected outputs

```
outputs/
├── checkpoints/<case_id>/<model_name>/{best_model.pt, final_model.pt}
├── scalers/<case_id>/<model_name>/{x_scaler,v_scaler,t_scaler}.joblib
├── metrics/<case_id>/<model_name>/{history.csv, run_config.json, metrics.json}
├── figures/<case_id>/<model_name>/*.png
├── predictions/
└── model_comparison.csv
```

## Project layout

```
Timeseries_prediction/
├── README.md  requirements.txt
├── configs/config.yaml
├── src/        # data, models, train, evaluate, predict, metrics, utils
├── scripts/    # inspect_dataset, train_one_case, train_all_cases,
│               # compare_models, plot_predictions, train_automl_baseline
├── outputs/    # checkpoints, scalers, metrics, figures, predictions
└── notebooks/quick_eda.ipynb   # EDA only — not part of the pipeline
```

## Reproducibility notes

* Fixed seeds across Python / NumPy / PyTorch (`src.utils.set_seed`).
* `pathlib.Path` everywhere; no hard-coded absolute paths (Linux & Windows safe).
* Output directories are created automatically.
* Missing files raise explicit errors — nothing is silently skipped.
* Runs from the project root (`Timeseries_prediction/`).
