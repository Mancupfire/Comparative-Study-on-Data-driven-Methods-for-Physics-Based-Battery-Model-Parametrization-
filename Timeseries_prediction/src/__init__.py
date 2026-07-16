"""Battery simulation time-series surrogate-model training package.

Modules
-------
data            : dataset discovery, alignment, splitting and Dataset classes
models          : MLP / RNN / LSTM / Bi-LSTM architectures and a model factory
train           : reusable training loop with early stopping and checkpointing
evaluate        : metric computation on inverse-transformed predictions
predict         : checkpoint / scaler loading and inference helpers
metrics         : voltage / temperature curve metrics
utils           : seeding, device selection, IO helpers

shared_data     : condition-aware point-wise / sequence datasets across all cases
shared_models   : shared MLP / RNN / LSTM / Bi-LSTM architectures
shared_train    : reusable training loop for shared models (point & sequence)
shared_evaluate : grouped physical-unit evaluation for shared models
"""

__all__ = [
    "data",
    "models",
    "train",
    "evaluate",
    "predict",
    "metrics",
    "utils",
    "shared_data",
    "shared_models",
    "shared_train",
    "shared_evaluate",
]
