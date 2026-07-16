"""Error-metric final extension: add Random Forest, XGBoost and CatBoost.

The final error-metric model set has eight families::

    ANN, MLP, Gated MLP, Deep Ensemble MLP   (reused, neural)
    ExtraTrees                               (reused, classical)
    Random Forest, XGBoost, CatBoost         (new, classical — this package)

The five reused families are taken verbatim from the completed grouped
benchmark run (same grouped split, 17 features, two targets, train-only
scalers, and stored predictions); only the three new tree models are trained
here.  Everything is written under the isolated
``outputs/Data_Batch_4/error_metric_final_extension`` namespace; the completed
benchmark outputs are never modified.
"""
