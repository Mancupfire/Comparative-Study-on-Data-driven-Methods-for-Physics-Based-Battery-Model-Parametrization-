"""Final filtered time-series protocol (Data_Batch_4_TSFiltered_0p8).

Adds three things on top of the existing per-case time-series pipeline:

* duration-ratio filtering of the source sequences (see
  ``scripts/build_batch4_tsfiltered.py``);
* valid-time masking so a held / extrapolated tail
  (``time_s > simulation_end_s``) never contributes to the loss or to any
  reported metric;
* a grouped sample-id split (700/150/150) reconstructed identically to the
  completed grouped error-metric benchmark.

The module is self-contained and writes only to the isolated
``outputs/Data_Batch_4_TSFiltered_0p8/time_series`` namespace.
"""
