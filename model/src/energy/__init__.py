"""Energy Monte Carlo propagation for runtime BHM posteriors.

Two variants matching the two pipeline stages:

  Heuristic filtering (simple Normal P_TASK):
    predict_energy_hf, run_energy_mc_hf

  String normalisation (form-specific Gaussian mixture P_TASK):
    sample_p_task_mixture, predict_energy_str_norm, run_energy_mc_str_norm

Both stages return energy in kJ.

Shared:
    decompose_variance_aggregate — population-level runtime/hardware
        variance share, computed once across a full set of predictions
        (not per-row). Use this, not a per-cell ratio, to reproduce the
        thesis's reported 99%/1% (filtering) and 97.5%/2.5% (string
        normalisation) figures.

All functions take posterior arrays and calibration parameters as
explicit arguments — no globals, no file paths.
"""
from .common import hdi_94, decompose_variance_aggregate
from .heuristic_filtering import predict_energy_hf, run_energy_mc_hf
from .string_normalisation import (
    sample_p_task_mixture,
    predict_energy_str_norm,
    run_energy_mc_str_norm,
)

__all__ = [
    "hdi_94",
    "decompose_variance_aggregate",
    "predict_energy_hf",
    "run_energy_mc_hf",
    "sample_p_task_mixture",
    "predict_energy_str_norm",
    "run_energy_mc_str_norm",
]