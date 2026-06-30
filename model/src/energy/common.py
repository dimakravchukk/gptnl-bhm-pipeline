"""Shared energy utilities: HDI computation and aggregate variance decomposition.

Variance decomposition is computed once across a full set of predictions,
not per-row inside the prediction loop — a single row's MC variance share
is a noisy estimate of a population-level quantity and should not be
treated as a per-row statistic. See decompose_variance_aggregate.
"""
from __future__ import annotations

import numpy as np
import arviz as az


def hdi_94(samples: np.ndarray) -> tuple[float, float]:
    """94% highest density interval, as a (lo, hi) tuple of floats."""
    lo, hi = az.hdi(samples, hdi_prob=0.94)
    return float(lo), float(hi)


def decompose_variance_aggregate(
    rt_samples_list: list[np.ndarray],
    P_samples_list: list[np.ndarray],
) -> dict:
    """Decompose total energy variance into runtime-model and hardware-power
    shares, aggregated across many (corpus, config) or (corpus, form) cells.

    For each cell i, with paired runtime draws rt_i (S,) and power draws P_i (S,):
        E_runtime_only_i = mean(P_i) * rt_i        (power fixed at its mean)
        E_power_only_i   = mean(rt_i) * P_i         (runtime fixed at its mean)

    Parameters
    ----------
    rt_samples_list : list of (S,) arrays, one per cell, runtime draws (seconds)
    P_samples_list  : list of (S,) arrays, one per cell, power draws (W),
                       same length and S as rt_samples_list

    Returns
    -------
    dict with keys:
        share_runtime, share_power   — population-level variance shares (sum to 1.0)
        per_cell_var_runtime, per_cell_var_power  — lists, one value per cell
        additivity_check   — mean of (var_runtime + var_power) / var_full per
                              cell; should be close to 1.0 if runtime and power
                              are genuinely independent (see thesis Eq. 3.11)
    """
    if len(rt_samples_list) != len(P_samples_list):
        raise ValueError(
            f"rt_samples_list and P_samples_list must have the same length, "
            f"got {len(rt_samples_list)} and {len(P_samples_list)}"
        )
    if len(rt_samples_list) == 0:
        raise ValueError("decompose_variance_aggregate: received empty input")

    var_runtime_list = []
    var_power_list = []
    additivity_ratios = []

    for rt, P in zip(rt_samples_list, P_samples_list):
        rt_mean = rt.mean()
        P_mean  = P.mean()

        E_full         = P * rt
        E_runtime_only = P_mean * rt
        E_power_only   = rt_mean * P

        var_full   = np.var(E_full)
        var_rt     = np.var(E_runtime_only)
        var_power  = np.var(E_power_only)

        var_runtime_list.append(float(var_rt))
        var_power_list.append(float(var_power))

        if var_full > 0:
            additivity_ratios.append((var_rt + var_power) / var_full)

    total_var_rt    = sum(var_runtime_list)
    total_var_power = sum(var_power_list)
    total_var       = total_var_rt + total_var_power

    return {
        "share_runtime": total_var_rt / total_var if total_var > 0 else 0.0,
        "share_power":   total_var_power / total_var if total_var > 0 else 0.0,
        "per_cell_var_runtime": var_runtime_list,
        "per_cell_var_power":   var_power_list,
        "additivity_check": float(np.mean(additivity_ratios)) if additivity_ratios else float("nan"),
        "additivity_check_std": float(np.std(additivity_ratios)) if additivity_ratios else float("nan"),
    }