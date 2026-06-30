"""Energy Monte Carlo propagation for the heuristic filtering stage
(simple Normal P_TASK calibration, single pooled distribution across
all corpora and configurations).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import hdi_94


def predict_energy_hf(
    alpha_c_post: np.ndarray,
    beta_c_post: np.ndarray,
    corpus_idx: int,
    alpha_cfg: float,
    total_chars: float,
    p_task_mean: float,
    p_task_std: float,
    p_task_min: float = 10.0,
    p_task_max: float = 50.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Propagate runtime + power uncertainty to energy for one (corpus, config) cell.

    Returns energy in kJ, matching the string normalisation stage's convention.

    Parameters
    ----------
    alpha_c_post : (S, J) posterior draws of corpus intercepts
    beta_c_post  : (S, J) posterior draws of corpus slopes on alpha_cfg
    corpus_idx   : column index in alpha_c_post / beta_c_post for this corpus
    alpha_cfg    : max_non_alpha_ratio - alpha_default for this config
    total_chars  : total character count for this corpus+config
    p_task_mean/std/min/max : power calibration parameters (W)
    rng          : numpy Generator; created fresh with seed 42 if None

    Returns
    -------
    dict with keys:
        E_samples, rt_samples, P_samples  — raw draws, kept separately so
            downstream variance decomposition doesn't require resampling
        E_mean_kJ, E_std_kJ
        E_hdi_lo_kJ, E_hdi_hi_kJ
        rt_mean_s, rt_std_s
    """
    if rng is None:
        rng = np.random.default_rng(42)

    S = alpha_c_post.shape[0]
    log_rt = (alpha_c_post[:, corpus_idx]
              + beta_c_post[:, corpus_idx] * alpha_cfg
              + np.log(total_chars))
    rt_samples = np.exp(log_rt)

    P_samples = rng.normal(p_task_mean, p_task_std, size=S)
    P_samples = np.clip(P_samples, p_task_min, p_task_max)

    E_samples = rt_samples * P_samples / 1000.0  # J -> kJ

    lo, hi = hdi_94(E_samples)
    return {
        "E_samples":   E_samples,
        "rt_samples":  rt_samples,
        "P_samples":   P_samples,
        "E_mean_kJ":   float(E_samples.mean()),
        "E_std_kJ":    float(E_samples.std()),
        "E_hdi_lo_kJ": lo,
        "E_hdi_hi_kJ": hi,
        "rt_mean_s":   float(rt_samples.mean()),
        "rt_std_s":    float(rt_samples.std()),
    }


def run_energy_mc_hf(
    df_lookup: "pd.DataFrame",
    alpha_c_post: np.ndarray,
    beta_c_post: np.ndarray,
    corpora: list[str],
    p_task_mean: float,
    p_task_std: float,
    p_task_min: float = 10.0,
    p_task_max: float = 50.0,
    rng_seed: int = 42,
) -> "pd.DataFrame":
    """Run energy MC for every (corpus, config) row in df_lookup.

    df_lookup must have columns: corpus, config, alpha_cfg, total_chars.
    corpora must be the ordered list that matches alpha_c_post columns
    — use dataset.corpora from a RuntimeDataset, or the exact same list
    passed to make_design_arrays() at fit time.

    NOTE ON REPRODUCIBILITY: a single Generator is created once and reused
    sequentially across all rows. Re-running predict_energy_hf for a single
    row in isolation will NOT reproduce the same draws as that row produced
    inside the full loop. Pass your own freshly-seeded Generator to
    predict_energy_hf directly if you need a single cell's result to be
    independently reproducible.

    Returns df_energy with one row per (corpus, config) containing all
    fields from predict_energy_hf plus corpus, config, alpha_cfg, total_chars.
    Rows whose corpus is not in `corpora` are skipped, with a printed count.
    """
    import pandas as pd

    corp2idx  = {c: i for i, c in enumerate(corpora)}
    rng       = np.random.default_rng(rng_seed)
    records   = []
    n_skipped = 0

    for _, row in df_lookup.iterrows():
        if row["corpus"] not in corp2idx:
            n_skipped += 1
            continue
        result = predict_energy_hf(
            alpha_c_post = alpha_c_post,
            beta_c_post  = beta_c_post,
            corpus_idx   = corp2idx[row["corpus"]],
            alpha_cfg    = float(row["alpha_cfg"]),
            total_chars  = float(row["total_chars"]),
            p_task_mean  = p_task_mean,
            p_task_std   = p_task_std,
            p_task_min   = p_task_min,
            p_task_max   = p_task_max,
            rng          = rng,
        )
        records.append({
            "corpus":      row["corpus"],
            "config":      row["config"],
            "alpha_cfg":   float(row["alpha_cfg"]),
            "total_chars": float(row["total_chars"]),
            **result,
        })

    if n_skipped:
        print(f"run_energy_mc_hf: skipped {n_skipped} rows with corpus not in `corpora`")

    return pd.DataFrame(records)