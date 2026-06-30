"""Energy Monte Carlo propagation for the string normalisation stage
(form-specific Gaussian mixture P_TASK calibration — see thesis
Section 2.d for the bimodality finding motivating the mixture over a
single Normal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


from .common import hdi_94


def sample_p_task_mixture(
    form_idx: int,
    p_task_mixture: dict[int, dict],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw power samples (W) from a form-specific Gaussian mixture.

    p_task_mixture maps form index -> {"means": [...], "stds": [...], "weights": [...]}.
    Component count is not fixed at 2 by this function — it follows
    whatever was fitted (see thesis: BIC selected k=2 for all three forms
    in the final model, though a 1-5 sweep showed k>2 was sometimes
    marginally preferred; k=2 was retained as the stable, parsimonious
    choice — see Section 2.d for the full discussion).
    """
    mix        = p_task_mixture[form_idx]
    means      = np.array(mix["means"])
    stds       = np.array(mix["stds"])
    weights    = np.array(mix["weights"])
    components = rng.choice(len(means), size=n_samples, p=weights)
    return rng.normal(means[components], stds[components])


def predict_energy_str_norm(
    alpha_c_post: np.ndarray,
    delta_corpus_post: np.ndarray,
    corpus_idx: int,
    form_idx: int,
    total_chars: float,
    p_task_mixture: dict[int, dict],
    p_task_min: float = 10.0,
    p_task_max: float = 50.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Propagate posterior uncertainty to energy samples (kJ) for one (corpus, form) cell.

    Returns a dict, matching predict_energy_hf's interface, so both stages'
    outputs can be passed to decompose_variance_aggregate identically.

    Returns
    -------
    dict with keys:
        E_samples, rt_samples, P_samples  — raw draws (S,), kept separately
            so downstream variance decomposition doesn't require resampling
        E_mean_kJ, E_std_kJ
        E_hdi_lo_kJ, E_hdi_hi_kJ
    """
    if rng is None:
        rng = np.random.default_rng(42)

    S       = alpha_c_post.shape[0]
    mu      = alpha_c_post[:, corpus_idx] + delta_corpus_post[:, corpus_idx, form_idx]
    runtime = np.exp(mu) * total_chars

    P_samples = sample_p_task_mixture(form_idx, p_task_mixture, S, rng)
    P_samples = np.clip(P_samples, p_task_min, p_task_max)

    E_samples = P_samples * runtime / 1000.0

    lo, hi = hdi_94(E_samples)
    return {
        "E_samples":   E_samples,
        "rt_samples":  runtime,
        "P_samples":   P_samples,
        "E_mean_kJ":   float(E_samples.mean()),
        "E_std_kJ":    float(E_samples.std()),
        "E_hdi_lo_kJ": lo,
        "E_hdi_hi_kJ": hi,
    }


def run_energy_mc_str_norm(
    df_lookup: "pd.DataFrame",
    alpha_c_post: np.ndarray,
    delta_corpus_post: np.ndarray,
    corpora: list[str],
    form_map: dict[str, int],
    p_task_mixture: dict[int, dict],
    p_task_min: float = 10.0,
    p_task_max: float = 50.0,
    rng_seed: int = 42,
) -> "pd.DataFrame":
    import pandas as pd

    corp2idx = {c: i for i, c in enumerate(corpora)}
    rng      = np.random.default_rng(rng_seed)
    records  = []
    skipped_pairs = []

    for _, row in df_lookup.iterrows():
        corpus = row["corpus"]
        form   = row["normalization_form"]
        if corpus not in corp2idx or form not in form_map:
            skipped_pairs.append((corpus, form))
            continue

        result = predict_energy_str_norm(
            alpha_c_post      = alpha_c_post,
            delta_corpus_post = delta_corpus_post,
            corpus_idx        = corp2idx[corpus],
            form_idx          = form_map[form],
            total_chars       = float(row["total_chars"]),
            p_task_mixture    = p_task_mixture,
            p_task_min        = p_task_min,
            p_task_max        = p_task_max,
            rng               = rng,
        )

        records.append({
            "corpus":             corpus,
            "normalization_form": form,
            "total_chars":        float(row["total_chars"]),
            **result,
        })

    if skipped_pairs:
        print(f"run_energy_mc_str_norm: skipped {len(skipped_pairs)} rows "
              f"with unrecognised (corpus, form): {skipped_pairs}")

    return pd.DataFrame(records)