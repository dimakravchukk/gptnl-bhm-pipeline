"""BHM model definitions for runtime prediction.

Two models, one per pipeline stage:
  model_offset_hf                    — heuristic filtering
  model_offset_interaction_no_ntasks — string normalisation (production model)

Both target log(runtime / total_chars). Fit them with fit_mcmc(), which
returns both the raw MCMC object and an ArviZ InferenceData in one call —
use load_and_validate_fit() when loading a previously-saved .nc file, to
catch model/fit mismatches before they propagate into downstream code.
"""
from __future__ import annotations

from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import pandas as pd
from data import RuntimeDataset


# Expected posterior variable names per model — used by load_and_validate_fit
# to catch the wrong-model-loaded class of bug before it propagates.
MODEL_VARS = {
    "model_offset_hf": {
        "gamma0", "sigma_alpha", "z", "alpha_c",
        "beta_alpha", "sigma_beta", "zb", "beta_c", "tau",
    },
    "model_offset_interaction_no_ntasks": {
        "gamma0", "sigma_alpha", "z_alpha", "alpha_c",
        "delta_raw", "delta", "sigma_delta", "eta_z", "delta_corpus", "tau",
    },
}


def model_offset_hf(c_idx, alpha_cfg, J, y=None):
    """Heuristic filtering: partial-pooled corpus intercepts + varying slopes on alpha_cfg.

    Parameters
    ----------
    c_idx     : (N,) int array  — corpus index per observation, values in [0, J)
    alpha_cfg : (N,) float array — max_non_alpha_ratio - alpha_default per obs
    J         : int              — number of corpora
    y         : (N,) float array — observed log(runtime/chars), omit for prior predictive
    """
    _validate_index_range(c_idx, J, "c_idx", "J")

    gamma0      = numpyro.sample("gamma0",      dist.Normal(-12.0, 5.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    z           = numpyro.sample("z",           dist.Normal(0, 1).expand([J]))
    alpha_c     = numpyro.deterministic("alpha_c", gamma0 + sigma_alpha * z)

    beta_alpha  = numpyro.sample("beta_alpha",  dist.Normal(0.0, 1.0))
    sigma_beta  = numpyro.sample("sigma_beta",  dist.HalfNormal(0.5))
    zb          = numpyro.sample("zb",          dist.Normal(0, 1).expand([J]))
    beta_c      = numpyro.deterministic("beta_c", beta_alpha + sigma_beta * zb)

    tau = numpyro.sample("tau", dist.HalfNormal(0.3))
    mu  = alpha_c[c_idx] + beta_c[c_idx] * alpha_cfg
    numpyro.sample("obs", dist.Normal(mu, tau), obs=y)


def model_offset_interaction_no_ntasks(c_idx, form_idx, J, K, y=None):
    """String normalisation: corpus intercepts + corpus×form interaction. No n_tasks term.

    beta_ntasks was approx 0 in every prior fit, so it was dropped. Predicted
    runtime is therefore constant across parallelism for a given (corpus, form).

    Parameters
    ----------
    c_idx    : (N,) int array — corpus index per observation, values in [0, J)
    form_idx : (N,) int array — normalisation form index per observation
               (NFC=0 baseline), values in [0, K)
    J        : int             — number of corpora
    K         : int            — number of normalisation forms (typically 3: NFC, NFD, NFKC)
    y        : (N,) float array — observed log(runtime/chars), omit for prior predictive
    """
    _validate_index_range(c_idx, J, "c_idx", "J")
    _validate_index_range(form_idx, K, "form_idx", "K")

    gamma0      = numpyro.sample("gamma0",      dist.Normal(-16.0, 2.0))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(1.0))
    z_alpha     = numpyro.sample("z_alpha",     dist.Normal(jnp.zeros(J), jnp.ones(J)))
    alpha_c     = numpyro.deterministic("alpha_c", gamma0 + sigma_alpha * z_alpha)

    # Population-level form effects — NFC is baseline (delta[0] = 0)
    delta_raw = numpyro.sample("delta_raw", dist.Normal(jnp.zeros(K - 1), 0.5 * jnp.ones(K - 1)))
    delta     = numpyro.deterministic("delta", jnp.concatenate([jnp.array([0.0]), delta_raw]))

    # Corpus-specific deviation from population form effect, partial-pooled
    sigma_delta  = numpyro.sample("sigma_delta", dist.HalfNormal(0.3))
    eta_z        = numpyro.sample("eta_z", dist.Normal(jnp.zeros((J, K - 1)), jnp.ones((J, K - 1))))
    eta_full     = jnp.concatenate([jnp.zeros((J, 1)), sigma_delta * eta_z], axis=1)
    delta_corpus = numpyro.deterministic("delta_corpus", delta[None, :] + eta_full)  # (J, K)

    tau = numpyro.sample("tau", dist.HalfNormal(0.3))
    mu  = alpha_c[c_idx] + delta_corpus[c_idx, form_idx]
    numpyro.sample("obs", dist.Normal(mu, tau), obs=y)


def _validate_index_range(idx_array, expected_max_exclusive, idx_name, max_name):
    """Raise a clear error if idx_array contains out-of-range indices.

    Prevents an unhelpful JAX/NumPyro indexing error deep inside MCMC
    tracing when c_idx or form_idx don't match the corpora/forms the
    caller intended — a class of bug that has previously surfaced as
    silently wrong posterior column attribution rather than a crash.
    """
    arr = np.asarray(idx_array)
    if arr.size == 0:
        return
    actual_max = int(arr.max())
    actual_min = int(arr.min())
    if actual_min < 0 or actual_max >= expected_max_exclusive:
        raise ValueError(
            f"{idx_name} contains indices outside [0, {max_name}) = "
            f"[0, {expected_max_exclusive}): observed range "
            f"[{actual_min}, {actual_max}]. This usually means {idx_name} "
            f"was built with a corpus/form ordering that doesn't match "
            f"{max_name}={expected_max_exclusive} — check that {idx_name} "
            f"and {max_name} were derived from the same RuntimeDataset."
        )


def fit_mcmc(
    model,
    rng_key,
    *args,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = 4,
    target_accept_prob: float = 0.95,
    progress_bar: bool = False,
    **kwargs,
) -> tuple[MCMC, az.InferenceData]:
    """Run NUTS MCMC and return both the raw MCMC object and its
    ArviZ InferenceData conversion, so flatten_posterior() can be used
    directly on the result without a separate manual conversion step.

    Returns
    -------
    (mcmc, idata) : the fitted MCMC object and its az.from_numpyro() conversion

    Examples
    --------
    mcmc, idata = fit_mcmc(model_offset_hf, jax.random.PRNGKey(0),
                           c_idx, alpha_cfg, J, y=y)
    alpha_c_post = flatten_posterior(idata, "alpha_c")

    mcmc, idata = fit_mcmc(model_offset_interaction_no_ntasks, jax.random.PRNGKey(0),
                           c_idx, form_idx, J, K, y=y)
    delta_corpus_post = flatten_posterior(idata, "delta_corpus")
    """
    mcmc = MCMC(
        NUTS(model, target_accept_prob=target_accept_prob),
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng_key, *args, **kwargs)
    idata = az.from_numpyro(mcmc)
    return mcmc, idata


def flatten_posterior(idata: az.InferenceData, var: str) -> np.ndarray:
    """Flatten ArviZ InferenceData (chain, draw, ...) -> (chain*draw, ...) for one variable."""
    v = idata.posterior[var].values
    return v.reshape(-1, *v.shape[2:])


def save_fit(idata: az.InferenceData, path: Path | str, model_name: str) -> None:
    """Save a fit to disk, with the model name embedded as metadata.

    model_name must be a key in MODEL_VARS (e.g. "model_offset_hf"). This
    is what load_and_validate_fit checks against on load, so a fit saved
    under the wrong model_name will be silently mislabeled — always pass
    the literal model function's name as a string, not a guess.
    """
    if model_name not in MODEL_VARS:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {list(MODEL_VARS)}"
        )
    idata.attrs["model_name"] = model_name
    idata.to_netcdf(str(path))


def load_and_validate_fit(path: Path | str, expected_model_name: str) -> az.InferenceData:
    """Load a saved .nc fit and verify it matches the expected model.

    Checks both the saved model_name attribute (if present — older fits
    saved without save_fit() won't have it) and the actual posterior
    variable names against MODEL_VARS[expected_model_name].

    Raises ValueError on any mismatch, rather than silently returning a
    fit from the wrong model — this is the single check that would have
    caught the wrong-model-loaded bugs encountered repeatedly during
    development (see thesis discussion of model selection, Section 2.h).
    """
    if expected_model_name not in MODEL_VARS:
        raise ValueError(
            f"Unknown expected_model_name {expected_model_name!r}; "
            f"expected one of {list(MODEL_VARS)}"
        )

    idata = az.from_netcdf(str(path))

    saved_name = idata.attrs.get("model_name")
    if saved_name is not None and saved_name != expected_model_name:
        raise ValueError(
            f"{path}: saved model_name={saved_name!r} does not match "
            f"expected {expected_model_name!r}"
        )

    actual_vars = set(idata.posterior.data_vars)
    expected_vars = MODEL_VARS[expected_model_name]
    missing = expected_vars - actual_vars
    if missing:
        raise ValueError(
            f"{path}: missing expected variables for {expected_model_name!r}: "
            f"{missing}. This usually means the wrong .nc file was loaded — "
            f"posterior contains {sorted(actual_vars)}."
        )

    return idata

def fit_anchored(
    base_csv_path: str,
    probe_csv_path: str,
    model,
    extra_design_cols: list[str],
    target_col: str = "log_rt_per_char",
    rng_key=None,
    save_to: str | None = None,
    **fit_kwargs,
) -> tuple["RuntimeDataset", "az.InferenceData"]:

    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    base_df, probe_df = pd.read_csv(base_csv_path), pd.read_csv(probe_csv_path)

    new_corpus = set(probe_df["corpus"]) - set(base_df["corpus"])
    if len(new_corpus) != 1:
        raise ValueError(f"probe must contain exactly one new corpus, got {new_corpus}")

    combined = pd.concat([base_df, probe_df], ignore_index=True)
    corpora  = sorted(combined["corpus"].unique())
    dataset  = RuntimeDataset(df=combined, corpora=corpora,
                              corp2idx={c: i for i, c in enumerate(corpora)},
                              stage="anchored")

    design_args = [dataset.df[c].values for c in extra_design_cols]
    J = len(corpora)
    model_args = [J, 3] if "form_idx" in extra_design_cols else [J]

    _, idata = fit_mcmc(model, rng_key, dataset.c_idx, *design_args, *model_args,
                        y=dataset.df[target_col].values, **fit_kwargs)

    if save_to:
        save_fit(idata, save_to, model.__name__)
        dataset.df.to_csv(base_csv_path, index=False)

    return dataset, idata