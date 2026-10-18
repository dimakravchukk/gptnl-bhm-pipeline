"""The BHMs from models.py behind the Predictor interface.

Seen corpus  : posterior predictive with that corpus's own parameters.
Unseen corpus: one population-level draw of the corpus parameters per
               posterior sample, shared across all rows of that corpus
               (centred on X_c·w for the metadata variants).
Both include observation noise tau.

Variants (name):
  bhm             production models
  bhm+meta        corpus metadata as group-level predictors on the intercept
  bhm+meta_slope  filtering only: metadata on intercept and slope
Metadata features are standardised on each fold's training corpora.
"""
from __future__ import annotations

import arviz as az
import jax
import numpy as np
import pandas as pd

from models import (
    fit_mcmc, flatten_posterior,
    model_offset_hf, model_offset_hf_meta,
    model_offset_interaction_no_ntasks, model_offset_interaction_no_ntasks_meta,
)

from .common import ALPHA, CORPUS, FORM_IDX, NORM_FORMS, TARGET, StageConfig

# The one sampler setting used for every benchmark, in-sample and sensitivity fit.
# Centred corpus effects: the non-centred production parameterisation left a
# gamma0 <-> z ridge (R-hat up to 2) because tau is tiny and every corpus is
# pinned by its data; centred gives 0 divergences and R-hat < 1.01 on every fold.
SAMPLER = dict(num_warmup=4000, num_samples=2000, num_chains=4, target_accept_prob=0.99)
MODEL_KWARGS = {"centered": True}


class BHMPredictor:

    def __init__(self, stage: StageConfig, seed: int = 0, meta: pd.DataFrame | None = None,
                 meta_on_slope: bool = False, model_kwargs: dict | None = None, **mcmc_kwargs):
        if meta_on_slope and (meta is None or stage.name != "hf"):
            raise ValueError("meta_on_slope needs metadata and the filtering stage")
        self.stage = stage
        self.seed = seed
        self.meta = meta
        self.meta_on_slope = meta_on_slope
        self.model_kwargs = model_kwargs or {}
        self.mcmc_kwargs = mcmc_kwargs
        self.name = "bhm" if meta is None else ("bhm+meta_slope" if meta_on_slope else "bhm+meta")

    @classmethod
    def from_posterior(cls, stage: StageConfig, idata: az.InferenceData, corpora: list[str]) -> "BHMPredictor":
        """Wrap an existing production fit (e.g. load_and_validate_fit of a .nc).

        corpora must be the ordering the fit was produced with.
        """
        self = cls(stage)
        self.corpora = list(corpora)
        self.corp2idx = {c: i for i, c in enumerate(self.corpora)}
        self.post = {v: flatten_posterior(idata, v) for v in self._post_vars()}
        n_cols = self.post["alpha_c"].shape[1]
        if n_cols != len(self.corpora):
            raise ValueError(f"fit has {n_cols} corpora, got ordering of {len(self.corpora)}")
        return self

    def _post_vars(self) -> list[str]:
        if self.stage.name == "hf":
            names = ["gamma0", "sigma_alpha", "alpha_c", "beta_alpha", "sigma_beta", "beta_c", "tau"]
        else:
            names = ["gamma0", "sigma_alpha", "alpha_c", "delta", "sigma_delta", "delta_corpus", "tau"]
        if self.meta is not None:
            names.append("w_alpha")
        if self.meta_on_slope:
            names.append("w_beta")
        return names

    def _standardised(self, corpora: list[str]) -> np.ndarray:
        return ((self.meta.loc[corpora] - self.x_mean) / self.x_std).to_numpy(float)

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "BHMPredictor":
        self.corpora = sorted(train_df[CORPUS].unique())
        self.corp2idx = {c: i for i, c in enumerate(self.corpora)}
        c_idx = train_df[CORPUS].map(self.corp2idx).to_numpy()
        y = train_df[TARGET].to_numpy(float)
        J = len(self.corpora)
        key = jax.random.PRNGKey(self.seed)

        if self.stage.name == "hf":
            args = [c_idx, train_df[ALPHA].to_numpy(float), J]
            model, extra = model_offset_hf, {}
            if self.meta is not None:
                model, extra = model_offset_hf_meta, {"meta_on_slope": self.meta_on_slope}
        else:
            args = [c_idx, train_df[FORM_IDX].to_numpy(int), J, len(NORM_FORMS)]
            model, extra = model_offset_interaction_no_ntasks, {}
            if self.meta is not None:
                model = model_offset_interaction_no_ntasks_meta
        if self.meta is not None:
            train_meta = self.meta.loc[self.corpora]
            self.x_mean, self.x_std = train_meta.mean(), train_meta.std(ddof=0)
            args.append(self._standardised(self.corpora))

        mcmc, idata = fit_mcmc(model, key, *args, y=y, **extra, **self.model_kwargs, **self.mcmc_kwargs)

        names = self._post_vars()
        self.post = {v: flatten_posterior(idata, v) for v in names}
        self.divergences = int(mcmc.get_extra_fields()["diverging"].sum())
        rhat = az.rhat(idata, var_names=names)
        self.rhat_max = float(max(np.nanmax(rhat[v].values) for v in names))
        return self

    # ── predict ──────────────────────────────────────────────────────────────
    def predict(self, test_df: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        S = self.post["tau"].shape[0]
        draw = rng.choice(S, size=n, replace=n > S)
        p = {k: v[draw] for k, v in self.post.items()}

        mu = np.empty((n, len(test_df)))
        for corpus, rows in test_df.groupby(CORPUS).indices.items():
            sub = test_df.iloc[rows]
            mu[:, rows] = self._corpus_mu(corpus, sub, p, rng)
        return mu + p["tau"][:, None] * rng.standard_normal(mu.shape)

    def _corpus_mu(self, corpus, sub, p, rng) -> np.ndarray:
        """(n, len(sub)) linear predictor for one corpus."""
        n = p["tau"].shape[0]
        j = self.corp2idx.get(corpus)

        if j is None:  # unseen: population draw, centred on the metadata prediction
            a_mean, b_mean = p["gamma0"], p.get("beta_alpha")
            if self.meta is not None:
                x = self._standardised([corpus])[0]
                a_mean = a_mean + p["w_alpha"] @ x
                if self.meta_on_slope:
                    b_mean = b_mean + p["w_beta"] @ x
            a = a_mean + p["sigma_alpha"] * rng.standard_normal(n)

        if self.stage.name == "hf":
            if j is None:
                b = b_mean + p["sigma_beta"] * rng.standard_normal(n)
            else:
                a, b = p["alpha_c"][:, j], p["beta_c"][:, j]
            return a[:, None] + b[:, None] * sub[ALPHA].to_numpy(float)[None, :]

        f = sub[FORM_IDX].to_numpy(int)
        if j is None:
            K = p["delta"].shape[1]
            eta = np.concatenate([np.zeros((n, 1)),
                                  p["sigma_delta"][:, None] * rng.standard_normal((n, K - 1))], axis=1)
            d = p["delta"] + eta
        else:
            a, d = p["alpha_c"][:, j], p["delta_corpus"][:, j, :]
        return a[:, None] + d[:, f]

    def diagnostics(self) -> dict:
        return {"divergences": self.divergences, "rhat_max": self.rhat_max,
                **{f"mcmc_{k}": v for k, v in self.mcmc_kwargs.items()},
                **{f"prior_{k}": v for k, v in self.model_kwargs.items()}}
