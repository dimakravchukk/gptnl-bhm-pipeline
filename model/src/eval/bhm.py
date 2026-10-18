"""The production BHMs from models.py behind the Predictor interface.

Seen corpus  : posterior predictive with that corpus's own parameters.
Unseen corpus: one population-level draw of the corpus parameters per
               posterior sample, shared across all rows of that corpus.
Both include observation noise tau.
"""
from __future__ import annotations

import arviz as az
import jax
import numpy as np
import pandas as pd

from models import fit_mcmc, flatten_posterior, model_offset_hf, model_offset_interaction_no_ntasks

from .common import ALPHA, CORPUS, FORM_IDX, NORM_FORMS, TARGET, StageConfig


class BHMPredictor:
    name = "bhm"

    def __init__(self, stage: StageConfig, seed: int = 0, **mcmc_kwargs):
        self.stage = stage
        self.seed = seed
        self.mcmc_kwargs = mcmc_kwargs

    # ── fit ──────────────────────────────────────────────────────────────────
    def fit(self, train_df: pd.DataFrame) -> "BHMPredictor":
        self.corpora = sorted(train_df[CORPUS].unique())
        self.corp2idx = {c: i for i, c in enumerate(self.corpora)}
        c_idx = train_df[CORPUS].map(self.corp2idx).to_numpy()
        y = train_df[TARGET].to_numpy(float)
        J = len(self.corpora)
        key = jax.random.PRNGKey(self.seed)

        if self.stage.name == "hf":
            mcmc, idata = fit_mcmc(model_offset_hf, key, c_idx, train_df[ALPHA].to_numpy(float), J,
                                   y=y, **self.mcmc_kwargs)
            names = ["gamma0", "sigma_alpha", "alpha_c", "beta_alpha", "sigma_beta", "beta_c", "tau"]
        else:
            K = len(NORM_FORMS)
            mcmc, idata = fit_mcmc(model_offset_interaction_no_ntasks, key, c_idx,
                                   train_df[FORM_IDX].to_numpy(int), J, K, y=y, **self.mcmc_kwargs)
            names = ["gamma0", "sigma_alpha", "alpha_c", "delta", "sigma_delta", "delta_corpus", "tau"]

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

        if self.stage.name == "hf":
            if j is None:
                a = p["gamma0"] + p["sigma_alpha"] * rng.standard_normal(n)
                b = p["beta_alpha"] + p["sigma_beta"] * rng.standard_normal(n)
            else:
                a, b = p["alpha_c"][:, j], p["beta_c"][:, j]
            return a[:, None] + b[:, None] * sub[ALPHA].to_numpy(float)[None, :]

        f = sub[FORM_IDX].to_numpy(int)
        if j is None:
            K = p["delta"].shape[1]
            a = p["gamma0"] + p["sigma_alpha"] * rng.standard_normal(n)
            eta = np.concatenate([np.zeros((n, 1)),
                                  p["sigma_delta"][:, None] * rng.standard_normal((n, K - 1))], axis=1)
            d = p["delta"] + eta
        else:
            a, d = p["alpha_c"][:, j], p["delta_corpus"][:, j, :]
        return a[:, None] + d[:, f]

    def diagnostics(self) -> dict:
        return {"divergences": self.divergences, "rhat_max": self.rhat_max}
