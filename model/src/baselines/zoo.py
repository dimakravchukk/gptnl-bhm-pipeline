"""Baselines for the LODO benchmark, all behind the Predictor interface.

  complete_pooling : one OLS on the design features, corpus ignored
  no_pooling       : one OLS per corpus, pooled residual sigma; cannot
                     predict an unseen corpus (returns NaN)
  rf, gbm, gp, mlp : f(design [+ per-corpus metadata]) + per-corpus offset

ML baselines are two-stage. f is fit on design features plus optional
corpus metadata; a per-corpus offset b_c is the mean out-of-fold residual of
corpus c's training rows, with observation noise the out-of-fold residual sd
after offsets.

For a corpus absent from training, uncertainty comes from an inner
leave-one-corpus-out over the training corpora: f is refit without corpus k
and scored on it. The held-out-corpus mean residuals m_k give the shared
corpus-level error, drawn from N(0, rms(m)) once per sample for all of that
corpus's rows (the analogue of the BHM's population draw); the within-corpus
spread of those residuals gives the row noise. This keeps metadata variants
honest: metadata that merely identifies training corpora fits them in-sample
but does not shrink the held-out residuals. Without metadata an unseen corpus
collapses to complete pooling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from eval.common import CORPUS, TARGET, StageConfig


def _with_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


class CompletePooling:
    """OLS y ~ design; predictive draws include coefficient uncertainty."""
    name = "complete_pooling"

    def __init__(self, stage: StageConfig):
        self.stage = stage

    def fit(self, train_df):
        X = _with_intercept(self.stage.design_matrix(train_df))
        y = train_df[TARGET].to_numpy(float)
        self.coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = max(len(y) - np.linalg.matrix_rank(X), 1)
        self.sigma = np.sqrt(((y - X @ self.coef) ** 2).sum() / dof)
        self.cov = self.sigma**2 * np.linalg.pinv(X.T @ X)
        return self

    def predict(self, test_df, n, rng):
        X = _with_intercept(self.stage.design_matrix(test_df))
        coefs = rng.multivariate_normal(self.coef, self.cov, size=n, method="svd")
        return coefs @ X.T + self.sigma * rng.standard_normal((n, len(X)))


class NoPooling:
    """Separate OLS per corpus (minimum-norm where a corpus doesn't identify
    every design column), with residual sigma pooled across corpora."""
    name = "no_pooling"

    def __init__(self, stage: StageConfig):
        self.stage = stage

    def fit(self, train_df):
        self.fits, sse, dof = {}, 0.0, 0
        for corpus, g in train_df.groupby(CORPUS):
            X = _with_intercept(self.stage.design_matrix(g))
            y = g[TARGET].to_numpy(float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.fits[corpus] = (coef, np.linalg.pinv(X.T @ X))
            sse += ((y - X @ coef) ** 2).sum()
            dof += len(y) - np.linalg.matrix_rank(X)
        self.sigma = np.sqrt(sse / max(dof, 1))
        return self

    def predict(self, test_df, n, rng):
        out = np.full((n, len(test_df)), np.nan)
        for corpus, rows in test_df.groupby(CORPUS).indices.items():
            if corpus not in self.fits:
                continue
            coef, xtx_inv = self.fits[corpus]
            X = _with_intercept(self.stage.design_matrix(test_df.iloc[rows]))
            coefs = rng.multivariate_normal(coef, self.sigma**2 * xtx_inv, size=n, method="svd")
            out[:, rows] = coefs @ X.T + self.sigma * rng.standard_normal((n, len(rows)))
        return out


class OffsetRegressor:
    """f(design [+ meta]) + per-corpus offset; see module docstring."""

    def __init__(self, name: str, estimator, stage: StageConfig,
                 meta: pd.DataFrame | None = None, n_folds: int = 5, seed: int = 0):
        self.name = name
        self.estimator = estimator
        self.stage = stage
        self.meta = meta
        self.n_folds = n_folds
        self.seed = seed

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        X = self.stage.design_matrix(df)
        if self.meta is not None:
            X = np.column_stack([X, self.meta.loc[df[CORPUS]].to_numpy(float)])
        return X

    def fit(self, train_df):
        X, y = self._features(train_df), train_df[TARGET].to_numpy(float)
        corpus = train_df[CORPUS].to_numpy()

        oof = np.empty_like(y)
        for tr, va in KFold(min(self.n_folds, len(y)), shuffle=True, random_state=self.seed).split(X):
            oof[va] = clone(self.estimator).fit(X[tr], y[tr]).predict(X[va])
        resid = pd.Series(y - oof, index=corpus)

        self.offset = resid.groupby(level=0).mean()
        self.sigma = float(np.std(resid - self.offset.loc[corpus].to_numpy(), ddof=1))
        self.unseen_sd, self.unseen_sigma = self._inner_lodo(X, y, corpus)
        self.model = clone(self.estimator).fit(X, y)
        return self

    def _inner_lodo(self, X, y, corpus) -> tuple[float, float]:
        """(corpus-level sd, within-corpus sd) of held-out-corpus residuals."""
        groups = np.unique(corpus)
        if len(groups) < 2:
            return 0.0, self.sigma
        means, within = [], []
        for k in groups:
            held = corpus == k
            r = y[held] - clone(self.estimator).fit(X[~held], y[~held]).predict(X[held])
            means.append(r.mean())
            within.append(r - r.mean())
        within = np.concatenate(within)
        return float(np.sqrt(np.mean(np.square(means)))), float(np.sqrt(np.sum(within**2) / (len(within) - len(groups))))

    def predict(self, test_df, n, rng):
        mu = self.model.predict(self._features(test_df))
        out = np.empty((n, len(test_df)))
        for corpus, rows in test_df.groupby(CORPUS).indices.items():
            if corpus in self.offset.index:
                b, sigma = np.full(n, self.offset[corpus]), self.sigma
            else:
                b, sigma = self.unseen_sd * rng.standard_normal(n), self.unseen_sigma
            out[:, rows] = (mu[rows][None, :] + b[:, None]
                            + sigma * rng.standard_normal((n, len(rows))))
        return out


def _ml_estimators(seed: int) -> dict:
    scaled = lambda est: TransformedTargetRegressor(make_pipeline(StandardScaler(), est),
                                                    transformer=StandardScaler())
    return {
        "rf":  RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=seed),
        "gbm": GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                         subsample=0.8, random_state=seed),
        "gp":  scaled(GaussianProcessRegressor(
                   ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(1e-2),
                   n_restarts_optimizer=2, random_state=seed)),
        "mlp": scaled(MLPRegressor(hidden_layer_sizes=(16, 16), alpha=1e-2, max_iter=3000,
                                   random_state=seed)),
    }


BASELINE_NAMES = ["complete_pooling", "no_pooling", "rf", "gbm", "gp", "mlp"]


def build_baselines(stage: StageConfig, meta: pd.DataFrame | None = None,
                    only: list[str] | None = None, seed: int = 0) -> dict:
    """{name: zero-arg factory}. ML baselines appear without metadata and,
    when `meta` is given, again as '<name>+meta'."""
    wanted = lambda n: only is None or n in only or n.split("+")[0] in only
    factories = {}
    if wanted("complete_pooling"):
        factories["complete_pooling"] = lambda: CompletePooling(stage)
    if wanted("no_pooling"):
        factories["no_pooling"] = lambda: NoPooling(stage)
    for name, est in _ml_estimators(seed).items():
        if wanted(name):
            factories[name] = lambda n=name, e=est: OffsetRegressor(n, e, stage, seed=seed)
        if meta is not None and wanted(f"{name}+meta"):
            factories[f"{name}+meta"] = (
                lambda n=f"{name}+meta", e=est: OffsetRegressor(n, e, stage, meta=meta, seed=seed))
    return factories
