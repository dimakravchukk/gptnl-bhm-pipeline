"""Shared benchmark plumbing: column names, per-stage config, the power
model, and the Predictor interface every model (BHM and baselines) implements.

Every model predicts y = log(runtime / total_chars). Runtime and energy are
derived afterwards by derive_runtime_energy() with the SAME fitted power model
for every model in a fold, so differences between models isolate the runtime model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from energy import sample_p_task_mixture

# ── column names ─────────────────────────────────────────────────────────────

CORPUS   = "corpus"
TARGET   = "log_rt_per_char"
CHARS    = "total_chars"
RUNTIME  = "runtime_s"
ENERGY   = "energy_kj_measured"
FORM     = "normalization_form"
FORM_IDX = "form_idx"
ALPHA    = "alpha_cfg"
CONFIG   = "config"

ZERO_SHOT = "zero_shot"
ANCHORED  = "anchored"

P_TASK_MIN, P_TASK_MAX = 10.0, 50.0  # clip range used by the energy/ module

NORM_FORMS = ["NFC", "NFD", "NFKC"]


# ── stage config ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class StageConfig:
    """Everything that differs between the two pipeline stages.

    cell_col    : column defining a "config cell" within a corpus (MAPE unit)
    job_key     : columns that uniquely identify a job (to drop probe rows
                  from the test set and to join measured energy)
    """
    name: str
    data_csv: str
    energy_csv: str
    cell_col: str
    job_key: list[str]

    def design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Non-corpus design features shared by all ML baselines, shape (rows, d)."""
        if self.name == "hf":
            return df[[ALPHA]].to_numpy(float)
        form = df[FORM_IDX].to_numpy(int)
        return np.stack([form == k for k in range(1, len(NORM_FORMS))], axis=1).astype(float)


STAGES = {
    "hf": StageConfig(
        name="hf",
        data_csv="results/df_train_hf.csv",
        energy_csv="results/df_energy_hf.csv",
        cell_col=CONFIG,
        job_key=[CORPUS, CONFIG, "n_tasks", "rep"],
    ),
    "sn": StageConfig(
        name="sn",
        data_csv="results/df_train_str_norm.csv",
        energy_csv="results/df_energy_str_norm.csv",
        cell_col=FORM,
        job_key=["run_id"],
    ),
}


# ── data assembly ────────────────────────────────────────────────────────────

def load_stage_data(stage: StageConfig, data_csv: str, energy_csv: str | None) -> pd.DataFrame:
    """Runtime jobs with measured energy (kJ) merged in; NaN where unmeasured."""
    df = pd.read_csv(data_csv)
    if stage.name == "sn" and FORM_IDX not in df:
        df[FORM_IDX] = df[FORM].map({f: i for i, f in enumerate(NORM_FORMS)})
    if energy_csv is None:
        df[ENERGY] = np.nan
        return df

    e = pd.read_csv(energy_csv)
    if stage.name == "hf":
        # energy CSV stores the filtering config in normalization_form and the
        # replicate only as the run_id suffix
        e = e.rename(columns={FORM: CONFIG})
        e["rep"] = e["run_id"].str.extract(r"_r(\d+)$")[0].astype(int)
    e = e[stage.job_key + [ENERGY]].drop_duplicates(stage.job_key)
    return df.merge(e, on=stage.job_key, how="left")


def resolve_corpora(requested: list[str] | None, available: list[str]) -> list[str]:
    """Map user-given corpus names onto the dataset's names, tolerating the
    filtering stage's `_str_norm` directory suffix (cc_github -> cc_github_str_norm)."""
    if not requested:
        return list(available)
    out = []
    for name in requested:
        hits = [c for c in available if c == name or c == f"{name}_str_norm"]
        if not hits:
            raise ValueError(f"unknown corpus {name!r}; available: {available}")
        out.append(hits[0])
    return out


def experimental_corpora() -> list[str]:
    """Union of corpus names across both stages' runtime data, with the
    filtering stage's `_str_norm` suffix stripped (the 9 experimental corpora)."""
    here = Path(__file__).resolve().parents[2]
    names = set()
    for st in STAGES.values():
        names |= {c.removesuffix("_str_norm") for c in pd.read_csv(here / st.data_csv)[CORPUS].unique()}
    return sorted(names)


def load_meta(meta_csv: str, reference_corpora: list[str],
              log: Callable[[str], None] = print) -> pd.DataFrame:
    """All usable per-corpus metadata features, one row per reference corpus.

    Accepts a wide file (a `corpus` column plus numeric feature columns) or
    the long stats format (dataset, metric, n_docs, mean, std, ...), which is
    pivoted to `{metric}_mean` / `{metric}_std` columns. Rows for corpora
    outside reference_corpora are ignored; a feature missing or NaN for any
    reference corpus is dropped (and logged), so every feature set is
    defined on exactly the same corpora.
    """
    raw = pd.read_csv(meta_csv)
    if {"dataset", "metric"} <= set(raw.columns):
        wide = raw.pivot(index="dataset", columns="metric", values=["mean", "std"])
        wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
        meta = wide[sorted(wide.columns)]
    else:
        meta = raw.set_index(CORPUS).select_dtypes("number")

    absent = sorted(set(reference_corpora) - set(meta.index))
    if absent:
        raise ValueError(f"{meta_csv} has no rows for corpora {absent}")
    meta = meta.loc[reference_corpora].astype(float)
    incomplete = meta.columns[meta.isna().any()].tolist()
    if incomplete:
        log(f"metadata: dropped {len(incomplete)} features missing for some corpus: {incomplete}")
    return meta.drop(columns=incomplete)


def meta_for_stage(meta: pd.DataFrame, cols: list[str], stage_corpora: list[str]) -> pd.DataFrame:
    """Select feature columns and re-index by the stage's corpus names
    (cc_github -> cc_github_str_norm for filtering)."""
    missing = sorted(set(cols) - set(meta.columns))
    if missing:
        raise ValueError(f"metadata features not available: {missing}")
    return pd.DataFrame({c: meta.loc[c.removesuffix("_str_norm"), cols] for c in stage_corpora}).T


# ── power model ──────────────────────────────────────────────────────────────

def job_power(df: pd.DataFrame) -> pd.Series:
    """Per-task power (W) of each job with measured energy: E / summed task runtime.

    The energy CSVs carry no power column; this is the job-average P_TASK that
    E = runtime * P_TASK implies, i.e. the quantity the sampler must reproduce.
    """
    d = df.dropna(subset=[ENERGY])
    return d[ENERGY] * 1000.0 / d[RUNTIME]


class PowerModel:
    """P_TASK sampler refit from data: one Normal for filtering, a
    2-component Gaussian mixture per form for string normalisation (thesis
    Section 2.d). Draws are clipped to [P_TASK_MIN, P_TASK_MAX] as in energy/."""

    def __init__(self, stage: StageConfig, n_components: int = 2, seed: int = 0):
        self.stage = stage
        self.n_components = n_components
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "PowerModel":
        P = job_power(df)
        if self.stage.name == "hf":
            self.params = {"mean": float(P.mean()), "std": float(P.std(ddof=1)), "n": int(len(P))}
            return self
        forms = df.loc[P.index, FORM_IDX].to_numpy(int)
        self.params = {}
        for k in sorted(set(forms)):
            gm = GaussianMixture(self.n_components, random_state=self.seed).fit(P[forms == k].to_numpy()[:, None])
            order = np.argsort(gm.means_.ravel())
            self.params[int(k)] = {
                "means":   gm.means_.ravel()[order].tolist(),
                "stds":    np.sqrt(gm.covariances_.ravel())[order].tolist(),
                "weights": gm.weights_[order].tolist(),
                "n":       int((forms == k).sum()),
            }
        return self

    def sample(self, df: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray:
        """Per-task power draws (W), shape (n, rows); independent per job."""
        if self.stage.name == "hf":
            P = rng.normal(self.params["mean"], self.params["std"], size=(n, len(df)))
        else:
            P = np.empty((n, len(df)))
            for j, f in enumerate(df[FORM_IDX].to_numpy(int)):
                P[:, j] = sample_p_task_mixture(f, self.params, n, rng)
        return np.clip(P, P_TASK_MIN, P_TASK_MAX)

    def to_json(self) -> str:
        return json.dumps(self.params, separators=(",", ":"))


# ── predictor interface ──────────────────────────────────────────────────────

class Predictor(Protocol):
    """fit(train_df) -> self
    predict(test_df, n, rng) -> (n, rows) samples of log(runtime/char).

    A corpus absent from train_df is an unseen corpus (zero-shot)."""
    name: str

    def fit(self, train_df: pd.DataFrame) -> "Predictor": ...

    def predict(self, test_df: pd.DataFrame, n: int, rng: np.random.Generator) -> np.ndarray: ...


def derive_runtime_energy(
    y_samples: np.ndarray, test_df: pd.DataFrame, power: PowerModel, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, rows) log-rt/char samples -> runtime (s), power (W), energy (kJ)."""
    R = np.exp(y_samples) * test_df[CHARS].to_numpy(float)[None, :]
    P = power.sample(test_df, y_samples.shape[0], rng)
    return R, P, R * P / 1000.0
