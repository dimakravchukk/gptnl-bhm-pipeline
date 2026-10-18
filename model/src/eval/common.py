"""Shared benchmark plumbing: column names, per-stage config, the power
model, and the Predictor interface every model (BHM and baselines) implements.

Every model predicts y = log(runtime / total_chars). Runtime and energy are
derived afterwards by derive_runtime_energy() with the SAME fitted power model
for every model in a fold, so differences between models isolate the runtime model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

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


def load_meta(meta_csv: str | None, meta_cols: list[str] | None, corpora: list[str]) -> pd.DataFrame | None:
    """Per-corpus metadata indexed by the dataset's corpus names.

    Rows are matched on exact name, then with the `_str_norm` suffix
    stripped, so one metadata file serves both stages.
    """
    if meta_csv is None:
        return None
    meta = pd.read_csv(meta_csv).set_index(CORPUS)
    cols = meta_cols or [c for c in meta.columns if pd.api.types.is_numeric_dtype(meta[c])]
    out = {}
    for c in corpora:
        key = c if c in meta.index else c.removesuffix("_str_norm")
        if key not in meta.index:
            raise ValueError(f"--meta has no row for corpus {c!r}")
        out[c] = meta.loc[key, cols].astype(float)
    return pd.DataFrame(out).T


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
