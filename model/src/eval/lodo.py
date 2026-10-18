"""Leave-one-corpus-out fold loop and per-(model, corpus, condition) metrics.

For each held-out corpus:
  zero_shot : train on all other corpora, test on every job of the held-out one.
  anchored  : additionally put the held-out corpus's probe rows (from a
              real probe CSV only) into the training set of EVERY model;
              test on the remaining jobs.

Point prediction is the median of each job's predictive samples for every
model (rt_mape, e_mape); the predictive mean is kept as rt_mape_mean /
e_mape_mean. Intervals and coverage use the full predictive (incl. noise).
The power model is refit per fold on the training rows only.
"""
from __future__ import annotations

import time
import zlib
from typing import Callable

import numpy as np
import pandas as pd

from energy import hdi_94

from .common import (
    ANCHORED, CORPUS, ENERGY, RUNTIME, ZERO_SHOT,
    PowerModel, Predictor, StageConfig, derive_runtime_energy,
)


def cell_mape(pred: np.ndarray, meas: np.ndarray, cells: np.ndarray) -> float:
    """MAPE (%) of per-cell predicted mean vs per-cell measured mean."""
    d = pd.DataFrame({"pred": pred, "meas": meas, "cell": cells}).groupby("cell").mean()
    return float((100 * (d["pred"] - d["meas"]).abs() / d["meas"]).mean())


def coverage94(samples: np.ndarray, measured: np.ndarray) -> float:
    """Share of individual jobs whose measurement lies in the row's 94% HDI."""
    inside = [lo <= m <= hi for m, (lo, hi) in
              ((m, hdi_94(samples[:, j])) for j, m in enumerate(measured))]
    return float(np.mean(inside))


def variance_shares(R: np.ndarray, P: np.ndarray, E: np.ndarray) -> tuple[float, float]:
    """(hw_share, additivity), each summed over rows before taking the ratio.

    hw_share   = Var(P * R_mean) / (Var(P_mean * R) + Var(P * R_mean))
    additivity = Var(E) / (Var(P_mean * R) + Var(P * R_mean))   (~1 if independent)
    """
    var_rt = np.var(P.mean(0) * R, axis=0).sum()
    var_hw = np.var(R.mean(0) * P, axis=0).sum()
    var_e  = np.var(E * 1000.0, axis=0).sum()  # E is kJ; R*P is J
    total = var_rt + var_hw
    return float(var_hw / total), float(var_e / total)


def compute_metrics(test_df: pd.DataFrame, R, P, E, stage: StageConfig) -> dict:
    cells = test_df[stage.cell_col].to_numpy()
    rt_meas = test_df[RUNTIME].to_numpy(float)
    out = {
        "rt_mape":      cell_mape(np.median(R, 0), rt_meas, cells),
        "rt_mape_mean": cell_mape(R.mean(0), rt_meas, cells),
        "rt_cov94":     coverage94(R, rt_meas),
    }
    has_e = test_df[ENERGY].notna().to_numpy()
    if has_e.any():
        e_meas = test_df.loc[has_e, ENERGY].to_numpy(float)
        out["e_mape"]      = cell_mape(np.median(E[:, has_e], 0), e_meas, cells[has_e])
        out["e_mape_mean"] = cell_mape(E[:, has_e].mean(0), e_meas, cells[has_e])
        out["e_cov94"]     = coverage94(E[:, has_e], e_meas)
    else:
        out["e_mape"] = out["e_mape_mean"] = out["e_cov94"] = np.nan
    out["hw_share"], out["additivity"] = variance_shares(R, P, E)
    out["n_energy"] = int(has_e.sum())
    return out


def _drop_jobs(df: pd.DataFrame, jobs: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    m = df.merge(jobs[key].drop_duplicates(), on=key, how="left", indicator=True)
    return df[(m["_merge"] == "left_only").to_numpy()]


def run_lodo(
    df: pd.DataFrame,
    stage: StageConfig,
    model_factories: dict[str, Callable[[], Predictor]],
    held_out: list[str],
    probes: pd.DataFrame | None,
    n_samples: int = 4000,
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    records = []
    for corpus in held_out:
        base_train = df[df[CORPUS] != corpus]
        test_all   = df[df[CORPUS] == corpus]

        folds = [(ZERO_SHOT, base_train, test_all)]
        probe_c = probes[probes[CORPUS] == corpus] if probes is not None else None
        if probe_c is not None and len(probe_c):
            train_an = pd.concat([base_train, probe_c[base_train.columns.intersection(probe_c.columns)]],
                                 ignore_index=True)
            test_an = _drop_jobs(test_all, probe_c, stage.job_key)
            folds.append((ANCHORED, train_an, test_an))
        else:
            log(f"[{corpus}] no probe rows — anchored condition skipped")

        for condition, train, test in folds:
            test = test.reset_index(drop=True)
            power = PowerModel(stage, seed=seed).fit(train)
            log(f"[{corpus} | {condition}] power model: {power.to_json()}")
            for name, factory in model_factories.items():
                t0 = time.time()
                rng = np.random.default_rng([seed, zlib.crc32(f"{corpus}|{condition}|{name}".encode())])
                model = factory().fit(train.reset_index(drop=True))
                y = model.predict(test, n_samples, rng)
                rec = {"model": name, "corpus": corpus, "condition": condition,
                       "n_train": len(train), "n_test": len(test)}
                if np.isnan(y).any():
                    rec["note"] = "no prediction for unseen corpus"
                else:
                    R, P, E = derive_runtime_energy(y, test, power, rng)
                    rec.update(compute_metrics(test, R, P, E, stage))
                if hasattr(model, "diagnostics"):
                    rec.update(model.diagnostics())
                rec["power_params"] = power.to_json()
                rec["fit_seconds"] = round(time.time() - t0, 1)
                records.append(rec)
                log(f"[{corpus} | {condition} | {name}] "
                    f"rt_mape={rec.get('rt_mape', np.nan):.1f}%  e_mape={rec.get('e_mape', np.nan):.1f}%  "
                    f"rt_cov94={rec.get('rt_cov94', np.nan):.2f}  ({rec['fit_seconds']}s)")
    return pd.DataFrame(records)
