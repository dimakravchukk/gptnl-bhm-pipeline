"""Hardware share of energy variance for the BHM, per stage and corpus.

    python report_hw_share.py --stage hf

hw_share = Var(P * R_mean) / (Var(P_mean * R) + Var(P * R_mean)), job-level,
summed over a corpus's jobs (eval.lodo.variance_shares). Columns:

  in_sample         BHM fit on all jobs, every corpus seen, tau included
  zero_shot         LODO (population draw + tau), from results/benchmark_{stage}.csv
  anchored          LODO with probe rows, from the same CSV (NaN until probes exist)
  thesis_style      same fit, in-sample, WITHOUT tau — comparable to the
                    thesis's reported split (not to the columns above)

The in-sample BHM is refit on all jobs with the benchmark's uniform sampler
setting (eval.bhm.SAMPLER, centred); --production uses the saved production
fits instead (non-centred; hf R-hat 1.016, sn 1 divergence).
Power is the PowerModel refit on all jobs of the stage. Rows: one per corpus,
'median' across corpora, and 'pooled' (variances summed over all jobs, the
aggregation used for the thesis figure; LODO columns are per-corpus only).
Writes results/hw_share_{stage}.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpyro  # noqa: E402

numpyro.set_host_device_count(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval.bhm import MODEL_KWARGS, SAMPLER, BHMPredictor  # noqa: E402
from eval.common import CORPUS, STAGES, PowerModel, derive_runtime_energy, load_stage_data  # noqa: E402
from eval.lodo import variance_shares  # noqa: E402
from models import load_and_validate_fit  # noqa: E402

PRODUCTION = {
    "hf": ("results/fit_hf_production.nc", "model_offset_hf"),
    "sn": ("results/fit_str_norm_production.nc", "model_offset_interaction_no_ntasks"),
}


def in_sample_bhm(stage, df, production: bool, seed: int = 0) -> BHMPredictor:
    """BHM fit on all jobs: refit with the uniform setting, or the saved production fit."""
    if production:
        nc, model_name = PRODUCTION[stage.name]
        return BHMPredictor.from_posterior(stage, load_and_validate_fit(HERE / nc, model_name),
                                           sorted(df[CORPUS].unique()))
    m = BHMPredictor(stage, seed=seed, model_kwargs=MODEL_KWARGS, **SAMPLER).fit(df)
    print(f"in-sample BHM ({stage.name}): {m.diagnostics()}")
    return m


def in_sample_shares(stage, df, model: BHMPredictor, n_samples, seed, with_tau: bool) -> dict:
    m = model
    tau = m.post["tau"]
    if not with_tau:
        m.post["tau"] = np.zeros_like(tau)
    power = PowerModel(stage, seed=seed).fit(df)
    rng = np.random.default_rng(seed)

    y = m.predict(df, n_samples, rng)
    m.post["tau"] = tau
    R, P, E = derive_runtime_energy(y, df, power, rng)
    out = {"pooled": variance_shares(R, P, E)[0]}
    for corpus, rows in df.groupby(CORPUS).indices.items():
        out[corpus] = variance_shares(R[:, rows], P[:, rows], E[:, rows])[0]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--benchmark", help="benchmark CSV (default: results/benchmark_{stage}.csv)")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--production", action="store_true", help="use the saved production fits")
    args = p.parse_args(argv)
    stage = STAGES[args.stage]

    df = load_stage_data(stage, HERE / stage.data_csv, HERE / stage.energy_csv)
    model = in_sample_bhm(stage, df, args.production, args.seed)
    table = pd.DataFrame({
        "in_sample":    in_sample_shares(stage, df, model, args.n_samples, args.seed, with_tau=True),
        "thesis_style": in_sample_shares(stage, df, model, args.n_samples, args.seed, with_tau=False),
    })

    bench_path = Path(args.benchmark) if args.benchmark else HERE / f"results/benchmark_{stage.name}.csv"
    bench = pd.read_csv(bench_path)
    bhm = bench[bench["model"] == "bhm"]
    for cond in ("zero_shot", "anchored"):
        table[cond] = bhm[bhm["condition"] == cond].set_index(CORPUS)["hw_share"]

    table = table[["in_sample", "zero_shot", "anchored", "thesis_style"]]
    per_corpus = table.drop(index="pooled")
    table = pd.concat([per_corpus, per_corpus.median().rename("median").to_frame().T, table.loc[["pooled"]]])
    table.index.name = CORPUS

    out = HERE / f"results/hw_share_{stage.name}{'_production' if args.production else ''}.csv"
    table.to_csv(out)
    print(f"hw_share — stage {stage.name}\n{(100 * table).round(1).to_string()}\nwrote {out}")
    return table


if __name__ == "__main__":
    main()
