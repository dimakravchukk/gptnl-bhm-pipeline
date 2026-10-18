"""In-sample calibration: every model fit on all jobs of a stage, scored on
those same jobs.

    python calibration.py --stage hf

Per job and model: PIT = P(predicted runtime <= measured runtime) and whether
the measurement lies in the 94% HDI. For the refit BHM also LOO-PIT
(az.loo_pit, PSIS leave-one-job-out) and its Pareto k per job; baselines and
--production have in-sample PIT only. The BHM is refit on all jobs with the
benchmark's uniform sampler setting (--production: the saved production fit);
baselines are refit on all jobs.
Also computes in-sample runtime and energy MAPE with the power model fit on
all jobs, two ways: cell-level (per-job median prediction vs measured mean per
config cell, as in the benchmark: rt_mape, e_mape) and job-level (each job's
median prediction vs that job's measurement: rt_mape_job, e_mape_job), which
includes job-to-job runtime noise and power variation.
Writes results/calibration_{stage}.csv (one row per job x model) and
results/insample_mape_{stage}.csv (one row per model).
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

from baselines.zoo import build_baselines  # noqa: E402
from energy import hdi_94  # noqa: E402
from eval.common import CHARS, CORPUS, ENERGY, RUNTIME, STAGES, PowerModel, derive_runtime_energy, load_stage_data  # noqa: E402
from eval.lodo import compute_metrics  # noqa: E402
from report_hw_share import in_sample_bhm  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--production", action="store_true", help="BHM from the saved production fit")
    args = p.parse_args(argv)
    stage = STAGES[args.stage]

    df = load_stage_data(stage, HERE / stage.data_csv, HERE / stage.energy_csv)
    power = PowerModel(stage, seed=args.seed).fit(df)
    models = {"bhm": in_sample_bhm(stage, df, args.production, args.seed)}
    models.update({name: f().fit(df) for name, f in build_baselines(stage, seed=args.seed).items()})

    chars, meas = df[CHARS].to_numpy(float), df[RUNTIME].to_numpy(float)
    loo_pit = pareto_k = None
    if not args.production:
        loo_pit, pareto_k = models["bhm"].loo_pit(seed=args.seed)
        print(f"LOO-PIT: {(pareto_k > 0.7).sum()} of {len(pareto_k)} jobs with Pareto k > 0.7 (max {pareto_k.max():.2f})")
    rows, mape = [], []
    for name, model in models.items():
        rng = np.random.default_rng(args.seed)
        R, P, E = derive_runtime_energy(model.predict(df, args.n_samples, rng), df, power, rng)
        met = compute_metrics(df, R, P, E, stage)
        has_e = df[ENERGY].notna().to_numpy()
        job = lambda pred, meas: float(100 * np.mean(np.abs(pred - meas) / meas))
        mape.append({"model": name, **{k: met[k] for k in ("rt_mape", "rt_mape_mean", "e_mape", "e_mape_mean")},
                     "rt_mape_job": job(np.median(R, 0), df[RUNTIME].to_numpy(float)),
                     "e_mape_job": job(np.median(E[:, has_e], 0), df.loc[has_e, ENERGY].to_numpy(float))})
        for j in range(len(df)):
            lo, hi = hdi_94(R[:, j])
            loo = name == "bhm" and loo_pit is not None
            rows.append({"model": name, CORPUS: df[CORPUS].iat[j], stage.cell_col: df[stage.cell_col].iat[j],
                         "pit": float((R[:, j] <= meas[j]).mean()), "in_hdi94": bool(lo <= meas[j] <= hi),
                         "pit_loo": float(loo_pit[j]) if loo else np.nan,
                         "pareto_k": float(pareto_k[j]) if loo else np.nan})

    suffix = "_production" if args.production else ""
    out = pd.DataFrame(rows)
    path = HERE / f"results/calibration_{stage.name}{suffix}.csv"
    out.to_csv(path, index=False)
    mape = pd.DataFrame(mape)
    mape.to_csv(HERE / f"results/insample_mape_{stage.name}{suffix}.csv", index=False)
    print(mape.round(3).to_string(index=False))
    print(out.groupby("model").agg(n_jobs=("pit", "size"), cov94=("in_hdi94", "mean"),
                                   pit_mean=("pit", "mean")).round(3).to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
