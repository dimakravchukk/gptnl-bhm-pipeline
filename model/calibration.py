"""In-sample calibration: every model fit on all jobs of a stage, scored on
those same jobs.

    python calibration.py --stage hf

Per job and model: PIT = P(predicted runtime <= measured runtime) and whether
the measurement lies in the 94% HDI. The BHM is refit on all jobs with the
benchmark's uniform sampler setting (--production: the saved production fit);
baselines are refit on all jobs.
Writes results/calibration_{stage}.csv (one row per job x model).
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
from eval.common import CHARS, CORPUS, RUNTIME, STAGES, load_stage_data  # noqa: E402
from report_hw_share import in_sample_bhm  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--production", action="store_true", help="BHM from the saved production fit")
    args = p.parse_args(argv)
    stage = STAGES[args.stage]

    df = load_stage_data(stage, HERE / stage.data_csv, None)
    models = {"bhm": in_sample_bhm(stage, df, args.production, args.seed)}
    models.update({name: f().fit(df) for name, f in build_baselines(stage, seed=args.seed).items()})

    chars, meas = df[CHARS].to_numpy(float), df[RUNTIME].to_numpy(float)
    rows = []
    for name, model in models.items():
        R = np.exp(model.predict(df, args.n_samples, np.random.default_rng(args.seed))) * chars[None, :]
        for j in range(len(df)):
            lo, hi = hdi_94(R[:, j])
            rows.append({"model": name, CORPUS: df[CORPUS].iat[j], stage.cell_col: df[stage.cell_col].iat[j],
                         "pit": float((R[:, j] <= meas[j]).mean()), "in_hdi94": bool(lo <= meas[j] <= hi)})

    out = pd.DataFrame(rows)
    path = HERE / f"results/calibration_{stage.name}{'_production' if args.production else ''}.csv"
    out.to_csv(path, index=False)
    print(out.groupby("model").agg(n_jobs=("pit", "size"), cov94=("in_hdi94", "mean"),
                                   pit_mean=("pit", "mean")).round(3).to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
