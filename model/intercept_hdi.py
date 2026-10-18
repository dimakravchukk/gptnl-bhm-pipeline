"""94% HDIs of the corpus intercepts alpha_c (and, for filtering, the slopes
beta_c): uniform-setting refit vs production fit.

    python intercept_hdi.py

For each stage, refits the BHM on all jobs with the benchmark's uniform sampler
setting (as calibration.py and report_hw_share.py do) and reads the saved
production fit. Writes results/intercept_hdi_{stage}.csv: one row per setting
x parameter x corpus with the posterior mean and 94% HDI. make_paper_assets.py
counts overlapping HDI pairs and the slope spread from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpyro  # noqa: E402

numpyro.set_host_device_count(4)

import arviz as az  # noqa: E402
import pandas as pd  # noqa: E402

from eval.common import CORPUS, STAGES, load_stage_data  # noqa: E402
from report_hw_share import in_sample_bhm  # noqa: E402


def corpus_hdis(model) -> pd.DataFrame:
    rows = []
    for param in ("alpha_c", "beta_c"):
        if param not in model.post:  # beta_c: filtering only
            continue
        a = model.post[param]
        for j, corpus in enumerate(model.corpora):
            lo, hi = az.hdi(a[:, j], hdi_prob=0.94)
            rows.append({"param": param, "corpus": corpus, "mean": a[:, j].mean(),
                         "hdi94_lower": lo, "hdi94_upper": hi})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    for st_name, stage in STAGES.items():
        df = load_stage_data(stage, HERE / stage.data_csv, HERE / stage.energy_csv)
        frames = []
        for setting, production in (("uniform", False), ("production", True)):
            m = in_sample_bhm(stage, df, production, args.seed)
            assert m.corpora == sorted(df[CORPUS].unique())
            frames.append(corpus_hdis(m).assign(setting=setting))
        out = HERE / f"results/intercept_hdi_{st_name}.csv"
        pd.concat(frames)[["setting", "param", "corpus", "mean", "hdi94_lower", "hdi94_upper"]].to_csv(out, index=False)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
