"""Prior sensitivity of the string-normalisation BHM to the gamma0 prior mean.

    python prior_sensitivity.py --alt-loc -14

Uses the benchmark's uniform sampler setting (eval.bhm.SAMPLER, centred).
Fits all normalisation jobs under gamma0 ~ N(-16, 2) (production) and
N(alt, 2), and compares posterior means of every population and corpus
parameter, in absolute terms and in units of the production posterior sd.
Writes results/prior_sensitivity_sn.csv. The zero-shot side of the check is
    python run_benchmark.py --stage sn --only bhm --gamma0-loc -14 \
        --out results/prior_sensitivity_sn_lodo.csv
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
from eval.common import NORM_FORMS, STAGES, load_stage_data  # noqa: E402


def summarise(m: BHMPredictor) -> pd.DataFrame:
    rows = []
    for v in ["gamma0", "sigma_alpha", "sigma_delta", "tau"]:
        rows.append((v, m.post[v]))
    for j, c in enumerate(m.corpora):
        rows.append((f"alpha_c[{c}]", m.post["alpha_c"][:, j]))
    for k, f in enumerate(NORM_FORMS[1:], start=1):
        rows.append((f"delta[{f}]", m.post["delta"][:, k]))
    return pd.DataFrame([{"param": n, "mean": x.mean(), "sd": x.std()} for n, x in rows]).set_index("param")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--alt-loc", type=float, default=-14.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    stage = STAGES["sn"]
    df = load_stage_data(stage, HERE / stage.data_csv, None)
    fits = {loc: BHMPredictor(stage, seed=args.seed, model_kwargs={**MODEL_KWARGS, "gamma0_loc": loc},
                              **SAMPLER).fit(df)
            for loc in (-16.0, args.alt_loc)}
    base, alt = (summarise(fits[loc]) for loc in (-16.0, args.alt_loc))
    out = pd.DataFrame({
        "mean_prod": base["mean"], "sd_prod": base["sd"], "mean_alt": alt["mean"],
        "diff": alt["mean"] - base["mean"], "diff_in_sd": (alt["mean"] - base["mean"]) / base["sd"],
    })
    for loc, m in fits.items():
        print(f"gamma0_loc={loc}: divergences={m.divergences} rhat_max={m.rhat_max:.4f}")
    path = HERE / "results/prior_sensitivity_sn.csv"
    out.to_csv(path)
    print(out.round(4).to_string())
    print(f"max |diff| = {out['diff'].abs().max():.4f} log units; "
          f"max |diff|/sd = {out['diff_in_sd'].abs().max():.3f}\nwrote {path}")


if __name__ == "__main__":
    main()
