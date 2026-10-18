"""Production (non-centred) vs uniform-setting refit (centred) BHM.

    python compare_production.py

  in-sample : runtime/energy MAPE (per-job median, per config cell) and
              94% coverage of the BHM fit on all jobs — saved production fit
              vs refit with eval.bhm.SAMPLER (centred)
  hw_share  : in-sample with tau and thesis-style (no tau), from
              results/hw_share_{stage}{,_production}.csv
  zero-shot : median MAPE per BHM variant, results/benchmark_{stage}.csv
              (uniform setting) vs results/benchmark_{stage}_noncentered.csv
              (production setting: non-centred, target_accept 0.95, 2000 warmup);
              baselines do not depend on the BHM setting.
Rows whose change exceeds --flag points are marked. Writes
results/production_vs_refit.csv.
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

from eval.common import STAGES, PowerModel, derive_runtime_energy, load_stage_data  # noqa: E402
from eval.lodo import compute_metrics  # noqa: E402
from report_hw_share import in_sample_bhm  # noqa: E402


def in_sample_metrics(stage, df, production: bool, n: int, seed: int) -> dict:
    m = in_sample_bhm(stage, df, production, seed)
    rng = np.random.default_rng(seed)
    y = m.predict(df, n, rng)
    R, P, E = derive_runtime_energy(y, df, PowerModel(stage, seed=seed).fit(df), rng)
    met = compute_metrics(df, R, P, E, stage)
    return {k: met[k] for k in ("rt_mape", "e_mape", "rt_cov94", "e_cov94")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flag", type=float, default=3.0, help="flag changes larger than this many points")
    args = p.parse_args(argv)

    rows = []
    for st_name, stage in STAGES.items():
        df = load_stage_data(stage, HERE / stage.data_csv, HERE / stage.energy_csv)
        prod = in_sample_metrics(stage, df, True, args.n_samples, args.seed)
        refit = in_sample_metrics(stage, df, False, args.n_samples, args.seed)
        for k in prod:
            scale = 100 if "cov" in k else 1
            rows.append({"stage": st_name, "block": "in-sample", "item": f"bhm {k}",
                         "production": prod[k] * scale, "refit": refit[k] * scale})

        hw_p = pd.read_csv(HERE / f"results/hw_share_{st_name}_production.csv", index_col="corpus")
        hw_r = pd.read_csv(HERE / f"results/hw_share_{st_name}.csv", index_col="corpus")
        for col in ("in_sample", "thesis_style"):
            for agg in ("median", "pooled"):
                rows.append({"stage": st_name, "block": "hw_share", "item": f"{col} ({agg})",
                             "production": 100 * hw_p.at[agg, col], "refit": 100 * hw_r.at[agg, col]})

        b_r = pd.read_csv(HERE / f"results/benchmark_{st_name}.csv")
        b_p = pd.read_csv(HERE / f"results/benchmark_{st_name}_noncentered.csv")
        for model in sorted(b_p["model"].unique()):
            for metric in ("rt_mape", "e_mape"):
                rows.append({"stage": st_name, "block": "zero-shot median", "item": f"{model} {metric}",
                             "production": b_p.loc[b_p.model == model, metric].median(),
                             "refit": b_r.loc[b_r.model == model, metric].median()})
        for model, g in b_p.groupby("model"):
            rows.append({"stage": st_name, "block": "zero-shot convergence", "item": f"{model} max R-hat / divergences",
                         "production": f"{g.rhat_max.max():.3f} / {int(g.divergences.sum())}",
                         "refit": (lambda h: f"{h.rhat_max.max():.3f} / {int(h.divergences.sum())}")(
                             b_r[b_r.model == model])})

    out = pd.DataFrame(rows)
    num = pd.to_numeric(out["production"], errors="coerce").notna()
    out["change"] = np.where(num, pd.to_numeric(out["refit"], errors="coerce")
                             - pd.to_numeric(out["production"], errors="coerce"), np.nan)
    out["flag"] = np.where(out["change"].abs() > args.flag, "<<", "")
    path = HERE / "results/production_vs_refit.csv"
    out.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.round(2).to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
