"""LODO benchmark: BHM vs baselines under identical conditions.

    python run_benchmark.py --stage hf
    python run_benchmark.py --stage hf --only bhm --held-out cc_github cc_openalex
    python run_benchmark.py --stage hf --probes results/probe_hf.csv
    python run_benchmark.py --stage sn --meta corpus_meta.csv --meta-cols mean_doc_len frac_non_ascii

Writes results/benchmark_{stage}.csv (one row per model x corpus x condition).

--probes  probe CSV (real small-scale probe runs, same columns as --data).
          Without it only the zero-shot condition runs. Probe jobs that also
          appear in --data are removed from the anchored test set.

The power model (Normal for hf, per-form 2-component GMM for sn) is refit in
every fold from the training rows' energy/runtime; its parameters are logged
and stored in the power_params column.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpyro  # noqa: E402

numpyro.set_host_device_count(4)  # run NUTS chains in parallel on CPU

import pandas as pd  # noqa: E402

from baselines.zoo import BASELINE_NAMES, build_baselines  # noqa: E402
from eval.bhm import BHMPredictor  # noqa: E402
from eval.common import CORPUS, STAGES, load_meta, load_stage_data, resolve_corpora  # noqa: E402
from eval.lodo import run_lodo  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--data", help="runtime jobs CSV (default: the stage's results/df_train_*.csv)")
    p.add_argument("--energy", help="measured energy CSV (default: results/df_energy_*.csv); 'none' to skip")
    p.add_argument("--probes", help="probe CSV enabling the anchored condition")
    p.add_argument("--meta", help="per-corpus metadata CSV with a 'corpus' column")
    p.add_argument("--meta-cols", nargs="+", help="metadata columns to use (default: all numeric)")
    p.add_argument("--held-out", nargs="+", help="corpora to hold out (default: all)")
    p.add_argument("--only", nargs="+", help=f"models to run: bhm {' '.join(BASELINE_NAMES)}")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=2000)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--num-chains", type=int, default=4)
    p.add_argument("--out", help="output CSV (default: results/benchmark_{stage}.csv)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    stage = STAGES[args.stage]

    data_csv = args.data or HERE / stage.data_csv
    energy_csv = None if args.energy == "none" else (args.energy or HERE / stage.energy_csv)
    df = load_stage_data(stage, data_csv, energy_csv)
    corpora = sorted(df[CORPUS].unique())
    held_out = resolve_corpora(args.held_out, corpora)

    probes = None
    if args.probes:
        probes = pd.read_csv(args.probes)
        probes[CORPUS] = resolve_corpora(probes[CORPUS].tolist(), corpora)

    meta = load_meta(args.meta, args.meta_cols, corpora)

    factories = {}
    if args.only is None or "bhm" in args.only:
        mcmc_kwargs = dict(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains)
        factories["bhm"] = lambda: BHMPredictor(stage, seed=args.seed, **mcmc_kwargs)
    factories.update(build_baselines(stage, meta=meta, only=args.only, seed=args.seed))
    if not factories:
        raise SystemExit(f"--only {args.only} selected no models")

    print(f"stage={stage.name}  jobs={len(df)}  held_out={held_out}  models={list(factories)}")
    results = run_lodo(df, stage, factories, held_out, probes, n_samples=args.n_samples, seed=args.seed)

    out = Path(args.out) if args.out else HERE / f"results/benchmark_{stage.name}.csv"
    results.to_csv(out, index=False)
    print(f"wrote {out} ({len(results)} rows)")
    return results


if __name__ == "__main__":
    main()
