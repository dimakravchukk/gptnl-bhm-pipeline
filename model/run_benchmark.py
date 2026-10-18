"""LODO benchmark: BHM vs baselines under identical conditions.

    python run_benchmark.py --stage hf
    python run_benchmark.py --stage hf --only bhm --held-out cc_github cc_openalex
    python run_benchmark.py --stage hf --probes results/probe_hf.csv
    python run_benchmark.py --stage sn --meta results/stats_by_dataset.csv --only +meta +meta_all --update

Writes results/benchmark_{stage}.csv (one row per model x corpus x condition).

--probes  probe CSV (real small-scale probe runs, same columns as --data).
          Without it only the zero-shot condition runs. Probe jobs that also
          appear in --data are removed from the anchored test set.

--meta    per-corpus metadata (wide, or the long stats_by_dataset format).
          Features missing for any of the experimental corpora are dropped.
          Each ML baseline then also runs as
            <model>+meta      the stage's feature set (--meta-cols, default
                              DEFAULT_META_COLS; OPTIONAL_META_COLS added
                              when present in the file)
            <model>+meta_all  every available feature
          and the BHM as bhm+meta (metadata on the corpus intercept) and, for
          filtering, bhm+meta_slope (intercept and slope); curated set only.
--update  replace matching (model, corpus, condition) rows in an existing
          --out file instead of overwriting it.

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

from baselines.zoo import BASELINE_NAMES, build_baselines, selected  # noqa: E402
from eval.bhm import MODEL_KWARGS, SAMPLER, BHMPredictor  # noqa: E402
from eval.common import (  # noqa: E402
    CORPUS, STAGES, experimental_corpora, load_meta, load_stage_data, meta_for_stage, resolve_corpora,
)
from eval.lodo import run_lodo  # noqa: E402

DEFAULT_META_COLS = {
    "hf": ["length_mean", "non_alpha_digit_ratio_mean", "stop_word_ratio_mean"],
    "sn": ["length_mean", "length_std", "avg_word_length_mean"],
}
OPTIONAL_META_COLS = {"hf": [], "sn": ["nfd_expansion_mean"]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--data", help="runtime jobs CSV (default: the stage's results/df_train_*.csv)")
    p.add_argument("--energy", help="measured energy CSV (default: results/df_energy_*.csv); 'none' to skip")
    p.add_argument("--probes", help="probe CSV enabling the anchored condition")
    p.add_argument("--meta", help="per-corpus metadata CSV (wide or long stats format)")
    p.add_argument("--meta-cols", nargs="+", help="features for the +meta variant (default: per stage)")
    p.add_argument("--held-out", nargs="+", help="corpora to hold out (default: all)")
    p.add_argument("--only", nargs="+",
                   help=f"models to run: bhm {' '.join(BASELINE_NAMES)}, full names like rf+meta, "
                        "or a variant suffix like +meta_all")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-warmup", type=int, default=SAMPLER["num_warmup"])
    p.add_argument("--num-samples", type=int, default=SAMPLER["num_samples"])
    p.add_argument("--num-chains", type=int, default=SAMPLER["num_chains"])
    p.add_argument("--target-accept", type=float, default=SAMPLER["target_accept_prob"])
    p.add_argument("--non-centered", action="store_true",
                   help="use the production (non-centred) parameterisation of the corpus effects")
    p.add_argument("--gamma0-loc", type=float, help="normalisation only: prior mean of gamma0 (default -16)")
    p.add_argument("--out", help="output CSV (default: results/benchmark_{stage}.csv)")
    p.add_argument("--update", action="store_true", help="upsert rows into an existing --out file")
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

    metas = {}
    if args.meta:
        meta = load_meta(args.meta, experimental_corpora())
        cols = args.meta_cols or (DEFAULT_META_COLS[stage.name]
                                  + [c for c in OPTIONAL_META_COLS[stage.name] if c in meta.columns])
        metas = {"meta": meta_for_stage(meta, cols, corpora),
                 "meta_all": meta_for_stage(meta, list(meta.columns), corpora)}
        print(f"+meta features ({len(cols)}): {cols}")
        print(f"+meta_all features ({meta.shape[1]}): {list(meta.columns)}")

    factories = {}
    mcmc_kwargs = dict(num_warmup=args.num_warmup, num_samples=args.num_samples,
                       num_chains=args.num_chains, target_accept_prob=args.target_accept)
    model_kwargs = {**MODEL_KWARGS, "centered": not args.non_centered}
    if args.gamma0_loc is not None:
        if stage.name != "sn":
            raise SystemExit("--gamma0-loc applies to the normalisation stage only")
        model_kwargs["gamma0_loc"] = args.gamma0_loc
    bhm_variants = {"bhm": {}}
    if "meta" in metas:
        bhm_variants["bhm+meta"] = {"meta": metas["meta"]}
        if stage.name == "hf":
            bhm_variants["bhm+meta_slope"] = {"meta": metas["meta"], "meta_on_slope": True}
    for name, kw in bhm_variants.items():
        if selected(name, args.only):
            factories[name] = lambda kw=kw: BHMPredictor(stage, seed=args.seed, model_kwargs=model_kwargs,
                                                         **kw, **mcmc_kwargs)
    print(f"sampler: {mcmc_kwargs}  model_kwargs: {model_kwargs}")
    factories.update(build_baselines(stage, metas=metas, only=args.only, seed=args.seed))
    if not factories:
        raise SystemExit(f"--only {args.only} selected no models")

    print(f"stage={stage.name}  jobs={len(df)}  held_out={held_out}  models={list(factories)}")
    results = run_lodo(df, stage, factories, held_out, probes, n_samples=args.n_samples, seed=args.seed)

    out = Path(args.out) if args.out else HERE / f"results/benchmark_{stage.name}.csv"
    if args.update and out.exists():
        key = ["model", CORPUS, "condition"]
        old = pd.read_csv(out)
        kept = old.merge(results[key], on=key, how="left", indicator=True)["_merge"] == "left_only"
        results = pd.concat([old[kept.to_numpy()], results], ignore_index=True)
    results.to_csv(out, index=False)
    print(f"wrote {out} ({len(results)} rows)")
    return results


if __name__ == "__main__":
    main()
