"""Forecast probe feasibility at full-scale target sizes from local data.

The local mock corpora are far smaller than the Snellius inputs, so a fraction
of them holds far fewer documents than the real probe will. This draws whole
documents (with replacement) from each local corpus's length distribution
until the full-scale target is reached, packs them with make_probes.pack, and
reports the shard ratio and minimum shard size over several draws.

    python forecast_balance.py --config probes.snellius.yaml --local probes.local.yaml
"""

import argparse

import numpy as np
import pyarrow.parquet as pq

from make_probes import balance, list_source_files, load_config, pack, text_lengths


def local_lengths(job) -> np.ndarray:
    parts = [text_lengths(pq.read_table(f, columns=[job.text_column])[job.text_column])
             for f in list_source_files(job)]
    lens = np.concatenate(parts)
    return lens[lens > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config with the full_corpus_chars to forecast for")
    ap.add_argument("--local", required=True, help="config whose inputs are available locally")
    ap.add_argument("--draws", type=int, default=20)
    args = ap.parse_args()

    from pathlib import Path
    target_cfg, local_cfg = load_config(Path(args.config)), load_config(Path(args.local))
    local = {(j.stage, j.corpus): j for j in local_cfg.jobs}
    rng = np.random.default_rng(0)

    print(f"{'stage':22s} {'corpus':14s} {'frac':>6s} {'target':>13s} {'docs':>8s} "
          f"{'ratio med':>9s} {'ratio max':>9s} {'min docs':>8s} {'P(feasible)':>11s}")
    for job in target_cfg.jobs:
        if (job.stage, job.corpus) not in local or not job.full_corpus_chars:
            continue
        lens = local_lengths(local[(job.stage, job.corpus)])
        for frac in target_cfg.size_fractions:
            target = int(np.ceil(frac * job.full_corpus_chars))
            ratios, mins, ok, ndocs = [], [], 0, []
            for _ in range(args.draws):
                # Enough draws to pass the target, then cut at the first crossing.
                n_guess = int(1.2 * target / lens.mean()) + 16
                sample = rng.choice(lens, size=n_guess)
                while sample.sum() < target:
                    sample = np.concatenate([sample, rng.choice(lens, size=n_guess)])
                k = int(np.searchsorted(np.cumsum(sample), target)) + 1
                sample = sample[:k]
                shards = pack(sample, [""] * k, target_cfg.n_workers)
                sc = [int(sample[s].sum()) for s in shards]
                sd = [len(s) for s in shards]
                r, feas, _ = balance(sc, sd, target_cfg.max_ratio, target_cfg.min_docs_per_worker)
                ratios.append(r); mins.append(min(sd)); ok += feas; ndocs.append(k)
            print(f"{job.stage:22s} {job.corpus:14s} {frac:6g} {target:13,d} {int(np.median(ndocs)):8d} "
                  f"{np.median(ratios):9.3f} {max(ratios):9.3f} {min(mins):8d} {ok / args.draws:11.2f}")


if __name__ == "__main__":
    main()
