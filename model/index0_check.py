"""Reproduce the suspected thesis zero-shot bug: an unseen corpus mapped to
corpus index 0 (am_stories).

    python index0_check.py

pandas `.map(corp2idx)` gives NaN for a corpus missing from the mapping;
`.astype(int)` turns NaN into 0 on this platform and JAX indexing then reads
index 0 silently. For each held-out filtering corpus, the fold is fit without
it (production model and setting) and its jobs are scored with every training
corpus's own parameters. Writes results/index0_check.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

import numpyro  # noqa: E402

numpyro.set_host_device_count(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eval.bhm import BHMPredictor  # noqa: E402
from eval.common import ALPHA, CHARS, CONFIG, CORPUS, RUNTIME, STAGES, load_stage_data  # noqa: E402
from eval.lodo import cell_mape  # noqa: E402

HELD_OUT = ["cc_github_str_norm", "cc_openalex_str_norm"]
THESIS = {"cc_github_str_norm": 1833.0, "cc_openalex_str_norm": 20.0}


def main():
    stage = STAGES["hf"]
    df = load_stage_data(stage, HERE / stage.data_csv, None)
    nan_idx = pd.Series(["unseen"]).map({"seen": 0}).to_numpy().astype(int)[0]
    print(f"NaN corpus index cast to int on this platform: {nan_idx}")
    rows = []
    for held in HELD_OUT:
        train, test = df[df[CORPUS] != held], df[df[CORPUS] == held]
        m = BHMPredictor(stage, seed=0, num_warmup=2000, num_samples=2000, num_chains=4,
                         target_accept_prob=0.95).fit(train)
        x = test[ALPHA].to_numpy(float)[None, :]
        for j, as_corpus in enumerate(m.corpora):
            rt = np.exp(m.post["alpha_c"][:, j, None] + m.post["beta_c"][:, j, None] * x) * test[CHARS].to_numpy(float)
            rows.append({"held_out": held, "scored_as": as_corpus, "index": j,
                         "rt_mape": cell_mape(np.median(rt, 0), test[RUNTIME].to_numpy(float), test[CONFIG].to_numpy()),
                         "thesis_rt_mape": THESIS[held]})
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "results/index0_check.csv", index=False)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
