"""tests/test_benchmark.py — LODO benchmark plumbing (synthetic data, fast)."""
import numpy as np
import pandas as pd
import pytest

from baselines.zoo import build_baselines
from eval.bhm import BHMPredictor
from eval.common import STAGES, PowerModel
from eval.lodo import cell_mape, run_lodo, variance_shares

FAST = dict(num_warmup=200, num_samples=200, num_chains=1)


@pytest.fixture
def hf_df():
    rng = np.random.default_rng(0)
    rows = []
    for corp, a in [("c0", -12.0), ("c1", -12.5), ("c2", -13.0), ("c3", -12.2)]:
        for cfg, x in [("ro_alpha060", -0.2), ("ro_alpha075", -0.05), ("ro_alpha080", 0.0)]:
            for rep in (1, 2):
                y = a + 0.3 * x + 0.02 * rng.normal()
                rows.append({"corpus": corp, "config": cfg, "alpha_cfg": x, "rep": rep, "n_tasks": 16,
                             "total_chars": 1e9, "log_rt_per_char": y, "runtime_s": np.exp(y) * 1e9,
                             "energy_kj_measured": np.exp(y) * 1e9 * 28.66 / 1000})
    return pd.DataFrame(rows)


def test_bhm_unseen_corpus_shares_population_draw(hf_df):
    stage = STAGES["hf"]
    m = BHMPredictor(stage, **FAST).fit(hf_df[hf_df.corpus != "c3"])
    test = hf_df[hf_df.corpus == "c3"].reset_index(drop=True)
    y = m.predict(test, 300, np.random.default_rng(0))
    assert y.shape == (300, len(test))
    # rows of the same config differ only by tau noise, so they are strongly
    # correlated across samples via the shared corpus draw
    same_cfg = test.index[test.config == "ro_alpha080"]
    assert np.corrcoef(y[:, same_cfg[0]], y[:, same_cfg[1]])[0, 1] > 0.5
    assert {"divergences", "rhat_max"} <= set(m.diagnostics())


def test_baselines_return_samples_and_no_pooling_skips_unseen(hf_df):
    stage = STAGES["hf"]
    train, test = hf_df[hf_df.corpus != "c3"], hf_df[hf_df.corpus == "c3"].reset_index(drop=True)
    for name, factory in build_baselines(stage).items():
        y = factory().fit(train).predict(test, 50, np.random.default_rng(0))
        assert y.shape == (50, len(test)), name
        assert np.isnan(y).all() if name == "no_pooling" else np.isfinite(y).all(), name


def test_power_model_recovers_hf_power(hf_df):
    pm = PowerModel(STAGES["hf"]).fit(hf_df)
    assert pm.params["mean"] == pytest.approx(28.66, rel=1e-6)
    P = pm.sample(hf_df, 10, np.random.default_rng(0))
    assert P.shape == (10, len(hf_df))


def test_power_model_sn_fits_two_components_per_form():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"form_idx": np.repeat([0, 1], 200), "runtime_s": 1000.0})
    P = np.where(rng.random(400) < 0.5, rng.normal(22, 1, 400), rng.normal(30, 1, 400))
    df["energy_kj_measured"] = P * df.runtime_s / 1000
    pm = PowerModel(STAGES["sn"]).fit(df)
    assert set(pm.params) == {0, 1}
    assert pm.params[0]["means"] == pytest.approx([22, 30], abs=0.5)


def test_run_lodo_anchored_puts_probes_in_training_and_not_in_test(hf_df):
    stage = STAGES["hf"]
    probes = hf_df[(hf_df.corpus == "c3") & (hf_df.rep == 1)]
    res = run_lodo(hf_df, stage, build_baselines(stage, only=["complete_pooling", "no_pooling"]),
                   held_out=["c3"], probes=probes, n_samples=100, log=lambda s: None)
    an = res[res.condition == "anchored"]
    assert (an.n_train == (hf_df.corpus != "c3").sum() + 3).all()
    assert (an.n_test == 3).all()
    assert an.set_index("model").loc["no_pooling", "rt_mape"] < 5


def test_run_lodo_without_probes_is_zero_shot_only(hf_df):
    stage = STAGES["hf"]
    res = run_lodo(hf_df, stage, build_baselines(stage, only=["complete_pooling"]),
                   held_out=["c0", "c3"], probes=None, n_samples=100, log=lambda s: None)
    assert set(res.condition) == {"zero_shot"}
    assert {"rt_mape", "rt_mape_mean", "e_mape", "e_mape_mean", "power_params"} <= set(res.columns)


def test_cell_mape_and_variance_shares():
    assert cell_mape(np.array([110.0, 90.0, 200.0]), np.array([100.0, 100.0, 100.0]),
                     np.array(["a", "a", "b"])) == pytest.approx(50.0)
    rng = np.random.default_rng(0)
    R = rng.normal(100, 10, size=(20000, 2))
    P = rng.normal(30, 3, size=(20000, 2))
    hw, add = variance_shares(R, P, R * P / 1000)
    assert hw == pytest.approx(0.5, abs=0.03)
    assert add == pytest.approx(1.0, abs=0.05)
