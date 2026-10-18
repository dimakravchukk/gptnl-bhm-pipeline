"""tests/test_energy.py — pytest, no real posteriors required."""
import numpy as np
import pandas as pd
import pytest

from energy import (
    hdi_94, decompose_variance_aggregate,
    predict_energy_hf, run_energy_mc_hf,
    sample_p_task_mixture, predict_energy_str_norm, run_energy_mc_str_norm,
)


# ── hdi_94 ───────────────────────────────────────────────────────────────────

def test_hdi_94_returns_floats_and_bracket_mean():
    rng = np.random.default_rng(0)
    samples = rng.normal(100, 5, size=5000)
    lo, hi = hdi_94(samples)
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < samples.mean() < hi


# ── predict_energy_hf ────────────────────────────────────────────────────────

@pytest.fixture
def fake_hf_posterior():
    S, J = 500, 3
    rng = np.random.default_rng(0)
    alpha_c_post = rng.normal(-13.0, 0.1, size=(S, J))
    beta_c_post  = rng.normal(-0.5, 0.05, size=(S, J))
    return alpha_c_post, beta_c_post

def test_predict_energy_hf_returns_kj_not_joules(fake_hf_posterior):
    alpha_c_post, beta_c_post = fake_hf_posterior
    result = predict_energy_hf(
        alpha_c_post, beta_c_post, corpus_idx=0, alpha_cfg=0.0,
        total_chars=1e9, p_task_mean=28.66, p_task_std=2.24,
        rng=np.random.default_rng(1),
    )
    # sanity: energy for ~1e9 chars at realistic rates should be O(1)-O(1000) kJ,
    # not O(1e6)-O(1e9) J — catches a units regression directly
    assert 0.01 < result["E_mean_kJ"] < 1e6

def test_predict_energy_hf_has_separate_rt_and_p_samples(fake_hf_posterior):
    alpha_c_post, beta_c_post = fake_hf_posterior
    result = predict_energy_hf(
        alpha_c_post, beta_c_post, corpus_idx=0, alpha_cfg=0.0,
        total_chars=1e9, p_task_mean=28.66, p_task_std=2.24,
        rng=np.random.default_rng(1),
    )
    assert "rt_samples" in result and "P_samples" in result
    assert len(result["rt_samples"]) == len(result["P_samples"]) == alpha_c_post.shape[0]
    # E_samples must equal the product of the two, confirming no hidden resampling
    np.testing.assert_allclose(
        result["E_samples"],
        result["rt_samples"] * result["P_samples"] / 1000.0,
    )

def test_predict_energy_hf_reproducible_with_seeded_rng(fake_hf_posterior):
    alpha_c_post, beta_c_post = fake_hf_posterior
    kwargs = dict(alpha_c_post=alpha_c_post, beta_c_post=beta_c_post,
                  corpus_idx=0, alpha_cfg=0.0, total_chars=1e9,
                  p_task_mean=28.66, p_task_std=2.24)
    r1 = predict_energy_hf(**kwargs, rng=np.random.default_rng(7))
    r2 = predict_energy_hf(**kwargs, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(r1["E_samples"], r2["E_samples"])

def test_predict_energy_hf_clips_power_samples(fake_hf_posterior):
    alpha_c_post, beta_c_post = fake_hf_posterior
    result = predict_energy_hf(
        alpha_c_post, beta_c_post, corpus_idx=0, alpha_cfg=0.0,
        total_chars=1e9, p_task_mean=28.66, p_task_std=50.0,  # huge std forces clipping
        p_task_min=10.0, p_task_max=50.0, rng=np.random.default_rng(1),
    )
    assert (result["P_samples"] >= 10.0).all()
    assert (result["P_samples"] <= 50.0).all()


def test_run_energy_mc_hf_skips_unknown_corpus(fake_hf_posterior, capsys):
    alpha_c_post, beta_c_post = fake_hf_posterior
    df_lookup = pd.DataFrame({
        "corpus": ["known", "unknown"],
        "config": ["c1", "c1"],
        "alpha_cfg": [0.0, 0.0],
        "total_chars": [1e9, 1e9],
    })
    df_energy = run_energy_mc_hf(
        df_lookup, alpha_c_post, beta_c_post, corpora=["known", "other", "third"],
        p_task_mean=28.66, p_task_std=2.24,
    )
    assert len(df_energy) == 1
    assert "skipped 1 rows" in capsys.readouterr().out


# ── sample_p_task_mixture / predict_energy_str_norm ─────────────────────────

@pytest.fixture
def fake_mixture():
    return {
        0: {"means": [22.0, 29.0], "stds": [2.0, 2.0], "weights": [0.5, 0.5]},  # NFC
        1: {"means": [21.0, 28.0], "stds": [2.0, 2.0], "weights": [0.4, 0.6]},  # NFD
        2: {"means": [21.0, 29.0], "stds": [2.0, 2.0], "weights": [0.6, 0.4]},  # NFKC
    }

def test_sample_p_task_mixture_respects_weights(fake_mixture):
    rng = np.random.default_rng(0)
    samples = sample_p_task_mixture(0, fake_mixture, n_samples=20000, rng=rng)
    # mean should land between the two component means, near the weighted average
    expected_mean = 0.5 * 22.0 + 0.5 * 29.0
    assert samples.mean() == pytest.approx(expected_mean, abs=0.3)

def test_sample_p_task_mixture_is_bimodal_not_collapsed(fake_mixture):
    rng = np.random.default_rng(0)
    samples = sample_p_task_mixture(0, fake_mixture, n_samples=20000, rng=rng)
    # crude bimodality check: density in the trough between modes should be
    # lower than density near either mode
    hist, edges = np.histogram(samples, bins=40)
    mid_idx = len(hist) // 2
    assert hist[mid_idx] < max(hist[:mid_idx//2].max(), hist[mid_idx + mid_idx//2:].max())


@pytest.fixture
def fake_str_norm_posterior():
    S, J, K = 500, 3, 3
    rng = np.random.default_rng(0)
    alpha_c_post = rng.normal(-14.0, 0.1, size=(S, J))
    delta_corpus_post = rng.normal(0.05, 0.02, size=(S, J, K))
    delta_corpus_post[:, :, 0] = 0.0  # NFC baseline fixed at 0
    return alpha_c_post, delta_corpus_post

def test_predict_energy_str_norm_nfc_baseline_zero_offset(fake_str_norm_posterior, fake_mixture):
    alpha_c_post, delta_corpus_post = fake_str_norm_posterior
    samples_nfc = predict_energy_str_norm(
        alpha_c_post, delta_corpus_post, corpus_idx=0, form_idx=0,
        total_chars=1e9, p_task_mixture=fake_mixture, rng=np.random.default_rng(1),
    )
    assert samples_nfc["E_samples"].shape == (500,)
    assert (samples_nfc["E_samples"] > 0).all()
    # NFC is the baseline form: runtime depends on alpha_c alone
    np.testing.assert_allclose(samples_nfc["rt_samples"], np.exp(alpha_c_post[:, 0]) * 1e9, rtol=1e-6)

def test_run_energy_mc_str_norm_skips_unknown_pairs(fake_str_norm_posterior, fake_mixture, capsys):
    alpha_c_post, delta_corpus_post = fake_str_norm_posterior
    df_lookup = pd.DataFrame({
        "corpus": ["c0", "ghost"],
        "normalization_form": ["NFC", "NFC"],
        "total_chars": [1e9, 1e9],
    })
    df_energy = run_energy_mc_str_norm(
        df_lookup, alpha_c_post, delta_corpus_post,
        corpora=["c0", "c1", "c2"], form_map={"NFC": 0, "NFD": 1, "NFKC": 2},
        p_task_mixture=fake_mixture,
    )
    assert len(df_energy) == 1
    assert "skipped 1 rows" in capsys.readouterr().out


# ── decompose_variance_aggregate ─────────────────────────────────────────────

def test_decompose_variance_aggregate_shares_sum_to_one():
    rng = np.random.default_rng(0)
    rt_list = [rng.normal(100, 5, 500) for _ in range(5)]
    P_list  = [rng.normal(28, 2, 500) for _ in range(5)]
    result = decompose_variance_aggregate(rt_list, P_list)
    assert result["share_runtime"] + result["share_power"] == pytest.approx(1.0)

def test_decompose_variance_aggregate_additivity_near_one():
    rng = np.random.default_rng(0)
    rt_list = [rng.normal(100, 5, 2000) for _ in range(10)]
    P_list  = [rng.normal(28, 2, 2000) for _ in range(10)]
    result = decompose_variance_aggregate(rt_list, P_list)
    assert result["additivity_check"] == pytest.approx(1.0, abs=0.05)

def test_decompose_variance_aggregate_power_dominated_case():
    # large power variance, tiny runtime variance -> share_power should dominate
    rng = np.random.default_rng(0)
    rt_list = [rng.normal(100, 0.01, 1000) for _ in range(5)]  # ~no runtime variance
    P_list  = [rng.normal(28, 5, 1000) for _ in range(5)]       # large power variance
    result = decompose_variance_aggregate(rt_list, P_list)
    assert result["share_power"] > 0.95

def test_decompose_variance_aggregate_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        decompose_variance_aggregate([np.array([1.0])], [])

def test_decompose_variance_aggregate_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        decompose_variance_aggregate([], [])