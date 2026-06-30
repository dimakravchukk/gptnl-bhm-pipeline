"""tests/test_models.py — pytest, fast (small synthetic fits, no real data)."""
import jax
import numpy as np
import pytest

from models import (
    model_offset_hf, model_offset_interaction_no_ntasks,
    fit_mcmc, flatten_posterior, _validate_index_range,
    save_fit, load_and_validate_fit, MODEL_VARS,
)


# ── _validate_index_range ────────────────────────────────────────────────────

def test_validate_index_range_passes_for_valid_indices():
    _validate_index_range(np.array([0, 1, 2]), expected_max_exclusive=3,
                          idx_name="c_idx", max_name="J")  # should not raise

def test_validate_index_range_rejects_out_of_bounds():
    with pytest.raises(ValueError, match="outside"):
        _validate_index_range(np.array([0, 1, 5]), expected_max_exclusive=3,
                              idx_name="c_idx", max_name="J")

def test_validate_index_range_rejects_negative():
    with pytest.raises(ValueError, match="outside"):
        _validate_index_range(np.array([-1, 0]), expected_max_exclusive=3,
                              idx_name="c_idx", max_name="J")

def test_validate_index_range_empty_array_is_noop():
    _validate_index_range(np.array([]), expected_max_exclusive=3,
                          idx_name="c_idx", max_name="J")  # should not raise


# ── tiny end-to-end fits (small N, few draws — fast, just checks plumbing) ──

@pytest.fixture
def tiny_hf_data():
    rng = np.random.default_rng(0)
    J = 2
    c_idx = np.array([0, 0, 1, 1] * 3)
    alpha_cfg = np.tile([-0.2, 0.0], 6)
    true_alpha = np.array([-13.0, -14.0])
    y = true_alpha[c_idx] + 0.01 * rng.normal(size=len(c_idx))
    return c_idx, alpha_cfg, J, y

def test_model_offset_hf_fits_and_recovers_reasonable_alpha(tiny_hf_data):
    c_idx, alpha_cfg, J, y = tiny_hf_data
    mcmc, idata = fit_mcmc(
        model_offset_hf, jax.random.PRNGKey(0), c_idx, alpha_cfg, J, y=y,
        num_warmup=200, num_samples=200, num_chains=2,
    )
    alpha_c_post = flatten_posterior(idata, "alpha_c")
    assert alpha_c_post.shape == (400, J)
    # recovered alpha should be in the right ballpark, not exact given tiny N
    assert -14.5 < alpha_c_post[:, 1].mean() < -13.5

def test_model_offset_hf_raises_on_bad_c_idx(tiny_hf_data):
    c_idx, alpha_cfg, J, y = tiny_hf_data
    bad_c_idx = c_idx.copy()
    bad_c_idx[0] = 99  # out of range for J=2
    with pytest.raises(ValueError, match="outside"):
        fit_mcmc(model_offset_hf, jax.random.PRNGKey(0), bad_c_idx, alpha_cfg, J, y=y,
                 num_warmup=10, num_samples=10, num_chains=1)


@pytest.fixture
def tiny_str_norm_data():
    rng = np.random.default_rng(0)
    J, K = 2, 3
    c_idx = np.tile([0, 0, 0, 1, 1, 1], 2)
    form_idx = np.tile([0, 1, 2], 4)
    true_alpha = np.array([-13.9, -14.1])
    y = true_alpha[c_idx] + 0.01 * rng.normal(size=len(c_idx))
    return c_idx, form_idx, J, K, y

def test_model_offset_interaction_fits_and_has_delta_corpus(tiny_str_norm_data):
    c_idx, form_idx, J, K, y = tiny_str_norm_data
    mcmc, idata = fit_mcmc(
        model_offset_interaction_no_ntasks, jax.random.PRNGKey(0),
        c_idx, form_idx, J, K, y=y,
        num_warmup=200, num_samples=200, num_chains=2,
    )
    delta_corpus_post = flatten_posterior(idata, "delta_corpus")
    assert delta_corpus_post.shape == (400, J, K)
    # NFC baseline (k=0) should be exactly zero for every corpus/draw
    np.testing.assert_allclose(delta_corpus_post[:, :, 0], 0.0)


# ── save_fit / load_and_validate_fit ─────────────────────────────────────────

def test_save_and_load_roundtrip(tiny_hf_data, tmp_path):
    c_idx, alpha_cfg, J, y = tiny_hf_data
    _, idata = fit_mcmc(model_offset_hf, jax.random.PRNGKey(0), c_idx, alpha_cfg, J, y=y,
                        num_warmup=50, num_samples=50, num_chains=1)
    path = tmp_path / "fit.nc"
    save_fit(idata, path, "model_offset_hf")
    loaded = load_and_validate_fit(path, "model_offset_hf")
    assert set(MODEL_VARS["model_offset_hf"]) <= set(loaded.posterior.data_vars)

def test_load_and_validate_fit_rejects_wrong_model(tiny_hf_data, tmp_path):
    c_idx, alpha_cfg, J, y = tiny_hf_data
    _, idata = fit_mcmc(model_offset_hf, jax.random.PRNGKey(0), c_idx, alpha_cfg, J, y=y,
                        num_warmup=50, num_samples=50, num_chains=1)
    path = tmp_path / "fit.nc"
    save_fit(idata, path, "model_offset_hf")
    with pytest.raises(ValueError, match="model_name"):
        load_and_validate_fit(path, "model_offset_interaction_no_ntasks")

def test_save_fit_rejects_unknown_model_name(tiny_hf_data, tmp_path):
    c_idx, alpha_cfg, J, y = tiny_hf_data
    _, idata = fit_mcmc(model_offset_hf, jax.random.PRNGKey(0), c_idx, alpha_cfg, J, y=y,
                        num_warmup=50, num_samples=50, num_chains=1)
    with pytest.raises(ValueError, match="Unknown model_name"):
        save_fit(idata, tmp_path / "fit.nc", "not_a_real_model")