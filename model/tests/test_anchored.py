"""tests/test_anchored.py — tests for fit_anchored (anchoring a new corpus)."""
import jax
import numpy as np
import pandas as pd
import pytest

from models import (
    model_offset_hf, model_offset_interaction_no_ntasks,
    fit_anchored, flatten_posterior,
)


# ── filtering ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_hf_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for corp, true_alpha in [("am_stories", -13.0), ("cc_github", -14.5)]:
        for cfg in [-0.2, -0.15, -0.05, 0.0]:
            for rep in range(3):
                rows.append({
                    "corpus": corp, "alpha_cfg": cfg,
                    "log_rt_per_char": true_alpha + 0.1 * cfg + 0.01 * rng.normal(),
                })
    path = tmp_path / "base_hf.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def probe_hf_csv(tmp_path):
    rng = np.random.default_rng(1)
    true_alpha_new = -13.8
    rows = [
        {"corpus": "new_corpus", "alpha_cfg": cfg,
         "log_rt_per_char": true_alpha_new + 0.1 * cfg + 0.01 * rng.normal()}
        for cfg in [-0.2, -0.15, -0.05, 0.0]
    ]
    path = tmp_path / "probe_hf.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fit_anchored_hf_adds_new_corpus(base_hf_csv, probe_hf_csv):
    dataset, idata = fit_anchored(
        str(base_hf_csv), str(probe_hf_csv),
        model_offset_hf, extra_design_cols=["alpha_cfg"],
        rng_key=jax.random.PRNGKey(0),
        num_warmup=200, num_samples=200, num_chains=2,  # fast settings for tests
    )
    assert "new_corpus" in dataset.corpora
    assert len(dataset.corpora) == 3

    alpha_c_post = flatten_posterior(idata, "alpha_c")
    assert alpha_c_post.shape[1] == 3

    new_idx = dataset.corp2idx["new_corpus"]
    assert -14.3 < alpha_c_post[:, new_idx].mean() < -13.3


def test_fit_anchored_rejects_corpus_already_in_base(base_hf_csv, tmp_path):
    bad_probe = pd.DataFrame([
        {"corpus": "am_stories", "alpha_cfg": 0.0, "log_rt_per_char": -13.0},
    ])
    path = tmp_path / "bad_probe.csv"
    bad_probe.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly one new corpus"):
        fit_anchored(str(base_hf_csv), str(path), model_offset_hf,
                    extra_design_cols=["alpha_cfg"], rng_key=jax.random.PRNGKey(0))


def test_fit_anchored_rejects_multiple_new_corpora(base_hf_csv, tmp_path):
    bad_probe = pd.DataFrame([
        {"corpus": "new_a", "alpha_cfg": 0.0, "log_rt_per_char": -13.0},
        {"corpus": "new_b", "alpha_cfg": 0.0, "log_rt_per_char": -13.0},
    ])
    path = tmp_path / "two_new.csv"
    bad_probe.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly one new corpus"):
        fit_anchored(str(base_hf_csv), str(path), model_offset_hf,
                    extra_design_cols=["alpha_cfg"], rng_key=jax.random.PRNGKey(0))


def test_fit_anchored_saves_when_requested(base_hf_csv, probe_hf_csv, tmp_path):
    nc_path = tmp_path / "fit_anchored.nc"
    dataset, idata = fit_anchored(
        str(base_hf_csv), str(probe_hf_csv),
        model_offset_hf, extra_design_cols=["alpha_cfg"],
        rng_key=jax.random.PRNGKey(0),
        num_warmup=200, num_samples=200, num_chains=2,
        save_to=str(nc_path),  # <-- this was missing
    )
    assert nc_path.exists()
    updated_base = pd.read_csv(base_hf_csv)
    assert "new_corpus" in updated_base["corpus"].values

# ── string normalisation ─────────────────────────────────────────────────────

@pytest.fixture
def base_sn_csv(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for corp in ["am_stories", "cc_github"]:
        for form_idx, form in enumerate(["NFC", "NFD", "NFKC"]):
            for rep in range(3):
                rows.append({
                    "corpus": corp, "form_idx": form_idx,
                    "log_rt_per_char": -14.0 + 0.05 * form_idx + 0.01 * rng.normal(),
                })
    path = tmp_path / "base_sn.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def probe_sn_csv(tmp_path):
    rng = np.random.default_rng(1)
    rows = [
        {"corpus": "new_corpus", "form_idx": form_idx,
         "log_rt_per_char": -13.8 + 0.05 * form_idx + 0.01 * rng.normal()}
        for form_idx in range(3)
    ]
    path = tmp_path / "probe_sn.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fit_anchored_str_norm_adds_new_corpus(base_sn_csv, probe_sn_csv):
    dataset, idata = fit_anchored(
        str(base_sn_csv), str(probe_sn_csv),
        model_offset_interaction_no_ntasks, extra_design_cols=["form_idx"],  # <-- fixed
        rng_key=jax.random.PRNGKey(0),
        num_warmup=200, num_samples=200, num_chains=2,
    )

    assert "new_corpus" in dataset.corpora
    delta_corpus_post = flatten_posterior(idata, "delta_corpus")
    n_samples = idata.posterior.dims["chain"] * idata.posterior.dims["draw"]
    assert delta_corpus_post.shape == (n_samples, len(dataset.corpora), 3)