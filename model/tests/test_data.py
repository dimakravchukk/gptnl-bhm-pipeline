"""tests/test_data.py — pytest, synthetic fixtures only, no real data required."""
import pandas as pd
import pytest

from data import (
    parse_runtime, reader_block, strip_task_suffix,
    make_design_arrays, RuntimeDataset,
    load_str_norm_jobs,
)


# ── parsing helpers ──────────────────────────────────────────────────────────

def test_parse_runtime_full_format():
    assert parse_runtime("Total Runtime: 1 hour 23 minute 45 second") == 1*3600+23*60+45

def test_parse_runtime_seconds_only():
    assert parse_runtime("Total Runtime: 12.5 second") == 12.5

def test_parse_runtime_missing():
    assert parse_runtime("nothing here") is None

def test_parse_runtime_zero_is_none():
    assert parse_runtime("Total Runtime: 0 second") is None

def test_reader_block_finds_match():
    blocks = [{"name": "x"}, {"name": "READER: Parquet", "stats": {"doc_len": {"total": 5}}}]
    assert reader_block(blocks)["stats"]["doc_len"]["total"] == 5

def test_reader_block_no_match():
    assert reader_block([{"name": "FORMAT: FTFY"}]) is None

@pytest.mark.parametrize("name,expected", [
    ("am_stories_32t", "am_stories"),
    ("cc_github", "cc_github"),
    ("woogle_str_norm_8t", "woogle_str_norm"),
])
def test_strip_task_suffix(name, expected):
    assert strip_task_suffix(name) == expected


# ── make_design_arrays / RuntimeDataset ─────────────────────────────────────

def test_make_design_arrays_sorted_default():
    df = pd.DataFrame({"corpus": ["b", "a", "b"]})
    corpora, corp2idx = make_design_arrays(df)
    assert corpora == ["a", "b"]
    assert corp2idx == {"a": 0, "b": 1}

def test_make_design_arrays_explicit_order_not_overridden():
    df = pd.DataFrame({"corpus": ["unseen"]})
    fixed = ["am_stories", "cc_github"]
    corpora, _ = make_design_arrays(df, corpora=fixed)
    assert corpora == fixed  # must NOT be recomputed from df

def test_runtime_dataset_valid_construction():
    df = pd.DataFrame({"corpus": ["a", "b", "a"]})
    ds = RuntimeDataset(df=df, corpora=["a", "b"], corp2idx={"a": 0, "b": 1}, stage="x")
    assert list(ds.c_idx) == [0, 1, 0]

def test_runtime_dataset_rejects_unknown_corpus():
    df = pd.DataFrame({"corpus": ["a", "ghost"]})
    with pytest.raises(ValueError, match="not in corp2idx"):
        RuntimeDataset(df=df, corpora=["a"], corp2idx={"a": 0}, stage="x")


# ── load_str_norm_jobs (the one loader fully testable without filesystem mocks) ──

@pytest.fixture
def synthetic_task_csv(tmp_path):
    rows = []
    specs = [
        ("am_stories_NFC_8t_r001", "am_stories", "NFC", 8, False),
        ("am_stories_NFD_8t_r001", "am_stories", "NFD", 8, False),
        ("rechtspraak_NFC_16t_extension_r001", "rechtspraak", "NFC", 16, True),
        ("bad_form_r001", "am_stories", "ZZZ", 8, False),
    ]
    for run_id, corpus, form, n_tasks, _ in specs:
        for t in range(2):
            rows.append({
                "run_id": run_id, "dataset": corpus, "normalization_form": form,
                "hpc_n_tasks": n_tasks, "json_read_error": "",
                "reader_time_total_s": 1.0, "ftfy_time_total_s": 0.5,
                "punctuation_time_total_s": 0.2, "whitespace_time_total_s": 0.1,
                "writer_time_total_s": 0.3,
                "reader_doc_len_total_chars": 1000, "reader_docs_total": 10,
            })
    path = tmp_path / "tasks.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

def test_load_str_norm_jobs_excludes_extension_by_default(synthetic_task_csv):
    registry = [{"corpus": "am_stories"}, {"corpus": "rechtspraak"}]
    df = load_str_norm_jobs(synthetic_task_csv, ["NFC", "NFD", "NFKC"], registry)
    assert "rechtspraak" not in df["corpus"].values  # excluded: run_id has "extension"

def test_load_str_norm_jobs_keeps_extension_when_disabled(synthetic_task_csv):
    registry = [{"corpus": "am_stories"}, {"corpus": "rechtspraak"}]
    df = load_str_norm_jobs(synthetic_task_csv, ["NFC", "NFD", "NFKC"], registry,
                            exclude_run_id_pattern=None)
    assert "rechtspraak" in df["corpus"].values

def test_load_str_norm_jobs_drops_unknown_form(synthetic_task_csv):
    registry = [{"corpus": "am_stories"}]
    df = load_str_norm_jobs(synthetic_task_csv, ["NFC", "NFD", "NFKC"], registry)
    assert "ZZZ" not in df["normalization_form"].values

def test_load_str_norm_jobs_computes_log_rt_per_char(synthetic_task_csv):
    registry = [{"corpus": "am_stories"}]
    df = load_str_norm_jobs(synthetic_task_csv, ["NFC", "NFD", "NFKC"], registry)
    row = df[df["normalization_form"] == "NFC"].iloc[0]
    expected_runtime = (1.0 + 0.5 + 0.2 + 0.1 + 0.3) * 2  # 2 tasks
    expected_chars = 1000 * 2
    assert row["runtime_s"] == pytest.approx(expected_runtime)
    assert row["log_rt_per_char"] == pytest.approx(
        __import__("numpy").log(expected_runtime / expected_chars)
    )