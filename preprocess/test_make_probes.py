"""Tests for make_probes.py.  Run: pytest preprocess -v"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

import make_probes as mp

MOCK = Path(__file__).resolve().parent.parent / "mock_data"


def make_corpus(root: Path, n_files=3, n_docs=2000, seed=0):
    """Synthetic corpus: text not first in column order, zstd, some empty/null
    text, non-ASCII characters (so len(text) != bytes)."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    schema = pa.schema([("id", pa.string()), ("n_char", pa.int64()),
                        ("text", pa.string()), ("score", pa.float64())])
    k = 0
    for f in range(n_files):
        texts = []
        for _ in range(n_docs):
            r = rng.random()
            texts.append(None if r < 0.01 else "" if r < 0.02
                         else "é€" * int(rng.lognormal(5, 1)))
        t = pa.table({"id": [f"doc{k + i}" for i in range(n_docs)],
                      "n_char": [0] * n_docs, "text": texts,
                      "score": rng.random(n_docs)}, schema=schema)
        pq.write_table(t, root / f"part_{f:04d}.parquet", compression="zstd", row_group_size=700)
        k += n_docs
    (root / "README.txt").write_text("not parquet")


def write_config(tmp: Path, corpus_dir: Path, fracs=(0.02, 0.08, 0.32)) -> Path:
    cfg = {"master_seed": 20261022, "n_workers": 8, "size_fractions": list(fracs),
           "oversample": 1.5, "max_ratio": 1.10, "min_docs_per_worker": 4,
           "text_column": "text", "id_column": "id", "glob": "**/*.parquet",
           "output_root": str(tmp / "probes"),
           "inputs": {"string_normalization": {"synth": {"path": str(corpus_dir)}}}}
    p = tmp / "probes.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def built(tmp_path):
    make_corpus(tmp_path / "corpus")
    s = mp.load_config(write_config(tmp_path, tmp_path / "corpus"))
    rows = mp.build(s.jobs[0], s)
    return tmp_path, s, rows


def probe_paths(root, frac):
    return mp.probe_dir(root / "probes", "string_normalization", "synth", frac)


def test_pack_greedy_and_deterministic():
    chars = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 10])
    ids = [str(i) for i in range(len(chars))]
    a = mp.pack(chars, ids, 3)
    assert a == mp.pack(chars, ids, 3)
    assert sorted(i for s in a for i in s) == list(range(len(chars)))
    assert all(s == sorted(s) for s in a)                 # sampled order restored
    assert a[0][0] == 0 and 10 in a[1]                    # ties -> id order, lowest shard


def test_seed_stable_and_distinct():
    assert mp.derive_seed(1, "a", "x") == mp.derive_seed(1, "a", "x")
    assert mp.derive_seed(1, "a", "x") != mp.derive_seed(1, "a", "y")


def test_chars_match_python_len():
    texts = ["", "abc", "é€😀", "é́", None]
    got = mp.text_lengths(pa.array(texts)).tolist()
    assert got == [len(t) if t else 0 for t in texts]


def test_outputs_preserve_schema_and_compression(built):
    tmp, s, rows = built
    src = sorted((tmp / "corpus").glob("*.parquet"))[0]
    for r in rows:
        if not r["feasible"]:
            continue
        files = sorted((probe_paths(tmp, r["size_frac"]) / "data").iterdir())
        assert [f.name for f in files] == [f"{w:05d}.parquet" for w in range(8)]
        for f in files:
            assert pq.read_schema(f).equals(pq.read_schema(src))
            assert pq.ParquetFile(f).metadata.row_group(0).column(2).compression == "ZSTD"


def test_manifest_matches_shards(built):
    tmp, s, rows = built
    assert any(r["feasible"] for r in rows)
    for r in rows:
        d = probe_paths(tmp, r["size_frac"])
        if not r["feasible"]:
            assert not d.exists()
            continue
        m = json.loads((d / "manifest.json").read_text())
        total = 0
        for sh, ids in zip(m["shards"], m["doc_ids"]):
            t = pq.read_table(d / sh["file"])
            assert t["id"].to_pylist() == ids
            chars = sum(len(x) for x in t["text"].to_pylist())
            assert chars == sh["chars"] and t.num_rows == sh["n_docs"]
            assert all(x for x in t["text"].to_pylist())   # empty/null dropped
            total += chars
        assert total == m["actual_chars"] >= m["target_chars"]
        assert m["ratio"] <= 1.10
        assert min(sh["n_docs"] for sh in m["shards"]) >= 4


def test_nested_and_reproducible(built, tmp_path):
    tmp, s, rows = built
    feas = [r["size_frac"] for r in rows if r["feasible"]]
    assert len(feas) >= 2
    ids = [{i for sh in json.loads((probe_paths(tmp, f) / "manifest.json").read_text())["doc_ids"]
            for i in sh} for f in feas]
    assert all(a < b for a, b in zip(ids, ids[1:]))

    before = (probe_paths(tmp, feas[0]) / "data" / "00000.parquet").read_bytes()
    mp.build(s.jobs[0], s)
    assert (probe_paths(tmp, feas[0]) / "data" / "00000.parquet").read_bytes() == before


def test_one_file_per_rank_in_datatrove(built):
    io = pytest.importorskip("datatrove.io")
    tmp, s, rows = built
    r = next(r for r in rows if r["feasible"])
    folder = io.get_datafolder(str(probe_paths(tmp, r["size_frac"]) / "data"))
    shards = [folder.get_shard(rank, 8, recursive=True) for rank in range(8)]
    assert shards == [[f"{w:05d}.parquet"] for w in range(8)]


@pytest.mark.skipif(not (MOCK / "kb-pd-books").exists(), reason="mock data not built")
@pytest.mark.parametrize("corpus_dir", ["american-stories", "cc_github_open_source", "kb-pd-books",
                                        "naturalis", "tweedekamer"])
def test_balance_on_mock_corpora(tmp_path, corpus_dir):
    """Every probe that is written meets the constraints, for short- and
    long-document corpora; infeasible ones say why."""
    s = mp.load_config(write_config(tmp_path, MOCK / corpus_dir, fracs=(0.001, 0.004, 0.016, 0.064)))
    for r in mp.build(s.jobs[0], s):
        if r["feasible"]:
            assert r["ratio"] <= 1.10 and r["min_shard_docs"] >= 4
        else:
            assert r["reason"] != "ok"


def test_doc_keys_locate_source_rows(built):
    """doc_keys point at the exact source rows, even when ids are not unique."""
    tmp, s, rows = built
    r = next(r for r in rows if r["feasible"])
    d = probe_paths(tmp, r["size_frac"])
    m = json.loads((d / "manifest.json").read_text())
    src = {}
    for sh, keys in zip(m["shards"], m["doc_keys"]):
        shard_text = pq.read_table(d / sh["file"])["text"].to_pylist()
        for text, k in zip(shard_text, keys):
            f, row = k.rsplit(":", 1)
            if f not in src:
                src[f] = pq.read_table(tmp / "corpus" / f)["text"].to_pylist()
            assert src[f][int(row)] == text
    all_keys = [k for ks in m["doc_keys"] for k in ks]
    assert len(set(all_keys)) == len(all_keys)
