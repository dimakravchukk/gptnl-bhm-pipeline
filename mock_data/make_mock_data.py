"""Build small local mock copies of the GPT-NL Public Corpus inputs.

For each corpus in the analysis, download one source parquet file from
huggingface.co/datasets/GPT-NL/GPT-NL_Public_Corpus and write a truncated copy
split over N_FILES files, keeping the HF directory layout, schema and column
order:

    mock_data/<hf_dir>/<hf_dir>_0001.parquet ... _000N.parquet

This is post-curation data: use it to build and debug the preprocessor, not
for measurements (filtering has already removed rejects, text is already
normalised).

Usage:
    python make_mock_data.py [--raw-dir DIR] [--out-dir DIR] [--max-chars N]
"""

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "GPT-NL/GPT-NL_Public_Corpus"
TEXT_COLUMN = "text"
N_FILES = 4

# Thesis short name -> HF directory. kb_books is kb-pd-books (see
# manage/datasets.yaml), not kb.
CORPORA = {
    "am_stories": "american-stories",
    "cc_github": "cc_github_open_source",
    "cc_openalex": "cc_openalex",
    "eu_parliament": "european-parliament",
    "kb_books": "kb-pd-books",
    "naturalis": "naturalis",
    "rechtspraak": "rechtspraak",
    "tweedekamer": "tweedekamer",
    "woogle": "woogle",
}


def build_corpus(short, hf_dir, raw_dir, out_dir, max_chars):
    src = hf_hub_download(
        REPO_ID, f"{hf_dir}/{hf_dir}_0001.parquet",
        repo_type="dataset", local_dir=raw_dir,
    )
    table = pq.read_table(src)

    # len(text) on the decoded string, same as the pipeline's total_chars.
    lengths = pc.utf8_length(table[TEXT_COLUMN]).to_pylist()

    # Keep whole documents until the char budget is reached, but always
    # enough for every output file to get at least 64 documents.
    n_rows, total = 0, 0
    for n in lengths:
        if total >= max_chars and n_rows >= 64 * N_FILES:
            break
        total += n
        n_rows += 1
    table = table.slice(0, n_rows)
    lengths = lengths[:n_rows]

    corpus_out = out_dir / hf_dir
    corpus_out.mkdir(parents=True, exist_ok=True)
    for old in corpus_out.glob("*.parquet"):
        old.unlink()
    per_file = -(-n_rows // N_FILES)
    for i in range(N_FILES):
        part = table.slice(i * per_file, per_file)
        if part.num_rows:
            pq.write_table(part, corpus_out / f"{hf_dir}_{i + 1:04d}.parquet")
    (corpus_out / "ALL_PARQUET_FILES_VALID").touch()

    n_char = table["n_char"].to_pylist()
    return {
        "short_name": short,
        "hf_dir": hf_dir,
        "source_file": f"{hf_dir}/{hf_dir}_0001.parquet",
        "columns": table.schema.names,
        "text_column": TEXT_COLUMN,
        "n_docs": n_rows,
        "total_chars": total,
        "mean_chars": total / n_rows,
        "max_chars": max(lengths),
        "n_char_matches_len_text": sum(a == b for a, b in zip(n_char, lengths)),
        "dataset_name_values": sorted(set(table["dataset_name"].to_pylist())),
    }


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=here / ".raw")
    ap.add_argument("--out-dir", type=Path, default=here)
    ap.add_argument("--max-chars", type=int, default=50_000_000)
    args = ap.parse_args()

    manifest = []
    for short, hf_dir in CORPORA.items():
        info = build_corpus(short, hf_dir, args.raw_dir, args.out_dir, args.max_chars)
        print(f"{short:14s} {info['n_docs']:7d} docs {info['total_chars']:>12,d} chars "
              f"max_doc={info['max_chars']:,d} "
              f"n_char==len {info['n_char_matches_len_text']}/{info['n_docs']}")
        manifest.append(info)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
