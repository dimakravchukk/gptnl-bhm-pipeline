"""Build probe shards for the anchoring experiments.

For each corpus x stage x size fraction, sample whole documents totalling a
target number of characters and split them over n_workers parquet shards with
near-equal character counts (one shard per pipeline rank).

    python make_probes.py --config probes.local.yaml [--stage S] [--corpus C] [--dry-run]

Sampling: one streaming Bernoulli pass (p = oversample x largest fraction),
then a seeded shuffle; each probe is the shortest prefix of the shuffled list
reaching its target, so smaller probes are subsets of larger ones.

Packing: largest-first greedy onto the lightest shard; a probe is feasible iff
max/min shard chars <= max_ratio and every shard has >= min_docs_per_worker
documents. Infeasible probes are not written.

Output (per probe):
    {output_root}/{stage}/{corpus}/size_{frac}/data/00000.parquet ... 0000{n-1}.parquet
    {output_root}/{stage}/{corpus}/size_{frac}/manifest.json
    {output_root}/summary.csv

Shards sit in data/ so that the pipeline's input_folder holds only the shard
files: datatrove lists input files recursively and sorted, without a glob by
default, and gives rank r files[r::world_size]. A manifest.json next to the
shards would be a ninth file assigned to rank 0.

Characters are len(text) on the decoded string (pyarrow utf8_length counts
code points, which equals Python len), the same as datatrove's doc_len stat
behind total_chars in the full-scale data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

log = logging.getLogger("make_probes")

BATCH_SIZE = 65_536
SUMMARY_FIELDS = [
    "stage", "corpus", "size_frac", "seed", "target_chars", "actual_chars",
    "full_corpus_chars", "scanned_corpus_chars", "cost_frac", "n_docs",
    "min_shard_docs", "ratio", "feasible", "n_empty_dropped", "p_bernoulli",
    "reason",
]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Job:
    stage: str
    corpus: str
    input_root: Path
    text_column: str
    id_column: str
    glob: str
    full_corpus_chars: int | None


@dataclass
class Settings:
    master_seed: int
    n_workers: int
    size_fractions: list[float]
    oversample: float
    max_ratio: float
    min_docs_per_worker: int
    output_root: Path
    jobs: list[Job] = field(default_factory=list)


def load_config(path: Path) -> Settings:
    cfg = yaml.safe_load(path.read_text())
    base = path.resolve().parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else base / p

    s = Settings(
        master_seed=int(cfg["master_seed"]),
        n_workers=int(cfg["n_workers"]),
        size_fractions=sorted(float(f) for f in cfg["size_fractions"]),
        oversample=float(cfg.get("oversample", 1.5)),
        max_ratio=float(cfg["max_ratio"]),
        min_docs_per_worker=int(cfg["min_docs_per_worker"]),
        output_root=resolve(cfg["output_root"]),
    )
    for stage, corpora in cfg["inputs"].items():
        for corpus, entry in corpora.items():
            s.jobs.append(Job(
                stage=stage,
                corpus=corpus,
                input_root=resolve(entry["path"]),
                text_column=entry.get("text_column", cfg.get("text_column", "text")),
                id_column=entry.get("id_column", cfg.get("id_column", "id")),
                glob=entry.get("glob", cfg.get("glob", "**/*.parquet")),
                full_corpus_chars=entry.get("full_corpus_chars"),
            ))
    return s


def derive_seed(master_seed: int, corpus: str, stage: str) -> int:
    """Stable across processes (unlike hash()), so every config of a corpus x
    stage sees the same sample."""
    h = hashlib.sha256(f"{master_seed}:{corpus}:{stage}".encode()).digest()
    return int.from_bytes(h[:8], "big")


# --------------------------------------------------------------------------
# 1. Sampling
# --------------------------------------------------------------------------

@dataclass
class Candidates:
    """Bernoulli-kept documents in stream order."""
    file_idx: np.ndarray
    row_idx: np.ndarray
    chars: np.ndarray       # 0 for empty / null text
    ids: list[str]
    scanned_chars: int      # len(text) summed over the whole corpus
    scanned_docs: int
    n_empty_total: int      # empty / null documents in the whole corpus


def list_source_files(job: Job) -> list[Path]:
    files = sorted(job.input_root.glob(job.glob), key=lambda f: f.relative_to(job.input_root).as_posix())
    if not files:
        raise FileNotFoundError(f"no files matching {job.glob!r} under {job.input_root}")
    return files


def text_lengths(col: pa.ChunkedArray | pa.Array) -> np.ndarray:
    return pc.fill_null(pc.utf8_length(col), 0).to_numpy(zero_copy_only=False).astype(np.int64)


def scan(files: list[Path], job: Job, p: float, seed: int) -> Candidates:
    rng = np.random.default_rng([seed, 0])
    fi_parts, ri_parts, ch_parts, ids = [], [], [], []
    scanned_chars = scanned_docs = n_empty = 0

    for fi, path in enumerate(files):
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        if job.text_column not in names:
            raise KeyError(f"{path}: no column {job.text_column!r} (columns: {names})")
        has_id = job.id_column in names
        cols = [job.text_column] + ([job.id_column] if has_id else [])

        row0 = 0
        for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
            n = batch.num_rows
            lens = text_lengths(batch.column(job.text_column))
            scanned_chars += int(lens.sum())
            scanned_docs += n
            n_empty += int((lens == 0).sum())

            keep = np.nonzero(rng.random(n) < p)[0]
            if keep.size:
                fi_parts.append(np.full(keep.size, fi, dtype=np.int32))
                ri_parts.append(keep + row0)
                ch_parts.append(lens[keep])
                if has_id:
                    id_col = batch.column(job.id_column).take(pa.array(keep))
                    ids.extend(str(v) for v in id_col.to_pylist())
                else:
                    ids.extend(f"{path.name}:{r}" for r in keep + row0)
            row0 += n

    cat = lambda parts, dt: np.concatenate(parts) if parts else np.empty(0, dtype=dt)
    return Candidates(
        file_idx=cat(fi_parts, np.int32), row_idx=cat(ri_parts, np.int64),
        chars=cat(ch_parts, np.int64), ids=ids,
        scanned_chars=scanned_chars, scanned_docs=scanned_docs, n_empty_total=n_empty,
    )


def shuffled_order(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, 1]).permutation(n)


def prefix_length(cum_chars: np.ndarray, target: int) -> int | None:
    """Shortest prefix with cumulative chars >= target, or None if too short."""
    if cum_chars.size == 0 or cum_chars[-1] < target:
        return None
    return int(np.searchsorted(cum_chars, target, side="left")) + 1


# --------------------------------------------------------------------------
# 2. Packing
# --------------------------------------------------------------------------

def pack(chars: np.ndarray, ids: list[str], n_workers: int) -> list[list[int]]:
    """Largest-first greedy. Returns per shard the indices into `chars`, in
    their original (sampled) order."""
    order = sorted(range(len(chars)), key=lambda i: (-int(chars[i]), ids[i], i))
    heap = [(0, s) for s in range(n_workers)]   # (load, shard): ties -> lowest index
    shards: list[list[int]] = [[] for _ in range(n_workers)]
    for i in order:
        load, s = heapq.heappop(heap)
        shards[s].append(i)
        heapq.heappush(heap, (load + int(chars[i]), s))
    return [sorted(sh) for sh in shards]


# --------------------------------------------------------------------------
# 3. Balance check
# --------------------------------------------------------------------------

def balance(shard_chars: list[int], shard_docs: list[int], max_ratio: float, min_docs: int):
    lo, hi = min(shard_chars), max(shard_chars)
    ratio = hi / lo if lo > 0 else float("inf")
    reasons = []
    if ratio > max_ratio:
        reasons.append(f"ratio {ratio:.3f} > {max_ratio}")
    if min(shard_docs) < min_docs:
        reasons.append(f"min shard docs {min(shard_docs)} < {min_docs}")
    return ratio, not reasons, "; ".join(reasons) or "ok"


# --------------------------------------------------------------------------
# 4. Output
# --------------------------------------------------------------------------

def source_compression(path: Path) -> dict[str, str]:
    md = pq.ParquetFile(path).metadata
    if md.num_row_groups == 0:
        return {}
    rg = md.row_group(0)
    out = {}
    for j in range(rg.num_columns):
        c = rg.column(j)
        codec = c.compression.lower()
        out[c.path_in_schema] = "none" if codec == "uncompressed" else codec
    return out


def gather_rows(files: list[Path], cand: Candidates, positions: np.ndarray,
                schema: pa.Schema) -> tuple[pa.Table, dict[int, int]]:
    """Read the candidates at `positions` (indices into cand) from the source
    files. Returns the rows and a map candidate position -> row in that table."""
    tables, pos_map, n_out = [], {}, 0
    by_file: dict[int, list[int]] = {}
    for pos in positions:
        by_file.setdefault(int(cand.file_idx[pos]), []).append(int(pos))

    for fi in sorted(by_file):
        pf = pq.ParquetFile(files[fi])
        rg_starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows
                                     for g in range(pf.metadata.num_row_groups)])
        wanted = sorted(by_file[fi], key=lambda q: cand.row_idx[q])
        rgs = np.searchsorted(rg_starts, [cand.row_idx[q] for q in wanted], side="right") - 1
        for g in np.unique(rgs):
            in_g = [q for q, r in zip(wanted, rgs) if r == g]
            local = pa.array([int(cand.row_idx[q] - rg_starts[g]) for q in in_g])
            t = pf.read_row_group(int(g)).take(local)
            t = t.select(schema.names).cast(schema)
            tables.append(t)
            for q in in_g:
                pos_map[q] = n_out
                n_out += 1
    table = pa.concat_tables(tables) if tables else schema.empty_table()
    return table, pos_map


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def probe_dir(root: Path, stage: str, corpus: str, frac: float) -> Path:
    return root / stage / corpus / f"size_{frac:g}"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def build(job: Job, s: Settings, dry_run: bool = False) -> list[dict]:
    files = list_source_files(job)
    seed = derive_seed(s.master_seed, job.corpus, job.stage)
    log.info("[%s/%s] %d source files under %s, seed %d",
             job.stage, job.corpus, len(files), job.input_root, seed)

    p = min(1.0, s.oversample * max(s.size_fractions))
    while True:
        cand = scan(files, job, p, seed)
        full = int(job.full_corpus_chars) if job.full_corpus_chars else cand.scanned_chars
        largest_target = int(np.ceil(max(s.size_fractions) * full))
        if cand.chars.sum() >= largest_target or p >= 1.0:
            break
        p = min(1.0, 2 * p)
        log.info("[%s/%s] candidates too short, retrying with p=%.4f", job.stage, job.corpus, p)

    if job.full_corpus_chars:
        rel = cand.scanned_chars / full - 1
        if abs(rel) > 0.005:
            log.warning("[%s/%s] scanned chars %d differ from full_corpus_chars %d by %+.2f%%: "
                        "is the input path the one the full-scale runs read?",
                        job.stage, job.corpus, cand.scanned_chars, full, 100 * rel)
    log.info("[%s/%s] scanned %d docs / %d chars (%d empty); kept %d candidates at p=%.4f",
             job.stage, job.corpus, cand.scanned_docs, cand.scanned_chars,
             cand.n_empty_total, len(cand.chars), p)

    perm = shuffled_order(len(cand.chars), seed)
    cum = np.cumsum(cand.chars[perm])

    probes = []
    for frac in s.size_fractions:
        target = int(np.ceil(frac * full))
        row = dict(stage=job.stage, corpus=job.corpus, size_frac=frac, seed=seed,
                   target_chars=target, full_corpus_chars=full,
                   scanned_corpus_chars=cand.scanned_chars, p_bernoulli=p)
        k = prefix_length(cum, target)
        if k is None:
            row.update(actual_chars=int(cum[-1]) if cum.size else 0, cost_frac=None, n_docs=0,
                       min_shard_docs=0, ratio=None, feasible=False, n_empty_dropped=0,
                       reason="insufficient candidates (corpus smaller than target)")
            probes.append((row, None))
            continue

        prefix = perm[:k]
        nonempty = prefix[cand.chars[prefix] > 0]   # still in sampled order
        n_empty = k - len(nonempty)
        sub_chars = cand.chars[nonempty]
        sub_ids = [cand.ids[q] for q in nonempty]
        shards = pack(sub_chars, sub_ids, s.n_workers)
        shard_chars = [int(sub_chars[sh].sum()) for sh in shards]
        shard_docs = [len(sh) for sh in shards]
        ratio, feasible, reason = balance(shard_chars, shard_docs, s.max_ratio, s.min_docs_per_worker)
        actual = int(sub_chars.sum())
        row.update(actual_chars=actual, cost_frac=actual / full, n_docs=len(nonempty),
                   min_shard_docs=min(shard_docs), ratio=ratio, feasible=feasible,
                   n_empty_dropped=n_empty, reason=reason)
        probes.append((row, (nonempty, shards, shard_chars)))
        log.info("[%s/%s] size %g: %d docs, %d chars, ratio %.4f, min docs %d -> %s",
                 job.stage, job.corpus, frac, len(nonempty), actual, ratio,
                 min(shard_docs), "feasible" if feasible else f"INFEASIBLE ({reason})")

    for row, _ in probes:
        d = probe_dir(s.output_root, job.stage, job.corpus, row["size_frac"])
        if not dry_run and d.exists():
            shutil.rmtree(d)

    to_write = [(row, sel) for row, sel in probes if row["feasible"]]
    if dry_run or not to_write:
        return [row for row, _ in probes]

    # Nested samples: the largest feasible probe holds every row we need.
    union = max((sel[0] for _, sel in to_write), key=len)
    schema = pq.ParquetFile(files[0]).schema_arrow
    compression = source_compression(files[0])
    rows, pos_map = gather_rows(files, cand, union, schema)
    commit, stamp = git_hash(), datetime.now(timezone.utc).isoformat()
    rel_files = [f.relative_to(job.input_root).as_posix() for f in files]
    key = lambda q: f"{rel_files[cand.file_idx[q]]}:{cand.row_idx[q]}"
    ids_unique = len(set(cand.ids)) == len(cand.ids)

    for row, (nonempty, shards, shard_chars) in to_write:
        d = probe_dir(s.output_root, job.stage, job.corpus, row["size_frac"])
        (d / "data").mkdir(parents=True)
        shard_info = []
        for w, sh in enumerate(shards):
            name = f"{w:05d}.parquet"
            take = pa.array([pos_map[int(nonempty[i])] for i in sh], type=pa.int64())
            pq.write_table(rows.take(take), d / "data" / name, compression=compression)
            shard_info.append({"file": f"data/{name}", "n_docs": len(sh), "chars": shard_chars[w],
                               "doc_ids": [cand.ids[int(nonempty[i])] for i in sh],
                               "doc_keys": [key(int(nonempty[i])) for i in sh]})
        manifest = {
            **{k: row[k] for k in ("corpus", "stage", "size_frac", "seed", "target_chars",
                                   "actual_chars", "full_corpus_chars", "cost_frac", "ratio",
                                   "feasible", "n_empty_dropped")},
            "scanned_corpus_chars": cand.scanned_chars,
            "p_bernoulli": p,
            "n_workers": s.n_workers,
            "text_column": job.text_column,
            "id_column": job.id_column,
            "input_root": str(job.input_root),
            "shards": [{k: v for k, v in sh.items() if k not in ("doc_ids", "doc_keys")}
                       for sh in shard_info],
            # id_column values are not per-document in most GPT-NL corpora;
            # doc_keys ("<source file>:<row>") locate each document exactly.
            "ids_unique": ids_unique,
            "doc_ids": [sh["doc_ids"] for sh in shard_info],
            "doc_keys": [sh["doc_keys"] for sh in shard_info],
            "source_files": rel_files,
            "script_git_hash": commit,
            "timestamp": stamp,
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return [row for row, _ in probes]


def write_summary(root: Path, rows: list[dict]) -> Path:
    path = root / "summary.csv"
    key = lambda r: (r["stage"], r["corpus"], f"{float(r['size_frac']):g}")
    new = {key(r): r for r in rows}
    kept = []
    if path.exists():
        with path.open() as f:
            kept = [r for r in csv.DictReader(f) if key(r) not in new]
    root.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(kept + rows, key=lambda r: (*key(r)[:2], float(r["size_frac"]))):
            w.writerow(r)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--stage", action="append", help="only these stages (repeatable)")
    ap.add_argument("--corpus", action="append", help="only these corpora (repeatable)")
    ap.add_argument("--dry-run", action="store_true", help="sample and pack, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = load_config(args.config)
    jobs = [j for j in s.jobs
            if (not args.stage or j.stage in args.stage)
            and (not args.corpus or j.corpus in args.corpus)]
    rows = []
    for job in jobs:
        rows.extend(build(job, s, dry_run=args.dry_run))
    if not args.dry_run:
        log.info("summary: %s", write_summary(s.output_root, rows))
    n_ok = sum(r["feasible"] for r in rows)
    log.info("%d/%d probes feasible", n_ok, len(rows))


if __name__ == "__main__":
    main()
