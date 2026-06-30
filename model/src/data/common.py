"""Shared parsing helpers and the stage-agnostic RuntimeDataset container.

These utilities are used by both heuristic_filtering.py and
string_normalisation.py loaders. Stage-specific parsing logic lives in
those modules, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_RT_RE = re.compile(r"Total Runtime:\s*(.+)")


def parse_runtime(text: str) -> float | None:
    """Parse a 'Total Runtime: X hour Y minute Z second' string to seconds.

    Used only by the heuristic filtering loader (task log text format).
    String normalisation runtime comes from a pre-aggregated CSV with its
    own timing columns and does not use this parser.
    """
    m = _RT_RE.search(text)
    if not m:
        return None
    total = 0.0
    for v, u in re.findall(r"([\d.]+)\s*(hour|minute|second)", m.group(1)):
        total += float(v) * {"hour": 3600, "minute": 60, "second": 1}[u]
    return total or None


def reader_block(blocks: list[dict]) -> dict | None:
    """Return the first Parquet READER block from a stats JSON, or None."""
    for b in blocks:
        if "READER" in b.get("name", "") and "Parquet" in b.get("name", ""):
            return b
    return None


def strip_task_suffix(name: str) -> str:
    """Remove trailing _<N>t suffixes: 'am_stories_32t' → 'am_stories'."""
    return re.sub(r"_\d+t$", "", name)


def make_design_arrays(
    df: pd.DataFrame,
    corpus_col: str = "corpus",
    corpora: list[str] | None = None,
) -> tuple[list[str], dict[str, int]]:
    """Return (corpora, corp2idx) derived from df, never hardcoded.

    Pass corpora explicitly if you need a fixed order (e.g. to match a
    saved fit's posterior column ordering).

    WARNING: always pass the same `corpora` list that was used when the
    fit you're loading was originally produced. A mismatched order will
    silently misattribute posterior parameters between corpora — this is
    not caught automatically here and will not raise an error. Prefer
    RuntimeDataset, which validates this at construction time.
    """
    if corpora is None:
        corpora = sorted(df[corpus_col].unique())
    corp2idx = {c: i for i, c in enumerate(corpora)}
    return corpora, corp2idx


@dataclass
class RuntimeDataset:
    """Stage-agnostic container: a runtime/energy DataFrame plus its fixed
    corpus ordering, bound together so they cannot silently drift apart.

    Always pass `dataset.corpora` / `dataset.corp2idx` to model fitting
    and prediction calls — never reconstruct corpus ordering separately
    from the DataFrame, since that is exactly the failure mode this class
    exists to prevent.
    """
    df: pd.DataFrame
    corpora: list[str]
    corp2idx: dict[str, int]
    stage: str  # "heuristic_filtering" | "string_normalisation"

    @property
    def c_idx(self):
        return self.df["corpus"].map(self.corp2idx).values

    def __post_init__(self):
        unknown = set(self.df["corpus"].unique()) - set(self.corp2idx)
        if unknown:
            raise ValueError(
                f"RuntimeDataset: df contains corpora not in corp2idx: {unknown}. "
                f"This usually means corpora was passed explicitly but doesn't "
                f"match what's actually in the data."
            )

    def __repr__(self):
        return (f"RuntimeDataset(stage={self.stage!r}, n_obs={len(self.df)}, "
                f"n_corpora={len(self.corpora)})")