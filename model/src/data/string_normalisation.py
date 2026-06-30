"""String normalisation stage: load runtime and EAR energy data from
pre-extracted CSVs.

DATA QUALITY NOTE — "extension" batch:
A subset of string-normalisation runs (run_id containing "extension") was
found to have unreliable job-level energy attribution — the per-task EAR
sum disagreed with the job-level energy field by up to several hundred
percent at high n_tasks, while agreeing within ~5-8% for the rest of the
dataset. The root cause was not fully resolved. load_str_norm_jobs
excludes these run_ids by default; pass exclude_run_id_pattern=None to
disable.

  load_str_norm_jobs(task_stats_csv, norm_forms, corpus_registry)
      → DataFrame  (one row per job)

  load_str_norm_ear(ear_csv)
      → DataFrame  (matched, full-node EAR power readings)

  load_str_norm_dataset(...) → RuntimeDataset  (load_str_norm_jobs + design arrays, bundled)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import make_design_arrays, RuntimeDataset

DEFAULT_EXCLUDE_RUN_ID_PATTERN = "extension"


def load_str_norm_jobs(
    task_stats_csv: Path | str,
    norm_forms: list[str],
    corpus_registry: list[dict],
    exclude_run_id_pattern: str | None = DEFAULT_EXCLUDE_RUN_ID_PATTERN,
) -> pd.DataFrame:
    """Aggregate task-level string-normalisation CSV to one row per job.

    Parameters
    ----------
    task_stats_csv  : path to stringnorm_task_level_stats_modules.csv
    norm_forms      : ordered list of forms, e.g. ["NFC", "NFD", "NFKC"] (index 0 = baseline)
    corpus_registry : list of dicts with at least {"corpus": <name>}
    exclude_run_id_pattern : substring match on run_id to exclude (default:
        "extension" — see module docstring). Pass None to disable.

    Returned columns:
        run_id, corpus, normalization_form, n_tasks, replicate,
        runtime_s, total_chars, total_docs, is_anchor,
        form_idx, log_rt_per_char, log_n_tasks
    """
    df_tasks = pd.read_csv(task_stats_csv)
    n_raw = len(df_tasks)
    df_tasks = df_tasks[
        df_tasks["json_read_error"].isna() | (df_tasks["json_read_error"] == "")
    ].copy()
    if len(df_tasks) < n_raw:
        print(f"load_str_norm_jobs: dropped {n_raw - len(df_tasks)} task rows with json_read_error")

    df_tasks["runtime_task"] = (
        df_tasks["reader_time_total_s"]
        + df_tasks["ftfy_time_total_s"]
        + df_tasks["punctuation_time_total_s"]
        + df_tasks["whitespace_time_total_s"]
        + df_tasks["writer_time_total_s"]
    )

    df = (
        df_tasks.groupby("run_id")
        .agg(
            corpus             = ("dataset",                    "first"),
            normalization_form = ("normalization_form",         "first"),
            n_tasks            = ("hpc_n_tasks",                "first"),
            runtime_s          = ("runtime_task",               "sum"),
            total_chars        = ("reader_doc_len_total_chars", "sum"),
            total_docs         = ("reader_docs_total",          "sum"),
        )
        .reset_index()
    )
    df["replicate"] = df["run_id"].str.extract(r"_(r\d+)$")[0]

    if exclude_run_id_pattern is not None:
        n_before = len(df)
        df = df[~df["run_id"].str.contains(exclude_run_id_pattern, na=False)].copy()
        n_excl = n_before - len(df)
        if n_excl:
            print(f"load_str_norm_jobs: excluded {n_excl} jobs matching "
                  f"run_id pattern '{exclude_run_id_pattern}' (see module docstring)")

    full_corpora = [e["corpus"] for e in corpus_registry]
    n_before = len(df)
    df = df[
        df["normalization_form"].isin(norm_forms) & df["corpus"].isin(full_corpora)
    ].copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"load_str_norm_jobs: dropped {n_before - n_after} jobs "
              f"(form not in {norm_forms} or corpus not in registry)")

    df["is_anchor"] = False

    if not df.empty:
        form2idx              = {f: i for i, f in enumerate(norm_forms)}
        df["form_idx"]        = df["normalization_form"].map(form2idx)
        df["log_rt_per_char"] = np.log(df["runtime_s"]) - np.log(df["total_chars"])
        df["log_n_tasks"]     = np.log(df["n_tasks"])

    return df


def load_str_norm_ear(
    ear_csv: Path | str,
    min_n_tasks: int = 8,
) -> pd.DataFrame:
    """Load and filter the pre-extracted string-normalisation EAR CSV.

    Keeps only matched, full-node runs (n_tasks >= min_n_tasks). The
    default of 8 matches the Rome nodes' minimum allocatable partition
    size (16 cores); runs below this granularity showed elevated
    cross-rank measurement noise in the heuristic filtering protocol
    comparison and are excluded by the same logic here.

    Returned columns are whatever is in the CSV; key ones used downstream:
        run_id, dataset, normalization_form, hpc_n_tasks,
        ear_dc_node_power_w, match_status
    """
    df = pd.read_csv(ear_csv)
    n_raw = len(df)
    df = df[
        (df["match_status"] == "matched_rank_to_ear_app") & (df["hpc_n_tasks"] >= min_n_tasks)
    ].copy()
    if len(df) < n_raw:
        print(f"load_str_norm_ear: kept {len(df)}/{n_raw} rows "
              f"(match_status='matched_rank_to_ear_app', n_tasks>={min_n_tasks})")
    return df


def load_str_norm_dataset(
    task_stats_csv: Path | str,
    norm_forms: list[str],
    corpus_registry: list[dict],
    exclude_run_id_pattern: str | None = DEFAULT_EXCLUDE_RUN_ID_PATTERN,
    corpora: list[str] | None = None,
) -> RuntimeDataset:
    """load_str_norm_jobs + design arrays, bundled into a RuntimeDataset."""
    df = load_str_norm_jobs(task_stats_csv, norm_forms, corpus_registry,
                            exclude_run_id_pattern)
    corpora, corp2idx = make_design_arrays(df, corpora=corpora)
    return RuntimeDataset(df=df, corpora=corpora, corp2idx=corp2idx,
                          stage="string_normalisation")